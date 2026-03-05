from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import requests

from benchmarks.core.artifacts import read_json, write_json
from benchmarks.core.executor import BenchmarkExecutor
from benchmarks.core.types import RunRecord, utc_now_iso


def _executor() -> BenchmarkExecutor:
    root = Path(__file__).resolve().parents[1]
    return BenchmarkExecutor(root)

def _parse_csv(value: str | None) -> List[str] | None:
    if not value:
        return None
    items = [v.strip() for v in value.split(",")]
    clean = [v for v in items if v]
    return clean or None

def _run_record_to_dict(row: RunRecord) -> Dict:
    return dict(row.__dict__)

def _persist_to_db(spec, metrics: dict) -> None:
    """Persist completed benchmark run to PostgreSQL. Never raises."""
    import asyncio as _asyncio
    import os as _os

    async def _insert():
        import asyncpg
        from datetime import datetime, timezone as _tz
        db_url = _os.environ.get(
            "DATABASE_URL",
            "postgresql://postgres:Pa55w0rd123%21@localhost:5432/ai_lab",
        )
        try:
            created_ts = datetime.fromisoformat(spec.created_at.replace("Z", "+00:00"))
        except (AttributeError, ValueError):
            created_ts = datetime.now(_tz.utc)

        suite_results = []
        try:
            results_path = (
                Path(__file__).resolve().parents[1] / "runs" / spec.run_id / "results.json"
            )
            suite_results = json.loads(results_path.read_text()).get("suite_results", [])
        except Exception:
            pass

        conn = await asyncpg.connect(db_url)
        try:
            await conn.execute(
                """
                INSERT INTO benchmark_runs
                    (run_id, agent_name, name, status, profile, model_name,
                     suites, baselines, suite_weights, aggregate_scores, suite_results,
                     created_at, completed_at)
                VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,NOW())
                ON CONFLICT (run_id) DO UPDATE SET
                    status           = EXCLUDED.status,
                    aggregate_scores = EXCLUDED.aggregate_scores,
                    suite_results    = EXCLUDED.suite_results,
                    completed_at     = EXCLUDED.completed_at
                """,
                spec.run_id,
                getattr(spec, "agent", "nexus-1"),
                getattr(spec, "name", ""),
                "completed",
                getattr(spec, "profile", ""),
                getattr(spec, "model_name", ""),
                json.dumps(list(getattr(spec, "suites", []))),
                json.dumps(list(getattr(spec, "baselines", []))),
                json.dumps(metrics.get("suite_weights", {})),
                json.dumps(metrics.get("aggregate_scores", [])),
                json.dumps(suite_results),
                created_ts,
            )
        finally:
            await conn.close()

    try:
        _asyncio.run(_insert())
        print("[DB] Benchmark run persisted to database.", flush=True)
    except Exception as e:
        print(f"[DB] Warning: could not persist benchmark run: {e}", flush=True)


def cmd_run(args: argparse.Namespace) -> int:
    try:
        import setproctitle
        setproctitle.setproctitle(f"nexus1-benchmark:{args.name}")
    except ImportError:
        pass

    ex = _executor()
    suites = _parse_csv(args.suites)
    baselines = _parse_csv(args.baselines)
    spec = ex.build_default_run_spec(
        name=args.name,
        suites=suites,
        baselines=baselines,
        profile=args.profile,
        model_name=args.model,
        use_4bit=args.use_4bit,
    )
    spec.max_new_tokens = args.max_new_tokens
    if args.seed is not None:
        spec.seed = int(args.seed)
    result = ex.run(spec)

    _persist_to_db(spec, result)

    if args.json:
        print(json.dumps(result, indent=2))
        return 0
    print(f"Run completed: {result.get('run_id', spec.run_id)}")
    aggregate = result.get("aggregate_scores", [])
    if not aggregate:
        print("No aggregate scores were produced.")
        return 0
    print("Aggregate scores:")
    for row in aggregate:
        bid = row.get("baseline_id", "?")
        score = float(row.get("overall_score", 0.0))
        print(f"- {bid}: {score:.4f}")
    return 0


def cmd_list(args: argparse.Namespace) -> int:
    ex = _executor()
    rows = ex.run_store.list(include_deleted=args.include_deleted, limit=args.limit)
    if args.json:
        print(json.dumps([_run_record_to_dict(row) for row in rows], indent=2))
        return 0
    if not rows:
        print("No runs found.")
        return 0
    for row in rows:
        print(
            f"{row.run_id}  status={row.status}  name={row.name}  "
            f"created={row.created_at}  deleted_at={row.deleted_at or '-'}"
        )
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    ex = _executor()
    row = ex.run_store.get(args.run_id)
    if row is None:
        print(f"Run not found: {args.run_id}", file=sys.stderr)
        return 1
    if args.json:
        print(json.dumps(_run_record_to_dict(row), indent=2))
        return 0
    print(f"Run ID: {row.run_id}")
    print(f"Name: {row.name}")
    print(f"Status: {row.status}")
    print(f"Created: {row.created_at}")
    print(f"Updated: {row.updated_at}")
    print(f"Deleted At: {row.deleted_at or '-'}")
    print(f"Spec: {row.spec_path or '-'}")
    print(f"Results: {row.results_path or '-'}")
    print(f"Metrics: {row.metrics_path or '-'}")
    if row.error:
        print(f"Error: {row.error}")
    return 0


def cmd_delete(args: argparse.Namespace) -> int:
    ex = _executor()
    row = ex.run_store.get(args.run_id)
    if row is None:
        print(f"Run not found: {args.run_id}", file=sys.stderr)
        return 1
    if row.deleted_at is not None:
        print(f"Run already soft-deleted: {args.run_id}")
        return 0
    if not ex.run_store.soft_delete(args.run_id):
        print(f"Failed to soft-delete run: {args.run_id}", file=sys.stderr)
        return 1
    print(f"Soft-deleted run: {args.run_id}")
    return 0


def cmd_restore(args: argparse.Namespace) -> int:
    ex = _executor()
    row = ex.run_store.get(args.run_id)
    if row is None:
        print(f"Run not found: {args.run_id}", file=sys.stderr)
        return 1
    if row.deleted_at is None:
        print(f"Run is already active: {args.run_id}")
        return 0
    if not ex.run_store.restore(args.run_id):
        print(f"Failed to restore run: {args.run_id}", file=sys.stderr)
        return 1
    print(f"Restored run: {args.run_id}")
    return 0


def _load_metrics(path: str) -> Dict:
    if not path:
        return {}
    p = Path(path)
    if not p.exists():
        return {}
    return read_json(p)


def cmd_compare(args: argparse.Namespace) -> int:
    ex = _executor()
    a = ex.run_store.get(args.run_a)
    b = ex.run_store.get(args.run_b)
    missing = []
    if a is None:
        missing.append(args.run_a)
    if b is None:
        missing.append(args.run_b)
    if missing:
        print(f"Run not found: {', '.join(missing)}", file=sys.stderr)
        return 1
    ma = _load_metrics(a.metrics_path or "")
    mb = _load_metrics(b.metrics_path or "")
    scores_a = {r["baseline_id"]: r["overall_score"] for r in ma.get("aggregate_scores", [])}
    scores_b = {r["baseline_id"]: r["overall_score"] for r in mb.get("aggregate_scores", [])}
    baseline_ids = sorted(set(scores_a.keys()) | set(scores_b.keys()))
    rows = []
    for bid in baseline_ids:
        sa = scores_a.get(bid)
        sb = scores_b.get(bid)
        rows.append(
            {
                "baseline_id": bid,
                "run_a": sa,
                "run_b": sb,
                "delta": None if sa is None or sb is None else (sb - sa),
            }
        )

    if args.json:
        print(
            json.dumps(
                {
                    "run_a": args.run_a,
                    "run_b": args.run_b,
                    "comparisons": rows,
                },
                indent=2,
            )
        )
        return 0

    print(f"Compare runs: {args.run_a} vs {args.run_b}")
    if not rows:
        print("No comparable aggregate scores found in either run.")
        return 0
    for row in rows:
        bid = row["baseline_id"]
        sa = row["run_a"]
        sb = row["run_b"]
        if sa is None or sb is None:
            print(f"- {bid}: non-overlap (run_a={sa}, run_b={sb})")
            continue
        print(f"- {bid}: run_a={sa:.4f} run_b={sb:.4f} delta={sb - sa:+.4f}")
    return 0


def cmd_refs_sync(args: argparse.Namespace) -> int:
    ex = _executor()
    snap_dir = ex.root / "reference_data" / "leaderboards" / args.leaderboard / "snapshots"
    snap_dir.mkdir(parents=True, exist_ok=True)
    out_path = snap_dir / f"{datetime.utcnow().strftime('%Y-%m-%d')}.json"

    payload = {}
    if args.input_json:
        payload = read_json(Path(args.input_json))
    elif args.source_url:
        try:
            resp = requests.get(args.source_url, timeout=30)
            resp.raise_for_status()
            payload = resp.json()
        except requests.RequestException as exc:
            print(f"Failed to fetch source URL: {exc}", file=sys.stderr)
            return 1
    else:
        # fallback: create a local reference snapshot from completed runs
        rows = ex.run_store.list(include_deleted=False, limit=200)
        models = []
        for row in rows:
            if row.status != "completed":
                continue
            metrics = _load_metrics(row.metrics_path or "")
            agg = metrics.get("aggregate_scores", [])
            if not agg:
                continue
            top = max(agg, key=lambda x: x.get("overall_score", 0.0))
            models.append(
                {
                    "name": f"{row.name}:{top.get('baseline_id')}",
                    "score": float(top.get("overall_score", 0.0)),
                }
            )
        payload = {
            "source": "local_completed_runs",
            "created_at": utc_now_iso(),
            "score_field": "score",
            "models": models,
        }
        if not models and out_path.exists():
            print(f"No completed runs found. Keeping existing snapshot: {out_path}")
            return 0

    if "models" not in payload:
        raise ValueError("Snapshot JSON must contain a 'models' field.")
    payload.setdefault("created_at", utc_now_iso())
    payload.setdefault("score_field", "score")
    payload.setdefault("source", args.source_url or "manual")
    write_json(out_path, payload, immutable=False)
    print(f"Snapshot written: {out_path}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Benchmark orchestration CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python -m benchmarks.cli.bench run --name quality-run\n"
            "  python -m benchmarks.cli.bench list --limit 20\n"
            "  python -m benchmarks.cli.bench status <run_id>\n"
            "  python -m benchmarks.cli.bench compare <run_a> <run_b>\n"
        ),
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_run = sub.add_parser("run", help="Run a benchmark job now.")
    p_run.add_argument("--name", default="benchmark-run")
    p_run.add_argument("--profile", default=None)
    p_run.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    p_run.add_argument("--use-4bit", action=argparse.BooleanOptionalAction, default=True)
    p_run.add_argument("--max-new-tokens", type=int, default=128)
    p_run.add_argument("--seed", type=int, default=None)
    p_run.add_argument("--suites", default=None, help="Comma-separated suite ids")
    p_run.add_argument("--baselines", default=None, help="Comma-separated baseline ids")
    p_run.add_argument("--json", action="store_true", help="Output JSON.")
    p_run.set_defaults(func=cmd_run)

    p_list = sub.add_parser("list", help="List benchmark runs.")
    p_list.add_argument("--include-deleted", action="store_true")
    p_list.add_argument("--limit", type=int, default=50)
    p_list.add_argument("--json", action="store_true", help="Output JSON.")
    p_list.set_defaults(func=cmd_list)

    p_status = sub.add_parser("status", help="Get run status.")
    p_status.add_argument("run_id")
    p_status.add_argument("--json", action="store_true", help="Output JSON.")
    p_status.set_defaults(func=cmd_status)

    p_delete = sub.add_parser("delete", help="Soft delete a run.")
    p_delete.add_argument("run_id")
    p_delete.set_defaults(func=cmd_delete)

    p_restore = sub.add_parser("restore", help="Restore a soft-deleted run.")
    p_restore.add_argument("run_id")
    p_restore.set_defaults(func=cmd_restore)

    p_compare = sub.add_parser("compare", help="Compare two run results.")
    p_compare.add_argument("run_a")
    p_compare.add_argument("run_b")
    p_compare.add_argument("--json", action="store_true", help="Output JSON.")
    p_compare.set_defaults(func=cmd_compare)

    p_refs = sub.add_parser("refs-sync", help="Sync reference leaderboard snapshot.")
    p_refs.add_argument("--leaderboard", default="galileo")
    input_group = p_refs.add_mutually_exclusive_group()
    input_group.add_argument("--source-url", default=None)
    input_group.add_argument("--input-json", default=None)
    p_refs.set_defaults(func=cmd_refs_sync)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    try:
        return int(args.func(args))
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
