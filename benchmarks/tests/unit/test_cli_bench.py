from __future__ import annotations

import argparse
import json

import pytest

from benchmarks.cli import bench
from benchmarks.core.run_store import RunStore
from benchmarks.core.types import RunSpec, utc_now_iso


class _ExecutorStub:
    def __init__(self, run_store: RunStore):
        self.run_store = run_store


def test_parse_csv_trim_and_empty():
    assert bench._parse_csv(None) is None
    assert bench._parse_csv("") is None
    assert bench._parse_csv("a, b , ,c") == ["a", "b", "c"]


def test_cmd_delete_missing_run_returns_error(tmp_path, monkeypatch, capsys):
    store = RunStore(tmp_path / "index.sqlite")
    monkeypatch.setattr(bench, "_executor", lambda: _ExecutorStub(store))

    rc = bench.cmd_delete(argparse.Namespace(run_id="missing"))
    out = capsys.readouterr()
    assert rc == 1
    assert "Run not found: missing" in out.err


def test_cmd_restore_states(tmp_path, monkeypatch, capsys):
    store = RunStore(tmp_path / "index.sqlite")
    spec = RunSpec(run_id="r1", name="run", created_at=utc_now_iso())
    store.create(spec)
    monkeypatch.setattr(bench, "_executor", lambda: _ExecutorStub(store))

    rc_active = bench.cmd_restore(argparse.Namespace(run_id="r1"))
    out_active = capsys.readouterr()
    assert rc_active == 0
    assert "already active" in out_active.out

    assert store.soft_delete("r1") is True
    rc_restore = bench.cmd_restore(argparse.Namespace(run_id="r1"))
    out_restore = capsys.readouterr()
    assert rc_restore == 0
    assert "Restored run: r1" in out_restore.out


def test_cmd_list_json_empty(tmp_path, monkeypatch, capsys):
    store = RunStore(tmp_path / "index.sqlite")
    monkeypatch.setattr(bench, "_executor", lambda: _ExecutorStub(store))

    rc = bench.cmd_list(
        argparse.Namespace(include_deleted=False, limit=50, json=True)
    )
    out = capsys.readouterr()
    assert rc == 0
    assert json.loads(out.out) == []


def test_refs_sync_rejects_mutually_exclusive_input_args():
    parser = bench.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "refs-sync",
                "--source-url",
                "https://example.com/snapshot.json",
                "--input-json",
                "snapshot.json",
            ]
        )
