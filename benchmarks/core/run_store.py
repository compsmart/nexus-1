from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import List, Optional

from .types import RunRecord, RunSpec, utc_now_iso


class RunStore:
    def __init__(self, sqlite_path: Path):
        self.sqlite_path = sqlite_path
        self.sqlite_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(str(self.sqlite_path))

    def _init_db(self) -> None:
        with self._connect() as con:
            con.execute(
                """
                CREATE TABLE IF NOT EXISTS runs (
                    run_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    status TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    deleted_at TEXT,
                    spec_path TEXT,
                    results_path TEXT,
                    metrics_path TEXT,
                    error TEXT
                )
                """
            )
            con.commit()

    def create(self, run_spec: RunSpec) -> RunRecord:
        now = utc_now_iso()
        rec = RunRecord(
            run_id=run_spec.run_id,
            name=run_spec.name,
            status="queued",
            created_at=run_spec.created_at,
            updated_at=now,
        )
        with self._connect() as con:
            con.execute(
                """
                INSERT INTO runs (
                    run_id, name, status, created_at, updated_at, deleted_at,
                    spec_path, results_path, metrics_path, error
                ) VALUES (?, ?, ?, ?, ?, NULL, NULL, NULL, NULL, NULL)
                """,
                (
                    rec.run_id,
                    rec.name,
                    rec.status,
                    rec.created_at,
                    rec.updated_at,
                ),
            )
            con.commit()
        return rec

    def update_status(self, run_id: str, status: str, error: Optional[str] = None) -> None:
        with self._connect() as con:
            con.execute(
                "UPDATE runs SET status=?, updated_at=?, error=? WHERE run_id=?",
                (status, utc_now_iso(), error, run_id),
            )
            con.commit()

    def update_artifacts(self, run_id: str, spec_path: str, results_path: str, metrics_path: str) -> None:
        with self._connect() as con:
            con.execute(
                """
                UPDATE runs
                SET spec_path=?, results_path=?, metrics_path=?, updated_at=?
                WHERE run_id=?
                """,
                (spec_path, results_path, metrics_path, utc_now_iso(), run_id),
            )
            con.commit()

    def get(self, run_id: str) -> Optional[RunRecord]:
        with self._connect() as con:
            row = con.execute(
                """
                SELECT run_id, name, status, created_at, updated_at, deleted_at,
                       spec_path, results_path, metrics_path, error
                FROM runs WHERE run_id=?
                """,
                (run_id,),
            ).fetchone()
        if row is None:
            return None
        return self._row_to_record(row)

    def list(self, include_deleted: bool = False, limit: int = 200) -> List[RunRecord]:
        clause = "" if include_deleted else "WHERE deleted_at IS NULL"
        query = (
            "SELECT run_id, name, status, created_at, updated_at, deleted_at, "
            "spec_path, results_path, metrics_path, error "
            f"FROM runs {clause} ORDER BY created_at DESC LIMIT ?"
        )
        with self._connect() as con:
            rows = con.execute(query, (limit,)).fetchall()
        return [self._row_to_record(r) for r in rows]

    def soft_delete(self, run_id: str) -> bool:
        with self._connect() as con:
            cur = con.execute(
                "UPDATE runs SET deleted_at=?, updated_at=? WHERE run_id=?",
                (utc_now_iso(), utc_now_iso(), run_id),
            )
            con.commit()
            return cur.rowcount > 0

    def restore(self, run_id: str) -> bool:
        with self._connect() as con:
            cur = con.execute(
                "UPDATE runs SET deleted_at=NULL, updated_at=? WHERE run_id=?",
                (utc_now_iso(), run_id),
            )
            con.commit()
            return cur.rowcount > 0

    @staticmethod
    def _row_to_record(row) -> RunRecord:
        return RunRecord(
            run_id=row[0],
            name=row[1],
            status=row[2],
            created_at=row[3],
            updated_at=row[4],
            deleted_at=row[5],
            spec_path=row[6],
            results_path=row[7],
            metrics_path=row[8],
            error=row[9],
        )
