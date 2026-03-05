from __future__ import annotations

from pathlib import Path

from benchmarks.core.run_store import RunStore
from benchmarks.core.types import RunSpec, utc_now_iso


def test_soft_delete_restore(tmp_path: Path):
    store = RunStore(tmp_path / "index.sqlite")
    spec = RunSpec(run_id="r1", name="run", created_at=utc_now_iso())
    store.create(spec)
    assert store.get("r1") is not None
    assert store.soft_delete("r1") is True
    row = store.get("r1")
    assert row is not None
    assert row.deleted_at is not None
    assert store.restore("r1") is True
    row = store.get("r1")
    assert row is not None
    assert row.deleted_at is None
    assert store.soft_delete("unknown") is False
    assert store.restore("unknown") is False

