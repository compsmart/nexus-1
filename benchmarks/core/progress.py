from __future__ import annotations

import threading
from pathlib import Path
from typing import Any, Dict

from .artifacts import append_jsonl
from .types import utc_now_iso


class NullProgressSink:
    def emit(self, event_type: str, message: str, payload: Dict[str, Any] | None = None) -> None:
        _ = (event_type, message, payload)


class JsonlProgressSink:
    def __init__(self, path: Path):
        self._path = path
        self._lock = threading.Lock()

    def emit(self, event_type: str, message: str, payload: Dict[str, Any] | None = None) -> None:
        row = {
            "timestamp": utc_now_iso(),
            "event_type": event_type,
            "message": message,
            "payload": payload or {},
        }
        with self._lock:
            append_jsonl(self._path, row)

