from __future__ import annotations

import queue
import threading
from typing import Optional

from .executor import BenchmarkExecutor
from .types import RunSpec


class RunScheduler:
    def __init__(self, executor: BenchmarkExecutor):
        self.executor = executor
        self._queue: "queue.Queue[RunSpec]" = queue.Queue()
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)

    def enqueue(self, run_spec: RunSpec) -> None:
        self._queue.put(run_spec)

    def queue_size(self) -> int:
        return self._queue.qsize()

    def _worker(self) -> None:
        while not self._stop_event.is_set():
            try:
                spec = self._queue.get(timeout=0.2)
            except queue.Empty:
                continue
            try:
                self.executor.run(spec)
            finally:
                self._queue.task_done()

