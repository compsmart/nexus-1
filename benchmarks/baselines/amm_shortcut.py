"""Clean AMM agent baseline for nexus-1 benchmarks.

All queries go through the real AGIAgent pipeline -- no shortcuts,
no regex pattern matching, no RAG bypass. This replaces the old
509-line shortcut-heavy baseline (D-413).

The class name AMMShortcutBaseline is kept for backward compatibility
with baselines.yaml references.
"""
from __future__ import annotations

import logging
import os
import sys
import time
from collections import deque
from typing import List

from benchmarks.core.types import BatchContext, Prediction
from .common import BaseBaseline

_log = logging.getLogger(__name__)

# Ensure nexus-1 root is importable
_NEXUS1_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _NEXUS1_DIR not in sys.path:
    sys.path.insert(0, _NEXUS1_DIR)


class AMMShortcutBaseline(BaseBaseline):
    """Nexus-1 baseline using the real AGIAgent pipeline.

    All shortcut logic has been removed. Queries go through
    AGIAgent.interact() which uses AMM retrieval + LLM reasoning.
    """

    def __init__(self, baseline_id: str, run_spec, config: dict):
        super().__init__(baseline_id, run_spec, config)
        self._agent = None
        self._run_spec = run_spec

    def _ensure_agent(self):
        if self._agent is not None:
            return
        from config import AgentConfig
        from agent import AGIAgent

        cfg = AgentConfig()
        cfg.model_name = getattr(self._run_spec, "model_name", "Qwen/Qwen2.5-7B-Instruct")
        cfg.use_4bit = getattr(self._run_spec, "use_4bit", True)
        cfg.autonomous_learning = False
        cfg.think_interval_secs = 9999.0
        cfg.idle_threshold_secs = 9999.0
        cfg.flush_interval_secs = 9999.0
        cfg.tool_routing_backend = "pattern"
        self._agent = AGIAgent(config=cfg)

    def _reset_state(self) -> None:
        if self._agent is None:
            return
        with self._agent.memory._lock:
            self._agent.memory._keys = deque(maxlen=self._agent.memory.max_slots)
            self._agent.memory._values = deque(maxlen=self._agent.memory.max_slots)
            self._agent.memory._metadata = deque(maxlen=self._agent.memory.max_slots)
            self._agent.memory._dedup_counts = {}
            self._agent.memory._version = 0
            self._agent.memory._dirty = False
            self._agent.memory._keys_tensor_cache = None
            self._agent.memory._keys_tensor_cache_version = -1
        self._agent._history.clear()
        self._agent.user_name = None

    def _seed_context_docs(self, docs: List[str]) -> None:
        self._ensure_agent()
        now = time.time()
        for doc in docs:
            if doc:
                self._agent.memory.add_memory(
                    doc[:2000],
                    {"type": "fact", "subject": "benchmark_context", "timestamp": now},
                )

    def answer(self, batch_ctx: BatchContext) -> List[Prediction]:
        self._ensure_agent()
        predictions: List[Prediction] = []
        for item in batch_ctx.items:
            self._reset_state()
            self._seed_context_docs(item.context_docs or [])
            result = self._agent.interact(item.prompt)
            predictions.append(
                self._prediction(
                    item.case_id,
                    result,
                    suite=batch_ctx.suite_id,
                    seeded_docs=len(item.context_docs or []),
                )
            )
        return predictions

    def close(self) -> None:
        if self._agent is not None:
            try:
                self._agent.stop()
            except Exception:
                pass
        self._agent = None
