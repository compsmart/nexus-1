from __future__ import annotations

import time
from collections import deque
from pathlib import Path
from typing import List

from agent import AGIAgent
from config import AgentConfig

from benchmarks.core.types import BatchContext, Prediction

from .common import BaseBaseline


class AMMAgentBaseline(BaseBaseline):
    def __init__(self, baseline_id: str, run_spec, config: dict):
        super().__init__(baseline_id, run_spec, config)
        cfg = AgentConfig()
        cfg.model_name = run_spec.model_name
        cfg.use_4bit = run_spec.use_4bit
        cfg.autonomous_learning = False
        cfg.retrieval_hop2_enabled = bool(self.config.get("retrieval_hop2_enabled", True))
        cfg.retrieval_hop2_strategy = self.config.get("retrieval_hop2_strategy", "hybrid")
        cfg.max_new_tokens_response = run_spec.max_new_tokens
        cfg.max_new_tokens_extraction = 64
        cfg.think_interval_secs = 9999.0
        cfg.idle_threshold_secs = 9999.0
        cfg.flush_interval_secs = 9999.0
        cfg.tool_routing_backend = "pattern"
        self.agent = AGIAgent(config=cfg)

        root = Path(self.config.get("benchmark_root", "benchmarks"))
        mem_path = root / "runs" / run_spec.run_id / "artifacts" / "amm_memory.json"
        mem_path.parent.mkdir(parents=True, exist_ok=True)
        self.agent.memory.save_path = str(mem_path)
        self.agent.memory._keys_path = str(mem_path) + ".pt"

    def _reset_state(self) -> None:
        with self.agent.memory._lock:
            self.agent.memory._keys = deque(maxlen=self.agent.memory.max_slots)
            self.agent.memory._values = deque(maxlen=self.agent.memory.max_slots)
            self.agent.memory._metadata = deque(maxlen=self.agent.memory.max_slots)
            self.agent.memory._dedup_counts = {}
            self.agent.memory._version = 0
            self.agent.memory._dirty = False
        self.agent._history.clear()
        self.agent.user_name = None

    def _seed_context_docs(self, docs: List[str]) -> None:
        now = time.time()
        for doc in docs:
            if not doc:
                continue
            self.agent.memory.add_memory(
                doc[:2000],
                {"type": "document", "subject": "benchmark_context", "timestamp": now},
            )

    def answer(self, batch_ctx: BatchContext) -> List[Prediction]:
        predictions: List[Prediction] = []
        for item in batch_ctx.items:
            self._reset_state()
            self._seed_context_docs(item.context_docs or [])
            answer = self.agent.interact(item.prompt)
            predictions.append(
                self._prediction(
                    item.case_id,
                    answer,
                    suite=batch_ctx.suite_id,
                    seeded_docs=len(item.context_docs or []),
                )
            )
        return predictions

    def close(self) -> None:
        self.agent.stop()
