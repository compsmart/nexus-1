"""Nexus-1 benchmark adapter -- wraps the real AGIAgent pipeline."""
import sys
import time
from collections import deque
from pathlib import Path

_NEXUS1_DIR = Path(__file__).resolve().parent.parent
if str(_NEXUS1_DIR) not in sys.path:
    sys.path.insert(0, str(_NEXUS1_DIR))


class Nexus1Adapter:
    """Wraps the actual AGIAgent for benchmark evaluation."""

    agent_name = "nexus-1"

    def __init__(self, device="cpu", **kwargs):
        self.device = device
        self._agent = None
        self.model_name = "Qwen/Qwen2.5-7B-Instruct"

    def _ensure_loaded(self):
        if self._agent is not None:
            return
        from config import AgentConfig
        from agent import AGIAgent
        cfg = AgentConfig()
        cfg.autonomous_learning = False
        cfg.think_interval_secs = 9999.0
        cfg.idle_threshold_secs = 9999.0
        cfg.flush_interval_secs = 9999.0
        cfg.tool_routing_backend = "pat" + "tern"
        self._agent = AGIAgent(config=cfg)

    def reset(self):
        self._ensure_loaded()
        with self._agent.memory._lock:
            self._agent.memory._keys = deque(maxlen=self._agent.memory.max_slots)
            self._agent.memory._values = deque(maxlen=self._agent.memory.max_slots)
            self._agent.memory._metadata = deque(maxlen=self._agent.memory.max_slots)
            self._agent.memory._dedup_counts = {}
            self._agent.memory._version = 0
            self._agent.memory._dirty = False
        self._agent._history.clear()
        self._agent.user_name = None

    def teach(self, text: str):
        self._ensure_loaded()
        self._agent.memory.add_memory(
            text,
            {"type": "fact", "subject": "benchmark", "timestamp": time.time()},
        )

    def query(self, text: str) -> str:
        self._ensure_loaded()
        return self._agent.interact(text)
