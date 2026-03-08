"""Clean adapter wrapping the actual nexus-1 AGIAgent for benchmarking.

No shortcuts -- all queries go through the real agent pipeline.
"""

import sys
from pathlib import Path

# Ensure nexus-1 root is importable
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
        self._agent = AGIAgent(config=cfg)

    def reset(self):
        self._ensure_loaded()
        # Clear adaptive modular memory
        self._agent.memory.bank.clear()
        if hasattr(self._agent.memory, '_snapshot_embeddings'):
            self._agent.memory._snapshot_embeddings = None
        self._agent._history.clear()

    def teach(self, text: str):
        self._ensure_loaded()
        self._agent.memory.store(text, mem_type="fact")

    def query(self, text: str) -> str:
        self._ensure_loaded()
        return self._agent.interact(text)
