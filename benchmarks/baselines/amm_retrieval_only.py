from __future__ import annotations

import re
import time
from collections import deque
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from config import AgentConfig
from memory import AdaptiveModularMemory

from benchmarks.core.types import BatchContext, Prediction

from .common import BaseBaseline, normalize_text


class AMMRetrievalOnlyBaseline(BaseBaseline):
    """Baseline that evaluates AMM retrieval support without running the LLM.

    It seeds each case's `context_docs` into AMM, runs hop-1 retrieval against
    the question, then runs a deterministic hop-2 retrieval based on simple
    entity extraction + query templates.

    The returned `Prediction.answer` is intentionally empty; the value is in
    `Prediction.metadata` (support hit flags and scores).
    """

    _CAP_ENTITY_RE = re.compile(
        r"\b([A-Z][A-Za-z0-9'\-]+(?:\s+[A-Z][A-Za-z0-9'\-]+){0,4})\b"
    )

    def __init__(self, baseline_id: str, run_spec, config: dict):
        super().__init__(baseline_id, run_spec, config)
        self.cfg = AgentConfig()
        self.top_k = int(self.config.get("retrieval_top_k", self.cfg.retrieval_top_k))
        self.threshold = float(self.config.get("retrieval_threshold", self.cfg.retrieval_threshold))
        self.hop2_enabled = bool(self.config.get("retrieval_hop2_enabled", True))
        self.hop2_max_entities = int(self.config.get("hop2_max_entities", 6))

        self._hop2_bridge_res = [
            re.compile(p, re.IGNORECASE)
            for p in (self.cfg.hop2_bridge_patterns or [])
            if p
        ]
        self._hop2_location_intent_res = [
            re.compile(p, re.IGNORECASE)
            for p in (self.cfg.hop2_location_intent_patterns or [])
            if p
        ]

        root = Path(self.config.get("benchmark_root", "benchmarks"))
        mem_path = root / "runs" / run_spec.run_id / "artifacts" / "amm_retrieval_only_memory.json"
        mem_path.parent.mkdir(parents=True, exist_ok=True)
        self.memory = AdaptiveModularMemory(
            model_name=self.cfg.memory_encoder,
            max_slots=int(self.config.get("max_memory_slots", 10_000)),
            save_path=str(mem_path),
            decay_enabled=True,
            decay_half_lives=self.cfg.memory_decay_half_lives,
            dedup_enabled=True,
            dedup_scope=self.cfg.memory_dedup_scope,
            dedup_types=self.cfg.memory_dedup_types,
        )

    def _reset_state(self) -> None:
        with self.memory._lock:
            self.memory._keys = deque(maxlen=self.memory.max_slots)
            self.memory._values = deque(maxlen=self.memory.max_slots)
            self.memory._metadata = deque(maxlen=self.memory.max_slots)
            self.memory._dedup_counts = {}
            self.memory._version = 0
            self.memory._dirty = False
            self.memory._keys_tensor_cache = None
            self.memory._keys_tensor_cache_version = -1

    def _seed_context_docs(self, docs: List[str]) -> None:
        now = time.time()
        for idx, doc in enumerate(docs or [], start=1):
            if not doc:
                continue
            tagged = f"[DOC{idx:02d}] {doc}"
            self.memory.add_memory(
                tagged[:4000],
                {
                    "type": "document",
                    "subject": "benchmark_context",
                    "doc_id": idx,
                    "timestamp": now,
                },
            )

    def _hop2_intent(self, question: str) -> str:
        if any(p.search(question) for p in self._hop2_location_intent_res):
            return "location"
        return "default"

    def _extract_entities_bridge(self, hop1_texts: List[str]) -> List[str]:
        entities: List[str] = []
        for text in hop1_texts:
            for pat in self._hop2_bridge_res:
                m = pat.search(text)
                if not m:
                    continue
                ent = (m.groupdict().get("entity") or "").strip(" .,!?:;\"'")
                if not ent:
                    continue
                if ent not in entities:
                    entities.append(ent)
                if len(entities) >= self.hop2_max_entities:
                    return entities
        return entities

    def _extract_entities_capitalized(self, hop1_texts: List[str]) -> List[str]:
        stop = {"The", "A", "An", "In", "On", "At", "Of", "For", "To", "From", "By"}
        entities: List[str] = []
        for text in hop1_texts:
            for m in self._CAP_ENTITY_RE.finditer(text or ""):
                ent = (m.group(1) or "").strip()
                if not ent or ent in stop:
                    continue
                if ent not in entities:
                    entities.append(ent)
                if len(entities) >= self.hop2_max_entities:
                    return entities
        return entities

    def _build_hop2_queries(self, question: str, hop1_texts: List[str]) -> Tuple[List[str], int]:
        # Prefer explicit bridge patterns; fall back to naive capitalized spans.
        entities = self._extract_entities_bridge(hop1_texts)
        if not entities:
            entities = self._extract_entities_capitalized(hop1_texts)
        if not entities:
            return [], 0

        intent = self._hop2_intent(question)
        templates = self.cfg.hop2_query_templates_by_intent.get(
            intent,
            self.cfg.hop2_query_templates_by_intent.get("default", []),
        )
        out: List[str] = []
        for ent in entities:
            for tmpl in templates:
                q = (tmpl or "").format(entity=ent).strip()
                if q and q not in out:
                    out.append(q)
        return out, len(entities)

    def _retrieve_hop2(self, question: str, hop1_rows: List[Tuple[str, Dict, float]]) -> Tuple[List[Tuple[str, Dict, float]], Dict[str, object]]:
        if not hop1_rows:
            return [], {"hop2_queries": [], "hop2_entities": 0}
        hop1_texts = [t for t, _m, _s in hop1_rows]
        queries, entity_count = self._build_hop2_queries(question, hop1_texts)
        if not queries:
            return [], {"hop2_queries": [], "hop2_entities": 0}

        aggregated: Dict[str, Tuple[str, Dict, float]] = {}
        for q in queries:
            rows = self.memory.retrieve(
                q,
                top_k=max(1, self.cfg.retrieval_hop2_pattern_top_k_per_query),
                threshold=max(0.0, min(1.0, self.threshold * self.cfg.retrieval_hop2_pattern_threshold_scale)),
                include_types={"document"},
            )
            for text, meta, score in rows:
                prev = aggregated.get(text)
                if prev is None or score > prev[2]:
                    aggregated[text] = (text, meta, score)

        ranked = sorted(aggregated.values(), key=lambda x: x[2], reverse=True)
        return ranked[: self.top_k], {"hop2_queries": queries, "hop2_entities": entity_count}

    @staticmethod
    def _answer_docs(docs: List[str], expected: str) -> List[int]:
        exp = normalize_text(expected)
        if not exp:
            return []
        idxs = []
        for i, d in enumerate(docs or [], start=1):
            if exp and exp in normalize_text(d):
                idxs.append(i)
        return idxs

    @staticmethod
    def _hit_any_answer_doc(retrieved_texts: List[str], expected: str) -> bool:
        exp = normalize_text(expected)
        if not exp:
            return False
        return any(exp in normalize_text(t) for t in (retrieved_texts or []))

    def answer(self, batch_ctx: BatchContext) -> List[Prediction]:
        predictions: List[Prediction] = []
        for item in batch_ctx.items:
            self._reset_state()
            self._seed_context_docs(item.context_docs or [])

            hop1 = self.memory.retrieve(
                item.prompt,
                top_k=self.top_k,
                threshold=self.threshold,
                include_types={"document"},
            )

            hop2: List[Tuple[str, Dict, float]] = []
            hop2_meta: Dict[str, object] = {"hop2_queries": [], "hop2_entities": 0}
            if self.hop2_enabled and hop1:
                hop2, hop2_meta = self._retrieve_hop2(item.prompt, hop1)

            hop1_texts = [t for t, _m, _s in hop1]
            hop2_texts = [t for t, _m, _s in hop2]

            hop1_hit = self._hit_any_answer_doc(hop1_texts, item.expected)
            hop2_hit = self._hit_any_answer_doc(hop2_texts, item.expected)
            any_hit = hop1_hit or hop2_hit

            predictions.append(
                self._prediction(
                    item.case_id,
                    "",
                    suite=batch_ctx.suite_id,
                    dataset=item.dataset,
                    seeded_docs=len(item.context_docs or []),
                    answer_doc_ids=self._answer_docs(item.context_docs or [], item.expected),
                    hop1_hit=hop1_hit,
                    hop2_hit=hop2_hit,
                    any_hit=any_hit,
                    hop1_max_score=max((s for _t, _m, s in hop1), default=0.0),
                    hop2_max_score=max((s for _t, _m, s in hop2), default=0.0),
                    hop1_count=len(hop1),
                    hop2_count=len(hop2),
                    **hop2_meta,
                )
            )
        return predictions

    def close(self) -> None:
        return
