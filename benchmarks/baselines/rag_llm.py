from __future__ import annotations

import math
import re
from collections import Counter
from typing import List, Tuple

from benchmarks.core.types import BatchContext, Prediction
from llm import LLMEngine

from .common import BaseBaseline


def _tokens(text: str) -> List[str]:
    return re.findall(r"[a-z0-9]+", (text or "").lower())


def _tf(counter: Counter) -> dict:
    total = sum(counter.values()) or 1
    return {k: v / total for k, v in counter.items()}


def _cosine(a: dict, b: dict) -> float:
    keys = set(a.keys()) | set(b.keys())
    dot = sum(a.get(k, 0.0) * b.get(k, 0.0) for k in keys)
    na = math.sqrt(sum(v * v for v in a.values())) or 1.0
    nb = math.sqrt(sum(v * v for v in b.values())) or 1.0
    return dot / (na * nb)


class RAGLLMBaseline(BaseBaseline):
    def __init__(self, baseline_id: str, run_spec, config: dict):
        super().__init__(baseline_id, run_spec, config)
        self.llm = LLMEngine(
            model_name=run_spec.model_name,
            use_4bit=run_spec.use_4bit,
        )
        self.top_k = int(self.config.get("top_k_docs", 5))
        self.max_context_chars = int(self.config.get("max_context_chars", 6000))

    def _rank_docs(self, question: str, docs: List[str]) -> List[Tuple[str, float]]:
        q = _tf(Counter(_tokens(question)))
        scored = []
        for doc in docs:
            d = _tf(Counter(_tokens(doc)))
            scored.append((doc, _cosine(q, d)))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored

    def answer(self, batch_ctx: BatchContext) -> List[Prediction]:
        predictions: List[Prediction] = []
        for item in batch_ctx.items:
            ranked = self._rank_docs(item.prompt, item.context_docs or [])
            chosen = [doc for doc, _score in ranked[: self.top_k]]
            context = "\n".join(f"- {d}" for d in chosen)
            context = context[: self.max_context_chars]
            prompt = (
                "You are an accurate QA model. Use only the provided context. "
                "If context is insufficient, answer with best effort and keep concise.\n"
                f"Context:\n{context}\n\nQuestion: {item.prompt}\nAnswer:"
            )
            answer = self.llm.generate(
                prompt,
                max_new_tokens=self.run_spec.max_new_tokens,
                temperature=0.2,
            )
            predictions.append(
                self._prediction(
                    item.case_id,
                    answer,
                    suite=batch_ctx.suite_id,
                    retrieved_docs=len(chosen),
                )
            )
        return predictions

