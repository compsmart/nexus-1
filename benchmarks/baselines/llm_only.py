from __future__ import annotations

from typing import List

from benchmarks.core.types import BatchContext, Prediction
from llm import LLMEngine

from .common import BaseBaseline


class LLMOnlyBaseline(BaseBaseline):
    def __init__(self, baseline_id: str, run_spec, config: dict):
        super().__init__(baseline_id, run_spec, config)
        self.llm = LLMEngine(
            model_name=run_spec.model_name,
            use_4bit=run_spec.use_4bit,
        )

    def answer(self, batch_ctx: BatchContext) -> List[Prediction]:
        predictions: List[Prediction] = []
        for item in batch_ctx.items:
            prompt = (
                "Answer the question accurately and concisely.\n"
                f"Question: {item.prompt}\nAnswer:"
            )
            answer = self.llm.generate(
                prompt,
                max_new_tokens=self.run_spec.max_new_tokens,
                temperature=0.2,
            )
            predictions.append(self._prediction(item.case_id, answer, suite=batch_ctx.suite_id))
        return predictions

