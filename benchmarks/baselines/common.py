from __future__ import annotations

import re
from typing import Iterable

from benchmarks.core.types import Prediction


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip().lower())


def keyword_match(answer: str, keywords: Iterable[str]) -> bool:
    low = normalize_text(answer)
    return all(normalize_text(k) in low for k in keywords if k)


class BaseBaseline:
    def __init__(self, baseline_id: str, run_spec, config: dict):
        self.baseline_id = baseline_id
        self.run_spec = run_spec
        self.config = config or {}

    def close(self) -> None:
        return

    def _prediction(self, case_id: str, answer: str, **metadata) -> Prediction:
        return Prediction(case_id=case_id, answer=answer, metadata=metadata)

