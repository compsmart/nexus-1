from __future__ import annotations

from typing import Dict, List

from benchmarks.core.types import BatchItem

from .prompts import build_prompt


def to_batch_items(cases: List[Dict]) -> List[BatchItem]:
    items: List[BatchItem] = []
    for row in cases:
        items.append(
            BatchItem(
                case_id=row["case_id"],
                prompt=build_prompt(row["question"]),
                expected=row.get("expected", ""),
                keywords=list(row.get("keywords", [])),
                category=row.get("category", "general"),
                dataset="galileo_agent",
                context_docs=list(row.get("context_docs", [])),
                metadata={"source": "galileo_agent_loader"},
            )
        )
    return items

