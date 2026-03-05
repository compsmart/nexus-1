from __future__ import annotations

from typing import Dict, List

from .normalizer import normalize_hotpot_row, normalize_two_wiki_row

try:
    from datasets import load_dataset
except Exception:
    load_dataset = None


def _fallback_cases() -> List[Dict]:
    return [
        {
            "question": "Which city is the lead of Project Atlas based in?",
            "answer": "Lisbon",
            "context_docs": ["Project Atlas is led by Dana.", "Dana is based in Lisbon."],
            "dataset": "fallback",
        },
        {
            "question": "Where does the manager of Project Orion work from?",
            "answer": "Nairobi",
            "context_docs": ["Project Orion is managed by Malik.", "Malik works from Nairobi."],
            "dataset": "fallback",
        },
    ]


def load_multihop_cases(dataset_cfg: Dict, split: str, dataset_samples: Dict[str, int], docs_per_case: int) -> List[Dict]:
    if load_dataset is None:
        return _fallback_cases()

    out: List[Dict] = []
    hotpot_n = int(dataset_samples.get("hotpot_qa", 0))
    two_wiki_n = int(dataset_samples.get("two_wiki", 0))
    datasets_cfg = dataset_cfg.get("datasets", {})

    if hotpot_n > 0:
        cfg = datasets_cfg.get("hotpot_qa", {})
        ds = load_dataset(
            cfg.get("dataset_id", "hotpot_qa"),
            cfg.get("config", "distractor"),
            split=f"{split}[:{hotpot_n}]",
        )
        for row in ds:
            out.append(normalize_hotpot_row(row, docs_per_case=docs_per_case))

    if two_wiki_n > 0:
        cfg = datasets_cfg.get("two_wiki", {})
        ds = load_dataset(
            cfg.get("dataset_id", "scholarly-shadows-syndicate/2wikimultihopqa"),
            split=f"{split}[:{two_wiki_n}]",
        )
        for row in ds:
            out.append(normalize_two_wiki_row(row, docs_per_case=docs_per_case))

    return out if out else _fallback_cases()

