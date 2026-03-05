from __future__ import annotations

from typing import Dict, List


def application_tasks(repetitions: int = 1) -> List[Dict]:
    base = [
        {"id": "LT01", "q": "Using Rule A, what is CODE(map)?", "keywords": ["NBQ"], "category": "rule_apply"},
        {"id": "LT02", "q": "Which city is the owner of Project Kappa located in?", "keywords": ["seoul"], "category": "multi_hop"},
        {"id": "LT03", "q": "What is the emergency protocol token?", "keywords": ["channel-7"], "category": "retention"},
    ]
    out: List[Dict] = []
    for rep in range(repetitions):
        for row in base:
            out.append({**row, "id": f"{row['id']}_R{rep:02d}"})
    return out

