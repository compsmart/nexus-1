from __future__ import annotations

from typing import Dict, List


def _base_cases() -> List[Dict]:
    return [
        {
            "case_id": "G001",
            "category": "multi_hop",
            "question": "Which city is the lead of Project Atlas based in?",
            "expected": "Lisbon",
            "keywords": ["lisbon"],
            "context_docs": [
                "Project Atlas is led by Dana.",
                "Dana is based in Lisbon.",
            ],
        },
        {
            "case_id": "G002",
            "category": "memory",
            "question": "What is my dog's name?",
            "expected": "Louis",
            "keywords": ["louis"],
            "context_docs": [
                "User personal fact: my dog is called Louis.",
            ],
        },
        {
            "case_id": "G003",
            "category": "reasoning",
            "question": "If all glibs are truds and no truds are silent, can any glib be silent?",
            "expected": "No",
            "keywords": ["no"],
            "context_docs": [],
        },
        {
            "case_id": "G004",
            "category": "learning",
            "question": "Using the rule from context, what is CODE(planet)?",
            "expected": "QMBOFU",
            "keywords": ["qmbofu"],
            "context_docs": [
                "Rule: CODE(x) shifts every alphabetic character by +1 and uppercases the result.",
                "Example: CODE(cat) = DBU",
            ],
        },
        {
            "case_id": "G005",
            "category": "multi_hop",
            "question": "Where does the manager of Project Orion work from?",
            "expected": "Nairobi",
            "keywords": ["nairobi"],
            "context_docs": [
                "Project Orion is managed by Malik.",
                "Malik works from Nairobi.",
            ],
        },
        {
            "case_id": "G006",
            "category": "memory",
            "question": "What is my oldest son's name?",
            "expected": "Jaxon",
            "keywords": ["jaxon"],
            "context_docs": [
                "User personal fact: my youngest son is Max.",
                "User personal fact: my oldest son is Jaxon.",
            ],
        },
        {
            "case_id": "G007",
            "category": "reasoning",
            "question": "A farmer has 17 sheep and all but 9 run away. How many are left?",
            "expected": "9",
            "keywords": ["9"],
            "context_docs": [],
        },
        {
            "case_id": "G008",
            "category": "learning",
            "question": "Apply the context rule: respond with the exact protocol token.",
            "expected": "channel-7",
            "keywords": ["channel-7"],
            "context_docs": [
                "Policy: if asked for the emergency protocol token, output exactly 'channel-7'.",
            ],
        },
    ]


def load_cases(num_cases: int) -> List[Dict]:
    base = _base_cases()
    if num_cases <= len(base):
        return base[:num_cases]
    out: List[Dict] = []
    idx = 0
    while len(out) < num_cases:
        row = dict(base[idx % len(base)])
        row["case_id"] = f"{row['case_id']}_R{idx // len(base):02d}"
        out.append(row)
        idx += 1
    return out

