from __future__ import annotations

from typing import Dict, List


def seed_facts() -> List[str]:
    return [
        "User identity: name=Alex",
        "User personal fact: my dog is called Bruno",
        "User personal fact: my cat is called Luna",
        "User personal fact: my favourite colour is dark green",
        "User personal fact: my job is software engineer",
        "User personal fact: I live in Edinburgh",
        "User personal fact: my oldest son is Jaxon",
        "User personal fact: my youngest son is Max",
    ]


def recall_questions(include_indirect_questions: bool = True) -> List[Dict]:
    rows = [
        {"id": "MR01", "q": "What is my name?", "keywords": ["alex"]},
        {"id": "MR02", "q": "What is my dog called?", "keywords": ["bruno"]},
        {"id": "MR03", "q": "What is my cat called?", "keywords": ["luna"]},
        {"id": "MR04", "q": "What is my favourite colour?", "keywords": ["green"]},
        {"id": "MR05", "q": "What do I do for work?", "keywords": ["software", "engineer"]},
        {"id": "MR06", "q": "Where do I live?", "keywords": ["edinburgh"]},
    ]
    if include_indirect_questions:
        rows.extend(
            [
                {"id": "MR07", "q": "Tell me my oldest son's name.", "keywords": ["jaxon"]},
                {"id": "MR08", "q": "Who is my youngest son?", "keywords": ["max"]},
            ]
        )
    return rows

