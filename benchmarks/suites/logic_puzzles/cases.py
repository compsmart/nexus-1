from __future__ import annotations

from typing import Dict, List


def puzzle_cases(include_hard_set: bool = False) -> List[Dict]:
    rows = [
        {
            "id": "LP01",
            "question": "A bat and ball cost $1.10 total. Bat costs $1 more than ball. Ball cost?",
            "expected": "0.05",
            "keywords": ["0.05"],
        },
        {
            "id": "LP02",
            "question": "What comes once in a minute, twice in a moment, never in a thousand years?",
            "expected": "m",
            "keywords": ["m"],
        },
        {
            "id": "LP03",
            "question": "If five machines make five widgets in five minutes, how long for 100 machines to make 100 widgets?",
            "expected": "5",
            "keywords": ["5"],
        },
        {
            "id": "LP04",
            "question": "True or false: If all bloops are razzies and all razzies are lazzies, all bloops are lazzies.",
            "expected": "true",
            "keywords": ["true"],
        },
    ]
    if include_hard_set:
        rows.extend(
            [
                {
                    "id": "LP05",
                    "question": "You have two ropes that each burn in 60 minutes non-uniformly. Measure 45 minutes.",
                    "expected": "light one rope both ends and the other one end",
                    "keywords": ["both", "ends", "other", "end"],
                },
                {
                    "id": "LP06",
                    "question": "Which number is next: 2, 3, 5, 9, 17, ?",
                    "expected": "33",
                    "keywords": ["33"],
                },
            ]
        )
    return rows

