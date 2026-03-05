from __future__ import annotations

import json
from typing import Dict, List


def normalize_hotpot_row(row: Dict, docs_per_case: int = 8) -> Dict:
    context = row.get("context", {})
    titles = list((context or {}).get("title", []))
    sentence_groups = list((context or {}).get("sentences", []))
    docs: List[str] = []
    for title, sents in zip(titles, sentence_groups):
        docs.append(f"{title}: {' '.join(sents)}")
    docs = docs[:docs_per_case]
    return {
        "question": row.get("question", ""),
        "answer": row.get("answer", ""),
        "context_docs": docs,
        "dataset": "hotpot_qa",
    }


def normalize_two_wiki_row(row: Dict, docs_per_case: int = 8) -> Dict:
    raw_context = row.get("context", "")
    docs: List[str] = []
    if isinstance(raw_context, str):
        try:
            parsed = json.loads(raw_context)
            if isinstance(parsed, list):
                for element in parsed:
                    if isinstance(element, list) and len(element) >= 2:
                        title = str(element[0])
                        body = element[1]
                        if isinstance(body, list):
                            docs.append(f"{title}: {' '.join(str(x) for x in body)}")
                        else:
                            docs.append(f"{title}: {str(body)}")
                    else:
                        docs.append(str(element))
            else:
                docs = [str(parsed)]
        except Exception:
            docs = [raw_context]
    elif isinstance(raw_context, list):
        docs = [str(x) for x in raw_context]
    else:
        docs = [str(raw_context)]
    docs = docs[:docs_per_case]
    return {
        "question": row.get("question", ""),
        "answer": row.get("answer", ""),
        "context_docs": docs,
        "dataset": "two_wiki",
    }

