from __future__ import annotations

import argparse
import re
import time
from collections import defaultdict
from pathlib import Path

from config import AgentConfig
from memory import AdaptiveModularMemory
from skills_store import SkillStore, slugify_skill_id

SKILL_PREFIX_RE = re.compile(r"^\[SKILL:\s*(?P<topic>[^\]]+)\]\s*(?P<body>.*)$")


def extract_topic_and_chunk(text: str) -> tuple[str, str]:
    match = SKILL_PREFIX_RE.match((text or "").strip())
    if not match:
        return "", text or ""
    topic_raw = match.group("topic")
    topic = topic_raw.split("\u00a7")[0].strip()
    return topic, match.group("body").strip()


def build_sections(topic: str, summary: str) -> dict:
    return {
        "Purpose": summary,
        "Preconditions": "- Legacy migrated skill. Review before publishing.",
        "Environment Variables": "- None declared in legacy memory. Add `${ENV_VAR}` placeholders if required.",
        "Endpoint": "Provider specific. Fill in explicit endpoint before publishing.",
        "Request Format": "Legacy migrated notes; convert to explicit request schema.",
        "Example Request": "```text\nNo canonical request payload was preserved in legacy chunks.\n```",
        "Example Response": "```text\nNo canonical response payload was preserved in legacy chunks.\n```",
        "Step-by-Step Procedure": (
            "1. Review migrated notes.\n"
            "2. Add concrete endpoint/auth details.\n"
            "3. Add request and response examples.\n"
            "4. Validate placeholders and failure modes."
        ),
        "Failure Modes": "- Missing structured request/response details due to legacy free-text storage.",
        "Validation Checklist": "- [ ] Skill reviewed and promoted to published.",
    }


def migrate(memory: AdaptiveModularMemory, store: SkillStore, pointer_type: str) -> tuple[int, int]:
    grouped = defaultdict(list)
    legacy_indices = defaultdict(list)

    with memory._lock:
        values = list(memory._values)
        metadata = list(memory._metadata)

    for idx, (text, meta) in enumerate(zip(values, metadata)):
        if (meta or {}).get("type") != "skill":
            continue
        topic, body = extract_topic_and_chunk(text)
        key = topic or (meta or {}).get("subject") or "legacy-skill"
        grouped[key].append((text, body, meta or {}))
        legacy_indices[key].append(idx)

    migrated = 0
    marked = 0
    for topic, rows in grouped.items():
        combined = " ".join(body for _text, body, _meta in rows).strip()
        summary = " ".join(combined.split())[:260] or f"Migrated legacy skill for {topic}."
        skill_id = f"{slugify_skill_id(topic)}-v1"
        source_urls = [r[2].get("source", "") for r in rows if r[2].get("source")]
        sections = build_sections(topic, summary)
        doc = store.create_or_update_draft(
            skill_id=skill_id,
            title=topic.title(),
            summary=summary,
            source_urls=source_urls,
            tags=[slugify_skill_id(topic)],
            requires_env=[],
            capabilities=["legacy_migrated"],
            sections=sections,
            owner="auto",
            trust_level="unverified",
        )

        pointer_text = f"[SKILL_REF] {doc.title} | {doc.skill_id} | {summary} | tags: migrated"
        pointer_meta = {
            "type": pointer_type,
            "subject": doc.skill_id,
            "skill_id": doc.skill_id,
            "skill_path": str(doc.path),
            "status": "draft",
            "version": doc.version,
            "summary": summary,
            "title": doc.title,
            "tags": ["migrated"],
            "timestamp": time.time(),
        }
        memory.upsert_by_meta(pointer_text, pointer_meta, match_keys=["type", "skill_id"])

        with memory._lock:
            for index in legacy_indices[topic]:
                meta = dict(memory._metadata[index] or {})
                meta["deprecated"] = True
                meta["replaced_by_skill_id"] = doc.skill_id
                memory._metadata[index] = meta
                marked += 1
            memory._dirty = True
            memory._version += 1

        migrated += 1

    if migrated:
        memory.flush()
    return migrated, marked


def main() -> int:
    parser = argparse.ArgumentParser(description="Migrate legacy AMM skill chunks to markdown skill drafts + pointer refs.")
    parser.add_argument("--memory-file", default="nexus_memory.json")
    args = parser.parse_args()

    cfg = AgentConfig()
    memory = AdaptiveModularMemory(
        model_name=cfg.memory_encoder,
        max_slots=cfg.max_memory_slots,
        save_path=args.memory_file,
        decay_enabled=cfg.memory_decay_enabled,
        decay_half_lives=cfg.memory_decay_half_lives,
        dedup_enabled=cfg.memory_dedup_enabled,
        dedup_scope=cfg.memory_dedup_scope,
        dedup_types=cfg.memory_dedup_types,
    )
    store = SkillStore(
        root_dir=cfg.skills_root_dir,
        drafts_dir=cfg.skills_drafts_dir,
        published_dir=cfg.skills_published_dir,
        require_env_placeholders=cfg.skills_require_env_placeholders,
    )

    migrated, marked = migrate(memory, store, cfg.memory_skill_pointer_type)
    print(f"migrated_skills={migrated}")
    print(f"marked_legacy_slots={marked}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
