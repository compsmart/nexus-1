from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import yaml

REQUIRED_FRONTMATTER_FIELDS = {
    "skill_id",
    "title",
    "status",
    "version",
    "created_at",
    "updated_at",
    "source_urls",
    "tags",
    "owner",
    "trust_level",
    "requires_env",
    "capabilities",
}

REQUIRED_SECTIONS = [
    "Purpose",
    "Preconditions",
    "Environment Variables",
    "Endpoint",
    "Request Format",
    "Example Request",
    "Example Response",
    "Step-by-Step Procedure",
    "Failure Modes",
    "Validation Checklist",
]

EXCERPT_SECTION_ORDER = [
    "Purpose",
    "Step-by-Step Procedure",
    "Request Format",
    "Example Request",
    "Example Response",
    "Failure Modes",
]

SECTION_HEADER_RE = re.compile(r"^##\s+(.+?)\s*$", re.MULTILINE)
FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n(.*)$", re.DOTALL)
ENV_PLACEHOLDER_RE = re.compile(r"\$\{([A-Z][A-Z0-9_]{1,63})\}")

# Keep these high-signal to avoid false positives while still catching obvious leaks.
INLINE_SECRET_PATTERNS = [
    re.compile(r"\bsk-[A-Za-z0-9]{16,}\b"),
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    re.compile(r"\b(?:api[_-]?key|secret|token)\b\s*[:=]\s*['\"]?[A-Za-z0-9_\-]{12,}" , re.IGNORECASE),
]


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def slugify_skill_id(text: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", (text or "").strip().lower()).strip("-")
    return slug or "skill"


def next_patch_version(version: str) -> str:
    parts = (version or "1.0.0").split(".")
    if len(parts) != 3 or not all(p.isdigit() for p in parts):
        return "1.0.0"
    major, minor, patch = (int(p) for p in parts)
    return f"{major}.{minor}.{patch + 1}"


def strip_frontmatter(markdown: str) -> tuple[Dict, str]:
    match = FRONTMATTER_RE.match(markdown.strip())
    if not match:
        raise ValueError("Skill markdown must include YAML frontmatter.")
    raw_frontmatter = match.group(1)
    body = match.group(2)
    frontmatter = yaml.safe_load(raw_frontmatter) or {}
    if not isinstance(frontmatter, dict):
        raise ValueError("YAML frontmatter must parse to a mapping.")
    return frontmatter, body


def split_sections(body: str) -> Dict[str, str]:
    matches = list(SECTION_HEADER_RE.finditer(body))
    sections: Dict[str, str] = {}
    if not matches:
        return sections

    for idx, header in enumerate(matches):
        section_name = header.group(1).strip()
        start = header.end()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(body)
        content = body[start:end].strip()
        sections[section_name] = content
    return sections


def redact_sensitive_text(text: str) -> str:
    redacted = text
    for pattern in INLINE_SECRET_PATTERNS:
        redacted = pattern.sub("[REDACTED]", redacted)
    return redacted


def has_inline_secret(text: str) -> bool:
    scrubbed = ENV_PLACEHOLDER_RE.sub("", text)
    return any(pattern.search(scrubbed) for pattern in INLINE_SECRET_PATTERNS)


@dataclass
class SkillDocument:
    skill_id: str
    title: str
    status: str
    version: str
    path: Path
    frontmatter: Dict
    sections: Dict[str, str]


class SkillStore:
    def __init__(
        self,
        root_dir: str = "skills",
        drafts_dir: str = "skills/drafts",
        published_dir: str = "skills/published",
        archive_dir: str = "skills/archive",
        index_path: Optional[str] = None,
        require_env_placeholders: bool = True,
    ) -> None:
        self.root_dir = Path(root_dir)
        self.drafts_dir = Path(drafts_dir)
        self.published_dir = Path(published_dir)
        self.archive_dir = Path(archive_dir)
        self.index_path = Path(index_path) if index_path else self.root_dir / "index.json"
        self.require_env_placeholders = require_env_placeholders

        self.root_dir.mkdir(parents=True, exist_ok=True)
        self.drafts_dir.mkdir(parents=True, exist_ok=True)
        self.published_dir.mkdir(parents=True, exist_ok=True)
        self.archive_dir.mkdir(parents=True, exist_ok=True)

    def _skill_path(self, skill_id: str, status: str) -> Path:
        if status == "published":
            return self.published_dir / f"{skill_id}.md"
        if status == "draft":
            return self.drafts_dir / f"{skill_id}.md"
        if status == "archived":
            return self.archive_dir / f"{skill_id}.md"
        raise ValueError(f"Unknown status '{status}'.")

    def _validate_frontmatter(self, frontmatter: Dict) -> List[str]:
        errors: List[str] = []
        missing = REQUIRED_FRONTMATTER_FIELDS - set(frontmatter.keys())
        if missing:
            errors.append(f"Missing frontmatter fields: {sorted(missing)}")

        if frontmatter.get("status") not in {"draft", "published", "archived"}:
            errors.append("status must be one of: draft, published, archived")
        if frontmatter.get("owner") not in {"auto", "manual"}:
            errors.append("owner must be 'auto' or 'manual'")
        if frontmatter.get("trust_level") not in {"unverified", "reviewed"}:
            errors.append("trust_level must be 'unverified' or 'reviewed'")

        for field in ("source_urls", "tags", "requires_env", "capabilities"):
            if not isinstance(frontmatter.get(field), list):
                errors.append(f"{field} must be a list")

        return errors

    def _validate_sections(self, sections: Dict[str, str]) -> List[str]:
        errors: List[str] = []
        for required in REQUIRED_SECTIONS:
            if required not in sections:
                errors.append(f"Missing section: {required}")
        return errors

    def _validate_placeholders(self, frontmatter: Dict, sections: Dict[str, str], markdown: str) -> List[str]:
        errors: List[str] = []
        if has_inline_secret(markdown):
            errors.append("Skill content contains inline secrets. Use ${ENV_VAR} placeholders only.")

        if not self.require_env_placeholders:
            return errors

        env_section = sections.get("Environment Variables", "")
        required_env = [str(v).strip() for v in frontmatter.get("requires_env", []) if str(v).strip()]
        for var_name in required_env:
            placeholder = f"${{{var_name}}}"
            if placeholder not in env_section:
                errors.append(
                    f"requires_env includes '{var_name}' but Environment Variables section is missing '{placeholder}'."
                )
        return errors

    def validate_markdown(self, markdown: str) -> List[str]:
        try:
            frontmatter, body = strip_frontmatter(markdown)
        except Exception as exc:
            return [str(exc)]

        sections = split_sections(body)
        errors = []
        errors.extend(self._validate_frontmatter(frontmatter))
        errors.extend(self._validate_sections(sections))
        errors.extend(self._validate_placeholders(frontmatter, sections, markdown))
        return errors

    def _render_markdown(self, frontmatter: Dict, sections: Dict[str, str]) -> str:
        ordered_frontmatter = dict(frontmatter)
        frontmatter_yaml = yaml.safe_dump(ordered_frontmatter, sort_keys=False, allow_unicode=False).strip()

        lines = ["---", frontmatter_yaml, "---", ""]
        for heading in REQUIRED_SECTIONS:
            lines.append(f"## {heading}")
            lines.append((sections.get(heading, "") or "").strip())
            lines.append("")
        return "\n".join(lines).strip() + "\n"

    def load_skill(
        self,
        skill_id: str,
        include_drafts: bool = True,
        include_published: bool = True,
    ) -> Optional[SkillDocument]:
        candidates: List[Path] = []
        if include_published:
            candidates.append(self.published_dir / f"{skill_id}.md")
        if include_drafts:
            candidates.append(self.drafts_dir / f"{skill_id}.md")

        for path in candidates:
            if path.exists():
                return self._load_by_path(path)
        return None

    def _load_by_path(self, path: Path) -> SkillDocument:
        raw = path.read_text(encoding="utf-8")
        frontmatter, body = strip_frontmatter(raw)
        sections = split_sections(body)
        return SkillDocument(
            skill_id=str(frontmatter.get("skill_id", "")),
            title=str(frontmatter.get("title", "")),
            status=str(frontmatter.get("status", "draft")),
            version=str(frontmatter.get("version", "1.0.0")),
            path=path,
            frontmatter=frontmatter,
            sections=sections,
        )

    def list_skills(self, include_drafts: bool = True, include_published: bool = True) -> List[SkillDocument]:
        docs: List[SkillDocument] = []
        if include_published:
            for path in sorted(self.published_dir.glob("*.md")):
                docs.append(self._load_by_path(path))
        if include_drafts:
            for path in sorted(self.drafts_dir.glob("*.md")):
                docs.append(self._load_by_path(path))
        docs.sort(key=lambda d: (d.skill_id, d.status))
        return docs

    def create_or_update_draft(
        self,
        skill_id: str,
        title: str,
        summary: str,
        source_urls: Iterable[str],
        tags: Iterable[str],
        requires_env: Iterable[str],
        capabilities: Iterable[str],
        sections: Dict[str, str],
        owner: str = "auto",
        trust_level: str = "unverified",
    ) -> SkillDocument:
        existing = self.load_skill(skill_id, include_drafts=True, include_published=True)
        now = utc_now_iso()
        version = next_patch_version(existing.version) if existing else "1.0.0"
        created_at = existing.frontmatter.get("created_at", now) if existing else now

        frontmatter = {
            "skill_id": skill_id,
            "title": title,
            "status": "draft",
            "version": version,
            "created_at": created_at,
            "updated_at": now,
            "source_urls": sorted(set(source_urls)),
            "tags": sorted(set(t for t in tags if t)),
            "owner": owner,
            "trust_level": trust_level,
            "requires_env": sorted(set(v for v in requires_env if v)),
            "capabilities": sorted(set(c for c in capabilities if c)),
            "summary": summary,
        }

        markdown = self._render_markdown(frontmatter, sections)
        errors = self.validate_markdown(markdown)
        if errors:
            raise ValueError("; ".join(errors))

        draft_path = self.drafts_dir / f"{skill_id}.md"
        draft_path.write_text(markdown, encoding="utf-8")
        self._write_index()
        return self._load_by_path(draft_path)

    def publish_skill(self, skill_id: str) -> SkillDocument:
        draft = self.load_skill(skill_id, include_drafts=True, include_published=False)
        if draft is None:
            raise FileNotFoundError(f"Draft skill not found: {skill_id}")

        existing_published = self.published_dir / f"{skill_id}.md"
        if existing_published.exists():
            archive_name = f"{skill_id}-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}.md"
            existing_published.replace(self.archive_dir / archive_name)

        frontmatter = dict(draft.frontmatter)
        frontmatter["status"] = "published"
        frontmatter["trust_level"] = "reviewed"
        frontmatter["updated_at"] = utc_now_iso()

        markdown = self._render_markdown(frontmatter, draft.sections)
        errors = self.validate_markdown(markdown)
        if errors:
            raise ValueError("; ".join(errors))

        published_path = self.published_dir / f"{skill_id}.md"
        published_path.write_text(markdown, encoding="utf-8")
        if draft.path.exists():
            draft.path.unlink()

        self._write_index()
        return self._load_by_path(published_path)

    def render_excerpt(
        self,
        doc: SkillDocument,
        max_chars: int = 3000,
        sections: Optional[Iterable[str]] = None,
        redact_sensitive: bool = True,
    ) -> str:
        selected = list(sections) if sections else EXCERPT_SECTION_ORDER
        parts: List[str] = []
        for name in selected:
            content = doc.sections.get(name, "").strip()
            if not content:
                continue
            parts.append(f"## {name}\n{content}")

        excerpt = "\n\n".join(parts).strip()
        if redact_sensitive:
            excerpt = redact_sensitive_text(excerpt)
        if len(excerpt) > max_chars:
            return excerpt[:max_chars].rstrip() + "\n... [truncated]"
        return excerpt

    def _write_index(self) -> None:
        rows = []
        for doc in self.list_skills(include_drafts=True, include_published=True):
            rows.append(
                {
                    "skill_id": doc.skill_id,
                    "title": doc.title,
                    "status": doc.status,
                    "version": doc.version,
                    "path": str(doc.path),
                    "updated_at": doc.frontmatter.get("updated_at", ""),
                    "tags": doc.frontmatter.get("tags", []),
                }
            )
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        self.index_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
