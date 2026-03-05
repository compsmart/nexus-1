from __future__ import annotations

from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from skills_store import SkillStore


def make_sections() -> dict:
    return {
        "Purpose": "Generate an image via API endpoint.",
        "Preconditions": "- API access enabled.",
        "Environment Variables": "- `${IMG_API_KEY}`",
        "Endpoint": "`POST https://api.example.com/v1/images`",
        "Request Format": "JSON body with prompt.",
        "Example Request": "```bash\ncurl -H \"Authorization: Bearer ${IMG_API_KEY}\"\n```",
        "Example Response": "```json\n{\"id\":\"img_1\"}\n```",
        "Step-by-Step Procedure": "1. Build payload.\n2. Call endpoint.",
        "Failure Modes": "- 401 auth failure.",
        "Validation Checklist": "- [ ] Placeholder-only credentials.",
    }


def test_create_validate_and_publish_skill(tmp_path: Path) -> None:
    store = SkillStore(
        root_dir=str(tmp_path / "skills"),
        drafts_dir=str(tmp_path / "skills" / "drafts"),
        published_dir=str(tmp_path / "skills" / "published"),
        archive_dir=str(tmp_path / "skills" / "archive"),
        require_env_placeholders=True,
    )

    doc = store.create_or_update_draft(
        skill_id="image-gen-openai-v1",
        title="Image Generation API",
        summary="Generate images through an API call.",
        source_urls=["https://example.com/docs"],
        tags=["image", "api"],
        requires_env=["IMG_API_KEY"],
        capabilities=["api_call", "image_generation"],
        sections=make_sections(),
    )
    assert doc.status == "draft"
    assert doc.path.exists()

    published = store.publish_skill("image-gen-openai-v1")
    assert published.status == "published"
    assert "published" in str(published.path)


def test_inline_secret_rejected(tmp_path: Path) -> None:
    store = SkillStore(
        root_dir=str(tmp_path / "skills"),
        drafts_dir=str(tmp_path / "skills" / "drafts"),
        published_dir=str(tmp_path / "skills" / "published"),
        archive_dir=str(tmp_path / "skills" / "archive"),
        require_env_placeholders=True,
    )
    bad_sections = make_sections()
    bad_sections["Example Request"] = "api_key=supersecretkeyvalue12345"

    with pytest.raises(ValueError):
        store.create_or_update_draft(
            skill_id="bad-skill-v1",
            title="Bad Skill",
            summary="Should fail",
            source_urls=["https://example.com"],
            tags=["bad"],
            requires_env=[],
            capabilities=["api_call"],
            sections=bad_sections,
        )


def test_excerpt_limit(tmp_path: Path) -> None:
    store = SkillStore(
        root_dir=str(tmp_path / "skills"),
        drafts_dir=str(tmp_path / "skills" / "drafts"),
        published_dir=str(tmp_path / "skills" / "published"),
        archive_dir=str(tmp_path / "skills" / "archive"),
        require_env_placeholders=True,
    )
    sections = make_sections()
    sections["Step-by-Step Procedure"] = "\n".join([f"{i}. step" for i in range(1, 200)])
    doc = store.create_or_update_draft(
        skill_id="long-skill-v1",
        title="Long Skill",
        summary="Long summary",
        source_urls=["https://example.com/docs"],
        tags=["long"],
        requires_env=["IMG_API_KEY"],
        capabilities=["api_call"],
        sections=sections,
    )
    excerpt = store.render_excerpt(doc, max_chars=240)
    assert len(excerpt) <= 260
    assert "truncated" in excerpt.lower()
