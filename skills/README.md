# Skill Catalog

Canonical skills are stored on disk as markdown files.

## Layout
- `drafts/` auto-generated or manually authored skills pending promotion.
- `published/` reviewed skills eligible for autonomous use.
- `archive/` previous published revisions.
- `templates/skill_template.md` authoring template.
- `index.json` lightweight index (rebuildable).

## Security Policy
- Do not store raw credentials in skill files.
- Use environment placeholders only, for example `${IMG_API_KEY}`.
- Publishing is blocked if inline secrets are detected.

## Frontmatter
Required fields:
- `skill_id`
- `title`
- `status` (`draft|published|archived`)
- `version`
- `created_at`
- `updated_at`
- `source_urls`
- `tags`
- `owner` (`auto|manual`)
- `trust_level` (`unverified|reviewed`)
- `requires_env`
- `capabilities`

## Required Sections
- `## Purpose`
- `## Preconditions`
- `## Environment Variables`
- `## Endpoint`
- `## Request Format`
- `## Example Request`
- `## Example Response`
- `## Step-by-Step Procedure`
- `## Failure Modes`
- `## Validation Checklist`
