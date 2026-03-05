---
skill_id: sample-skill-v1
title: Sample Skill
status: draft
version: 1.0.0
created_at: 2026-01-01T00:00:00+00:00
updated_at: 2026-01-01T00:00:00+00:00
source_urls:
  - https://example.com
tags:
  - sample
owner: manual
trust_level: unverified
requires_env:
  - SAMPLE_API_KEY
capabilities:
  - api_call
summary: One line summary of what this skill does.
---

## Purpose
Describe what this skill solves.

## Preconditions
- Required tooling
- Required permissions

## Environment Variables
- `${SAMPLE_API_KEY}`

## Endpoint
`POST https://api.example.com/v1/action`

## Request Format
JSON payload with required fields and auth header.

## Example Request
```bash
curl -X POST "https://api.example.com/v1/action" \
  -H "Authorization: Bearer ${SAMPLE_API_KEY}" \
  -H "Content-Type: application/json" \
  -d '{"prompt":"hello"}'
```

## Example Response
```json
{"id":"123","status":"ok"}
```

## Step-by-Step Procedure
1. Prepare environment variables.
2. Validate endpoint and payload.
3. Send request.
4. Parse response.
5. Handle retries on transient failures.

## Failure Modes
- Invalid credentials.
- Rate limiting.
- Endpoint timeout.

## Validation Checklist
- [ ] Uses env placeholders only.
- [ ] Request and response examples are valid.
- [ ] Failure handling is documented.
