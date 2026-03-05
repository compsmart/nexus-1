---
skill_id: image-gen-http-api-v1
title: Image Generation via HTTP API
status: published
version: 1.0.0
created_at: '2026-02-27T17:39:07.765354+00:00'
updated_at: '2026-02-27T17:39:08.242768+00:00'
source_urls:
- https://example.com/image-api-docs
tags:
- api
- generation
- image
owner: manual
trust_level: reviewed
requires_env:
- IMG_API_KEY
capabilities:
- api_call
- image_generation
- json_request
summary: Generate images by sending a prompt to an HTTP endpoint with bearer auth
  and JSON payload.
---

## Purpose
Generate images by sending a prompt to an HTTP endpoint with bearer auth and JSON payload.

## Preconditions
- Network access to the endpoint.
- Correct API permissions and auth scope.
- Request/response schema awareness.

## Environment Variables
- `${IMG_API_KEY}`

## Endpoint
`POST https://api.example.com/v1/images`

## Request Format
POST JSON with prompt, size, style and auth header.

## Example Request
```bash
curl -X POST "https://api.example.com/v1/images" \
  -H "Authorization: Bearer ${IMG_API_KEY}" \
  -H "Content-Type: application/json" \
  -d '{"prompt":"a cinematic mountain sunrise","size":"1024x1024"}'
```

## Example Response
```json
{"id":"img_123","status":"ok","url":"https://cdn.example.com/img_123.png"}
```

## Step-by-Step Procedure
1. Export ${IMG_API_KEY}.
2. Build JSON payload with prompt and image params.
3. Call POST endpoint.
4. Verify status and parse URL.
5. Retry on 429/5xx with backoff.

## Failure Modes
- Missing/invalid IMG_API_KEY (401/403).
- Rate limiting (429).
- Invalid schema (400).
- Timeout/network errors.

## Validation Checklist
- [ ] Uses `${ENV_VAR}` placeholders only.
- [ ] Request and response examples are valid.
- [ ] Error handling is documented.
