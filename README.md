# NAICS Classifier API (NAICS 2017)

Hybrid **vector search (OpenAI embeddings)** + **LLM re-rank (OpenAI chat)** over NAICS code pages scraped from naics.com.

## Quick start (local)

```bash
cd /home/curl/.openclaw/workspace-dealmatch

# 1) install deps
npm install

# 2) build embeddings index (writes out/naics-2017-embeddings.jsonl)
export OPENAI_API_KEY="sk-..."
npm run build:index

# 3) run API server
export PORT=8787
npm start
```

Open the local UI:
- http://localhost:8787/

## API

### Health

`GET /health`

Example:
```bash
curl -sS https://YOUR_HOST/health
```

### Classify

`POST /classify`

Request JSON:
```json
{ "text": "<business description, up to 10,000 chars>" }
```

Response JSON:
```json
{
  "top": [
    { "code": "238160", "title": "Roofing Contractors", "score": 100, "source_url": "..." },
    { "code": "238170", "title": "Siding Contractors", "score": 42, "source_url": "..." },
    { "code": "238110", "title": "Poured Concrete Foundation and Structure Contractors", "score": 19, "source_url": "..." }
  ]
}
```

Example curl:
```bash
curl -sS -X POST https://YOUR_HOST/classify \
  -H 'content-type: application/json' \
  -d '{"text":"We install and repair residential and commercial roofs, including skylights, metal roofing, and roof coatings."}'
```

## Render deploy

This repo is ready for Render as a **Web Service**.

### 1) Create the service
- Root Directory: (repo root)
- Build Command:
  ```bash
  npm ci && npm run build:index
  ```
- Start Command:
  ```bash
  npm start
  ```

### 2) Environment variables (Render dashboard)
Set:
- `OPENAI_API_KEY` (required)

Optional:
- `EMBED_MODEL` (default `text-embedding-3-small`)
- `LLM_MODEL` (default `gpt-4o-mini`)
- `TOPK` (default `25`)

### Notes
- The embeddings index is built during Render build. That means Render must have `OPENAI_API_KEY` available during build.
- If you prefer faster deploys and no build-time API usage, you can prebuild `out/naics-2017-embeddings.jsonl` locally and commit it, then change the Render Build Command to `npm ci`.
