# Agentic Researcher (API + worker)

Deployable human-in-the-loop domain discovery + verification + graph building.

## Local dev (quick)

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

export DATABASE_URL="postgresql+psycopg://postgres:postgres@localhost:5432/dealmatch"
export OPENAI_API_KEY=... 
export SERPER_API_KEY=...

uvicorn app.main:app --reload

# in another terminal
python -m worker.run
```

## Environment variables

- `DATABASE_URL` (Postgres)
- `OPENAI_API_KEY`
- `OPENAI_MODEL_STRATEGY` (optional)
- `SERPER_API_KEY`

## Render deployment

This repo includes a `render.yaml` blueprint that creates:
- a Postgres database
- a Web service (FastAPI)
- a Worker service (background tasks)

On Render: **New → Blueprint** and select this repo.

You must set secrets:
- `OPENAI_API_KEY`
- `SERPER_API_KEY`

Optional overrides:
- `OPENAI_PLANNER_MODEL` (default `gpt-5`)
- `OPENAI_WORKER_MODEL` (default `gpt-4.1-mini`)

## Notes

Legacy scripts from the OpenClaw workspace remain at repo root for now; the new system lives under `app/`, `worker/`, `dealmatch/`.
