# dealmatch (API + worker)

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

## Notes

Legacy scripts from the OpenClaw workspace remain at repo root for now; the new system lives under `app/`, `worker/`, `dealmatch/`.
