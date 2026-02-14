#!/usr/bin/env bash
set -euo pipefail

# Run DB migrations (best-effort for now)
if command -v alembic >/dev/null 2>&1; then
  echo "Running alembic migrations..." >&2
  alembic upgrade head || true
fi

# Ensure base tables exist (fresh DB bootstrap)
python -m dealmatch.db.init_db || true

exec uvicorn app.main:app --host 0.0.0.0 --port "${PORT:-8000}"
