#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH="/app:${PYTHONPATH:-}"

# Ensure schema exists before worker starts pulling tasks
if command -v alembic >/dev/null 2>&1; then
  echo "Running alembic migrations..." >&2
  alembic upgrade head || true
fi

# Ensure base tables exist (fresh DB bootstrap)
python -m dealmatch.db.init_db || true

exec python -m worker.run
