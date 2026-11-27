#!/usr/bin/env bash
set -euo pipefail

PORT="${PORT:-8888}"
if [[ -z "${JUPYTER_TOKEN:-}" ]]; then
  JUPYTER_TOKEN="$(python - <<'PY'
import secrets
print(secrets.token_urlsafe(24))
PY
)"
  echo "Generated one-session Jupyter token: ${JUPYTER_TOKEN}"
else
  echo "Using JUPYTER_TOKEN from the environment."
fi

exec uv run jupyter server \
  --no-browser \
  --ip=127.0.0.1 \
  --port="${PORT}" \
  --ServerApp.allow_remote_access=false \
  --ServerApp.token="${JUPYTER_TOKEN}" \
  --ServerApp.password='' \
  --ServerApp.root_dir="${JUPYTER_ROOT:-$PWD}"
