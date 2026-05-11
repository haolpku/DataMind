#!/usr/bin/env bash
# Start a local `claude-code-router` instance for the SDK backend.
#
# Usage:
#   ./scripts/start_ccr.sh
#
# Env vars (all have defaults; override as needed):
#   CCR_PORT          listen port (default: 13456)
#   CCR_HOME          CCR's config dir (default: /tmp/ccr-home)
#   CCR_SERVER_ENTRY  path to CCR's bundled server index.js
#   UPSTREAM_BASE     OpenAI-compatible gateway URL
#   UPSTREAM_KEY      API key for the upstream gateway
#   UPSTREAM_MODEL    primary model name (default: claude-sonnet-4-6)
#   UPSTREAM_FALLBACK cheap model for background/think routes
#
# When DataMind runs with DATAMIND__AGENT__BACKEND=sdk, its agent loop
# talks to CCR on http://127.0.0.1:$CCR_PORT via the Anthropic
# /v1/messages protocol, and CCR translates to OpenAI format before
# forwarding to the real upstream.
#
# On exit (Ctrl-C or SIGTERM), CCR is killed cleanly.

set -euo pipefail

CCR_PORT="${CCR_PORT:-13456}"
CCR_HOME="${CCR_HOME:-/tmp/ccr-home}"

UPSTREAM_BASE="${UPSTREAM_BASE:?must set UPSTREAM_BASE (OpenAI-compatible gateway URL)}"
UPSTREAM_KEY="${UPSTREAM_KEY:?must set UPSTREAM_KEY}"
UPSTREAM_MODEL="${UPSTREAM_MODEL:-claude-sonnet-4-6}"
UPSTREAM_FALLBACK="${UPSTREAM_FALLBACK:-claude-haiku-4-5-20251001}"

# Default: use the vendored CCR inside KDD-CUP-DataAgent (sibling repo).
# If you cloned DataMind standalone, install CCR separately and point
# CCR_SERVER_ENTRY at its packages/server/dist/index.js.
DEFAULT_ENTRY="/Users/lianghao/Desktop/DataMind/KDD-CUP-DataAgent/vendor/claude-code-router/packages/server/dist/index.js"
CCR_SERVER_ENTRY="${CCR_SERVER_ENTRY:-$DEFAULT_ENTRY}"

if [[ ! -f "$CCR_SERVER_ENTRY" ]]; then
  echo "[fatal] CCR server entry not found: $CCR_SERVER_ENTRY" >&2
  echo "[hint]  set CCR_SERVER_ENTRY to the path of claude-code-router's packages/server/dist/index.js" >&2
  exit 2
fi

if ! command -v node >/dev/null 2>&1; then
  echo "[fatal] node is required but not on PATH" >&2
  exit 2
fi

mkdir -p "$CCR_HOME/.claude-code-router/logs"

# Normalise upstream URL — CCR's transformer expects /v1/chat/completions.
BASE_NORM="${UPSTREAM_BASE%/}"
case "$BASE_NORM" in
  */v1/chat/completions) ;;                          # already correct
  */v1/messages)         BASE_NORM="${BASE_NORM%/v1/messages}/v1/chat/completions" ;;
  */v1)                  BASE_NORM="${BASE_NORM}/chat/completions" ;;
  *)                     BASE_NORM="${BASE_NORM}/v1/chat/completions" ;;
esac

cat > "$CCR_HOME/.claude-code-router/config.json" <<JSON
{
  "HOST": "127.0.0.1",
  "PORT": ${CCR_PORT},
  "APIKEY": "",
  "LOG": true,
  "LOG_LEVEL": "info",
  "API_TIMEOUT_MS": 600000,
  "Providers": [
    {
      "name": "openai",
      "api_base_url": "${BASE_NORM}",
      "api_key": "${UPSTREAM_KEY}",
      "models": ["${UPSTREAM_MODEL}", "${UPSTREAM_FALLBACK}"],
      "transformer": {"use": ["anthropic"]}
    }
  ],
  "Router": {
    "default": "openai,${UPSTREAM_MODEL}",
    "background": "openai,${UPSTREAM_FALLBACK}",
    "think": "openai,${UPSTREAM_MODEL}",
    "longContext": "openai,${UPSTREAM_MODEL}",
    "webSearch": "openai,${UPSTREAM_MODEL}",
    "image": "openai,${UPSTREAM_MODEL}"
  }
}
JSON

echo "[ccr] upstream = ${BASE_NORM}"
echo "[ccr] listen   = http://127.0.0.1:${CCR_PORT}"
echo "[ccr] config   = ${CCR_HOME}/.claude-code-router/config.json"

# Kill any stale CCR on the same port.
if lsof -ti:"${CCR_PORT}" >/dev/null 2>&1; then
  echo "[ccr] port ${CCR_PORT} busy — killing previous instance"
  lsof -ti:"${CCR_PORT}" | xargs kill -9 2>/dev/null || true
  sleep 0.5
fi

# Run in foreground — Ctrl-C stops CCR cleanly.
HOME="$CCR_HOME" \
  HTTP_PROXY="" HTTPS_PROXY="" ALL_PROXY="" \
  http_proxy="" https_proxy="" all_proxy="" \
  exec node "$CCR_SERVER_ENTRY"
