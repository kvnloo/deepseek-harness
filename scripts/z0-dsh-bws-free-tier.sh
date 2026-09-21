#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PATCH="apps/cli/config/examples/z0-free-tier/cordis.patch.yml"

if ! command -v bws >/dev/null 2>&1; then
  echo "error: bws is not installed or not on PATH" >&2
  exit 127
fi

if [[ -z "${BWS_ACCESS_TOKEN:-}" ]]; then
  echo "error: BWS_ACCESS_TOKEN is not available to this shell" >&2
  echo "load the machine-account token from the existing OS-keyring/bootstrap path; do not put it in this repo" >&2
  exit 2
fi

if [[ -z "${BWS_DSH_PROJECT_ID:-}" ]]; then
  echo "error: BWS_DSH_PROJECT_ID is required" >&2
  echo "use a least-privilege BWS project containing only GROQ_API_KEY and CEREBRAS_API_KEY" >&2
  exit 2
fi

cd "$ROOT"

# bws run maps POSIX-compliant secret names directly into the child process.
# The checks occur inside the injected shell and never print either value.
exec bws run --project-id "$BWS_DSH_PROJECT_ID" -- \
  'test -n "$GROQ_API_KEY" || { echo "error: GROQ_API_KEY missing from BWS project" >&2; exit 3; };    test -n "$CEREBRAS_API_KEY" || { echo "error: CEREBRAS_API_KEY missing from BWS project" >&2; exit 3; };    exec pnpm dsh web --patch apps/cli/config/examples/z0-free-tier/cordis.patch.yml'
