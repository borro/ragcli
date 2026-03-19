#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ARTIFACT_DIR="${RAGCLI_MUTATION_ARTIFACT_DIR:-$ROOT_DIR/.artifacts/mutation}"
OUTPUT_PATH="${RAGCLI_MUTATION_OUTPUT:-$ARTIFACT_DIR/mutation-report.json}"
GREMLINS_VERSION="${RAGCLI_MUTATION_GREMLINS_VERSION:-v0.6.0}"
OUTPUT_STATUSES="${RAGCLI_MUTATION_OUTPUT_STATUSES:-lctv}"

mkdir -p "$ARTIFACT_DIR"
cd "$ROOT_DIR"

export RAGCLI_MUTATION="${RAGCLI_MUTATION:-1}"

if [[ $# -gt 1 ]]; then
  echo "Usage: $0 [git-ref]" >&2
  exit 2
fi

args=(
  unleash
  --output "$OUTPUT_PATH"
  --output-statuses "$OUTPUT_STATUSES"
)

if [[ $# -eq 1 && -n "$1" ]]; then
  args+=(
    --diff "$1"
  )
fi

if command -v gremlins >/dev/null 2>&1; then
  exec gremlins "${args[@]}"
fi

exec go run "github.com/go-gremlins/gremlins/cmd/gremlins@${GREMLINS_VERSION}" "${args[@]}"
