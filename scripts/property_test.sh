#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RAPID_CHECKS="${RAGCLI_PROPERTY_RAPID_CHECKS:-500}"
GO_TEST_COUNT="${RAGCLI_PROPERTY_GO_TEST_COUNT:-1}"

PACKAGES=(
  ./internal/map
  ./internal/retrieval
  ./internal/aitools/files
)

cd "$ROOT_DIR"

go test \
  -count="$GO_TEST_COUNT" \
  -run Property \
  "${PACKAGES[@]}" \
  -args -rapid.checks="$RAPID_CHECKS"
