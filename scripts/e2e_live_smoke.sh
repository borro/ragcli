#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ARTIFACT_DIR="${RAGCLI_E2E_ARTIFACT_DIR:-${TMPDIR:-/tmp}/ragcli-e2e-live}"
LOG_DIR="$ARTIFACT_DIR/logs"
BIN_PATH="$ARTIFACT_DIR/ragcli"
FIXTURE_PATH="$ROOT_DIR/testdata/e2e/live_smoke_fixture.txt"
PROMPT="${RAGCLI_E2E_PROMPT:-What is the smoke test project code?}"
EXPECTED_TOKEN="${RAGCLI_E2E_EXPECTED_TOKEN:-RAGCLI-SMOKE-742}"
STRICT_MODE="${RAGCLI_E2E_REQUIRED:-0}"

mkdir -p "$ARTIFACT_DIR" "$LOG_DIR"

skip_or_fail() {
  local message="$1"
  if [[ "$STRICT_MODE" == "1" ]]; then
    echo "Live E2E smoke failed: $message" >&2
    exit 1
  fi

  echo "Live E2E smoke skipped: $message" >&2
  exit 0
}

COMMON_MODEL="${RAGCLI_E2E_CHAT_MODEL:-${LLM_MODEL:-}}"
EMBEDDING_MODEL="${RAGCLI_E2E_EMBEDDING_MODEL:-${EMBEDDING_MODEL:-}}"

MAP_MODEL="${RAGCLI_E2E_MAP_MODEL:-$COMMON_MODEL}"
TOOLS_MODEL="${RAGCLI_E2E_TOOLS_MODEL:-$COMMON_MODEL}"
TOOLS_RAG_MODEL="${RAGCLI_E2E_TOOLS_RAG_MODEL:-$TOOLS_MODEL}"
RAG_MODEL="${RAGCLI_E2E_RAG_MODEL:-$COMMON_MODEL}"
HYBRID_MODEL="${RAGCLI_E2E_HYBRID_MODEL:-$COMMON_MODEL}"

missing=()
for name in LLM_API_URL OPENAI_API_KEY; do
  if [[ -z "${!name:-}" ]]; then
    missing+=("$name")
  fi
done
if [[ -z "$COMMON_MODEL" ]]; then
  missing+=("LLM_MODEL or RAGCLI_E2E_CHAT_MODEL")
fi
if [[ -z "$EMBEDDING_MODEL" ]]; then
  missing+=("EMBEDDING_MODEL or RAGCLI_E2E_EMBEDDING_MODEL")
fi
if [[ ${#missing[@]} -gt 0 ]]; then
  skip_or_fail "missing required environment: ${missing[*]}"
fi

if [[ ! -f "$FIXTURE_PATH" ]]; then
  echo "Live E2E smoke failed: fixture not found at $FIXTURE_PATH" >&2
  exit 1
fi

echo "Building ragcli for live smoke..."
go build -o "$BIN_PATH" ./cmd/ragcli

COMMON_FLAGS=(
  --api-url "$LLM_API_URL"
  --api-key "$OPENAI_API_KEY"
  --lang en
  --raw
  --debug
)

RUN_FAILURES=0

run_case() {
  local name="$1"
  local input_mode="$2"
  local require_sources="$3"
  local stdout_path="$LOG_DIR/${name}.stdout.log"
  local stderr_path="$LOG_DIR/${name}.stderr.log"
  shift 3

  local args=("$@")

  echo "=== $name ==="
  set +e
  if [[ "$input_mode" == "stdin" ]]; then
    "$BIN_PATH" "${args[@]}" <"$FIXTURE_PATH" >"$stdout_path" 2>"$stderr_path"
  else
    "$BIN_PATH" "${args[@]}" >"$stdout_path" 2>"$stderr_path"
  fi
  local status=$?
  set -e

  if [[ $status -ne 0 ]]; then
    echo "FAIL $name: exit code $status" >&2
    RUN_FAILURES=$((RUN_FAILURES + 1))
    return
  fi

  if ! grep -Fq "$EXPECTED_TOKEN" "$stdout_path"; then
    echo "FAIL $name: stdout missing expected token $EXPECTED_TOKEN" >&2
    RUN_FAILURES=$((RUN_FAILURES + 1))
    return
  fi

  if grep -Fq "level=ERROR" "$stderr_path"; then
    echo "FAIL $name: stderr contains level=ERROR" >&2
    RUN_FAILURES=$((RUN_FAILURES + 1))
    return
  fi

  if [[ "$require_sources" == "1" ]] && ! grep -Fq "Sources:" "$stdout_path"; then
    echo "FAIL $name: stdout missing Sources section" >&2
    RUN_FAILURES=$((RUN_FAILURES + 1))
    return
  fi

  echo "PASS $name"
}

run_case "map-path" "path" "0" \
  "${COMMON_FLAGS[@]}" map --model "$MAP_MODEL" --path "$FIXTURE_PATH" "$PROMPT"
run_case "map-stdin" "stdin" "0" \
  "${COMMON_FLAGS[@]}" map --model "$MAP_MODEL" "$PROMPT"

run_case "tools-path" "path" "0" \
  "${COMMON_FLAGS[@]}" tools --model "$TOOLS_MODEL" --path "$FIXTURE_PATH" "$PROMPT"
run_case "tools-stdin" "stdin" "0" \
  "${COMMON_FLAGS[@]}" tools --model "$TOOLS_MODEL" "$PROMPT"

run_case "tools-rag-path" "path" "0" \
  "${COMMON_FLAGS[@]}" tools --rag --model "$TOOLS_RAG_MODEL" --embedding-model "$EMBEDDING_MODEL" --path "$FIXTURE_PATH" "$PROMPT"
run_case "tools-rag-stdin" "stdin" "0" \
  "${COMMON_FLAGS[@]}" tools --rag --model "$TOOLS_RAG_MODEL" --embedding-model "$EMBEDDING_MODEL" "$PROMPT"

run_case "rag-path" "path" "1" \
  "${COMMON_FLAGS[@]}" rag --model "$RAG_MODEL" --embedding-model "$EMBEDDING_MODEL" --path "$FIXTURE_PATH" "$PROMPT"
run_case "rag-stdin" "stdin" "1" \
  "${COMMON_FLAGS[@]}" rag --model "$RAG_MODEL" --embedding-model "$EMBEDDING_MODEL" "$PROMPT"

run_case "hybrid-path" "path" "1" \
  "${COMMON_FLAGS[@]}" hybrid --model "$HYBRID_MODEL" --embedding-model "$EMBEDDING_MODEL" --path "$FIXTURE_PATH" "$PROMPT"
run_case "hybrid-stdin" "stdin" "1" \
  "${COMMON_FLAGS[@]}" hybrid --model "$HYBRID_MODEL" --embedding-model "$EMBEDDING_MODEL" "$PROMPT"

if [[ $RUN_FAILURES -ne 0 ]]; then
  echo "Live E2E smoke failed. Logs: $LOG_DIR" >&2
  exit 1
fi

echo "Live E2E smoke passed. Logs: $LOG_DIR"
