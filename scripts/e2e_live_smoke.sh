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

BASE_FLAGS=(
  --api-url "$LLM_API_URL"
  --api-key "$OPENAI_API_KEY"
  --lang en
  --raw
)

DEBUG_FLAGS=(--debug)
VERBOSE_FLAGS=(--verbose)

RUN_FAILURES=0

assert_monotonic_progress() {
  local stderr_path="$1"

  awk '
    {
      if (match($0, /\([0-9]+%\):/)) {
        value = substr($0, RSTART + 1, RLENGTH - 4) + 0
        seen = 1
        if (have_prev && value < prev) {
          printf "progress decreased from %d to %d: %s\n", prev, value, $0 > "/dev/stderr"
          exit 1
        }
        prev = value
        have_prev = 1
      }
    }
    END {
      if (!seen) {
        print "verbose log does not contain progress percentages" > "/dev/stderr"
        exit 1
      }
    }
  ' "$stderr_path"
}

run_case() {
  local name="$1"
  local input_mode="$2"
  local require_sources="$3"
  local stderr_profile="$4"
  local stdout_path="$LOG_DIR/${name}.stdout.log"
  local stderr_path="$LOG_DIR/${name}.stderr.log"
  shift 4

  local expected_stderr=()
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --stderr-marker)
        expected_stderr+=("$2")
        shift 2
        ;;
      --)
        shift
        break
        ;;
      *)
        break
        ;;
    esac
  done

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

  case "$stderr_profile" in
    verbose)
      for marker in "${expected_stderr[@]}"; do
        if ! grep -Fq "$marker" "$stderr_path"; then
          echo "FAIL $name: stderr missing verbose marker $marker" >&2
          RUN_FAILURES=$((RUN_FAILURES + 1))
          return
        fi
      done
      if ! assert_monotonic_progress "$stderr_path"; then
        echo "FAIL $name: stderr progress percent decreased or is missing" >&2
        RUN_FAILURES=$((RUN_FAILURES + 1))
        return
      fi
      ;;
    debug)
      ;;
    *)
      echo "FAIL $name: unknown stderr profile $stderr_profile" >&2
      RUN_FAILURES=$((RUN_FAILURES + 1))
      return
      ;;
  esac

  echo "PASS $name"
}

run_case "map-path" "path" "0" "verbose" \
  --stderr-marker "splitting file into chunks" \
  -- \
  "${BASE_FLAGS[@]}" "${VERBOSE_FLAGS[@]}" map --model "$MAP_MODEL" --path "$FIXTURE_PATH" "$PROMPT"
run_case "map-stdin" "stdin" "0" "debug" \
  "${BASE_FLAGS[@]}" "${DEBUG_FLAGS[@]}" map --model "$MAP_MODEL" "$PROMPT"

run_case "tools-path" "path" "0" "verbose" \
  --stderr-marker "semantic index is not needed: search_rag is disabled in this tool session" \
  -- \
  "${BASE_FLAGS[@]}" "${VERBOSE_FLAGS[@]}" tools --model "$TOOLS_MODEL" --path "$FIXTURE_PATH" "$PROMPT"
run_case "tools-stdin" "stdin" "0" "debug" \
  "${BASE_FLAGS[@]}" "${DEBUG_FLAGS[@]}" tools --model "$TOOLS_MODEL" "$PROMPT"

run_case "tools-rag-path" "path" "0" "verbose" \
  --stderr-marker "reading input document" \
  --stderr-marker "preparing tool session" \
  -- \
  "${BASE_FLAGS[@]}" "${VERBOSE_FLAGS[@]}" tools --rag --model "$TOOLS_RAG_MODEL" --embedding-model "$EMBEDDING_MODEL" --path "$FIXTURE_PATH" "$PROMPT"
run_case "tools-rag-stdin" "stdin" "0" "debug" \
  "${BASE_FLAGS[@]}" "${DEBUG_FLAGS[@]}" tools --rag --model "$TOOLS_RAG_MODEL" --embedding-model "$EMBEDDING_MODEL" "$PROMPT"

run_case "rag-path" "path" "1" "verbose" \
  --stderr-marker "sending question to embeddings API" \
  -- \
  "${BASE_FLAGS[@]}" "${VERBOSE_FLAGS[@]}" rag --model "$RAG_MODEL" --embedding-model "$EMBEDDING_MODEL" --path "$FIXTURE_PATH" "$PROMPT"
run_case "rag-stdin" "stdin" "1" "debug" \
  "${BASE_FLAGS[@]}" "${DEBUG_FLAGS[@]}" rag --model "$RAG_MODEL" --embedding-model "$EMBEDDING_MODEL" "$PROMPT"

run_case "hybrid-path" "path" "1" "verbose" \
  --stderr-marker "checking whether the file fits in the model context" \
  -- \
  "${BASE_FLAGS[@]}" "${VERBOSE_FLAGS[@]}" hybrid --model "$HYBRID_MODEL" --embedding-model "$EMBEDDING_MODEL" --path "$FIXTURE_PATH" "$PROMPT"
run_case "hybrid-stdin" "stdin" "1" "debug" \
  "${BASE_FLAGS[@]}" "${DEBUG_FLAGS[@]}" hybrid --model "$HYBRID_MODEL" --embedding-model "$EMBEDDING_MODEL" "$PROMPT"

if [[ $RUN_FAILURES -ne 0 ]]; then
  echo "Live E2E smoke failed. Logs: $LOG_DIR" >&2
  exit 1
fi

echo "Live E2E smoke passed. Logs: $LOG_DIR"
