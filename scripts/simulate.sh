#!/usr/bin/env bash
set -euo pipefail

usage() {
    echo "Usage: $0 --config <path> [--port <port>] [--server-tag <label>]"
    echo ""
    echo "  --config      Path to the simulation config JSON (required)"
    echo "  --port        Port of the local server (default: 8000)"
    echo "  --server-tag  Label for this server (default: vllm-local)"
    exit 1
}

CONFIG=""
PORT="8000"
SERVER_TAG="vllm-local"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --config)     CONFIG="$2";     shift 2 ;;
        --port)       PORT="$2";       shift 2 ;;
        --server-tag) SERVER_TAG="$2"; shift 2 ;;
        *) echo "Unknown argument: $1"; usage ;;
    esac
done

[[ -z "$CONFIG" ]] && { echo "Error: --config is required"; usage; }

BASENAME=$(basename "$CONFIG" .json)
TIMESTAMP=$(date +%Y-%m-%d_%H-%M-%S)
RUN_LABEL="${BASENAME}_${TIMESTAMP}"
OUTPUT="results/${RUN_LABEL}.json"

mkdir -p results

openclaw-bench simulate \
    --config "$CONFIG" \
    --base-url "http://localhost:${PORT}" \
    --server-label "$SERVER_TAG" \
    --run-label "$RUN_LABEL" \
    --output "$OUTPUT"
