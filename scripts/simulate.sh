#!/usr/bin/env bash
# Run a benchmark for one concurrency profile.
#
# Usage:
#   bash scripts/simulate.sh c64              # uses configs/openclaw_c64.json
#   bash scripts/simulate.sh c128 my-server   # custom server label
set -euo pipefail

PROFILE="${1:?usage: simulate.sh <profile> [server-label]}"
SERVER_LABEL="${2:-lightllm}"
CONFIG="configs/openclaw_${PROFILE}.json"

if [ ! -f "$CONFIG" ]; then
  echo "error: config not found: $CONFIG"
  echo "hint:  run  bash scripts/gen_config.sh  first"
  exit 1
fi

openclaw-bench simulate \
  --config "$CONFIG" \
  --output "results/openclaw_${PROFILE}_$(date +%Y-%m-%d_%H-%M-%S).json" \
  --run-label "run-${PROFILE}" \
  --server-label "$SERVER_LABEL" \
  --base-url http://localhost:17888
