#!/usr/bin/env bash
# Generate simulation configs for all concurrency profiles.
set -euo pipefail

PROFILES="c32 c64 c128 c256"

for profile in $PROFILES; do
  echo "==> generating openclaw_${profile} ..."
  openclaw-bench generate-config \
    --control-file "control/openclaw_${profile}.json" \
    --output "configs/openclaw_${profile}.json"
done

echo "done."
