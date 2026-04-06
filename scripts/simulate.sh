openclaw-bench simulate \
  --config configs/openclaw.json \
  --output results/openclaw_$(date +%Y-%m-%d_%H-%M-%S).json \
  --run-label run-a \
  --server-label lightllm \
  --base-url http://localhost:17888