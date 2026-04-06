# OpenClaw Bench

`openclaw-bench` is a deterministic benchmark harness for OpenAI-compatible chat-completions services, designed around the usage pattern you described for OpenClaw:

- a fixed system prompt
- multi-turn user conversations
- random user arrivals instead of a fake flat concurrency constant
- token-based session growth and stopping rules
- streaming latency measurement for TTFT and TPOT

The simulator always calls `POST /v1/chat/completions`. It keeps later turns deterministic by rolling forward a pre-generated assistant placeholder instead of the live model output, which makes results from different servers comparable when they share the same config UUID.

## Design Notes

- `users_per_minute` is treated as an arrival rate, not literal concurrency. Real concurrency emerges from arrival rate, service latency, and user think time.
- `think_time_seconds` is a distribution, not a fixed delay. Each turn samples from the configured distribution during config generation and stores the exact sampled values in the config JSON.
- All workload sizing is token-based using a real tokenizer resolved from the configured model by default. Character counts are not used for stopping decisions.
- A session stops before sending the next request if the estimated prompt tokens would cross `context_threshold * max_context_tokens`, or if prompt plus the next planned assistant budget would exceed the hard context limit.

## Project Layout

- `generate-config`: consumes a control JSON and emits a deterministic simulation config with a unique config UUID.
- `simulate`: replays the config against an OpenAI-compatible endpoint and writes a result JSON with raw request metrics plus aggregate summaries.
- `serve-dashboard`: launches a local FastAPI dashboard that accepts result JSON uploads and compares runs grouped by config UUID.

## Quick Start

Install the package in editable mode:

```bash
python -m pip install -e .
```

Generate a deterministic workload:

```bash
openclaw-bench generate-config \
  --control-file examples/control.example.json \
  --output out/config.json
```

Run the benchmark against a local vLLM server. `simulate` now defaults to `http://localhost:8000`, so `--base-url` is optional:

```bash
openclaw-bench simulate \
  --config out/config.json \
  --output out/result-server-a.json \
  --run-label run-a \
  --server-label vllm-local
```

For OpenRouter or another hosted OpenAI-compatible gateway, override `--base-url` and provide an API key if the server requires one:

```bash
export OPENROUTER_API_KEY=your_key_here

openclaw-bench simulate \
  --config out/config.json \
  --output out/result-server-a.json \
  --base-url https://openrouter.ai/api \
  --run-label run-a \
  --server-label router-a \
  --http-referer https://your-benchmark-host.example \
  --title openclaw-bench
```

Launch the comparison UI:

```bash
openclaw-bench serve-dashboard --host 127.0.0.1 --port 8000
```

Then open `http://127.0.0.1:8000`, upload multiple result JSON files, and compare only the runs that share a config UUID.

## Control File Shape

Use [`examples/control.example.json`](/workspace/openclaw-bench/examples/control.example.json) as the starting point.

Important fields:

- `total_users`: total sessions to simulate.
- `arrival.users_per_minute`: session arrival rate.
- `think_time_seconds`: post-response user think-time distribution.
- `initial_user_tokens`: size of the first user turn.
- `context_increment_tokens`: size of later user turns.
- `assistant_tokens`: deterministic assistant placeholder size and `max_tokens` cap for the live request.
- `max_context_tokens`: hard model context window, for example `262144` for a 256k target.
- `context_threshold`: session stops before the next request once the prompt estimate reaches this fraction of the hard limit.

Supported numeric distributions:

- `constant`
- `uniform`
- `triangular`
- `lognormal`

Supported tokenizer modes:

- `model`: default. Resolve the tokenizer from `request.model`, using `tiktoken` for known model families and `transformers` for model-backed tokenizers.
- `tiktoken`: explicit `tiktoken` mode if you need to force a known encoding family.
- `tokenizer_file`: load a local `tokenizer.json` for a model-specific tokenizer without relying on a remote registry lookup.
- `regex`: offline fallback for smoke tests only, not recommended for production benchmarking.

## Output Metrics

Each result JSON contains:

- per-request raw timings and token counts
- per-session completion/failure counts
- aggregate throughput in requests/sec and output tokens/sec
- TTFT percentiles: `p5`, `p10`, `p50`, `p90`, `p99`
- TPOT percentiles: `p5`, `p10`, `p50`, `p90`, `p99`
- total latency percentiles
- peak in-flight request count

## Validation

Run the unit tests:

```bash
pytest
```

## API Notes

The simulator is written for OpenAI-compatible chat completions and is compatible with OpenRouter’s `/v1/chat/completions` path. OpenRouter’s current docs also indicate that usage is automatically included in responses, including streaming responses:

- https://openrouter.ai/docs/guides/guides/administration/usage-accounting
