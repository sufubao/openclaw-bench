from __future__ import annotations

import math
import random
from pathlib import Path

from openclaw_bench.models import ArrivalConfig, GenerationInput, NumericDistribution
from openclaw_bench.tokenizer import TokenizerAdapter


DEFAULT_SYSTEM_PROMPT = """You are OpenClaw, a coding-focused AI assistant working inside a user workspace.

Be concise, technically accurate, and explicit about assumptions. Prefer actionable engineering guidance, preserve the user's existing code patterns, and explain tradeoffs when there is real ambiguity.

When the user asks for code changes, reason about the smallest safe change that solves the problem. Keep responses structured around the actual task, not generic advice. Pay attention to tests, regressions, edge cases, and performance when they matter."""


SESSION_SCENARIOS: list[dict[str, object]] = [
    {
        "name": "repo-debugging",
        "initial": [
            "I need help debugging a flaky regression in a request router. The failure only shows up under parallel load and the error rate increases after a deploy.",
            "The latest change broke a hot path in our service. Please analyze the failure mode and suggest a focused fix rather than a full rewrite.",
        ],
        "follow_up": [
            "Based on the earlier analysis, I pulled a smaller repro and I need you to reason through the edge cases before I patch it.",
            "I found another clue after trying your previous suggestion. Please refine the diagnosis and keep the patch surface limited.",
        ],
        "constraints": [
            "The codebase is large, the team wants backward compatibility, and we cannot add new infrastructure dependencies.",
            "Please keep the response grounded in production debugging rather than academic alternatives.",
        ],
    },
    {
        "name": "test-failure-triage",
        "initial": [
            "A CI job started failing after a refactor. I need help tracing the root cause from logs, config snippets, and a partial diff.",
            "Please review a failing integration test and tell me what behavior likely regressed, what to inspect next, and how to contain the fallout.",
        ],
        "follow_up": [
            "I reran the suite with more logging and now I need a narrower explanation of the failure path.",
            "The failure was intermittent, but I captured another trace. Please fold it into the previous reasoning and identify the most likely bug.",
        ],
        "constraints": [
            "Avoid broad refactors and assume the team will accept only low-risk changes this week.",
            "The answer should prioritize concrete failure modes, not a generic testing checklist.",
        ],
    },
    {
        "name": "large-context-refactor",
        "initial": [
            "I am planning a medium-sized refactor across several modules and I need a staged approach that preserves behavior while reducing coupling.",
            "Please read through the architecture notes, a few code fragments, and some rough constraints, then recommend a realistic incremental plan.",
        ],
        "follow_up": [
            "I incorporated part of the prior plan and now I need a deeper review of the boundary between modules.",
            "I have additional implementation notes. Please revise the refactor sequence while keeping the migration safe for existing callers.",
        ],
        "constraints": [
            "The team cares about migration safety, operational simplicity, and avoiding hidden performance regressions.",
            "Keep the answer practical: specific sequencing, risks, and rollback points matter more than idealized architecture.",
        ],
    },
]


FILLER_BLOCKS = [
    """Repository notes:
- request_router.py dispatches based on capability and tenant policy
- session_state.py tracks rolling transcript metadata and token budgets
- benchmark_worker.py fans out load using asyncio tasks and streams responses
- observability hooks emit latency histograms and per-request labels""",
    """Recent logs:
2026-03-01T10:11:24Z INFO router selected upstream=provider-a shard=2 queue_depth=4
2026-03-01T10:11:24Z WARN cache miss for session=session-4916 prompt_tokens=18342
2026-03-01T10:11:25Z ERROR upstream timeout after 2.04s request_id=req_91a tenant=enterprise
2026-03-01T10:11:25Z INFO retry budget exhausted for request_id=req_91a""",
    """Code fragment:
```python
async def dispatch(request, upstream):
    stream = await upstream.create_stream(request.payload)
    async for event in stream:
        if event.kind == "token":
            yield event.text
        elif event.kind == "usage":
            request.usage = event.payload
```
""",
    """Config excerpt:
```yaml
router:
  provider_order: [primary, overflow, backup]
  retry_budget: 1
  request_timeout_seconds: 45
  cache:
    enabled: true
    max_prompt_tokens: 200000
```""",
    """Shell transcript:
```bash
$ pytest tests/test_router.py -k timeout -vv
FAILED tests/test_router.py::test_stream_timeout_recovers - AssertionError
$ tail -n 20 logs/service.log
... upstream disconnected before terminal usage chunk ...
```""",
    """Diff excerpt:
```diff
- history.append({"role": "assistant", "content": reply})
+ history.extend(patch.generated_history_items())
- if budget > threshold:
+ if budget >= threshold:
```
""",
    """Structured payload:
```json
{
  "tenant": "acme-enterprise",
  "session": "sess_8120",
  "features": ["streaming", "long-context", "tool-routing"],
  "observed": {
    "ttft_ms": 1980,
    "tpot_ms": 42,
    "tail_latency_ms": 9110
  }
}
```""",
]


ASSISTANT_FILLER_BLOCKS = [
    """Suggested approach:
1. Confirm whether the transcript budget is computed before or after appending the next user turn.
2. Compare real upstream usage tokens with the local estimator on a few long sessions.
3. Isolate whether retries or delayed usage accounting are creating hidden prompt growth.
4. Add one targeted regression test for the exact boundary condition.""",
    """Patch shape:
```python
estimated_prompt = estimator(messages + [next_user_message])
if estimated_prompt >= stop_threshold:
    return SessionStop("context_threshold")

response = await client.create(...)
history.append(build_placeholder(response_budget))
```""",
    """Risk notes:
- If prompt estimation systematically undercounts, the benchmark will overshoot the intended ceiling.
- If assistant history is non-deterministic across runs, server-to-server comparisons stop being apples-to-apples.
- If TTFT is measured on role-only chunks, the number will look artificially optimistic.""",
]


TURN_TYPE_BY_INDEX = {
    0: "initial",
    1: "clarification",
    2: "follow_up",
}


def _count_static_text(text: str, tokenizer: TokenizerAdapter) -> int:
    cache = getattr(tokenizer, "_openclaw_static_count_cache", None)
    if cache is None:
        cache = {}
        setattr(tokenizer, "_openclaw_static_count_cache", cache)
    if text not in cache:
        cache[text] = tokenizer.count(text)
    return cache[text]


def load_system_prompt(control: GenerationInput) -> str:
    if control.system_prompt:
        return control.system_prompt.strip()
    if control.system_prompt_path:
        return Path(control.system_prompt_path).read_text(encoding="utf-8").strip()
    return DEFAULT_SYSTEM_PROMPT.strip()


def sample_distribution(distribution: NumericDistribution, rng: random.Random) -> float:
    if distribution.kind == "constant":
        value = float(distribution.value)
    elif distribution.kind == "uniform":
        value = rng.uniform(float(distribution.low), float(distribution.high))
    elif distribution.kind == "triangular":
        value = rng.triangular(float(distribution.low), float(distribution.high), float(distribution.mode))
    else:
        mean = float(distribution.mean)
        stdev = float(distribution.stdev)
        variance_ratio = (stdev**2) / (mean**2)
        sigma_sq = math.log1p(variance_ratio)
        sigma = math.sqrt(sigma_sq)
        mu = math.log(mean) - (sigma_sq / 2)
        value = rng.lognormvariate(mu, sigma)

    if distribution.clip_min is not None:
        value = max(value, distribution.clip_min)
    if distribution.clip_max is not None:
        value = min(value, distribution.clip_max)
    if distribution.integer:
        return int(round(value))
    return value


def sample_arrival_offsets(total_users: int, arrival: ArrivalConfig, rng: random.Random) -> list[float]:
    if total_users <= 0:
        return []
    if arrival.kind == "poisson":
        offsets = [0.0]
        rate = arrival.users_per_minute / 60.0
        for _ in range(total_users - 1):
            offsets.append(offsets[-1] + rng.expovariate(rate))
        return offsets
    window = float(arrival.arrival_window_seconds)
    offsets = sorted(rng.uniform(0.0, window) for _ in range(total_users))
    if not offsets:
        return offsets
    minimum = offsets[0]
    return [offset - minimum for offset in offsets]


def choose_scenario(rng: random.Random) -> dict[str, object]:
    return rng.choice(SESSION_SCENARIOS)


def _trimmed_join(parts: list[str], target_tokens: int, tokenizer: TokenizerAdapter) -> str:
    text = "\n\n".join(part.strip() for part in parts if part.strip())
    return tokenizer.trim_to_tokens(text, target_tokens).strip()


def _expand_to_budget(
    base_parts: list[str],
    target_tokens: int,
    filler_pool: list[str],
    tokenizer: TokenizerAdapter,
    rng: random.Random,
) -> str:
    if target_tokens <= 0:
        return ""
    parts = list(base_parts)
    shuffled_fillers = list(filler_pool)
    rng.shuffle(shuffled_fillers)

    base_text = "\n\n".join(part.strip() for part in parts if part.strip())
    base_count = tokenizer.count(base_text)

    if base_count >= target_tokens:
        return tokenizer.trim_to_tokens(base_text, target_tokens).strip()

    remaining = target_tokens - base_count
    if shuffled_fillers:
        avg_filler_tokens = sum(_count_static_text(f, tokenizer) for f in shuffled_fillers) / len(shuffled_fillers)
        if avg_filler_tokens > 0:
            needed = int(remaining / avg_filler_tokens) + 2
        else:
            needed = 1
    else:
        needed = 0

    for i in range(needed):
        parts.append(shuffled_fillers[i % len(shuffled_fillers)])

    text = "\n\n".join(part.strip() for part in parts if part.strip())
    return tokenizer.trim_to_tokens(text, target_tokens).strip()


def build_user_turn(
    scenario: dict[str, object],
    turn_index: int,
    target_tokens: int,
    tokenizer: TokenizerAdapter,
    rng: random.Random,
) -> tuple[str, str]:
    name = str(scenario["name"])
    if turn_index == 0:
        stem = rng.choice(list(scenario["initial"]))
        turn_type = "initial"
    else:
        stem = rng.choice(list(scenario["follow_up"]))
        turn_type = TURN_TYPE_BY_INDEX.get(turn_index, rng.choice(["follow_up", "constraint_change", "evidence_dump"]))
    constraint = rng.choice(list(scenario["constraints"]))
    preamble = [
        f"Scenario: {name.replace('-', ' ')}.",
        stem,
        constraint,
        "Use the material below as the current conversation state and focus on the next practical step.",
    ]
    message = _expand_to_budget(preamble, target_tokens, FILLER_BLOCKS, tokenizer, rng)
    return turn_type, message


def build_assistant_placeholder(
    scenario: dict[str, object],
    turn_index: int,
    target_tokens: int,
    tokenizer: TokenizerAdapter,
    rng: random.Random,
) -> str:
    base = [
        f"Working notes for {scenario['name']}:",
        "The previous answer analyzed the likely root cause, narrowed the risk surface, and proposed a minimal next action.",
        "Key conclusions are preserved here only to grow the rolling transcript deterministically across benchmark runs.",
    ]
    if turn_index > 0:
        base.append("The follow-up also included edge cases, rollback guidance, and one concrete test to add before shipping.")
    return _expand_to_budget(base, target_tokens, ASSISTANT_FILLER_BLOCKS, tokenizer, rng)
