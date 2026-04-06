from __future__ import annotations

from openclaw_bench.config_generator import generate_config
from openclaw_bench.models import (
    ArrivalConfig,
    GenerationInput,
    NumericDistribution,
    RequestTemplate,
    TokenizerSpec,
    WorkloadControls,
)
from openclaw_bench.tokenizer import build_tokenizer


def build_control() -> GenerationInput:
    return GenerationInput(
        seed=7,
        tokenizer=TokenizerSpec(kind="regex"),
        request=RequestTemplate(model="openai/gpt-4.1-mini"),
        workload=WorkloadControls(
            total_users=4,
            arrival=ArrivalConfig(kind="poisson", users_per_minute=30),
            think_time_seconds=NumericDistribution(kind="constant", value=5),
            initial_user_tokens=NumericDistribution(kind="constant", value=120, integer=True),
            context_increment_tokens=NumericDistribution(kind="constant", value=80, integer=True),
            assistant_tokens=NumericDistribution(kind="constant", value=64, integer=True),
            max_context_tokens=2048,
            context_threshold=0.8,
            max_turns_per_user=5,
        ),
    )


def test_generate_config_is_deterministic() -> None:
    control = build_control()
    first = generate_config(control)
    second = generate_config(control)
    assert first.model_dump(exclude={"config_uuid", "generated_at"}) == second.model_dump(
        exclude={"config_uuid", "generated_at"}
    )


def test_config_respects_context_threshold() -> None:
    control = build_control()
    control.workload.max_context_tokens = 512
    config = generate_config(control)
    threshold = int(control.workload.max_context_tokens * control.workload.context_threshold)
    for session in config.users:
        for turn in session.turns:
            assert turn.estimated_prompt_tokens < threshold


def test_config_token_estimates_match_chat_estimator() -> None:
    control = build_control()
    control.workload.total_users = 2
    control.workload.max_turns_per_user = 4
    config = generate_config(control)
    tokenizer = build_tokenizer(control.tokenizer, fallback_model_name=control.request.model)

    for session in config.users:
        history = [{"role": "system", "content": config.system_prompt}]
        for turn in session.turns:
            prompt_messages = history + [{"role": "user", "content": turn.user_message}]
            assert turn.estimated_prompt_tokens == tokenizer.estimate_chat_tokens(prompt_messages)
            history.extend(
                [
                    {"role": "user", "content": turn.user_message},
                    {"role": "assistant", "content": turn.assistant_placeholder},
                ]
            )
        assert session.estimated_terminal_context_tokens == tokenizer.estimate_chat_tokens(history)
