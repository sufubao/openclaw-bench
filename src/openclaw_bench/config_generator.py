from __future__ import annotations

import json
import random
from pathlib import Path

from openclaw_bench.models import GenerationInput, SessionPlan, SimulationConfig, TurnPlan
from openclaw_bench.scenario import (
    build_assistant_placeholder,
    build_user_turn,
    choose_scenario,
    load_system_prompt,
    sample_arrival_offsets,
    sample_distribution,
)
from openclaw_bench.tokenizer import build_tokenizer


def _estimate_chat_message_tokens(content_tokens: int) -> int:
    return 3 + content_tokens


def load_generation_input(path: str | Path) -> GenerationInput:
    raw = Path(path).read_text(encoding="utf-8")
    return GenerationInput.model_validate(json.loads(raw))


def generate_config(control: GenerationInput) -> SimulationConfig:
    rng = random.Random(control.seed)
    print(f"[generate-config] seed={control.seed}, loading tokenizer (kind={control.tokenizer.kind}) ...")
    tokenizer = build_tokenizer(control.tokenizer, fallback_model_name=control.request.model)
    print("[generate-config] tokenizer loaded.")
    system_prompt = load_system_prompt(control)
    system_prompt_tokens = tokenizer.count(system_prompt)
    print(f"[generate-config] system prompt: {system_prompt_tokens} tokens")
    arrival_offsets = sample_arrival_offsets(control.workload.total_users, control.workload.arrival, rng)
    threshold_tokens = int(control.workload.max_context_tokens * control.workload.context_threshold)
    print(f"[generate-config] total_users={control.workload.total_users}, context_threshold_tokens={threshold_tokens}")

    users: list[SessionPlan] = []
    total_planned_requests = 0
    total_estimated_prompt_tokens = 0
    empty_sessions = 0

    for user_index, arrival_offset in enumerate(arrival_offsets):
        scenario = choose_scenario(rng)
        history_token_total = 3 + _estimate_chat_message_tokens(system_prompt_tokens)
        turns: list[TurnPlan] = []
        stop_reason = "context_threshold"

        while True:
            turn_index = len(turns)
            if control.workload.max_turns_per_user and turn_index >= control.workload.max_turns_per_user:
                stop_reason = "max_turns_per_user"
                break

            distribution = (
                control.workload.initial_user_tokens
                if turn_index == 0
                else control.workload.context_increment_tokens
            )
            user_tokens = max(1, int(sample_distribution(distribution, rng)))
            assistant_tokens = max(1, int(sample_distribution(control.workload.assistant_tokens, rng)))
            think_time = max(0.0, float(sample_distribution(control.workload.think_time_seconds, rng)))

            turn_type, user_message = build_user_turn(scenario, turn_index, user_tokens, tokenizer, rng)
            user_message_tokens = tokenizer.count(user_message)
            estimated_prompt_tokens = history_token_total + _estimate_chat_message_tokens(user_message_tokens)

            if estimated_prompt_tokens >= threshold_tokens:
                stop_reason = "context_threshold"
                break
            if estimated_prompt_tokens + assistant_tokens > control.workload.max_context_tokens:
                stop_reason = "max_context_guard"
                break

            assistant_placeholder = build_assistant_placeholder(
                scenario,
                turn_index,
                assistant_tokens,
                tokenizer,
                rng,
            )
            turn = TurnPlan(
                turn_index=turn_index,
                turn_type=turn_type,
                user_message=user_message,
                user_tokens=user_message_tokens,
                assistant_placeholder=assistant_placeholder,
                assistant_placeholder_tokens=tokenizer.count(assistant_placeholder),
                max_output_tokens=assistant_tokens,
                think_time_after_seconds=think_time,
                estimated_prompt_tokens=estimated_prompt_tokens,
            )
            turns.append(turn)
            history_token_total += _estimate_chat_message_tokens(turn.user_tokens)
            history_token_total += _estimate_chat_message_tokens(turn.assistant_placeholder_tokens)

        if not turns:
            empty_sessions += 1

        session = SessionPlan(
            session_id=f"user-{user_index + 1:05d}",
            scenario=str(scenario["name"]),
            arrival_offset_seconds=arrival_offset,
            estimated_terminal_context_tokens=history_token_total,
            stop_reason=stop_reason,
            turns=turns,
        )
        total_planned_requests += len(turns)
        total_estimated_prompt_tokens += sum(turn.estimated_prompt_tokens for turn in turns)
        users.append(session)
        if (user_index + 1) % 10 == 0 or user_index == len(arrival_offsets) - 1:
            print(f"[generate-config] planned {user_index + 1}/{control.workload.total_users} users, {total_planned_requests} requests so far")

    summary = {
        "total_users": control.workload.total_users,
        "planned_requests": total_planned_requests,
        "users_without_requests": empty_sessions,
        "estimated_prompt_tokens": total_estimated_prompt_tokens,
        "arrival_span_seconds": arrival_offsets[-1] if arrival_offsets else 0.0,
        "context_threshold_tokens": threshold_tokens,
    }
    return SimulationConfig(
        seed=control.seed,
        tokenizer=control.tokenizer,
        request=control.request,
        workload=control.workload,
        system_prompt=system_prompt,
        system_prompt_tokens=system_prompt_tokens,
        users=users,
        summary=summary,
    )


def write_config(config: SimulationConfig, output_path: str | Path) -> Path:
    output = Path(output_path)
    output.write_text(config.model_dump_json(indent=2), encoding="utf-8")
    return output
