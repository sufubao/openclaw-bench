from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, model_validator


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class TokenizerSpec(StrictModel):
    kind: Literal["model", "tiktoken", "tokenizer_file", "regex"] = "model"
    model_name: str | None = None
    encoding_name: str | None = None
    tokenizer_path: str | None = None
    trust_remote_code: bool = True
    local_files_only: bool = False

    @model_validator(mode="after")
    def validate_tokenizer(self) -> "TokenizerSpec":
        if self.kind == "tokenizer_file" and not self.tokenizer_path:
            raise ValueError("tokenizer_file requires 'tokenizer_path'")
        return self


class NumericDistribution(StrictModel):
    kind: Literal["constant", "uniform", "triangular", "lognormal"]
    integer: bool = False
    value: float | None = None
    low: float | None = None
    high: float | None = None
    mode: float | None = None
    mean: float | None = None
    stdev: float | None = None
    clip_min: float | None = None
    clip_max: float | None = None

    @model_validator(mode="after")
    def validate_distribution(self) -> "NumericDistribution":
        if self.kind == "constant" and self.value is None:
            raise ValueError("constant distributions require 'value'")
        if self.kind == "uniform":
            if self.low is None or self.high is None:
                raise ValueError("uniform distributions require 'low' and 'high'")
        if self.kind == "triangular":
            if self.low is None or self.high is None or self.mode is None:
                raise ValueError("triangular distributions require 'low', 'mode', and 'high'")
        if self.kind == "lognormal":
            if self.mean is None or self.stdev is None:
                raise ValueError("lognormal distributions require 'mean' and 'stdev'")
        if self.low is not None and self.high is not None and self.low > self.high:
            raise ValueError("'low' must be <= 'high'")
        if self.clip_min is not None and self.clip_max is not None and self.clip_min > self.clip_max:
            raise ValueError("'clip_min' must be <= 'clip_max'")
        return self


class ArrivalConfig(StrictModel):
    kind: Literal["poisson", "uniform_window"] = "poisson"
    users_per_minute: float = Field(gt=0)
    arrival_window_seconds: float | None = Field(default=None, gt=0)

    @model_validator(mode="after")
    def validate_arrival(self) -> "ArrivalConfig":
        if self.kind == "uniform_window" and self.arrival_window_seconds is None:
            raise ValueError("uniform_window arrivals require 'arrival_window_seconds'")
        return self


class RequestTemplate(StrictModel):
    model: str
    temperature: float = 0.2
    top_p: float = 0.95
    max_retries: int = Field(default=0, ge=0)
    timeout_seconds: float = Field(default=300.0, gt=0)
    headers: dict[str, str] = Field(default_factory=dict)
    extra_body: dict[str, Any] = Field(default_factory=dict)


class WorkloadControls(StrictModel):
    total_users: int = Field(gt=0)
    arrival: ArrivalConfig
    think_time_seconds: NumericDistribution
    initial_user_tokens: NumericDistribution
    context_increment_tokens: NumericDistribution
    assistant_tokens: NumericDistribution
    max_context_tokens: int = Field(default=262_144, gt=0)
    context_threshold: float = Field(default=0.9, gt=0.0, lt=1.0)
    max_turns_per_user: int | None = Field(default=None, gt=0)


class GenerationInput(StrictModel):
    seed: int = 42
    tokenizer: TokenizerSpec = Field(default_factory=TokenizerSpec)
    request: RequestTemplate
    workload: WorkloadControls
    system_prompt: str | None = None
    system_prompt_path: str | None = None

    @model_validator(mode="after")
    def validate_prompt_source(self) -> "GenerationInput":
        if self.system_prompt and self.system_prompt_path:
            raise ValueError("provide either 'system_prompt' or 'system_prompt_path', not both")
        return self


class TurnPlan(StrictModel):
    turn_index: int = Field(ge=0)
    turn_type: Literal["initial", "clarification", "follow_up", "constraint_change", "evidence_dump"]
    user_message: str
    user_tokens: int = Field(gt=0)
    assistant_placeholder: str
    assistant_placeholder_tokens: int = Field(gt=0)
    max_output_tokens: int = Field(gt=0)
    think_time_after_seconds: float = Field(ge=0)
    estimated_prompt_tokens: int = Field(gt=0)


class SessionPlan(StrictModel):
    session_id: str
    scenario: str
    arrival_offset_seconds: float = Field(ge=0)
    estimated_terminal_context_tokens: int = Field(ge=0)
    stop_reason: str
    turns: list[TurnPlan] = Field(default_factory=list)


class SimulationConfig(StrictModel):
    schema_version: str = "1.0"
    config_uuid: str = Field(default_factory=lambda: str(uuid4()))
    generated_at: str = Field(default_factory=utc_now_iso)
    seed: int
    tokenizer: TokenizerSpec
    request: RequestTemplate
    workload: WorkloadControls
    system_prompt: str
    system_prompt_tokens: int = Field(gt=0)
    users: list[SessionPlan]
    summary: dict[str, Any]


class TurnResult(StrictModel):
    request_id: str = Field(default_factory=lambda: str(uuid4()))
    session_id: str
    scenario: str
    turn_index: int = Field(ge=0)
    status: Literal["completed", "http_error", "request_error"]
    http_status: int | None = None
    error: str | None = None
    finish_reason: str | None = None
    estimated_prompt_tokens: int = Field(gt=0)
    actual_prompt_tokens: int | None = Field(default=None, ge=0)
    max_output_tokens: int = Field(gt=0)
    actual_completion_tokens: int | None = Field(default=None, ge=0)
    ttft_seconds: float | None = Field(default=None, ge=0)
    tpot_seconds: float | None = Field(default=None, ge=0)
    total_latency_seconds: float | None = Field(default=None, ge=0)
    started_at: str
    completed_at: str | None = None
    started_at_offset_seconds: float = Field(ge=0)
    completed_at_offset_seconds: float | None = Field(default=None, ge=0)
    output_text: str | None = None
    output_preview: str | None = None
    output_sha256: str | None = None
    usage: dict[str, Any] | None = None


class SessionResult(StrictModel):
    session_id: str
    scenario: str
    planned_turns: int = Field(ge=0)
    completed_turns: int = Field(ge=0)
    failed_turns: int = Field(ge=0)
    stop_reason: str


class BenchmarkResult(StrictModel):
    schema_version: str = "1.0"
    run_uuid: str = Field(default_factory=lambda: str(uuid4()))
    config_uuid: str
    run_label: str
    server_label: str
    base_url: str
    model: str
    started_at: str = Field(default_factory=utc_now_iso)
    completed_at: str | None = None
    duration_seconds: float | None = None
    summary: dict[str, Any] = Field(default_factory=dict)
    session_results: list[SessionResult] = Field(default_factory=list)
    request_results: list[TurnResult] = Field(default_factory=list)
