from __future__ import annotations

import asyncio
import hashlib
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx
from tqdm.auto import tqdm

from openclaw_bench.metrics import busy_seconds, describe, describe_trimmed, peak_concurrency
from openclaw_bench.models import BenchmarkResult, SessionPlan, SessionResult, SimulationConfig, TurnResult
from openclaw_bench.tokenizer import build_tokenizer


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_config(path: str | Path) -> SimulationConfig:
    raw = Path(path).read_text(encoding="utf-8")
    return SimulationConfig.model_validate_json(raw)


def normalize_base_url(base_url: str) -> str:
    stripped = base_url.rstrip("/")
    if stripped.endswith("/v1/chat/completions"):
        return stripped[: -len("/v1/chat/completions")]
    return stripped


class SimulationRunner:
    def __init__(
        self,
        config: SimulationConfig,
        base_url: str,
        api_key: str | None,
        run_label: str,
        server_label: str,
        extra_headers: dict[str, str] | None = None,
    ):
        self.config = config
        self.base_url = normalize_base_url(base_url)
        self.api_key = api_key
        self.run_label = run_label
        self.server_label = server_label
        self.extra_headers = extra_headers or {}
        self.endpoint = f"{self.base_url}/v1/chat/completions"
        self.tokenizer = build_tokenizer(config.tokenizer, fallback_model_name=config.request.model)
        self.run_started_wall = utc_now_iso()
        self.run_started_perf = time.perf_counter()
        self.request_results: list[TurnResult] = []
        self.session_results: list[SessionResult] = []
        self._request_lock = asyncio.Lock()
        self._progress_bar: Any | None = None

    async def run(self) -> BenchmarkResult:
        headers = {
            "Content-Type": "application/json",
            **self.config.request.headers,
            **self.extra_headers,
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        timeout = httpx.Timeout(timeout=self.config.request.timeout_seconds, connect=self.config.request.timeout_seconds)
        planned_requests = sum(len(session.turns) for session in self.config.users)
        async with httpx.AsyncClient(timeout=timeout, headers=headers) as client:
            await self._run_warmup(client)
            # Reset timing so warmup does not count toward metrics.
            self.run_started_wall = utc_now_iso()
            self.run_started_perf = time.perf_counter()
            with tqdm(total=planned_requests, desc="simulate", unit="req", disable=None) as progress:
                self._progress_bar = progress
                try:
                    tasks = [asyncio.create_task(self._run_session(client, session)) for session in self.config.users]
                    if tasks:
                        await asyncio.gather(*tasks)
                finally:
                    self._progress_bar = None

        completed_at = utc_now_iso()
        duration_seconds = max(time.perf_counter() - self.run_started_perf, 0.0)
        summary = self._build_summary(duration_seconds)
        return BenchmarkResult(
            config_uuid=self.config.config_uuid,
            run_label=self.run_label,
            server_label=self.server_label,
            base_url=self.base_url,
            model=self.config.request.model,
            started_at=self.run_started_wall,
            completed_at=completed_at,
            duration_seconds=duration_seconds,
            summary=summary,
            session_results=self.session_results,
            request_results=self.request_results,
        )

    async def _run_session(self, client: httpx.AsyncClient, session: SessionPlan) -> None:
        await self._sleep_until(session.arrival_offset_seconds)
        history = [{"role": "system", "content": self.config.system_prompt}]
        completed_turns = 0
        failed_turns = 0

        for turn in session.turns:
            messages = history + [{"role": "user", "content": turn.user_message}]
            result = await self._execute_turn(client, session, turn.turn_index, turn.estimated_prompt_tokens, turn.max_output_tokens, messages)
            async with self._request_lock:
                self.request_results.append(result)
                self._advance_progress_unlocked(1)
            if result.status != "completed":
                failed_turns += 1
                break
            completed_turns += 1
            assistant_content = result.output_text if result.output_text else turn.assistant_placeholder
            history.extend(
                [
                    {"role": "user", "content": turn.user_message},
                    {"role": "assistant", "content": assistant_content},
                ]
            )
            if turn.think_time_after_seconds > 0 and turn.turn_index < len(session.turns) - 1:
                await asyncio.sleep(turn.think_time_after_seconds)

        self.session_results.append(
            SessionResult(
                session_id=session.session_id,
                scenario=session.scenario,
                planned_turns=len(session.turns),
                completed_turns=completed_turns,
                failed_turns=failed_turns,
                stop_reason=session.stop_reason,
            )
        )

    def _advance_progress_unlocked(self, amount: int) -> None:
        if amount <= 0 or self._progress_bar is None:
            return
        self._progress_bar.update(amount)

    async def _run_warmup(self, client: httpx.AsyncClient) -> None:
        warmup = self.config.warmup
        if warmup.num_requests <= 0:
            return
        print(f"[warmup] sending {warmup.num_requests} requests (concurrency={warmup.max_concurrency}) ...")
        semaphore = asyncio.Semaphore(warmup.max_concurrency)
        completed = 0
        failed = 0

        async def _single(index: int) -> None:
            nonlocal completed, failed
            payload: dict[str, Any] = {
                "model": self.config.request.model,
                "messages": [{"role": "user", "content": warmup.prompt}],
                "max_tokens": warmup.max_tokens,
                "stream": False,
                **self.config.request.extra_body,
            }
            async with semaphore:
                try:
                    resp = await client.post(self.endpoint, json=payload)
                    if resp.status_code < 400:
                        completed += 1
                    else:
                        failed += 1
                except Exception:
                    failed += 1

        tasks = [asyncio.create_task(_single(i)) for i in range(warmup.num_requests)]
        await asyncio.gather(*tasks)
        print(f"[warmup] done — {completed} ok, {failed} failed")

    async def _execute_turn(
        self,
        client: httpx.AsyncClient,
        session: SessionPlan,
        turn_index: int,
        estimated_prompt_tokens: int,
        max_output_tokens: int,
        messages: list[dict[str, str]],
    ) -> TurnResult:
        payload: dict[str, Any] = {
            "model": self.config.request.model,
            "messages": messages,
            "stream": True,
            "temperature": self.config.request.temperature,
            "top_p": self.config.request.top_p,
            "max_tokens": max_output_tokens,
            **self.config.request.extra_body,
        }
        started_at_wall = utc_now_iso()
        started_at_perf = time.perf_counter()
        attempts = 0

        while True:
            try:
                return await self._stream_request(
                    client=client,
                    payload=payload,
                    session=session,
                    turn_index=turn_index,
                    estimated_prompt_tokens=estimated_prompt_tokens,
                    max_output_tokens=max_output_tokens,
                    started_at_wall=started_at_wall,
                    started_at_perf=started_at_perf,
                )
            except (httpx.HTTPError, asyncio.TimeoutError) as exc:
                if attempts >= self.config.request.max_retries:
                    completed_at_perf = time.perf_counter()
                    return TurnResult(
                        session_id=session.session_id,
                        scenario=session.scenario,
                        turn_index=turn_index,
                        status="request_error",
                        error=str(exc),
                        estimated_prompt_tokens=estimated_prompt_tokens,
                        max_output_tokens=max_output_tokens,
                        started_at=started_at_wall,
                        completed_at=utc_now_iso(),
                        started_at_offset_seconds=max(started_at_perf - self.run_started_perf, 0.0),
                        completed_at_offset_seconds=max(completed_at_perf - self.run_started_perf, 0.0),
                    )
                attempts += 1

    async def _stream_request(
        self,
        client: httpx.AsyncClient,
        payload: dict[str, Any],
        session: SessionPlan,
        turn_index: int,
        estimated_prompt_tokens: int,
        max_output_tokens: int,
        started_at_wall: str,
        started_at_perf: float,
    ) -> TurnResult:
        first_token_at: float | None = None
        finish_reason: str | None = None
        content_parts: list[str] = []
        usage: dict[str, Any] | None = None

        async with client.stream("POST", self.endpoint, json=payload) as response:
            if response.status_code >= 400:
                body = await response.aread()
                completed_at_perf = time.perf_counter()
                return TurnResult(
                    session_id=session.session_id,
                    scenario=session.scenario,
                    turn_index=turn_index,
                    status="http_error",
                    http_status=response.status_code,
                    error=body.decode("utf-8", errors="replace"),
                    estimated_prompt_tokens=estimated_prompt_tokens,
                    max_output_tokens=max_output_tokens,
                    started_at=started_at_wall,
                    completed_at=utc_now_iso(),
                    started_at_offset_seconds=max(started_at_perf - self.run_started_perf, 0.0),
                    completed_at_offset_seconds=max(completed_at_perf - self.run_started_perf, 0.0),
                )

            async for line in response.aiter_lines():
                if not line or not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if not data or data == "[DONE]":
                    continue
                payload_chunk = json.loads(data)
                usage = payload_chunk.get("usage") or usage
                for choice in payload_chunk.get("choices", []):
                    delta = choice.get("delta") or {}
                    content = delta.get("content")
                    if isinstance(content, str) and content:
                        if first_token_at is None:
                            first_token_at = time.perf_counter()
                        content_parts.append(content)
                    if choice.get("finish_reason") is not None:
                        finish_reason = choice["finish_reason"]

        completed_at_perf = time.perf_counter()
        completion_text = "".join(content_parts)
        actual_prompt_tokens = None
        actual_completion_tokens = None
        if usage:
            actual_prompt_tokens = usage.get("prompt_tokens")
            actual_completion_tokens = usage.get("completion_tokens")
        if actual_completion_tokens is None:
            actual_completion_tokens = self.tokenizer.count(completion_text)
        if actual_prompt_tokens is None:
            actual_prompt_tokens = estimated_prompt_tokens
        ttft_seconds = None
        tpot_seconds = None
        if first_token_at is not None:
            ttft_seconds = max(first_token_at - started_at_perf, 0.0)
            if actual_completion_tokens and actual_completion_tokens > 1:
                tpot_seconds = max((completed_at_perf - first_token_at) / (actual_completion_tokens - 1), 0.0)
        return TurnResult(
            session_id=session.session_id,
            scenario=session.scenario,
            turn_index=turn_index,
            status="completed",
            finish_reason=finish_reason,
            estimated_prompt_tokens=estimated_prompt_tokens,
            actual_prompt_tokens=actual_prompt_tokens,
            max_output_tokens=max_output_tokens,
            actual_completion_tokens=actual_completion_tokens,
            ttft_seconds=ttft_seconds,
            tpot_seconds=tpot_seconds,
            total_latency_seconds=max(completed_at_perf - started_at_perf, 0.0),
            started_at=started_at_wall,
            completed_at=utc_now_iso(),
            started_at_offset_seconds=max(started_at_perf - self.run_started_perf, 0.0),
            completed_at_offset_seconds=max(completed_at_perf - self.run_started_perf, 0.0),
            output_text=completion_text or None,
            output_preview=completion_text[:200] or None,
            output_sha256=hashlib.sha256(completion_text.encode("utf-8")).hexdigest() if completion_text else None,
            usage=usage,
        )

    async def _sleep_until(self, target_offset_seconds: float) -> None:
        while True:
            elapsed = time.perf_counter() - self.run_started_perf
            remaining = target_offset_seconds - elapsed
            if remaining <= 0:
                return
            await asyncio.sleep(min(remaining, 0.5))

    def _build_summary(self, duration_seconds: float) -> dict[str, Any]:
        completed = [result for result in self.request_results if result.status == "completed"]
        prompt_tokens = [float(result.actual_prompt_tokens or result.estimated_prompt_tokens) for result in completed]
        completion_tokens = [float(result.actual_completion_tokens or 0) for result in completed]
        ttft_values = [result.ttft_seconds for result in completed if result.ttft_seconds is not None]
        tpot_values = [result.tpot_seconds for result in completed if result.tpot_seconds is not None]
        latency_values = [result.total_latency_seconds for result in completed if result.total_latency_seconds is not None]
        duration = max(duration_seconds, 1e-9)

        # Server-busy time: wall-clock seconds where >= 1 request was in-flight.
        busy = max(busy_seconds(self.request_results), 1e-9)

        trim_frac = self.config.trim_percent / 100.0

        # Build the trimmed request set (middle portion by total latency).
        trimmed_results = self._trim_requests_by_latency(completed, trim_frac)
        tr_prompt = [float(r.actual_prompt_tokens or r.estimated_prompt_tokens) for r in trimmed_results]
        tr_completion = [float(r.actual_completion_tokens or 0) for r in trimmed_results]
        tr_ttft = [r.ttft_seconds for r in trimmed_results if r.ttft_seconds is not None]
        tr_tpot = [r.tpot_seconds for r in trimmed_results if r.tpot_seconds is not None]
        tr_latency = [r.total_latency_seconds for r in trimmed_results if r.total_latency_seconds is not None]
        tr_busy = max(busy_seconds(trimmed_results), 1e-9)

        return {
            "planned_requests": sum(len(session.turns) for session in self.config.users),
            "completed_requests": len(completed),
            "failed_requests": len(self.request_results) - len(completed),
            "completed_sessions": sum(1 for session in self.session_results if session.failed_turns == 0),
            "duration_seconds": duration,
            "busy_seconds": busy,
            "request_throughput_rps": len(completed) / busy,
            "request_throughput_rpm": (len(completed) / busy) * 60.0,
            "prompt_token_throughput_tps": sum(prompt_tokens) / busy,
            "completion_token_throughput_tps": sum(completion_tokens) / busy,
            "total_token_throughput_tps": (sum(prompt_tokens) + sum(completion_tokens)) / busy,
            "peak_inflight_requests": peak_concurrency(self.request_results),
            "ttft_seconds": describe(ttft_values),
            "tpot_seconds": describe(tpot_values),
            "total_latency_seconds": describe(latency_values),
            "prompt_tokens": describe(prompt_tokens),
            "completion_tokens": describe(completion_tokens),
            "trimmed": {
                "trim_percent": self.config.trim_percent,
                "included_requests": len(trimmed_results),
                "excluded_requests": len(completed) - len(trimmed_results),
                "busy_seconds": tr_busy,
                "request_throughput_rps": len(trimmed_results) / tr_busy,
                "request_throughput_rpm": (len(trimmed_results) / tr_busy) * 60.0,
                "prompt_token_throughput_tps": sum(tr_prompt) / tr_busy,
                "completion_token_throughput_tps": sum(tr_completion) / tr_busy,
                "total_token_throughput_tps": (sum(tr_prompt) + sum(tr_completion)) / tr_busy,
                "ttft_seconds": describe(tr_ttft),
                "tpot_seconds": describe(tr_tpot),
                "total_latency_seconds": describe(tr_latency),
                "prompt_tokens": describe(tr_prompt),
                "completion_tokens": describe(tr_completion),
            },
        }

    @staticmethod
    def _trim_requests_by_latency(completed: list[TurnResult], trim_fraction: float) -> list[TurnResult]:
        """Return the middle portion of *completed* requests sorted by total latency."""
        if not completed or trim_fraction <= 0:
            return completed
        by_latency = sorted(completed, key=lambda r: r.total_latency_seconds or 0.0)
        n = len(by_latency)
        lower = int(n * trim_fraction)
        upper = n - int(n * trim_fraction)
        if lower >= upper:
            return completed
        return by_latency[lower:upper]


async def simulate_config(
    config: SimulationConfig,
    base_url: str,
    api_key: str | None,
    run_label: str,
    server_label: str,
    extra_headers: dict[str, str] | None = None,
) -> BenchmarkResult:
    runner = SimulationRunner(
        config=config,
        base_url=base_url,
        api_key=api_key,
        run_label=run_label,
        server_label=server_label,
        extra_headers=extra_headers,
    )
    return await runner.run()


def resolve_api_key(explicit_key: str | None = None, env_var: str = "OPENAI_API_KEY") -> str | None:
    if explicit_key is not None:
        return explicit_key or None

    env_names = [env_var]
    if env_var != "OPENROUTER_API_KEY":
        env_names.append("OPENROUTER_API_KEY")

    for name in env_names:
        if not name:
            continue
        api_key = os.getenv(name)
        if api_key:
            return api_key
    return None


def write_result(result: BenchmarkResult, output_path: str | Path, full: bool = False) -> Path:
    output = Path(output_path)
    exclude = None if full else {"request_results": True, "session_results": True}
    output.write_text(result.model_dump_json(indent=2, exclude=exclude), encoding="utf-8")
    return output
