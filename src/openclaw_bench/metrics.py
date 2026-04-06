from __future__ import annotations

from statistics import fmean

from openclaw_bench.models import TurnResult


PERCENTILES = (5, 10, 50, 90, 99)


def percentile(values: list[float], percentile_rank: float) -> float:
    if not values:
        raise ValueError("cannot compute percentile of empty values")
    if len(values) == 1:
        return values[0]
    ordered = sorted(values)
    rank = (len(ordered) - 1) * (percentile_rank / 100.0)
    lower = int(rank)
    upper = min(lower + 1, len(ordered) - 1)
    weight = rank - lower
    return ordered[lower] + ((ordered[upper] - ordered[lower]) * weight)


def describe(values: list[float]) -> dict[str, float | int]:
    if not values:
        return {"count": 0}
    ordered = sorted(values)
    summary: dict[str, float | int] = {
        "count": len(ordered),
        "mean": fmean(ordered),
        "min": ordered[0],
        "max": ordered[-1],
    }
    for pct in PERCENTILES:
        summary[f"p{pct}"] = percentile(ordered, pct)
    return summary


def peak_concurrency(request_results: list[TurnResult]) -> int:
    events: list[tuple[float, int]] = []
    for result in request_results:
        if result.completed_at_offset_seconds is None:
            continue
        events.append((result.started_at_offset_seconds, 1))
        events.append((result.completed_at_offset_seconds, -1))
    events.sort(key=lambda item: (item[0], item[1]))
    active = 0
    peak = 0
    for _, delta in events:
        active += delta
        peak = max(peak, active)
    return peak
