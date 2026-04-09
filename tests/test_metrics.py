from __future__ import annotations

from openclaw_bench.metrics import busy_seconds, describe, describe_trimmed, percentile, trim_sorted
from openclaw_bench.models import TurnResult


def test_percentile_interpolates() -> None:
    values = [1.0, 2.0, 3.0, 4.0]
    assert percentile(values, 50) == 2.5


def test_describe_contains_expected_percentiles() -> None:
    summary = describe([1.0, 2.0, 3.0, 4.0, 5.0])
    assert summary["count"] == 5
    assert summary["p5"] >= 1.0
    assert summary["p99"] <= 5.0


def test_trim_sorted_drops_extremes() -> None:
    ordered = list(range(1, 11))  # [1..10]
    trimmed = trim_sorted(ordered, 0.1)
    assert trimmed == [2, 3, 4, 5, 6, 7, 8, 9]


def test_trim_sorted_noop_when_zero() -> None:
    ordered = [1.0, 2.0, 3.0]
    assert trim_sorted(ordered, 0.0) == ordered


def test_trim_sorted_preserves_when_too_few() -> None:
    ordered = [1.0, 2.0]
    assert trim_sorted(ordered, 0.5) == ordered


def test_describe_trimmed_excludes_extremes() -> None:
    values = list(range(1, 21))  # [1..20], 10 % = 2 from each side
    summary = describe_trimmed(values, 0.1)
    assert summary["count"] == 16
    assert summary["min"] == 3
    assert summary["max"] == 18


def _make_turn(start: float, end: float) -> TurnResult:
    return TurnResult(
        session_id="s1",
        scenario="test",
        turn_index=0,
        status="completed",
        estimated_prompt_tokens=1,
        max_output_tokens=1,
        started_at="t",
        started_at_offset_seconds=start,
        completed_at_offset_seconds=end,
    )


def test_busy_seconds_no_gap() -> None:
    # Two overlapping requests: [0,10] and [5,15] → busy = 15s
    results = [_make_turn(0, 10), _make_turn(5, 15)]
    assert busy_seconds(results) == 15.0


def test_busy_seconds_with_gap() -> None:
    # Two requests with a gap: [0,5] and [10,20] → busy = 15s (gap excluded)
    results = [_make_turn(0, 5), _make_turn(10, 20)]
    assert busy_seconds(results) == 15.0
