from __future__ import annotations

from openclaw_bench.metrics import describe, describe_trimmed, percentile, trim_sorted


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
