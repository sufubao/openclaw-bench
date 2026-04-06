from __future__ import annotations

from openclaw_bench.metrics import describe, percentile


def test_percentile_interpolates() -> None:
    values = [1.0, 2.0, 3.0, 4.0]
    assert percentile(values, 50) == 2.5


def test_describe_contains_expected_percentiles() -> None:
    summary = describe([1.0, 2.0, 3.0, 4.0, 5.0])
    assert summary["count"] == 5
    assert summary["p5"] >= 1.0
    assert summary["p99"] <= 5.0
