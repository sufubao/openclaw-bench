from __future__ import annotations

from pathlib import Path

from openclaw_bench.dashboard import _load_results_dir


def test_load_results_dir_reads_valid_metrics_file() -> None:
    results, errors = _load_results_dir(Path("/workspace/openclaw-bench/results"))

    assert len(results) == 1
    assert results[0][0].endswith("results/tiny.json")
    assert errors == []


def test_load_results_dir_reports_missing_directory(tmp_path: Path) -> None:
    results, errors = _load_results_dir(tmp_path / "missing-results")

    assert results == []
    assert len(errors) == 1
    assert "results directory not found" in errors[0]
