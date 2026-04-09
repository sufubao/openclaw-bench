from __future__ import annotations

import base64
import json
import os
import tempfile
from io import BytesIO
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "openclaw-bench-mpl"))

import matplotlib
matplotlib.use("Agg")

from fastapi import FastAPI, File, Request, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from matplotlib import pyplot as plt

from openclaw_bench.metrics import PERCENTILES
from openclaw_bench.models import BenchmarkResult

TEMPLATE_DIR = Path(__file__).parent / "templates"
DEFAULT_RESULTS_DIR = Path.cwd() / "results"


def build_dashboard_app() -> FastAPI:
    app = FastAPI(title="OpenClaw Bench Dashboard")
    templates = Jinja2Templates(directory=str(TEMPLATE_DIR))

    @app.get("/", response_class=HTMLResponse)
    async def home(request: Request) -> HTMLResponse:
        results, errors = _load_results_dir(DEFAULT_RESULTS_DIR)
        return templates.TemplateResponse(
            request,
            "dashboard.html",
            {
                "request": request,
                "groups": _build_groups(results),
                "errors": errors,
                "files_loaded": len(results),
                "results_dir": _display_path(DEFAULT_RESULTS_DIR),
            },
        )

    @app.post("/compare", response_class=HTMLResponse)
    async def compare(request: Request, files: list[UploadFile] = File(...)) -> HTMLResponse:
        results: list[tuple[str, BenchmarkResult]] = []
        errors: list[str] = []
        for uploaded in files:
            try:
                payload = await uploaded.read()
                model = BenchmarkResult.model_validate(json.loads(payload))
                results.append((uploaded.filename or "uploaded-result.json", model))
            except Exception as exc:
                errors.append(f"{uploaded.filename or 'unknown file'}: {exc}")

        groups = _build_groups(results)
        return templates.TemplateResponse(
            request,
            "dashboard.html",
            {
                "request": request,
                "groups": groups,
                "errors": errors,
                "files_loaded": len(results),
                "results_dir": None,
            },
        )

    return app


def _load_results_dir(results_dir: Path) -> tuple[list[tuple[str, BenchmarkResult]], list[str]]:
    if not results_dir.exists():
        return [], [f"results directory not found: {_display_path(results_dir)}"]

    results: list[tuple[str, BenchmarkResult]] = []
    errors: list[str] = []
    for path in sorted(results_dir.rglob("*.json")):
        try:
            model = BenchmarkResult.model_validate_json(path.read_text(encoding="utf-8"))
            results.append((_display_path(path), model))
        except Exception as exc:
            errors.append(f"{_display_path(path)}: {exc}")
    return results, errors


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(Path.cwd()))
    except ValueError:
        return str(path)


def _build_groups(results: list[tuple[str, BenchmarkResult]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[tuple[str, BenchmarkResult]]] = {}
    for filename, result in results:
        grouped.setdefault(result.config_uuid, []).append((filename, result))

    rendered_groups: list[dict[str, Any]] = []
    for config_uuid, group_results in sorted(grouped.items(), key=lambda item: item[0]):
        summary_rows = [_summary_row(filename, result) for filename, result in group_results]

        # Build sidebar info: collect unique models and server labels
        models = sorted({r.model.split("/")[-1] for _, r in group_results})
        servers = sorted({r.server_label for _, r in group_results})

        rendered_groups.append(
            {
                "config_uuid": config_uuid,
                "short_uuid": config_uuid[:8],
                "models": models,
                "servers": servers,
                "run_count": len(group_results),
                "rows": summary_rows,
                "throughput_chart": _bar_chart(
                    group_results,
                    metric_key="completion_token_throughput_tps",
                    title="Completion Token Throughput",
                    ylabel="tokens / second",
                ),
                "ttft_chart": _percentile_chart(
                    group_results,
                    metric_group="ttft_seconds",
                    title="TTFT Percentiles",
                    ylabel="seconds",
                ),
                "tpot_chart": _percentile_chart(
                    group_results,
                    metric_group="tpot_seconds",
                    title="TPOT Percentiles",
                    ylabel="seconds / token",
                ),
                "latency_chart": _bar_chart(
                    group_results,
                    metric_key="request_throughput_rps",
                    title="Request Throughput",
                    ylabel="requests / second",
                ),
                "tr_throughput_chart": _trimmed_bar_chart(
                    group_results,
                    metric_key="completion_token_throughput_tps",
                    title="Trimmed Completion Token Throughput",
                    ylabel="tokens / second",
                ),
                "tr_ttft_chart": _trimmed_percentile_chart(
                    group_results,
                    metric_group="ttft_seconds",
                    title="Trimmed TTFT Percentiles",
                    ylabel="seconds",
                ),
            }
        )
    return rendered_groups


def _summary_row(filename: str, result: BenchmarkResult) -> dict[str, Any]:
    ttft = result.summary.get("ttft_seconds", {})
    tpot = result.summary.get("tpot_seconds", {})
    trimmed = result.summary.get("trimmed", {})
    tr_ttft = trimmed.get("ttft_seconds", {})
    tr_tpot = trimmed.get("tpot_seconds", {})
    tr_tps = trimmed.get("completion_token_throughput_tps")
    return {
        "filename": filename,
        "run_label": result.run_label,
        "server_label": result.server_label,
        "run_uuid": result.run_uuid,
        "model": result.model,
        "completed_requests": result.summary.get("completed_requests"),
        "failed_requests": result.summary.get("failed_requests"),
        "request_throughput_rps": _fmt(result.summary.get("request_throughput_rps")),
        "completion_token_throughput_tps": _fmt(result.summary.get("completion_token_throughput_tps")),
        "tpm": _fmt(
            (result.summary.get("completion_token_throughput_tps") or 0) * 60
            if result.summary.get("completion_token_throughput_tps") is not None
            else None
        ),
        "ttft_p50": _fmt(ttft.get("p50")),
        "ttft_p90": _fmt(ttft.get("p90")),
        "ttft_p99": _fmt(ttft.get("p99")),
        "tpot_p50": _fmt(tpot.get("p50")),
        "tpot_p90": _fmt(tpot.get("p90")),
        "tpot_p99": _fmt(tpot.get("p99")),
        # Trimmed (middle-80 %) metrics
        "trim_percent": trimmed.get("trim_percent"),
        "tr_included": trimmed.get("included_requests"),
        "tr_tps": _fmt(tr_tps),
        "tr_tpm": _fmt(tr_tps * 60 if tr_tps else None),
        "tr_ttft_p50": _fmt(tr_ttft.get("p50")),
        "tr_ttft_p90": _fmt(tr_ttft.get("p90")),
        "tr_ttft_p99": _fmt(tr_ttft.get("p99")),
        "tr_tpot_p50": _fmt(tr_tpot.get("p50")),
        "tr_tpot_p90": _fmt(tr_tpot.get("p90")),
        "tr_tpot_p99": _fmt(tr_tpot.get("p99")),
    }


def _fmt(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, (int, float)):
        return f"{value:.3f}"
    return str(value)


_CHART_PALETTE = ["#818cf8", "#22d3ee", "#fbbf24", "#f87171", "#34d399", "#a78bfa"]
_CHART_BAR_COLORS = ["#818cf8", "#22d3ee", "#fbbf24", "#f87171", "#34d399", "#a78bfa"]


def _apply_chart_style():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Inter", "Helvetica Neue", "Arial", "sans-serif"],
        "font.size": 11,
        "axes.facecolor": "#fafbfd",
        "axes.edgecolor": "#e2e8f0",
        "axes.grid": True,
        "grid.color": "#edf2f7",
        "grid.linestyle": "-",
        "grid.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.spines.left": False,
        "axes.spines.bottom": True,
        "xtick.color": "#94a3b8",
        "ytick.color": "#94a3b8",
        "text.color": "#64748b",
    })


def _chart_to_base64() -> str:
    buffer = BytesIO()
    plt.savefig(buffer, format="png", bbox_inches="tight", dpi=150, facecolor="#fafbfd")
    plt.close()
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def _bar_chart(
    results: list[tuple[str, BenchmarkResult]],
    metric_key: str,
    title: str,
    ylabel: str,
) -> str | None:
    labels = [f"{result.server_label}\n{result.run_label}" for _, result in results]
    values = [float(result.summary.get(metric_key, 0.0) or 0.0) for _, result in results]
    if not any(values):
        return None
    _apply_chart_style()
    fig, ax = plt.subplots(figsize=(7, 3.5))
    colors = [_CHART_BAR_COLORS[i % len(_CHART_BAR_COLORS)] for i in range(len(labels))]
    bars = ax.bar(labels, values, color=colors, alpha=0.9, width=0.55, edgecolor="white", linewidth=1.5)
    ax.bar_label(bars, fmt="%.1f", fontsize=9, color="#475569", padding=6)
    ax.set_title(title, fontsize=13, fontweight=600, color="#1e293b", pad=14, loc="left")
    ax.set_ylabel(ylabel, fontsize=10, color="#64748b")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=0, ha="center", fontsize=10)
    fig.tight_layout()
    return _chart_to_base64()


def _trimmed_bar_chart(
    results: list[tuple[str, BenchmarkResult]],
    metric_key: str,
    title: str,
    ylabel: str,
) -> str | None:
    """Bar chart reading from summary['trimmed'][metric_key]."""
    labels = [f"{result.server_label}\n{result.run_label}" for _, result in results]
    values = [float((result.summary.get("trimmed") or {}).get(metric_key, 0.0) or 0.0) for _, result in results]
    if not any(values):
        return None
    _apply_chart_style()
    fig, ax = plt.subplots(figsize=(7, 3.5))
    colors = [_CHART_BAR_COLORS[i % len(_CHART_BAR_COLORS)] for i in range(len(labels))]
    bars = ax.bar(labels, values, color=colors, alpha=0.9, width=0.55, edgecolor="white", linewidth=1.5)
    ax.bar_label(bars, fmt="%.1f", fontsize=9, color="#475569", padding=6)
    ax.set_title(title, fontsize=13, fontweight=600, color="#1e293b", pad=14, loc="left")
    ax.set_ylabel(ylabel, fontsize=10, color="#64748b")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=0, ha="center", fontsize=10)
    fig.tight_layout()
    return _chart_to_base64()


def _trimmed_percentile_chart(
    results: list[tuple[str, BenchmarkResult]],
    metric_group: str,
    title: str,
    ylabel: str,
) -> str | None:
    """Percentile chart reading from summary['trimmed'][metric_group]."""
    _apply_chart_style()
    fig, ax = plt.subplots(figsize=(7, 3.5))
    plotted = False
    for idx, (_, result) in enumerate(results):
        metrics = (result.summary.get("trimmed") or {}).get(metric_group, {})
        ys = [metrics.get(f"p{p}") for p in PERCENTILES]
        if not any(v is not None for v in ys):
            continue
        plotted = True
        color = _CHART_PALETTE[idx % len(_CHART_PALETTE)]
        ax.plot(
            list(PERCENTILES),
            [float(v or 0.0) for v in ys],
            marker="o", markersize=5, linewidth=2.2, color=color,
            label=f"{result.server_label} | {result.run_label}", zorder=3,
        )
    if not plotted:
        plt.close()
        return None
    ax.set_title(title, fontsize=13, fontweight=600, color="#1e293b", pad=14, loc="left")
    ax.set_xlabel("Percentile", fontsize=10, color="#64748b")
    ax.set_ylabel(ylabel, fontsize=10, color="#64748b")
    ax.set_xticks(list(PERCENTILES))
    ax.legend(frameon=True, framealpha=0.95, edgecolor="#e2e8f0", fontsize=9, fancybox=True)
    fig.tight_layout()
    return _chart_to_base64()


def _percentile_chart(
    results: list[tuple[str, BenchmarkResult]],
    metric_group: str,
    title: str,
    ylabel: str,
) -> str | None:
    _apply_chart_style()
    fig, ax = plt.subplots(figsize=(7, 3.5))
    plotted = False
    for idx, (_, result) in enumerate(results):
        metrics = result.summary.get(metric_group, {})
        ys = [metrics.get(f"p{percentile}") for percentile in PERCENTILES]
        if not any(value is not None for value in ys):
            continue
        plotted = True
        color = _CHART_PALETTE[idx % len(_CHART_PALETTE)]
        ax.plot(
            list(PERCENTILES),
            [float(value or 0.0) for value in ys],
            marker="o",
            markersize=5,
            linewidth=2.2,
            color=color,
            label=f"{result.server_label} | {result.run_label}",
            zorder=3,
        )
    if not plotted:
        plt.close()
        return None
    ax.set_title(title, fontsize=13, fontweight=600, color="#1e293b", pad=14, loc="left")
    ax.set_xlabel("Percentile", fontsize=10, color="#64748b")
    ax.set_ylabel(ylabel, fontsize=10, color="#64748b")
    ax.set_xticks(list(PERCENTILES))
    ax.legend(frameon=True, framealpha=0.95, edgecolor="#e2e8f0", fontsize=9, fancybox=True)
    fig.tight_layout()
    return _chart_to_base64()
