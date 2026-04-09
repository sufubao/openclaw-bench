#!/usr/bin/env python3
"""Print a concise benchmark summary table from result JSON files."""
from __future__ import annotations

import json
import sys
from pathlib import Path


# ── table helpers ──────────────────────────────────────────────────────────────

def _fmt(value, precision=2, suffix=""):
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.{precision}f}{suffix}"
    return f"{value}{suffix}"


def _print_table(title: str, columns: list[str], rows: list[list[str]]) -> None:
    if not rows:
        return
    widths = [len(c) for c in columns]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))

    border_top = "┌" + "┬".join("─" * (w + 2) for w in widths) + "┐"
    border_mid = "├" + "┼".join("─" * (w + 2) for w in widths) + "┤"
    border_bot = "└" + "┴".join("─" * (w + 2) for w in widths) + "┘"

    def fmt_row(cells):
        return "│" + "│".join(f" {c:<{widths[i]}} " for i, c in enumerate(cells)) + "│"

    print()
    print(f"  {title}")
    print(f"  {border_top}")
    print(f"  {fmt_row(columns)}")
    print(f"  {border_mid}")
    for idx, row in enumerate(rows):
        print(f"  {fmt_row(row)}")
        if idx < len(rows) - 1:
            print(f"  {border_mid}")
    print(f"  {border_bot}")
    print()


# ── data loading ──────────────────────────────────────────────────────────────

def _load_results(paths: list[Path]) -> list[tuple[str, dict]]:
    results = []
    for p in sorted(paths):
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            if "summary" in data:
                results.append((p.name, data))
        except Exception as exc:
            print(f"  skip {p.name}: {exc}", file=sys.stderr)
    return results


# ── report ────────────────────────────────────────────────────────────────────

def report(results_dir: Path) -> None:
    json_files = sorted(results_dir.glob("*.json"))
    if not json_files:
        print(f"No result files in {results_dir}")
        return

    results = _load_results(json_files)
    if not results:
        print("No valid benchmark results found.")
        return

    # ── overview table ────────────────────────────────────────────────────
    overview_cols = ["File", "Server", "Run", "Duration", "Busy", "Reqs", "Fail", "Peak"]
    overview_rows = []
    for fname, data in results:
        s = data["summary"]
        overview_rows.append([
            fname,
            data.get("server_label", "-"),
            data.get("run_label", "-"),
            _fmt(data.get("duration_seconds"), 0, "s"),
            _fmt(s.get("busy_seconds"), 0, "s"),
            str(s.get("completed_requests", 0)),
            str(s.get("failed_requests", 0)),
            str(s.get("peak_inflight_requests", "-")),
        ])
    _print_table("Overview", overview_cols, overview_rows)

    # ── throughput table (based on server-busy time) ─────────────────────
    tp_cols = ["File", "TPS (in+out)", "In TPS", "Out TPS", "Out TPM", "Req/s"]
    tp_rows = []
    for fname, data in results:
        s = data["summary"]
        tp_rows.append([
            fname,
            _fmt(s.get("total_token_throughput_tps"), 1),
            _fmt(s.get("prompt_token_throughput_tps"), 1),
            _fmt(s.get("completion_token_throughput_tps"), 1),
            _fmt((s.get("completion_token_throughput_tps") or 0) * 60, 0),
            _fmt(s.get("request_throughput_rps")),
        ])
    _print_table("Throughput (server-busy time only)", tp_cols, tp_rows)

    # ── latency table ─────────────────────────────────────────────────────
    lat_cols = ["File", "TTFT p50", "TTFT p90", "TTFT p99", "TPOT p50", "TPOT p90", "Latency p50"]
    lat_rows = []
    for fname, data in results:
        s = data["summary"]
        ttft = s.get("ttft_seconds", {})
        tpot = s.get("tpot_seconds", {})
        lat = s.get("total_latency_seconds", {})
        lat_rows.append([
            fname,
            _fmt(ttft.get("p50"), 3, "s"),
            _fmt(ttft.get("p90"), 3, "s"),
            _fmt(ttft.get("p99"), 3, "s"),
            _fmt(tpot.get("p50"), 4, "s"),
            _fmt(tpot.get("p90"), 4, "s"),
            _fmt(lat.get("p50"), 2, "s"),
        ])
    _print_table("Latency (all requests)", lat_cols, lat_rows)

    # ── trimmed table (if any result has it) ──────────────────────────────
    has_trimmed = any(data["summary"].get("trimmed") for _, data in results)
    if has_trimmed:
        tr_cols = ["File", "Trim%", "Included", "TPS (in+out)", "Out TPS", "TTFT p50", "TTFT p90", "TPOT p50", "TPOT p90"]
        tr_rows = []
        for fname, data in results:
            tr = data["summary"].get("trimmed")
            if not tr:
                tr_rows.append([fname] + ["-"] * (len(tr_cols) - 1))
                continue
            tr_ttft = tr.get("ttft_seconds", {})
            tr_tpot = tr.get("tpot_seconds", {})
            tr_rows.append([
                fname,
                _fmt(tr.get("trim_percent"), 0, "%"),
                str(tr.get("included_requests", "-")),
                _fmt(tr.get("total_token_throughput_tps"), 1),
                _fmt(tr.get("completion_token_throughput_tps"), 1),
                _fmt(tr_ttft.get("p50"), 3, "s"),
                _fmt(tr_ttft.get("p90"), 3, "s"),
                _fmt(tr_tpot.get("p50"), 4, "s"),
                _fmt(tr_tpot.get("p90"), 4, "s"),
            ])
        _print_table("Trimmed Metrics (middle 80%, server-busy time)", tr_cols, tr_rows)


# ── main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    results_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("results")
    if not results_dir.is_dir():
        print(f"error: {results_dir} is not a directory", file=sys.stderr)
        sys.exit(1)
    report(results_dir)
