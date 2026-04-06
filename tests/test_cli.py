from __future__ import annotations

from openclaw_bench.cli import build_parser


def test_simulate_defaults_to_local_vllm() -> None:
    parser = build_parser()
    args = parser.parse_args(["simulate", "--config", "out/config.json", "--output", "out/result.json"])

    assert args.base_url == "http://localhost:8000"
    assert args.api_key_env == "OPENAI_API_KEY"
    assert args.server_label == "vllm-local"
