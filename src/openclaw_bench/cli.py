from __future__ import annotations

import argparse
import asyncio

import uvicorn

from openclaw_bench.config_generator import generate_config, load_generation_input, write_config
from openclaw_bench.simulator import load_config, resolve_api_key, simulate_config, write_result

DEFAULT_VLLM_BASE_URL = "http://localhost:8000"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="OpenClaw multi-user benchmark harness")
    subparsers = parser.add_subparsers(dest="command", required=True)

    generate_parser = subparsers.add_parser("generate-config", help="Generate a deterministic simulation config JSON")
    generate_parser.add_argument("--control-file", required=True, help="JSON file describing workload distributions")
    generate_parser.add_argument("--output", required=True, help="Where to write the generated simulation config")

    simulate_parser = subparsers.add_parser("simulate", help="Execute a deterministic config against /v1/chat/completions")
    simulate_parser.add_argument("--config", required=True, help="Generated simulation config JSON")
    simulate_parser.add_argument("--output", required=True, help="Where to write the benchmark result JSON")
    simulate_parser.add_argument(
        "--base-url",
        default=DEFAULT_VLLM_BASE_URL,
        help=f"Base URL for the OpenAI-compatible server. Defaults to the local vLLM server at {DEFAULT_VLLM_BASE_URL}",
    )
    simulate_parser.add_argument("--api-key", help="Optional API key override")
    simulate_parser.add_argument(
        "--api-key-env",
        default="OPENAI_API_KEY",
        help="Environment variable used to resolve the API key. OPENROUTER_API_KEY is also checked as a fallback",
    )
    simulate_parser.add_argument("--run-label", default="benchmark-run", help="Human-friendly label for this benchmark run")
    simulate_parser.add_argument("--server-label", default="vllm-local", help="Label used when comparing result files")
    simulate_parser.add_argument("--http-referer", help="Optional OpenRouter HTTP-Referer header")
    simulate_parser.add_argument("--title", help="Optional OpenRouter X-Title header")

    dashboard_parser = subparsers.add_parser("serve-dashboard", help="Launch the local comparison dashboard")
    dashboard_parser.add_argument("--host", default="127.0.0.1")
    dashboard_parser.add_argument("--port", default=8000, type=int)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "generate-config":
        control = load_generation_input(args.control_file)
        config = generate_config(control)
        output = write_config(config, args.output)
        print(f"wrote config {config.config_uuid} to {output}")
        return

    if args.command == "simulate":
        config = load_config(args.config)
        headers: dict[str, str] = {}
        if args.http_referer:
            headers["HTTP-Referer"] = args.http_referer
        if args.title:
            headers["X-Title"] = args.title
        api_key = resolve_api_key(explicit_key=args.api_key, env_var=args.api_key_env)
        result = asyncio.run(
            simulate_config(
                config=config,
                base_url=args.base_url,
                api_key=api_key,
                run_label=args.run_label,
                server_label=args.server_label,
                extra_headers=headers,
            )
        )
        output = write_result(result, args.output)
        print(f"wrote result {result.run_uuid} to {output}")
        return

    if args.command == "serve-dashboard":
        from openclaw_bench.dashboard import build_dashboard_app

        app = build_dashboard_app()
        uvicorn.run(app, host=args.host, port=args.port)
        return

    parser.error(f"unknown command: {args.command}")


if __name__ == "__main__":
    main()
