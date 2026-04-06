from __future__ import annotations

from openclaw_bench.simulator import resolve_api_key


def test_resolve_api_key_returns_none_when_unset(monkeypatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)

    assert resolve_api_key() is None


def test_resolve_api_key_falls_back_to_legacy_openrouter_env(monkeypatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("OPENROUTER_API_KEY", "legacy-key")

    assert resolve_api_key() == "legacy-key"


def test_resolve_api_key_respects_explicit_override(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "env-key")

    assert resolve_api_key(explicit_key="explicit-key") == "explicit-key"
