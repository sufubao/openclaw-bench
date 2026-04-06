from __future__ import annotations

from openclaw_bench.models import TokenizerSpec
from openclaw_bench.tokenizer import _normalized_model_candidates, build_tokenizer


class DummyEncoding:
    def encode(self, text: str) -> list[int]:
        return [1 for _ in text.split()]

    def decode(self, tokens: list[int]) -> str:
        return " ".join("tok" for _ in tokens)


def test_model_candidates_strip_provider_and_variant_suffix() -> None:
    assert _normalized_model_candidates("openai/gpt-4.1-mini:beta") == [
        "openai/gpt-4.1-mini:beta",
        "openai/gpt-4.1-mini",
        "gpt-4.1-mini:beta",
        "gpt-4.1-mini",
    ]


def test_default_tokenizer_loads_from_request_model(monkeypatch) -> None:
    monkeypatch.setattr("openclaw_bench.tokenizer.AutoTokenizer.from_pretrained", lambda *args, **kwargs: (_ for _ in ()).throw(OSError("skip transformers")))

    def fake_encoding_for_model(model_name: str) -> DummyEncoding:
        if model_name == "gpt-4.1-mini":
            return DummyEncoding()
        raise KeyError(model_name)

    monkeypatch.setattr("openclaw_bench.tokenizer.tiktoken.encoding_for_model", fake_encoding_for_model)
    tokenizer = build_tokenizer(TokenizerSpec(), fallback_model_name="openai/gpt-4.1-mini:beta")
    assert tokenizer.count("alpha beta gamma") == 3
