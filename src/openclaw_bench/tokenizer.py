from __future__ import annotations

import re
from pathlib import Path
from typing import Sequence

import tiktoken
from tokenizers import Tokenizer as HFTokenizer
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from openclaw_bench.models import TokenizerSpec


class TokenizerAdapter:
    def __init__(self, encoding: tiktoken.Encoding):
        self.encoding = encoding

    def encode(self, text: str) -> list[int]:
        return self.encoding.encode(text)

    def decode(self, tokens: Sequence[int]) -> str:
        return self.encoding.decode(list(tokens))

    def count(self, text: str) -> int:
        return len(self.encode(text))

    def trim_to_tokens(self, text: str, token_budget: int) -> str:
        if token_budget <= 0:
            return ""
        tokens = self.encode(text)
        return self.decode(tokens[:token_budget])

    def estimate_chat_tokens(self, messages: list[dict[str, str]]) -> int:
        total = 3
        for message in messages:
            total += 3
            total += self.count(message.get("content", ""))
            if "name" in message:
                total += 1
        return total


class RegexTokenizerAdapter(TokenizerAdapter):
    TOKEN_PATTERN = re.compile(r"\w+|[^\w\s]", re.UNICODE)

    def __init__(self):
        pass

    def encode(self, text: str) -> list[int]:
        return list(range(len(self.TOKEN_PATTERN.findall(text))))

    def decode(self, tokens: Sequence[int]) -> str:
        return " ".join(f"tok{i}" for i in tokens)

    def count(self, text: str) -> int:
        return len(self.TOKEN_PATTERN.findall(text))

    def trim_to_tokens(self, text: str, token_budget: int) -> str:
        if token_budget <= 0:
            return ""
        fragments = self.TOKEN_PATTERN.findall(text)
        return " ".join(fragments[:token_budget])


class TokenizerFileAdapter(TokenizerAdapter):
    def __init__(self, tokenizer: HFTokenizer):
        self.tokenizer = tokenizer

    def encode(self, text: str) -> list[int]:
        return self.tokenizer.encode(text).ids

    def decode(self, tokens: Sequence[int]) -> str:
        return self.tokenizer.decode(list(tokens))


class TransformersTokenizerAdapter(TokenizerAdapter):
    def __init__(self, tokenizer: PreTrainedTokenizerBase):
        self.tokenizer = tokenizer

    def encode(self, text: str) -> list[int]:
        return self.tokenizer.encode(text, add_special_tokens=False)

    def decode(self, tokens: Sequence[int]) -> str:
        return self.tokenizer.decode(list(tokens), skip_special_tokens=False)


def _normalized_model_candidates(model_name: str) -> list[str]:
    raw = model_name.strip()
    if not raw:
        return []

    candidates: list[str] = []
    seen: set[str] = set()

    def add(candidate: str) -> None:
        value = candidate.strip()
        if value and value not in seen:
            seen.add(value)
            candidates.append(value)

    add(raw)
    add(raw.rsplit(":", 1)[0])
    if "/" in raw:
        suffix = raw.split("/", 1)[1]
        add(suffix)
        add(suffix.rsplit(":", 1)[0])
    return candidates


def _build_tiktoken_from_model(model_name: str) -> TokenizerAdapter | None:
    for candidate in _normalized_model_candidates(model_name):
        try:
            return TokenizerAdapter(tiktoken.encoding_for_model(candidate))
        except KeyError:
            continue
    return None


def _build_transformers_from_model(spec: TokenizerSpec, model_name: str) -> TokenizerAdapter | None:
    for candidate in _normalized_model_candidates(model_name):
        local_only = spec.local_files_only or Path(candidate).exists()
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                candidate,
                use_fast=True,
                trust_remote_code=spec.trust_remote_code,
                local_files_only=local_only,
            )
        except Exception:
            continue
        return TransformersTokenizerAdapter(tokenizer)
    return None


def _resolve_model_name(spec: TokenizerSpec, fallback_model_name: str | None) -> str:
    model_name = spec.model_name or fallback_model_name
    if not model_name:
        raise RuntimeError("tokenizer kind 'model' requires a model name or request.model fallback")
    return model_name


def build_tokenizer(spec: TokenizerSpec, fallback_model_name: str | None = None) -> TokenizerAdapter:
    if spec.kind == "regex":
        return RegexTokenizerAdapter()
    if spec.kind == "tokenizer_file":
        return TokenizerFileAdapter(HFTokenizer.from_file(spec.tokenizer_path))
    if spec.kind == "model":
        model_name = _resolve_model_name(spec, fallback_model_name)
        tokenizer = _build_tiktoken_from_model(model_name)
        if tokenizer is not None:
            return tokenizer
        tokenizer = _build_transformers_from_model(spec, model_name)
        if tokenizer is not None:
            return tokenizer
        raise RuntimeError(f"could not load a tokenizer from model '{model_name}'")
    model_name = spec.model_name or fallback_model_name
    encoding_name = spec.encoding_name
    if model_name:
        tokenizer = _build_tiktoken_from_model(model_name)
        if tokenizer is not None:
            return tokenizer
    return TokenizerAdapter(tiktoken.get_encoding(encoding_name or "cl100k_base"))
