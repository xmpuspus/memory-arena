"""OpenRouter LLM provider.

OpenRouter (https://openrouter.ai) is a unified gateway for many open and closed
LLMs (Llama 3.3, Qwen, DeepSeek, Gemini Flash, etc). The wire format is the
OpenAI Chat Completions API, so we reuse the OpenAI SDK with a different base
URL + API key.

This module is intentionally a thin wrapper around openai.AsyncOpenAI — the
behavioural contract matches :class:`OpenAIProvider` so the rest of the LLM
client (retry, timeout, cost accounting, streaming) stays provider-agnostic.

Cost accounting uses a small per-model price table below. Update prices as
OpenRouter changes them — verified 2026-05-11 from per-model OpenRouter pages.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator

from memory_arena.llm.providers import LLMProvider, ProviderResponse

log = logging.getLogger(__name__)


OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


# Per-million-token pricing (USD), verified from openrouter.ai/<model> pages on
# 2026-05-11. Keep this table small — only the curated set we benchmark with.
OPENROUTER_PRICING: dict[str, dict[str, float]] = {
    "meta-llama/llama-3.3-70b-instruct": {"input": 0.10, "output": 0.32},
    "qwen/qwen-2.5-72b-instruct": {"input": 0.36, "output": 0.40},
    "deepseek/deepseek-chat": {"input": 0.32, "output": 0.89},
    "google/gemini-2.0-flash-001": {"input": 0.10, "output": 0.40},
}


# Slug prefixes that should auto-route to OpenRouter in the LLMClient factory.
OPENROUTER_MODEL_PREFIXES: tuple[str, ...] = (
    "meta-llama/",
    "qwen/",
    "deepseek/",
    "google/",
    "mistralai/",
    "anthropic/",  # OpenRouter mirrors Anthropic too
    "openai/",  # ditto
)


def is_openrouter_model(model: str) -> bool:
    """Return True if a model id looks like an OpenRouter slug (provider/model)."""
    return any(model.startswith(p) for p in OPENROUTER_MODEL_PREFIXES)


class OpenRouterProvider(LLMProvider):
    """OpenRouter backend — OpenAI-compatible chat completions."""

    def __init__(self, api_key: str, base_url: str = OPENROUTER_BASE_URL):
        import openai

        self.client = openai.AsyncOpenAI(api_key=api_key, base_url=base_url)
        self.last_stream_response: ProviderResponse | None = None

    async def complete(
        self,
        model: str,
        system: str,
        user: str,
        max_tokens: int = 4096,
        temperature: float = 0,
    ) -> ProviderResponse:
        response = await self.client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        choice = response.choices[0]
        usage = response.usage
        return ProviderResponse(
            text=choice.message.content or "",
            input_tokens=usage.prompt_tokens if usage else 0,
            output_tokens=usage.completion_tokens if usage else 0,
            model=model,
        )

    async def stream_text(
        self,
        model: str,
        system: str,
        user: str,
        max_tokens: int = 4096,
        temperature: float = 0,
    ) -> AsyncIterator[str]:
        stream = await self.client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
            stream_options={"include_usage": True},
        )
        last_usage = None
        async for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
            if getattr(chunk, "usage", None):
                last_usage = chunk.usage

        self.last_stream_response = ProviderResponse(
            text="",
            input_tokens=last_usage.prompt_tokens if last_usage else 0,
            output_tokens=last_usage.completion_tokens if last_usage else 0,
            model=model,
        )
