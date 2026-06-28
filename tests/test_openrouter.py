"""Tests for the OpenRouter LLM provider."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import openai
import pytest

from memory_arena.llm.openrouter import (
    OPENROUTER_BASE_URL,
    OPENROUTER_MODEL_PREFIXES,
    OPENROUTER_PRICING,
    OpenRouterProvider,
    is_openrouter_model,
)


class TestSlugRouting:
    def test_known_prefixes_route_to_openrouter(self):
        for prefix in OPENROUTER_MODEL_PREFIXES:
            assert is_openrouter_model(f"{prefix}some-model")

    def test_bare_anthropic_model_does_not_route(self):
        # Plain Claude model names (no "provider/" prefix) stay on the native
        # Anthropic provider.
        assert not is_openrouter_model("claude-sonnet-4-6")
        assert not is_openrouter_model("gpt-4o-mini")
        assert not is_openrouter_model("llama3.1:8b")


class TestPricingTable:
    def test_curated_models_priced(self):
        # The four models the audit asked us to support must be in the table.
        for slug in (
            "meta-llama/llama-3.3-70b-instruct",
            "qwen/qwen-2.5-72b-instruct",
            "deepseek/deepseek-chat",
            "google/gemini-2.0-flash-001",
        ):
            entry = OPENROUTER_PRICING[slug]
            assert entry["input"] > 0
            assert entry["output"] > 0


class TestOpenRouterProvider:
    def test_constructor_uses_openrouter_base_url(self):
        with patch("openai.AsyncOpenAI") as ctor:
            OpenRouterProvider(api_key="sk-or-test")
            ctor.assert_called_once()
            kwargs = ctor.call_args.kwargs
            assert kwargs["api_key"] == "sk-or-test"
            assert kwargs["base_url"] == OPENROUTER_BASE_URL

    def test_constructor_respects_custom_base_url(self):
        with patch("openai.AsyncOpenAI") as ctor:
            OpenRouterProvider(api_key="sk", base_url="https://example.test/v1")
            assert ctor.call_args.kwargs["base_url"] == "https://example.test/v1"

    @pytest.mark.asyncio
    async def test_complete_returns_text_and_usage(self):
        mock_response = MagicMock()
        mock_choice = MagicMock()
        mock_choice.message.content = "ok"
        mock_response.choices = [mock_choice]
        mock_response.usage.prompt_tokens = 42
        mock_response.usage.completion_tokens = 7

        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

        with patch("openai.AsyncOpenAI", return_value=mock_client):
            p = OpenRouterProvider(api_key="sk-or-test")
            resp = await p.complete(
                model="meta-llama/llama-3.3-70b-instruct",
                system="be terse",
                user="hi",
            )
        assert resp.text == "ok"
        assert resp.input_tokens == 42
        assert resp.output_tokens == 7
        assert resp.model == "meta-llama/llama-3.3-70b-instruct"

    @pytest.mark.asyncio
    async def test_complete_handles_none_content(self):
        mock_response = MagicMock()
        mock_choice = MagicMock()
        mock_choice.message.content = None
        mock_response.choices = [mock_choice]
        mock_response.usage = None

        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

        with patch("openai.AsyncOpenAI", return_value=mock_client):
            p = OpenRouterProvider(api_key="sk")
            resp = await p.complete(model="qwen/qwen-2.5-72b-instruct", system="", user="")
        assert resp.text == ""
        assert resp.input_tokens == 0
        assert resp.output_tokens == 0

    @pytest.mark.asyncio
    async def test_complete_propagates_rate_limit_error(self):
        """A 429 from upstream should bubble up so the LLMClient retry loop sees it."""
        mock_client = AsyncMock()
        # Use a real OpenAI exception type so tenacity's retry matcher works.
        mock_client.chat.completions.create = AsyncMock(
            side_effect=openai.RateLimitError(
                "rate limited",
                response=MagicMock(status_code=429, request=MagicMock()),
                body={"error": "too many"},
            )
        )
        with patch("openai.AsyncOpenAI", return_value=mock_client):
            p = OpenRouterProvider(api_key="sk")
            with pytest.raises(openai.RateLimitError):
                await p.complete(model="deepseek/deepseek-chat", system="", user="hi")

    @pytest.mark.asyncio
    async def test_complete_propagates_server_error(self):
        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(
            side_effect=openai.InternalServerError(
                "boom",
                response=MagicMock(status_code=500, request=MagicMock()),
                body=None,
            )
        )
        with patch("openai.AsyncOpenAI", return_value=mock_client):
            p = OpenRouterProvider(api_key="sk")
            with pytest.raises(openai.InternalServerError):
                await p.complete(model="google/gemini-2.0-flash-001", system="", user="hi")


class TestLLMClientAutoRoutes:
    """When the caller passes an OpenRouter slug into LLMClient, the provider
    factory should pick OpenRouterProvider regardless of settings.llm_provider."""

    def test_slug_overrides_default_provider(self, monkeypatch):
        from memory_arena.llm.client import LLMClient
        from memory_arena.llm.openrouter import OpenRouterProvider

        monkeypatch.setattr("memory_arena.settings.settings.llm_provider", "anthropic")
        monkeypatch.setattr("memory_arena.settings.settings.openrouter_api_key", "sk-or-test")

        with patch("openai.AsyncOpenAI"):
            c = LLMClient(model="meta-llama/llama-3.3-70b-instruct")

        assert isinstance(c._provider, OpenRouterProvider)
        assert c._models["generate"] == "meta-llama/llama-3.3-70b-instruct"
