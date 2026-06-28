"""Tests for memory-arena specific settings (memory strategies, postgres, vendor keys)."""

from __future__ import annotations

import os
from unittest.mock import patch

from memory_arena.settings import Settings


class TestMemoryStrategySettings:
    def test_full_context_token_budget_default(self):
        s = Settings()
        assert s.full_context_token_budget == 150000

    def test_recency_window_n_default(self):
        s = Settings()
        assert s.recency_window_n == 20

    def test_recall_default_top_k(self):
        s = Settings()
        assert s.recall_default_top_k == 10

    def test_full_context_via_env(self):
        with patch.dict(os.environ, {"FULL_CONTEXT_TOKEN_BUDGET": "50000"}):
            s = Settings(_env_file=None)
            assert s.full_context_token_budget == 50000

    def test_recency_window_via_env(self):
        with patch.dict(os.environ, {"RECENCY_WINDOW_N": "5"}):
            s = Settings(_env_file=None)
            assert s.recency_window_n == 5

    def test_full_context_legacy_alias(self):
        # Backward-compat: MEM_ARENA_-prefixed names still resolve.
        with patch.dict(os.environ, {"MEM_ARENA_FULL_CONTEXT_TOKEN_BUDGET": "12345"}):
            s = Settings(_env_file=None)
            assert s.full_context_token_budget == 12345


class TestPostgresSettings:
    def test_postgres_host_default(self):
        s = Settings()
        assert s.postgres_host == "localhost"

    def test_postgres_port_default(self):
        s = Settings()
        assert s.postgres_port == 5432

    def test_postgres_user_default(self):
        s = Settings()
        assert s.postgres_user == "memarena"

    def test_postgres_database_default(self):
        s = Settings()
        assert s.postgres_database == "memarena"

    def test_postgres_via_env(self):
        with patch.dict(os.environ, {"POSTGRES_HOST": "db.example.com"}):
            s = Settings(_env_file=None)
            assert s.postgres_host == "db.example.com"


class TestVendorSettings:
    def test_mem0_collection_prefix_default(self):
        s = Settings()
        assert s.mem0_collection_prefix == "mem0"

    def test_mem0_api_key_empty_default(self):
        s = Settings()
        assert s.mem0_api_key == ""

    def test_zep_api_key_empty_default(self):
        s = Settings()
        assert s.zep_api_key == ""

    def test_graphiti_group_prefix_default(self):
        s = Settings()
        assert s.graphiti_group_prefix == "ma"


class TestJudgeModel:
    def test_judge_model_is_opus(self):
        s = Settings()
        assert "opus" in s.judge_model

    def test_judge_distinct_from_generate(self):
        s = Settings()
        assert s.judge_model != s.generate_model
