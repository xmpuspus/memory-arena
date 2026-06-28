"""Shared test fixtures for memory-arena."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from memory_arena.llm.client import LLMResponse
from memory_arena.sessions.schema import (
    Constraints,
    FactAssertion,
    GroundTruth,
    QuestionRecord,
    Session,
    Turn,
)


@pytest.fixture
def sample_turn() -> Turn:
    return Turn(
        id="session_03_turn_001",
        session_id="session_03",
        role="user",
        content="I work as a software engineer at a fintech startup, focusing on payments.",
        timestamp="2026-03-12T10:00:00Z",
    )


@pytest.fixture
def sample_session(sample_turn: Turn) -> Session:
    return Session(
        id="session_03",
        user_id="user_42",
        timestamp="2026-03-12T10:00:00Z",
        turns=[
            sample_turn,
            Turn(
                id="session_03_turn_002",
                session_id="session_03",
                role="assistant",
                content="That's an interesting field. What are you currently working on?",
                timestamp="2026-03-12T10:00:30Z",
            ),
            Turn(
                id="session_03_turn_003",
                session_id="session_03",
                role="user",
                content="A new fraud detection pipeline using transformer embeddings.",
                timestamp="2026-03-12T10:01:00Z",
            ),
        ],
        metadata={"category": "information_extraction"},
    )


@pytest.fixture
def sample_sessions(sample_session: Session) -> list[Session]:
    second = Session(
        id="session_05",
        user_id="user_42",
        timestamp="2026-03-19T11:00:00Z",
        turns=[
            Turn(
                id="session_05_turn_001",
                session_id="session_05",
                role="user",
                content="My favourite programming language is Python.",
                timestamp="2026-03-19T11:00:00Z",
            ),
            Turn(
                id="session_05_turn_002",
                session_id="session_05",
                role="assistant",
                content="Got it.",
                timestamp="2026-03-19T11:00:10Z",
            ),
        ],
    )
    return [sample_session, second]


@pytest.fixture
def sample_question() -> QuestionRecord:
    return QuestionRecord(
        id="longmemeval-extract-001",
        category="information_extraction",
        hops=1,
        question="What kind of work does the user do?",
        ground_truth=GroundTruth(
            answer="Software engineer at a fintech startup, working on payments.",
            supporting_session_ids=["session_03"],
            supporting_turn_ids=["session_03_turn_001"],
        ),
        constraints=Constraints(
            must_mention=["software engineer", "fintech"],
            must_not_claim=["currently unemployed"],
            abstention_expected=False,
            max_tokens=200,
        ),
    )


@pytest.fixture
def abstention_question() -> QuestionRecord:
    return QuestionRecord(
        id="longmemeval-abstain-001",
        category="abstention",
        question="What is the user's social security number?",
        ground_truth=GroundTruth(
            answer="The user has not shared this information.",
            supporting_session_ids=[],
        ),
        constraints=Constraints(abstention_expected=True),
    )


@pytest.fixture
def update_question() -> QuestionRecord:
    return QuestionRecord(
        id="longmemeval-update-001",
        category="knowledge_update",
        question="What city does the user live in?",
        ground_truth=GroundTruth(
            answer="Manila",
            supporting_session_ids=["session_07"],
            fact_versions=[
                FactAssertion(value="Tokyo", source_session_id="session_03"),
                FactAssertion(value="Manila", source_session_id="session_07"),
            ],
        ),
        constraints=Constraints(must_mention=["Manila"]),
    )


@pytest.fixture
def temporal_question() -> QuestionRecord:
    return QuestionRecord(
        id="longmemeval-temporal-001",
        category="temporal",
        question="When did the user first mention they enjoy mountain biking?",
        ground_truth=GroundTruth(
            answer="In session_04 on 2026-03-15.",
            supporting_session_ids=["session_04"],
            valid_as_of="2026-03-15",
        ),
        constraints=Constraints(must_mention=["2026"]),
    )


@pytest.fixture
def mock_llm_client():
    client = AsyncMock()
    client.classify.return_value = "NO"
    client.generate.return_value = LLMResponse(
        text="The user is a software engineer at a fintech startup [session_03].",
        input_tokens=100,
        output_tokens=50,
        cost_usd=0.0005,
    )
    client.judge.return_value = LLMResponse(
        text='{"accuracy": 90, "completeness": 85, "rationale": "Matches the reference closely."}',
        input_tokens=200,
        output_tokens=30,
        cost_usd=0.001,
    )
    return client


@pytest.fixture
def mock_neo4j_driver():
    driver = MagicMock()
    session = AsyncMock()
    result = AsyncMock()
    summary = MagicMock()
    summary.counters.nodes_created = 5
    summary.counters.relationships_created = 3
    result.consume.return_value = summary
    result.data.return_value = []
    session.run.return_value = result
    ctx = AsyncMock()
    ctx.__aenter__ = AsyncMock(return_value=session)
    ctx.__aexit__ = AsyncMock(return_value=False)
    driver.session.return_value = ctx
    return driver


@pytest.fixture
def mock_chroma_collection():
    collection = MagicMock()
    collection.query.return_value = {
        "ids": [["session_03_turn_001"]],
        "documents": [["user: I work as a software engineer at a fintech startup"]],
        "metadatas": [
            [
                {
                    "session_id": "session_03",
                    "turn_id": "session_03_turn_001",
                    "user_id": "user_42",
                    "role": "user",
                    "timestamp": "2026-03-12T10:00:00Z",
                }
            ]
        ],
        "distances": [[0.12]],
    }
    collection.upsert.return_value = None
    return collection


@pytest.fixture
def mock_chroma_client(mock_chroma_collection):
    client = MagicMock()
    client.get_or_create_collection.return_value = mock_chroma_collection
    client.delete_collection.return_value = None
    return client


@pytest.fixture
def longmemeval_raw_record() -> dict:
    """A single raw LongMemEval record."""
    return {
        "question_id": "qa_extract_001",
        "question_type": "single-session-user",
        "question": "What kind of work does the user do?",
        "answer": "Software engineer at a fintech startup",
        "haystack_session_ids": ["session_03", "session_05"],
        "haystack_dates": ["2026-03-12", "2026-03-19"],
        "haystack_sessions": [
            [
                {"role": "user", "content": "I work as a software engineer at a fintech startup."},
                {"role": "assistant", "content": "That's interesting."},
            ],
            [
                {"role": "user", "content": "My favourite programming language is Python."},
                {"role": "assistant", "content": "Got it."},
            ],
        ],
        "answer_session_ids": ["session_03"],
        "user_id": "user_42",
    }
