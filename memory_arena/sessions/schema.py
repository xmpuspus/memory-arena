"""Pydantic v2 models for chat sessions, turns, facts, and benchmark questions.

Universal schema covering LongMemEval, LoCoMo, and custom chat session corpora.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class Turn(BaseModel):
    """A single user-or-assistant message inside a session."""

    model_config = ConfigDict(extra="forbid")

    id: str
    session_id: str
    role: str  # "user" | "assistant" | "system"
    content: str
    timestamp: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class Session(BaseModel):
    """An ordered chat session: many turns belonging to the same user."""

    model_config = ConfigDict(extra="forbid")

    id: str
    user_id: str = "default"
    timestamp: str | None = None
    turns: list[Turn] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def turn_count(self) -> int:
        return len(self.turns)

    def to_messages(self) -> list[dict[str, str]]:
        """Format as OpenAI/Anthropic-style messages for LLM ingestion."""
        return [{"role": t.role, "content": t.content} for t in self.turns]


class FactAssertion(BaseModel):
    """A single fact derived from a session — used for ground truth and update tracking."""

    model_config = ConfigDict(extra="forbid")

    value: str
    valid_at: str | None = None
    invalid_at: str | None = None
    source_session_id: str
    source_turn_id: str | None = None
    confidence: float = 1.0


class GroundTruth(BaseModel):
    """Ground truth for a benchmark question."""

    model_config = ConfigDict(extra="forbid")

    answer: str
    supporting_session_ids: list[str] = Field(default_factory=list)
    supporting_turn_ids: list[str] = Field(default_factory=list)
    valid_as_of: str | None = None
    fact_versions: list[FactAssertion] = Field(default_factory=list)


class Constraints(BaseModel):
    """Evaluator constraints attached to a question."""

    model_config = ConfigDict(extra="forbid")

    must_mention: list[str] = Field(default_factory=list)
    must_not_claim: list[str] = Field(default_factory=list)
    abstention_expected: bool = False
    max_tokens: int = 500


class QuestionRecord(BaseModel):
    """A benchmark question over a chat-session corpus."""

    model_config = ConfigDict(extra="forbid")

    id: str
    # See CATEGORIES tuple for valid values.
    category: str
    hops: int = 1
    question: str
    ground_truth: GroundTruth
    constraints: Constraints = Field(default_factory=Constraints)
    type: str = ""
    tier: int = 0
    expected_chunks: list[str] = Field(default_factory=list)


CATEGORIES = (
    "information_extraction",
    "multi_session_reasoning",
    "temporal",
    "knowledge_update",
    "abstention",
)


def category_to_tier(category: str) -> int:
    """Map memory-arena categories to legacy tier numbers (1-5) for runner compatibility."""
    return {c: i for i, c in enumerate(CATEGORIES, start=1)}.get(category, 0)
