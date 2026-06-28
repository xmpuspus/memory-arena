"""LongMemEval and JSONL corpus loaders.

LongMemEval upstream repo: https://github.com/xiaowu0162/LongMemEval (MIT, ICLR 2025).
Source dataset: data/longmemeval_s.json — 500 questions across 5 categories with
multi-session chat history.
"""

from __future__ import annotations

import json
from pathlib import Path

from memory_arena.sessions.schema import (
    CATEGORIES,
    Constraints,
    FactAssertion,
    GroundTruth,
    QuestionRecord,
    Session,
    Turn,
    category_to_tier,
)

_LONGMEMEVAL_CATEGORY_MAP = {
    "single-session-user": "information_extraction",
    "single-session-assistant": "information_extraction",
    "single-session-preference": "information_extraction",
    "multi-session": "multi_session_reasoning",
    "temporal-reasoning": "temporal",
    "knowledge-update": "knowledge_update",
    "abstention": "abstention",
}


class LongMemEvalLoader:
    """Loader for LongMemEval_S JSON dumps.

    The upstream JSON layout is a list of question objects where each question
    carries the haystack of sessions in `haystack_sessions`. We normalize to
    Session and Turn rows plus a question manifest pointing back at the sessions
    that should support each answer.
    """

    def __init__(self, raw_path: str | Path):
        self.raw_path = Path(raw_path)

    def load_raw(self) -> list[dict]:
        if not self.raw_path.exists():
            raise FileNotFoundError(
                f"LongMemEval raw file not found: {self.raw_path}. "
                "Download from https://github.com/xiaowu0162/LongMemEval (data/longmemeval_s.json)"
            )
        with self.raw_path.open() as f:
            return json.load(f)

    def normalize(
        self,
    ) -> tuple[list[Session], list[QuestionRecord]]:
        """Return (sessions, questions). Sessions are deduped across questions."""
        records = self.load_raw()
        sessions: dict[str, Session] = {}
        questions: list[QuestionRecord] = []

        for q_idx, item in enumerate(records):
            qid = item.get("question_id") or f"longmemeval-{q_idx:04d}"
            raw_category = item.get("question_type", "single-session-user")
            category = _LONGMEMEVAL_CATEGORY_MAP.get(raw_category, "information_extraction")

            haystack_session_ids = item.get("haystack_session_ids") or []
            haystack_sessions = item.get("haystack_sessions") or []
            haystack_dates = item.get("haystack_dates") or [None] * len(haystack_sessions)
            answer_session_ids = item.get("answer_session_ids") or []

            for sess_id, turns_payload, ts in zip(
                haystack_session_ids,
                haystack_sessions,
                haystack_dates,
                strict=False,
            ):
                if sess_id in sessions:
                    continue
                turns: list[Turn] = []
                for t_idx, msg in enumerate(turns_payload or []):
                    turns.append(
                        Turn(
                            id=f"{sess_id}_turn_{t_idx:03d}",
                            session_id=sess_id,
                            role=msg.get("role", "user"),
                            content=msg.get("content", ""),
                            timestamp=msg.get("timestamp") or ts,
                            metadata={"has_answer": msg.get("has_answer", False)},
                        )
                    )
                sessions[sess_id] = Session(
                    id=sess_id,
                    user_id=item.get("user_id", "default"),
                    timestamp=ts,
                    turns=turns,
                    metadata={"category": category},
                )

            answer = item.get("answer", "")
            # Upstream LongMemEval sometimes has int/float answers; coerce to string.
            if not isinstance(answer, str):
                answer = "" if answer is None else str(answer)
            ground_truth = GroundTruth(
                answer=answer,
                supporting_session_ids=list(answer_session_ids),
                supporting_turn_ids=[],
                valid_as_of=item.get("question_date"),
                fact_versions=self._extract_fact_versions(item),
            )
            constraints = Constraints(
                must_mention=item.get("must_mention", []) or [],
                must_not_claim=item.get("must_not_claim", []) or [],
                abstention_expected=(category == "abstention"),
                max_tokens=item.get("max_tokens", 500),
            )
            questions.append(
                QuestionRecord(
                    id=qid,
                    category=category,
                    hops=1 if category == "information_extraction" else 2,
                    question=item.get("question", ""),
                    ground_truth=ground_truth,
                    constraints=constraints,
                    type=category,
                    tier=category_to_tier(category),
                )
            )

        return list(sessions.values()), questions

    @staticmethod
    def _extract_fact_versions(item: dict) -> list[FactAssertion]:
        versions = item.get("fact_versions") or []
        out: list[FactAssertion] = []
        for v in versions:
            out.append(
                FactAssertion(
                    value=v.get("value", ""),
                    valid_at=v.get("valid_at"),
                    invalid_at=v.get("invalid_at"),
                    source_session_id=v.get("source_session_id", ""),
                    source_turn_id=v.get("source_turn_id"),
                    confidence=float(v.get("confidence", 1.0)),
                )
            )
        return out

    def write_processed(
        self,
        sessions: list[Session],
        questions: list[QuestionRecord],
        out_dir: str | Path,
    ) -> tuple[Path, Path]:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)

        sessions_path = out / "sessions.jsonl"
        with sessions_path.open("w") as f:
            for s in sessions:
                f.write(s.model_dump_json() + "\n")

        questions_path = out / "questions.jsonl"
        with questions_path.open("w") as f:
            for q in questions:
                f.write(q.model_dump_json() + "\n")
        return sessions_path, questions_path


def load_sessions(corpus: str = "longmemeval-s") -> list[Session]:
    """Load normalized sessions for a corpus."""
    from memory_arena.paths import session_jsonl

    path = session_jsonl(corpus)
    if not path.exists():
        return []
    sessions: list[Session] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            sessions.append(Session.model_validate_json(line))
    return sessions


def load_questions_jsonl(corpus: str = "longmemeval-s") -> list[QuestionRecord]:
    from memory_arena.paths import question_jsonl

    path = question_jsonl(corpus)
    if not path.exists():
        return []
    out: list[QuestionRecord] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(QuestionRecord.model_validate_json(line))
    return out


__all__ = [
    "CATEGORIES",
    "LongMemEvalLoader",
    "load_questions_jsonl",
    "load_sessions",
]
