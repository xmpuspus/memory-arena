"""7-axis evaluator for memory-arena.

Axes (4 lifted from kb-arena + 3 memory-specific):
  1. structural — must_mention / must_not_claim / max_tokens
  2. sources    — was at least one supporting_session_id cited?
  3. judge      — Opus rates accuracy 0..100 against the reference
  4. memo       — identical (answer, reference) pairs cached in-process
  5. temporal_correctness — answer's claimed timestamp is inside ground_truth.valid_as_of window
  6. update_precision     — answer reflects the latest fact version, not an earlier one
  7. abstention_f1        — abstention questions get F1 over an abstain/no-abstain classifier
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from pydantic import BaseModel, ConfigDict, Field

from memory_arena.llm.client import LLMClient
from memory_arena.sessions.schema import Constraints, GroundTruth, QuestionRecord

_JUDGE_SYSTEM = (
    "You are a strict judge comparing a candidate answer to a reference answer for a question "
    "about a chat history. Output a JSON object with this exact shape:\n"
    '{"accuracy": <0-100>, "completeness": <0-100>, "rationale": "<one sentence>"}\n'
    "Be strict: an answer that misses key facts should score below 60. An answer that "
    "contradicts the reference should score below 30. Reply with only the JSON object."
)


_ABSTAIN_SYSTEM = (
    "Classify whether the following answer abstains from answering (says it does not know, "
    "cannot find the information, or refuses to answer). Reply with only YES or NO."
)


_TEMPORAL_SYSTEM = (
    "Extract any explicit time marker mentioned in the answer (date, week-of, month, year, "
    "or relative phrase like 'last week'). Reply with only the time marker, or NONE."
)


class MemoryScore(BaseModel):
    """7-axis evaluation score."""

    model_config = ConfigDict(extra="forbid")

    accuracy: float = 0.0
    completeness: float = 0.0
    structural_pass: bool = True
    structural_fails: list[str] = Field(default_factory=list)
    sources_pass: bool = False
    judge_score: float = 0.0
    judge_rationale: str = ""
    temporal_correct: bool = False
    # update_precision_correct can be None when the question does not
    # carry fact_versions to check — runner aggregates only non-None.
    update_precision_correct: bool | None = False
    abstained: bool = False
    abstention_correct: bool = False
    cost_usd: float = 0.0
    tokens_used: int = 0


# Module-level memo cache. Keyed by (question_id, answer-hash, reference-hash)
# so long answers that diverge after the first 500 characters do not collide.
_judge_cache: dict[tuple[str, str, str], dict] = {}


def _evaluate_structural(
    answer: str,
    constraints: Constraints,
) -> tuple[bool, list[str]]:
    fails: list[str] = []
    answer_lower = answer.lower()
    for required in constraints.must_mention:
        if required.lower() not in answer_lower:
            fails.append(f"missing must_mention: {required}")
    for forbidden in constraints.must_not_claim:
        if forbidden.lower() in answer_lower:
            fails.append(f"contains must_not_claim: {forbidden}")
    if constraints.max_tokens > 0:
        approx_tokens = max(1, len(answer) // 4)
        if approx_tokens > constraints.max_tokens * 1.5:
            fails.append(f"exceeds max_tokens by 50%: {approx_tokens} > {constraints.max_tokens}")
    return (len(fails) == 0), fails


def _evaluate_sources(
    answer: str,
    supporting_session_ids: list[str],
    ground_truth: GroundTruth,
) -> bool:
    expected = set(ground_truth.supporting_session_ids)
    if not expected:
        return True  # no labeled sources -> can't fail
    return bool(set(supporting_session_ids) & expected)


async def _evaluate_judge(
    answer: str,
    reference: str,
    llm: LLMClient,
    question_id: str = "",
) -> tuple[float, float, str, float, int]:
    import hashlib

    a_hash = hashlib.sha256(answer.strip().encode("utf-8")).hexdigest()
    r_hash = hashlib.sha256(reference.strip().encode("utf-8")).hexdigest()
    key = (question_id, a_hash, r_hash)
    if key in _judge_cache:
        cached = _judge_cache[key]
        return (
            cached["accuracy"],
            cached["completeness"],
            cached["rationale"],
            0.0,
            0,
        )
    if not reference:
        return 0.0, 0.0, "no reference provided", 0.0, 0
    resp = await llm.judge(answer=answer, reference=reference, system_prompt=_JUDGE_SYSTEM)
    text = resp.text.strip()
    accuracy = 0.0
    completeness = 0.0
    rationale = ""
    try:
        import json

        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            data = json.loads(match.group(0))
            accuracy = float(data.get("accuracy", 0))
            completeness = float(data.get("completeness", 0))
            rationale = data.get("rationale", "")
    except Exception:
        rationale = f"judge parse failure: {text[:80]}"
    _judge_cache[key] = {
        "accuracy": accuracy,
        "completeness": completeness,
        "rationale": rationale,
    }
    return accuracy, completeness, rationale, resp.cost_usd, resp.total_tokens


async def _classify_abstention(answer: str, llm: LLMClient) -> tuple[bool, float, int]:
    if _quick_abstain_match(answer):
        return True, 0.0, 0
    resp = await llm.classify(
        query=answer[:1000], system_prompt=_ABSTAIN_SYSTEM, allowed_values=["YES", "NO"]
    )
    return resp.upper().startswith("Y"), 0.0, 0


def _quick_abstain_match(answer: str) -> bool:
    a = answer.lower()
    needles = (
        "i do not have that information",
        "i don't have that information",
        "i don't know",
        "i do not know",
        "i'm not sure",
        "i am not sure",
        "no information",
        "cannot answer",
        "unable to answer",
    )
    return any(n in a for n in needles)


async def _check_temporal(
    answer: str,
    valid_as_of: str | None,
    llm: LLMClient,
) -> bool:
    if not valid_as_of:
        return True
    resp = await llm.classify(query=answer[:1000], system_prompt=_TEMPORAL_SYSTEM)
    if resp.upper() == "NONE" or not resp:
        return False
    return _temporal_overlap(resp, valid_as_of)


def _temporal_overlap(claimed: str, valid_as_of: str) -> bool:
    """Naive substring overlap on year/month/day numbers. Tighten in v0.2."""
    claimed_nums = set(re.findall(r"\d+", claimed))
    expected_nums = set(re.findall(r"\d+", valid_as_of))
    return bool(claimed_nums & expected_nums) if expected_nums else False


async def _check_update_precision(
    answer: str,
    ground_truth: GroundTruth,
) -> bool | None:
    """Answer must reflect the latest fact version, not an earlier one.

    Returns None when fact_versions is empty/missing — there's nothing to
    check, and reporting True would inflate the per-category metric to
    1.0 across every strategy regardless of whether the question
    actually exercised an update. The runner aggregates only non-None
    values.
    """
    if not ground_truth.fact_versions:
        return None
    a = answer.lower()
    latest = ground_truth.fact_versions[-1].value.lower()
    if latest and latest in a:
        return True
    earlier = [v.value.lower() for v in ground_truth.fact_versions[:-1]]
    if any(v and v in a for v in earlier):
        return False
    return False


async def evaluate_memory_answer(
    answer: str,
    ground_truth: GroundTruth,
    constraints: Constraints,
    question: QuestionRecord,
    llm: LLMClient,
    supporting_session_ids: list[str] | None = None,
) -> MemoryScore:
    """Run all 7 axes and return a MemoryScore."""
    score = MemoryScore()
    if supporting_session_ids is None:
        supporting_session_ids = []

    # 1. structural
    structural_pass, fails = _evaluate_structural(answer, constraints)
    score.structural_pass = structural_pass
    score.structural_fails = fails

    # 2. sources
    score.sources_pass = _evaluate_sources(answer, supporting_session_ids, ground_truth)

    # 3. judge
    accuracy, completeness, rationale, cost, toks = await _evaluate_judge(
        answer, ground_truth.answer, llm, question_id=question.id
    )
    score.judge_score = accuracy
    score.judge_rationale = rationale
    score.completeness = completeness / 100.0 if completeness > 1 else completeness
    score.cost_usd += cost
    score.tokens_used += toks

    # 5. temporal
    if question.category == "temporal":
        score.temporal_correct = await _check_temporal(answer, ground_truth.valid_as_of, llm)
    else:
        score.temporal_correct = True

    # 6. update precision. _check_update_precision returns None when the
    # question has no fact_versions to verify; preserve that so the
    # runner can aggregate only the questions that actually exercised an
    # update instead of polluting the metric with structural-trues.
    if question.category == "knowledge_update":
        score.update_precision_correct = await _check_update_precision(answer, ground_truth)
    else:
        score.update_precision_correct = True

    # 7. abstention
    abstained, _, _ = await _classify_abstention(answer, llm)
    score.abstained = abstained
    score.abstention_correct = abstained == constraints.abstention_expected

    # Composite accuracy: judge score normalized to 0-1, dampened by structural and sources
    base = accuracy / 100.0 if accuracy > 1 else accuracy
    if not structural_pass:
        base *= 0.5
    if not score.sources_pass and ground_truth.supporting_session_ids:
        base *= 0.8
    if question.category == "abstention":
        base = 1.0 if score.abstention_correct else 0.0
    if question.category == "temporal" and not score.temporal_correct:
        base *= 0.5
    if question.category == "knowledge_update" and score.update_precision_correct is False:
        base *= 0.5
    score.accuracy = max(0.0, min(1.0, base))

    return score


def clear_eval_cache() -> None:
    _judge_cache.clear()


@dataclass
class EvaluatorBundle:
    """Convenience wrapper for callers that batch many evaluations."""

    llm: LLMClient = field(default_factory=LLMClient)

    async def evaluate(
        self,
        answer: str,
        question: QuestionRecord,
        supporting_session_ids: list[str] | None = None,
    ) -> MemoryScore:
        return await evaluate_memory_answer(
            answer=answer,
            ground_truth=question.ground_truth,
            constraints=question.constraints,
            question=question,
            llm=self.llm,
            supporting_session_ids=supporting_session_ids,
        )


__all__ = [
    "EvaluatorBundle",
    "MemoryScore",
    "clear_eval_cache",
    "evaluate_memory_answer",
]
