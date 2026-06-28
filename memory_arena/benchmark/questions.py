"""Question loading for memory-arena.

Sources, in order of preference:
    1. datasets/<corpus>/processed/questions.jsonl       (after `ingest-sessions`)
    2. datasets/<corpus>/questions/smoke_synthetic/*.yaml (the canonical smoke subset)
    3. datasets/<corpus>/questions/smoke/*.yaml          (legacy smoke layout)
    4. datasets/<corpus>/questions/*.yaml                (full set, when present)
"""

from __future__ import annotations

from pathlib import Path

import yaml

from memory_arena.sessions.loaders import load_questions_jsonl
from memory_arena.sessions.schema import (
    CATEGORIES,
    Constraints,
    GroundTruth,
    QuestionRecord,
    category_to_tier,
)


def discover_corpora() -> list[str]:
    """Return corpus names that have at least one question file."""
    from memory_arena.paths import datasets_root

    base = datasets_root()
    if not base.exists():
        return []
    out: list[str] = []
    for d in sorted(base.iterdir()):
        if not d.is_dir() or d.name.startswith("."):
            continue
        if (d / "processed" / "questions.jsonl").exists():
            out.append(d.name)
            continue
        q_dir = d / "questions"
        if q_dir.is_dir() and any(q_dir.rglob("*.yaml")):
            out.append(d.name)
    return out


def _yaml_to_record(item: dict) -> QuestionRecord:
    gt_raw = item.get("ground_truth", {}) or {}
    gt = GroundTruth(
        answer=gt_raw.get("answer", ""),
        supporting_session_ids=list(gt_raw.get("supporting_session_ids", []) or []),
        supporting_turn_ids=list(gt_raw.get("supporting_turn_ids", []) or []),
        valid_as_of=gt_raw.get("valid_as_of"),
    )
    constraints_raw = item.get("constraints", {}) or {}
    constraints = Constraints(
        must_mention=list(constraints_raw.get("must_mention", []) or []),
        must_not_claim=list(constraints_raw.get("must_not_claim", []) or []),
        abstention_expected=bool(constraints_raw.get("abstention_expected", False)),
        max_tokens=int(constraints_raw.get("max_tokens", 500)),
    )
    category = item.get("category", "information_extraction")
    return QuestionRecord(
        id=item["id"],
        category=category,
        hops=int(item.get("hops", 1)),
        question=item["question"],
        ground_truth=gt,
        constraints=constraints,
        type=category,
        tier=category_to_tier(category),
    )


def load_memory_questions(
    corpus: str = "longmemeval-s",
    subset: str = "smoke",
) -> list[QuestionRecord]:
    """Load questions from JSONL or YAML."""
    from memory_arena.paths import datasets_root

    base = datasets_root() / corpus

    # Custom path support: subset can be a directory or file.
    custom = Path(subset)
    if custom.exists():
        if custom.is_file() and custom.suffix in (".yaml", ".yml"):
            return _load_yaml_file(custom)
        if custom.is_dir():
            return _load_yaml_dir(custom)

    if subset == "full":
        jsonl = base / "processed" / "questions.jsonl"
        if jsonl.exists():
            return load_questions_jsonl(corpus)
        # fall through to YAML
        yaml_dir = base / "questions"
        if yaml_dir.exists():
            return _load_yaml_dir(yaml_dir, recursive=False)
        return []

    # smoke (default). On-disk layout uses smoke_synthetic/ (the bundled
    # LongMemEval-S subset of synthetic abstention questions); older
    # corpora may still use smoke/. Try the canonical path first, fall
    # back to legacy.
    smoke_dir = base / "questions" / "smoke_synthetic"
    if smoke_dir.exists():
        return _load_yaml_dir(smoke_dir)
    smoke_dir = base / "questions" / "smoke"
    if smoke_dir.exists():
        return _load_yaml_dir(smoke_dir)

    # If only JSONL exists, take first 30 questions
    jsonl_records = load_questions_jsonl(corpus)
    if jsonl_records:
        return _balanced_subset(jsonl_records, per_category=6)
    return []


def _balanced_subset(records: list[QuestionRecord], per_category: int = 6) -> list[QuestionRecord]:
    by_cat: dict[str, list[QuestionRecord]] = {c: [] for c in CATEGORIES}
    for r in records:
        if r.category in by_cat:
            by_cat[r.category].append(r)
    out: list[QuestionRecord] = []
    for cat in CATEGORIES:
        out.extend(by_cat[cat][:per_category])
    return out


def _load_yaml_file(path: Path) -> list[QuestionRecord]:
    with path.open() as f:
        data = yaml.safe_load(f)
    if not data:
        return []
    if isinstance(data, dict) and "questions" in data:
        items = data["questions"]
    else:
        items = data
    return [_yaml_to_record(item) for item in items]


def _load_yaml_dir(path: Path, recursive: bool = False) -> list[QuestionRecord]:
    pattern = "**/*.yaml" if recursive else "*.yaml"
    out: list[QuestionRecord] = []
    for p in sorted(path.glob(pattern)):
        out.extend(_load_yaml_file(p))
    return out


__all__ = ["discover_corpora", "load_memory_questions"]
