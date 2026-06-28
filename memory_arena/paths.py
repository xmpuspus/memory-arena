"""Path resolution: bundled-data-first, then local CWD fallback.

When `pip install memory-arena` lands the wheel in site-packages, there is no
`./datasets/` or `./results/` next to the user. The demo command needs to find
the bundled smoke corpus and a result snapshot anyway.

This module returns the bundled location if it exists, else falls back to the
working directory. Local dev (running from a checkout with `./datasets/`)
still reads the live tree because that path takes precedence — actually no,
bundled wins by default to avoid the case where someone runs `memory-arena
demo` from a random directory and gets nothing.

Override behavior with the env vars:
    MEM_ARENA_DATASETS_PATH=./datasets   # use local first
    MEM_ARENA_RESULTS_PATH=./results
"""

from __future__ import annotations

import os
from pathlib import Path

_PACKAGE_ROOT = Path(__file__).resolve().parent
_BUNDLED_DATA = _PACKAGE_ROOT / "data"


def _has_real_corpus(path: Path) -> bool:
    """True if `path` contains an actual corpus dir (not just .gitkeep)."""
    if not path.is_dir():
        return False
    for child in path.iterdir():
        if child.is_dir() and not child.name.startswith("."):
            return True
    return False


def datasets_root() -> Path:
    """Where corpus data lives. Local working directory wins, else bundled."""
    override = os.environ.get("MEM_ARENA_DATASETS_PATH")
    if override:
        return Path(override).resolve()
    # Local checkout takes precedence over bundled — Xavier iterates locally
    # and would be surprised if the wheel-bundled snapshot shadowed his edits.
    local = Path("datasets")
    if _has_real_corpus(local):
        return local.resolve()
    if _has_real_corpus(_BUNDLED_DATA):
        return _BUNDLED_DATA
    # Neither exists; return local so callers get a clean missing-file error.
    return local.resolve()


def results_root() -> Path:
    """Where benchmark result JSONs live. Local wins, else bundled snapshot."""
    override = os.environ.get("MEM_ARENA_RESULTS_PATH")
    if override:
        return Path(override).resolve()
    local_results = Path("results")
    if local_results.is_dir() and any(local_results.glob("longmemeval-s_*.json")):
        return local_results.resolve()
    bundled_results = _BUNDLED_DATA / "results_snapshot"
    if bundled_results.is_dir() and any(bundled_results.glob("longmemeval-s_*.json")):
        return bundled_results
    return local_results.resolve()


def session_jsonl(corpus: str) -> Path:
    return datasets_root() / corpus / "processed" / "sessions.jsonl"


def question_jsonl(corpus: str) -> Path:
    return datasets_root() / corpus / "processed" / "questions.jsonl"


def smoke_questions_dir(corpus: str) -> Path:
    return datasets_root() / corpus / "questions" / "smoke"


def raw_corpus_path(corpus: str) -> Path:
    return datasets_root() / corpus / "raw" / f"{corpus.replace('-', '_')}.json"


__all__ = [
    "datasets_root",
    "question_jsonl",
    "raw_corpus_path",
    "results_root",
    "session_jsonl",
    "smoke_questions_dir",
]
