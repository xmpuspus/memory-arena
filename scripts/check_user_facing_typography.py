"""Lint user-facing markdown for AI-fingerprint typography + count drift.

CI rule: README + docs/*.md + CONTRIBUTING.md must not contain:
  - em-dashes (U+2014). Use ", " or ": " or "." instead.
  - en-dashes (U+2013) in narrative text. Use "-" for number ranges too.
  - the literal phrase "16 agent-memory architectures" (current count is 19).

The single file `docs/per-question-comparison.md` is exempt because it
quotes LLM output verbatim; the LLM emits em-dashes and rewriting them
would falsify the transcript.

Run:
    python scripts/check_user_facing_typography.py
Exit code 0 = clean. Exit code 1 = at least one violation; offending
lines are printed to stderr.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

USER_FACING = [
    "README.md",
    "CONTRIBUTING.md",
    "docs/FAQ.md",
    "docs/case-studies.md",
    "docs/decision-guide.md",
    "docs/vendor-pins.md",
]

# Files that intentionally contain verbatim LLM output (em-dashes are
# part of the model's response, not editorial copy).
LLM_TRANSCRIPT_EXEMPT = {"docs/per-question-comparison.md"}

BANNED_TYPOGRAPHY = {
    "—": "em-dash (-- U+2014); use ', ' or ': ' or '.'",
    "–": "en-dash (-- U+2013); use '-'",
}

BANNED_PHRASES = [
    (
        "16 agent-memory architectures",
        "current count is 19; update to '19 agent-memory architectures'",
    ),
    (
        "16 strategies",
        "current count is 19 (where the claim applies; check carefully)",
    ),
]


def main() -> int:
    violations: list[str] = []
    for rel in USER_FACING:
        path = REPO_ROOT / rel
        if not path.exists():
            print(f"WARN: {rel} not found, skipping", file=sys.stderr)
            continue
        text = path.read_text()
        for i, line in enumerate(text.splitlines(), start=1):
            for ch, reason in BANNED_TYPOGRAPHY.items():
                if ch in line:
                    snippet = line.strip()[:140]
                    violations.append(f"{rel}:{i}: {reason}  >>>  {snippet}")
            for phrase, reason in BANNED_PHRASES:
                if phrase in line:
                    snippet = line.strip()[:140]
                    violations.append(f"{rel}:{i}: phrase '{phrase}' -- {reason}  >>>  {snippet}")

    if violations:
        print("Typography / count lint failures:", file=sys.stderr)
        for v in violations:
            print(f"  {v}", file=sys.stderr)
        print(
            f"\nExempt (verbatim LLM output, em-dashes allowed): {sorted(LLM_TRANSCRIPT_EXEMPT)}",
            file=sys.stderr,
        )
        return 1
    print("Typography + count lint: OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
