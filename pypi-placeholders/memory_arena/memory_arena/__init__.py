"""Placeholder package — did you mean memory-arena (with hyphen)?"""

import sys

_MSG = (
    "WARNING: 'memoryarena' is a placeholder. The real package is 'memory-arena'.\n"
    "  pip install memory-arena\n"
    "  https://github.com/xmpuspus/memory-arena\n"
)

print(_MSG, file=sys.stderr)
