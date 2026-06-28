#!/bin/bash
# Memory Arena release pipeline. Run after benchmarks complete.
#
# Prerequisites:
#   - results/longmemeval-s_<strategy>_seed{N}.json files exist for each
#     strategy and seed (run `memory-arena benchmark --seed N` for N in 0,1,2)
#   - .venv set up with `pip install -e '.[dev]'` and matplotlib + playwright
#
# Outputs:
#   - results/<strategy>_summary.json (3-seed bootstrap mean ± 95% CI)
#   - docs/hero.png (Pareto frontier chart)
#   - docs/screenshot-{home,benchmark,recall-lab}.png (1440x900)
#   - dist/memory_arena-${VERSION}-py3-none-any.whl
#   - README headline table updated between BENCHMARK_TABLE_START/END markers
#   - memory_arena/data/results_snapshot/ refreshed for the wheel

set -e
cd "$(dirname "$0")/.."

# Read version from memory_arena/__init__.py — the single source of truth.
# pyproject.toml declares `dynamic = ["version"]`, so reading [project][version]
# there fails. Fall back to system python3 if .venv isn't set up yet.
if [ -x .venv/bin/python ]; then
  PY=.venv/bin/python
else
  PY=python3
fi
VERSION=$($PY -c 'from memory_arena import __version__; print(__version__)')
echo "Releasing memory-arena ${VERSION}"

echo "=== aggregate bootstrap summaries ==="
.venv/bin/python scripts/aggregate_bootstrap.py

echo ""
echo "=== build hero chart ==="
.venv/bin/python scripts/build_hero_chart.py

echo ""
echo "=== render README headline table ==="
.venv/bin/python scripts/render_readme.py

echo ""
echo "=== bundle results into wheel data ==="
rm -rf memory_arena/data/results_snapshot/*
cp results/longmemeval-s_*.json memory_arena/data/results_snapshot/ 2>/dev/null || true
echo "snapshot files: $(ls memory_arena/data/results_snapshot/ | wc -l)"

echo ""
echo "=== capture dashboard screenshots ==="
pkill -f 'uvicorn memory' 2>/dev/null || true
sleep 1
.venv/bin/python scripts/capture_screenshots.py

echo ""
echo "=== build wheel ==="
rm -rf dist
.venv/bin/python -m build --wheel
ls -lh dist/

echo ""
echo "=== verify wheel installs cleanly + memory-arena report works ==="
rm -rf /tmp/verify-final
.venv/bin/python -m venv /tmp/verify-final
/tmp/verify-final/bin/pip install "dist/memory_arena-${VERSION}-py3-none-any.whl" 2>&1 | tail -3
( cd /tmp && /tmp/verify-final/bin/memory-arena report --corpus longmemeval-s | head -25 )

echo ""
echo "=== final lint + tests ==="
.venv/bin/python -m ruff check .
.venv/bin/python -m ruff format --check .
.venv/bin/python -m pytest tests/ -q --ignore=tests/live | tail -3

echo ""
echo "=== READY TO TAG ==="
echo "Next steps:"
echo "  git add -A && git reset HEAD .env.example && git commit -m 'v${VERSION} final numbers + screenshots + hero chart'"
echo "  git tag v${VERSION}"
echo "  Then per LAUNCH_PENDING_HUMAN_ACTIONS.md: create the GitHub repo,"
echo "  open the PR, merge, and push the tag to fire publish.yml."
