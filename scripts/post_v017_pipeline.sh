#!/bin/bash
# Post-benchmark aggregation and chart regeneration for v0.1.7.
# Runs after the full 19-strategy seed_0 sweep completes.
set -euo pipefail
cd "$(dirname "$0")/.."

echo "==> Bootstrap aggregation..."
.venv/bin/python scripts/aggregate_bootstrap.py longmemeval-s

echo "==> Re-rendering charts (hero, taxonomy, pairwise)..."
.venv/bin/python scripts/build_hero_chart.py
.venv/bin/python scripts/build_taxonomy_chart.py
.venv/bin/python scripts/build_pairwise_chart.py

echo "==> Figure sizes:"
.venv/bin/python -c "from PIL import Image; [print(p, Image.open(p).size) for p in ['docs/hero.png','docs/taxonomy.png','docs/pairwise.png']]"

echo "==> Refresh README with new numbers (manual review required after)."
.venv/bin/python scripts/render_readme.py 2>/dev/null || echo "  (no render_readme.py, do manually)"

echo "==> Cost summary across the seed-0 run:"
.venv/bin/python -c "
import json, glob
total = 0.0
for p in sorted(glob.glob('results/longmemeval-s_*_seed0.json')):
    d = json.load(open(p))
    cost = d.get('total_cost_usd', 0)
    total += cost
    print(f'  {d[\"strategy\"]:25} acc={d[\"accuracy\"]:.2%} cost=\${cost:.4f}')
print(f'  {\"TOTAL\":25}                \${total:.4f}')
"
