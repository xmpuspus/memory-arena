#!/usr/bin/env bash
# Re-capture screenshots and re-record the demo GIF after a fresh benchmark run.
# Assumes:
#   - results/longmemeval-s_*.json populated for all 16 strategies
#   - virtualenv at .venv/ activated
#   - Docker neo4j + postgres running

set -euo pipefail

cd "$(dirname "$0")/.."

# Kill any existing serve and free its port.
PID=$(lsof -ti :8765 2>/dev/null || true)
if [ -n "$PID" ]; then
  kill "$PID" 2>/dev/null || true
fi

# Restart serve with the latest results.
memory-arena serve --port 8765 > /tmp/memory-arena-serve.log 2>&1 &
disown

# Wait for /api/health.
until curl -s http://localhost:8765/api/health > /dev/null 2>&1; do sleep 1; done
echo "[serve] up"

agent-browser open "http://localhost:8765/" >/dev/null
agent-browser wait 2000 >/dev/null
agent-browser screenshot --full docs/screenshot-home.png >/dev/null
echo "[home] captured"

agent-browser open "http://localhost:8765/benchmark/" >/dev/null
agent-browser wait 3000 >/dev/null
agent-browser screenshot --full docs/screenshot-benchmark.png >/dev/null
echo "[benchmark] captured"

agent-browser open "http://localhost:8765/recall-lab/" >/dev/null
agent-browser wait 2000 >/dev/null
agent-browser screenshot --full docs/screenshot-recall-lab.png >/dev/null
echo "[recall-lab] captured"

vhs docs/demo.tape >/dev/null
echo "[demo gif] recorded"
