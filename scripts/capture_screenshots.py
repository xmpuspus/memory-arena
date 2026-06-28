"""Capture dashboard screenshots for the README.

Starts a memory-arena server in the background, navigates to each page, and
saves a 1440x900 PNG to docs/. Idempotent — kills the server when done.

Usage:
    python scripts/capture_screenshots.py
"""

from __future__ import annotations

import socket
import subprocess
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

REPO_ROOT = Path(__file__).resolve().parents[1]
DOCS = REPO_ROOT / "docs"

PAGES = [
    ("/", "screenshot-home.png"),
    ("/benchmark/", "screenshot-benchmark.png"),
    ("/recall-lab/", "screenshot-recall-lab.png"),
]


def _free_port(start: int = 8090) -> int:
    for p in range(start, start + 50):
        with socket.socket() as s:
            if s.connect_ex(("127.0.0.1", p)) != 0:
                return p
    raise RuntimeError("no free port")


def _wait_until_up(url: str, timeout_s: float = 30.0) -> None:
    import urllib.request

    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            urllib.request.urlopen(url, timeout=2.0)
            return
        except Exception:
            time.sleep(0.5)
    raise RuntimeError(f"server did not come up at {url}")


def main() -> None:
    port = _free_port()
    server = subprocess.Popen(
        [
            str(REPO_ROOT / ".venv/bin/uvicorn"),
            "memory_arena.chatbot.api:app",
            "--port",
            str(port),
            "--host",
            "127.0.0.1",
        ],
        cwd=str(REPO_ROOT),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        base = f"http://127.0.0.1:{port}"
        _wait_until_up(f"{base}/api/health")
        with sync_playwright() as p:
            browser = p.chromium.launch()
            ctx = browser.new_context(
                viewport={"width": 1440, "height": 900},
                device_scale_factor=2,
            )
            for path, fname in PAGES:
                page = ctx.new_page()
                page.goto(f"{base}{path}", wait_until="networkidle")
                # Give the data fetch + chart render a beat to settle.
                page.wait_for_timeout(1500)
                page.screenshot(path=str(DOCS / fname), full_page=True)
                print(f"wrote {DOCS / fname}")
                page.close()
            browser.close()
    finally:
        server.terminate()
        server.wait(timeout=5)


if __name__ == "__main__":
    main()
