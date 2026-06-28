"""Record the dashboard tour as a README-ready GIF + a LinkedIn-ready MP4.

Drives the bundled dashboard via Playwright (headless, viewport-only —
no browser chrome per /record-demo skill rule 5), reads the step list
from `demo-flow.yaml`, and uses the skill's mandated smooth-scroll
pattern (per-frame `scrollBy` at 15px / 30ms) so the recording stays
fluid on every screen size.

Usage:
    python scripts/record_dashboard_tour.py
    python scripts/record_dashboard_tour.py --port 8765 --max-size 8

Outputs:
    docs/dashboard-tour.gif   README-embeddable, target <10 MB
    docs/dashboard-tour.mp4   LinkedIn-ready, 1280x720, ~50 s

Requires: playwright + chromium (in dev extras), ffmpeg on PATH.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import re
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DOCS = REPO_ROOT / "docs"
TMP = REPO_ROOT / "tmp"
FLOW_FILE = REPO_ROOT / "demo-flow.yaml"
TOUR_MP4 = DOCS / "dashboard-tour.mp4"
TOUR_GIF = DOCS / "dashboard-tour.gif"


def _parse_duration(s: str | int | float) -> float:
    """'8s' -> 8.0, 8 -> 8.0."""
    if isinstance(s, int | float):
        return float(s)
    m = re.match(r"^\s*(\d+(?:\.\d+)?)\s*s?\s*$", str(s))
    if not m:
        raise ValueError(f"bad duration: {s!r}")
    return float(m.group(1))


def _load_flow() -> dict:
    import yaml  # pyyaml is in core deps

    return yaml.safe_load(FLOW_FILE.read_text())


def _wait_for_port(port: int, timeout: float = 30.0) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=1):
                return
        except OSError:
            time.sleep(0.3)
    raise RuntimeError(f"server on :{port} never came up within {timeout}s")


async def _smooth_scroll(page, distance: int, step: int = 15, delay_ms: int = 30) -> None:
    """Per-frame scrollBy — fluid in the recording on every renderer.

    The browser's CSS smooth-scroll is renderer-dependent; per-frame JS
    eval is deterministic. /record-demo skill rule 6.
    """
    n_steps = max(1, distance // step)
    for _ in range(n_steps):
        await page.evaluate(f"window.scrollBy(0, {step})")
        await asyncio.sleep(delay_ms / 1000)


async def _drive(flow: dict) -> Path:
    """Run the flow steps in a recorded headless browser."""
    from playwright.async_api import async_playwright

    TMP.mkdir(parents=True, exist_ok=True)
    raw_dir = TMP / "tour-raw"
    if raw_dir.exists():
        shutil.rmtree(raw_dir)
    raw_dir.mkdir(parents=True)

    viewport = flow.get("viewport", {"width": 1280, "height": 720})
    base_url = flow["base_url"]
    steps = flow["steps"]

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context(
            viewport=viewport,
            record_video_dir=str(raw_dir),
            record_video_size=viewport,
        )
        page = await context.new_page()

        for step in steps:
            url = f"{base_url}{step['page']}"
            wait_until = step.get("wait", "networkidle")
            if wait_until in {"networkidle", "load", "domcontentloaded"}:
                await page.goto(url, wait_until=wait_until)
            else:
                await page.goto(url)
                await asyncio.sleep(_parse_duration(wait_until))

            duration = _parse_duration(step.get("duration", "5s"))
            scroll = step.get("scroll")

            if scroll:
                # Pause briefly so the viewer registers the page, then scroll.
                await asyncio.sleep(min(2.0, duration * 0.25))
                distance = int(scroll.get("distance", 300))
                # speed: smooth (30ms/step) | fast (10ms/step)
                delay = 10 if scroll.get("speed") == "fast" else 30
                await _smooth_scroll(page, distance, delay_ms=delay)
                # Linger after scroll so the viewer can read the bottom content.
                remaining = max(0.0, duration - 2.0 - (distance // 15) * (delay / 1000))
                await asyncio.sleep(remaining)
            else:
                await asyncio.sleep(duration)

        await context.close()
        await browser.close()

    webms = sorted(raw_dir.glob("*.webm"))
    if not webms:
        raise RuntimeError("playwright produced no webm")
    return webms[-1]


def _transcode_mp4(webm: Path) -> None:
    DOCS.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(webm),
            "-c:v",
            "libx264",
            "-preset",
            "slow",
            "-crf",
            "22",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            "-vf",
            "scale=1280:-2:flags=lanczos",
            str(TOUR_MP4),
        ],
        check=True,
        stderr=subprocess.DEVNULL,
    )


def _make_gif(webm: Path, max_mb: float) -> None:
    """Convert webm -> palette-optimized GIF, iterating until under max_mb.

    Two knobs to shrink: width (960 -> 720 -> 600) and fps (12 -> 10 -> 8).
    """
    palette = TMP / "palette.png"
    candidates = [(960, 12), (960, 10), (720, 10), (720, 8), (600, 8)]
    last_size = -1
    for width, fps in candidates:
        # Build palette
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-i",
                str(webm),
                "-vf",
                f"fps={fps},scale={width}:-2:flags=lanczos,palettegen=stats_mode=diff",
                str(palette),
            ],
            check=True,
            stderr=subprocess.DEVNULL,
        )
        # Apply palette
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-i",
                str(webm),
                "-i",
                str(palette),
                "-lavfi",
                f"fps={fps},scale={width}:-2:flags=lanczos[x];"
                "[x][1:v]paletteuse=dither=bayer:bayer_scale=4",
                str(TOUR_GIF),
            ],
            check=True,
            stderr=subprocess.DEVNULL,
        )
        size_mb = TOUR_GIF.stat().st_size / (1024 * 1024)
        last_size = size_mb
        print(f"  [gif] {width}x?  {fps}fps -> {size_mb:.1f} MB")
        if size_mb <= max_mb:
            return
    raise RuntimeError(f"could not get GIF under {max_mb} MB (last: {last_size:.1f} MB)")


def _verify(webm: Path) -> None:
    """Ground-truth checks: ffprobe nb_frames > 1, GIF + MP4 both readable.

    Per skill rule 9: never trust PIL frame counts; ffprobe is authoritative.
    """
    for path in (TOUR_MP4, TOUR_GIF):
        out = subprocess.check_output(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=nb_frames,width,height,duration",
                "-of",
                "default=noprint_wrappers=1",
                str(path),
            ],
            text=True,
        )
        print(f"  [{path.suffix[1:]}] {path.name}")
        for line in out.strip().splitlines():
            print(f"        {line}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=None, help="override demo-flow.yaml port")
    ap.add_argument("--max-size", type=float, default=None, help="max GIF size in MB")
    args = ap.parse_args()

    if not shutil.which("ffmpeg"):
        raise SystemExit("ffmpeg not on PATH (brew install ffmpeg)")

    flow = _load_flow()
    if args.port is not None:
        flow["base_url"] = re.sub(r":\d+", f":{args.port}", flow["base_url"])
    port = int(re.search(r":(\d+)", flow["base_url"]).group(1))
    max_mb = args.max_size or float(flow.get("max_size_mb", 10))

    server = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "memory_arena.cli",
            "serve",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        env={**os.environ, "PYTHONUNBUFFERED": "1"},
    )
    try:
        _wait_for_port(port)
        webm = asyncio.run(_drive(flow))
        print(f"[recorded] {webm} ({webm.stat().st_size / 1024:.0f} KB)")
        _transcode_mp4(webm)
        print(f"[mp4] {TOUR_MP4} ({TOUR_MP4.stat().st_size / 1024:.0f} KB)")
        _make_gif(webm, max_mb)
        print(f"[gif] {TOUR_GIF} ({TOUR_GIF.stat().st_size / 1024:.0f} KB)")
        _verify(webm)
    finally:
        server.terminate()
        try:
            server.wait(timeout=5)
        except subprocess.TimeoutExpired:
            server.kill()


if __name__ == "__main__":
    main()
