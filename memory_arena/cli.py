"""Memory Arena CLI — multi-stage pipeline.

init-corpus -> ingest-sessions -> build-memory -> benchmark -> report -> serve

Each command is independently runnable and re-runnable.
"""

from __future__ import annotations

import logging
import re

import typer
from rich.console import Console
from rich.logging import RichHandler

app = typer.Typer(
    name="memory-arena",
    help=(
        "Benchmark agent-memory architectures: Mem0, Graphiti, Cognee, "
        "LangMem, Memori, naive vector, recency window, full context."
    ),
    no_args_is_help=True,
)
console = Console()


# Corpus names flow into Path("datasets") / corpus / ... . Without validation a
# user (or attacker) could pass `../../etc/passwd`. Match the same allowlist
# the FastAPI handler uses for {corpus} path params.
_CORPUS_SLUG_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")


def _validate_corpus_slug(corpus: str) -> str:
    if not _CORPUS_SLUG_RE.match(corpus):
        raise typer.BadParameter("invalid corpus name")
    return corpus


def _print_version_and_exit(value: bool) -> None:
    if not value:
        return
    try:
        from importlib.metadata import version

        v = version("memory-arena")
    except Exception:
        from memory_arena import __version__ as v
    console.print(f"memory-arena {v}")
    raise typer.Exit()


@app.callback()
def _setup(
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable debug logging"),
    version: bool = typer.Option(
        False,
        "--version",
        callback=_print_version_and_exit,
        is_eager=True,
        help="Show memory-arena version and exit.",
    ),
) -> None:
    level = logging.DEBUG if verbose else logging.WARNING
    logging.basicConfig(
        level=level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(rich_tracebacks=True, show_path=verbose)],
    )


def _preflight(needs_anthropic: bool = False, needs_openai: bool = False) -> None:
    from memory_arena.settings import settings

    errors: list[str] = []
    if needs_anthropic and not settings.anthropic_api_key:
        errors.append("Anthropic API key required. Set ANTHROPIC_API_KEY in .env")
    if needs_openai and not settings.openai_api_key:
        errors.append("OpenAI API key required. Set OPENAI_API_KEY in .env")
    if errors:
        for e in errors:
            console.print(f"[red]{e}[/red]")
        raise typer.Exit(1)


@app.command()
def init_corpus(
    name: str = typer.Argument(..., help="Corpus name (e.g. longmemeval-s)"),
):
    """Scaffold a new corpus directory."""
    from pathlib import Path

    name = _validate_corpus_slug(name)
    base = Path("datasets") / name
    if base.exists():
        console.print(f"[yellow]Corpus directory already exists: {base}[/yellow]")
        return

    for subdir in ["raw", "processed", "questions", "questions/smoke"]:
        (base / subdir).mkdir(parents=True, exist_ok=True)

    console.print(f"[green]Created corpus scaffold:[/green] {base}/")
    console.print("  raw/         <- drop your session JSON here")
    console.print("  processed/   <- normalized sessions.jsonl + questions.jsonl")
    console.print("  questions/   <- benchmark questions (YAML)")


@app.command()
def download_longmemeval(
    out_dir: str = typer.Option(
        "datasets/longmemeval-s/raw",
        "--out-dir",
        help="Where to save longmemeval_s.json",
    ),
):
    """Download LongMemEval_S dataset from upstream GitHub."""
    import urllib.request
    from pathlib import Path

    url = "https://raw.githubusercontent.com/xiaowu0162/LongMemEval/main/data/longmemeval_s.json"
    out = Path(out_dir) / "longmemeval_s.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    console.print(f"Downloading {url} -> {out}")
    try:
        urllib.request.urlretrieve(url, out)  # nosec B310 - trusted dataset URL
    except Exception as exc:
        console.print(f"[red]Download failed: {exc}[/red]")
        raise typer.Exit(1) from exc
    console.print(f"[green]Saved {out}[/green]")


@app.command()
def ingest_sessions(
    corpus: str = typer.Option("longmemeval-s", help="Corpus name"),
):
    """Stage 1: Normalize raw session JSON into JSONL."""
    from pathlib import Path

    from memory_arena.sessions.loaders import LongMemEvalLoader

    corpus = _validate_corpus_slug(corpus)
    raw_path = Path("datasets") / corpus / "raw" / "longmemeval_s.json"
    if not raw_path.exists():
        console.print(
            f"[red]Raw file not found: {raw_path}[/red]\nRun: memory-arena download-longmemeval"
        )
        raise typer.Exit(1)
    loader = LongMemEvalLoader(raw_path)
    sessions, questions = loader.normalize()
    out_dir = Path("datasets") / corpus / "processed"
    s_path, q_path = loader.write_processed(sessions, questions, out_dir)
    console.print(
        f"[green]Wrote {len(sessions)} sessions to {s_path}[/green]\n"
        f"[green]Wrote {len(questions)} questions to {q_path}[/green]"
    )


@app.command()
def build_memory(
    corpus: str = typer.Option("longmemeval-s", help="Corpus name"),
    strategy: str = typer.Option("all", help="Strategy or 'all'"),
):
    """Stage 2: Set up memory backends for each strategy.

    For most strategies this is a no-op (setup happens in benchmark). For
    naive_vector and graph-backed strategies, collections/indices are created.
    """
    corpus = _validate_corpus_slug(corpus)
    console.print(
        "[dim]build-memory is a no-op in v0.1: backends are set up per-run inside benchmark.[/dim]"
    )
    console.print(f"  corpus={corpus} strategy={strategy}")


@app.command()
def benchmark(
    corpus: str = typer.Option("longmemeval-s", help="Corpus name"),
    strategy: str = typer.Option("all", help="Comma-separated names or 'all'"),
    questions: str = typer.Option("smoke", help="Question subset: smoke, full, or path"),
    cost_cap: float = typer.Option(5.0, help="Halt if cumulative cost exceeds (USD)"),
    top_k: int = typer.Option(5, "--top-k", help="Held constant across all strategies."),
    seed: int | None = typer.Option(
        None,
        "--seed",
        help=(
            "Pin RNG seed for one run; output suffixed _seed{N} so multi-seed "
            "bootstraps don't overwrite."
        ),
    ),
    dry_run: bool = typer.Option(False, "--dry-run"),
):
    """Stage 3: Run the benchmark across strategies."""
    import asyncio

    corpus = _validate_corpus_slug(corpus)
    _preflight(needs_anthropic=True)

    from memory_arena.benchmark.runner import run_memory_benchmark

    if dry_run:
        from memory_arena.strategies import STRATEGY_NAMES

        names = STRATEGY_NAMES if strategy == "all" else [s.strip() for s in strategy.split(",")]
        console.print("[bold]Dry run: benchmark[/bold]")
        console.print(f"  corpus: {corpus}")
        console.print(f"  questions: {questions}")
        console.print(f"  strategies: {', '.join(names)}")
        console.print(f"  top_k: {top_k}")
        console.print(f"  cost cap: ${cost_cap:.2f}")
        console.print(f"  seed: {seed if seed is not None else '(none)'}")
        return

    # TODO(U11, docs/audits/2026-04-30): live cumulative-cost ticker in the
    # Rich progress display. Requires a callback hook from
    # memory_arena.benchmark.runner.run_memory_benchmark — owned by Agent A.
    # When the hook lands, wire a Rich Progress with a "$" column that updates
    # after each question.
    asyncio.run(
        run_memory_benchmark(
            corpus=corpus,
            strategy=strategy,
            questions=questions,
            cost_cap=cost_cap,
            top_k=top_k,
            seed=seed,
        )
    )


@app.command()
def report(
    corpus: str = typer.Option("longmemeval-s", help="Corpus to report on"),
    output: str | None = typer.Option(None, help="Output path"),
    fmt: str = typer.Option("rich", "--format", help="rich | json"),
):
    """Generate a benchmark report from the latest run.

    Prefers per-strategy bootstrap _summary.json files when present;
    otherwise reads per-seed files and dedupes by strategy.
    """
    import json
    import sys
    from pathlib import Path

    from memory_arena.paths import results_root

    corpus = _validate_corpus_slug(corpus)
    results_dir = results_root()
    summary_files = sorted(results_dir.glob(f"{corpus}_*_summary.json"))
    if summary_files:
        files = summary_files
    else:
        all_files = sorted(results_dir.glob(f"{corpus}_*.json"))
        per_strategy: dict[str, Path] = {}
        for p in all_files:
            stem = p.stem
            prefix = f"{corpus}_"
            rest = stem[len(prefix) :] if stem.startswith(prefix) else stem
            if rest.endswith("_summary"):
                continue
            strategy = rest.rsplit("_seed", 1)[0]
            per_strategy[strategy] = p
        files = sorted(per_strategy.values())
    if not files:
        console.print(f"[red]No results found for {corpus}[/red]")
        raise typer.Exit(1)

    rows: list[dict] = []
    for p in files:
        try:
            data = json.loads(p.read_text())
            rows.append(data)
        except Exception:
            continue

    # Sort by accuracy descending so the report matches the dashboard
    # leaderboard ordering. Strategies with no accuracy field sink to the
    # bottom (treated as 0).
    rows.sort(key=lambda r: r.get("accuracy") or 0.0, reverse=True)

    if fmt == "json":
        sys.stdout.write(json.dumps(rows, indent=2) + "\n")
        return

    console.print(f"[bold]Memory Arena Report: {corpus}[/bold]")
    for row in rows:
        f1 = row.get("abstention_f1")
        f1_str = f"{f1:.2f}" if f1 is not None else "—"
        console.print(
            f"  {row.get('strategy', '?')}: "
            f"acc={row.get('accuracy', 0):.2%}, "
            f"abst_f1={f1_str}, "
            f"cost=${row.get('total_cost_usd', 0):.4f}"
        )
    if output:
        Path(output).write_text(json.dumps(rows, indent=2))
        console.print(f"Wrote {output}")


@app.command(name="recall-lab")
def recall_lab(
    corpus: str = typer.Option("longmemeval-s", help="Corpus to evaluate"),
    top_k: int = typer.Option(10, "--top-k"),
    strategies: str = typer.Option("all", help="Strategy filter"),
    min_recall: float = typer.Option(0.30, "--min-recall"),
):
    """Run retrieval-only benchmark (turn-level Recall@k)."""
    import asyncio

    corpus = _validate_corpus_slug(corpus)
    _preflight(needs_openai=True)

    from memory_arena.benchmark.recall_lab import run_recall_lab

    exit_code = asyncio.run(run_recall_lab(corpus, strategies, top_k, min_recall))
    if exit_code:
        raise typer.Exit(exit_code)


def _pick_free_port(start: int, host: str = "127.0.0.1", span: int = 20) -> int:
    """Find a free port. If start==0, OS picks any free port; else scan span."""
    import socket

    if start == 0:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind((host, 0))
            return s.getsockname()[1]
    for c in range(start, start + span):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(("127.0.0.1", c)) != 0:
                return c
    return start


def _warn_if_lan_bind(host: str) -> None:
    if host == "0.0.0.0":  # nosec B104 - warning helper, not a real bind
        console.print(
            "[yellow]binding 0.0.0.0 — dashboard reachable from your local network. "
            "Use 127.0.0.1 if you don't want this.[/yellow]"
        )


@app.command()
def serve(
    host: str = typer.Option("127.0.0.1", "--host", help="Host to bind to (default 127.0.0.1)"),
    port: int = typer.Option(
        8000,
        "--port",
        help="Port to listen on. Pass 0 to auto-pick a free port.",
    ),
):
    """Launch the FastAPI dashboard server."""
    import uvicorn

    actual_port = _pick_free_port(port) if port == 0 else port
    _warn_if_lan_bind(host)
    console.print(f"dashboard at http://127.0.0.1:{actual_port}/")
    uvicorn.run("memory_arena.chatbot.api:app", host=host, port=actual_port)


@app.command()
def arena(
    host: str = typer.Option("127.0.0.1", "--host", help="Host to bind to (default 127.0.0.1)"),
    port: int = typer.Option(
        8000,
        "--port",
        help="Port to listen on. Pass 0 to auto-pick a free port.",
    ),
) -> None:
    """Alias for `serve` — the head-to-head ELO arena lives in the dashboard."""
    serve(host=host, port=port)


@app.command()
def demo(
    host: str = typer.Option("127.0.0.1", "--host", help="Host to bind to (default 127.0.0.1)"),
    port: int = typer.Option(
        8000,
        "--port",
        help="Port to listen on. Pass 0 to auto-pick a free port.",
    ),
):
    """Launch the demo server with pre-computed results."""
    import webbrowser
    from threading import Timer

    from memory_arena.paths import results_root

    results_dir = results_root()
    if not list(results_dir.glob("longmemeval-s_*.json")):
        console.print(
            "[yellow]No pre-computed results found. The dashboard home page will "
            "still load but the benchmark table will be empty.[/yellow]"
        )

    actual_port = _pick_free_port(port if port != 0 else 0)

    _warn_if_lan_bind(host)
    console.print(f"dashboard at http://127.0.0.1:{actual_port}/")

    def _open():
        webbrowser.open(f"http://127.0.0.1:{actual_port}/")

    Timer(1.5, _open).start()
    import uvicorn

    uvicorn.run("memory_arena.chatbot.api:app", host=host, port=actual_port)


@app.command()
def health(fmt: str = typer.Option("rich", "--format", help="rich | json")):
    """Pipeline status: API keys, services, strategies."""
    import json
    import sys

    from memory_arena.settings import settings
    from memory_arena.strategies import STRATEGY_REGISTRY

    state = {
        "api_keys": {
            "anthropic": bool(settings.anthropic_api_key),
            "openai": bool(settings.openai_api_key),
        },
        "strategies_registered": list(STRATEGY_REGISTRY.keys()),
    }

    if fmt == "json":
        sys.stdout.write(json.dumps(state, indent=2) + "\n")
        return

    console.print("[bold]Memory Arena Health[/bold]")
    console.print(f"  Anthropic key: {'set' if state['api_keys']['anthropic'] else 'missing'}")
    console.print(f"  OpenAI key:    {'set' if state['api_keys']['openai'] else 'missing'}")
    console.print(f"  Strategies:    {', '.join(state['strategies_registered'])}")


if __name__ == "__main__":
    app()
