"""Recall Lab: retrieval-only benchmark, ~10x cheaper than full benchmark.

Per strategy: setup -> ingest_session(...) -> recall(question) -> teardown.
Only the IR metrics are computed; the LLM judge is skipped. Useful for tuning
top_k, vector store choice, embedding model, etc.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from rich.console import Console

from memory_arena.benchmark.questions import load_memory_questions
from memory_arena.benchmark.recall_metrics import (
    aggregate_recall_metrics,
    compute_memory_recall_metrics,
)
from memory_arena.sessions.loaders import load_sessions
from memory_arena.settings import settings
from memory_arena.strategies import STRATEGY_REGISTRY, get_strategy

console = Console()


async def _run_one(strategy, sessions, questions, run_id: str, top_k: int) -> dict:
    rows: list[dict] = []
    try:
        await strategy.setup(run_id)
    except Exception as exc:
        return {"strategy": strategy.name, "error": f"setup: {exc}", "rows": rows}
    try:
        for s in sessions:
            try:
                await strategy.ingest_session(s)
            except Exception:
                continue
        for q in questions:
            try:
                result = await strategy.recall(q.question, top_k=top_k)
            except Exception:
                rows.append({"question_id": q.id, "error": True})
                continue
            ir = compute_memory_recall_metrics(
                retrieved_session_ids=result.supporting_session_ids,
                expected_session_ids=set(q.ground_truth.supporting_session_ids),
                retrieved_turn_ids=result.supporting_turn_ids,
                expected_turn_ids=set(q.ground_truth.supporting_turn_ids),
                k=top_k,
            )
            ir["question_id"] = q.id
            ir["category"] = q.category
            rows.append(ir)
    finally:
        try:
            await strategy.teardown()
        except Exception:
            pass
    return {"strategy": strategy.name, "rows": rows}


async def run_recall_lab(
    corpus: str,
    strategies: str = "all",
    top_k: int = 10,
    min_recall: float = 0.30,
) -> int:
    sessions = load_sessions(corpus)
    questions = load_memory_questions(corpus, subset="smoke")
    if not sessions or not questions:
        console.print(f"[red]No data for corpus {corpus}.[/red]")
        return 1

    if strategies == "all":
        names = list(STRATEGY_REGISTRY.keys())
    else:
        names = [s.strip() for s in strategies.split(",") if s.strip()]

    instances = []
    for n in names:
        try:
            instances.append(get_strategy(n))
        except Exception as exc:
            console.print(f"[yellow]Skipping {n}: {exc}[/yellow]")
    if not instances:
        return 1

    run_id = uuid4().hex[:8]
    coros = [_run_one(s, sessions, questions, run_id, top_k) for s in instances]
    out = await asyncio.gather(*coros)

    results_dir = Path(settings.results_path)
    results_dir.mkdir(exist_ok=True, parents=True)
    payload = {
        "run_id": run_id,
        "corpus": corpus,
        "top_k": top_k,
        "started_at": datetime.now(UTC).isoformat(),
        "strategies": [],
    }

    failed = False
    for r in out:
        agg = aggregate_recall_metrics(r.get("rows", []))
        if agg["mean_session_recall_at_k"] < min_recall:
            failed = True
        payload["strategies"].append({"name": r["strategy"], "metrics": agg, "rows": r["rows"]})
        console.print(
            f"  {r['strategy']}: session_recall@{top_k}={agg['mean_session_recall_at_k']:.2%}"
        )

    import json

    out_path = results_dir / f"recall_lab_{corpus}_{run_id}.json"
    out_path.write_text(json.dumps(payload, indent=2))
    console.print(f"[green]Wrote {out_path}[/green]")
    return 1 if failed else 0
