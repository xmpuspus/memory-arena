"""Memory benchmark runner.

Per-strategy lifecycle:
    setup(run_id) -> ingest_session(...) sequentially -> recall(query) per question
    -> teardown()

Strategies run in parallel across each other (each in its own asyncio task) but
sessions ingest sequentially within a strategy because vendor systems
(Graphiti, Mem0g) reason over insertion order. Cost cap halts the run when
exceeded.
"""

from __future__ import annotations

import asyncio
import logging
import os
import platform
import subprocess
import sys
import time
from datetime import UTC, datetime
from importlib import metadata as importlib_metadata
from pathlib import Path
from uuid import uuid4

from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn

from memory_arena.benchmark.evaluator import evaluate_memory_answer
from memory_arena.benchmark.questions import load_memory_questions
from memory_arena.benchmark.recall_metrics import compute_memory_recall_metrics
from memory_arena.llm.client import LLMClient
from memory_arena.sessions.loaders import load_sessions
from memory_arena.settings import settings
from memory_arena.strategies import STRATEGY_REGISTRY, get_strategy
from memory_arena.strategies.base import MemoryStrategy

logger = logging.getLogger(__name__)

# Strategies whose ingest path failed >50% of the time at default config get
# the "config-failed-at-default" status badge instead of being silently scored
# against an empty store. The headline table shows the badge; the README
# explicitly invites vendors to PR a working default config in v0.2.
_INGEST_HEALTH_THRESHOLD = 0.5

# Vendor SDKs whose internal LLM calls go through the vendor's own client and
# aren't visible to memory-arena. Their result JSON's cost field is the
# memory-arena-paid cost only (e.g. the Sonnet generation step). Footnote in
# the table.
_VENDOR_INTERNAL_COST_NOT_MEASURED = {"langmem", "memori", "mem0", "mem0g", "graphiti", "cognee"}

# Optional vendor SDKs that may not be installed; importlib_metadata.version
# raises PackageNotFoundError for missing ones. Track those we want stamped.
_TRACKED_PACKAGES = (
    "memory-arena",
    "anthropic",
    "openai",
    "chromadb",
    "neo4j",
    "fastapi",
    "pydantic",
    "rank-bm25",
    "tiktoken",
    "tenacity",
    "numpy",
    "scikit-learn",
    "mem0ai",
    "graphiti-core",
    "cognee",
    "langmem",
    "langgraph",
    "memori",
    "psycopg",
    "s3fs",
)


def _git_sha() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parents[2],
            stderr=subprocess.DEVNULL,
        )
        return out.decode().strip()
    except Exception:
        return "unknown"


def _package_versions() -> dict[str, str]:
    out: dict[str, str] = {}
    for pkg in _TRACKED_PACKAGES:
        try:
            out[pkg] = importlib_metadata.version(pkg)
        except importlib_metadata.PackageNotFoundError:
            out[pkg] = "not-installed"
    return out


def _build_run_metadata(top_k: int, seed: int | None) -> dict:
    return {
        "commit_sha": _git_sha(),
        "started_utc": datetime.now(UTC).isoformat(),
        "host": f"{platform.system()} {platform.release()} ({platform.machine()})",
        "python": sys.version.split()[0],
        "top_k": top_k,
        "top_k_normalized": True,
        "seed": seed,
        "models": {
            "anthropic_generate": settings.generate_model,
            "anthropic_fast": settings.fast_model,
            "anthropic_judge": settings.judge_model,
            "openai_generate": settings.openai_generate_model,
            "openai_fast": settings.openai_fast_model,
            "openai_judge": settings.openai_judge_model,
            "embedding": settings.embedding_model,
        },
        "package_versions": _package_versions(),
    }


console = Console()


class CostCapExceededError(Exception):
    pass


def _check_cost_cap(
    strategy_name: str,
    cumulative_cost: dict[str, float],
    cost_cap: float,
    phase: str,
    errors_list: list[str],
    context: str = "",
) -> bool:
    """Return True when the strategy has breached the cost cap and the loop should halt.

    Called BOTH before each ingest/recall iteration AND after the cost addition.
    The before-call check prevents new spend once we're past the cap; the
    after-call check guarantees the next iteration sees the cap-breached state
    even if the previous before-call check was the last one to pass.

    We accept that the breaching call counts. Better to overspend by one call
    than to skip it post-hoc and produce inconsistent state.

    Returns True if the cap is breached (caller should `break`). Logs a WARNING
    and appends a message with overshoot to errors_list on first breach.
    """
    if cost_cap <= 0:
        return False
    spent = cumulative_cost.get(strategy_name, 0.0)
    if spent < cost_cap:
        return False
    overshoot = spent - cost_cap
    suffix = f" during {phase}"
    if context:
        suffix += f" of {context}"
    msg = (
        f"cost cap reached: ${spent:.2f} (cap=${cost_cap:.2f}, "
        f"overshoot=${overshoot:.2f}){suffix}"
    )
    # Only record the breach once per strategy/phase so repeated post-call
    # checks don't spam errors_list. Detect via a flag-like sentinel: scan
    # errors_list for a prior cap-reached line for this phase.
    already_logged = any(
        e.startswith("cost cap reached") and f"during {phase}" in e for e in errors_list
    )
    if not already_logged:
        logger.warning("strategy=%s %s", strategy_name, msg)
        errors_list.append(msg)
    return True


async def _run_strategy(
    strategy: MemoryStrategy,
    sessions: list,
    questions: list,
    run_id: str,
    top_k: int,
    llm: LLMClient,
    cost_cap: float,
    cumulative_cost: dict[str, float],
    progress: Progress,
    task_id,
) -> dict:
    started = datetime.now(UTC).isoformat()
    out = {
        "strategy": strategy.name,
        "run_id": run_id,
        "started_at": started,
        "ingest_records": [],
        "recall_records": [],
        "errors": [],
    }

    try:
        await strategy.setup(run_id)
    except Exception as exc:
        out["errors"].append(f"setup: {exc}")
        # Drain any pre-failure swallowed errors so they aren't lost when setup
        # also raised. Most strategies set self._errors=[] inside setup, so this
        # is a no-op when setup fails on its first line, but it covers
        # partial-setup paths (config_set / prune / register) cleanly.
        swallowed = list(getattr(strategy, "_errors", []) or [])
        if swallowed:
            out["errors"].extend(swallowed)
            logger.warning(
                "strategy=%s setup failed with %d swallowed errors",
                strategy.name,
                len(swallowed),
            )
        out["swallowed_error_count"] = len(swallowed)
        out["finished_at"] = datetime.now(UTC).isoformat()
        return out

    # Sequential ingest. The cost cap is checked twice per iteration:
    # before the call (skip new spend once we're past the cap) and after
    # the cost addition (guarantee the next iteration sees the breached
    # state). The breaching call's cost still counts.
    for session in sessions:
        if _check_cost_cap(
            strategy.name,
            cumulative_cost,
            cost_cap,
            phase="ingest",
            errors_list=out["errors"],
            context=f"session={session.id}",
        ):
            break
        try:
            rec = await strategy.ingest_session(session)
            out["ingest_records"].append(rec.model_dump())
            # cost_usd may be None when the strategy can't measure its own cost
            # (a vendor SDK's server-side LLM loop). Treat as 0 for the cap
            # check; the field stays None on the record so the dashboard
            # renders "—".
            cumulative_cost[strategy.name] += rec.cost_usd or 0.0
            if _check_cost_cap(
                strategy.name,
                cumulative_cost,
                cost_cap,
                phase="ingest",
                errors_list=out["errors"],
                context=f"session={session.id}",
            ):
                break
        except Exception as exc:
            out["errors"].append(f"ingest {session.id}: {exc}")

    # Recall + evaluate
    accuracies: list[float] = []
    abstention_calls: list[dict] = []
    update_calls: list[dict] = []
    temporal_calls: list[dict] = []
    total_cost = 0.0
    total_recall_latency = 0.0
    recall_metrics_acc: list[dict] = []
    # Track whether THIS strategy can produce a fair Recall@k. Vendors that
    # store extracted facts (LangMem, Cognee, Memori) can't return source
    # session_ids, so the IR axis is structurally zero and not the vendor's
    # fault. Aggregator emits null for those.
    recall_at_k_measurable_runs: list[bool] = []

    for q in questions:
        if _check_cost_cap(
            strategy.name,
            cumulative_cost,
            cost_cap,
            phase="recall",
            errors_list=out["errors"],
            context=f"question={q.id}",
        ):
            break
        t0 = time.perf_counter()
        try:
            result = await strategy.recall(q.question, top_k=top_k)
        except Exception as exc:
            out["recall_records"].append(
                {
                    "question_id": q.id,
                    "category": q.category,
                    "answer": f"[ERROR] {exc}",
                    "error": True,
                    "latency_ms": (time.perf_counter() - t0) * 1000,
                }
            )
            continue
        latency_ms = result.latency_ms or (time.perf_counter() - t0) * 1000
        total_recall_latency += latency_ms
        recall_cost = result.cost_usd or 0.0
        cumulative_cost[strategy.name] += recall_cost
        total_cost += recall_cost
        # Post-call cap check: if this call pushed us over, halt before the
        # next iteration. The breaching call's cost still counts and its
        # score is recorded below — we just don't start another one.
        breached_after = _check_cost_cap(
            strategy.name,
            cumulative_cost,
            cost_cap,
            phase="recall",
            errors_list=out["errors"],
            context=f"question={q.id}",
        )

        score = await evaluate_memory_answer(
            answer=result.answer,
            ground_truth=q.ground_truth,
            constraints=q.constraints,
            question=q,
            llm=llm,
            supporting_session_ids=result.supporting_session_ids,
        )
        accuracies.append(score.accuracy)
        if q.category == "abstention":
            abstention_calls.append({"expected": True, "actual": score.abstained})
        if q.category == "knowledge_update" and score.update_precision_correct is not None:
            # Mirror the abstention_f1 null-safe pattern: only count the
            # question toward update_precision if its fact_versions were
            # actually populated. Otherwise the metric inflates to 1.0
            # across every strategy regardless of whether they handled
            # the update.
            update_calls.append({"correct": score.update_precision_correct})
        if q.category == "temporal":
            temporal_calls.append({"correct": score.temporal_correct})

        ir = compute_memory_recall_metrics(
            retrieved_session_ids=result.supporting_session_ids,
            expected_session_ids=set(q.ground_truth.supporting_session_ids),
            retrieved_turn_ids=result.supporting_turn_ids,
            expected_turn_ids=set(q.ground_truth.supporting_turn_ids),
            k=top_k,
        )
        recall_metrics_acc.append(ir)
        # Prefer the class-level signal (set on the strategy class) over the
        # per-RecallResult flag. Strategies whose data model can't carry
        # session_ids declare this once on the class; the per-result field
        # is kept for backward compat but takes a back seat to the class
        # attribute when it's an explicit False.
        cls_measurable = getattr(strategy.__class__, "recall_at_k_measurable", True)
        run_measurable = bool(cls_measurable) and bool(result.recall_at_k_measurable)
        recall_at_k_measurable_runs.append(run_measurable)

        out["recall_records"].append(
            {
                "question_id": q.id,
                "category": q.category,
                "answer": result.answer,
                "supporting_session_ids": result.supporting_session_ids,
                "supporting_turn_ids": result.supporting_turn_ids,
                "score": score.model_dump(),
                "ir": ir,
                "latency_ms": latency_ms,
                "cost_usd": result.cost_usd,
                "tokens_used": result.tokens_used,
                "recall_at_k_measurable": run_measurable,
                "error": False,
            }
        )
        progress.advance(task_id)
        # Surface the headline cost axis live: show running spend per strategy as
        # it accrues, not just at end-of-strategy. progress is duck-typed, so
        # guard the optional update call.
        if hasattr(progress, "update"):
            progress.update(
                task_id,
                description=(
                    f"[cyan]{strategy.name}[/cyan] "
                    f"[dim]${cumulative_cost[strategy.name]:.3f}[/dim]"
                ),
            )
        # The breaching call's record is now persisted. Halt before the
        # next iteration would spend more.
        if breached_after:
            break

    try:
        await strategy.teardown()
    except Exception as exc:
        out["errors"].append(f"teardown: {exc}")

    # Drain any swallowed-exception accumulator that strategies maintain. We
    # surface SDK-level cleanup, retry-and-continue, and best-effort failures
    # so a hostile reviewer can audit whether a strategy's accuracy was a
    # benchmark result or a long string of silently ignored vendor errors.
    swallowed = list(getattr(strategy, "_errors", []) or [])
    if swallowed:
        out["errors"].extend(swallowed)
        logger.warning(
            "strategy=%s finished with %d swallowed errors; see results JSON",
            strategy.name,
            len(swallowed),
        )
    out["swallowed_error_count"] = len(swallowed)

    out["finished_at"] = datetime.now(UTC).isoformat()
    out["accuracy"] = sum(accuracies) / len(accuracies) if accuracies else 0.0
    out["total_cost_usd"] = cumulative_cost[strategy.name]
    out["recall_cost_usd"] = total_cost
    out["avg_recall_latency_ms"] = (
        total_recall_latency / len(out["recall_records"]) if out["recall_records"] else 0.0
    )
    # Per-category metrics only count toward the strategy when at least one
    # question of that category was actually evaluated. Otherwise return None
    # so the dashboard can render "—" instead of a misleading default.
    out["abstention_f1"] = _abstention_f1(abstention_calls) if abstention_calls else None
    out["abstention_n"] = len(abstention_calls)
    out["update_precision"] = _ratio([c["correct"] for c in update_calls]) if update_calls else None
    out["update_n"] = len(update_calls)
    out["temporal_correctness"] = (
        _ratio([c["correct"] for c in temporal_calls]) if temporal_calls else None
    )
    out["temporal_n"] = len(temporal_calls)
    # Recall@k axis: only meaningful when the strategy can return source
    # session_ids. If every answer flagged it as not measurable (LangMem,
    # Cognee, Memori), emit null so the dashboard can render "—" instead of
    # attributing a structural-zero to the vendor.
    recall_at_k_measurable = (
        any(recall_at_k_measurable_runs) if recall_at_k_measurable_runs else True
    )
    out["recall_at_k_measurable"] = recall_at_k_measurable
    if recall_at_k_measurable and recall_metrics_acc:
        out["mean_session_recall_at_k"] = sum(
            r["session_recall_at_k"] for r in recall_metrics_acc
        ) / len(recall_metrics_acc)
        out["mean_session_hit_at_k"] = sum(r["session_hit_at_k"] for r in recall_metrics_acc) / len(
            recall_metrics_acc
        )
    else:
        out["mean_session_recall_at_k"] = None
        out["mean_session_hit_at_k"] = None
    # Ingest health badge — drives the table's "config-failed-at-default" label.
    n_ingest = len(out["ingest_records"])
    n_failed = sum(1 for r in out["ingest_records"] if r.get("error"))
    out["ingest_total"] = n_ingest
    out["ingest_failed"] = n_failed
    out["ingest_failure_rate"] = (n_failed / n_ingest) if n_ingest else 0.0
    if n_ingest and n_failed / n_ingest > _INGEST_HEALTH_THRESHOLD:
        out["status"] = "config-failed-at-default"
    else:
        out["status"] = "ok" if accuracies else "no-questions-evaluated"
    out["cost_measurable"] = strategy.name not in _VENDOR_INTERNAL_COST_NOT_MEASURED
    out["top_k"] = top_k
    out["questions_evaluated"] = len(out["recall_records"])

    # Per-category accuracy breakdown so the dashboard can show how each
    # strategy splits across the 5 question types.
    by_category: dict[str, list[float]] = {}
    for rec in out["recall_records"]:
        cat = rec.get("category") or "unknown"
        score = (rec.get("score") or {}).get("accuracy", 0.0)
        if rec.get("error"):
            continue
        by_category.setdefault(cat, []).append(score)
    out["accuracy_by_category"] = {
        cat: {"accuracy": sum(vs) / len(vs), "n": len(vs)} for cat, vs in by_category.items()
    }
    return out


async def _run_strategy_with_timeout(coro, strategy_name: str, timeout_s: int) -> dict:
    """Run one strategy under a wall-clock budget.

    A single strategy stuck in a non-LLM await (a vendor SDK socket, a Neo4j /
    FalkorDB auth hang, a Chroma lock) would otherwise block the whole
    ``asyncio.gather`` batch forever, since the gather waits on every task. On
    timeout we cancel the coroutine and return a structured error result so the
    run finishes and every other strategy still writes its JSON.
    """
    try:
        return await asyncio.wait_for(coro, timeout=timeout_s)
    except TimeoutError:
        console.print(
            f"[red]{strategy_name} exceeded the {timeout_s}s wall-clock budget; "
            f"recorded as an error, batch continues[/red]"
        )
        return {
            "strategy": strategy_name,
            "errors": [f"strategy exceeded {timeout_s}s wall-clock budget (timed out)"],
            "timed_out": True,
        }


def _abstention_f1(calls: list[dict]) -> float:
    if not calls:
        return 0.0
    tp = sum(1 for c in calls if c["expected"] and c["actual"])
    fp = sum(1 for c in calls if not c["expected"] and c["actual"])
    fn = sum(1 for c in calls if c["expected"] and not c["actual"])
    if tp + fp == 0 or tp + fn == 0:
        return 0.0
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def _ratio(bools: list[bool]) -> float:
    if not bools:
        return 0.0
    return sum(1 for b in bools if b) / len(bools)


def _resolve_strategies(strategy: str) -> list[MemoryStrategy]:
    if strategy == "all":
        names = list(STRATEGY_REGISTRY.keys())
    else:
        names = [s.strip() for s in strategy.split(",") if s.strip()]
    instances: list[MemoryStrategy] = []
    for n in names:
        try:
            instances.append(get_strategy(n))
        except Exception as exc:
            console.print(f"[yellow]Skipping {n}: {exc}[/yellow]")
    return instances


async def run_memory_benchmark(
    corpus: str = "longmemeval-s",
    strategy: str = "all",
    questions: str = "smoke",
    cost_cap: float = 5.0,
    top_k: int = 5,
    seed: int | None = None,
) -> None:
    """Top-level benchmark entrypoint called from the CLI.

    top_k=5 is held constant across all 17 strategies. The audit
    (docs/audits/2026-04-30) showed the v0.1.4 headline "winner" had 2x the
    retrieval budget of the strategy it beat; this default eliminates that
    confound.

    Pass seed=N to run a deterministic seed for the bootstrap rerun. The
    output JSON is suffixed _seed{N} so multi-seed runs don't overwrite.
    """
    run_id = uuid4().hex[:8]
    sessions = load_sessions(corpus)
    qrecords = load_memory_questions(corpus, subset=questions)
    if not sessions:
        console.print(
            f"[red]No sessions found for corpus {corpus}. Run ingest-sessions first.[/red]"
        )
        return
    if not qrecords:
        console.print(f"[red]No questions for corpus {corpus} subset={questions}.[/red]")
        return

    instances = _resolve_strategies(strategy)
    if not instances:
        console.print("[red]No strategies available.[/red]")
        return

    if seed is not None:
        # Pin RNG seeds so within-seed reruns are deterministic. Anthropic API
        # is still non-deterministic; we capture seed in metadata so post-hoc
        # debugging knows which seed produced which numbers.
        os.environ["PYTHONHASHSEED"] = str(seed)
        try:
            import random

            random.seed(seed)
            import numpy as np  # type: ignore

            np.random.seed(seed)
        except ImportError:
            pass

    llm = LLMClient()
    cumulative_cost: dict[str, float] = {s.name: 0.0 for s in instances}

    results_dir = Path(settings.results_path)
    results_dir.mkdir(parents=True, exist_ok=True)
    run_dir = results_dir / f"run_{run_id}"
    run_dir.mkdir(parents=True, exist_ok=True)
    run_metadata = _build_run_metadata(top_k, seed)

    console.print(f"[bold]Memory Arena run_id={run_id}[/bold]")
    console.print(
        f"  corpus={corpus} sessions={len(sessions)} questions={len(qrecords)} "
        f"strategies={[s.name for s in instances]} top_k={top_k} cost_cap=${cost_cap}"
    )

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        tasks = {
            s.name: progress.add_task(f"[cyan]{s.name}", total=len(qrecords)) for s in instances
        }
        coros = [
            _run_strategy_with_timeout(
                _run_strategy(
                    s,
                    sessions,
                    qrecords,
                    run_id,
                    top_k,
                    llm,
                    cost_cap,
                    cumulative_cost,
                    progress,
                    tasks[s.name],
                ),
                s.name,
                settings.benchmark_strategy_timeout_s,
            )
            for s in instances
        ]
        results = await asyncio.gather(*coros, return_exceptions=True)

    import json

    for r in results:
        if isinstance(r, Exception):
            console.print(f"[red]Strategy crashed: {r}[/red]")
            continue
        # Strategies that fail in setup write a result with errors but no
        # aggregated metrics. Backfill defaults so the JSON shape is uniform.
        for key, default in (
            ("accuracy", 0.0),
            ("abstention_f1", 0.0),
            ("update_precision", 0.0),
            ("temporal_correctness", 0.0),
            ("mean_session_recall_at_k", 0.0),
            ("mean_session_hit_at_k", 0.0),
            ("avg_recall_latency_ms", 0.0),
            ("recall_cost_usd", 0.0),
            ("top_k", top_k),
        ):
            r.setdefault(key, default)
        if "total_cost_usd" not in r:
            r["total_cost_usd"] = cumulative_cost.get(r.get("strategy", ""), 0.0)
        # Stamp every result JSON with reproducibility metadata: commit SHA,
        # SDK versions, model IDs, host, run timestamp, seed. Required for
        # third parties to bisect "why does my rerun give different numbers."
        r["metadata"] = run_metadata

        seed_suffix = f"_seed{seed}" if seed is not None else ""
        out_path = results_dir / f"{corpus}_{r['strategy']}{seed_suffix}.json"
        run_path = run_dir / f"{corpus}_{r['strategy']}{seed_suffix}.json"

        out_path.write_text(json.dumps(r, indent=2, default=str))
        run_path.write_text(json.dumps(r, indent=2, default=str))

        # Append to a per-run cost log so the bootstrap orchestrator can sum
        # cumulative spend across seeds and halt before the $50 cap.
        cost_log = results_dir / ".costs.jsonl"
        with cost_log.open("a") as fh:
            fh.write(
                json.dumps(
                    {
                        "ts": datetime.now(UTC).isoformat(),
                        "corpus": corpus,
                        "strategy": r.get("strategy"),
                        "seed": seed,
                        "cost_usd": r.get("total_cost_usd", 0.0),
                        "questions_evaluated": r.get("questions_evaluated", 0),
                    }
                )
                + "\n"
            )

        if r.get("errors"):
            console.print(
                f"  [yellow]{r['strategy']}: errors={r['errors'][0]} "
                f"acc={r['accuracy']:.2%} cost=${r['total_cost_usd']:.4f}[/yellow]"
            )
        else:
            f1 = r.get("abstention_f1")
            f1_str = f"{f1:.2f}" if f1 is not None else "—"
            console.print(
                f"  {r['strategy']}: acc={r['accuracy']:.2%} "
                f"abst_f1={f1_str} "
                f"cost=${r['total_cost_usd']:.4f}"
            )


# Backward-compat alias for code paths still expecting kb-arena's name
run_benchmark = run_memory_benchmark
STRATEGY_NAMES = list(STRATEGY_REGISTRY.keys())


__all__ = [
    "CostCapExceededError",
    "STRATEGY_NAMES",
    "run_benchmark",
    "run_memory_benchmark",
]
