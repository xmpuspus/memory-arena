"""Memory Arena FastAPI dashboard.

Serves the Next.js static bundle and exposes /api endpoints for the dashboard
to read benchmark results, list strategies, and stream the demo recall flow.
"""

from __future__ import annotations

import json
import re
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi import Path as FPath
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from memory_arena import __version__ as _ma_version
from memory_arena.settings import settings


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield


app = FastAPI(
    title="Memory Arena",
    description="Benchmark agent-memory architectures",
    version=_ma_version,
    lifespan=lifespan,
)

# CORS: only localhost / 127.0.0.1 by default. Wildcard origins are never
# allowed — the dashboard ships as a same-origin static bundle, so the only
# real cross-origin caller is a local Next.js dev server.
_DEFAULT_ORIGINS = [
    "http://localhost:3000",
    "http://localhost:3001",
    "http://localhost:8000",
    "http://127.0.0.1:3000",
    "http://127.0.0.1:3001",
    "http://127.0.0.1:8000",
]
origins = settings.cors_origins or _DEFAULT_ORIGINS
# Refuse a wildcard CORS even if someone overrides via env var; fall back to
# the safe default and warn at startup.
if any(o.strip() == "*" for o in origins):
    import logging

    logging.getLogger(__name__).warning(
        "Refusing to honor allow_origins=['*']; falling back to localhost allowlist."
    )
    origins = _DEFAULT_ORIGINS
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


# Path-segment validator — used on every {corpus} / {strategy} path param so a
# request like `/api/recall-records/..%2F..%2Fetc/passwd/x` can never reach the
# filesystem layer.
_SLUG_RE = re.compile(r"^[a-z0-9_\-]+$")


def _safe_slug(value: str, kind: str) -> str:
    if not _SLUG_RE.match(value):
        raise HTTPException(
            status_code=400,
            detail=f"invalid {kind}: must match [a-z0-9_-]+",
        )
    return value


class HealthResponse(BaseModel):
    status: str
    strategies: list[str]
    has_results: bool


@app.get("/api/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    from memory_arena.paths import results_root
    from memory_arena.strategies import STRATEGY_REGISTRY

    results_dir = results_root()
    has_results = (
        bool(list(results_dir.glob("longmemeval-s_*.json"))) if results_dir.exists() else False
    )
    return HealthResponse(
        status="ok",
        strategies=list(STRATEGY_REGISTRY.keys()),
        has_results=has_results,
    )


@app.get("/api/corpora")
async def list_corpora() -> dict:
    from memory_arena.paths import datasets_root

    base = datasets_root()
    out: list[dict] = []
    if base.exists():
        for d in sorted(base.iterdir()):
            if not d.is_dir() or d.name.startswith("."):
                continue
            q_dir = d / "questions"
            count = 0
            if q_dir.is_dir():
                for q in q_dir.glob("*.yaml"):
                    try:
                        count += q.read_text().count("- id:")
                    except OSError:
                        pass
            out.append({"name": d.name, "label": d.name.replace("-", " ").title(), "count": count})
    if not out:
        out = [{"name": "longmemeval-s", "label": "LongMemEval-S", "count": 30}]
    return {"corpora": out}


@app.get("/api/strategies")
async def list_strategies() -> dict:
    from memory_arena.strategies import STRATEGY_REGISTRY

    return {"strategies": [{"name": n, "available": True} for n in STRATEGY_REGISTRY]}


@app.get("/api/results/{corpus}")
async def get_results(
    corpus: str = FPath(..., pattern=r"^[a-z0-9_\-]+$"),
) -> dict:
    corpus = _safe_slug(corpus, "corpus")
    from memory_arena.paths import results_root

    results_dir = results_root()
    # Prefer per-strategy bootstrap summary files (_summary.json). Fall back to
    # per-seed files only if no summary exists. Without this filter the
    # dashboard renders one row per (strategy, seed) which looks like 51 rows
    # of duplicates instead of 17 strategies.
    summary_files = sorted(results_dir.glob(f"{corpus}_*_summary.json"))
    if summary_files:
        files = summary_files
    else:
        # Pick the latest per-seed file per strategy (or single non-seed file).
        all_files = sorted(results_dir.glob(f"{corpus}_*.json"))
        per_strategy: dict[str, Path] = {}
        for p in all_files:
            stem = p.stem
            prefix = f"{corpus}_"
            rest = stem[len(prefix) :] if stem.startswith(prefix) else stem
            if rest.endswith("_summary"):
                continue
            strategy = rest.rsplit("_seed", 1)[0]
            per_strategy[strategy] = p  # last one wins (sorted = highest seed)
        files = sorted(per_strategy.values())
    if not files:
        raise HTTPException(status_code=404, detail=f"No results for corpus: {corpus}")
    rows: list[dict] = []
    for p in files:
        try:
            data = json.loads(p.read_text())
        except Exception:
            continue
        # Project a flat row that the dashboard expects. Per-category metrics
        # use null when no question of that category was evaluated, so the UI
        # can show "—" rather than a misleading default.
        rows.append(
            {
                "strategy": data.get("strategy"),
                "accuracy": data.get("accuracy", 0.0),
                "mean_session_recall_at_k": data.get("mean_session_recall_at_k", 0.0),
                "mean_session_hit_at_k": data.get("mean_session_hit_at_k", 0.0),
                "avg_recall_latency_ms": data.get("avg_recall_latency_ms", 0.0),
                "total_cost_usd": data.get("total_cost_usd", 0.0),
                "abstention_f1": data.get("abstention_f1"),
                "abstention_n": data.get("abstention_n", 0),
                "update_precision": data.get("update_precision"),
                "update_n": data.get("update_n", 0),
                "temporal_correctness": data.get("temporal_correctness"),
                "temporal_n": data.get("temporal_n", 0),
                "accuracy_by_category": data.get("accuracy_by_category", {}),
                # Summary JSONs (bootstrap mean) carry questions_evaluated_per_seed
                # as a list. Per-seed JSONs carry it as an int. Fall back to the
                # length of recall_records on legacy v0.1.4 files.
                "questions_evaluated": (
                    data.get("questions_evaluated")
                    if isinstance(data.get("questions_evaluated"), int)
                    else max(data.get("questions_evaluated_per_seed", [0]))
                    if data.get("questions_evaluated_per_seed")
                    else len(data.get("recall_records", []))
                ),
                "errors": len(data.get("errors", [])),
                "run_id": data.get("run_id"),
            }
        )
    return {"corpus": corpus, "results": rows}


@app.get("/api/benchmark/{corpus}")
async def get_benchmark(
    corpus: str = FPath(..., pattern=r"^[a-z0-9_\-]+$"),
) -> dict:
    return await get_results(corpus)


@app.get("/api/recall-records/{corpus}/{strategy}")
async def get_recall_records(
    corpus: str = FPath(..., pattern=r"^[a-z0-9_\-]+$"),
    strategy: str = FPath(..., pattern=r"^[a-z0-9_\-]+$"),
) -> dict:
    """Per-question recall records for the Recall Lab page."""
    from memory_arena.paths import results_root
    from memory_arena.strategies import STRATEGY_REGISTRY

    corpus = _safe_slug(corpus, "corpus")
    strategy = _safe_slug(strategy, "strategy")
    # Strategy must be a registered strategy. Without this, anything matching
    # the slug regex would let an attacker probe for {corpus}_{anything}.json
    # files inside results_root.
    if strategy not in STRATEGY_REGISTRY:
        raise HTTPException(status_code=404, detail=f"unknown strategy: {strategy}")

    results_dir = results_root().resolve()
    # The bundled wheel snapshot ships only `_seed{N}.json` and `_summary.json`
    # (no bare `{corpus}_{strategy}.json`), so prefer the bare file when present
    # (latest local run) and fall back to seed 0, which carries the same
    # `recall_records`. Summary files don't, so they are not a candidate.
    path = None
    for name in (f"{corpus}_{strategy}.json", f"{corpus}_{strategy}_seed0.json"):
        candidate = (results_dir / name).resolve()
        # Defense in depth: ensure the resolved path is inside results_root even
        # though the slug regex already forbids '..' / '/'.
        try:
            candidate.relative_to(results_dir)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail="invalid path") from exc
        if candidate.exists():
            path = candidate
            break

    if path is None:
        raise HTTPException(status_code=404, detail=f"No results for {corpus}/{strategy}")
    try:
        data = json.loads(path.read_text())
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    # Decorate each record with a per-record measurability flag and a
    # strategy-level measurability flag so the dashboard can show
    # "Recall not measurable for this strategy" instead of pretending HIT/MISS.
    return {
        "corpus": corpus,
        "strategy": strategy,
        "recall_at_k_measurable": data.get("recall_at_k_measurable"),
        "top_k": data.get("top_k"),
        "records": data.get("recall_records", []),
    }


# Mount the bundled Next.js static dashboard if present
_static_path = Path(__file__).parent.parent / "static"
if _static_path.exists() and any(_static_path.iterdir()):
    app.mount("/", StaticFiles(directory=str(_static_path), html=True), name="static")
