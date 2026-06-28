"""Memory strategies for agent memory benchmarking.

Lifecycle: setup(run_id) -> ingest_session(...) -> recall(query) -> teardown()

Always-available strategies (no vendor SDK):
    full_context, recency_window, naive_vector, bm25, hybrid_rrf, hyde,
    persona_profile, reflection, raptor, karpathy_llm_wiki, hipporag2, amem,
    qiss (quantum-inspired reranker, pure NumPy)

Optional strategies (require an extra SDK):
    mem0, mem0g, graphiti, graphiti_falkor, cognee, langmem, memori,
    sqr (Qiskit Aer SWAP-test reranker, needs the [quantum] extra)
"""

from __future__ import annotations

import logging

from memory_arena.strategies.amem import AMEMStrategy
from memory_arena.strategies.base import IngestRecord, MemoryStrategy, RecallResult
from memory_arena.strategies.bm25 import BM25Strategy
from memory_arena.strategies.full_context import FullContextStrategy
from memory_arena.strategies.hipporag2 import HippoRAG2Strategy
from memory_arena.strategies.hybrid_rrf import HybridRRFStrategy
from memory_arena.strategies.hyde import HydeStrategy
from memory_arena.strategies.karpathy_llm_wiki import KarpathyLlmWikiStrategy
from memory_arena.strategies.naive_vector import NaiveVectorStrategy
from memory_arena.strategies.persona_profile import PersonaProfileStrategy
from memory_arena.strategies.quantum.qiss import QISSStrategy
from memory_arena.strategies.recency_window import RecencyWindowStrategy
from memory_arena.strategies.reflection import ReflectionStrategy

logger = logging.getLogger(__name__)


def _try_import(name: str, dotted: str, attr: str):
    try:
        module = __import__(dotted, fromlist=[attr])
        return name, getattr(module, attr)
    except Exception as exc:
        logger.debug("%s unavailable: %s", name, exc)
        return None


def _register_optional() -> dict[str, type]:
    optional: dict[str, type] = {}
    candidates = [
        ("mem0", "memory_arena.strategies.mem0", "Mem0Strategy"),
        ("mem0g", "memory_arena.strategies.mem0g", "Mem0GraphStrategy"),
        ("graphiti", "memory_arena.strategies.graphiti", "GraphitiStrategy"),
        ("graphiti_falkor", "memory_arena.strategies.graphiti_falkor", "GraphitiFalkorStrategy"),
        ("cognee", "memory_arena.strategies.cognee", "CogneeStrategy"),
        ("langmem", "memory_arena.strategies.langmem", "LangMemStrategy"),
        ("memori", "memory_arena.strategies.memori", "MemoriStrategy"),
        ("raptor", "memory_arena.strategies.raptor", "RaptorStrategy"),
        # sqr imports qiskit/qiskit_aer at module top; _try_import drops it
        # cleanly when the [quantum] extra isn't installed, the same gate as the
        # vendor SDKs, keeps the core install + CI light.
        ("sqr", "memory_arena.strategies.quantum.sqr", "SQRStrategy"),
    ]
    for name, dotted, attr in candidates:
        result = _try_import(name, dotted, attr)
        if result is not None:
            optional[result[0]] = result[1]
    return optional


STRATEGY_REGISTRY: dict[str, type[MemoryStrategy]] = {
    "full_context": FullContextStrategy,
    "recency_window": RecencyWindowStrategy,
    "naive_vector": NaiveVectorStrategy,
    "bm25": BM25Strategy,
    "hybrid_rrf": HybridRRFStrategy,
    "hyde": HydeStrategy,
    "persona_profile": PersonaProfileStrategy,
    "reflection": ReflectionStrategy,
    "karpathy_llm_wiki": KarpathyLlmWikiStrategy,
    "hipporag2": HippoRAG2Strategy,
    "amem": AMEMStrategy,
    "qiss": QISSStrategy,
    **_register_optional(),
}


STRATEGY_NAMES = list(STRATEGY_REGISTRY.keys())


def get_strategy(name: str) -> MemoryStrategy:
    """Instantiate a strategy by name."""
    cls = STRATEGY_REGISTRY.get(name)
    if cls is None:
        raise ValueError(
            f"Unknown strategy: {name}. Available: {list(STRATEGY_REGISTRY)}. "
            "Vendor strategies (mem0, mem0g, graphiti, cognee, langmem, memori) "
            "require their SDKs to be installed: pip install 'memory-arena[all-systems]'"
        )
    return cls()


def register_plugin_strategy(module_path: str) -> None:
    """Import a user module and register its MemoryStrategy subclass.

    Convention: the module should expose a ``__strategy__`` attribute that
    points at the MemoryStrategy subclass to register::

        # my_plugin.py
        class MyStrategy(MemoryStrategy):
            name = "my_strategy"
            ...

        __strategy__ = MyStrategy

    If ``__strategy__`` is unset, the loader falls back to scanning
    ``dir(module)`` for a single MemoryStrategy subclass defined in the
    plugin module itself (i.e. ``cls.__module__ == module_path``). This
    avoids picking up ``MemoryStrategy`` imported as a type-hint base.
    """
    import importlib

    mod = importlib.import_module(module_path)

    # Preferred: explicit __strategy__ attribute.
    explicit = getattr(mod, "__strategy__", None)
    if explicit is not None:
        if not (
            isinstance(explicit, type)
            and issubclass(explicit, MemoryStrategy)
            and explicit is not MemoryStrategy
        ):
            raise ValueError(
                f"{module_path}.__strategy__ must be a MemoryStrategy subclass, got {explicit!r}"
            )
        cls = explicit
        name = getattr(cls, "name", module_path.split(".")[-1])
        STRATEGY_REGISTRY[name] = cls
        logger.info("Registered plugin strategy: %s from %s (via __strategy__)", name, module_path)
        return

    # Fallback: scan, but only consider classes defined in the plugin module
    # itself so that imported base classes / type hints are not picked up.
    candidates = [
        obj
        for name in dir(mod)
        if not name.startswith("_")
        for obj in [getattr(mod, name)]
        if (
            isinstance(obj, type)
            and issubclass(obj, MemoryStrategy)
            and obj is not MemoryStrategy
            and getattr(obj, "__module__", "") == module_path
        )
    ]
    if not candidates:
        raise ValueError(
            f"No MemoryStrategy subclass found in {module_path}. "
            "Set `__strategy__ = YourStrategy` at module top-level."
        )
    if len(candidates) > 1:
        raise ValueError(
            f"Multiple MemoryStrategy subclasses in {module_path}: "
            f"{[c.__name__ for c in candidates]}. Set `__strategy__` to disambiguate."
        )

    cls = candidates[0]
    name = getattr(cls, "name", module_path.split(".")[-1])
    STRATEGY_REGISTRY[name] = cls
    logger.info("Registered plugin strategy: %s from %s", name, module_path)


__all__ = [
    "AMEMStrategy",
    "BM25Strategy",
    "FullContextStrategy",
    "HippoRAG2Strategy",
    "HybridRRFStrategy",
    "HydeStrategy",
    "IngestRecord",
    "KarpathyLlmWikiStrategy",
    "MemoryStrategy",
    "NaiveVectorStrategy",
    "PersonaProfileStrategy",
    "QISSStrategy",
    "RecallResult",
    "RecencyWindowStrategy",
    "ReflectionStrategy",
    "STRATEGY_NAMES",
    "STRATEGY_REGISTRY",
    "get_strategy",
    "register_plugin_strategy",
]
