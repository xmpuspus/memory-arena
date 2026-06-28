"""Custom exception hierarchy for Memory Arena."""


class MemoryArenaError(Exception):
    """Base exception for all Memory Arena errors."""


class IngestError(MemoryArenaError):
    """Error during session ingestion or normalization."""


class GraphError(MemoryArenaError):
    """Error during Neo4j graph operations."""


class StrategyError(MemoryArenaError):
    """Error during strategy setup, ingest, or recall."""


class MemorySystemError(MemoryArenaError):
    """Error from a vendor memory system (mem0, graphiti, cognee, etc)."""


class EvaluationError(MemoryArenaError):
    """Error during benchmark evaluation."""


class LLMError(MemoryArenaError):
    """Error during LLM API calls."""


# Backwards-compat alias for older imports
KBArenaError = MemoryArenaError
