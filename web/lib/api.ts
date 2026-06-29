export const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "";

export const STRATEGIES = [
  "full_context",
  "recency_window",
  "naive_vector",
  "bm25",
  "hybrid_rrf",
  "hyde",
  "persona_profile",
  "reflection",
  "raptor",
  "karpathy_llm_wiki",
  "hipporag2",
  "amem",
  "qiss",
  "sqr",
  "mem0",
  "graphiti",
  "graphiti_falkor",
  "cognee",
  "langmem",
  "memori",
] as const;

export type Strategy = (typeof STRATEGIES)[number];

export const STRATEGY_LABELS: Record<Strategy, string> = {
  full_context: "Full Context",
  recency_window: "Recency Window",
  naive_vector: "Naive Vector",
  bm25: "BM25",
  hybrid_rrf: "Hybrid RRF",
  hyde: "HyDE",
  persona_profile: "Persona Profile",
  reflection: "Reflection",
  raptor: "RAPTOR",
  karpathy_llm_wiki: "Karpathy LLM Wiki",
  hipporag2: "HippoRAG 2",
  amem: "A-MEM",
  qiss: "QISS (Quantum-Inspired)",
  sqr: "SQR (SWAP Test)",
  mem0: "Mem0",
  graphiti: "Graphiti",
  graphiti_falkor: "Graphiti (FalkorDB)",
  cognee: "Cognee",
  langmem: "LangMem",
  memori: "Memori",
};

export const STRATEGY_COLORS: Record<Strategy, string> = {
  full_context: "#64748b",
  recency_window: "#94a3b8",
  naive_vector: "#3b82f6",
  bm25: "#0ea5e9",
  hybrid_rrf: "#06b6d4",
  hyde: "#14b8a6",
  persona_profile: "#10b981",
  reflection: "#84cc16",
  raptor: "#eab308",
  karpathy_llm_wiki: "#fb923c",
  hipporag2: "#dc2626",
  amem: "#7c3aed",
  qiss: "#6a4c93",
  sqr: "#3f3074",
  mem0: "#8b5cf6",
  graphiti: "#f59e0b",
  graphiti_falkor: "#b45309",
  cognee: "#ec4899",
  langmem: "#f43f5e",
  memori: "#d946ef",
};

export const STRATEGY_DESCRIPTIONS: Record<Strategy, string> = {
  full_context:
    "Stuff every ingested turn into the prompt up to the model context limit. Ceiling baseline for what an LLM could answer with perfect recall.",
  recency_window:
    "Last N turns across all sessions. Cheapest baseline. Approximates a chatbot with no memory beyond its current scrollback.",
  naive_vector:
    "Embed every turn with text-embedding-3-large, retrieve top-k by cosine similarity. The 'put it in a vector DB' default.",
  bm25:
    "Pure-Python lexical baseline using rank-bm25. Old-school keyword search, what Google did before vectors.",
  hybrid_rrf:
    "Reciprocal Rank Fusion (k=60) over a vector ranking and a BM25 ranking. Captures both meaning matches and exact-keyword matches.",
  hyde:
    "Hypothetical Document Embeddings: ask the LLM to write a plausible answer first, embed THAT, retrieve real turns by similarity, then answer.",
  persona_profile:
    "Build a one-page JSON profile of the user via Haiku once before recall, then stuff that profile into every answer's system context.",
  reflection:
    "Generative-agents pattern (Park et al. 2023): every N sessions write a synthetic LLM-authored summary, index it alongside raw turns.",
  raptor:
    "Hierarchical k-means clustering with LLM cluster summaries (Sarthi et al. 2024). 4 levels deep, branch=4. Like a table of contents for the chat.",
  karpathy_llm_wiki:
    "Karpathy's LLM-maintained wiki: every session triggers an LLM call that creates or appends entity pages with [[wikilinks]] and [session=...] citations. Periodic lint pass merges duplicates and resolves contradictions.",
  hipporag2:
    "HippoRAG 2 (Gutierrez et al., ICML 2025): hippocampus-inspired memory. Haiku open-IE extracts (subject, predicate, object) triples per session into a networkx graph. Synonym edges link near-duplicate entities. At recall, query embeddings seed personalized PageRank over the graph; passages are scored by PPR mass on their entities.",
  amem:
    "A-MEM (NeurIPS 2025, Xu et al.): the LLM emits structured memory notes (content + keywords + context + tags) per session, embeds them, and a periodic link-evolution pass adds directed edges between related notes. Retrieval fans out from each top-k hit to its linked neighbors.",
  qiss:
    "Quantum-Inspired Semantic Similarity: reranks naive_vector's candidates by quantum-state fidelity Tr(rho_q rho_d) = cosine squared over the same embeddings. Pure NumPy, no new deps. An optional multi-query superposition mode adds interference cross-terms that classical rank fusion cannot express.",
  sqr:
    "Simulated Quantum Reranker: runs a real SWAP-test circuit on the Qiskit Aer simulator to estimate each query-document overlap. Embeddings are PCA-reduced to 2^n_qubits dims, amplitude-encoded, and compared in exact statevector mode. The accuracy cost of that dimensionality reduction is reported honestly.",
  mem0:
    "Mem0 SDK (v2), leveled to the harness: Chroma vector store, OpenAI embeddings, Anthropic Sonnet for fact extraction during ingest (same model as the pure-Python baselines).",
  graphiti:
    "Zep's Graphiti: temporal knowledge graph with valid_at/invalid_at edges. Each session ingests as one episode. Searches return time-aware facts.",
  graphiti_falkor:
    "Zep's Graphiti running on FalkorDB (a Redis-based, GraphBLAS graph engine) instead of Neo4j. Identical ingest and recall to graphiti; the graph database is the only variable, so the latency and cost deltas isolate the engine and put FalkorDB's headline latency claims to an independent test.",
  cognee:
    "Open-source knowledge-graph memory. Cognify extracts entities/relationships into a graph; search answers via the graph as context.",
  langmem:
    "LangChain's memory store: extracts facts as conversations happen, stores them in LangGraph's InMemoryStore, recalls by similarity.",
  memori:
    "SQL-native agent memory. Stores extracted facts in Postgres for SQL-style recall. Memori 3.x with autocommit BYODB conn.",
};

export const CATEGORIES = [
  "information_extraction",
  "multi_session_reasoning",
  "temporal",
  "knowledge_update",
] as const;

export type Category = (typeof CATEGORIES)[number];

export const CATEGORY_INFO: Record<
  Category,
  { label: string; description: string }
> = {
  information_extraction: {
    label: "Information Extraction",
    description:
      "Single-fact recall from one session. Example: 'What programming language did the user say they prefer?'",
  },
  multi_session_reasoning: {
    label: "Multi-Session Reasoning",
    description:
      "Combine facts from 2+ sessions to answer. Example: 'Given the user's job and city, what is their commute like?'",
  },
  temporal: {
    label: "Temporal",
    description:
      "Time-aware queries that require knowing when something was said. Example: 'When did the user first mention mountain biking?'",
  },
  knowledge_update: {
    label: "Knowledge Update",
    description:
      "User changed their mind across sessions. Test: does the system reflect the latest version, not an earlier one?",
  },
};

export interface CorpusInfo {
  name: string;
  label: string;
  count?: number;
}

export const DEFAULT_CORPORA: CorpusInfo[] = [
  { name: "longmemeval-s", label: "LongMemEval-S", count: 16 },
];

export const CORPORA = DEFAULT_CORPORA;

export async function fetchCorpora(): Promise<CorpusInfo[]> {
  try {
    const res = await fetch(`${API_URL}/api/corpora`);
    if (!res.ok) return DEFAULT_CORPORA;
    const data = await res.json();
    return data.corpora?.length ? data.corpora : DEFAULT_CORPORA;
  } catch {
    return DEFAULT_CORPORA;
  }
}

export interface BenchmarkRow {
  strategy: Strategy | string;
  accuracy: number;
  mean_session_recall_at_k: number;
  mean_session_hit_at_k?: number;
  avg_recall_latency_ms: number;
  total_cost_usd: number;
  // Per-category metrics are null when no question of that category was
  // evaluated. The dashboard renders "—" for null.
  abstention_f1: number | null;
  abstention_n: number;
  update_precision: number | null;
  update_n: number;
  temporal_correctness: number | null;
  temporal_n: number;
  questions_evaluated?: number;
  errors?: number;
  run_id?: string;
  accuracy_by_category?: Record<string, { accuracy: number; n: number }>;
}

export async function fetchBenchmarkResults(
  corpus: string = "longmemeval-s"
): Promise<BenchmarkRow[]> {
  try {
    const res = await fetch(`${API_URL}/api/benchmark/${corpus}`);
    if (!res.ok) return MOCK_BENCHMARK_DATA;
    const data = await res.json();
    return data.results?.length ? data.results : MOCK_BENCHMARK_DATA;
  } catch {
    return MOCK_BENCHMARK_DATA;
  }
}

export interface Source {
  title: string;
  url?: string;
}

export const MOCK_BENCHMARK_DATA: BenchmarkRow[] = [
  {
    strategy: "naive_vector",
    accuracy: 0.4,
    mean_session_recall_at_k: 0.89,
    avg_recall_latency_ms: 3548,
    total_cost_usd: 0.087,
    abstention_f1: null,
    abstention_n: 0,
    update_precision: null,
    update_n: 0,
    temporal_correctness: null,
    temporal_n: 0,
    questions_evaluated: 16,
  },
];

// IR metrics computed by memory_arena.benchmark.recall_metrics. The dashboard
// reads session_hit_at_k to render HIT/MISS in the Recall Lab.
export interface RecallIR {
  k: number;
  session_hit_at_k: number;
  session_recall_at_k: number;
  session_precision_at_k: number;
  session_mrr: number;
  session_ndcg_at_k: number;
  turn_hit_at_k: number;
  turn_recall_at_k: number;
}

export interface RecallRecord {
  question_id: string;
  category: string;
  answer?: string;
  // The strategy's retrieved supporting session ids — not the gold-truth
  // expected ones. Expected supporting ids live in the question file and
  // aren't echoed back in recall_records.
  supporting_session_ids: string[];
  supporting_turn_ids?: string[];
  ir?: RecallIR | null;
  recall_at_k_measurable?: boolean | null;
  latency_ms?: number;
  cost_usd?: number;
  error?: boolean;
}

export interface RecallRecordsResponse {
  corpus: string;
  strategy: string;
  recall_at_k_measurable: boolean | null;
  top_k: number | null;
  records: RecallRecord[];
}

export async function fetchRecallRecords(
  corpus: string,
  strategy: string
): Promise<RecallRecordsResponse | null> {
  try {
    const res = await fetch(`${API_URL}/api/recall-records/${corpus}/${strategy}`);
    if (!res.ok) return null;
    return await res.json();
  } catch {
    return null;
  }
}
