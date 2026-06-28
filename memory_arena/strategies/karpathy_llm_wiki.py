"""Karpathy's LLM Wiki strategy.

Three-layer architecture (https://gist.github.com/karpathy/442a6bf555914893e9891c11519de94f):

- Raw sources: chat sessions (immutable).
- The wiki: LLM-maintained markdown pages with [[wikilinks]]. Each page is one
  entity / topic. Claims cite the source session via ``[session=<id>]`` markers.
- The schema: encoded in the ingest / select / lint system prompts below.

Three workflows:

- Ingest: LLM reads the session, emits JSON ops to create new entity pages or
  append to existing ones, and an entry is appended to the running log.
- Recall: LLM reads the index + recent log, picks pages to load, then synthesizes
  an answer with citations.
- Lint: every ``LINT_EVERY`` ingested sessions, the LLM scans the page set and
  emits rewrites/merges to resolve contradictions, deduplicate entities, and
  drop stale claims.

Cost is heavier than naive_vector because every session triggers an LLM ingest
call, and every recall is a two-stage LLM exchange (page selection then answer).
The expected payoff is on knowledge_update and multi_session_reasoning, where
consolidating facts onto a page that gets edited as the user changes their mind
is structurally a better fit than dropping every turn into a vector store.
"""

from __future__ import annotations

import json
import logging
import re
import shutil
import time
from pathlib import Path
from tempfile import gettempdir

from memory_arena.llm.client import LLMClient
from memory_arena.sessions.schema import Session
from memory_arena.strategies.base import IngestRecord, MemoryStrategy, RecallResult

logger = logging.getLogger(__name__)

_INGEST_SYSTEM = (
    "You are maintaining a wiki of entity pages built from chat sessions.\n"
    "Page names are lowercase-kebab-case (e.g. user-job, paris-trip, "
    "python-preference).\n"
    "Use [[other-page]] to cross-link related entities. Cite sources as "
    "[session=<id>] at the end of every claim.\n\n"
    "Output ONLY a JSON array of operations. No prose, no markdown fences.\n"
    "[\n"
    '  {"op": "create", "page": "<name>", "content": "<markdown body>"},\n'
    '  {"op": "append", "page": "<name>", "content": "<one-line update with citation>"}\n'
    "]\n\n"
    "Rules:\n"
    "- Use create when the entity does not yet have a page.\n"
    "- Use append when the entity already has a page (just add the new fact).\n"
    "- Every claim must end with a [session=<id>] citation.\n"
    "- 1-3 sentences per fact. Keep it tight.\n"
    "- Skip the session if it contains nothing worth remembering."
)

_SELECT_SYSTEM = (
    "You are choosing which wiki pages to read to answer a query.\n"
    "Output ONLY a JSON array of page names (max 5). No prose, no fences."
)

_RECALL_SYSTEM = (
    "You are an assistant answering using a wiki of entity pages from a chat history. "
    "Use only information that appears in the retrieved pages. Cite sessions by "
    "their id in square brackets. If the pages do not contain the answer, say "
    '"I do not have that information."'
)

_LINT_SYSTEM = (
    "You are linting a wiki for contradictions, duplicate entities, and stale "
    "claims that have been superseded.\n\n"
    "Output ONLY a JSON array of fixes. Empty array if everything looks fine.\n"
    "[\n"
    '  {"op": "rewrite", "page": "<name>", "content": "<full corrected body>",'
    ' "reason": "<short>"},\n'
    '  {"op": "merge", "from": "<page>", "into": "<page>", "reason": "<short>"}\n'
    "]\n\n"
    "Rules:\n"
    "- Newer claim wins on knowledge updates. Prefer rewriting the page to "
    "  reflect the latest version while preserving all [session=<id>] citations.\n"
    "- Merge when two pages clearly describe the same entity.\n"
    "- Do not invent facts. Only consolidate or correct what is already on the pages.\n"
    "- Keep [[wikilinks]] intact."
)


_PAGE_NAME_RE = re.compile(r"[^a-z0-9_-]+")
_SESSION_CITE_RE = re.compile(r"\[session=([^\]]+)\]")


def _safe_parse_json_list(text: str) -> list:
    """Parse a JSON array tolerantly: strip markdown fences, find outermost brackets."""
    if not text:
        return []
    text = text.strip()
    if text.startswith("```"):
        text = text.split("```", 2)[1]
        if text.lstrip().startswith("json"):
            text = text.split("\n", 1)[1] if "\n" in text else ""
        text = text.strip("` \n")
    i = text.find("[")
    j = text.rfind("]")
    if i == -1 or j == -1 or j < i:
        return []
    try:
        parsed = json.loads(text[i : j + 1])
        return parsed if isinstance(parsed, list) else []
    except json.JSONDecodeError:
        return []


def _sanitize_page_name(raw: str) -> str:
    name = _PAGE_NAME_RE.sub("-", str(raw).lower()).strip("-")
    return name[:80]  # cap length so we don't write absurd file names


class KarpathyLlmWikiStrategy(MemoryStrategy):
    """LLM-maintained markdown wiki with periodic lint pass."""

    name = "karpathy_llm_wiki"

    LINT_EVERY = 10  # lint after every N ingested sessions

    def __init__(self) -> None:
        super().__init__()
        self._wiki_dir: Path | None = None
        self._llm: LLMClient | None = None
        self._index_lines: list[str] = []
        self._log_lines: list[str] = []
        self._sessions_since_lint = 0
        self._errors: list[dict] = []

    async def setup(self, run_id: str) -> None:
        self.run_id = run_id
        base = Path(gettempdir()) / "memory_arena_karpathy_wiki" / run_id
        if base.exists():
            shutil.rmtree(base, ignore_errors=True)
        (base / "pages").mkdir(parents=True, exist_ok=True)
        self._wiki_dir = base
        self._llm = LLMClient()
        self._index_lines = []
        self._log_lines = []
        self._sessions_since_lint = 0
        self._errors = []

    def _existing_pages(self) -> list[str]:
        if self._wiki_dir is None:
            return []
        return sorted(p.stem for p in (self._wiki_dir / "pages").glob("*.md"))

    def _read_page(self, name: str) -> str:
        if self._wiki_dir is None:
            return ""
        p = self._wiki_dir / "pages" / f"{name}.md"
        return p.read_text() if p.exists() else ""

    def _write_or_append(self, raw_name: str, content: str) -> str:
        """Write a new page or append to an existing one. Returns sanitized name."""
        if self._wiki_dir is None:
            return ""
        name = _sanitize_page_name(raw_name)
        if not name:
            return ""
        p = self._wiki_dir / "pages" / f"{name}.md"
        body = content.strip()
        if not body:
            return ""
        if p.exists():
            with p.open("a") as f:
                f.write(f"\n{body}\n")
        else:
            p.write_text(f"# {name}\n\n{body}\n")
            preview = body[:120].replace("\n", " ")
            self._index_lines.append(f"- [[{name}]]: {preview}")
        return name

    def _format_session(self, session: Session) -> str:
        body_lines = [f"Session {session.id} ({session.timestamp or 'unknown'}):"]
        for turn in session.turns:
            body_lines.append(f"{turn.role}: {turn.content}")
        return "\n".join(body_lines)

    async def ingest_session(self, session: Session) -> IngestRecord:
        if self._llm is None or self._wiki_dir is None:
            raise RuntimeError("setup() not called")
        start = self._start_timer()
        cost = 0.0
        tokens = 0
        err = ""
        ops_count = 0

        existing = self._existing_pages()
        existing_str = ", ".join(existing) if existing else "(none)"
        body = self._format_session(session)
        prompt = (
            f"Existing wiki pages: {existing_str}\n\n"
            f"{body}\n\n"
            "Return the JSON array of operations to create or update entity pages."
        )

        try:
            resp = await self._llm.generate(prompt, "", _INGEST_SYSTEM)
            cost += resp.cost_usd
            tokens += resp.total_tokens
            ops = _safe_parse_json_list(resp.text)
            page_names: list[str] = []
            for op in ops:
                if not isinstance(op, dict):
                    continue
                page = op.get("page", "")
                content = op.get("content", "")
                if not page or not content:
                    continue
                # Make sure the citation is present so recall can recover the session id.
                if "[session=" not in content:
                    content = f"{content.rstrip('.').rstrip()} [session={session.id}]"
                written = self._write_or_append(page, content)
                if written:
                    page_names.append(written)
            ops_count = len(page_names)
            ts = session.timestamp or "-"
            self._log_lines.append(f"## [{ts}] ingest | session={session.id} | pages={page_names}")
        except Exception as exc:
            err = f"ingest: {exc}"
            logger.warning(
                "strategy=%s ingest session=%s failed: %s",
                self.name,
                session.id,
                exc,
            )
            self._errors.append(
                {
                    "phase": "ingest",
                    "session_id": session.id,
                    "error": str(exc),
                    "type": type(exc).__name__,
                }
            )

        # Periodic lint pass to resolve contradictions and merge duplicates.
        self._sessions_since_lint += 1
        if not err and self._sessions_since_lint >= self.LINT_EVERY:
            try:
                lint_cost, lint_tokens = await self._lint()
                cost += lint_cost
                tokens += lint_tokens
            except Exception as lint_exc:
                err = f"lint: {lint_exc}"
                logger.warning(
                    "strategy=%s lint after session=%s failed: %s",
                    self.name,
                    session.id,
                    lint_exc,
                )
                self._errors.append(
                    {
                        "phase": "ingest",
                        "step": "lint",
                        "session_id": session.id,
                        "error": str(lint_exc),
                        "type": type(lint_exc).__name__,
                    }
                )
            self._sessions_since_lint = 0

        elapsed = (self._start_timer() - start) * 1000
        return IngestRecord(
            session_id=session.id,
            latency_ms=elapsed,
            tokens_used=tokens,
            cost_usd=cost,
            facts_extracted=ops_count,
            error=err,
        )

    async def _lint(self) -> tuple[float, int]:
        """Scan the wiki and apply rewrite/merge fixes. Returns (cost, tokens)."""
        if self._llm is None or self._wiki_dir is None:
            return 0.0, 0
        pages = self._existing_pages()
        if len(pages) < 2:
            return 0.0, 0
        # Cap the lint context: the 30 largest pages are most likely to harbor
        # contradictions. (Sorting by size keeps cost bounded.)
        sample = sorted(
            pages,
            key=lambda n: (self._wiki_dir / "pages" / f"{n}.md").stat().st_size,
            reverse=True,
        )[:30]
        body_parts = [f"=== [[{name}]] ===\n{self._read_page(name)}" for name in sample]
        body = "\n\n".join(body_parts)

        resp = await self._llm.generate(
            f"Wiki pages:\n{body}\n\nReturn the JSON array of fixes.",
            "",
            _LINT_SYSTEM,
        )
        fixes = _safe_parse_json_list(resp.text)
        for fix in fixes:
            if not isinstance(fix, dict):
                continue
            op = fix.get("op", "")
            if op == "rewrite":
                page_name = _sanitize_page_name(fix.get("page", ""))
                content = (fix.get("content") or "").strip()
                if not page_name or not content:
                    continue
                page_path = self._wiki_dir / "pages" / f"{page_name}.md"
                if page_path.exists():
                    page_path.write_text(content + "\n")
            elif op == "merge":
                src = _sanitize_page_name(fix.get("from", ""))
                dst = _sanitize_page_name(fix.get("into", ""))
                if not src or not dst or src == dst:
                    continue
                src_path = self._wiki_dir / "pages" / f"{src}.md"
                dst_path = self._wiki_dir / "pages" / f"{dst}.md"
                if src_path.exists() and dst_path.exists():
                    with dst_path.open("a") as f:
                        f.write(f"\n\n<!-- merged from [[{src}]] -->\n")
                        f.write(src_path.read_text())
                    src_path.unlink()
                    # drop the orphan from the index
                    self._index_lines = [
                        line for line in self._index_lines if f"[[{src}]]" not in line
                    ]
        return resp.cost_usd, resp.total_tokens

    async def recall(self, query: str, top_k: int = 10) -> RecallResult:
        if self._llm is None or self._wiki_dir is None:
            raise RuntimeError("setup() not called")
        start = self._start_timer()
        cost = 0.0
        tokens = 0

        # Stage 1: Haiku-class call to pick which pages to read.
        index_str = "\n".join(self._index_lines) or "(empty)"
        log_tail = "\n".join(self._log_lines[-20:]) or "(empty)"
        existing = self._existing_pages()
        existing_str = ", ".join(existing) if existing else "(none)"
        select_prompt = (
            f"Wiki index:\n{index_str}\n\n"
            f"Recent log:\n{log_tail}\n\n"
            f"Available pages: {existing_str}\n\n"
            f"Query: {query}\n\n"
            "Which pages should I read?"
        )
        page_names: list[str] = []
        try:
            select_resp = await self._llm.generate(select_prompt, "", _SELECT_SYSTEM)
            cost += select_resp.cost_usd
            tokens += select_resp.total_tokens
            picked = _safe_parse_json_list(select_resp.text)
            page_names = [_sanitize_page_name(p) for p in picked if isinstance(p, str)][:5]
        except Exception as exc:
            logger.warning("strategy=%s recall page-selection failed: %s", self.name, exc)
            self._errors.append(
                {
                    "phase": "recall",
                    "step": "page_selection",
                    "error": str(exc),
                    "type": type(exc).__name__,
                }
            )
            page_names = []
        if not page_names:
            page_names = existing[: min(5, top_k)]

        retrieval_start = time.perf_counter()
        contexts: list[dict] = []
        cited_session_ids: set[str] = set()
        for name in page_names:
            content = self._read_page(name)
            if content:
                contexts.append({"page": name, "content": content})
                cited_session_ids.update(_SESSION_CITE_RE.findall(content))
        retrieval_ms = (time.perf_counter() - retrieval_start) * 1000

        retrieved_block = "\n\n".join(f"=== [[{c['page']}]] ===\n{c['content']}" for c in contexts)

        gen_start = time.perf_counter()
        try:
            gen_resp = await self._llm.generate(query, retrieved_block, _RECALL_SYSTEM)
            cost += gen_resp.cost_usd
            tokens += gen_resp.total_tokens
            answer = gen_resp.text
        except Exception as exc:
            logger.warning("strategy=%s recall generation failed: %s", self.name, exc)
            self._errors.append(
                {
                    "phase": "recall",
                    "step": "generation",
                    "error": str(exc),
                    "type": type(exc).__name__,
                }
            )
            answer = f"[ERROR] generation failed: {exc}"
        gen_ms = (time.perf_counter() - gen_start) * 1000

        latency = (self._start_timer() - start) * 1000
        return RecallResult(
            answer=answer,
            supporting_session_ids=sorted(cited_session_ids),
            supporting_turn_ids=[],
            retrieved_memories=contexts,
            strategy=self.name,
            latency_ms=latency,
            retrieval_latency_ms=retrieval_ms,
            generation_latency_ms=gen_ms,
            tokens_used=tokens,
            cost_usd=cost,
        )

    async def teardown(self) -> None:
        if self._wiki_dir is not None and self._wiki_dir.exists():
            shutil.rmtree(self._wiki_dir, ignore_errors=True)
        self._wiki_dir = None
        self._llm = None
