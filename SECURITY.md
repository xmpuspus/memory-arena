# Security Policy

## Supported Versions

| Version | Supported |
|---------|-----------|
| 0.1.x   | Yes       |

## Reporting a Vulnerability

If you discover a security vulnerability, please report it responsibly:

1. **Do not** open a public GitHub issue
2. Email security concerns to xavier@xmpuspus.dev
3. Include a description of the vulnerability, steps to reproduce, and potential impact
4. You'll receive a response within 48 hours

## Security Model

### API Keys

- All API keys are loaded from environment variables via `pydantic-settings`
- Keys are never logged, serialized to disk, or included in error messages
- The `MEM_ARENA_DEBUG=false` default ensures production error responses are generic

### Network

- CORS is configured with explicit allowed origins - never `*` in production
- Neo4j and Postgres bind to `127.0.0.1` only by default; never exposed to all interfaces
- `MEM_ARENA_NEO4J_PASSWORD` is required (no default fallback), docker-compose refuses to start without it

### Database

- All Neo4j queries use parameterized Cypher, no string interpolation
- The chatbot API uses read-only query patterns (`MATCH`/`RETURN`)
- Graph mutations only occur during the `benchmark` and `recall-lab` runs, scoped by `run_id`, and reverted by each strategy's `teardown()`

### Dependencies

Most direct dependencies are pinned to exact versions; vendor SDKs
(`cognee`, `langmem`, `memori`) and a few large libs (`numpy`,
`scikit-learn`) use bounded ranges to track their fast-moving releases
(e.g. `cognee>=1.0,<2.0`, `numpy>=1.26,<3.0`). The exactly-pinned
extras (`mem0ai==0.1.114`, `graphiti-core==0.13.0`,
`langchain-neo4j==0.4.0`) cover the integrations whose APIs have
broken under us in the past. We track upstream advisories on the
ranged deps and refresh on release.

### Input Validation

- All API request bodies are validated by Pydantic v2 with strict type checking
- Strategy names are validated against the registry; unknown strategies return a structured error
- Session and question records are validated against the `Session` / `QuestionRecord` Pydantic models at load time, with `ConfigDict(extra="forbid")` on response models

## Known Limitations

- The chatbot API binds to `0.0.0.0` by default in `cli.py serve`. In production, bind to `127.0.0.1` and use a reverse proxy. The bundled `docker-compose.yml` already binds to `127.0.0.1`.
- LLM responses are not sanitized for XSS before rendering in the frontend. The Next.js frontend uses React's built-in escaping, but custom integrations should sanitize output.
- **AI prompt injection in source corpora.** The LLM judge reads candidate session content end-to-end. If you point memory-arena at an untrusted corpus (e.g. user-generated chat logs or scraped data), prompt-injection payloads embedded in session text can manipulate the judge's score. We recommend running on trusted corpora only. For untrusted corpora, sanitize sessions and run with `--judge-blind` once that flag lands (v0.2).
- The benchmark issues many concurrent calls to vendor SDKs and LLM providers under default settings. Run only against your own paid keys; do not point at shared organization keys without informing the team.
