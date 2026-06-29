# Memory Arena Dashboard

Next.js 14 + Tailwind dashboard for Memory Arena. Static-exported and shipped inside the Python wheel at `memory_arena/static/`.

## Pages

- `/` - Home: 20 strategies and 4 question categories.
- `/benchmark` - Per-strategy comparison table. Pulls real numbers from `/api/benchmark/{corpus}`.
- `/recall-lab` - Per-question HIT/MISS drill-down for retrieval quality.

## Develop

```bash
cd web
npm install
npx next dev -p 3001
```

The dev server expects the FastAPI backend on port 8000 (or whatever `NEXT_PUBLIC_API_URL` points to). Without an API, pages fall back to `MOCK_BENCHMARK_DATA` from `lib/api.ts`.

## Build the Static Bundle

```bash
cd web && npx next build
cp -R out/* ../memory_arena/static/
```

The Python `memory-arena serve` command mounts that bundle at `/` and the FastAPI app on `/api/*`, so the same origin serves both.

## Style

- Cloudwright light theme via CSS custom properties in `app/globals.css`.
- Tailwind utilities for layout. No dynamic class strings (PurgeCSS would drop them).
- No em-dashes in user-facing copy.
