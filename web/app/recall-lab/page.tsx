"use client";

import { useEffect, useMemo, useState } from "react";
import {
  STRATEGIES,
  STRATEGY_LABELS,
  STRATEGY_COLORS,
  STRATEGY_DESCRIPTIONS,
  CORPORA,
  fetchBenchmarkResults,
  fetchRecallRecords,
  type Strategy,
  type RecallRecord,
} from "@/lib/api";

type Verdict = "HIT" | "MISS" | "N/A";

function verdict(rec: RecallRecord, measurable: boolean | null): Verdict {
  // Strategy-level not measurable (e.g. full_context dumps everything; mem0g
  // hides retrieval inside the SDK). The runner already flags these so the
  // dashboard does not pretend HIT/MISS.
  if (measurable === false) return "N/A";
  if (rec.recall_at_k_measurable === false) return "N/A";
  if (!rec.ir) return "N/A";
  return rec.ir.session_hit_at_k > 0 ? "HIT" : "MISS";
}

function fmtMs(value: number | undefined): string {
  if (value === undefined || value === null) return "—";
  if (value < 1000) return `${value.toFixed(0)}ms`;
  return `${(value / 1000).toFixed(1)}s`;
}

export default function RecallLabPage() {
  const [corpus] = useState(CORPORA[0]?.name ?? "longmemeval-s");
  const [availableStrategies, setAvailableStrategies] = useState<string[]>([]);
  const [strategy, setStrategy] = useState<string>("");
  const [data, setData] = useState<{
    recall_at_k_measurable: boolean | null;
    top_k: number | null;
    records: RecallRecord[];
  } | null>(null);
  const [loading, setLoading] = useState(false);
  const [notFound, setNotFound] = useState(false);

  // Populate strategy dropdown from the benchmark results so we only show
  // strategies that actually have a result file for this corpus.
  useEffect(() => {
    fetchBenchmarkResults(corpus).then((rows) => {
      const names = rows
        .map((r) => r.strategy as string)
        .filter((n): n is string => typeof n === "string" && n.length > 0);
      // Keep declaration order from STRATEGIES so the dropdown is stable.
      const ordered = (STRATEGIES as readonly string[]).filter((s) =>
        names.includes(s)
      );
      const fallback = ordered.length > 0 ? ordered : names;
      setAvailableStrategies(fallback);
      if (!strategy && fallback.length) {
        // Prefer naive_vector as the default since it's the most pedagogical.
        setStrategy(fallback.includes("naive_vector") ? "naive_vector" : fallback[0]);
      }
    });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [corpus]);

  useEffect(() => {
    if (!strategy) return;
    setLoading(true);
    setNotFound(false);
    fetchRecallRecords(corpus, strategy).then((res) => {
      setLoading(false);
      if (!res) {
        setData(null);
        setNotFound(true);
        return;
      }
      setData({
        recall_at_k_measurable: res.recall_at_k_measurable ?? null,
        top_k: res.top_k ?? null,
        records: res.records ?? [],
      });
    });
  }, [corpus, strategy]);

  const counts = useMemo(() => {
    if (!data) return { hit: 0, miss: 0, na: 0, total: 0 };
    let hit = 0,
      miss = 0,
      na = 0;
    for (const r of data.records) {
      const v = verdict(r, data.recall_at_k_measurable);
      if (v === "HIT") hit++;
      else if (v === "MISS") miss++;
      else na++;
    }
    return { hit, miss, na, total: data.records.length };
  }, [data]);

  const measurable = data?.recall_at_k_measurable;

  return (
    <div className="max-w-6xl mx-auto px-6 py-12 space-y-12">
      <section className="space-y-3">
        <h1
          className="text-3xl font-bold tracking-tight"
          style={{ color: "var(--foreground)" }}
        >
          Recall Lab
        </h1>
        <p
          className="text-base leading-relaxed max-w-3xl"
          style={{ color: "var(--muted)" }}
        >
          Retrieval-only view: did the strategy fetch a labelled supporting
          session inside the top-k result set, before the LLM judge ever sees
          the answer? HIT means the gold supporting session id appeared in the
          retrieved set; MISS means it did not. Cheaper to inspect than the
          full benchmark and useful for tuning top_k, embeddings, and store
          choice.
        </p>
        <div className="flex flex-wrap items-center gap-3">
          <label className="text-sm" style={{ color: "var(--muted)" }}>
            Strategy
          </label>
          <select
            className="text-sm px-3 py-1.5 rounded border"
            value={strategy}
            onChange={(e) => setStrategy(e.target.value)}
            style={{
              borderColor: "var(--border)",
              background: "var(--card)",
              color: "var(--foreground)",
            }}
          >
            {availableStrategies.length === 0 && (
              <option value="">(loading…)</option>
            )}
            {availableStrategies.map((s) => (
              <option key={s} value={s}>
                {STRATEGY_LABELS[s as Strategy] ?? s}
              </option>
            ))}
          </select>
          {data?.top_k != null && (
            <span className="text-xs" style={{ color: "var(--muted)" }}>
              top_k = {data.top_k}
            </span>
          )}
          {data && (
            <span className="text-xs" style={{ color: "var(--muted)" }}>
              {counts.total} questions · {counts.hit} HIT · {counts.miss} MISS
              {counts.na > 0 ? ` · ${counts.na} N/A` : ""}
            </span>
          )}
        </div>
        {strategy && STRATEGY_DESCRIPTIONS[strategy as Strategy] && (
          <p
            className="text-xs leading-relaxed max-w-3xl pt-1"
            style={{ color: "var(--muted)" }}
          >
            {STRATEGY_DESCRIPTIONS[strategy as Strategy]}
          </p>
        )}
      </section>

      {measurable === false && (
        <section
          className="rounded-lg border p-4"
          style={{
            borderColor: "var(--border)",
            background: "var(--card)",
            color: "var(--muted)",
          }}
        >
          <p className="text-sm">
            Recall not measurable for this strategy. The strategy either
            doesn&apos;t expose a per-question retrieved-id list (e.g.
            full_context dumps every session by design) or wraps recall inside
            a vendor SDK that doesn&apos;t surface ranking. Accuracy and the
            other end-to-end axes are still on the Benchmark page.
          </p>
        </section>
      )}

      {loading && (
        <p className="text-sm" style={{ color: "var(--muted)" }}>
          Loading…
        </p>
      )}

      {notFound && !loading && (
        <p className="text-sm" style={{ color: "var(--muted)" }}>
          No recall records on disk for {corpus}/{strategy}. Run{" "}
          <code style={{ color: "var(--foreground)" }}>
            memory-arena benchmark --corpus {corpus} --strategy {strategy}
          </code>{" "}
          first.
        </p>
      )}

      {data && data.records.length > 0 && (
        <section className="space-y-4">
          <h2
            className="text-xl font-semibold"
            style={{ color: "var(--foreground)" }}
          >
            HIT / MISS by question
          </h2>
          <div className="space-y-3">
            {data.records.map((rec) => {
              const v = verdict(rec, data.recall_at_k_measurable);
              const color =
                v === "HIT"
                  ? "var(--accent)"
                  : v === "MISS"
                    ? "var(--muted)"
                    : "var(--muted)";
              const stratColor =
                STRATEGY_COLORS[strategy as Strategy] ?? "var(--muted)";
              return (
                <div
                  key={rec.question_id}
                  className="rounded-lg border p-4 space-y-2"
                  style={{
                    borderColor:
                      v === "HIT" ? "var(--accent)" : "var(--border)",
                    background: "var(--card)",
                  }}
                >
                  <div className="flex items-baseline justify-between gap-3 flex-wrap">
                    <h3
                      className="text-sm font-mono"
                      style={{ color: "var(--foreground)" }}
                    >
                      <span
                        className="inline-block w-2 h-2 rounded-full mr-2 align-middle"
                        style={{ background: stratColor }}
                      />
                      {rec.question_id}
                    </h3>
                    <div className="flex items-center gap-2">
                      <span
                        className="text-xs px-2 py-0.5 rounded"
                        style={{
                          background: "var(--background)",
                          color: "var(--muted)",
                          border: "1px solid var(--border)",
                        }}
                      >
                        {rec.category}
                      </span>
                      <span
                        className="text-xs font-semibold tabular-nums"
                        style={{ color }}
                      >
                        {v}
                      </span>
                    </div>
                  </div>
                  <div
                    className="grid grid-cols-1 md:grid-cols-3 gap-3 text-xs"
                    style={{ color: "var(--muted)" }}
                  >
                    <div className="space-y-1">
                      <div className="font-semibold uppercase tracking-wide opacity-70">
                        Retrieved
                      </div>
                      <div
                        className="font-mono leading-relaxed break-all"
                        style={{ color: "var(--foreground)" }}
                      >
                        {rec.supporting_session_ids?.length
                          ? rec.supporting_session_ids.join(", ")
                          : "(none)"}
                      </div>
                    </div>
                    <div className="space-y-1">
                      <div className="font-semibold uppercase tracking-wide opacity-70">
                        IR
                      </div>
                      <div
                        className="tabular-nums"
                        style={{ color: "var(--foreground)" }}
                      >
                        {rec.ir
                          ? `recall@${rec.ir.k} = ${(rec.ir.session_recall_at_k * 100).toFixed(0)}%, mrr = ${rec.ir.session_mrr.toFixed(2)}`
                          : "—"}
                      </div>
                    </div>
                    <div className="space-y-1">
                      <div className="font-semibold uppercase tracking-wide opacity-70">
                        Cost / latency
                      </div>
                      <div
                        className="tabular-nums"
                        style={{ color: "var(--foreground)" }}
                      >
                        {rec.cost_usd != null
                          ? `$${rec.cost_usd.toFixed(4)}`
                          : "—"}{" "}
                        · {fmtMs(rec.latency_ms)}
                      </div>
                    </div>
                  </div>
                </div>
              );
            })}
          </div>
        </section>
      )}
    </div>
  );
}
