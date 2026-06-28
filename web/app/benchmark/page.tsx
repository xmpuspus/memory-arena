"use client";

import { useEffect, useMemo, useState } from "react";
import {
  STRATEGY_LABELS,
  STRATEGY_COLORS,
  CATEGORIES,
  CATEGORY_INFO,
  CORPORA,
  fetchBenchmarkResults,
  type BenchmarkRow,
  type Strategy,
} from "@/lib/api";
import InfoTip from "@/components/InfoTip";

function pct(value: number | null | undefined): string {
  if (value === null || value === undefined) return "—";
  return `${(value * 100).toFixed(1)}%`;
}

function dollar(value: number): string {
  return `$${value.toFixed(3)}`;
}

function ms(value: number): string {
  if (value < 1000) return `${value.toFixed(0)}ms`;
  return `${(value / 1000).toFixed(1)}s`;
}

function f1(value: number | null | undefined): string {
  if (value === null || value === undefined) return "—";
  return value.toFixed(2);
}

type SortKey = "accuracy" | "recall" | "latency" | "cost" | "name";

export default function BenchmarkPage() {
  const [corpus, setCorpus] = useState(CORPORA[0]?.name ?? "longmemeval-s");
  const [rows, setRows] = useState<BenchmarkRow[]>([]);
  const [sortKey, setSortKey] = useState<SortKey>("accuracy");
  const [sortDesc, setSortDesc] = useState<boolean>(true);

  useEffect(() => {
    fetchBenchmarkResults(corpus).then(setRows);
  }, [corpus]);

  const sorted = useMemo(() => {
    const copy = [...rows];
    copy.sort((a, b) => {
      const av = (() => {
        switch (sortKey) {
          case "accuracy":
            return a.accuracy ?? 0;
          case "recall":
            return a.mean_session_recall_at_k ?? 0;
          case "latency":
            return a.avg_recall_latency_ms ?? 0;
          case "cost":
            return a.total_cost_usd ?? 0;
          case "name":
            return a.strategy ?? "";
        }
      })();
      const bv = (() => {
        switch (sortKey) {
          case "accuracy":
            return b.accuracy ?? 0;
          case "recall":
            return b.mean_session_recall_at_k ?? 0;
          case "latency":
            return b.avg_recall_latency_ms ?? 0;
          case "cost":
            return b.total_cost_usd ?? 0;
          case "name":
            return b.strategy ?? "";
        }
      })();
      if (av < bv) return sortDesc ? 1 : -1;
      if (av > bv) return sortDesc ? -1 : 1;
      return 0;
    });
    return copy;
  }, [rows, sortKey, sortDesc]);

  const handleSort = (key: SortKey) => {
    if (sortKey === key) {
      setSortDesc(!sortDesc);
    } else {
      setSortKey(key);
      setSortDesc(key !== "latency" && key !== "cost"); // ascending makes sense for latency / cost
    }
  };

  const arrow = (key: SortKey) => {
    if (sortKey !== key) return "";
    return sortDesc ? " ▼" : " ▲";
  };

  return (
    <div className="max-w-6xl mx-auto px-6 py-12 space-y-12">
      <section className="space-y-3">
        <h1
          className="text-3xl font-bold tracking-tight"
          style={{ color: "var(--foreground)" }}
        >
          Benchmark
        </h1>
        <p
          className="text-base leading-relaxed max-w-3xl"
          style={{ color: "var(--muted)" }}
        >
          Each strategy ingests the same chat-session corpus, recalls answers
          to the same questions, and is scored by the same evaluator. Click a
          column header to sort. Per-category metrics show the count of
          questions in parentheses; &quot;—&quot; means no question of that category
          was evaluated.
        </p>
        <div className="flex items-center gap-3">
          <label className="text-sm" style={{ color: "var(--muted)" }}>
            Corpus
          </label>
          <select
            className="text-sm px-3 py-1.5 rounded border"
            value={corpus}
            onChange={(e) => setCorpus(e.target.value)}
            style={{
              borderColor: "var(--border)",
              background: "var(--card)",
              color: "var(--foreground)",
            }}
          >
            {CORPORA.map((c) => (
              <option key={c.name} value={c.name}>
                {c.label}
              </option>
            ))}
          </select>
          <span className="text-xs" style={{ color: "var(--muted)" }}>
            {sorted.length} strategies
          </span>
        </div>
      </section>

      <section className="space-y-4">
        <h2
          className="text-xl font-semibold"
          style={{ color: "var(--foreground)" }}
        >
          Headline metrics
        </h2>
        <div
          className="rounded-lg border overflow-hidden"
          style={{ borderColor: "var(--border)", background: "var(--card)" }}
        >
          <table className="w-full text-sm">
            <thead style={{ background: "var(--background)" }}>
              <tr>
                <th
                  className="text-left px-4 py-2.5 font-semibold cursor-pointer"
                  onClick={() => handleSort("name")}
                >
                  Strategy{arrow("name")}
                </th>
                <th
                  className="text-right px-3 py-2.5 font-semibold cursor-pointer"
                  onClick={() => handleSort("accuracy")}
                >
                  <span className="inline-flex items-center justify-end">
                    Accuracy{arrow("accuracy")}
                    <InfoTip text="LLM-judge accuracy averaged over all evaluated questions. Opus grades each answer 0..100 against the reference; the headline number reports the mean / 100." />
                  </span>
                </th>
                <th
                  className="text-right px-3 py-2.5 font-semibold cursor-pointer"
                  onClick={() => handleSort("recall")}
                >
                  <span className="inline-flex items-center justify-end">
                    Recall@5{arrow("recall")}
                    <InfoTip text="Mean session-level recall@k: did the strategy retrieve the gold-truth supporting session id inside its top-k results? Top-k held at 5 across all strategies." />
                  </span>
                </th>
                <th
                  className="text-right px-3 py-2.5 font-semibold cursor-pointer"
                  onClick={() => handleSort("latency")}
                >
                  <span className="inline-flex items-center justify-end">
                    Latency{arrow("latency")}
                    <InfoTip text="Average wall-clock recall latency per question, end-to-end (retrieval + generation). Lower is better." />
                  </span>
                </th>
                <th
                  className="text-right px-3 py-2.5 font-semibold cursor-pointer"
                  onClick={() => handleSort("cost")}
                >
                  <span className="inline-flex items-center justify-end">
                    Cost{arrow("cost")}
                    <InfoTip text="Total USD spent across the run (ingest LLM calls + recall LLM calls). Vendor-internal costs the SDK pays out-of-band are not included — see the README footnote." />
                  </span>
                </th>
                <th className="text-right px-3 py-2.5 font-semibold">
                  <span className="inline-flex items-center justify-end">
                    Questions
                    <InfoTip text="Number of questions evaluated for this strategy on this corpus. The cost cap can halt a run mid-corpus, in which case this drops below the corpus total." />
                  </span>
                </th>
              </tr>
            </thead>
            <tbody>
              {sorted.map((r) => {
                const strat = r.strategy as Strategy;
                const label = STRATEGY_LABELS[strat] ?? r.strategy;
                const color = STRATEGY_COLORS[strat] ?? "var(--muted)";
                return (
                  <tr key={r.strategy} style={{ borderTop: "1px solid var(--border)" }}>
                    <td className="px-4 py-2.5">
                      <span
                        className="inline-block w-2.5 h-2.5 rounded-full mr-2 align-middle"
                        style={{ background: color }}
                      />
                      <span style={{ color: "var(--foreground)" }}>{label}</span>
                    </td>
                    <td className="px-3 py-2.5 text-right tabular-nums font-semibold" style={{ color: "var(--foreground)" }}>
                      {pct(r.accuracy)}
                    </td>
                    <td className="px-3 py-2.5 text-right tabular-nums">
                      {pct(r.mean_session_recall_at_k)}
                    </td>
                    <td className="px-3 py-2.5 text-right tabular-nums">
                      {ms(r.avg_recall_latency_ms ?? 0)}
                    </td>
                    <td className="px-3 py-2.5 text-right tabular-nums">
                      {dollar(r.total_cost_usd ?? 0)}
                    </td>
                    <td className="px-3 py-2.5 text-right tabular-nums" style={{ color: "var(--muted)" }}>
                      {r.questions_evaluated ?? "—"}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </section>

      <section className="space-y-4">
        <h2
          className="text-xl font-semibold"
          style={{ color: "var(--foreground)" }}
        >
          Memory-specific axes
        </h2>
        <p className="text-sm" style={{ color: "var(--muted)" }}>
          These three metrics only count over questions in the relevant
          category. The number in parentheses is the question count. &quot;—&quot; means
          no question of that category was evaluated, so the metric is not
          defined.
        </p>
        <div
          className="rounded-lg border overflow-hidden"
          style={{ borderColor: "var(--border)", background: "var(--card)" }}
        >
          <table className="w-full text-sm">
            <thead style={{ background: "var(--background)" }}>
              <tr>
                <th className="text-left px-4 py-2.5 font-semibold">Strategy</th>
                <th className="text-right px-3 py-2.5 font-semibold">
                  Update Precision
                </th>
                <th className="text-right px-3 py-2.5 font-semibold">
                  Temporal Correctness
                </th>
                <th className="text-right px-3 py-2.5 font-semibold">
                  Abstention F1
                </th>
              </tr>
            </thead>
            <tbody>
              {sorted.map((r) => {
                const strat = r.strategy as Strategy;
                const label = STRATEGY_LABELS[strat] ?? r.strategy;
                const color = STRATEGY_COLORS[strat] ?? "var(--muted)";
                return (
                  <tr key={r.strategy} style={{ borderTop: "1px solid var(--border)" }}>
                    <td className="px-4 py-2.5">
                      <span
                        className="inline-block w-2.5 h-2.5 rounded-full mr-2 align-middle"
                        style={{ background: color }}
                      />
                      <span style={{ color: "var(--foreground)" }}>{label}</span>
                    </td>
                    <td className="px-3 py-2.5 text-right tabular-nums">
                      {pct(r.update_precision)}{" "}
                      <span style={{ color: "var(--muted)" }}>
                        ({r.update_n ?? 0})
                      </span>
                    </td>
                    <td className="px-3 py-2.5 text-right tabular-nums">
                      {pct(r.temporal_correctness)}{" "}
                      <span style={{ color: "var(--muted)" }}>
                        ({r.temporal_n ?? 0})
                      </span>
                    </td>
                    <td className="px-3 py-2.5 text-right tabular-nums">
                      {f1(r.abstention_f1)}{" "}
                      <span style={{ color: "var(--muted)" }}>
                        ({r.abstention_n ?? 0})
                      </span>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </section>

      <section className="space-y-4">
        <h2
          className="text-xl font-semibold"
          style={{ color: "var(--foreground)" }}
        >
          Question categories
        </h2>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {CATEGORIES.map((c) => (
            <div
              key={c}
              className="rounded-lg border p-4"
              style={{ borderColor: "var(--border)", background: "var(--card)" }}
            >
              <h3
                className="text-sm font-semibold mb-1"
                style={{ color: "var(--foreground)" }}
              >
                {CATEGORY_INFO[c].label}
              </h3>
              <p
                className="text-xs leading-relaxed"
                style={{ color: "var(--muted)" }}
              >
                {CATEGORY_INFO[c].description}
              </p>
            </div>
          ))}
        </div>
      </section>
    </div>
  );
}
