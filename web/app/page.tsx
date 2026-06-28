"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import {
  STRATEGIES,
  STRATEGY_LABELS,
  STRATEGY_COLORS,
  STRATEGY_DESCRIPTIONS,
  CATEGORIES,
  CATEGORY_INFO,
  CORPORA,
  fetchCorpora,
  type CorpusInfo,
  type Strategy,
} from "@/lib/api";

function StrategyCard({
  label,
  desc,
  color,
}: {
  label: string;
  desc: string;
  color: string;
}) {
  return (
    <div
      className="rounded-lg border p-4 flex flex-col gap-2"
      style={{
        borderColor: "var(--border)",
        background: "var(--card)",
        boxShadow: "0 1px 3px rgba(0,0,0,0.06)",
      }}
    >
      <div className="flex items-center gap-2">
        <span className="w-2.5 h-2.5 rounded-full shrink-0" style={{ background: color }} />
        <h3 className="text-sm font-semibold" style={{ color: "var(--foreground)" }}>
          {label}
        </h3>
      </div>
      <p className="text-xs leading-relaxed" style={{ color: "var(--muted)" }}>{desc}</p>
    </div>
  );
}

export default function Home() {
  const [corpora, setCorpora] = useState<CorpusInfo[]>(CORPORA);

  useEffect(() => {
    fetchCorpora().then(setCorpora);
  }, []);

  return (
    <div className="max-w-5xl mx-auto px-6 py-12 space-y-16">
      <section className="space-y-4">
        <h1 className="text-3xl font-bold tracking-tight" style={{ color: "var(--foreground)" }}>
          Memory Arena
        </h1>
        <p className="text-lg leading-relaxed max-w-3xl" style={{ color: "var(--muted)" }}>
          Which memory architecture works best for your agent? Sixteen
          systems (Mem0, Graphiti, Cognee, LangMem, Memori, plus
          pure-Python baselines and advanced retrievers like HyDE, RAPTOR,
          Reflection, and Karpathy&apos;s LLM Wiki) run on the same
          LongMemEval corpus, scored by the same evaluator. Empirical
          evidence, not vendor-tuned numbers.
        </p>
        <div className="flex gap-3 pt-2 flex-wrap">
          <Link
            href="/benchmark"
            className="px-4 py-2 rounded-lg text-sm font-medium transition-opacity hover:opacity-80"
            style={{ background: "var(--accent)", color: "#fff" }}
          >
            View benchmarks
          </Link>
          <Link
            href="/recall-lab"
            className="px-4 py-2 rounded-lg text-sm font-medium border transition-opacity hover:opacity-80"
            style={{ borderColor: "var(--border)", color: "var(--foreground)" }}
          >
            Recall lab
          </Link>
          <Link
            href="/benchmark"
            className="px-4 py-2 rounded-lg text-sm font-medium border transition-opacity hover:opacity-80"
            style={{ borderColor: "var(--border)", color: "var(--foreground)" }}
          >
            Leaderboard
          </Link>
          <a
            href="https://github.com/xmpuspus/memory-arena"
            target="_blank"
            rel="noopener noreferrer"
            className="px-4 py-2 rounded-lg text-sm font-medium border transition-opacity hover:opacity-80"
            style={{ borderColor: "var(--border)", color: "var(--muted)" }}
          >
            GitHub
          </a>
        </div>
        {corpora.length > 0 && (
          <p className="text-xs pt-2" style={{ color: "var(--muted)" }}>
            Active corpora: {corpora.map((c) => c.label).join(", ")}
          </p>
        )}
      </section>

      <section className="space-y-4">
        <h2 className="text-xl font-semibold" style={{ color: "var(--foreground)" }}>
          How it works
        </h2>
        <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
          {[
            {
              step: "1",
              title: "Same corpus",
              desc: "All 16 strategies ingest the LongMemEval-S sessions in the same order. Each is namespaced by run id so concurrent runs do not contaminate each other.",
            },
            {
              step: "2",
              title: "7-axis evaluator",
              desc: "Structural checks, source attribution, LLM judge (Opus), plus three memory-specific axes: temporal correctness, update precision, and abstention F1.",
            },
            {
              step: "3",
              title: "Honest cost",
              desc: "Ingest-side LLM calls (entity extraction, summarization) are tracked separately from recall cost. Most vendor numbers hide ingest cost. Memory Arena does not.",
            },
          ].map((item) => (
            <div
              key={item.step}
              className="rounded-lg border p-4 space-y-2"
              style={{ borderColor: "var(--border)", background: "var(--card)" }}
            >
              <span
                className="inline-flex items-center justify-center w-6 h-6 rounded-full text-xs font-bold"
                style={{ background: "var(--accent)", color: "#fff" }}
              >
                {item.step}
              </span>
              <h3 className="text-sm font-semibold" style={{ color: "var(--foreground)" }}>
                {item.title}
              </h3>
              <p className="text-xs leading-relaxed" style={{ color: "var(--muted)" }}>
                {item.desc}
              </p>
            </div>
          ))}
        </div>
      </section>

      <section className="space-y-4">
        <h2 className="text-xl font-semibold" style={{ color: "var(--foreground)" }}>
          The 16 strategies
        </h2>
        <p className="text-sm" style={{ color: "var(--muted)" }}>
          Three baselines, seven advanced retrieval patterns (BM25, Hybrid RRF,
          HyDE, Persona Profile, Reflection, RAPTOR, Karpathy&apos;s LLM Wiki),
          and six vendor SDKs (Mem0, Mem0+Graph, Graphiti, Cognee, LangMem,
          Memori).
        </p>
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
          {STRATEGIES.map((s: Strategy) => (
            <StrategyCard
              key={s}
              label={STRATEGY_LABELS[s]}
              desc={STRATEGY_DESCRIPTIONS[s]}
              color={STRATEGY_COLORS[s]}
            />
          ))}
        </div>
      </section>

      <section className="space-y-4">
        <h2 className="text-xl font-semibold" style={{ color: "var(--foreground)" }}>
          The 5 question categories
        </h2>
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
          {CATEGORIES.map((c) => (
            <div
              key={c}
              className="rounded-lg border p-4"
              style={{ borderColor: "var(--border)", background: "var(--card)" }}
            >
              <h3 className="text-sm font-semibold mb-1" style={{ color: "var(--foreground)" }}>
                {CATEGORY_INFO[c].label}
              </h3>
              <p className="text-xs leading-relaxed" style={{ color: "var(--muted)" }}>
                {CATEGORY_INFO[c].description}
              </p>
            </div>
          ))}
        </div>
      </section>
    </div>
  );
}
