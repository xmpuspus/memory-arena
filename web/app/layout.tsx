import type { Metadata } from "next";
import "./globals.css";
import Nav from "@/components/Nav";
import ClientErrorBoundary from "@/components/ClientErrorBoundary";

export const metadata: Metadata = {
  title: "Memory Arena: Agent Memory Benchmark",
  description:
    "Benchmark 16 agent-memory architectures (Mem0, Graphiti, Cognee, LangMem, Memori, Karpathy LLM Wiki, plus baselines and advanced retrievers) on the same evaluator under the same configs.",
};

export default function RootLayout({
  children,
}: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en">
      <body className="min-h-screen" style={{ background: "var(--background)", color: "var(--foreground)" }}>
        <Nav />
        <main>
          <ClientErrorBoundary>{children}</ClientErrorBoundary>
        </main>
        <footer className="border-t mt-16 py-8 text-center text-sm" style={{ borderColor: "var(--border)", color: "var(--muted)" }}>
          <a
            href="https://github.com/xmpuspus/memory-arena"
            target="_blank"
            rel="noopener noreferrer"
            className="hover:opacity-70 transition-opacity"
            style={{ color: "var(--accent)" }}
          >
            GitHub
          </a>
          <span className="mx-2">·</span>
          <span>Memory Arena: Agent Memory Benchmark</span>
        </footer>
      </body>
    </html>
  );
}
