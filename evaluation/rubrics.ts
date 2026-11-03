/**
 * rubrics.ts — Scoring functions of doc/rubrics.md and the markdown report.
 */

export const WEIGHTS = { relevance: 0.3, precision: 0.35, completeness: 0.2, format: 0.15 };
export const THRESHOLDS = { global: 70, relevance: 60, precision: 65 };

// Phrases that indicate the assistant refused although data was available
const REJECTION_PHRASES = [
  "don't have enough information",
  "do not have enough information",
  "couldn't find",
  "could not find",
  "no information",
  "no tengo información",
  "no encuentro",
  "no dispongo",
  "no hay datos",
  "no se encontraron",
];

export interface TestCase {
  id: string;
  category: string | null;
  question: string;
  expected_keywords: string[];
  expected_docs: string[];
}

export interface Scores {
  relevance: number;
  precision: number;
  completeness: number;
  format: number;
  global: number;
}

export interface CaseResult {
  id: string;
  question: string;
  category: string | null;
  answer: string;
  retrievedTitles: string[];
  topScore: number;
  elapsed: number;
  error: string | null;
  scores: Scores;
}

/** Rubric 1: share of expected documents among the retrieved titles (0-1). */
export function evalRelevance(retrievedTitles: string[], expectedDocs: string[]): number {
  if (!expectedDocs.length) return 1;
  const found = expectedDocs.filter((exp) =>
    retrievedTitles.some((t) => t.toLowerCase().includes(exp.toLowerCase())),
  ).length;
  return found / expectedDocs.length;
}

/** Rubric 2: share of expected keywords present in the answer (0-1). */
export function evalPrecision(answer: string, expectedKeywords: string[]): number {
  if (!expectedKeywords.length) return 1;
  const lower = answer.toLowerCase();
  return expectedKeywords.filter((kw) => lower.includes(kw.toLowerCase())).length / expectedKeywords.length;
}

/** Rubric 3: answers without wrongly refusing, with a reasonable length (0-1). */
export function evalCompleteness(answer: string, hasExpectedData: boolean): number {
  const lower = answer.toLowerCase();
  if (hasExpectedData && REJECTION_PHRASES.some((p) => lower.includes(p))) return 0;
  const length = answer.trim().length;
  if (length < 30) return 0.25;
  return length >= 100 ? 1 : 0.75;
}

/** Rubric 4: language of the question (English, Spanish accepted), length, structure (0-1). */
export function evalFormat(answer: string): number {
  const words = answer.toLowerCase().split(/\s+/);
  const english = ["the", "and", "of", "in", "is", "with", "for", "to", "a"];
  const spanish = ["de", "el", "la", "en", "que", "es", "con", "por"];
  let score = 0;
  if ([english, spanish].some((ind) => ind.filter((w) => words.includes(w)).length >= 3)) score += 0.5;
  const length = answer.trim().length;
  if (length >= 50 && length <= 2000) score += 0.25;
  if (answer.includes(".") && answer.includes(",")) score += 0.25;
  return Math.min(score, 1);
}

const pct = (x: number) => Math.round(x * 1000) / 10;

export function score(test: TestCase, answer: string, retrievedTitles: string[]): Scores {
  const relevance = evalRelevance(retrievedTitles, test.expected_docs);
  const precision = evalPrecision(answer, test.expected_keywords);
  const completeness = evalCompleteness(answer, test.expected_docs.length > 0);
  const format = evalFormat(answer);
  const global =
    relevance * WEIGHTS.relevance +
    precision * WEIGHTS.precision +
    completeness * WEIGHTS.completeness +
    format * WEIGHTS.format;
  return {
    relevance: pct(relevance),
    precision: pct(precision),
    completeness: pct(completeness),
    format: pct(format),
    global: pct(global),
  };
}

export function averages(results: CaseResult[]): Scores {
  const avg = (k: keyof Scores) => results.reduce((s, r) => s + r.scores[k], 0) / (results.length || 1);
  return {
    relevance: avg("relevance"),
    precision: avg("precision"),
    completeness: avg("completeness"),
    format: avg("format"),
    global: avg("global"),
  };
}

/** Threshold violations from doc/rubrics.md (empty = pass). */
export function failures(results: CaseResult[]): string[] {
  const avg = averages(results);
  const out: string[] = [];
  if (avg.global < THRESHOLDS.global) out.push(`Global accuracy ${avg.global.toFixed(1)}% < ${THRESHOLDS.global}%`);
  if (avg.relevance < THRESHOLDS.relevance) out.push(`Avg. relevance ${avg.relevance.toFixed(1)}% < ${THRESHOLDS.relevance}%`);
  if (avg.precision < THRESHOLDS.precision) out.push(`Avg. precision ${avg.precision.toFixed(1)}% < ${THRESHOLDS.precision}%`);
  const zero = results.filter((r) => r.scores.precision === 0).map((r) => r.id);
  if (zero.length) out.push(`Questions with 0% precision: ${zero.join(", ")}`);
  const errors = results.filter((r) => r.error).map((r) => r.id);
  if (errors.length) out.push(`Questions with errors: ${errors.join(", ")}`);
  return out;
}

export function report(results: CaseResult[], meta: { url: string; provider: string; elapsed: number }): string {
  const avg = averages(results);
  const passed = results.filter((r) => r.scores.global >= THRESHOLDS.global).length;
  const lines = [
    "# RAG System Evaluation Report",
    "",
    `**Date**: ${new Date().toISOString().replace("T", " ").slice(0, 19)} UTC`,
    `**Target**: ${meta.url}`,
    `**LLM**: ${meta.provider}`,
    `**Tests executed**: ${results.length}`,
    `**Total time**: ${meta.elapsed.toFixed(1)}s`,
    "",
    "## Accuracy Summary",
    "",
    "| Metric | Value | Threshold |",
    "|---|---|---|",
    `| **Global Accuracy** | **${avg.global.toFixed(1)}%** | >= ${THRESHOLDS.global}% |`,
    `| Avg. Relevance | ${avg.relevance.toFixed(1)}% | >= ${THRESHOLDS.relevance}% |`,
    `| Avg. Precision | ${avg.precision.toFixed(1)}% | >= ${THRESHOLDS.precision}% |`,
    `| Avg. Completeness | ${avg.completeness.toFixed(1)}% | - |`,
    `| Avg. Format | ${avg.format.toFixed(1)}% | - |`,
    `| Tests passed (>= ${THRESHOLDS.global}%) | ${passed}/${results.length} | |`,
    "",
    "## Detail by Test",
    "",
    "| ID | Category | Relevance | Precision | Completeness | Format | **Global** | Time |",
    "|---|---|---|---|---|---|---|---|",
    ...results.map((r) => {
      const s = r.scores;
      const status = s.global >= THRESHOLDS.global ? "✅" : "❌";
      return (
        `| ${r.id} | ${r.category ?? "all"} | ${s.relevance}% | ${s.precision}% | ${s.completeness}% | ` +
        `${s.format}% | **${s.global}%** ${status} | ${r.elapsed.toFixed(1)}s |`
      );
    }),
    "",
    "## Response Details",
    "",
  ];
  for (const r of results) {
    lines.push(
      `### ${r.id}: ${r.question}`,
      "",
      `- **Global score**: ${r.scores.global}%`,
      `- **Top retrieval score**: ${r.topScore.toFixed(4)}`,
      `- **Retrieved docs**: ${r.retrievedTitles.join(", ") || "none"}`,
      "",
      "**Answer**:",
      "",
      `> ${r.answer.slice(0, 500).replace(/\n/g, "\n> ")}${r.answer.length > 500 ? "..." : ""}`,
      "",
    );
    if (r.error) lines.push(`⚠️ Error: ${r.error}`, "");
  }
  return lines.join("\n");
}
