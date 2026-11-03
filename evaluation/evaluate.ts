/**
 * evaluate.ts — Run the question battery against a deployed Worker, score it
 * with the rubrics of doc/rubrics.md and write evaluation/results.md.
 *
 *   npm run evaluate -- --url https://rag-catalog.<subdomain>.workers.dev [--strict]
 *   (or set DEMO_URL)
 *
 * Questions are spaced out to respect the demo's rate limit (10/min per IP).
 * When the free daily AI quota is exhausted the run is skipped with a warning.
 * --strict exits with code 1 if the thresholds of doc/rubrics.md are not met.
 */

import { readFileSync, writeFileSync } from "node:fs";

import { type CaseResult, type TestCase, failures, report, score } from "./rubrics";

const DELAY_MS = 7000;

function arg(name: string): string | undefined {
  const i = process.argv.indexOf(name);
  return i >= 0 ? process.argv[i + 1] : undefined;
}

const annotate = (level: "warning" | "error", message: string) =>
  process.env.GITHUB_ACTIONS ? console.log(`::${level}::${message}`) : console.log(`${level.toUpperCase()}: ${message}`);

const url = (arg("--url") ?? process.env.DEMO_URL ?? "").replace(/\/$/, "");
if (!url) {
  console.error("Pass --url <worker url> or set DEMO_URL");
  process.exit(2);
}

const tests: TestCase[] = JSON.parse(readFileSync(new URL("questions.json", import.meta.url), "utf-8")).tests;
const config = (await (await fetch(`${url}/api/config`)).json()) as { provider: string; model: string };
const results: CaseResult[] = [];
const t0 = Date.now();

for (const [i, test] of tests.entries()) {
  if (i > 0) await new Promise((r) => setTimeout(r, DELAY_MS));
  const start = Date.now();
  const resp = await fetch(`${url}/api/query`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question: test.question, category: test.category }),
  });
  const body = (await resp.json().catch(() => ({}))) as {
    answer?: string;
    error?: string;
    top_score?: number;
    docs?: { title: string }[];
  };

  if (resp.status === 429 && /quota/i.test(body.error ?? "")) {
    annotate("warning", `Evaluation skipped: the free daily AI quota is used up (${test.id}).`);
    process.exit(0);
  }

  const answer = body.answer ?? "";
  const titles = (body.docs ?? []).map((d) => d.title);
  const result: CaseResult = {
    id: test.id,
    question: test.question,
    category: test.category,
    answer,
    retrievedTitles: titles,
    topScore: body.top_score ?? 0,
    elapsed: (Date.now() - start) / 1000,
    error: resp.ok ? null : `HTTP ${resp.status}: ${body.error ?? ""}`,
    scores: score(test, answer, titles),
  };
  results.push(result);
  console.log(`[${i + 1}/${tests.length}] ${test.id}: global ${result.scores.global}%`);
}

const markdown = report(results, {
  url,
  provider: `${config.provider} (${config.model})`,
  elapsed: (Date.now() - t0) / 1000,
});
writeFileSync(new URL("results.md", import.meta.url), markdown);
if (process.env.GITHUB_STEP_SUMMARY) {
  writeFileSync(process.env.GITHUB_STEP_SUMMARY, markdown.split("## Response Details")[0], { flag: "a" });
}

const problems = failures(results);
for (const p of problems) annotate("error", p);
console.log(problems.length ? "Evaluation below thresholds" : "Evaluation passed");
process.exit(problems.length && process.argv.includes("--strict") ? 1 : 0);
