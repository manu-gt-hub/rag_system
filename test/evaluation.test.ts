/** evaluation.test.ts — Rubric scoring of doc/rubrics.md. */

import { describe, expect, it } from "vitest";

import { evalCompleteness, evalFormat, evalPrecision, evalRelevance, failures, report, score } from "../evaluation/rubrics";

describe("rubrics", () => {
  it("relevance is the share of expected docs retrieved", () => {
    expect(evalRelevance(["28 años después", "Alone (Sola)"], ["28 años después", "28 semanas después", "Alone"])).toBeCloseTo(2 / 3);
    expect(evalRelevance([], [])).toBe(1);
  });

  it("precision is the share of expected keywords in the answer", () => {
    expect(evalPrecision("Zack Snyder directed 300, about Sparta.", ["Zack Snyder", "300", "Sparta", "Leonidas"])).toBe(0.75);
  });

  it("completeness penalizes wrong refusals", () => {
    expect(evalCompleteness("I don't have enough information in your catalog to answer that.", true)).toBe(0);
    expect(evalCompleteness("Short.", false)).toBe(0.25);
    expect(evalCompleteness("x".repeat(120), true)).toBe(1);
  });

  it("format checks language, length and structure", () => {
    expect(evalFormat("This is the answer, with a title in bold and the rating of the movie.")).toBe(1);
    expect(evalFormat("ok")).toBe(0);
  });

  it("weights the global score and reports threshold failures", () => {
    const test = { id: "T", category: null, question: "q", expected_keywords: ["Alien"], expected_docs: ["Alien"] };
    const answer =
      "I recommend **Alien**, the classic of science fiction and horror from 1979 directed by Ridley Scott, with a rating of 8.2.";
    const scores = score(test, answer, ["Alien"]);
    expect(scores.global).toBe(100);
    const bad = { ...scores, precision: 0, global: 50 };
    const result = { id: "T", question: "q", category: null, answer: "", retrievedTitles: [], topScore: 0, elapsed: 1, error: null };
    expect(failures([{ ...result, scores }])).toEqual([]);
    expect(failures([{ ...result, scores: bad }])).toContain("Questions with 0% precision: T");
    expect(report([{ ...result, scores }], { url: "u", provider: "p", elapsed: 1 })).toContain("**Global Accuracy** | **100.0%**");
  });
});
