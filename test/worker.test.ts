/**
 * worker.test.ts — Tests without a Cloudflare account.
 *
 * The retrieval and API tests load the real catalog (build/d1-data.sql, from
 * `npm run index -- --build-only`, which `npm test` runs first) into Node's
 * built-in SQLite, which also has FTS5, and fake Vectorize / Workers AI.
 */

import { existsSync, readFileSync } from "node:fs";
import { createRequire } from "node:module";
import { describe, expect, it } from "vitest";

import type { Env } from "../src/env";
import worker, { buildRetrievalQuery } from "../src/index";
import { fuse, search, TITLE_BONUS } from "../src/retriever";
import { detectCategory, ftsQuery, normalize, queryTerms, titleKey } from "../src/text";

// ---------------------------------------------------------------------------
// text.ts
// ---------------------------------------------------------------------------

describe("text", () => {
  it("normalizes accents", () => {
    expect(normalize("Películas de Acción ÑÚ")).toBe("peliculas de accion nu");
  });

  it("drops filler words in both languages", () => {
    expect(queryTerms("¿Qué contenido tengo de Danny Boyle?")).toEqual(["danny", "boyle"]);
    expect(queryTerms("What do I have by Danny Boyle?")).toEqual(["danny", "boyle"]);
    expect(queryTerms("¿Quién dirigió 300?")).toContain("300");
  });

  it("detects a single category", () => {
    expect(detectCategory("Recommend me a horror movie")).toBe("peliculas");
    expect(detectCategory("¿Qué películas de terror tengo?")).toBe("peliculas");
    expect(detectCategory("An RPG for the Game Boy")).toBe("videojuegos");
    expect(detectCategory("Any TV show like Breaking Bad?")).toBe("series");
  });

  it("returns null for ambiguous or missing categories", () => {
    expect(detectCategory("What do I have by Danny Boyle?")).toBeNull();
    expect(detectCategory("Una serie basada en un libro")).toBeNull();
    expect(detectCategory("¿Tengo Juego de tronos?")).toBeNull();
  });

  it("builds FTS5 queries with prefixes for long terms", () => {
    expect(ftsQuery(["danny", "boyle"], "AND")).toBe('"danny"* AND "boyle"*');
    expect(ftsQuery(["peliculas", "300"], "OR")).toBe('"pelicula"* OR "300"');
  });

  it("finds a title inside a question", () => {
    expect(titleKey("What rating does Algunos hombres buenos have?")).toContain(titleKey("Algunos hombres buenos"));
    expect(titleKey("Aliens and robots")).not.toContain(titleKey("Alien"));
  });
});

// ---------------------------------------------------------------------------
// Fusion and conversation handling
// ---------------------------------------------------------------------------

describe("fuse", () => {
  it("ranks a document found by both searches first", () => {
    const candidates = new Map([
      [1, { vectorRank: 1, vectorScore: 0.8 }],
      [2, { vectorRank: 2, vectorScore: 0.7, textRank: 1 }],
    ]);
    expect(fuse(candidates, false, 0.3, 5).map((d) => d.id)).toEqual([2, 1]);
  });

  it("gives the title bonus and keeps keyword hits under the threshold", () => {
    const candidates = new Map([
      [1, { vectorRank: 1, vectorScore: 0.9 }],
      [2, { titleMatch: true }],
      [3, { vectorRank: 2, vectorScore: 0.1 }],
    ]);
    const result = fuse(candidates, false, 0.3, 5);
    expect(result.map((d) => d.id)).toEqual([2, 1]);
    expect(result[0].rrf).toBeCloseTo(TITLE_BONUS);
    expect(result[0].score).toBeNull();
  });
});

describe("buildRetrievalQuery", () => {
  const history = [
    { role: "user" as const, content: "horror movies" },
    { role: "assistant" as const, content: "I recommend..." },
  ];

  it("leaves standalone questions untouched", () => {
    const q = "Recommend me a science fiction book with spaceships and aliens";
    expect(buildRetrievalQuery(q, history)).toBe(q);
  });

  it("joins short follow-ups with the previous question", () => {
    expect(buildRetrievalQuery("any more recent ones?", history)).toBe("horror movies any more recent ones?");
  });
});

// ---------------------------------------------------------------------------
// Retrieval against the real catalog in SQLite (fake Vectorize / Workers AI)
// ---------------------------------------------------------------------------

const DATA = new URL("../build/d1-data.sql", import.meta.url);

export function fakeEnv(vectorMatches: { id: string; score: number }[] = []): Env {
  const { DatabaseSync } = createRequire(import.meta.url)("node:sqlite");
  const db = new DatabaseSync(":memory:");
  db.exec(readFileSync(DATA, "utf-8")); // schema + data

  const statement = (sql: string, args: unknown[] = []) => ({
    bind: (...a: unknown[]) => statement(sql, a),
    all: async () => ({ results: db.prepare(sql).all(...args) }),
  });
  return {
    DB: {
      prepare: (sql: string) => statement(sql),
      batch: (stmts: ReturnType<typeof statement>[]) => Promise.all(stmts.map((s) => s.all())),
    },
    VECTORIZE: { query: async () => ({ matches: vectorMatches, count: vectorMatches.length }) },
    AI: {
      run: async (model: string) =>
        model.includes("bge") ? { data: [[0]] } : { response: "I recommend **28 años después**, a horror movie." },
    },
    ASSETS: { fetch: async () => new Response("<html></html>") },
    LLM_PROVIDER: "workers-ai",
    WORKERS_AI_MODEL: "@cf/meta/llama-3.3-70b-instruct-fp8-fast",
    TOP_K: "5",
    SIMILARITY_THRESHOLD: "0.35",
  } as unknown as Env;
}

describe.skipIf(!existsSync(DATA))("search (keyword side, real catalog)", () => {
  const env = fakeEnv();
  const titles = async (q: string, cat: string | null = null) =>
    (await search(env, q, cat, 5, 0.35)).map((d) => d.title);

  it("finds a director's movies by name", async () => {
    expect(await titles("What do I have by Danny Boyle?")).toContain("28 años después");
  });

  it("finds a Spanish title quoted in an English question", async () => {
    const result = await titles("What rating does the movie Algunos hombres buenos have?", "peliculas");
    expect(result[0]).toBe("Algunos hombres buenos");
  });

  it("matches English questions through bilingual genres", async () => {
    expect(await titles("horror", "peliculas")).not.toHaveLength(0);
  });

  it("filters by category", async () => {
    const docs = await search(env, "Nicolas Cage", "peliculas", 5, 0.35);
    expect(docs.length).toBeGreaterThan(0);
    expect(docs.every((d) => d.category === "peliculas")).toBe(true);
  });

  it("merges vector matches with keyword matches", async () => {
    const withVectors = fakeEnv([{ id: "1", score: 0.9 }]);
    const docs = await search(withVectors, "Danny Boyle", null, 5, 0.35);
    expect(docs[0].id).toBe(1); // in both lists
    expect(docs[0].score).toBe(0.9);
  });
});

describe.skipIf(!existsSync(DATA))("API", () => {
  const env = fakeEnv();
  const post = (path: string, body: unknown) =>
    worker.fetch(new Request(`https://demo${path}`, { method: "POST", body: JSON.stringify(body) }), env);

  it("answers a chat message with sources", async () => {
    const resp = await post("/api/chat", { message: "What do I have by Danny Boyle?", history: [] });
    expect(resp.status).toBe(200);
    const body = (await resp.json()) as { answer: string; docs: { title: string }[]; provider: string };
    expect(body.answer).toContain("28 años después");
    expect(body.docs.map((d) => d.title)).toContain("28 años después");
    expect(body.provider).toContain("workers-ai");
  });

  it("returns only retrieval results on /api/search", async () => {
    const body = (await (await post("/api/search", { question: "Nicolas Cage" })).json()) as {
      answer: string | null;
      docs: unknown[];
    };
    expect(body.answer).toBeNull();
    expect(body.docs.length).toBeGreaterThan(0);
  });

  it("detects the category from the question", async () => {
    const body = (await (await post("/api/query", { question: "Recommend me a horror movie" })).json()) as {
      category: string;
    };
    expect(body.category).toBe("peliculas");
  });

  it("validates input and returns stable error codes", async () => {
    const missing = await post("/api/query", {});
    expect(missing.status).toBe(400);
    expect(((await missing.json()) as { code: string }).code).toBe("missing_question");
    const unknown = await post("/api/query", { question: "x", category: "comics" });
    expect(((await unknown.json()) as { code: string }).code).toBe("unknown_category");
    const get = await worker.fetch(new Request("https://demo/api/query"), env);
    expect(get.status).toBe(405);
  });

  it("answers in the UI language when nothing is found", async () => {
    const ask = async (lang?: string) =>
      ((await (await post("/api/query", { question: "qwxzvb", lang })).json()) as { answer: string; docs: unknown[] });
    const es = await ask("es");
    expect(es.docs).toHaveLength(0);
    expect(es.answer).toMatch(/^No he encontrado/);
    expect((await ask()).answer).toMatch(/^I couldn't find/);
  });

  it("serves config and health", async () => {
    const config = (await (await worker.fetch(new Request("https://demo/api/config"), env)).json()) as {
      categories: Record<string, string>;
    };
    expect(config.categories.peliculas).toBe("Movies");
    expect((await worker.fetch(new Request("https://demo/health"), env)).status).toBe(200);
  });
});
