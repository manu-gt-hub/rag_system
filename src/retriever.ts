/**
 * retriever.ts — Hybrid search:
 *   1. Vector search in Vectorize (bge-m3 embeddings, cosine).
 *   2. Full-text search in D1 (FTS5, title > keywords > content), documents
 *      containing all terms first.
 *   3. Documents whose whole title appears in the question.
 * Lists are merged with Reciprocal Rank Fusion, plus a bonus for documents
 * containing every query term and a double bonus for title matches.
 *
 * Degraded mode: when the free daily Workers AI quota is used up the question
 * cannot be embedded, so search continues with D1 alone (keywords and titles)
 * instead of failing, and reports it with `keywordOnly`.
 */

import type { Env } from "./env";
import { isQuotaError, QuotaExceededError } from "./generator";
import { ftsQuery, queryTerms, titleKey } from "./text";

export const EMBEDDING_MODEL = "@cf/baai/bge-m3";
export const RRF_K = 60;
export const ALL_TERMS_BONUS = 1 / RRF_K;
export const TITLE_BONUS = 2 / RRF_K;
const CANDIDATES = 40;
const MIN_TITLE_KEY_CHARS = 5;

export interface RetrievedDoc {
  id: number;
  category: string;
  title: string;
  content: string;
  metadata: Record<string, unknown>;
  score: number | null; // cosine similarity, null when found only by keyword
  rrf: number;
  keywordHit: boolean;
}

export interface SearchResult {
  docs: RetrievedDoc[];
  /** Vector search was skipped because the AI quota is used up. */
  keywordOnly: boolean;
}

interface Candidate {
  vectorRank?: number;
  vectorScore?: number;
  textRank?: number;
  allTerms?: boolean;
  titleMatch?: boolean;
}

/** Fuse the candidate rankings (pure function, unit-tested). */
export function fuse(
  candidates: Map<number, Candidate>,
  multiTerm: boolean,
  threshold: number,
  k: number,
): { id: number; rrf: number; score: number | null; keywordHit: boolean }[] {
  return [...candidates.entries()]
    .map(([id, c]) => {
      const rrf =
        (c.vectorRank ? 1 / (RRF_K + c.vectorRank) : 0) +
        (c.textRank ? 1 / (RRF_K + c.textRank) : 0) +
        (multiTerm && c.allTerms ? ALL_TERMS_BONUS : 0) +
        (c.titleMatch ? TITLE_BONUS : 0);
      const keywordHit = Boolean(c.textRank || c.titleMatch);
      return { id, rrf, score: c.vectorScore ?? null, keywordHit };
    })
    // Keyword matches are kept even with a low cosine score
    .filter((d) => d.keywordHit || (d.score ?? 0) >= threshold)
    .sort((a, b) => b.rrf - a.rrf)
    .slice(0, k);
}

export async function embed(env: Env, text: string): Promise<number[]> {
  try {
    const out = (await env.AI.run(EMBEDDING_MODEL, { text: [text] })) as { data: number[][] };
    return out.data[0];
  } catch (e) {
    if (isQuotaError(e)) throw new QuotaExceededError(String(e));
    throw e;
  }
}

export async function search(
  env: Env,
  query: string,
  category: string | null,
  k: number,
  threshold: number,
): Promise<SearchResult> {
  const terms = queryTerms(query);
  const candidates = new Map<number, Candidate>();
  const get = (id: number) => candidates.get(id) ?? candidates.set(id, {}).get(id)!;

  // Vector search and keyword searches run in parallel. Without AI quota the
  // vector side is skipped (null) and the keyword side still answers.
  const vectorSearch = embed(env, query)
    .then((vector) =>
      env.VECTORIZE.query(vector, {
        topK: CANDIDATES,
        returnMetadata: "none",
        ...(category ? { filter: { category } } : {}),
      }),
    )
    .catch((e) => {
      if (e instanceof QuotaExceededError) return null;
      throw e;
    });

  const catSql = category ? "AND d.category = ?" : "";
  const catArgs = category ? [category] : [];
  const statements = [
    env.DB.prepare(
      `SELECT d.id FROM docs d WHERE length(d.title_key) >= ${MIN_TITLE_KEY_CHARS}
         AND instr(?, d.title_key) > 0 ${catSql} LIMIT ${CANDIDATES}`,
    ).bind(titleKey(query), ...catArgs),
  ];
  if (terms.length) {
    const ftsSql = (match: string) =>
      env.DB.prepare(
        `SELECT d.id FROM docs_fts JOIN docs d ON d.id = docs_fts.rowid
          WHERE docs_fts MATCH ? ${catSql}
          ORDER BY bm25(docs_fts, 10.0, 5.0, 1.0) LIMIT ${CANDIDATES}`,
      ).bind(match, ...catArgs);
    statements.push(ftsSql(ftsQuery(terms, "AND")), ftsSql(ftsQuery(terms, "OR")));
  }
  const [vectorResult, batch] = await Promise.all([vectorSearch, env.DB.batch<{ id: number }>(statements)]);

  const keywordOnly = vectorResult === null;
  (vectorResult?.matches ?? []).forEach((m, i) => {
    const c = get(Number(m.id));
    c.vectorRank = i + 1;
    c.vectorScore = m.score;
  });
  batch[0].results.forEach((r) => (get(r.id).titleMatch = true));
  if (terms.length) {
    const allTerms = new Set(batch[1].results.map((r) => r.id));
    // Text ranking: documents with all terms first, then any term (bm25 order)
    const ranked = [...batch[1].results, ...batch[2].results.filter((r) => !allTerms.has(r.id))];
    ranked.slice(0, CANDIDATES).forEach((r, i) => {
      const c = get(r.id);
      c.textRank = i + 1;
      c.allTerms = allTerms.has(r.id);
    });
  }

  const top = fuse(candidates, terms.length > 1, threshold, k);
  if (!top.length) return { docs: [], keywordOnly };

  const placeholders = top.map(() => "?").join(", ");
  const rows = await env.DB.prepare(
    `SELECT id, category, title, content, metadata FROM docs WHERE id IN (${placeholders})`,
  )
    .bind(...top.map((t) => t.id))
    .all<{ id: number; category: string; title: string; content: string; metadata: string }>();
  const byId = new Map(rows.results.map((r) => [r.id, r]));

  const docs = top
    .filter((t) => byId.has(t.id))
    .map((t) => {
      const r = byId.get(t.id)!;
      return {
        id: r.id,
        category: r.category,
        title: r.title,
        content: r.content,
        metadata: r.metadata ? JSON.parse(r.metadata) : {},
        score: t.score,
        rrf: t.rrf,
        keywordHit: t.keywordHit,
      };
    });
  return { docs, keywordOnly };
}
