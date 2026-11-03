/**
 * index.ts — Worker entry point and RAG pipeline. Static assets (public/) are
 * served by Cloudflare before reaching this code. API:
 *   POST /api/chat    {message, history?, category?, top_k?, lang?}  chat with memory
 *   POST /api/query   {question, category?, top_k?}           single question
 *   POST /api/search  {question, category?, top_k?}           retrieval only, no LLM
 *   GET  /api/config  LLM, embedding model and category labels for the UI
 *   GET  /health
 */

import type { Env } from "./env";
import { activeProvider, generate, QuotaExceededError, type ChatMessage } from "./generator";
import { EMBEDDING_MODEL, search } from "./retriever";
import { CATEGORIES, CATEGORY_LABELS, detectCategory, queryTerms } from "./text";

const HISTORY_MESSAGES = 6;
const FOLLOW_UP_MAX_TERMS = 3;
const MAX_MESSAGE_CHARS = 1000;

const json = (body: unknown, status = 200) =>
  new Response(JSON.stringify(body), { status, headers: { "Content-Type": "application/json" } });

/** Errors carry a stable `code` so the UI can show them in its own language. */
const fail = (code: string, error: string, status: number) => json({ error, code }, status);

// Server-generated answer text; LLM answers follow the language of the question
const NO_RESULTS: Record<string, string> = {
  en: "I couldn't find anything related in your catalog. Try rephrasing the question or changing the category.",
  es: "No he encontrado nada relacionado en tu catálogo. Prueba a reformular la pregunta o a cambiar la categoría.",
};

/** Enrich short follow-up questions with the previous user message. */
export function buildRetrievalQuery(question: string, history: ChatMessage[]): string {
  if (!history.length || queryTerms(question).length > FOLLOW_UP_MAX_TERMS) return question;
  const previous = history.filter((m) => m.role === "user" && m.content).map((m) => m.content);
  return previous.length ? `${previous[previous.length - 1]} ${question}` : question;
}

function cleanHistory(raw: unknown): ChatMessage[] {
  if (!Array.isArray(raw)) return [];
  return raw
    .filter((m) => m && (m.role === "user" || m.role === "assistant") && typeof m.content === "string")
    .map((m) => ({ role: m.role, content: m.content.slice(0, 4000) }) as ChatMessage)
    .slice(-HISTORY_MESSAGES);
}

type Mode = "chat" | "query" | "search";
const MODES: Record<string, Mode> = { "/api/chat": "chat", "/api/query": "query", "/api/search": "search" };

async function handleQuery(request: Request, env: Env, mode: Mode): Promise<Response> {
  const data = (await request.json().catch(() => ({}))) as Record<string, unknown>;
  const question = String(data.question ?? data.message ?? "").trim();
  if (!question) return fail("missing_question", "The 'question' field is required.", 400);
  if (question.length > MAX_MESSAGE_CHARS) return fail("message_too_long", "Message too long.", 400);
  const lang = data.lang === "es" ? "es" : "en";

  let category = typeof data.category === "string" && data.category ? data.category : null;
  if (category === "todas") category = null;
  if (category && !CATEGORIES.includes(category)) return fail("unknown_category", `Unknown category: ${category}`, 400);

  const t0 = Date.now();
  const history = mode === "chat" ? cleanHistory(data.history) : [];
  const retrievalQuery = buildRetrievalQuery(question, history);
  category ??= detectCategory(retrievalQuery);
  const topK = Math.min(Number(data.top_k) || Number(env.TOP_K) || 5, 10);

  const docs = await search(env, retrievalQuery, category, topK, Number(env.SIMILARITY_THRESHOLD));
  const { provider, model } = activeProvider(env);
  let answer: string | null = null;
  if (mode === "search") {
    // Retrieval only: used by monitoring, spends no LLM quota
  } else if (!docs.length) {
    answer = NO_RESULTS[lang];
  } else {
    try {
      answer = await generate(env, question, docs, history);
    } catch (e) {
      if (e instanceof QuotaExceededError) {
        return fail("quota_exceeded", "The free daily AI quota of this demo is used up. Please try again tomorrow.", 429);
      }
      throw e;
    }
  }

  const scores = docs.map((d) => d.score ?? 0);
  return json({
    query: question,
    answer,
    elapsed_seconds: (Date.now() - t0) / 1000,
    provider: `${provider} (${model})`,
    category,
    retrieval_query: retrievalQuery,
    top_score: scores.length ? Math.max(...scores) : 0,
    num_docs_retrieved: docs.length,
    docs: docs.map((d) => ({
      title: d.title,
      category: d.category,
      score: d.score === null ? null : Math.round(d.score * 10000) / 10000,
      keyword_hit: d.keywordHit,
      year: d.metadata.year ?? null,
      genre: d.metadata.genre ?? null,
      rating: d.metadata.rating ?? null,
      poster_url: d.metadata.poster_url ?? null,
    })),
  });
}

export default {
  async fetch(request: Request, env: Env): Promise<Response> {
    const { pathname } = new URL(request.url);

    if (pathname === "/health") return json({ status: "ok" });
    if (pathname === "/api/config") {
      return json({ ...activeProvider(env), embedding_model: EMBEDDING_MODEL, categories: CATEGORY_LABELS });
    }
    const mode = MODES[pathname];
    if (mode) {
      if (request.method !== "POST") return fail("method_not_allowed", "Method not allowed", 405);
      // Protects the free daily quota (and any API key) from abuse
      const ip = request.headers.get("CF-Connecting-IP") ?? "unknown";
      if (env.RATE_LIMITER && !(await env.RATE_LIMITER.limit({ key: ip })).success) {
        return fail("rate_limited", "Too many requests, please slow down.", 429);
      }
      try {
        return await handleQuery(request, env, mode);
      } catch (e) {
        console.error("Error processing query", e);
        return fail("internal_error", "Internal error processing the question.", 500);
      }
    }
    return env.ASSETS.fetch(request);
  },
} satisfies ExportedHandler<Env>;
