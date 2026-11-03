/**
 * generator.ts — Answer generation. Default: a Workers AI model (free daily
 * allocation). OpenAI / Anthropic are used when LLM_PROVIDER says so and the
 * matching secret is set (wrangler secret put OPENAI_API_KEY).
 */

import type { Env } from "./env";
import type { RetrievedDoc } from "./retriever";
import { SYSTEM_PROMPT } from "./prompt";

export interface ChatMessage {
  role: "user" | "assistant";
  content: string;
}

// Context is trimmed per document to keep Workers AI usage (Neurons) low
const MAX_DOC_CHARS = 1200;
const MAX_TOKENS = 700;

export class QuotaExceededError extends Error {}

/** Workers AI daily free allocation used up: HTTP 429 / error code 3036 (chat and embedding calls). */
export const isQuotaError = (e: unknown): boolean => /3036|daily free allocation|429/i.test(String(e));

export function buildContext(docs: RetrievedDoc[]): string {
  return docs
    .map((d, i) => {
      const relevance = d.score === null ? "keyword match" : d.score.toFixed(3);
      return `--- Document ${i + 1} (relevance: ${relevance}) ---\n${d.content.slice(0, MAX_DOC_CHARS)}\n`;
    })
    .join("\n");
}

export function activeProvider(env: Env): { provider: string; model: string } {
  if (env.LLM_PROVIDER === "openai" && env.OPENAI_API_KEY) {
    return { provider: "openai", model: env.OPENAI_CHAT_MODEL };
  }
  if (env.LLM_PROVIDER === "anthropic" && env.ANTHROPIC_API_KEY) {
    return { provider: "anthropic", model: env.ANTHROPIC_CHAT_MODEL };
  }
  return { provider: "workers-ai", model: env.WORKERS_AI_MODEL };
}

export async function generate(
  env: Env,
  question: string,
  docs: RetrievedDoc[],
  history: ChatMessage[],
): Promise<string> {
  const messages: ChatMessage[] = [
    ...history,
    { role: "user", content: `Context retrieved from my catalog:\n${buildContext(docs)}\n\nQuestion: ${question}` },
  ];
  const { provider, model } = activeProvider(env);
  if (provider === "openai") return openai(env, model, messages);
  if (provider === "anthropic") return anthropic(env, model, messages);
  return workersAi(env, model, messages);
}

async function workersAi(env: Env, model: string, messages: ChatMessage[]): Promise<string> {
  try {
    const out = (await env.AI.run(model as keyof AiModels, {
      messages: [{ role: "system", content: SYSTEM_PROMPT }, ...messages],
      max_tokens: MAX_TOKENS,
      temperature: 0.3,
    })) as { response?: string };
    return (out.response ?? "").trim();
  } catch (e) {
    if (isQuotaError(e)) throw new QuotaExceededError(String(e));
    throw e;
  }
}

async function openai(env: Env, model: string, messages: ChatMessage[]): Promise<string> {
  const resp = await fetch("https://api.openai.com/v1/chat/completions", {
    method: "POST",
    headers: { Authorization: `Bearer ${env.OPENAI_API_KEY}`, "Content-Type": "application/json" },
    body: JSON.stringify({
      model,
      messages: [{ role: "system", content: SYSTEM_PROMPT }, ...messages],
      temperature: 0.3,
      max_tokens: MAX_TOKENS,
    }),
  });
  if (!resp.ok) throw new Error(`OpenAI ${resp.status}: ${await resp.text()}`);
  const body = (await resp.json()) as { choices: { message: { content: string } }[] };
  return body.choices[0].message.content.trim();
}

async function anthropic(env: Env, model: string, messages: ChatMessage[]): Promise<string> {
  // Anthropic requires the conversation to start with a user turn
  while (messages.length && messages[0].role !== "user") messages = messages.slice(1);
  const resp = await fetch("https://api.anthropic.com/v1/messages", {
    method: "POST",
    headers: {
      "x-api-key": env.ANTHROPIC_API_KEY!,
      "anthropic-version": "2023-06-01",
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ model, system: SYSTEM_PROMPT, messages, temperature: 0.3, max_tokens: MAX_TOKENS }),
  });
  if (!resp.ok) throw new Error(`Anthropic ${resp.status}: ${await resp.text()}`);
  const body = (await resp.json()) as { content: { text: string }[] };
  return body.content[0].text.trim();
}
