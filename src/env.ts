/** Bindings and variables declared in wrangler.jsonc. */
export interface Env {
  AI: Ai;
  VECTORIZE: VectorizeIndex;
  DB: D1Database;
  ASSETS: Fetcher;
  RATE_LIMITER?: RateLimit;

  // vars
  LLM_PROVIDER: string; // "workers-ai" | "openai" | "anthropic"
  WORKERS_AI_MODEL: string;
  OPENAI_CHAT_MODEL: string;
  ANTHROPIC_CHAT_MODEL: string;
  TOP_K: string;
  SIMILARITY_THRESHOLD: string;

  // secrets (optional, only for the paid providers)
  OPENAI_API_KEY?: string;
  ANTHROPIC_API_KEY?: string;
}
