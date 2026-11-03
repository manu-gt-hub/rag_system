# AGENTS.md — RAG Catalog Harness Engineering

## Overview

RAG (Retrieval-Augmented Generation) recommendation chatbot over a personal
multimedia catalog, running entirely on Cloudflare's free tier: a TypeScript
Worker serves the chat UI and API, Vectorize stores the embeddings, D1 holds
the documents and an FTS5 keyword index, and Workers AI provides embeddings
(`bge-m3`) and answers (Llama 3.3 70B). OpenAI or Anthropic can replace the
LLM through Worker secrets.

Detailed architecture, diagrams and technology rationale: `doc/architecture.md`.

## Architecture

```
┌─────────────┐     ┌──────────────────┐     ┌────────────────┐
│  Chat UI    │────▶│ Worker (TS)      │────▶│ Workers AI     │
│  (static)   │     │ RAG pipeline     │     │ bge-m3 + Llama │
└─────────────┘     └────────┬─────────┘     └────────────────┘
                             │
               ┌─────────────┴─────────────┐
        ┌──────▼───────┐            ┌──────▼───────┐
        │  Vectorize   │            │  D1 (SQLite) │
        │  vectors     │            │  docs + FTS5 │
        └──────────────┘            └──────────────┘
```

## Agents / Modules

### 1. Indexer Agent (`indexer/`)
- **Role**: Turns `resources/indexes/*/index.json` into semantic documents
  and loads them into Vectorize and D1.
- **Responsibilities**:
  - Build one document per recommendable item (`documents.ts`): series are
    grouped by show, music by artist/album folder, movies/games/books are
    deduplicated; genres are stored in Spanish and English.
  - Compute a signature (catalog + document format + embedding model) and skip
    work when D1 already holds it (`--if-changed`).
  - Embed with Workers AI `bge-m3` (REST API, batches of 50, cached per signature).
  - Upsert vectors into Vectorize and load documents + FTS5 + signature into D1.

### 2. Retriever Agent (`src/retriever.ts`)
- **Role**: Hybrid search.
- **Responsibilities**:
  - Embed the question with the same model used for indexing.
  - Query Vectorize (cosine, optional category filter) and D1 (title match,
    FTS5 all-terms and any-term) in parallel.
  - Fuse the lists with Reciprocal Rank Fusion plus all-terms and title bonuses.
  - Return the top-K documents with scores.
  - Without AI quota, fall back to D1 only (`keywordOnly`) instead of failing.

### 3. Generator Agent (`src/generator.ts`)
- **Role**: Generate the answer with retrieved context.
- **Responsibilities**:
  - Build the prompt (`src/prompt.ts`) with context, history and question.
  - Call Workers AI (default), OpenAI or Anthropic.
  - Report an exhausted free daily quota as a clear 429.

### 4. RAG Pipeline (`src/index.ts`)
- **Role**: Orchestrator and API.
- **Responsibilities**:
  - `/api/chat` (with memory), `/api/query`, `/api/search` (retrieval only),
    `/api/config`, `/health`; rate limiting per IP.
  - Enrich short follow-ups with the previous question; detect category.
  - Invoke retriever, then generator; return answer, sources and metrics.

### 5. Evaluator Agent (`evaluation/`)
- **Role**: Test harness measuring RAG quality on the live deployment.
- **Responsibilities**:
  - Load the question battery (`evaluation/questions.json`).
  - Run each question against a deployed URL.
  - Score with the rubrics of `doc/rubrics.md` (`evaluation/rubrics.ts`).
  - Write `evaluation/results.md`; fail under thresholds with `--strict`.

### 6. Web UI (`public/index.html`)
- **Role**: Chat interface served as static assets.
- **Responsibilities**:
  - Keep the conversation history client-side and send it to `/api/chat`.
  - Render answers (light markdown) and the source documents with scores.

## Harness Engineering

1. **Defined rubrics** in `doc/rubrics.md` — relevance, precision,
   completeness and format, with acceptance thresholds.
2. **Test suite** — `evaluation/questions.json` for quality; `test/` (vitest)
   for code, including retrieval against the real catalog in SQLite and the
   chat UI running in happy-dom (language switch, escaping, chat round trip).
3. **Automated evaluation** — every deploy that can change answer quality
   (`src/`, `indexer/`, `resources/`, `evaluation/`, `schema.sql`,
   `wrangler.jsonc`) evaluates the live demo and fails the evaluation job
   under thresholds; other deploys skip it to save the free AI quota.
4. **Versioned results** — each run publishes `evaluation/results.md` as a CI
   artifact and in the run summary.

## Configuration

- `npm install`, `npx wrangler login`, `cp .env.example .env`
  (`CLOUDFLARE_ACCOUNT_ID`, `CLOUDFLARE_API_TOKEN`).
- `npm run setup` (creates D1 + Vectorize), `npm run index`, `npm run deploy`.
- Tests need no account: `npm test`.
- CI/CD: secrets `CLOUDFLARE_API_TOKEN`, `CLOUDFLARE_ACCOUNT_ID`; variable `DEMO_URL`.
- See `README.md` for full instructions.

## LLM Provider

`LLM_PROVIDER` in `wrangler.jsonc`:
- **workers-ai** (default, free): `@cf/meta/llama-3.3-70b-instruct-fp8-fast`.
- **openai**: needs `wrangler secret put OPENAI_API_KEY`.
- **anthropic**: needs `wrangler secret put ANTHROPIC_API_KEY`.

Embeddings are always Workers AI `bge-m3`; changing the LLM needs no re-index.
