# AGENTS.md — RAG System Harness Engineering

## Overview

RAG (Retrieval-Augmented Generation) recommendation chatbot over a personal
multimedia catalog. Uses pgvector as vector store, OpenAI or local (fastembed)
embeddings and OpenAI/Anthropic for response generation from JSON indexes of
movies, books, series, music, video games and documentaries.

Detailed architecture, diagrams and technology rationale: `doc/architecture.md`.

## Architecture

```
┌─────────────┐     ┌──────────────┐     ┌──────────────┐
│  Web UI     │────▶│ RAG Pipeline │────▶│  LLM (GPT /  │
│  (Flask)    │     │              │     │  Claude)     │
└─────────────┘     └──────┬───────┘     └──────────────┘
                           │
                    ┌──────▼───────┐
                    │  pgvector    │
                    │ (PostgreSQL) │
                    └──────┬───────┘
                           │
                    ┌──────▼───────┐
                    │  Embeddings  │
                    │  (OpenAI /   │
                    │   local)     │
                    └──────────────┘
```

## Agents / Modules

### 1. Indexer Agent (`src/indexer.py`)
- **Role**: Loads `resources/indexes/*.json` files, generates embeddings and
  stores them in pgvector.
- **Responsibilities**:
  - Parse each `index.json` (movies, books, series, music, video games, documentaries).
  - Build one semantic document per recommendable item: series are grouped by
    show, music by artist/album folder, movies/games are deduplicated.
  - Generate embeddings (OpenAI `text-embedding-3-small`, or the local
    multilingual model when there is no OpenAI key).
  - Insert vectors + Spanish `tsvector` into PostgreSQL/pgvector, build HNSW
    and GIN indexes, and record the embedding signature in `index_meta`.
  - `--if-needed` skips work when the index matches the current model.

### 2. Retriever Agent (`src/retriever.py`)
- **Role**: Semantic search against pgvector.
- **Responsibilities**:
  - Receive a user query.
  - Generate query embedding with the same model used for indexing.
  - Run cosine similarity search and Spanish full-text search in one SQL
    query, fused with Reciprocal Rank Fusion.
  - Return the top-K most relevant documents with scores.

### 3. Generator Agent (`src/generator.py`)
- **Role**: Generate responses using an LLM with retrieved context.
- **Responsibilities**:
  - Build recommendation prompt with context (retrieved documents),
    conversation history and question.
  - Call OpenAI GPT-4o-mini or Anthropic Claude.
  - Return natural-language response with references.

### 4. RAG Pipeline (`src/rag_pipeline.py`)
- **Role**: Orchestrator connecting Retriever → Generator.
- **Responsibilities**:
  - Receive user question and chat history.
  - Enrich short follow-ups with the previous question; detect category.
  - Invoke retriever to obtain context.
  - Invoke generator with context + question.
  - Return final response with metadata (time, docs used, scores).

### 5. Evaluator Agent (`tests/run_evaluation.py`)
- **Role**: Test harness to measure RAG quality.
- **Responsibilities**:
  - Load question/expected-answer battery (`tests/test_questions.json`).
  - Execute each question through the pipeline.
  - Evaluate accuracy with rubrics defined in `doc/rubrics.md`.
  - Generate report in `results/evaluation_results.md`.

### 6. Web UI (`src/web_app.py`)
- **Role**: Chat web interface.
- **Responsibilities**:
  - Serve the chat page (history kept client-side).
  - Display response + context documents + scores.
  - `/api/chat` (with memory) and `/api/query` endpoints.

## Harness Engineering

The harness engineering approach is applied as follows:

1. **Defined rubrics** in `doc/rubrics.md` — objective criteria to evaluate
   relevance, precision, completeness and format of responses.
2. **Test suite** in `tests/test_questions.json` — question battery with
   expected answers per category (movies, books, etc.).
3. **Automated evaluation** — `run_evaluation.py` runs tests, calculates % accuracy
   per rubric and generates a markdown report.
4. **Versioned results** — each run generates a report in
   `results/evaluation_results.md` with timestamp, scores and details.

## Configuration

- Copy `.env.example` → `.env` and fill in at least one API key.
- `docker compose up --build` starts pgvector, indexes and serves on :8000.
- Without Docker: `bash setup.sh`, a PostgreSQL with pgvector, then
  `python -m src.indexer` and `python -m src.web_app`.
- Unit tests without keys: `pytest tests/test_unit.py`.
- See `README.md` for full instructions.

## LLM Provider

Supports two providers (`LLM_PROVIDER` in `.env`, auto-detected from keys):
- **openai**: GPT-4o-mini for generation, text-embedding-3-small for embeddings.
- **anthropic**: Claude Haiku 4.5 for generation. Embeddings via OpenAI if its
  key is present, otherwise the local fastembed model (`EMBEDDING_PROVIDER`).
