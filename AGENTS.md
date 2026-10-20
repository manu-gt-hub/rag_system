# AGENTS.md — RAG System Harness Engineering

## Overview

RAG (Retrieval-Augmented Generation) system over a personal multimedia catalog.
Uses pgvector as vector store, OpenAI/Anthropic embeddings and contextual
response generation from JSON indexes of movies, books, series, music,
video games and documentaries.

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
                    │   Anthropic) │
                    └──────────────┘
```

## Agents / Modules

### 1. Indexer Agent (`src/indexer.py`)
- **Role**: Loads `resources/indexes/*.json` files, generates embeddings and
  stores them in pgvector.
- **Responsibilities**:
  - Parse each `index.json` (movies, books, series, music, video games, documentaries).
  - Build documents combining title, synopsis, genre, director/author, actors.
  - Generate embeddings via API (OpenAI `text-embedding-3-small` or Anthropic).
  - Insert vectors into PostgreSQL with the pgvector extension.

### 2. Retriever Agent (`src/retriever.py`)
- **Role**: Semantic search against pgvector.
- **Responsibilities**:
  - Receive a user query.
  - Generate query embedding.
  - Execute cosine similarity search in pgvector.
  - Return the top-K most relevant documents with scores.

### 3. Generator Agent (`src/generator.py`)
- **Role**: Generate responses using an LLM with retrieved context.
- **Responsibilities**:
  - Build prompt with context (retrieved documents) + question.
  - Call OpenAI GPT-4o-mini or Anthropic Claude.
  - Return natural-language response with references.

### 4. RAG Pipeline (`src/rag_pipeline.py`)
- **Role**: Orchestrator connecting Retriever → Generator.
- **Responsibilities**:
  - Receive user question.
  - Invoke retriever to obtain context.
  - Invoke generator with context + question.
  - Return final response with metadata (time, docs used, scores).

### 5. Evaluator Agent (`tests/run_evaluation.py`)
- **Role**: Test harness to measure RAG quality.
- **Responsibilities**:
  - Load question/expected-answer battery (`tests/test_questions.json`).
  - Execute each question through the pipeline.
  - Evaluate accuracy with rubrics defined in `doc/rubricas.md`.
  - Generate report in `results/evaluation_results.md`.

### 6. Web UI (`src/web_app.py`)
- **Role**: Simple local web interface.
- **Responsibilities**:
  - Serve question form.
  - Display response + context documents + scores.
  - `/api/query` endpoint for programmatic integration.

## Harness Engineering

The harness engineering approach is applied as follows:

1. **Defined rubrics** in `doc/rubricas.md` — objective criteria to evaluate
   relevance, precision, completeness and format of responses.
2. **Test suite** in `tests/test_questions.json` — question battery with
   expected answers per category (movies, books, etc.).
3. **Automated evaluation** — `run_evaluation.py` runs tests, calculates % accuracy
   per rubric and generates a markdown report.
4. **Versioned results** — each run generates a report in
   `results/evaluation_results.md` with timestamp, scores and details.

## Configuration

- Copy `.env.example` → `.env` and fill in API keys.
- Run `bash setup.sh` to create venv and install dependencies.
- PostgreSQL with pgvector extension must be available.
- See `README.md` for full instructions.

## LLM Provider

Supports two providers (configurable via `LLM_PROVIDER` in `.env`):
- **openai**: GPT-4o-mini for generation, text-embedding-3-small for embeddings.
- **anthropic**: Claude 3.5 Haiku for generation (embeddings via OpenAI).
