# RAG System — AI-Powered Multimedia Catalog

RAG (Retrieval-Augmented Generation) system over your personal catalog of
movies, books, series, music, video games and documentaries. Uses pgvector
as vector store and OpenAI/Anthropic as LLM provider.

## Project Structure

```
rag_system/
├── AGENTS.md                # Agent definitions and architecture
├── README.md                # This file
├── .env.example             # Configuration template (API keys)
├── requirements.txt         # Python dependencies
├── setup.sh                 # Installation script (venv)
├── doc/
│   └── rubricas.md          # Evaluation rubrics
├── src/
│   ├── __init__.py
│   ├── config.py            # Centralized configuration
│   ├── embeddings.py        # Embedding generation (OpenAI)
│   ├── indexer.py           # Load JSON → pgvector
│   ├── retriever.py         # Semantic search
│   ├── generator.py         # LLM generation (OpenAI/Anthropic)
│   ├── rag_pipeline.py      # RAG orchestrator
│   └── web_app.py           # Web UI (Flask)
├── templates/
│   └── index.html           # Web interface
├── tests/
│   ├── test_questions.json  # Test question battery
│   └── run_evaluation.py    # Evaluation script with rubrics
├── results/
│   └── evaluation_results.md  # Accuracy results (generated)
└── resources/
    └── indexes/             # Catalog JSON indexes
        ├── peliculas/
        ├── libros/
        ├── series/
        ├── musica/
        ├── videojuegos/
        └── documentales/
```

## Prerequisites

- Python 3.10+
- PostgreSQL 15+ with **pgvector** extension installed
- **OpenAI** and/or **Anthropic** API key

## Installation

```bash
# 1. Clone and enter the directory
cd rag_system

# 2. Run setup (creates venv, installs deps, copies .env)
bash setup.sh

# 3. Activate the virtual environment
source venv/bin/activate

# 4. Edit .env with your API keys and PostgreSQL credentials
nano .env
```

## Usage

### 1. Index the catalog

```bash
python -m src.indexer
```

Loads all `resources/indexes/*/index.json`, generates embeddings and
inserts them into pgvector.

### 2. Start the web UI

```bash
python -m src.web_app
```

Open `http://localhost:5000` in your browser. Type natural-language
questions and get answers based on your catalog.

### 3. API REST

```bash
curl -X POST http://localhost:5000/api/query \
  -H "Content-Type: application/json" \
  -d '{"question": "¿Qué películas de terror tengo?", "category": "peliculas"}'
```

### 4. Run evaluation

```bash
python -m tests.run_evaluation
```

Runs the 10-question test battery, evaluates with the rubrics
defined in `doc/rubricas.md` and generates the accuracy report in
`results/evaluation_results.md`.

## API Token Configuration

Edit the `.env` file (copied from `.env.example`):

```env
# Provider: "openai" or "anthropic"
LLM_PROVIDER=openai

# OpenAI
OPENAI_API_KEY=sk-your-key-here

# Anthropic (alternative)
ANTHROPIC_API_KEY=sk-ant-your-key-here
```

## Evaluation Rubrics

Each test question is evaluated across 4 dimensions:

| Rubric | Weight | Description |
|---|---|---|
| Relevance | 30% | Are the retrieved documents the correct ones? |
| Precision | 35% | Does the answer contain the expected data? |
| Completeness | 20% | Is the answer complete and not erroneously rejected? |
| Format | 15% | In Spanish, well structured, adequate length? |

Acceptance threshold: **>= 70% global accuracy**.

## Architecture

See `AGENTS.md` for full architecture details and module roles.

```
User → Web UI (Flask) → RAG Pipeline → Retriever (pgvector) → Generator (LLM) → Response
```