# RAG System — Recommendation chatbot for your media catalog

A chatbot that **recommends and answers questions** about a personal catalog of
movies, TV series, books, music, video games and documentaries. It uses RAG:
each catalog item becomes a semantic vector stored in **PostgreSQL + pgvector**;
each question is embedded with the same model, the most similar items are
retrieved (hybrid vector + full-text search) and an LLM (**OpenAI** or
**Anthropic**) writes the recommendation using only that context.

📐 Architecture, diagrams and technology choices: [doc/architecture.md](doc/architecture.md)

## Quick start (Docker)

Requirement: [Docker](https://docs.docker.com/get-docker/) with `docker compose`.

```bash
git clone <repo> && cd rag_system
cp .env.example .env          # Windows PowerShell: copy .env.example .env
# Edit .env and set at least one key:
#   OPENAI_API_KEY=sk-...      and/or   ANTHROPIC_API_KEY=sk-ant-...
docker compose up --build
```

Open **http://localhost:8000** and ask, for example:

- *Recommend me a horror movie* → then *any more recent ones?*
- *A science fiction book to read*
- *What do I have by Danny Boyle?*
- *An RPG for the Game Boy*

On first start the container indexes the catalog (about 2,400 documents) and
then serves the web app. Later starts reuse the index. If you change the
embedding provider, it re-indexes automatically.

| Keys in `.env` | LLM | Embeddings | Indexing cost |
|---|---|---|---|
| Only `OPENAI_API_KEY` | GPT-4o-mini | OpenAI `text-embedding-3-small` | a few cents, ~1 min |
| Only `ANTHROPIC_API_KEY` | Claude Haiku 4.5 | Local multilingual model (fastembed, CPU) | free; ~220 MB download the first time, a few minutes |
| Both | `LLM_PROVIDER` (default: openai) | `EMBEDDING_PROVIDER` (default: openai) | — |

Anthropic does not offer an embeddings API, so without an OpenAI key a local
model is used and no other account is needed.

The bot replies in the language of the question. The catalog data itself is
in Spanish, so titles are shown as they appear in the catalog.

### Check that it works

```bash
# Service health
curl http://localhost:8000/health

# Chat with memory
curl -X POST http://localhost:8000/api/chat -H "Content-Type: application/json" \
  -d '{"message": "Recommend me a horror movie", "history": []}'

# Single question (optional category filter)
curl -X POST http://localhost:8000/api/query -H "Content-Type: application/json" \
  -d '{"question": "Who directed 300?", "category": "peliculas"}'

# Unit tests (no keys needed)
docker compose exec app python -m pytest tests/test_unit.py -v

# Rubric-based evaluation against the real pipeline (calls the LLM API)
docker compose exec app python -m tests.run_evaluation
docker compose exec app cat results/evaluation_results.md
```

Useful commands:

```bash
docker compose logs -f app                        # follow indexing and requests
docker compose exec app python -m src.indexer     # force a re-index
docker compose down -v                            # stop and delete data and index
```

If ports 8000 or 5432 are already in use, change them in `.env`
(`APP_HOST_PORT`, `POSTGRES_HOST_PORT`).

## API

| Endpoint | Body | Description |
|---|---|---|
| `POST /api/chat` | `{"message", "history": [{"role","content"}], "category"?, "top_k"?}` | Chat with conversational memory |
| `POST /api/query` | `{"question", "category"?, "top_k"?}` | Single question |
| `GET /health` | — | Liveness check |

`category`: `peliculas` (movies), `libros` (books), `series`, `musica` (music),
`videojuegos` (video games), `documentales` (documentaries). These keys match
the catalog folders. If omitted, it is inferred when the question names exactly
one category.

The response includes `answer`, `docs` (title, category, similarity, year,
genre, rating, poster), `category`, `elapsed_seconds` and `provider`.

## Development without Docker

Requires Python 3.10+ and PostgreSQL with pgvector (you can start just the
database with `docker compose up -d db`).

```bash
bash setup.sh                 # creates venv, installs dependencies, copies .env
source venv/bin/activate      # Windows: venv\Scripts\activate
python -m src.indexer         # index (add --if-needed to skip when up to date)
python -m src.web_app         # http://localhost:8000
pytest tests/test_unit.py     # fast tests
python -m tests.run_evaluation
```

## Project structure

```
rag_system/
├── AGENTS.md                # Agent/module definitions
├── docker-compose.yml       # db (pgvector) + app
├── Dockerfile               # App image
├── docker/entrypoint.sh     # indexes if needed, then starts gunicorn
├── .env.example             # Configuration template (API keys)
├── requirements.txt
├── setup.sh                 # Local install (venv)
├── doc/
│   ├── architecture.md      # Diagrams, data flow and technology choices
│   └── rubrics.md           # Evaluation rubrics
├── src/
│   ├── config.py            # Configuration and provider auto-detection
│   ├── text_utils.py        # Normalization, search terms, category detection
│   ├── embeddings.py        # OpenAI or local (fastembed) embeddings
│   ├── indexer.py           # JSON → documents → pgvector
│   ├── retriever.py         # Hybrid search (vector + full-text, RRF)
│   ├── generator.py         # Recommendation prompt + LLM
│   ├── rag_pipeline.py      # Orchestrator
│   └── web_app.py           # Chat UI + API (Flask)
├── templates/index.html     # Chat interface
├── tests/
│   ├── test_unit.py         # Tests without keys or database
│   ├── test_rag_ci.py       # Rubric thresholds (CI)
│   ├── run_evaluation.py    # Evaluation report
│   └── test_questions.json  # Question battery
├── results/evaluation_results.md
└── resources/indexes/       # Catalog JSON per category
```

## Evaluation (harness)

Each question in `tests/test_questions.json` is scored with the rubrics in
[doc/rubrics.md](doc/rubrics.md):

| Rubric | Weight | What it measures |
|---|---|---|
| Relevance | 30% | Are the expected documents retrieved? |
| Precision | 35% | Does the answer contain the expected data? |
| Completeness | 20% | Does it answer without wrongly refusing? |
| Format | 15% | Right language, well structured, reasonable length? |

Acceptance threshold: **>= 70% global accuracy**. The
`.github/workflows/rag-evaluation.yml` workflow runs it on every push using the
repository secrets.
