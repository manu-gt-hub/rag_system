"""
config.py — Centralized configuration for the RAG System.
Loads variables from .env and exposes constants for all modules.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env from the project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")


def _secret(name: str) -> str:
    """Read an API key, treating the .env.example placeholders as unset."""
    value = os.getenv(name, "").strip()
    return "" if not value or "XXXX" in value else value


# ---------------------------------------------------------------------------
# OpenAI
# ---------------------------------------------------------------------------
OPENAI_API_KEY = _secret("OPENAI_API_KEY")
OPENAI_EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
OPENAI_CHAT_MODEL = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")

# ---------------------------------------------------------------------------
# Anthropic
# ---------------------------------------------------------------------------
ANTHROPIC_API_KEY = _secret("ANTHROPIC_API_KEY")
ANTHROPIC_CHAT_MODEL = os.getenv("ANTHROPIC_CHAT_MODEL", "claude-haiku-4-5")

# ---------------------------------------------------------------------------
# LLM Provider: "openai" | "anthropic". If unset, the first provider with a
# key is used (OpenAI preferred).
# ---------------------------------------------------------------------------
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "").strip().lower() or (
    "anthropic" if ANTHROPIC_API_KEY and not OPENAI_API_KEY else "openai"
)
LLM_KEY_CONFIGURED = bool({"openai": OPENAI_API_KEY, "anthropic": ANTHROPIC_API_KEY}.get(LLM_PROVIDER))

# ---------------------------------------------------------------------------
# PostgreSQL + pgvector
# ---------------------------------------------------------------------------
POSTGRES_HOST = os.getenv("POSTGRES_HOST", "localhost")
POSTGRES_PORT = int(os.getenv("POSTGRES_PORT", "5432"))
POSTGRES_DB = os.getenv("POSTGRES_DB", "rag_system")
POSTGRES_USER = os.getenv("POSTGRES_USER", "postgres")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", "postgres")

POSTGRES_DSN = (
    f"host={POSTGRES_HOST} port={POSTGRES_PORT} dbname={POSTGRES_DB} "
    f"user={POSTGRES_USER} password={POSTGRES_PASSWORD}"
)

# ---------------------------------------------------------------------------
# Embeddings: "openai" (API) | "local" (fastembed, no key needed). If unset,
# OpenAI is used when its key is present, otherwise the local model.
# ---------------------------------------------------------------------------
EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "").strip().lower() or (
    "openai" if OPENAI_API_KEY else "local"
)
LOCAL_EMBEDDING_MODEL = os.getenv(
    "LOCAL_EMBEDDING_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)
EMBEDDING_DIMENSION = int(os.getenv(
    "EMBEDDING_DIMENSION", "1536" if EMBEDDING_PROVIDER == "openai" else "384"
))
# Max characters of a document sent to the embedding model (long OCR'd book
# extracts and big music collections are truncated, not chunked).
MAX_EMBED_CHARS = int(os.getenv("MAX_EMBED_CHARS", "4000"))

# ---------------------------------------------------------------------------
# Retriever
# ---------------------------------------------------------------------------
TOP_K = int(os.getenv("TOP_K", "5"))
SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.2"))
# Candidates fetched from each search (vector + full-text) before fusion
RETRIEVAL_CANDIDATES = int(os.getenv("RETRIEVAL_CANDIDATES", "40"))

# ---------------------------------------------------------------------------
# Chat
# ---------------------------------------------------------------------------
# Previous user/assistant messages kept as conversational memory
CHAT_HISTORY_MESSAGES = int(os.getenv("CHAT_HISTORY_MESSAGES", "6"))

# ---------------------------------------------------------------------------
# Data paths
# ---------------------------------------------------------------------------
INDEXES_DIR = PROJECT_ROOT / "resources" / "indexes"
RESULTS_DIR = PROJECT_ROOT / "results"
TEMPLATES_DIR = PROJECT_ROOT / "templates"
TESTS_DIR = PROJECT_ROOT / "tests"

# ---------------------------------------------------------------------------
# Web
# ---------------------------------------------------------------------------
FLASK_HOST = os.getenv("FLASK_HOST", "0.0.0.0")
FLASK_PORT = int(os.getenv("FLASK_PORT", "8000"))
FLASK_DEBUG = os.getenv("FLASK_DEBUG", "true").lower() == "true"

# ---------------------------------------------------------------------------
# Available content categories
# ---------------------------------------------------------------------------
CONTENT_CATEGORIES = [
    "peliculas", "libros", "series", "musica",
    "videojuegos", "documentales",
]

# Display names for the web UI
CATEGORY_LABELS = {
    "peliculas": "Movies",
    "libros": "Books",
    "series": "TV series",
    "musica": "Music",
    "videojuegos": "Video games",
    "documentales": "Documentaries",
}
