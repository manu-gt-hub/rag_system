"""
indexer.py — Loads JSON indexes, generates embeddings and stores them in pgvector.
Runnable as: python -m src.indexer
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List

import psycopg2
from pgvector.psycopg2 import register_vector

from src.config import (
    CONTENT_CATEGORIES,
    EMBEDDING_DIMENSION,
    INDEXES_DIR,
    POSTGRES_DSN,
)
from src.embeddings import EmbeddingService

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Document construction from JSON
# ---------------------------------------------------------------------------

def _build_document(item: Dict[str, Any], category: str) -> str:
    """Build a plain-text document from an index record."""
    parts = [f"[{category.upper()}]"]

    titulo = item.get("titulo") or item.get("nombre") or item.get("title", "")
    if titulo:
        parts.append(f"Título: {titulo}")

    for field in ("año", "genero", "género", "director", "autor", "artista",
                  "desarrollador", "plataforma", "discografica"):
        value = item.get(field)
        if value:
            parts.append(f"{field.capitalize()}: {value}")

    actores = item.get("actores") or item.get("reparto", [])
    if actores:
        parts.append(f"Reparto: {', '.join(actores[:8])}")

    sinopsis = item.get("sinopsis") or item.get("descripcion", "")
    if sinopsis:
        parts.append(f"Sinopsis: {sinopsis}")

    rating = item.get("rating") or item.get("vote_average")
    if rating:
        parts.append(f"Valoración: {rating}")

    return "\n".join(parts)


def load_indexes() -> List[Dict[str, Any]]:
    """Load all JSON indexes and return a list of {category, text, metadata}."""
    documents = []
    for category in CONTENT_CATEGORIES:
        index_path = INDEXES_DIR / category / "index.json"
        if not index_path.exists():
            logger.warning("Index not found at %s, skipping.", index_path)
            continue

        with open(index_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        items = data.get(category, data.get(list(data.keys())[0], []))
        logger.info("Loaded %d records from %s", len(items), category)

        for item in items:
            text = _build_document(item, category)
            documents.append({
                "category": category,
                "text": text,
                "metadata": item,
            })

    logger.info("Total documents: %d", len(documents))
    return documents


# ---------------------------------------------------------------------------
# pgvector: table creation and insertion
# ---------------------------------------------------------------------------

def init_pgvector(conn) -> None:
    """Create pgvector extension and documents table if they don't exist."""
    with conn.cursor() as cur:
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        cur.execute(f"""
            CREATE TABLE IF NOT EXISTS documents (
                id          SERIAL PRIMARY KEY,
                category    TEXT NOT NULL,
                title       TEXT,
                content     TEXT NOT NULL,
                metadata    JSONB,
                embedding   vector({EMBEDDING_DIMENSION})
            );
        """)
        cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_documents_embedding
            ON documents USING ivfflat (embedding vector_cosine_ops)
            WITH (lists = 100);
        """)
    conn.commit()


def insert_documents(
    conn,
    documents: List[Dict[str, Any]],
    embeddings: List[List[float]],
) -> int:
    """Insert documents with their embeddings into pgvector. Returns count inserted."""
    count = 0
    with conn.cursor() as cur:
        for doc, emb in zip(documents, embeddings):
            title = (doc["metadata"].get("titulo")
                     or doc["metadata"].get("nombre")
                     or doc["metadata"].get("title", ""))
            cur.execute(
                """
                INSERT INTO documents (category, title, content, metadata, embedding)
                VALUES (%s, %s, %s, %s, %s)
                """,
                (
                    doc["category"],
                    title,
                    doc["text"],
                    json.dumps(doc["metadata"], ensure_ascii=False),
                    emb,
                ),
            )
            count += 1
    conn.commit()
    return count


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_indexing() -> None:
    """Full indexing pipeline."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    # 1. Load documents
    documents = load_indexes()
    if not documents:
        logger.error("No documents found to index.")
        return

    # 2. Generate embeddings
    logger.info("Generating embeddings for %d documents...", len(documents))
    embedding_svc = EmbeddingService()
    texts = [doc["text"] for doc in documents]
    embeddings = embedding_svc.embed_batch(texts)

    # 3. Connect to PostgreSQL and insert
    logger.info("Connecting to PostgreSQL...")
    conn = psycopg2.connect(POSTGRES_DSN)
    register_vector(conn)

    logger.info("Initializing pgvector schema...")
    init_pgvector(conn)

    # Clear table before re-indexing
    with conn.cursor() as cur:
        cur.execute("TRUNCATE TABLE documents;")
    conn.commit()

    logger.info("Inserting documents...")
    inserted = insert_documents(conn, documents, embeddings)
    logger.info("Indexing complete: %d documents inserted.", inserted)

    conn.close()


if __name__ == "__main__":
    run_indexing()
