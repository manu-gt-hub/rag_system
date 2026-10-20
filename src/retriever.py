"""
retriever.py — Semantic search against pgvector.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import psycopg2
from pgvector.psycopg2 import register_vector

from src.config import POSTGRES_DSN, TOP_K, SIMILARITY_THRESHOLD
from src.embeddings import EmbeddingService

logger = logging.getLogger(__name__)


@dataclass
class RetrievedDoc:
    """Retrieved document with its similarity score."""
    id: int
    category: str
    title: str
    content: str
    metadata: Dict[str, Any]
    score: float


class Retriever:
    """Semantic search over pgvector."""

    def __init__(
        self,
        dsn: str | None = None,
        top_k: int | None = None,
        threshold: float | None = None,
    ):
        self.dsn = dsn or POSTGRES_DSN
        self.top_k = top_k or TOP_K
        self.threshold = threshold or SIMILARITY_THRESHOLD
        self.embedding_svc = EmbeddingService()
        self._conn = None

    def _get_conn(self):
        if self._conn is None or self._conn.closed:
            self._conn = psycopg2.connect(self.dsn)
            register_vector(self._conn)
        return self._conn

    def search(
        self,
        query: str,
        top_k: int | None = None,
        category_filter: str | None = None,
    ) -> List[RetrievedDoc]:
        """Search for documents similar to the query.

        Args:
            query: User question.
            top_k: Number of results (override).
            category_filter: Filter by category (peliculas, libros, etc.).

        Returns:
            List of RetrievedDoc sorted by descending similarity.
        """
        k = top_k or self.top_k
        query_embedding = self.embedding_svc.embed_text(query)

        conn = self._get_conn()
        with conn.cursor() as cur:
            if category_filter:
                cur.execute(
                    """
                    SELECT id, category, title, content, metadata,
                           1 - (embedding <=> %s::vector) AS score
                    FROM documents
                    WHERE category = %s
                    ORDER BY embedding <=> %s::vector
                    LIMIT %s
                    """,
                    (query_embedding, category_filter, query_embedding, k),
                )
            else:
                cur.execute(
                    """
                    SELECT id, category, title, content, metadata,
                           1 - (embedding <=> %s::vector) AS score
                    FROM documents
                    ORDER BY embedding <=> %s::vector
                    LIMIT %s
                    """,
                    (query_embedding, query_embedding, k),
                )

            rows = cur.fetchall()

        results = []
        for row in rows:
            score = float(row[5])
            if score >= self.threshold:
                results.append(RetrievedDoc(
                    id=row[0],
                    category=row[1],
                    title=row[2],
                    content=row[3],
                    metadata=row[4] if isinstance(row[4], dict) else {},
                    score=score,
                ))

        logger.info(
            "Query: '%s' -> %d results (top score: %.3f)",
            query[:60],
            len(results),
            results[0].score if results else 0.0,
        )
        return results

    def close(self):
        if self._conn and not self._conn.closed:
            self._conn.close()
