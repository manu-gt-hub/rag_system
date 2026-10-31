"""
retriever.py — Hybrid semantic search against pgvector.

Two candidate lists are computed inside PostgreSQL:
  1. Vector search: cosine distance between the query embedding and the
     document embeddings (HNSW index).
  2. Full-text search: Spanish tsvector match on the meaningful query terms
     (titles, people and genres weigh more than synopsis text).
Both lists are merged with Reciprocal Rank Fusion (RRF), plus a bonus for
documents containing every term of a multi-term query, so exact names
("Danny Boyle", "Nicolas Cage") are found even when the embedding of a
short question is not discriminative enough.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List

import psycopg2
from pgvector.psycopg2 import register_vector

from src.config import (
    EMBEDDING_DIMENSION,
    POSTGRES_DSN,
    RETRIEVAL_CANDIDATES,
    SIMILARITY_THRESHOLD,
    TOP_K,
)
from src.embeddings import EmbeddingService
from src.text_utils import query_terms

logger = logging.getLogger(__name__)

RRF_K = 60  # standard RRF damping constant
ALL_TERMS_BONUS = 1.0 / RRF_K  # worth as much as a first place in one list

HYBRID_SQL = f"""
WITH vec AS (
    SELECT id, row_number() OVER (ORDER BY embedding <=> %(emb)s::vector({EMBEDDING_DIMENSION})) AS r
    FROM documents
    WHERE %(cat)s::text IS NULL OR category = %(cat)s
    ORDER BY embedding <=> %(emb)s::vector({EMBEDDING_DIMENSION})
    LIMIT %(n)s
),
fts AS (
    -- Documents matching ALL terms first, then any term, by weighted rank
    SELECT id, tsv @@ q_all AS all_terms,
           row_number() OVER (ORDER BY tsv @@ q_all DESC, ts_rank_cd(tsv, q) DESC) AS r
    FROM documents,
         to_tsquery('spanish', %(tsq)s) AS q,
         to_tsquery('spanish', %(tsq_all)s) AS q_all
    WHERE %(tsq)s <> '' AND tsv @@ q
      AND (%(cat)s::text IS NULL OR category = %(cat)s)
    ORDER BY tsv @@ q_all DESC, ts_rank_cd(tsv, q) DESC
    LIMIT %(n)s
)
SELECT d.id, d.category, d.title, d.content, d.metadata,
       1 - (d.embedding <=> %(emb)s::vector({EMBEDDING_DIMENSION})) AS score,
       COALESCE(1.0 / ({RRF_K} + vec.r), 0) + COALESCE(1.0 / ({RRF_K} + fts.r), 0)
       -- bonus for matching every term of a multi-term query ("danny & boyle")
       + CASE WHEN %(multi)s AND fts.all_terms THEN {ALL_TERMS_BONUS} ELSE 0 END AS rrf,
       fts.id IS NOT NULL AS keyword_hit
FROM documents d
LEFT JOIN vec ON vec.id = d.id
LEFT JOIN fts ON fts.id = d.id
WHERE vec.id IS NOT NULL OR fts.id IS NOT NULL
ORDER BY rrf DESC
LIMIT %(k)s
"""


@dataclass
class RetrievedDoc:
    """Retrieved document with its similarity score."""
    id: int
    category: str
    title: str
    content: str
    metadata: Dict[str, Any]
    score: float          # cosine similarity (0-1)
    rrf: float = 0.0      # fused ranking score
    keyword_hit: bool = False


def build_tsquery(text: str, operator: str = "|") -> str:
    """OR (or AND) query of the meaningful terms (each term is [a-z0-9]+, safe for to_tsquery)."""
    return f" {operator} ".join(query_terms(text))


class Retriever:
    """Hybrid search over pgvector."""

    def __init__(
        self,
        dsn: str | None = None,
        top_k: int | None = None,
        threshold: float | None = None,
    ):
        self.dsn = dsn or POSTGRES_DSN
        self.top_k = top_k or TOP_K
        self.threshold = SIMILARITY_THRESHOLD if threshold is None else threshold
        self.embedding_svc = EmbeddingService()
        self._conn = None

    def _get_conn(self):
        if self._conn is None or self._conn.closed:
            self._conn = psycopg2.connect(self.dsn)
            self._conn.autocommit = True
            register_vector(self._conn)
            with self._conn.cursor() as cur:
                # Wider HNSW search so category filtering still leaves enough rows
                cur.execute("SET hnsw.ef_search = 200;")
        return self._conn

    def search(
        self,
        query: str,
        top_k: int | None = None,
        category_filter: str | None = None,
    ) -> List[RetrievedDoc]:
        """Search for documents relevant to the query.

        Args:
            query: User question (or question enriched with chat context).
            top_k: Number of results (override).
            category_filter: Filter by category (peliculas, libros, etc.).

        Returns:
            List of RetrievedDoc sorted by descending relevance (RRF).
        """
        k = top_k or self.top_k
        query_embedding = self.embedding_svc.embed_text(query)
        params = {
            "emb": query_embedding,
            "cat": category_filter,
            "tsq": build_tsquery(query),
            "tsq_all": build_tsquery(query, "&"),
            "multi": len(query_terms(query)) > 1,
            "n": max(RETRIEVAL_CANDIDATES, k),
            "k": k * 3,  # extra rows so the threshold filter can drop some
        }

        conn = self._get_conn()
        try:
            with conn.cursor() as cur:
                cur.execute(HYBRID_SQL, params)
                rows = cur.fetchall()
        except psycopg2.errors.UndefinedTable:
            raise RuntimeError(
                "The 'documents' table does not exist. Run the indexer first: "
                "python -m src.indexer"
            )
        except psycopg2.errors.DataException as e:
            raise RuntimeError(
                f"Embedding dimension mismatch ({e.pgerror.strip()}). The index was built "
                "with a different embedding provider; re-run: python -m src.indexer"
            )

        results = []
        for row in rows:
            score = float(row[5])
            keyword_hit = bool(row[7])
            # Keyword matches are kept even with a low cosine score
            if score >= self.threshold or keyword_hit:
                results.append(RetrievedDoc(
                    id=row[0],
                    category=row[1],
                    title=row[2],
                    content=row[3],
                    metadata=row[4] if isinstance(row[4], dict) else {},
                    score=score,
                    rrf=float(row[6]),
                    keyword_hit=keyword_hit,
                ))
        results = results[:k]

        logger.info(
            "Query: '%s' [%s] -> %d results (top: %s)",
            query[:60],
            category_filter or "all",
            len(results),
            results[0].title if results else "-",
        )
        return results

    def close(self):
        if self._conn and not self._conn.closed:
            self._conn.close()
