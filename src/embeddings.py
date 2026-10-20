"""
embeddings.py — Embedding generation via OpenAI API.
Always uses OpenAI for embeddings (Anthropic does not offer an embeddings endpoint).
"""

from __future__ import annotations

import logging
from typing import List

import numpy as np
from openai import OpenAI

from src.config import OPENAI_API_KEY, OPENAI_EMBEDDING_MODEL, EMBEDDING_DIMENSION

logger = logging.getLogger(__name__)


class EmbeddingService:
    """Wrapper over the OpenAI embeddings API."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        dimension: int | None = None,
    ):
        self.api_key = api_key or OPENAI_API_KEY
        self.model = model or OPENAI_EMBEDDING_MODEL
        self.dimension = dimension or EMBEDDING_DIMENSION
        self.client = OpenAI(api_key=self.api_key)

    def embed_text(self, text: str) -> List[float]:
        """Generate an embedding for a single text."""
        response = self.client.embeddings.create(
            input=text,
            model=self.model,
            dimensions=self.dimension,
        )
        return response.data[0].embedding

    def embed_batch(self, texts: List[str], batch_size: int = 100) -> List[List[float]]:
        """Generate embeddings for a batch of texts."""
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            logger.info("Embedding batch %d-%d / %d", i, i + len(batch), len(texts))
            response = self.client.embeddings.create(
                input=batch,
                model=self.model,
                dimensions=self.dimension,
            )
            all_embeddings.extend([d.embedding for d in response.data])
        return all_embeddings

    @staticmethod
    def cosine_similarity(a: List[float], b: List[float]) -> float:
        """Cosine similarity between two vectors."""
        a_np, b_np = np.array(a), np.array(b)
        return float(np.dot(a_np, b_np) / (np.linalg.norm(a_np) * np.linalg.norm(b_np)))
