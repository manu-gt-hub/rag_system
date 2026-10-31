"""
embeddings.py — Embedding generation.

Two backends (EMBEDDING_PROVIDER):
  - openai: text-embedding-3-small via API (1536 dims).
  - local:  multilingual sentence-transformers model run with fastembed/ONNX
            (384 dims, CPU only). Used when there is no OpenAI key, since
            Anthropic does not offer an embeddings endpoint.
Indexing and querying must use the same backend.
"""

from __future__ import annotations

import logging
from typing import List

import numpy as np

from src.config import (
    EMBEDDING_DIMENSION,
    EMBEDDING_PROVIDER,
    LOCAL_EMBEDDING_MODEL,
    OPENAI_API_KEY,
    OPENAI_EMBEDDING_MODEL,
)

logger = logging.getLogger(__name__)


class EmbeddingService:
    """Wrapper over the configured embeddings backend."""

    def __init__(
        self,
        provider: str | None = None,
        api_key: str | None = None,
        model: str | None = None,
        dimension: int | None = None,
    ):
        self.provider = provider or EMBEDDING_PROVIDER
        self.dimension = dimension or EMBEDDING_DIMENSION

        if self.provider == "openai":
            from openai import OpenAI
            self.model = model or OPENAI_EMBEDDING_MODEL
            self.client = OpenAI(api_key=api_key or OPENAI_API_KEY)
        elif self.provider == "local":
            from fastembed import TextEmbedding
            self.model = model or LOCAL_EMBEDDING_MODEL
            logger.info("Loading local embedding model %s ...", self.model)
            self.client = TextEmbedding(model_name=self.model)
        else:
            raise ValueError(f"Unsupported embedding provider: {self.provider}")

    def embed_text(self, text: str) -> List[float]:
        """Generate an embedding for a single text."""
        return self.embed_batch([text])[0]

    def embed_batch(self, texts: List[str], batch_size: int = 100) -> List[List[float]]:
        """Generate embeddings for a batch of texts."""
        texts = [t if t.strip() else "-" for t in texts]  # APIs reject empty inputs
        if self.provider == "local":
            return [v.tolist() for v in self.client.embed(texts, batch_size=32)]

        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            if len(texts) > batch_size:
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
