"""
rag_pipeline.py — RAG pipeline orchestrator: Retriever → Generator.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from src.retriever import Retriever, RetrievedDoc
from src.generator import Generator

logger = logging.getLogger(__name__)


@dataclass
class RAGResponse:
    """Full response from the RAG pipeline."""
    query: str
    answer: str
    retrieved_docs: List[RetrievedDoc]
    elapsed_seconds: float
    provider: str
    top_score: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "answer": self.answer,
            "elapsed_seconds": round(self.elapsed_seconds, 3),
            "provider": self.provider,
            "top_score": round(self.top_score, 4),
            "num_docs_retrieved": len(self.retrieved_docs),
            "docs": [
                {
                    "title": d.title,
                    "category": d.category,
                    "score": round(d.score, 4),
                }
                for d in self.retrieved_docs
            ],
        }


class RAGPipeline:
    """Main pipeline: receives question, retrieves context, generates response."""

    def __init__(self, provider: str | None = None):
        self.retriever = Retriever()
        self.generator = Generator(provider=provider)
        self.provider = self.generator.provider

    def query(
        self,
        question: str,
        top_k: int | None = None,
        category_filter: str | None = None,
    ) -> RAGResponse:
        """Execute the full pipeline.

        Args:
            question: Natural-language question.
            top_k: Number of docs to retrieve.
            category_filter: Filter by category.

        Returns:
            RAGResponse with answer, docs and metrics.
        """
        t0 = time.time()

        # 1. Retrieve relevant documents
        docs = self.retriever.search(
            query=question,
            top_k=top_k,
            category_filter=category_filter,
        )

        # 2. Generate response with context
        if docs:
            answer = self.generator.generate(question, docs)
            top_score = docs[0].score
        else:
            answer = "No relevant documents found for your question."
            top_score = 0.0

        elapsed = time.time() - t0

        logger.info(
            "Pipeline completed in %.2fs — %d docs, top_score=%.3f",
            elapsed, len(docs), top_score,
        )

        return RAGResponse(
            query=question,
            answer=answer,
            retrieved_docs=docs,
            elapsed_seconds=elapsed,
            provider=self.provider,
            top_score=top_score,
        )

    def close(self):
        self.retriever.close()
