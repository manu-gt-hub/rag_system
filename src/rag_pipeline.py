"""
rag_pipeline.py — RAG pipeline orchestrator: Retriever → Generator.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, List

from src.config import CHAT_HISTORY_MESSAGES
from src.retriever import Retriever, RetrievedDoc
from src.generator import Generator
from src.text_utils import detect_category, query_terms

logger = logging.getLogger(__name__)

# A follow-up with this many meaningful terms or fewer ("any more recent
# ones?") is searched together with the previous user message.
FOLLOW_UP_MAX_TERMS = 3


@dataclass
class RAGResponse:
    """Full response from the RAG pipeline."""
    query: str
    answer: str
    retrieved_docs: List[RetrievedDoc]
    elapsed_seconds: float
    provider: str
    top_score: float = 0.0
    category: str | None = None
    retrieval_query: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "answer": self.answer,
            "elapsed_seconds": round(self.elapsed_seconds, 3),
            "provider": self.provider,
            "category": self.category,
            "retrieval_query": self.retrieval_query,
            "top_score": round(self.top_score, 4),
            "num_docs_retrieved": len(self.retrieved_docs),
            "docs": [
                {
                    "title": d.title,
                    "category": d.category,
                    "score": round(d.score, 4),
                    "keyword_hit": d.keyword_hit,
                    "year": d.metadata.get("year"),
                    "genre": d.metadata.get("genre"),
                    "rating": d.metadata.get("rating"),
                    "poster_url": d.metadata.get("poster_url"),
                }
                for d in self.retrieved_docs
            ],
        }


def build_retrieval_query(question: str, history: List[Dict[str, str]] | None) -> str:
    """Enrich short follow-up questions with the previous user message."""
    if not history or len(query_terms(question)) > FOLLOW_UP_MAX_TERMS:
        return question
    previous = [m["content"] for m in history if m.get("role") == "user" and m.get("content")]
    return f"{previous[-1]} {question}" if previous else question


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
        history: List[Dict[str, str]] | None = None,
        auto_category: bool = True,
    ) -> RAGResponse:
        """Execute the full pipeline.

        Args:
            question: Natural-language question.
            top_k: Number of docs to retrieve.
            category_filter: Filter by category. If None and auto_category is
                set, it is inferred when the question names exactly one category.
            history: Previous chat messages (oldest first).
            auto_category: Infer the category from the question.

        Returns:
            RAGResponse with answer, docs and metrics.
        """
        t0 = time.time()
        history = (history or [])[-CHAT_HISTORY_MESSAGES:] if CHAT_HISTORY_MESSAGES else []

        retrieval_query = build_retrieval_query(question, history)
        if category_filter is None and auto_category:
            category_filter = detect_category(retrieval_query)

        # 1. Retrieve relevant documents
        docs = self.retriever.search(
            query=retrieval_query,
            top_k=top_k,
            category_filter=category_filter,
        )

        # 2. Generate response with context
        if docs:
            answer = self.generator.generate(question, docs, history=history)
            top_score = max(d.score for d in docs)
        else:
            answer = ("I couldn't find anything related in your catalog. "
                      "Try rephrasing the question or changing the category.")
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
            category=category_filter,
            retrieval_query=retrieval_query,
        )

    def close(self):
        self.retriever.close()
