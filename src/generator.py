"""
generator.py — LLM response generation (OpenAI / Anthropic).
"""

from __future__ import annotations

import logging
from typing import Dict, List

from src.config import (
    LLM_PROVIDER,
    OPENAI_API_KEY,
    OPENAI_CHAT_MODEL,
    ANTHROPIC_API_KEY,
    ANTHROPIC_CHAT_MODEL,
)

logger = logging.getLogger(__name__)

NO_KEY_MESSAGE = (
    "No LLM API key configured for provider '{provider}'. Set OPENAI_API_KEY or "
    "ANTHROPIC_API_KEY in .env and restart (docker compose up -d)."
)

SYSTEM_PROMPT = """\
You are the recommendation assistant for the user's personal media catalog:
movies, TV series, books, music, video games and documentaries. You answer
questions and recommend content using ONLY the documents in the context,
which are what the user has in their collection.

Rules:
- Reply in the language the user writes in (English by default), in a
  friendly and concise tone. Catalog data is in Spanish: translate genres and
  synopses when needed, but never translate titles.
- Quote titles exactly as they appear in the context, in **bold**.
- For a recommendation, suggest 1 to 5 titles from the context that fit the
  request. For each one give its type (movie, book...), year if known and one
  sentence on why it fits (genre, director/author, cast, synopsis themes).
  Include the rating when available.
- For a specific question (who directed it, what it is about, its rating...),
  answer it directly with the data in the context.
- Ignore context documents that do not fit the request.
- Never make up titles or data that are not in the context.
- Only if no document is useful, say you don't have enough information in
  their catalog and suggest how to rephrase the question.
- You may end with a short question to refine the recommendation.
"""


def _build_context(docs: list) -> str:
    """Format retrieved documents as context for the prompt."""
    parts = []
    for i, doc in enumerate(docs, 1):
        parts.append(f"--- Document {i} (relevance: {doc.score:.3f}) ---")
        parts.append(doc.content)
        parts.append("")
    return "\n".join(parts)


class Generator:
    """Response generator using OpenAI or Anthropic."""

    def __init__(self, provider: str | None = None):
        self.provider = (provider or LLM_PROVIDER).lower()

        if self.provider == "openai":
            if not OPENAI_API_KEY:
                raise ValueError(NO_KEY_MESSAGE.format(provider=self.provider))
            from openai import OpenAI
            self.client = OpenAI(api_key=OPENAI_API_KEY)
            self.model = OPENAI_CHAT_MODEL
        elif self.provider == "anthropic":
            if not ANTHROPIC_API_KEY:
                raise ValueError(NO_KEY_MESSAGE.format(provider=self.provider))
            import anthropic
            self.client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
            self.model = ANTHROPIC_CHAT_MODEL
        else:
            raise ValueError(f"Unsupported LLM provider: {self.provider}")

    def generate(
        self,
        query: str,
        retrieved_docs: list,
        history: List[Dict[str, str]] | None = None,
    ) -> str:
        """Generate a response using the LLM with retrieved context.

        Args:
            query: User question.
            retrieved_docs: List of RetrievedDoc from the retriever.
            history: Previous chat turns as [{"role": "user"|"assistant", "content": ...}].

        Returns:
            Plain-text (light markdown) response.
        """
        context = _build_context(retrieved_docs)
        user_message = f"""Context retrieved from my catalog:\n{context}\n\nQuestion: {query}"""
        messages = [
            {"role": m["role"], "content": m["content"]}
            for m in (history or [])
            if m.get("role") in ("user", "assistant") and m.get("content")
        ]
        messages.append({"role": "user", "content": user_message})

        if self.provider == "openai":
            return self._generate_openai(messages)
        else:
            return self._generate_anthropic(messages)

    def _generate_openai(self, messages: List[Dict[str, str]]) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "system", "content": SYSTEM_PROMPT}, *messages],
            temperature=0.3,
            max_tokens=1024,
        )
        return response.choices[0].message.content.strip()

    def _generate_anthropic(self, messages: List[Dict[str, str]]) -> str:
        # Anthropic requires the conversation to start with a user turn
        while messages and messages[0]["role"] != "user":
            messages = messages[1:]
        response = self.client.messages.create(
            model=self.model,
            system=SYSTEM_PROMPT,
            messages=messages,
            temperature=0.3,
            max_tokens=1024,
        )
        return response.content[0].text.strip()
