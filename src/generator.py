"""
generator.py — LLM response generation (OpenAI / Anthropic).
"""

from __future__ import annotations

import logging
from typing import List

from src.config import (
    LLM_PROVIDER,
    OPENAI_API_KEY,
    OPENAI_CHAT_MODEL,
    ANTHROPIC_API_KEY,
    ANTHROPIC_CHAT_MODEL,
)

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """\
Eres un asistente experto en catálogo multimedia. Respondes preguntas sobre
películas, libros, series, música, videojuegos y documentales usando SOLO la
información proporcionada en el contexto.

Reglas:
- Responde en español.
- Cita títulos exactos cuando sea posible.
- Si no encuentras la información en el contexto, di "No tengo información
  suficiente para responder".
- Sé conciso pero completo.
- Incluye valoraciones/ratings cuando estén disponibles.
"""


def _build_context(docs: list) -> str:
    """Format retrieved documents as context for the prompt."""
    parts = []
    for i, doc in enumerate(docs, 1):
        parts.append(f"--- Documento {i} (score: {doc.score:.3f}) ---")
        parts.append(doc.content)
        parts.append("")
    return "\n".join(parts)


class Generator:
    """Response generator using OpenAI or Anthropic."""

    def __init__(self, provider: str | None = None):
        self.provider = (provider or LLM_PROVIDER).lower()

        if self.provider == "openai":
            from openai import OpenAI
            self.client = OpenAI(api_key=OPENAI_API_KEY)
            self.model = OPENAI_CHAT_MODEL
        elif self.provider == "anthropic":
            import anthropic
            self.client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
            self.model = ANTHROPIC_CHAT_MODEL
        else:
            raise ValueError(f"Unsupported LLM provider: {self.provider}")

    def generate(self, query: str, retrieved_docs: list) -> str:
        """Generate a response using the LLM with retrieved context.

        Args:
            query: User question.
            retrieved_docs: List of RetrievedDoc from the retriever.

        Returns:
            Plain-text response.
        """
        context = _build_context(retrieved_docs)
        user_message = f"""Contexto recuperado:\n{context}\n\nPregunta: {query}"""

        if self.provider == "openai":
            return self._generate_openai(user_message)
        else:
            return self._generate_anthropic(user_message)

    def _generate_openai(self, user_message: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
            temperature=0.3,
            max_tokens=1024,
        )
        return response.choices[0].message.content.strip()

    def _generate_anthropic(self, user_message: str) -> str:
        response = self.client.messages.create(
            model=self.model,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_message}],
            temperature=0.3,
            max_tokens=1024,
        )
        return response.content[0].text.strip()
