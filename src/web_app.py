"""
web_app.py — Local web service (Flask) for the RAG System.
Runnable as: python -m src.web_app
"""

from __future__ import annotations

import logging
from flask import Flask, render_template, request, jsonify

from src.config import (
    CATEGORY_LABELS,
    CONTENT_CATEGORIES,
    EMBEDDING_PROVIDER,
    FLASK_DEBUG,
    FLASK_HOST,
    FLASK_PORT,
    LLM_KEY_CONFIGURED,
    LLM_PROVIDER,
    TEMPLATES_DIR,
)
from src.rag_pipeline import RAGPipeline

logger = logging.getLogger(__name__)
# Configured at import time so it also applies under gunicorn
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
for _noisy in ("httpx", "huggingface_hub"):
    logging.getLogger(_noisy).setLevel(logging.WARNING)

app = Flask(__name__, template_folder=str(TEMPLATES_DIR))
pipeline: RAGPipeline | None = None


def get_pipeline() -> RAGPipeline:
    global pipeline
    if pipeline is None:
        pipeline = RAGPipeline()
    return pipeline


def _run_query(data: dict, history: list | None = None):
    """Shared handler for /api/query and /api/chat."""
    question = (data.get("question") or data.get("message") or "").strip()
    if not question:
        return jsonify({"error": "The 'question' field is required."}), 400

    category = data.get("category")
    if category in ("todas", ""):
        category = None
    if category and category not in CONTENT_CATEGORIES:
        return jsonify({"error": f"Unknown category: {category}"}), 400

    try:
        rag = get_pipeline()
        response = rag.query(
            question=question,
            top_k=data.get("top_k"),
            category_filter=category,
            history=history,
        )
        return jsonify(response.to_dict())
    except Exception as e:
        logger.exception("Error processing query")
        return jsonify({"error": str(e)}), 500


@app.route("/")
def index():
    """Chat page."""
    return render_template(
        "index.html",
        categories=CONTENT_CATEGORIES,
        category_labels=CATEGORY_LABELS,
        llm_provider=LLM_PROVIDER,
        llm_ready=LLM_KEY_CONFIGURED,
        embedding_provider=EMBEDDING_PROVIDER,
    )


@app.route("/api/query", methods=["POST"])
def api_query():
    """Single-question endpoint (no conversation memory).

    Body JSON:
        {
            "question": "...",
            "top_k": 5,              // optional
            "category": "peliculas"  // optional
        }

    Returns:
        JSON with response, retrieved docs and metrics.
    """
    return _run_query(request.get_json(force=True) or {})


@app.route("/api/chat", methods=["POST"])
def api_chat():
    """Chatbot endpoint with conversation memory.

    Body JSON:
        {
            "message": "...",
            "history": [{"role": "user"|"assistant", "content": "..."}],  // optional
            "category": "peliculas",                                      // optional
            "top_k": 5                                                    // optional
        }
    """
    data = request.get_json(force=True) or {}
    history = data.get("history") or []
    if not isinstance(history, list):
        return jsonify({"error": "'history' must be a list of messages."}), 400
    return _run_query(data, history=history)


@app.route("/health")
def health():
    return jsonify({"status": "ok"})


def main():
    logger.info("Starting RAG Web UI at http://%s:%d", FLASK_HOST, FLASK_PORT)
    app.run(host=FLASK_HOST, port=FLASK_PORT, debug=FLASK_DEBUG)


if __name__ == "__main__":
    main()
