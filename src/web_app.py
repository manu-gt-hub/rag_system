"""
web_app.py — Local web service (Flask) for the RAG System.
Runnable as: python -m src.web_app
"""

from __future__ import annotations

import logging
from flask import Flask, render_template, request, jsonify

from src.config import FLASK_HOST, FLASK_PORT, FLASK_DEBUG, TEMPLATES_DIR, CONTENT_CATEGORIES
from src.rag_pipeline import RAGPipeline

logger = logging.getLogger(__name__)

app = Flask(__name__, template_folder=str(TEMPLATES_DIR))
pipeline: RAGPipeline | None = None


def get_pipeline() -> RAGPipeline:
    global pipeline
    if pipeline is None:
        pipeline = RAGPipeline()
    return pipeline


@app.route("/")
def index():
    """Main page with question form."""
    return render_template("index.html", categories=CONTENT_CATEGORIES)


@app.route("/api/query", methods=["POST"])
def api_query():
    """API endpoint for RAG queries.

    Body JSON:
        {
            "question": "...",
            "top_k": 5,           // optional
            "category": "peliculas"  // optional
        }

    Returns:
        JSON with response, retrieved docs and metrics.
    """
    data = request.get_json(force=True)
    question = data.get("question", "").strip()

    if not question:
        return jsonify({"error": "The 'question' field is required."}), 400

    top_k = data.get("top_k")
    category = data.get("category")
    if category == "todas":
        category = None

    try:
        rag = get_pipeline()
        response = rag.query(
            question=question,
            top_k=top_k,
            category_filter=category,
        )
        return jsonify(response.to_dict())
    except Exception as e:
        logger.exception("Error processing query")
        return jsonify({"error": str(e)}), 500


@app.route("/health")
def health():
    return jsonify({"status": "ok"})


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    logger.info("Starting RAG Web UI at http://%s:%d", FLASK_HOST, FLASK_PORT)
    app.run(host=FLASK_HOST, port=FLASK_PORT, debug=FLASK_DEBUG)


if __name__ == "__main__":
    main()
