"""
run_evaluation.py — Test harness: runs a battery of questions, evaluates with
rubrics and generates an accuracy report in results/evaluation_results.md.

Runnable as: python -m tests.run_evaluation
"""

from __future__ import annotations

import json
import logging
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

from src.config import RESULTS_DIR, TESTS_DIR
from src.rag_pipeline import RAGPipeline

logger = logging.getLogger(__name__)

# Phrases that indicate an erroneous rejection
REJECTION_PHRASES = [
    "no tengo información",
    "no encuentro",
    "no dispongo",
    "no hay datos",
    "no se encontraron",
]


# ---------------------------------------------------------------------------
# Per-rubric evaluation functions
# ---------------------------------------------------------------------------

def eval_relevancia(retrieved_titles: List[str], expected_docs: List[str]) -> float:
    """Rubric 1: Retrieval relevance (0.0 - 1.0)."""
    if not expected_docs:
        return 1.0
    found = 0
    for expected in expected_docs:
        expected_lower = expected.lower()
        for title in retrieved_titles:
            if expected_lower in title.lower():
                found += 1
                break
    return found / len(expected_docs)


def eval_precision(answer: str, expected_keywords: List[str]) -> float:
    """Rubric 2: Answer precision (0.0 - 1.0)."""
    if not expected_keywords:
        return 1.0
    answer_lower = answer.lower()
    found = sum(1 for kw in expected_keywords if kw.lower() in answer_lower)
    return found / len(expected_keywords)


def eval_completitud(answer: str, has_expected_data: bool) -> float:
    """Rubric 3: Completeness (0.0 - 1.0)."""
    # Penalize if it rejects when data is available
    answer_lower = answer.lower()
    has_rejection = any(phrase in answer_lower for phrase in REJECTION_PHRASES)

    if has_expected_data and has_rejection:
        return 0.0

    # Minimum reasonable length (>30 characters)
    if len(answer.strip()) < 30:
        return 0.25

    # Adequate length (>100 characters = complete answer)
    if len(answer.strip()) >= 100:
        return 1.0

    return 0.75


def eval_formato(answer: str) -> float:
    """Rubric 4: Format and clarity (0.0 - 1.0)."""
    score = 0.0

    # Spanish check: look for typical words
    spanish_indicators = ["de", "el", "la", "en", "que", "es", "con", "por"]
    answer_words = answer.lower().split()
    spanish_count = sum(1 for w in spanish_indicators if w in answer_words)
    if spanish_count >= 3:
        score += 0.5

    # Reasonable length (neither too short nor excessively long)
    length = len(answer.strip())
    if 50 <= length <= 2000:
        score += 0.25

    # Has structure (periods, commas, paragraphs)
    if "." in answer and "," in answer:
        score += 0.25

    return min(score, 1.0)


def evaluate_single(
    test_case: Dict[str, Any],
    pipeline: RAGPipeline,
) -> Dict[str, Any]:
    """Evaluate a test case and return metrics."""
    question = test_case["question"]
    category = test_case.get("category")
    expected_keywords = test_case.get("expected_keywords", [])
    expected_docs = test_case.get("expected_docs", [])

    # Run pipeline
    t0 = time.time()
    try:
        response = pipeline.query(
            question=question,
            category_filter=category,
        )
        answer = response.answer
        retrieved_titles = [d.title for d in response.retrieved_docs]
        top_score = response.top_score
        error = None
    except Exception as e:
        answer = ""
        retrieved_titles = []
        top_score = 0.0
        error = str(e)
    elapsed = time.time() - t0

    # Evaluate rubrics
    relevancia = eval_relevancia(retrieved_titles, expected_docs)
    precision = eval_precision(answer, expected_keywords)
    completitud = eval_completitud(answer, bool(expected_docs))
    formato = eval_formato(answer)

    # Weighted global score
    global_score = (
        relevancia * 0.30
        + precision * 0.35
        + completitud * 0.20
        + formato * 0.15
    )

    return {
        "id": test_case["id"],
        "question": question,
        "category": category,
        "answer": answer,
        "retrieved_titles": retrieved_titles,
        "top_score": top_score,
        "elapsed": round(elapsed, 3),
        "error": error,
        "rubricas": {
            "relevancia": round(relevancia * 100, 1),
            "precision": round(precision * 100, 1),
            "completitud": round(completitud * 100, 1),
            "formato": round(formato * 100, 1),
            "global": round(global_score * 100, 1),
        },
    }


# ---------------------------------------------------------------------------
# Markdown report generation
# ---------------------------------------------------------------------------

def generate_report(results: List[Dict], elapsed_total: float) -> str:
    """Generate the markdown evaluation report."""
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    n = len(results)

    # Aggregated metrics
    avg = lambda key: sum(r["rubricas"][key] for r in results) / n if n else 0
    avg_relevancia = avg("relevancia")
    avg_precision = avg("precision")
    avg_completitud = avg("completitud")
    avg_formato = avg("formato")
    avg_global = avg("global")

    passed = sum(1 for r in results if r["rubricas"]["global"] >= 70)
    failed = n - passed

    lines = [
        f"# RAG System Evaluation Report",
        f"",
        f"**Date**: {now}",
        f"**Tests executed**: {n}",
        f"**Total time**: {elapsed_total:.1f}s",
        f"**LLM Provider**: configured in .env",
        f"",
        f"## Accuracy Summary",
        f"",
        f"| Metric | Value | Threshold |",
        f"|---|---|---|",
        f"| **Global Accuracy** | **{avg_global:.1f}%** | >= 70% |",
        f"| Avg. Relevance | {avg_relevancia:.1f}% | >= 60% |",
        f"| Avg. Precision | {avg_precision:.1f}% | >= 65% |",
        f"| Avg. Completeness | {avg_completitud:.1f}% | - |",
        f"| Avg. Format | {avg_formato:.1f}% | - |",
        f"| Tests passed (>= 70%) | {passed}/{n} |",
        f"| Tests failed (< 70%) | {failed}/{n} |",
        f"",
        f"## Detail by Test",
        f"",
        f"| ID | Category | Relevance | Precision | Completeness | Format | **Global** | Time |",
        f"|---|---|---|---|---|---|---|---|",
    ]

    for r in results:
        rb = r["rubricas"]
        cat = r["category"] or "todas"
        status = "✅" if rb["global"] >= 70 else "❌"
        lines.append(
            f"| {r['id']} | {cat} | {rb['relevancia']}% | {rb['precision']}% | "
            f"{rb['completitud']}% | {rb['formato']}% | **{rb['global']}%** {status} | {r['elapsed']}s |"
        )

    lines.extend([
        f"",
        f"## Response Details",
        f"",
    ])

    for r in results:
        rb = r["rubricas"]
        lines.extend([
            f"### {r['id']}: {r['question']}",
            f"",
            f"- **Category**: {r['category'] or 'all'}",
            f"- **Global score**: {rb['global']}%",
            f"- **Top retrieval score**: {r['top_score']:.4f}",
            f"- **Retrieved docs**: {', '.join(r['retrieved_titles']) or 'none'}",
            f"- **Time**: {r['elapsed']}s",
            f"",
            f"**Answer**:",
            f"",
            f"> {r['answer'][:500]}{'...' if len(r['answer']) > 500 else ''}",
            f"",
        ])
        if r["error"]:
            lines.append(f"⚠️ Error: {r['error']}")
            lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_evaluation() -> None:
    """Run the full test battery and generate a report."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    # Load tests
    test_file = TESTS_DIR / "test_questions.json"
    with open(test_file, "r", encoding="utf-8") as f:
        test_data = json.load(f)
    tests = test_data["tests"]
    logger.info("Loaded %d tests from %s", len(tests), test_file)

    # Initialize pipeline
    pipeline = RAGPipeline()

    # Run evaluation
    results = []
    t0 = time.time()
    for i, test_case in enumerate(tests, 1):
        logger.info("[%d/%d] Evaluating %s: %s", i, len(tests), test_case["id"], test_case["question"])
        result = evaluate_single(test_case, pipeline)
        results.append(result)
        logger.info(
            "  -> Global: %s%% | Rel: %s%% | Prec: %s%%",
            result["rubricas"]["global"],
            result["rubricas"]["relevancia"],
            result["rubricas"]["precision"],
        )
    elapsed_total = time.time() - t0

    pipeline.close()

    # Generate report
    report = generate_report(results, elapsed_total)

    # Save
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = RESULTS_DIR / "evaluation_results.md"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(report)

    logger.info("")
    logger.info("=" * 50)
    logger.info("EVALUATION COMPLETED")
    logger.info("=" * 50)
    avg_global = sum(r["rubricas"]["global"] for r in results) / len(results)
    logger.info("Global accuracy: %.1f%%", avg_global)
    logger.info("Report saved to: %s", output_path)
    logger.info("=" * 50)


if __name__ == "__main__":
    run_evaluation()
