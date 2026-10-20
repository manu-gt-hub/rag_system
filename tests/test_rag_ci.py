"""
test_rag_ci.py — Pytest tests runnable from GitHub Actions.

Wraps the RAG evaluation and asserts against the acceptance
thresholds defined in doc/rubricas.md.

Runnable as:
    pytest tests/test_rag_ci.py -v
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List

import pytest

from src.config import TESTS_DIR
from src.rag_pipeline import RAGPipeline
from tests.run_evaluation import (
    eval_completitud,
    eval_formato,
    eval_precision,
    eval_relevancia,
    evaluate_single,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Acceptance thresholds (see doc/rubricas.md)
# ---------------------------------------------------------------------------
THRESHOLD_GLOBAL = 70.0          # Global accuracy >= 70%
THRESHOLD_RELEVANCIA = 60.0      # Avg. relevance >= 60%
THRESHOLD_PRECISION = 65.0       # Avg. precision >= 65%
THRESHOLD_INDIVIDUAL_PRECISION = 0.0  # No question with 0% precision


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def pipeline():
    """Initialize the RAG pipeline once for all tests in this module."""
    p = RAGPipeline()
    yield p
    p.close()


@pytest.fixture(scope="module")
def test_cases() -> List[Dict[str, Any]]:
    """Load the test question battery."""
    test_file = TESTS_DIR / "test_questions.json"
    with open(test_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["tests"]


@pytest.fixture(scope="module")
def evaluation_results(pipeline, test_cases) -> List[Dict[str, Any]]:
    """Run the full evaluation and cache results for the module."""
    results = []
    for tc in test_cases:
        result = evaluate_single(tc, pipeline)
        results.append(result)
    return results


# ---------------------------------------------------------------------------
# Individual tests per question
# ---------------------------------------------------------------------------

def _get_test_ids(test_file: Path) -> List[str]:
    """Extract IDs for parametrization (called at collection time)."""
    with open(test_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    return [t["id"] for t in data["tests"]]


class TestIndividualQuestions:
    """Each test suite question as an individual test."""

    def test_question_not_zero_precision(self, evaluation_results, test_cases):
        """No question should have 0% precision."""
        zero_precision = [
            r for r in evaluation_results
            if r["rubricas"]["precision"] <= THRESHOLD_INDIVIDUAL_PRECISION
        ]
        if zero_precision:
            failed_ids = [r["id"] for r in zero_precision]
            pytest.fail(
                f"Questions with 0% precision: {failed_ids}. "
                f"See details in results/evaluation_results.md"
            )

    def test_each_question_retrieves_docs(self, evaluation_results):
        """Each question should retrieve at least 1 document."""
        no_docs = [
            r for r in evaluation_results
            if len(r["retrieved_titles"]) == 0
        ]
        if no_docs:
            failed_ids = [r["id"] for r in no_docs]
            pytest.fail(f"Questions with no retrieved documents: {failed_ids}")

    def test_no_errors(self, evaluation_results):
        """No question should produce execution errors."""
        errors = [
            r for r in evaluation_results
            if r["error"] is not None
        ]
        if errors:
            details = [(r["id"], r["error"]) for r in errors]
            pytest.fail(f"Questions with errors: {details}")


# ---------------------------------------------------------------------------
# Aggregate accuracy tests
# ---------------------------------------------------------------------------

class TestAggregateAccuracy:
    """Verify global thresholds per doc/rubricas.md."""

    def test_global_accuracy_above_threshold(self, evaluation_results):
        """Accuracy global >= 70%."""
        n = len(evaluation_results)
        avg_global = sum(r["rubricas"]["global"] for r in evaluation_results) / n
        assert avg_global >= THRESHOLD_GLOBAL, (
            f"Global accuracy {avg_global:.1f}% < threshold {THRESHOLD_GLOBAL}%"
        )

    def test_relevancia_above_threshold(self, evaluation_results):
        """Relevancia media >= 60%."""
        n = len(evaluation_results)
        avg = sum(r["rubricas"]["relevancia"] for r in evaluation_results) / n
        assert avg >= THRESHOLD_RELEVANCIA, (
            f"Avg. relevance {avg:.1f}% < threshold {THRESHOLD_RELEVANCIA}%"
        )

    def test_precision_above_threshold(self, evaluation_results):
        """Precisión media >= 65%."""
        n = len(evaluation_results)
        avg = sum(r["rubricas"]["precision"] for r in evaluation_results) / n
        assert avg >= THRESHOLD_PRECISION, (
            f"Avg. precision {avg:.1f}% < threshold {THRESHOLD_PRECISION}%"
        )

    def test_pass_rate(self, evaluation_results):
        """At least 70% of questions must pass (>= 70% global)."""
        n = len(evaluation_results)
        passed = sum(1 for r in evaluation_results if r["rubricas"]["global"] >= 70)
        rate = (passed / n) * 100
        assert rate >= 70, (
            f"Only {passed}/{n} ({rate:.0f}%) questions passed. Minimum: 70%"
        )


# ---------------------------------------------------------------------------
# Basic connectivity test (smoke test, no API keys required)
# ---------------------------------------------------------------------------

class TestSmoke:
    """Quick tests that verify configuration without calling APIs."""

    def test_test_questions_file_exists(self):
        path = TESTS_DIR / "test_questions.json"
        assert path.exists(), f"Not found: {path}"

    def test_test_questions_valid_json(self):
        path = TESTS_DIR / "test_questions.json"
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        assert "tests" in data
        assert len(data["tests"]) > 0

    def test_all_test_cases_have_required_fields(self):
        path = TESTS_DIR / "test_questions.json"
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        required = {"id", "question", "expected_keywords", "expected_docs"}
        for tc in data["tests"]:
            missing = required - set(tc.keys())
            assert not missing, (
                f"Test {tc.get('id', '?')} is missing fields: {missing}"
            )

    def test_config_loads(self):
        from src.config import (
            LLM_PROVIDER, OPENAI_API_KEY, INDEXES_DIR, CONTENT_CATEGORIES,
        )
        assert LLM_PROVIDER in ("openai", "anthropic")
        assert INDEXES_DIR.exists()
        assert len(CONTENT_CATEGORIES) > 0

    def test_indexes_exist(self):
        from src.config import INDEXES_DIR, CONTENT_CATEGORIES
        for cat in CONTENT_CATEGORIES:
            index_path = INDEXES_DIR / cat / "index.json"
            assert index_path.exists(), f"Missing index: {index_path}"
