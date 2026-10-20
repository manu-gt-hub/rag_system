# Evaluation Rubrics — RAG System

This document defines the objective criteria for evaluating the quality of
RAG system responses. Each test question is evaluated according to these rubrics.

## 1. Retrieval Relevance (0-100%)

| Score | Criteria |
|---|---|
| 100% | All retrieved documents are directly relevant to the question |
| 75% | Most documents are relevant, minor noise |
| 50% | At least half of the documents are relevant |
| 25% | Only 1 relevant document among those retrieved |
| 0% | None of the retrieved documents is relevant |

**How it is measured**: Retrieved document titles are compared against the
`expected_docs` defined in the test. Calculated as `docs_found / docs_expected`.

## 2. Answer Precision (0-100%)

| Score | Criteria |
|---|---|
| 100% | The answer contains ALL expected keywords and is factually correct |
| 75% | Contains most expected keywords (>75%) |
| 50% | Contains at least half of the expected keywords |
| 25% | Contains some keywords but is missing critical information |
| 0% | Contains no expected keywords or is incorrect |

**How it is measured**: The `expected_keywords` from the test are searched in the
generated answer (case-insensitive). Calculated as `keywords_found / keywords_expected`.

## 3. Completeness (0-100%)

| Score | Criteria |
|---|---|
| 100% | Answers all aspects of the question with sufficient detail |
| 75% | Answers the main topic but omits secondary details |
| 50% | Partial answer, missing important information |
| 25% | Very incomplete answer |
| 0% | Does not answer the question or erroneously says "no information available" |

**How it is measured**: Verifies the answer does not contain rejection phrases
("no tengo información", "no encuentro") when data is actually available.
Also checks minimum expected length.

## 4. Format and Clarity (0-100%)

| Score | Criteria |
|---|---|
| 100% | Response in Spanish, well structured, correctly cites titles |
| 75% | Good structure but with minor formatting issues |
| 50% | Readable but disorganized |
| 25% | Hard to follow |
| 0% | Incoherent or in the wrong language |

**How it is measured**: Checks for Spanish language, presence of cited titles,
and reasonable length (neither too short nor excessively long).

## Global Score

The global score for each question is calculated as a weighted average:

```
Score = (Relevance * 0.30) + (Precision * 0.35) + (Completeness * 0.20) + (Format * 0.15)
```

## Acceptance Thresholds

| Metric | Minimum acceptable threshold |
|---|---|
| Global accuracy | >= 70% |
| Avg. relevance | >= 60% |
| Avg. precision | >= 65% |
| No question with | 0% precision |
