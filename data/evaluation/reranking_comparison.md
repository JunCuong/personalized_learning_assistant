# Reranking comparison

Eval file: `eval_questions_revised.csv`

## Baseline (FAISS only)

| Metric | Value |
|--------|-------|
| Hit@3 | 0.6666666666666666 |
| Hit@5 | 0.8 |
| MRR | 0.5744444444444444 |

## Lightweight rerank (candidate_k=10, semantic 0.75 + keyword 0.25)

| Metric | Value |
|--------|-------|
| Hit@3 | 0.7333333333333333 |
| Hit@5 | 0.8666666666666667 |
| MRR | 0.5777777777777777 |

**Reranking improved:** True

**Recommendation:** enable_reranking_optional: Reranking improved metrics without regressions. Keep disabled by default in app; users may opt in via checkbox or --use-reranking.
