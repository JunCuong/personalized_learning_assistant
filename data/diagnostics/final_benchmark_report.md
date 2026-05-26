# Final benchmark report

Generated: 2026-05-26T04:12:17.598494+00:00

## 1. System status

| Field | Value |
|-------|-------|
| Timestamp | 2026-05-26T04:12:17.598494+00:00 |
| Python | 3.13.5 |
| Gemini model | gemini-2.5-flash |
| Gemini keys configured | 4 |
| Local fallback allowed | False |
| Gemini quota exhausted | True |
| Local fallback used | False |
| Reranking enabled in run | True |
| Retrieval-only mode | False |

**Note:** Generation latency could not be measured reliably because all Gemini keys were quota-exhausted.

## 2. Resource loading times

| Resource | Cold (s) | Warm (s) |
|----------|----------|----------|
| Embedder | 6.369 | 0.129 |
| Retriever / FAISS | 0.003 | 0.004 |
| LLM client init | 4.142 | — |

## 3. Retrieval latency

Average without reranking: **0.043s**  
Average with reranking: **0.039s**

| Query | No rerank (s) | With rerank (s) | Top1 (no rerank) | Top1 (rerank) |
|-------|---------------|-----------------|------------------|---------------|
| What is retrieval augmented generation? | 0.049 | 0.043 | RAG.pdf p13 | RAG.pdf p13 |
| What is MLOps? | 0.041 | 0.036 | Scalable_MLOps_Architecture.pptx.pdf p3 | Scalable_MLOps_Architecture.pptx.pdf p3 |
| What is Kolmogorov-Arnold Network? | 0.038 | 0.038 | Kolmogorov–Arnold Networks (KAN).pdf.pdf p1 | Kolmogorov–Arnold Networks (KAN).pdf.pdf p1 |

## 4. Generation latency

| Mode | Retrieve (s) | Generate (s) | Total (s) | Status | Output len | Top1 source |
|------|------------|--------------|-----------|--------|------------|-------------|
| ask | 0.045 | 6.902 | 6.946 | quota_error | 154 | RAG.pdf p13 |
| summarize | 0.06 | 0.075 | 0.135 | quota_error | 95 | Federated Learning AI.pdf p12 |
| mcq | 0.067 | 0.008 | 0.075 | quota_error | 95 | RAG.pdf p6 |

## 5. Demo readiness summary

| Check | Result |
|-------|--------|
| Mode switching (expected) | fast (Streamlit @st.cache_resource; not measured here) |
| Ask ≤ 10s | unknown_due_quota |
| Summary ≤ 15s | unknown_due_quota |
| MCQ ≤ 20s | unknown_due_quota |
| Main bottleneck | gemini_quota |

## 6. Recommendations

Generation latency could not be measured reliably because all Gemini keys were quota-exhausted. Retrieval benchmarks are still valid. Add quota or use --allow-local-fallback only for offline tests (slow).
