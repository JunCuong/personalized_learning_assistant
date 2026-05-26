# Performance benchmark

| Metric | Cold | Warm |
|--------|------|------|
| Embedder load | 9.13s | 0s (cached) |
| FAISS load | 0.03s | 0s (cached) |
| LLM init | 9.29s | 0s (cached) |
| Avg Ask Question | 4.74s | 7.85s |
| Summary | 8.55s | 14.81s |
| MCQ (3) | 12.82s | 7.23s |

**Allow local fallback:** False
**Gemini quota exhausted (any run):** True
**Local fallback used:** False

**Last warm ask — retrieve:** 0.09s | **generate:** 7.44s | **total:** 7.54s

**10s warm Ask target:** MET

**Bottleneck:** gemini_generate (LLM API latency dominates retrieve+format)

**Recommendation:** Warm Ask meets 10s demo target; keep Streamlit caches warm.
