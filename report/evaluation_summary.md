# Evaluation Summary

## Dataset Summary

| Item | Value |
|---|---:|
| Total PDFs | 6 |
| Usable PDFs after pypdf + OCR | 6/6 |
| OCR tools | Poppler + Tesseract |
| PDFs requiring OCR | KAN and MLOps PDFs |
| Chunks indexed | 137 |
| Sources in FAISS | 6 |
| Embedding model | sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 |
| Vector store | FAISS |
| LLM | Gemini 2.5 Flash |

## Retrieval Metrics

| Metric | Baseline | With Lightweight Reranking |
|---|---:|---:|
| Hit@3 | 66.7% | 73.3% |
| Hit@5 | 80.0% | 86.7% |
| MRR | 0.574 | about 0.578 |
| Source Hit@3 | Not primary reported metric | 15/15 |
| Source+Page Hit@3 | Not primary reported metric | 11/15 |

Source+Page Hit@3 is stricter than Source Hit@3 because the same concept may appear on multiple pages or in adjacent slide sections.

## Latency Benchmark

| Component | Observation |
|---|---|
| Retrieval | Fast and local |
| Average retrieval latency | about 0.04s |
| Reranking overhead | negligible |
| Generation | Depends on Gemini API latency and quota |
| Local fallback | Optional, explicit, and slow |

## Qualitative Demo Results

| Feature | Expected Demo Behavior |
|---|---|
| Question answering | Produces grounded answer and source references |
| KAN question | Retrieves KAN PDF source |
| MLOps question | Retrieves Scalable_MLOps_Architecture.pptx.pdf source |
| RAG summary | Produces 3-5 useful bullets without repeated intro text |
| MCQ generation | Produces structured MCQ cards when generation succeeds, with clear X/Y count otherwise |
| Reranking toggle | Can improve ordering for source/page retrieval |

## Limitations

- Dataset size is small.
- OCR quality depends on document scan quality.
- Gemini quota or high-demand errors can affect generation.
- Local fallback is intentionally explicit because it is slower and less capable.
- Exact page matching is strict and may understate useful retrieval when related content appears nearby.
- QA keyword scoring may penalize correct paraphrases.
