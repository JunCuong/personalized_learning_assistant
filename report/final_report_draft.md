# AI-Powered Learning Assistant for Course Materials

## Abstract

This project implements a retrieval-augmented learning assistant for querying, summarizing, and generating multiple-choice questions from course materials. The system processes PDF documents using text extraction and OCR fallback, stores page-level chunks in a FAISS vector index, retrieves relevant context using sentence embeddings, optionally applies lightweight reranking, and uses Gemini 2.5 Flash for grounded generation. The final application provides both a Streamlit interface and a FastAPI backend. Evaluation on a revised 15-question set shows that reranking improves retrieval quality, with Hit@3 increasing from 66.7% to 73.3% and Hit@5 increasing from 80.0% to 86.7%.

## 1. Introduction

Course materials are often distributed across multiple PDFs, slides, and lecture documents. Students may need to search across these files, identify relevant pages, summarize concepts, and create revision questions. Manual search is time-consuming, especially when documents include scanned or slide-based content.

This project develops an AI learning assistant that uses retrieval-augmented generation (RAG) to answer questions using only uploaded course documents. The system is designed to preserve source traceability by showing retrieved sources and page references alongside generated answers.

## 2. Problem Statement

The main problem is to build a practical assistant that can retrieve relevant information from heterogeneous course PDFs and generate useful study outputs. The system must handle both text-based and OCR-needed documents, provide fast local retrieval, and avoid unsupported generation when retrieved context is insufficient.

## 3. Objectives

- Process a small course-material dataset into searchable chunks.
- Support both direct PDF text extraction and OCR fallback.
- Build a FAISS vector store using sentence-transformer embeddings.
- Retrieve relevant chunks quickly for user queries.
- Improve retrieval quality with lightweight reranking.
- Generate grounded answers, summaries, and MCQs using Gemini.
- Provide a usable Streamlit interface and FastAPI backend.
- Evaluate retrieval quality and latency using a revised evaluation set.

## 4. Dataset and Course Materials

The dataset contains 6 PDFs in total, and all 6 were usable after pypdf extraction plus OCR fallback. OCR was required for the Kolmogorov-Arnold Networks (KAN) and MLOps PDFs. OCR uses Poppler for PDF page rendering and Tesseract for text recognition.

After processing, the knowledge base contains 137 chunks from 6 sources in FAISS. The documents cover topics including Retrieval-Augmented Generation, Federated Learning, MLOps, Multimodal Machine Learning, KAN, and course guidance material.

## 5. System Architecture

The architecture follows a standard RAG pipeline:

1. Users interact through Streamlit or FastAPI.
2. PDFs are loaded from uploads or the dataset folder.
3. pypdf extracts text where possible.
4. OCR fallback with Poppler and Tesseract handles low-text pages.
5. Page-level documents are chunked.
6. Chunks are embedded with `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`.
7. FAISS stores and searches vectors.
8. The retriever returns top chunks.
9. Optional lightweight reranking combines FAISS semantic score with keyword overlap.
10. Gemini 2.5 Flash generates answers, summaries, and MCQs from retrieved context.
11. Outputs include source references for transparency.

## 6. Methodology

### PDF Loading

PDFs are loaded from the dataset folder or user uploads. Each document is processed page by page so that retrieved evidence can be traced back to source files and pages.

### OCR Fallback

The system first attempts text extraction with pypdf. If a page has insufficient extracted text, OCR fallback is used. Poppler renders the PDF page into an image, and Tesseract extracts text from that image. This was important for scanned or slide-like PDFs, especially KAN and MLOps materials.

### Chunking

Extracted page text is split into chunks suitable for embedding and retrieval. The final index contains 137 chunks. Chunking balances context size with retrieval precision: smaller chunks improve targeting, while larger chunks preserve enough context for generation.

### Embedding

Chunks are embedded using `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`. This model provides semantic representations that allow retrieval by meaning rather than exact keyword matching.

### FAISS Vector Store

FAISS stores the embedding vectors and performs fast local nearest-neighbor search. The vector store contains 6 document sources.

### Retrieval

For each user query, the retriever embeds the query and searches FAISS for relevant chunks. Average retrieval latency is about 0.04 seconds, making retrieval fast enough for interactive use.

### Lightweight Reranking

The optional reranker combines FAISS semantic similarity with keyword overlap. This is not a learned reranker, but it improves ranking when important query terms are present in the retrieved chunks. The reranking overhead is negligible.

### Gemini Generation

Gemini 2.5 Flash is used for generated answers, summaries, and MCQs. Prompts instruct the model to use only retrieved context and to cite sources where appropriate. If Gemini quota is exhausted, the app rotates configured keys. The app avoids silently switching to slow local fallback unless the user explicitly enables it.

### Streamlit Interface

The Streamlit interface supports uploading PDFs, inspecting the dataset, rebuilding the knowledge base, asking questions, summarizing topics, generating MCQs, viewing retrieved sources, toggling reranking, and explicitly enabling local fallback.

### FastAPI Backend

The FastAPI backend exposes the system functionality programmatically. This makes the project usable beyond the Streamlit demo and supports integration with other clients.

## 7. Implementation

The implementation is organized around modular source files for PDF loading, OCR, chunking, embeddings, FAISS storage, retrieval, LLM access, and generation tasks. The Streamlit app acts as the main interactive interface, while FastAPI provides backend access.

The system uses Gemini key rotation for API resilience. Local fallback uses flan-t5, but it is slow and only enabled explicitly by the user. This avoids unexpected latency and prevents the system from silently producing lower-quality responses when Gemini is unavailable.

Generation features include:

- Question answering with source citation.
- Topic summarization with duplicate cleanup and fallback bullets from retrieved context.
- MCQ generation with structured parsing, validation, and fallback behavior when model output is malformed or incomplete.

## 8. Evaluation

The revised evaluation set contains 15 questions. Retrieval was evaluated using Hit@3, Hit@5, MRR, source-level hit rate, and source+page hit rate.

Baseline retrieval results:

- Hit@3: 66.7%
- Hit@5: 80.0%
- MRR: 0.574

With lightweight reranking:

- Hit@3: 73.3%
- Hit@5: 86.7%
- MRR: about 0.578
- Source Hit@3: 15/15
- Source+Page Hit@3: 11/15

Source-level retrieval is easier than exact page-level retrieval. Exact page matching is stricter because a concept may appear across multiple pages, or the retrieved chunk may come from a nearby page with related content. Therefore, Source+Page Hit@3 should be interpreted as a stricter metric than Source Hit@3.

QA keyword scoring can also be strict because generated answers may use correct paraphrases rather than the same words as the reference answer.

## 9. Results

The system successfully builds a searchable knowledge base from 6 usable PDFs and supports interactive study workflows. Retrieval is fast and local, with average latency around 0.04 seconds. Reranking improves Hit@3 and Hit@5 while adding negligible overhead.

The Streamlit demo supports:

- Uploading PDFs.
- Inspecting dataset quality.
- Building or rebuilding the knowledge base.
- Asking questions.
- Summarizing topics.
- Generating MCQs.
- Viewing retrieved sources.
- Toggling lightweight reranking.
- Enabling optional local fallback.

Generation quality depends on Gemini availability and quota. When Gemini quota is exhausted, the app rotates keys and reports the issue rather than silently switching to local fallback.

## 10. Discussion

The project demonstrates that a lightweight RAG pipeline can support useful study assistance over course materials. The combination of OCR fallback, semantic retrieval, and source display is important because course PDFs may not always contain extractable text.

Reranking provides measurable improvement without adding significant latency. The improvement is modest but useful, especially in a small dataset where keyword overlap can help distinguish similar semantic results.

The strictness of page-level evaluation should be considered when interpreting results. A retrieved source may be correct even if the page differs from the expected page, especially when slide decks repeat headings or distribute a concept across adjacent pages.

## 11. Limitations

- The dataset is small, with only 6 PDFs.
- OCR quality depends on page rendering and scan quality.
- Gemini generation depends on external API availability and quota.
- Local fallback is slow and less capable than Gemini.
- Lightweight reranking is heuristic and not a learned reranker.
- Exact page-level retrieval can be difficult when documents contain repeated headings or related content across pages.
- QA keyword scoring can penalize correct paraphrased answers.
- The system does not yet include user feedback loops for improving retrieval or generation.

## 12. Conclusion

This project delivers a functional AI-powered learning assistant for course materials. It processes PDFs with OCR fallback, builds a FAISS vector store, retrieves relevant chunks quickly, improves ranking with lightweight reranking, and generates study outputs through Gemini. Evaluation shows improved retrieval with reranking, and the Streamlit application provides a practical interface for asking questions, summarizing topics, and generating MCQs.

## 13. References / Tools Used

- Python
- Streamlit
- FastAPI
- pypdf
- Poppler
- Tesseract OCR
- sentence-transformers
- `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`
- FAISS
- Gemini 2.5 Flash
- flan-t5 optional local fallback
- pandas and supporting Python utilities
