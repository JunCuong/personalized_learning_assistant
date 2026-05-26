# 5-Minute Demo Script

## 0:00-0:30 - Dashboard

Open the Streamlit dashboard. Briefly explain that this is an AI-powered learning assistant for course PDFs. Point out the main workflow: build or inspect the knowledge base, ask questions, summarize topics, and generate MCQs.

## 0:30-1:00 - Dataset and OCR

Explain that the dataset contains 6 PDFs, all usable after pypdf extraction plus OCR fallback. Mention that OCR uses Poppler and Tesseract, and OCR was needed for the KAN and MLOps PDFs.

Show the dataset inspection feature if available. Emphasize that OCR helps handle scanned or slide-based documents.

## 1:00-1:40 - MLOps Question

Go to the Ask tab and enter:

```text
What is MLOps?
```

Show the generated answer. Open retrieved sources and point out the source from:

```text
Scalable_MLOps_Architecture.pptx.pdf
```

Explain that the system retrieves context first and then asks Gemini to answer using the retrieved context.

## 1:40-2:20 - KAN Question

Ask:

```text
What is Kolmogorov-Arnold Network?
```

Show that the source comes from the KAN PDF. Explain that this document required OCR, demonstrating why the OCR fallback was important.

## 2:20-2:50 - RAG Question

Ask:

```text
What is retrieval augmented generation?
```

Show the answer and source references. Briefly explain that retrieved sources make the generated answer more transparent.

## 2:50-3:30 - Summary

Go to the Summary tab and enter:

```text
Summarize retrieval augmented generation.
```

Show the 3-5 bullet summary. Mention that the summary is cleaned to remove duplicated lines, intro text, and incomplete bullets.

## 3:30-4:15 - MCQ Generation

Go to the MCQ tab. Enter:

```text
RAG components and benefits
```

Set the number of questions to 3 and generate MCQs. Show the structured cards. Explain that the system validates structured MCQs and reports the exact X/Y count if any question cannot be structured.

## 4:15-5:00 - Evaluation and Reranking

Explain the retrieval evaluation:

- Revised evaluation set: 15 questions.
- Baseline Hit@3: 66.7%.
- Baseline Hit@5: 80.0%.
- Baseline MRR: 0.574.
- Reranking Hit@3: 73.3%.
- Reranking Hit@5: 86.7%.
- Reranking MRR: about 0.578.
- Source Hit@3 with reranking: 15/15.
- Source+Page Hit@3 with reranking: 11/15.

Toggle lightweight reranking and explain that it combines FAISS semantic score with keyword overlap. Mention that source-level retrieval is easier than exact page-level retrieval, so Source+Page Hit@3 is the stricter metric.
