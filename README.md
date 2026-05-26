# Personalized Learning Assistant (Track 3)

A RAG-based study assistant that lets students upload course PDFs, ask questions, get grounded answers with source references, generate summaries, and create multiple-choice quizzes.

## Features

- Upload PDFs and build a FAISS knowledge base
- Ask natural-language questions (answers grounded in retrieved chunks)
- Source references: PDF filename + page number
- Summarize topics from your documents
- Generate MCQs with answer keys and explanations
- Retrieval + keyword QA evaluation script
- Streamlit UI + FastAPI backend

## Folder structure

```
Final_project/
??? app.py                 # Streamlit frontend
??? api.py                 # FastAPI backend
??? inspect_dataset.py     # PDF inspection (Step 1)
??? run_evaluation.py      # Evaluation runner
??? requirements.txt
??? README.md
??? .env.example
??? dataset/               # Your PDF files
??? data/
?   ??? processed/         # documents.json, chunks.json
?   ??? evaluation/        # eval_questions.csv, results
??? vector_store/faiss_index/
??? notebooks/final_notebook.ipynb
??? src/                   # RAG pipeline modules
??? report/
??? slides/
```

## Dataset inspection (Step 1)

Six PDFs were found in `dataset/`. Initial inspection:

| Status | Count | Notes |
|--------|-------|-------|
| OK | 4 | Text extracts successfully |
| WARNING | 2 | Likely scanned/image PDFs (0 chars): `Kolmogorov-Arnold Networks (KAN).pdf.pdf`, `Scalable_MLOps_Architecture.pptx.pdf` |

Run inspection anytime:

```powershell
cd D:\Cuong\01_Programming\1_1_Learning\VNUK\4_2\advanced_data_challenge\Final_project
python inspect_dataset.py
```

Scanned PDFs are skipped for chunking (empty pages). For OCR support, you would need an extra dependency (not included by default).

## Setup (Windows PowerShell)

```powershell
cd D:\Cuong\01_Programming\1_1_Learning\VNUK\4_2\advanced_data_challenge\Final_project
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
```

### Gemini API key (recommended)

Use **GEMINI_API_KEY only** in `.env` (see `.env.example`). Leave `GOOGLE_API_KEY` commented unless you have a second real key.

Get a key from [Google AI Studio](https://aistudio.google.com/apikey):

```powershell
$env:GEMINI_API_KEY="your_gemini_api_key_here"
```

Or in `.env`:

```
GEMINI_API_KEY=your_gemini_api_key_here
# GOOGLE_API_KEY=
```

If no key is set, the app falls back to local `google/flan-t5-base` (slower, shorter outputs).

### OCR setup (scanned PDFs)

OCR runs **only** when pypdf extracts fewer than 30 characters on a page.

| Requirement | Details |
|-------------|---------|
| Tesseract | `C:\Program Files\Tesseract-OCR\tesseract.exe` (override via `TESSERACT_CMD`) |
| Python packages | `pytesseract`, `pdf2image`, `Pillow` |
| Poppler | Required by `pdf2image` on Windows |

**Install Poppler (required for OCR on scanned slides):**

1. Download [poppler-windows releases](https://github.com/oschwartz10612/poppler-windows/releases)
2. Extract and add the `Library\bin` folder to PATH, **or** set in `.env`:

```
POPPLER_PATH=C:\path\to\poppler\Library\bin
```

If Poppler is missing, OCR pages stay empty and diagnostics report `ocr_dependencies.ready = false`.

Text-based PDFs still use **pypdf only** (no unnecessary OCR).

### Default models

| Component | Default |
|-----------|---------|
| Embeddings | `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` |
| Generation (API) | `gemini-2.0-flash` (override via `GEMINI_MODEL_NAME`) |
| Generation (local) | `google/flan-t5-base` |
| Chunk size / overlap | 1000 / 200 characters |
| Retrieval top-k | 5 |

## Run Streamlit

```powershell
.\venv\Scripts\activate
streamlit run app.py
```

1. Upload PDFs (optional) � saved to `dataset/`
2. Click **Build / Rebuild Knowledge Base**
3. Choose mode: Ask / Summarize / MCQ

## Run FastAPI

```powershell
.\venv\Scripts\activate
uvicorn api:app --reload
```

Endpoints:

- `GET /` � health check
- `POST /build-index` � full ingestion pipeline
- `POST /ask` � `{"question": "...", "top_k": 5}`
- `POST /summarize` � `{"topic": "...", "top_k": 8}`
- `POST /generate-mcq` � `{"topic": "...", "num_questions": 5, "top_k": 10}`

PDF upload via API is not implemented; use Streamlit or copy files into `dataset/`.

## Build index from CLI

```powershell
python -c "from src.pipeline import build_knowledge_base; print(build_knowledge_base())"
```

## Evaluation

Edit `data/evaluation/eval_questions.csv` with real questions and expected source/page/keywords (use `|` for multiple keywords).

```powershell
python run_evaluation.py --eval-file data/evaluation/eval_questions.csv --mode retrieval
python run_evaluation.py --eval-file data/evaluation/eval_questions_proposed.csv --mode retrieval
python build_proposed_eval.py
```

Outputs (per eval file stem):

- `data/evaluation/{stem}_results.csv` � retrieval metrics per question
- `data/evaluation/evaluation_qa_results.csv` � keyword QA scores
- `data/evaluation/evaluation_summary.json` � Hit@3, Hit@5, MRR, mean keyword score

**Note:** Template eval rows are placeholders � verify `expected_page` against your built index for accurate metrics.

## Notebook

```powershell
jupyter notebook notebooks/final_notebook.ipynb
```

## Diagnostics and Review

Run all automated inspections and export reports to `data/diagnostics/`:

```powershell
cd D:\Cuong\01_Programming\1_1_Learning\VNUK\4_2\advanced_data_challenge\Final_project
.\venv\Scripts\activate
python run_diagnostics.py
```

Individual scripts:

```powershell
python inspect_dataset.py
python inspect_ocr.py
python inspect_chunks.py
python inspect_retrieval.py
python inspect_evaluation.py
python inspect_evaluation.py --eval-file data/evaluation/eval_questions_proposed.csv
python inspect_llm.py
```

Optional filters:

```powershell
python inspect_ocr.py --full
python inspect_ocr.py --source "Kolmogorov-Arnold Networks (KAN).pdf.pdf" --full
python inspect_chunks.py --source "RAG.pdf" --limit 20
python inspect_retrieval.py --query "What is RAG?" --top-k 5
```

Outputs include CSV/JSON reports (dataset, chunks, retrieval, evaluation, LLM) and `diagnostics_run_summary.json`. See `data/diagnostics/manual_test_checklist.md` for manual Streamlit/FastAPI tests.

## Known limitations

- Character-based chunking (not semantic)
- OCR requires Poppler + Tesseract on Windows; without Poppler, scanned PDFs stay empty
- Local flan-t5 fallback has limited context length and quality vs Gemini
- MCQ JSON parsing may fail on small local models; raw text is shown instead
- Evaluation keyword matching is simple substring match, not semantic similarity
- First embedding model download requires internet

## Engineering defaults (minor)

- FAISS `IndexFlatIP` with L2-normalized embeddings (cosine similarity)
- Sources in answers come only from retrieved chunks, not from model invention
- UTF-8 for all JSON/CSV I/O
