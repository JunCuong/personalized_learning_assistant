# Manual Test Checklist

Use this checklist after `python run_diagnostics.py` passes automated checks.

---

## Streamlit (`streamlit run app.py`)

| Step | Action | Expected result |
|------|--------|-----------------|
| 1 | Run `streamlit run app.py` from project root | App opens in browser without import errors |
| 2 | Confirm dataset path shown at top | Path points to `Final_project/dataset` |
| 3 | Upload a small PDF (optional) | File saved message; file appears in `dataset/` |
| 4 | Click **Build / Rebuild Knowledge Base** | Success message; JSON summary shows chunk_count > 0 |
| 5 | Mode: **Ask Question** — e.g. "What is RAG?" | Answer text appears; not empty error |
| 6 | Check **Sources** table | Rows show `source`, `page`, `rank`, `score`; at least one row |
| 7 | Mode: **Generate MCQ** — topic "RAG", 2 questions | MCQs or raw text shown; sources table populated |
| 8 | Sidebar: **Inspect dataset PDFs** | Table with OK/WARNING/ERROR per PDF |

---

## FastAPI (`uvicorn api:app --reload`)

| Step | Action | Expected result |
|------|--------|-----------------|
| 1 | Run `uvicorn api:app --reload` | Server starts on `http://127.0.0.1:8000` |
| 2 | Open `http://127.0.0.1:8000/docs` | Swagger UI loads |
| 3 | **GET /** | `{"status":"ok","index_built":true,...}` |
| 4 | **POST /build-index** (if index missing) | JSON with `status: ok`, `index_summary` |
| 5 | **POST /ask** — `{"question":"What is federated learning?","top_k":5}` | `answer` string + `sources` list |
| 6 | **POST /generate-mcq** — `{"topic":"RAG","num_questions":2,"top_k":8}` | `mcqs` or `raw_text` + `sources` |

---

## Notes

- If Gemini quota is exceeded (429), answers may use local `flan-t5` (shorter/slower).
- Scanned PDFs (0 extracted text) will not appear in retrieval results.
- Re-run `python run_diagnostics.py` after adding PDFs or rebuilding the index.
