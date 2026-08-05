# Gemini RAG Pipeline

Async retrieval-augmented generation API built with **Python**, **Gemini**, **FAISS**, **LangChain**, and **FastAPI**.

## Features

- Optimized recursive chunking with overlap for stronger top-k retrieval
- Gemini embeddings (`text-embedding-004`) + FAISS cosine/relevance search
- Grounded answers via Gemini chat models
- Async FastAPI endpoints that handle concurrent queries via a thread pool
- Structured logging and error handling throughout

## Project layout

```
gemini-rag-pipeline/
├── app/
│   ├── api/routes.py      # /health /query /ingest endpoints
│   ├── rag/               # chunking, embeddings, FAISS, retrieval, pipeline
│   ├── models/schemas.py
│   ├── config.py
│   └── main.py
├── data/sample_docs/      # starter corpus
├── scripts/ingest.py      # CLI ingest
├── scripts/load_test.py   # concurrent request smoke test
├── requirements.txt
└── .env.example
```

## Quick start

### 1. Create a virtual environment

```powershell
cd C:\Users\user\Downloads\gemini-rag-pipeline
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 2. Configure your API key

```powershell
copy .env.example .env
```

Edit `.env` and set `GEMINI_API_KEY` from [Google AI Studio](https://aistudio.google.com/apikey).

### 3. Ingest sample documents

```powershell
python scripts/ingest.py
```

### 4. Run the API

```powershell
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

Open interactive docs at [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs).

## API

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/api/v1/health` | Service + index status |
| `POST` | `/api/v1/query` | Ask a question against the index |
| `POST` | `/api/v1/ingest` | Rebuild/index `data/sample_docs` |
| `POST` | `/api/v1/ingest/upload` | Upload `.txt` / `.md` / `.pdf` files |

### Example query

```powershell
curl -X POST http://127.0.0.1:8000/api/v1/query `
  -H "Content-Type: application/json" `
  -d "{\"question\": \"What is RAG and why does chunk overlap help?\"}"
```

### Concurrent load test

With the server running:

```powershell
python scripts/load_test.py --concurrency 10 --rounds 2
```

## Resume-aligned highlights

- End-to-end RAG: chunking → embeddings → FAISS → semantic retrieval → Gemini generation
- Tunable chunk size/overlap and relevance-score filtering for better top-3 hit rate
- Async FastAPI handlers for 10+ concurrent requests under load testing
