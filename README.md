# Gemini RAG Pipeline

Async retrieval-augmented generation API built with **Python**, **Gemini**, **FAISS**, **LangChain**, and **FastAPI**.

## What this project does

1. Load documents (`.txt`, `.md`, `.pdf`)
2. Split them with optimized overlapping chunking
3. Embed chunks with Gemini (`text-embedding-004`)
4. Index vectors in **FAISS** using **cosine similarity** (L2-normalized + inner product)
5. Retrieve top-k passages for a question
6. Generate a grounded answer with Gemini via FastAPI (async + thread pool for concurrency)

## Project layout

```
gemini-rag-pipeline/
├── app/
│   ├── api/routes.py         # REST endpoints
│   ├── rag/
│   │   ├── chunking.py       # document load + recursive chunking
│   │   ├── embeddings.py     # Gemini (or demo) embeddings
│   │   ├── vector_index.py   # FAISS / NumPy cosine index
│   │   ├── indexer.py        # persist / load / search wrapper
│   │   ├── retriever.py      # top-k semantic retrieval
│   │   └── pipeline.py       # end-to-end RAG orchestration
│   ├── models/schemas.py     # request/response models
│   ├── config.py             # settings from .env
│   └── main.py               # FastAPI app entrypoint
├── data/sample_docs/         # starter corpus
├── scripts/ingest.py         # CLI ingest
├── scripts/smoke_test.py     # offline end-to-end check
├── scripts/load_test.py      # concurrent request test
├── requirements.txt
└── .env.example
```

## Quick start

```powershell
cd C:\Users\user\Downloads\gemini-rag-pipeline
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env
```

Edit `.env` and set `GEMINI_API_KEY` from [Google AI Studio](https://aistudio.google.com/apikey). Keep `DEMO_MODE=false`.

```powershell
python scripts\ingest.py
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

Open [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs).

### Offline smoke test (no API key)

```powershell
python scripts\smoke_test.py
```

## API

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/api/v1/health` | Service + index status |
| `POST` | `/api/v1/query` | Ask a question |
| `POST` | `/api/v1/ingest` | Index `data/sample_docs` |
| `POST` | `/api/v1/ingest/upload` | Upload `.txt` / `.md` / `.pdf` |

### Example query

```powershell
curl -X POST http://127.0.0.1:8000/api/v1/query `
  -H "Content-Type: application/json" `
  -d "{\"question\": \"What is RAG and why does chunk overlap help?\"}"
```

### Concurrent load test

```powershell
python scripts\load_test.py --concurrency 10 --rounds 2
```
