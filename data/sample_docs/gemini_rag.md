# Gemini API for RAG

Google Gemini models can power both the generation and embedding stages of a RAG
pipeline.

## Chat models

Flash-class models such as `gemini-2.0-flash` offer low latency and strong
instruction following, which is ideal for interactive document Q&A.

## Embedding models

`text-embedding-004` produces dense vectors for semantic search. Embed documents
once during ingestion and embed each query at request time.

## Grounded generation prompt pattern

Provide a system instruction that restricts answers to the retrieved context.
Include source filenames and relevance scores so the model can cite evidence.
If context is insufficient, the model should admit uncertainty instead of guessing.

## Operational tips

- Keep temperature low (around 0.1–0.3) for factual Q&A
- Log latency for retrieval and generation separately when optimizing
- Use FastAPI + asyncio so multiple clients can query concurrently
