# Retrieval-Augmented Generation (RAG)

Retrieval-Augmented Generation combines information retrieval with large language models.
Instead of relying only on parametric memory, a RAG system retrieves relevant passages
from a document corpus and conditions the LLM on that context.

## Why RAG helps

- Grounds answers in your private documents
- Reduces hallucinations compared to closed-book generation
- Makes updates easy: re-index documents without retraining a model

## Typical pipeline stages

1. Load and clean source documents (PDF, Markdown, text)
2. Split text into overlapping chunks that preserve semantic boundaries
3. Embed each chunk into a dense vector representation
4. Store vectors in a similarity index such as FAISS
5. At query time, embed the question and retrieve top-k nearest chunks
6. Pass retrieved context plus the question to an LLM for grounded generation

## Chunking tips

Optimal chunk size depends on the domain. Overlaps of 10–20% help keep sentences
that would otherwise be split across boundaries. Recursive splitting on paragraphs
and sentences usually beats naive fixed-length windows for top-3 hit rate.
