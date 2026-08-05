"""End-to-end RAG pipeline: ingest → retrieve → grounded Gemini generation."""

from __future__ import annotations

import asyncio
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from langchain_core.documents import Document
from langchain_google_genai import ChatGoogleGenerativeAI

from app.config import Settings, get_settings
from app.models.schemas import (
    IngestResponse,
    QueryResponse,
    SourceDocument,
)
from app.rag.chunking import chunk_documents, load_document, load_documents_from_dir
from app.rag.embeddings import build_embeddings
from app.rag.indexer import VectorStoreManager
from app.rag.retriever import SemanticRetriever

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a helpful assistant that answers questions using ONLY the provided context.
If the context does not contain enough information, say you don't know based on the available documents.
Be concise, accurate, and cite source filenames when relevant.
Do not invent facts outside the context."""


class RAGPipeline:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self._executor = ThreadPoolExecutor(max_workers=8)
        self._lock = asyncio.Lock()

        self.embeddings = build_embeddings(self.settings)
        self.store_manager = VectorStoreManager(self.embeddings, self.settings)
        self.retriever = SemanticRetriever(self.store_manager, self.settings)
        self.llm = ChatGoogleGenerativeAI(
            model=self.settings.gemini_chat_model,
            google_api_key=self.settings.gemini_api_key,
            temperature=0.2,
        )

        loaded = self.store_manager.load()
        if loaded:
            logger.info("Pipeline ready with existing vector store")
        else:
            logger.info("Pipeline ready; awaiting document ingestion")

    async def _run_sync(self, fn, *args, **kwargs):
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            self._executor,
            lambda: fn(*args, **kwargs),
        )

    def _build_context(self, hits: list[tuple[Document, float]]) -> str:
        parts = []
        for i, (doc, score) in enumerate(hits, start=1):
            source = doc.metadata.get("source", "unknown")
            parts.append(
                f"[Source {i}: {source} | relevance={score:.3f}]\n{doc.page_content}"
            )
        return "\n\n---\n\n".join(parts)

    def _generate(self, question: str, context: str) -> str:
        prompt = (
            f"{SYSTEM_PROMPT}\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {question}\n\n"
            f"Answer:"
        )
        response = self.llm.invoke(prompt)
        content = response.content
        if isinstance(content, list):
            return " ".join(str(part) for part in content)
        return str(content)

    async def query(
        self,
        question: str,
        top_k: int | None = None,
        include_sources: bool = True,
    ) -> QueryResponse:
        started = time.perf_counter()
        try:
            hits = await self._run_sync(self.retriever.retrieve, question, top_k)
            if not hits:
                answer = (
                    "I couldn't find relevant information in the indexed documents "
                    "to answer that question."
                )
                sources: list[SourceDocument] = []
            else:
                context = self._build_context(hits)
                answer = await self._run_sync(self._generate, question, context)
                sources = [
                    SourceDocument(
                        content=doc.page_content,
                        metadata=dict(doc.metadata),
                        score=float(score),
                    )
                    for doc, score in hits
                ]

            latency_ms = (time.perf_counter() - started) * 1000
            logger.info(
                "Query completed in %.1fms (retrieved=%d)",
                latency_ms,
                len(hits),
            )
            return QueryResponse(
                answer=answer,
                sources=sources if include_sources else [],
                latency_ms=round(latency_ms, 2),
                retrieved_count=len(hits),
            )
        except Exception:
            logger.exception("Query failed for question=%r", question[:120])
            raise

    async def ingest_paths(self, paths: list[Path], replace: bool = False) -> IngestResponse:
        started = time.perf_counter()
        async with self._lock:
            documents: list[Document] = []
            for path in paths:
                documents.append(await self._run_sync(load_document, path))

            chunks = await self._run_sync(chunk_documents, documents, self.settings)

            if replace:
                await self._run_sync(self.store_manager.clear)
                count = await self._run_sync(self.store_manager.build, chunks)
            elif self.store_manager.is_ready:
                count = await self._run_sync(self.store_manager.add_documents, chunks)
            else:
                count = await self._run_sync(self.store_manager.build, chunks)

            latency_ms = (time.perf_counter() - started) * 1000
            return IngestResponse(
                message="Documents ingested successfully",
                documents_processed=len(documents),
                chunks_indexed=count,
                latency_ms=round(latency_ms, 2),
            )

    async def ingest_directory(
        self,
        directory: Path | None = None,
        replace: bool = True,
    ) -> IngestResponse:
        started = time.perf_counter()
        docs_dir = directory or self.settings.docs_dir

        async with self._lock:
            documents = await self._run_sync(load_documents_from_dir, docs_dir)
            if not documents:
                raise ValueError(f"No supported documents found in {docs_dir}")

            chunks = await self._run_sync(chunk_documents, documents, self.settings)

            if replace or not self.store_manager.is_ready:
                await self._run_sync(self.store_manager.clear)
                count = await self._run_sync(self.store_manager.build, chunks)
            else:
                count = await self._run_sync(self.store_manager.add_documents, chunks)

            latency_ms = (time.perf_counter() - started) * 1000
            logger.info(
                "Ingested %d docs → %d chunks in %.1fms",
                len(documents),
                count,
                latency_ms,
            )
            return IngestResponse(
                message=f"Indexed documents from {docs_dir}",
                documents_processed=len(documents),
                chunks_indexed=count,
                latency_ms=round(latency_ms, 2),
            )

    def health(self) -> dict:
        return {
            "status": "ok",
            "vectorstore_ready": self.store_manager.is_ready,
            "indexed_chunks": self.store_manager.indexed_count(),
        }
