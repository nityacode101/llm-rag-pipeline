"""Vector store manager: build / persist / load the FAISS cosine index."""

from __future__ import annotations

import logging
import shutil
from pathlib import Path

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from app.config import Settings
from app.rag.vector_index import DocumentVectorIndex

logger = logging.getLogger(__name__)


class VectorStoreManager:
    def __init__(self, embeddings: Embeddings, settings: Settings) -> None:
        self.embeddings = embeddings
        self.settings = settings
        self.index = DocumentVectorIndex(embeddings)

    @property
    def index_path(self) -> Path:
        return self.settings.vectorstore_dir

    @property
    def is_ready(self) -> bool:
        return self.index.is_ready

    @property
    def backend_name(self) -> str:
        return self.index.backend_name

    def indexed_count(self) -> int:
        return self.index.count()

    def build(self, chunks: list[Document]) -> int:
        if not chunks:
            raise ValueError("No chunks to index")

        texts = [c.page_content for c in chunks]
        metadatas = [dict(c.metadata) for c in chunks]
        logger.info("Building vector index from %d chunks...", len(chunks))
        count = self.index.build(texts, metadatas)
        self.persist()
        logger.info(
            "Index ready (%s backend, %d vectors)",
            self.backend_name,
            count,
        )
        return count

    def add_documents(self, chunks: list[Document]) -> int:
        if not chunks:
            return self.indexed_count()
        if not self.is_ready:
            return self.build(chunks)

        texts = [c.page_content for c in chunks]
        metadatas = [dict(c.metadata) for c in chunks]
        logger.info("Adding %d chunks to existing index...", len(chunks))
        count = self.index.add(texts, metadatas)
        self.persist()
        return count

    def persist(self) -> None:
        self.index.save(self.index_path)

    def load(self) -> bool:
        ok = self.index.load(self.index_path)
        if not ok:
            logger.warning("No vector index found at %s", self.index_path)
        return ok

    def clear(self) -> None:
        self.index.clear()
        if self.index_path.exists():
            shutil.rmtree(self.index_path)
            self.index_path.mkdir(parents=True, exist_ok=True)
        logger.info("Cleared vector index")

    def similarity_search(
        self,
        query: str,
        k: int,
        score_threshold: float,
    ) -> list[tuple[Document, float]]:
        hits = self.index.similarity_search(
            query,
            k=k,
            score_threshold=score_threshold,
        )
        results: list[tuple[Document, float]] = []
        for item, score in hits:
            results.append(
                (
                    Document(
                        page_content=item["page_content"],
                        metadata=item.get("metadata", {}),
                    ),
                    score,
                )
            )
        return results
