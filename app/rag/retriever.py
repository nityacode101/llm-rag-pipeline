"""Semantic retrieval over the FAISS/cosine vector store."""

from __future__ import annotations

import logging

from langchain_core.documents import Document

from app.config import Settings
from app.rag.indexer import VectorStoreManager

logger = logging.getLogger(__name__)


class SemanticRetriever:
    def __init__(self, store_manager: VectorStoreManager, settings: Settings) -> None:
        self.store_manager = store_manager
        self.settings = settings

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
    ) -> list[tuple[Document, float]]:
        if not self.store_manager.is_ready:
            raise RuntimeError(
                "Vector store is not ready. Ingest documents first via /ingest."
            )

        k = top_k or self.settings.top_k
        results = self.store_manager.similarity_search(
            query,
            k=k,
            score_threshold=self.settings.similarity_threshold,
        )

        logger.info(
            "Retrieved %d/%d candidates (threshold=%.2f, backend=%s)",
            len(results),
            k,
            self.settings.similarity_threshold,
            self.store_manager.backend_name,
        )
        return results
