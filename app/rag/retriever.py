"""Semantic retrieval over FAISS using cosine similarity scores."""

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
        if not self.store_manager.is_ready or self.store_manager.store is None:
            raise RuntimeError(
                "Vector store is not ready. Ingest documents first via /ingest."
            )

        k = top_k or self.settings.top_k
        # FAISS L2 / IP distance → LangChain returns similarity score when available.
        # score_threshold filters weak matches to reduce hallucinations.
        results = self.store_manager.store.similarity_search_with_relevance_scores(
            query,
            k=k,
            score_threshold=self.settings.similarity_threshold,
        )

        logger.info(
            "Retrieved %d/%d candidates for query (threshold=%.2f)",
            len(results),
            k,
            self.settings.similarity_threshold,
        )
        return results
