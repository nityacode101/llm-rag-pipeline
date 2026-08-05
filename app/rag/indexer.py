"""FAISS vector store: build, persist, and load indexes."""

from __future__ import annotations

import logging
import shutil
from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from app.config import Settings

logger = logging.getLogger(__name__)

INDEX_NAME = "index"


class VectorStoreManager:
    def __init__(self, embeddings: Embeddings, settings: Settings) -> None:
        self.embeddings = embeddings
        self.settings = settings
        self.store: FAISS | None = None

    @property
    def index_path(self) -> Path:
        return self.settings.vectorstore_dir

    @property
    def is_ready(self) -> bool:
        return self.store is not None

    def indexed_count(self) -> int:
        if self.store is None:
            return 0
        return len(self.store.index_to_docstore_id)

    def build(self, chunks: list[Document]) -> int:
        if not chunks:
            raise ValueError("No chunks to index")

        logger.info("Building FAISS index from %d chunks...", len(chunks))
        self.store = FAISS.from_documents(chunks, self.embeddings)
        self.persist()
        count = self.indexed_count()
        logger.info("FAISS index ready with %d vectors", count)
        return count

    def add_documents(self, chunks: list[Document]) -> int:
        if not chunks:
            return 0
        if self.store is None:
            return self.build(chunks)

        logger.info("Adding %d chunks to existing FAISS index...", len(chunks))
        self.store.add_documents(chunks)
        self.persist()
        return self.indexed_count()

    def persist(self) -> None:
        if self.store is None:
            return
        self.index_path.mkdir(parents=True, exist_ok=True)
        self.store.save_local(str(self.index_path), index_name=INDEX_NAME)
        logger.info("Persisted FAISS index to %s", self.index_path)

    def load(self) -> bool:
        faiss_file = self.index_path / f"{INDEX_NAME}.faiss"
        if not faiss_file.exists():
            logger.warning("No FAISS index found at %s", self.index_path)
            return False

        try:
            self.store = FAISS.load_local(
                str(self.index_path),
                self.embeddings,
                index_name=INDEX_NAME,
                allow_dangerous_deserialization=True,
            )
            logger.info(
                "Loaded FAISS index (%d vectors) from %s",
                self.indexed_count(),
                self.index_path,
            )
            return True
        except Exception:
            logger.exception("Failed to load FAISS index")
            self.store = None
            return False

    def clear(self) -> None:
        self.store = None
        if self.index_path.exists():
            shutil.rmtree(self.index_path)
            self.index_path.mkdir(parents=True, exist_ok=True)
        logger.info("Cleared FAISS index")
