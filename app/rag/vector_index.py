"""FAISS vector index with L2-normalized vectors (cosine similarity via inner product).

Falls back to a NumPy cosine index if the FAISS native library cannot load
(e.g. Windows Application Control blocking the DLL).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Protocol

import numpy as np

logger = logging.getLogger(__name__)


def _normalize(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    return vectors / norms


class _IndexBackend(Protocol):
    def add(self, vectors: np.ndarray) -> None: ...

    def search(self, query: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]: ...

    @property
    def ntotal(self) -> int: ...

    def save(self, path: Path) -> None: ...

    @classmethod
    def load(cls, path: Path) -> "_IndexBackend": ...


class FaissIPIndex:
    """FAISS IndexFlatIP over L2-normalized vectors ≈ cosine similarity."""

    def __init__(self, dim: int, index: Any | None = None) -> None:
        import faiss

        self.dim = dim
        self._index = index if index is not None else faiss.IndexFlatIP(dim)

    def add(self, vectors: np.ndarray) -> None:
        vectors = np.asarray(vectors, dtype=np.float32)
        if vectors.ndim != 2:
            raise ValueError("vectors must be 2-D")
        self._index.add(_normalize(vectors))

    def search(self, query: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        query = np.asarray(query, dtype=np.float32)
        if query.ndim == 1:
            query = query.reshape(1, -1)
        scores, indices = self._index.search(_normalize(query), k)
        return scores, indices

    @property
    def ntotal(self) -> int:
        return int(self._index.ntotal)

    def save(self, path: Path) -> None:
        import faiss

        faiss.write_index(self._index, str(path))

    @classmethod
    def load(cls, path: Path) -> "FaissIPIndex":
        import faiss

        index = faiss.read_index(str(path))
        return cls(dim=index.d, index=index)


class NumpyCosineIndex:
    """Pure-NumPy cosine similarity index used when FAISS is unavailable."""

    def __init__(self, dim: int, matrix: np.ndarray | None = None) -> None:
        self.dim = dim
        self._matrix = (
            matrix
            if matrix is not None
            else np.zeros((0, dim), dtype=np.float32)
        )

    def add(self, vectors: np.ndarray) -> None:
        vectors = _normalize(np.asarray(vectors, dtype=np.float32))
        if self._matrix.size == 0:
            self._matrix = vectors
        else:
            self._matrix = np.vstack([self._matrix, vectors])

    def search(self, query: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        if self.ntotal == 0:
            return np.zeros((1, 0), dtype=np.float32), np.full((1, 0), -1)

        query = _normalize(np.asarray(query, dtype=np.float32).reshape(1, -1))
        scores = query @ self._matrix.T
        k = min(k, self.ntotal)
        top_idx = np.argsort(-scores, axis=1)[:, :k]
        top_scores = np.take_along_axis(scores, top_idx, axis=1)
        return top_scores.astype(np.float32), top_idx.astype(np.int64)

    @property
    def ntotal(self) -> int:
        return int(self._matrix.shape[0])

    def save(self, path: Path) -> None:
        np.save(path, self._matrix)

    @classmethod
    def load(cls, path: Path) -> "NumpyCosineIndex":
        matrix = np.load(path, allow_pickle=False)
        return cls(dim=matrix.shape[1], matrix=matrix)


def create_index(dim: int) -> tuple[_IndexBackend, str]:
    try:
        index = FaissIPIndex(dim)
        # Touch the native path early so DLL failures surface here.
        _ = index.ntotal
        logger.info("Using FAISS IndexFlatIP backend (cosine via inner product)")
        return index, "faiss"
    except Exception as exc:
        logger.warning(
            "FAISS unavailable (%s: %s). Falling back to NumPy cosine index.",
            type(exc).__name__,
            exc,
        )
        return NumpyCosineIndex(dim), "numpy"


class DocumentVectorIndex:
    """Embed + store documents and retrieve by cosine similarity."""

    META_FILE = "meta.json"
    DOCS_FILE = "documents.json"
    FAISS_FILE = "index.faiss"
    NUMPY_FILE = "index.npy"

    def __init__(self, embeddings: Any) -> None:
        self.embeddings = embeddings
        self.backend: _IndexBackend | None = None
        self.backend_name: str = "none"
        self.documents: list[dict[str, Any]] = []

    @property
    def is_ready(self) -> bool:
        return self.backend is not None and self.backend.ntotal > 0

    def count(self) -> int:
        return self.backend.ntotal if self.backend else 0

    def build(self, texts: list[str], metadatas: list[dict[str, Any]]) -> int:
        if not texts:
            raise ValueError("No texts to index")
        if len(texts) != len(metadatas):
            raise ValueError("texts and metadatas length mismatch")

        vectors = np.asarray(
            self.embeddings.embed_documents(texts),
            dtype=np.float32,
        )
        self.backend, self.backend_name = create_index(vectors.shape[1])
        self.backend.add(vectors)
        self.documents = [
            {"page_content": text, "metadata": meta}
            for text, meta in zip(texts, metadatas)
        ]
        return self.count()

    def add(self, texts: list[str], metadatas: list[dict[str, Any]]) -> int:
        if not texts:
            return self.count()
        if self.backend is None:
            return self.build(texts, metadatas)

        vectors = np.asarray(
            self.embeddings.embed_documents(texts),
            dtype=np.float32,
        )
        self.backend.add(vectors)
        self.documents.extend(
            {"page_content": text, "metadata": meta}
            for text, meta in zip(texts, metadatas)
        )
        return self.count()

    def similarity_search(
        self,
        query: str,
        k: int = 3,
        score_threshold: float = 0.0,
    ) -> list[tuple[dict[str, Any], float]]:
        if not self.is_ready or self.backend is None:
            raise RuntimeError("Vector index is empty. Ingest documents first.")

        query_vec = np.asarray(
            self.embeddings.embed_query(query),
            dtype=np.float32,
        )
        scores, indices = self.backend.search(query_vec, k)

        hits: list[tuple[dict[str, Any], float]] = []
        for score, idx in zip(scores[0], indices[0]):
            if int(idx) < 0:
                continue
            sim = float(score)
            if sim < score_threshold:
                continue
            hits.append((self.documents[int(idx)], sim))
        return hits

    def save(self, directory: Path) -> None:
        if self.backend is None:
            return
        directory.mkdir(parents=True, exist_ok=True)

        if self.backend_name == "faiss":
            self.backend.save(directory / self.FAISS_FILE)
        else:
            self.backend.save(directory / self.NUMPY_FILE)

        (directory / self.DOCS_FILE).write_text(
            json.dumps(self.documents, ensure_ascii=False),
            encoding="utf-8",
        )

        meta = {
            "backend": self.backend_name,
            "count": self.count(),
            "dim": getattr(self.backend, "dim", None),
        }
        (directory / self.META_FILE).write_text(
            json.dumps(meta, indent=2),
            encoding="utf-8",
        )
        logger.info(
            "Saved %s index (%d vectors) to %s",
            self.backend_name,
            self.count(),
            directory,
        )

    def load(self, directory: Path) -> bool:
        meta_path = directory / self.META_FILE
        docs_path = directory / self.DOCS_FILE
        if not meta_path.exists() or not docs_path.exists():
            return False

        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        backend_name = meta.get("backend", "faiss")

        try:
            loaded = json.loads(docs_path.read_text(encoding="utf-8"))
            if not isinstance(loaded, list):
                raise ValueError("documents.json must contain a list")
            self.documents = loaded

            if backend_name == "faiss" and (directory / self.FAISS_FILE).exists():
                self.backend = FaissIPIndex.load(directory / self.FAISS_FILE)
                self.backend_name = "faiss"
            elif (directory / self.NUMPY_FILE).exists():
                self.backend = NumpyCosineIndex.load(directory / self.NUMPY_FILE)
                self.backend_name = "numpy"
            else:
                logger.warning("Index files missing under %s", directory)
                return False

            logger.info(
                "Loaded %s index with %d vectors",
                self.backend_name,
                self.count(),
            )
            return True
        except Exception:
            logger.exception("Failed to load vector index from %s", directory)
            self.backend = None
            self.documents = []
            return False

    def clear(self) -> None:
        self.backend = None
        self.backend_name = "none"
        self.documents = []
