"""Gemini embedding client (with optional local demo embeddings)."""

from __future__ import annotations

import hashlib
import logging
import re

import numpy as np
from langchain_core.embeddings import Embeddings

from app.config import Settings

logger = logging.getLogger(__name__)


class HashingEmbeddings(Embeddings):
    """Deterministic local embeddings for offline/demo runs (no API key)."""

    def __init__(self, dim: int = 256) -> None:
        self.dim = dim

    def _embed(self, text: str) -> list[float]:
        tokens = re.findall(r"[a-z0-9]+", text.lower())
        vec = np.zeros(self.dim, dtype=np.float32)
        if not tokens:
            vec[0] = 1.0
            return vec.tolist()
        for token in tokens:
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            idx = int.from_bytes(digest[:4], "little") % self.dim
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vec[idx] += sign
        norm = float(np.linalg.norm(vec)) or 1.0
        return (vec / norm).tolist()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)


def build_embeddings(settings: Settings) -> Embeddings:
    if settings.demo_mode:
        logger.warning(
            "DEMO_MODE enabled — using local hashing embeddings "
            "(not Gemini). Set DEMO_MODE=false and GEMINI_API_KEY for production."
        )
        return HashingEmbeddings(dim=256)

    if not settings.gemini_api_key or settings.gemini_api_key.startswith("your_"):
        raise ValueError(
            "GEMINI_API_KEY is missing. Copy .env.example to .env and set your key, "
            "or set DEMO_MODE=true for an offline smoke run."
        )

    from langchain_google_genai import GoogleGenerativeAIEmbeddings

    logger.info("Initializing Gemini embeddings: %s", settings.gemini_embedding_model)
    return GoogleGenerativeAIEmbeddings(
        model=settings.gemini_embedding_model,
        google_api_key=settings.gemini_api_key,
    )
