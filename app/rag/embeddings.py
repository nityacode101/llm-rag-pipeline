"""Gemini embedding client used for indexing and query vectors."""

from __future__ import annotations

import logging

from langchain_google_genai import GoogleGenerativeAIEmbeddings

from app.config import Settings

logger = logging.getLogger(__name__)


def build_embeddings(settings: Settings) -> GoogleGenerativeAIEmbeddings:
    if not settings.gemini_api_key or settings.gemini_api_key.startswith("your_"):
        raise ValueError(
            "GEMINI_API_KEY is missing. Copy .env.example to .env and set your key."
        )

    logger.info("Initializing embeddings model: %s", settings.gemini_embedding_model)
    return GoogleGenerativeAIEmbeddings(
        model=settings.gemini_embedding_model,
        google_api_key=settings.gemini_api_key,
    )
