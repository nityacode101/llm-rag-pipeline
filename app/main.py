"""FastAPI entrypoint for the Gemini RAG Pipeline."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app import __version__
from app.api.routes import init_pipeline, router
from app.config import get_settings
from app.logging_config import setup_logging

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    settings = get_settings()
    setup_logging(settings.log_level)
    logger.info("Starting Gemini RAG Pipeline v%s", __version__)
    try:
        init_pipeline()
        logger.info("RAG pipeline initialized")
    except Exception:
        logger.exception(
            "Pipeline init deferred — set GEMINI_API_KEY and restart, "
            "or first successful request will retry"
        )
    yield
    logger.info("Shutting down Gemini RAG Pipeline")


app = FastAPI(
    title="Gemini RAG Pipeline",
    description=(
        "Async RAG API with chunking, Gemini embeddings, FAISS indexing, "
        "semantic retrieval, and grounded generation."
    ),
    version=__version__,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/api/v1", tags=["rag"])


@app.exception_handler(Exception)
async def unhandled_exception_handler(_request: Request, exc: Exception):
    logger.exception("Unhandled server error: %s", exc)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"},
    )


@app.get("/")
async def root():
    return {
        "name": "Gemini RAG Pipeline",
        "version": __version__,
        "docs": "/docs",
        "health": "/api/v1/health",
    }


def run() -> None:
    import uvicorn

    settings = get_settings()
    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        reload=False,
        log_level=settings.log_level.lower(),
    )


if __name__ == "__main__":
    run()
