"""FastAPI routes for the RAG pipeline."""

from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile

from app.config import Settings, get_settings
from app.models.schemas import HealthResponse, IngestResponse, QueryRequest, QueryResponse
from app.rag.pipeline import RAGPipeline

logger = logging.getLogger(__name__)

router = APIRouter()

_pipeline: RAGPipeline | None = None


def get_pipeline() -> RAGPipeline:
    global _pipeline
    if _pipeline is None:
        try:
            _pipeline = RAGPipeline()
        except Exception as exc:
            logger.exception("Failed to initialize RAG pipeline")
            raise HTTPException(status_code=503, detail=str(exc)) from exc
    return _pipeline


def init_pipeline() -> RAGPipeline:
    global _pipeline
    _pipeline = RAGPipeline()
    return _pipeline


@router.get("/health", response_model=HealthResponse)
async def health(pipeline: Annotated[RAGPipeline, Depends(get_pipeline)]) -> HealthResponse:
    info = pipeline.health()
    return HealthResponse(**info)


@router.post("/query", response_model=QueryResponse)
async def query(
    body: QueryRequest,
    pipeline: Annotated[RAGPipeline, Depends(get_pipeline)],
) -> QueryResponse:
    try:
        return await pipeline.query(
            question=body.question,
            top_k=body.top_k,
            include_sources=body.include_sources,
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Unhandled query error")
        raise HTTPException(status_code=500, detail=f"Query failed: {exc}") from exc


@router.post("/ingest", response_model=IngestResponse)
async def ingest_sample_docs(
    pipeline: Annotated[RAGPipeline, Depends(get_pipeline)],
    replace: bool = Query(default=True, description="Rebuild index from scratch"),
) -> IngestResponse:
    try:
        return await pipeline.ingest_directory(replace=replace)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Ingest failed")
        raise HTTPException(status_code=500, detail=f"Ingest failed: {exc}") from exc


@router.post("/ingest/upload", response_model=IngestResponse)
async def ingest_upload(
    pipeline: Annotated[RAGPipeline, Depends(get_pipeline)],
    settings: Annotated[Settings, Depends(get_settings)],
    files: list[UploadFile] = File(...),
    replace: bool = Query(default=False),
) -> IngestResponse:
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded")

    saved: list[Path] = []
    try:
        for upload in files:
            if not upload.filename:
                continue
            dest = settings.uploads_dir / Path(upload.filename).name
            with dest.open("wb") as out:
                shutil.copyfileobj(upload.file, out)
            saved.append(dest)
            logger.info("Saved upload: %s", dest.name)

        if not saved:
            raise HTTPException(status_code=400, detail="No valid files uploaded")

        return await pipeline.ingest_paths(saved, replace=replace)
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Upload ingest failed")
        raise HTTPException(status_code=500, detail=f"Upload ingest failed: {exc}") from exc
