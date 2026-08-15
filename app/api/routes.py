"""FastAPI routes for the RAG pipeline."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile

from app.api.auth import require_app_token
from app.config import Settings, get_settings
from app.models.schemas import HealthResponse, IngestResponse, QueryRequest, QueryResponse
from app.rag.chunking import SUPPORTED_EXTENSIONS
from app.rag.pipeline import RAGPipeline

logger = logging.getLogger(__name__)

router = APIRouter()

_pipeline: RAGPipeline | None = None
MAX_UPLOAD_FILES = 5


def get_pipeline() -> RAGPipeline:
    global _pipeline
    if _pipeline is None:
        try:
            _pipeline = RAGPipeline()
        except Exception:
            logger.exception("Failed to initialize RAG pipeline")
            raise HTTPException(
                status_code=503,
                detail="Service is not configured. Set GEMINI_API_KEY or DEMO_MODE=true.",
            ) from None
    return _pipeline


def init_pipeline() -> RAGPipeline:
    global _pipeline
    _pipeline = RAGPipeline()
    return _pipeline


@router.get("/health", response_model=HealthResponse)
async def health(pipeline: Annotated[RAGPipeline, Depends(get_pipeline)]) -> HealthResponse:
    info = pipeline.health()
    return HealthResponse(**info)


@router.post("/query", response_model=QueryResponse, dependencies=[Depends(require_app_token)])
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
    except Exception:
        logger.exception("Unhandled query error")
        raise HTTPException(status_code=500, detail="Query failed") from None


@router.post("/ingest", response_model=IngestResponse, dependencies=[Depends(require_app_token)])
async def ingest_sample_docs(
    pipeline: Annotated[RAGPipeline, Depends(get_pipeline)],
    replace: bool = Query(default=True, description="Rebuild index from scratch"),
) -> IngestResponse:
    try:
        return await pipeline.ingest_directory(replace=replace)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception:
        logger.exception("Ingest failed")
        raise HTTPException(status_code=500, detail="Ingest failed") from None


@router.post(
    "/ingest/upload",
    response_model=IngestResponse,
    dependencies=[Depends(require_app_token)],
)
async def ingest_upload(
    pipeline: Annotated[RAGPipeline, Depends(get_pipeline)],
    settings: Annotated[Settings, Depends(get_settings)],
    files: list[UploadFile] = File(...),
    replace: bool = Query(default=False),
) -> IngestResponse:
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded")
    if len(files) > MAX_UPLOAD_FILES:
        raise HTTPException(
            status_code=400,
            detail=f"Too many files. Maximum is {MAX_UPLOAD_FILES}.",
        )

    saved: list[Path] = []
    try:
        for upload in files:
            if not upload.filename:
                continue
            name = Path(upload.filename).name
            suffix = Path(name).suffix.lower()
            if suffix not in SUPPORTED_EXTENSIONS:
                raise HTTPException(
                    status_code=400,
                    detail=f"Unsupported file type: {suffix}. Use .txt, .md, or .pdf.",
                )
            dest = settings.uploads_dir / name
            written = 0
            too_large = False
            with dest.open("wb") as out:
                while True:
                    chunk = await upload.read(1024 * 1024)
                    if not chunk:
                        break
                    written += len(chunk)
                    if written > settings.max_upload_bytes:
                        too_large = True
                        break
                    out.write(chunk)
            if too_large:
                dest.unlink(missing_ok=True)
                raise HTTPException(status_code=413, detail="File is too large.")
            saved.append(dest)
            logger.info("Saved upload: %s", dest.name)

        if not saved:
            raise HTTPException(status_code=400, detail="No valid files uploaded")

        return await pipeline.ingest_paths(saved, replace=replace)
    except HTTPException:
        raise
    except Exception:
        logger.exception("Upload ingest failed")
        raise HTTPException(status_code=500, detail="Upload ingest failed") from None
