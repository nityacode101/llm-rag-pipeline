"""Optimized document chunking for higher retrieval hit rates."""

from __future__ import annotations

import logging
from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import Settings

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".txt", ".md", ".pdf"}


def build_splitter(settings: Settings) -> RecursiveCharacterTextSplitter:
    """Recursive splitter preserves semantic boundaries better than fixed-size cuts."""
    return RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
        is_separator_regex=False,
    )


def load_text_file(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def load_pdf_file(path: Path) -> str:
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    pages = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        if text.strip():
            pages.append(f"[Page {i + 1}]\n{text}")
    return "\n\n".join(pages)


def load_document(path: Path) -> Document:
    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported file type: {suffix}")

    if suffix == ".pdf":
        content = load_pdf_file(path)
    else:
        content = load_text_file(path)

    if not content.strip():
        raise ValueError(f"Empty document: {path.name}")

    return Document(
        page_content=content,
        metadata={"source": path.name, "path": str(path), "filetype": suffix},
    )


def load_documents_from_dir(directory: Path) -> list[Document]:
    docs: list[Document] = []
    if not directory.exists():
        logger.warning("Documents directory does not exist: %s", directory)
        return docs

    for path in sorted(directory.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue
        try:
            docs.append(load_document(path))
            logger.info("Loaded document: %s", path.name)
        except Exception:
            logger.exception("Failed to load %s", path)

    return docs


def chunk_documents(
    documents: list[Document],
    settings: Settings,
) -> list[Document]:
    if not documents:
        return []

    splitter = build_splitter(settings)
    chunks = splitter.split_documents(documents)

    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = i
        chunk.metadata["chunk_size"] = len(chunk.page_content)

    logger.info(
        "Chunked %d documents into %d chunks (size=%d, overlap=%d)",
        len(documents),
        len(chunks),
        settings.chunk_size,
        settings.chunk_overlap,
    )
    return chunks
