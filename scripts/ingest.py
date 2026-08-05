"""CLI helper to ingest sample documents into FAISS."""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.config import get_settings
from app.logging_config import setup_logging
from app.rag.pipeline import RAGPipeline


async def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest documents into the RAG index")
    parser.add_argument(
        "--dir",
        type=Path,
        default=None,
        help="Directory of docs to ingest (default: data/sample_docs)",
    )
    parser.add_argument(
        "--append",
        action="store_true",
        help="Append to existing index instead of rebuilding",
    )
    args = parser.parse_args()

    settings = get_settings()
    setup_logging(settings.log_level)

    pipeline = RAGPipeline(settings)
    result = await pipeline.ingest_directory(
        directory=args.dir,
        replace=not args.append,
    )
    print(
        f"{result.message}\n"
        f"  documents: {result.documents_processed}\n"
        f"  chunks:    {result.chunks_indexed}\n"
        f"  latency:   {result.latency_ms:.1f} ms"
    )


if __name__ == "__main__":
    asyncio.run(main())
