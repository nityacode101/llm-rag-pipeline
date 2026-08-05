"""End-to-end offline smoke test (DEMO_MODE, no Gemini key required)."""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Force demo mode before settings are loaded.
os.environ["DEMO_MODE"] = "true"

from app.config import get_settings
from app.logging_config import setup_logging
from app.rag.pipeline import RAGPipeline


async def main() -> None:
    get_settings.cache_clear()
    settings = get_settings()
    setup_logging(settings.log_level)

    # Isolate smoke index from any previous Gemini-backed store.
    settings.vectorstore_dir = settings.data_dir / "vectorstore_demo"
    settings.vectorstore_dir.mkdir(parents=True, exist_ok=True)

    pipeline = RAGPipeline(settings)
    ingest = await pipeline.ingest_directory(replace=True)
    print("INGEST:", ingest.model_dump())

    health = pipeline.health()
    print("HEALTH:", health)

    result = await pipeline.query("What is retrieval-augmented generation?")
    print("QUERY answer:", result.answer[:300].replace("\n", " "))
    print("QUERY latency_ms:", result.latency_ms)
    print("QUERY retrieved_count:", result.retrieved_count)
    print("QUERY top_score:", result.sources[0].score if result.sources else None)

    if ingest.chunks_indexed < 1:
        raise SystemExit("FAIL: no chunks indexed")
    if result.retrieved_count < 1:
        raise SystemExit("FAIL: retrieval returned nothing")
    if health.get("backend") not in {"faiss", "numpy"}:
        raise SystemExit(f"FAIL: unexpected backend {health.get('backend')}")

    print("SMOKE TEST PASSED")


if __name__ == "__main__":
    asyncio.run(main())
