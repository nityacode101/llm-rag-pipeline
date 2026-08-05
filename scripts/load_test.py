"""Simple concurrent load test against /api/v1/query."""

from __future__ import annotations

import argparse
import asyncio
import statistics
import time

import httpx


QUESTIONS = [
    "What is retrieval-augmented generation?",
    "How does FAISS help with vector search?",
    "Which Gemini embedding model is recommended?",
    "Why use chunk overlap when splitting documents?",
    "How can FastAPI handle concurrent RAG queries?",
]


async def one_query(client: httpx.AsyncClient, question: str) -> float:
    started = time.perf_counter()
    response = await client.post(
        "/api/v1/query",
        json={"question": question, "include_sources": True},
        timeout=120.0,
    )
    response.raise_for_status()
    return (time.perf_counter() - started) * 1000


async def run(base_url: str, concurrency: int, rounds: int) -> None:
    questions = (QUESTIONS * ((concurrency * rounds // len(QUESTIONS)) + 1))[
        : concurrency * rounds
    ]
    latencies: list[float] = []

    async with httpx.AsyncClient(base_url=base_url) as client:
        health = await client.get("/api/v1/health")
        health.raise_for_status()
        print("health:", health.json())

        for r in range(rounds):
            batch = questions[r * concurrency : (r + 1) * concurrency]
            print(f"round {r + 1}/{rounds}: firing {len(batch)} concurrent requests...")
            results = await asyncio.gather(*(one_query(client, q) for q in batch))
            latencies.extend(results)

    print("\nLoad test summary")
    print(f"  requests : {len(latencies)}")
    print(f"  concurrency: {concurrency}")
    print(f"  avg_ms   : {statistics.mean(latencies):.1f}")
    print(f"  p50_ms   : {statistics.median(latencies):.1f}")
    print(f"  max_ms   : {max(latencies):.1f}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--concurrency", type=int, default=10)
    parser.add_argument("--rounds", type=int, default=2)
    args = parser.parse_args()
    asyncio.run(run(args.base_url, args.concurrency, args.rounds))


if __name__ == "__main__":
    main()
