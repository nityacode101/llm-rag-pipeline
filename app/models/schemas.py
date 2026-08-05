from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=4000)
    top_k: int | None = Field(default=None, ge=1, le=20)
    include_sources: bool = True


class SourceDocument(BaseModel):
    content: str
    metadata: dict
    score: float


class QueryResponse(BaseModel):
    answer: str
    sources: list[SourceDocument] = []
    latency_ms: float
    retrieved_count: int


class IngestResponse(BaseModel):
    message: str
    documents_processed: int
    chunks_indexed: int
    latency_ms: float


class HealthResponse(BaseModel):
    status: str
    vectorstore_ready: bool
    indexed_chunks: int
