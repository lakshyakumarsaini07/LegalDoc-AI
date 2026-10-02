"""FastAPI wrapper exposing the RAG, summarization, and generation
pipelines as a REST API.

Run with: uvicorn src.api.main:app --host 0.0.0.0 --port 8000
"""
from __future__ import annotations

import time
from contextlib import asynccontextmanager
from functools import lru_cache

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from src.config import ALLOWED_ORIGINS, API_KEY, has_llm_configured, vector_store_ready
from src.generation.doc_generator import DocumentGenerator
from src.summarization.summarize import SummarizationService
from src.utils.logger import logger


@lru_cache(maxsize=1)
def get_query_pipeline():
    from src.rag.query_pipeline import RAGQueryPipeline

    return RAGQueryPipeline()


@asynccontextmanager
async def lifespan(app: FastAPI):
    if vector_store_ready():
        try:
            get_query_pipeline()
            logger.info("Vector store loaded at startup.")
        except Exception as exc:  # noqa: BLE001
            logger.error(f"Failed to load vector store at startup: {exc}")
    else:
        logger.warning("Vector store not ready — /chat will fail until pipeline.py is run.")
    yield


app = FastAPI(title="LegalDoc AI API", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled error on {request.method} {request.url.path}: {exc}")
    return JSONResponse(status_code=500, content={"detail": "Internal server error."})


@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    duration_ms = (time.perf_counter() - start) * 1000
    logger.info(f"{request.method} {request.url.path} -> {response.status_code} ({duration_ms:.0f}ms)")
    return response


def require_api_key(x_api_key: str | None = Header(default=None)) -> None:
    if API_KEY and x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Missing or invalid X-API-Key header.")


# ------------------------------------------------------------------
# Request / response schemas
# ------------------------------------------------------------------
class ChatRequest(BaseModel):
    query: str = Field(..., min_length=1)
    top_k: int | None = Field(default=None, ge=1, le=20)
    enable_eval: bool = False


class SourceItem(BaseModel):
    index: int
    doc_id: str | None = None
    chunk_id: str | None = None
    title: str | None = None
    court: str | None = None
    judge: str | None = None
    decision_year: int | None = None
    excerpt: str


class EvalScore(BaseModel):
    accuracy: float
    completeness: float
    relevance: float
    average: float
    feedback: str


class ChatResponse(BaseModel):
    answer: str
    sources: list[SourceItem]
    used_gemini: bool
    eval: EvalScore | None = None


class SummarizeRequest(BaseModel):
    text: str = Field(..., min_length=1)


class SummarizeResponse(BaseModel):
    summary: str
    used_gemini: bool
    chunks: int


class GenerateRequest(BaseModel):
    document_type: str = Field(..., min_length=1)
    requirements: str = Field(..., min_length=1)


class DocumentResponse(BaseModel):
    document: str
    used_gemini: bool


class RefineRequest(BaseModel):
    document: str = Field(..., min_length=1)
    instruction: str = Field(..., min_length=1)


class HealthResponse(BaseModel):
    status: str
    llm_configured: bool
    vector_store_ready: bool


class StatsResponse(BaseModel):
    indexed_chunks: int
    indexed_documents: int


# ------------------------------------------------------------------
# Routes
# ------------------------------------------------------------------
@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(
        status="ok",
        llm_configured=has_llm_configured(),
        vector_store_ready=vector_store_ready(),
    )


@app.get("/stats", response_model=StatsResponse, dependencies=[Depends(require_api_key)])
def stats():
    if not vector_store_ready():
        raise HTTPException(status_code=503, detail="Vector store not built. Run pipeline.py first.")

    pipeline = get_query_pipeline()
    docs = pipeline.retriever.docs
    unique_doc_ids = {d.metadata.get("doc_id") for d in docs if d.metadata.get("doc_id")}
    return StatsResponse(indexed_chunks=len(docs), indexed_documents=len(unique_doc_ids))


@app.post("/chat", response_model=ChatResponse, dependencies=[Depends(require_api_key)])
def chat(request: ChatRequest):
    if not vector_store_ready():
        raise HTTPException(status_code=503, detail="Vector store not built. Run pipeline.py first.")

    pipeline = get_query_pipeline()
    return pipeline.answer(request.query, k=request.top_k, enable_eval=request.enable_eval)


@app.post("/summarize", response_model=SummarizeResponse, dependencies=[Depends(require_api_key)])
def summarize(request: SummarizeRequest):
    return SummarizationService().summarize(request.text)


@app.post("/generate", response_model=DocumentResponse, dependencies=[Depends(require_api_key)])
def generate(request: GenerateRequest):
    return DocumentGenerator().generate(request.document_type, request.requirements)


@app.post("/refine", response_model=DocumentResponse, dependencies=[Depends(require_api_key)])
def refine(request: RefineRequest):
    return DocumentGenerator().refine(request.document, request.instruction)
