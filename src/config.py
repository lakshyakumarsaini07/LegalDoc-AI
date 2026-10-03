"""Central configuration for LegalDoc AI.

All paths, model names, and tunables live here so every module reads
from one source instead of hardcoding literals.
"""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# ------------------------------------------------------------------
# Paths
# ------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data"
PDF_DIR = DATA_DIR / "pdfs"
JSON_DIR = DATA_DIR / "json"

PDF_TEXT_DIR = BASE_DIR / "pdf_text_data"
PROCESSED_DATA_DIR = BASE_DIR / "processed_data"
CHUNKS_DATA_DIR = BASE_DIR / "chunks_data"
VECTOR_STORE_DIR = BASE_DIR / "vector_store"
LOG_DIR = BASE_DIR / "logs"

# ------------------------------------------------------------------
# Embedding / retrieval
# ------------------------------------------------------------------
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")  # FAISS backend only (local, 384-dim)

# Supabase backend uses Gemini's hosted embedding API instead of local
# sentence-transformers/torch, so the backend has a small enough dependency
# footprint to run as a Vercel serverless function.
GEMINI_EMBEDDING_MODEL = os.getenv("GEMINI_EMBEDDING_MODEL", "models/gemini-embedding-001")
GEMINI_EMBEDDING_DIM = int(os.getenv("GEMINI_EMBEDDING_DIM", "768"))

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "800"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "150"))

# "faiss" (local file-based index) or "supabase" (pgvector, shared/cloud)
VECTOR_STORE_BACKEND = os.getenv("VECTOR_STORE_BACKEND", "faiss").strip().lower()

# ------------------------------------------------------------------
# Supabase (only used when VECTOR_STORE_BACKEND=supabase)
# ------------------------------------------------------------------
SUPABASE_URL = os.getenv("SUPABASE_URL", "").strip()
SUPABASE_KEY = os.getenv("SUPABASE_KEY", "").strip()  # service role key (server-side ingestion + queries)
SUPABASE_TABLE = os.getenv("SUPABASE_TABLE", "legal_chunks")
SUPABASE_MATCH_FUNCTION = os.getenv("SUPABASE_MATCH_FUNCTION", "match_legal_chunks")


def has_supabase_configured() -> bool:
    return bool(SUPABASE_URL and SUPABASE_KEY)


def vector_store_ready() -> bool:
    """Whether the configured vector store backend is ready to serve retrieval."""
    if VECTOR_STORE_BACKEND == "supabase":
        return has_supabase_configured()
    return VECTOR_STORE_DIR.exists()

RETRIEVER_TOP_K = int(os.getenv("RETRIEVER_TOP_K", "5"))
RETRIEVER_VECTOR_CANDIDATES = int(os.getenv("RETRIEVER_VECTOR_CANDIDATES", "20"))
RETRIEVER_BM25_CANDIDATES = int(os.getenv("RETRIEVER_BM25_CANDIDATES", "20"))

# ------------------------------------------------------------------
# LLM (Gemini)
# ------------------------------------------------------------------
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "").strip()
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-flash-latest")
GEMINI_TEMPERATURE = float(os.getenv("GEMINI_TEMPERATURE", "0.2"))

# ------------------------------------------------------------------
# Evaluation loop
# ------------------------------------------------------------------
EVAL_SCORE_THRESHOLD = float(os.getenv("EVAL_SCORE_THRESHOLD", "4.0"))
EVAL_MAX_ITERATIONS = int(os.getenv("EVAL_MAX_ITERATIONS", "2"))

# ------------------------------------------------------------------
# API
# ------------------------------------------------------------------
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8000"))

# If set, /chat, /summarize, /generate, /refine require header `X-API-Key: <value>`.
# Left unset, the API is open (fine for local/dev use behind your own network boundary).
API_KEY = os.getenv("API_KEY", "").strip()

# Comma-separated list of allowed CORS origins, or "*" for any.
ALLOWED_ORIGINS = [o.strip() for o in os.getenv("ALLOWED_ORIGINS", "*").split(",") if o.strip()]


def has_llm_configured() -> bool:
    return bool(GOOGLE_API_KEY)
