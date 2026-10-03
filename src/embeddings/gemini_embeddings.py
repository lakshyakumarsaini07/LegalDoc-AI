"""Gemini-hosted embeddings for the Supabase backend.

Deliberately does not import anything from vector_store.py (FAISS,
langchain_huggingface) — those pull in sentence-transformers/torch, which
are far too large for a Vercel serverless function. This module's only
heavy-ish dependency is langchain-google-genai, imported lazily.
"""
from __future__ import annotations

from src.config import GEMINI_EMBEDDING_DIM, GEMINI_EMBEDDING_MODEL


def get_gemini_embeddings(model_name: str = GEMINI_EMBEDDING_MODEL):
    from langchain_google_genai import GoogleGenerativeAIEmbeddings

    return GoogleGenerativeAIEmbeddings(model=model_name, output_dimensionality=GEMINI_EMBEDDING_DIM)


def to_native(value):
    """Coerce numpy/pandas scalars to plain Python types so metadata stays JSON-serializable."""
    if hasattr(value, "item") and type(value).__module__ == "numpy":
        return value.item()
    return value
