"""Vercel serverless entrypoint. Exposes the same FastAPI app as local/Docker
use (src/api/main.py); Vercel's Python runtime detects the `app` ASGI
callable in this file automatically.

Requires VECTOR_STORE_BACKEND=supabase (set in Vercel project env vars) —
the FAISS backend needs sentence-transformers/torch, which are far too large
for a serverless function (see api/requirements.txt, which deliberately
excludes them).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.api.main import app  # noqa: E402

__all__ = ["app"]
