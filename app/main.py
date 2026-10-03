"""Vercel serverless entrypoint, at a location its FastAPI zero-config
detection actually recognizes (project root, or inside src/ or app/ —
NOT api/, which was silently not picked up). Exposes the same FastAPI app as
local/Docker use (src/api/main.py).

Requires VECTOR_STORE_BACKEND=supabase (set in Vercel project env vars) —
the FAISS backend needs sentence-transformers/torch, which are far too large
for a serverless function (see requirements.txt at the project root, which
deliberately excludes them).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.api.main import app  # noqa: E402

__all__ = ["app"]
