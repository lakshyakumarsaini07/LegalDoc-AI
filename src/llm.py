"""Thin interface around the Gemini chat model.

Every pipeline (RAG, summarization, generation, evaluation) calls
``get_llm()`` instead of instantiating a chat model directly. When no
``GOOGLE_API_KEY`` is configured this returns ``None`` and callers fall
back to a degraded (retrieval-only / template-only) mode instead of
crashing, so the app is usable before a key is provisioned.
"""
from __future__ import annotations

from functools import lru_cache

from src.config import GEMINI_MODEL, GEMINI_TEMPERATURE, has_llm_configured
from src.utils.logger import logger


@lru_cache(maxsize=1)
def get_llm(temperature: float | None = None):
    if not has_llm_configured():
        logger.warning("GOOGLE_API_KEY not set — LLM features are disabled (fallback mode).")
        return None

    from langchain_google_genai import ChatGoogleGenerativeAI

    return ChatGoogleGenerativeAI(
        model=GEMINI_MODEL,
        temperature=GEMINI_TEMPERATURE if temperature is None else temperature,
    )


def invoke_text(prompt: str, *, temperature: float | None = None) -> str | None:
    """Invoke the LLM with a plain-text prompt, returning the response text or None."""
    llm = get_llm(temperature=temperature)
    if llm is None:
        return None

    try:
        response = llm.invoke(prompt)
        # `.text` (not `.content`) normalizes both plain-string responses and the
        # list-of-content-block format newer Gemini models return.
        return response.text
    except Exception as exc:  # noqa: BLE001 - surface any provider error to caller as None
        logger.error(f"LLM invocation failed: {exc}")
        return None
