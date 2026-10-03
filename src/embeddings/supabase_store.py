"""Supabase (Postgres + pgvector) vector store — a drop-in alternative to the
local FAISS index for shared/cloud deployment (and the only backend light
enough to run as a Vercel serverless function).

Requires `supabase/schema.sql` to have been applied to the target Supabase
project, and SUPABASE_URL / SUPABASE_KEY set in .env. Select this backend
with VECTOR_STORE_BACKEND=supabase.

Deliberately avoids importing vector_store.py (FAISS/pandas/pyarrow) at
module level — pandas is only needed for `build()` (local ingestion, never
run on Vercel), so that import is deferred into the method that needs it.
"""
from __future__ import annotations

import re
import time

from langchain_core.documents import Document

from src.config import (
    CHUNKS_DATA_DIR,
    SUPABASE_KEY,
    SUPABASE_MATCH_FUNCTION,
    SUPABASE_TABLE,
    SUPABASE_URL,
    has_supabase_configured,
)
from src.embeddings.gemini_embeddings import get_gemini_embeddings, to_native
from src.utils.logger import logger

_METADATA_FIELDS = ["chunk_id", "doc_id", "title", "court", "judge", "decision_year"]


def _get_client():
    if not has_supabase_configured():
        raise RuntimeError(
            "SUPABASE_URL / SUPABASE_KEY not set. Required when VECTOR_STORE_BACKEND=supabase."
        )

    from supabase import create_client

    return create_client(SUPABASE_URL, SUPABASE_KEY)


def _retry_delay_seconds(exc: Exception, default: float = 65.0) -> float:
    """Parse Gemini's suggested backoff ("Please retry in 27.1s") out of the error,
    falling back to a conservative default if the message format ever changes."""
    match = re.search(r"retry in ([\d.]+)s", str(exc))
    return float(match.group(1)) + 3 if match else default


def _row_to_document(row: dict) -> Document:
    metadata = {field: row[field] for field in _METADATA_FIELDS if row.get(field) is not None}
    metadata["chunk_text"] = row.get("chunk_text", row.get("content", ""))
    return Document(page_content=row.get("content", ""), metadata=metadata)


class SupabaseVectorStore:
    def __init__(
        self,
        table_name: str = SUPABASE_TABLE,
        match_function: str = SUPABASE_MATCH_FUNCTION,
    ):
        self.client = _get_client()
        self.table_name = table_name
        self.match_function = match_function
        self.embeddings = get_gemini_embeddings()

    def _embed_with_retry(self, texts: list[str], max_retries: int = 8) -> list[list[float]]:
        """Gemini's embed_content has no true batch endpoint — langchain issues one
        request per text, so a free-tier quota (100 req/min) can be hit mid-batch.
        React to the 429 by sleeping for the API's own suggested delay and retrying,
        rather than guessing a fixed pace up front."""
        for attempt in range(max_retries):
            try:
                return self.embeddings.embed_documents(texts)
            except Exception as exc:  # noqa: BLE001 - provider-specific rate-limit error
                if attempt == max_retries - 1:
                    raise
                delay = _retry_delay_seconds(exc)
                logger.warning(
                    f"Embedding request rate-limited, waiting {delay:.0f}s "
                    f"(attempt {attempt + 1}/{max_retries})"
                )
                time.sleep(delay)

    def _existing_chunk_ids(self, page_size: int = 1000) -> set[str]:
        ids: set[str] = set()
        start = 0

        while True:
            response = (
                self.client.table(self.table_name)
                .select("chunk_id")
                .range(start, start + page_size - 1)
                .execute()
            )
            rows = response.data
            if not rows:
                break

            ids.update(row["chunk_id"] for row in rows)
            if len(rows) < page_size:
                break
            start += page_size

        return ids

    def build(
        self,
        chunk_source=CHUNKS_DATA_DIR,
        batch_size: int = 50,
        pause_seconds: float = 40.0,
        force: bool = False,
    ) -> None:
        import pandas as pd

        from src.embeddings.vector_store import load_chunk_dataframe

        df = load_chunk_dataframe(chunk_source)

        if not force:
            existing_ids = self._existing_chunk_ids()
            if existing_ids:
                before = len(df)
                df = df[~df["chunk_id"].isin(existing_ids)]
                logger.info(
                    f"Found {len(existing_ids)} chunks already in Supabase — "
                    f"skipping {before - len(df)}, embedding the remaining {len(df)}."
                )

        if df.empty:
            logger.info("Nothing to embed — Supabase table already up to date.")
            return

        logger.info(f"Embedding {len(df)} chunks and upserting to Supabase table '{self.table_name}'")

        for start in range(0, len(df), batch_size):
            batch = df.iloc[start : start + batch_size]
            contents = [
                row.get("contextual_text") or row.get("chunk_text", "") for _, row in batch.iterrows()
            ]
            vectors = self._embed_with_retry(contents)

            rows = []
            for (_, row), content, vector in zip(batch.iterrows(), contents, vectors):
                rows.append(
                    {
                        "chunk_id": row["chunk_id"],
                        "doc_id": row["doc_id"],
                        "title": to_native(row.get("title")) if pd.notna(row.get("title")) else None,
                        "court": to_native(row.get("court")) if pd.notna(row.get("court")) else None,
                        "judge": to_native(row.get("judge")) if pd.notna(row.get("judge")) else None,
                        "decision_year": (
                            int(row["decision_year"]) if pd.notna(row.get("decision_year")) else None
                        ),
                        "content": content,
                        "chunk_text": row.get("chunk_text", ""),
                        "embedding": vector,
                    }
                )

            self.client.table(self.table_name).upsert(rows, on_conflict="chunk_id").execute()
            logger.info(f"Upserted {start + len(rows)}/{len(df)} chunks")

            is_last_batch = start + batch_size >= len(df)
            if not is_last_batch:
                # Proactive pacing: Gemini's free-tier quota is 100 embed_content
                # requests/minute, and each text in a batch is its own request (no
                # true batch endpoint). Staying well under that avoids ever
                # triggering the SDK's own internal retry-with-backoff on 429,
                # which has been observed to stack into multi-minute stalls.
                time.sleep(pause_seconds)

        logger.info("Supabase vector store build complete.")

    def similarity_search(self, query: str, k: int = 5) -> list[Document]:
        query_embedding = self.embeddings.embed_query(query)
        response = self.client.rpc(
            self.match_function,
            {"query_embedding": query_embedding, "match_count": k},
        ).execute()
        return [_row_to_document(row) for row in response.data]

    def get_all_documents(self, page_size: int = 1000) -> list[Document]:
        """Full corpus, for building the in-process BM25 index (Supabase has no
        built-in BM25; Postgres full-text search would need a parallel setup)."""
        docs: list[Document] = []
        start = 0

        while True:
            response = (
                self.client.table(self.table_name)
                .select("chunk_id, doc_id, title, court, judge, decision_year, content, chunk_text")
                .range(start, start + page_size - 1)
                .execute()
            )
            rows = response.data
            if not rows:
                break

            docs.extend(_row_to_document(row) for row in rows)
            if len(rows) < page_size:
                break
            start += page_size

        return docs
