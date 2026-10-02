"""Supabase (Postgres + pgvector) vector store — a drop-in alternative to the
local FAISS index for shared/cloud deployment.

Requires `supabase/schema.sql` to have been applied to the target Supabase
project, and SUPABASE_URL / SUPABASE_KEY set in .env. Select this backend
with VECTOR_STORE_BACKEND=supabase.
"""
from __future__ import annotations

import pandas as pd
from langchain_core.documents import Document

from src.config import (
    CHUNKS_DATA_DIR,
    EMBEDDING_MODEL,
    SUPABASE_KEY,
    SUPABASE_MATCH_FUNCTION,
    SUPABASE_TABLE,
    SUPABASE_URL,
    has_supabase_configured,
)
from src.embeddings.vector_store import _to_native, get_embeddings, load_chunk_dataframe
from src.utils.logger import logger

_METADATA_FIELDS = ["chunk_id", "doc_id", "title", "court", "judge", "decision_year"]


def _get_client():
    if not has_supabase_configured():
        raise RuntimeError(
            "SUPABASE_URL / SUPABASE_KEY not set. Required when VECTOR_STORE_BACKEND=supabase."
        )

    from supabase import create_client

    return create_client(SUPABASE_URL, SUPABASE_KEY)


def _row_to_document(row: dict) -> Document:
    metadata = {field: row[field] for field in _METADATA_FIELDS if row.get(field) is not None}
    metadata["chunk_text"] = row.get("chunk_text", row.get("content", ""))
    return Document(page_content=row.get("content", ""), metadata=metadata)


class SupabaseVectorStore:
    def __init__(
        self,
        table_name: str = SUPABASE_TABLE,
        match_function: str = SUPABASE_MATCH_FUNCTION,
        model_name: str = EMBEDDING_MODEL,
    ):
        self.client = _get_client()
        self.table_name = table_name
        self.match_function = match_function
        self.embeddings = get_embeddings(model_name)

    def build(self, chunk_source=CHUNKS_DATA_DIR, batch_size: int = 200) -> None:
        df = load_chunk_dataframe(chunk_source)
        logger.info(f"Embedding {len(df)} chunks and upserting to Supabase table '{self.table_name}'")

        for start in range(0, len(df), batch_size):
            batch = df.iloc[start : start + batch_size]
            contents = [
                row.get("contextual_text") or row.get("chunk_text", "") for _, row in batch.iterrows()
            ]
            vectors = self.embeddings.embed_documents(contents)

            rows = []
            for (_, row), content, vector in zip(batch.iterrows(), contents, vectors):
                rows.append(
                    {
                        "chunk_id": row["chunk_id"],
                        "doc_id": row["doc_id"],
                        "title": _to_native(row.get("title")) if pd.notna(row.get("title")) else None,
                        "court": _to_native(row.get("court")) if pd.notna(row.get("court")) else None,
                        "judge": _to_native(row.get("judge")) if pd.notna(row.get("judge")) else None,
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
