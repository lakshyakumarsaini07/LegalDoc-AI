"""Hybrid retriever: FAISS vector search + BM25 keyword search, combined
via Reciprocal Rank Fusion (RRF).

Pure vector similarity can drown out exact legal terms of art
("indemnity", "force majeure", specific case/CNR numbers). BM25 catches
those; RRF blends the two rankings without needing to normalize scores
from two different scales.
"""
from __future__ import annotations

from pathlib import Path

from langchain_core.documents import Document
from rank_bm25 import BM25Okapi

from src.config import (
    RETRIEVER_BM25_CANDIDATES,
    RETRIEVER_TOP_K,
    RETRIEVER_VECTOR_CANDIDATES,
    VECTOR_STORE_BACKEND,
    VECTOR_STORE_DIR,
)
from src.utils.logger import logger


def _tokenize(text: str) -> list[str]:
    return text.lower().split()


def _doc_key(doc: Document) -> str:
    return str(doc.metadata.get("chunk_id") or doc.page_content)


def reciprocal_rank_fusion(
    ranked_lists: list[list[Document]],
    k: int = RETRIEVER_TOP_K,
    rrf_k: int = 60,
) -> list[Document]:
    """Merge multiple ranked document lists into one, keyed by rrf_k / (rank + rrf_k)."""
    scores: dict[str, float] = {}
    doc_by_key: dict[str, Document] = {}

    for ranked in ranked_lists:
        for rank, doc in enumerate(ranked):
            key = _doc_key(doc)
            doc_by_key[key] = doc
            scores[key] = scores.get(key, 0.0) + 1.0 / (rrf_k + rank + 1)

    ordered_keys = sorted(scores, key=lambda key: scores[key], reverse=True)
    return [doc_by_key[key] for key in ordered_keys[:k]]


def _corpus_documents(vector_store) -> list[Document]:
    """Enumerate every document in the store, for building the BM25 index.

    Works for both backends: FAISS exposes an in-memory docstore; the
    Supabase store paginates the table directly.
    """
    if hasattr(vector_store, "get_all_documents"):
        return vector_store.get_all_documents()
    return list(vector_store.docstore._dict.values())


class HybridRetriever:
    """Combines vector search (FAISS or Supabase/pgvector) with a BM25 keyword
    index built over the same corpus."""

    def __init__(
        self,
        vector_store,
        k: int = RETRIEVER_TOP_K,
        vector_candidates: int = RETRIEVER_VECTOR_CANDIDATES,
        bm25_candidates: int = RETRIEVER_BM25_CANDIDATES,
    ):
        self.vector_store = vector_store
        self.k = k
        self.vector_candidates = vector_candidates
        self.bm25_candidates = bm25_candidates
        self._build_bm25()

    def _build_bm25(self) -> None:
        self.docs: list[Document] = _corpus_documents(self.vector_store)
        tokenized_corpus = [_tokenize(doc.page_content) for doc in self.docs]
        self.bm25 = BM25Okapi(tokenized_corpus)
        logger.info(f"Built BM25 index over {len(self.docs)} chunks")

    def retrieve(self, query: str, k: int | None = None) -> list[Document]:
        k = k or self.k

        vector_hits = self.vector_store.similarity_search(query, k=self.vector_candidates)

        bm25_scores = self.bm25.get_scores(_tokenize(query))
        top_bm25_idx = bm25_scores.argsort()[::-1][: self.bm25_candidates]
        bm25_hits = [self.docs[i] for i in top_bm25_idx if bm25_scores[i] > 0]

        return reciprocal_rank_fusion([vector_hits, bm25_hits], k=k)

    # LangChain-style alias
    def invoke(self, query: str) -> list[Document]:
        return self.retrieve(query)


def load_retriever(
    vector_store_dir: Path | str = VECTOR_STORE_DIR,
    k: int = RETRIEVER_TOP_K,
    backend: str = VECTOR_STORE_BACKEND,
) -> HybridRetriever:
    if backend == "supabase":
        from src.embeddings.supabase_store import SupabaseVectorStore

        vector_store = SupabaseVectorStore()
    else:
        from src.embeddings.vector_store import load_store

        vector_store = load_store(vector_store_dir)

    return HybridRetriever(vector_store, k=k)


def main() -> None:
    retriever = load_retriever()
    query = "Tell me about the case on ARVIND SINGH SANGWAN. What was the final decision?"
    results = retriever.retrieve(query)

    for i, doc in enumerate(results, start=1):
        print(f"\n--- Result {i} ({doc.metadata.get('doc_id')}) ---")
        print(doc.metadata)
        print(doc.metadata.get("chunk_text", doc.page_content)[:300])


if __name__ == "__main__":
    main()
