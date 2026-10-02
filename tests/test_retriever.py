from langchain_core.documents import Document

from src.rag.retriever import reciprocal_rank_fusion


def _doc(chunk_id: str) -> Document:
    return Document(page_content=f"content for {chunk_id}", metadata={"chunk_id": chunk_id})


def test_rrf_prefers_docs_ranked_highly_in_multiple_lists():
    vector_hits = [_doc("a"), _doc("b"), _doc("c")]
    bm25_hits = [_doc("b"), _doc("a"), _doc("d")]

    fused = reciprocal_rank_fusion([vector_hits, bm25_hits], k=4)
    fused_ids = [doc.metadata["chunk_id"] for doc in fused]

    # "a" and "b" appear near the top of both lists, so they should be fused to the top.
    assert set(fused_ids[:2]) == {"a", "b"}
    assert "c" in fused_ids
    assert "d" in fused_ids


def test_rrf_deduplicates_by_chunk_id():
    same_doc_a = _doc("a")
    same_doc_b = _doc("a")

    fused = reciprocal_rank_fusion([[same_doc_a], [same_doc_b]], k=5)

    assert len(fused) == 1


def test_rrf_respects_k_limit():
    docs = [_doc(str(i)) for i in range(10)]
    fused = reciprocal_rank_fusion([docs], k=3)
    assert len(fused) == 3
