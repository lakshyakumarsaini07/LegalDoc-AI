import numpy as np
import pandas as pd

from src.embeddings.vector_store import ChunkedVectorStore


def test_build_documents_embeds_contextual_text_and_keeps_raw_metadata():
    df = pd.DataFrame(
        [
            {
                "chunk_id": "doc1_0",
                "doc_id": "doc1",
                "title": "Sample Case",
                "court": "High Court",
                "decision_year": 2022,
                "judge": "Justice X",
                "contextual_text": "Case Title: Sample Case\nContent: indemnity clause text",
                "chunk_text": "indemnity clause text",
            }
        ]
    )

    documents = ChunkedVectorStore.build_documents(df)

    assert len(documents) == 1
    doc = documents[0]
    assert doc.page_content == "Case Title: Sample Case\nContent: indemnity clause text"
    assert doc.metadata["chunk_text"] == "indemnity clause text"
    assert doc.metadata["doc_id"] == "doc1"
    assert doc.metadata["decision_year"] == 2022


def test_build_documents_falls_back_to_chunk_text_when_no_contextual_text():
    df = pd.DataFrame([{"chunk_id": "doc1_0", "doc_id": "doc1", "chunk_text": "raw text only"}])

    documents = ChunkedVectorStore.build_documents(df)

    assert documents[0].page_content == "raw text only"


def test_build_documents_converts_numpy_scalars_to_native_types():
    # Parquet partition columns commonly come back as numpy/pandas scalar dtypes,
    # which json.dumps (and therefore FastAPI's default encoder) cannot serialize.
    df = pd.DataFrame(
        [{"chunk_id": "doc1_0", "doc_id": "doc1", "chunk_text": "text", "decision_year": np.int64(2022)}]
    )

    documents = ChunkedVectorStore.build_documents(df)

    year = documents[0].metadata["decision_year"]
    assert year == 2022
    assert isinstance(year, int) and not isinstance(year, np.generic)


def test_build_documents_omits_none_metadata_fields():
    df = pd.DataFrame([{"chunk_id": "doc1_0", "doc_id": "doc1", "chunk_text": "text", "court": None}])

    documents = ChunkedVectorStore.build_documents(df)

    assert "court" not in documents[0].metadata
