"""Build and load the FAISS vector store from chunked parquet data.

Embeds the contextual chunk text (case title/court/judge/year prefix +
content) since that context measurably improves retrieval for short or
ambiguous chunks, while the raw chunk text is kept in metadata for
citations/display.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
import numpy as np
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

from src.config import CHUNKS_DATA_DIR, EMBEDDING_MODEL, VECTOR_STORE_DIR
from src.utils.logger import logger

load_dotenv()


def _to_native(value):
    """Coerce numpy/pandas scalars to plain Python types so metadata stays JSON-serializable."""
    if isinstance(value, np.generic):
        return value.item()
    return value


def load_chunk_dataframe(chunk_source: Path | str) -> pd.DataFrame:
    source = Path(chunk_source)
    if not source.exists():
        raise FileNotFoundError(f"Chunk source not found: {source}")

    return pd.read_parquet(source)


def get_embeddings(model_name: str = EMBEDDING_MODEL) -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(model_name=model_name)


def load_store(
    vector_store_dir: Path | str = VECTOR_STORE_DIR,
    model_name: str = EMBEDDING_MODEL,
) -> FAISS:
    """Load a previously built FAISS store from disk."""
    vector_store_dir = Path(vector_store_dir)
    if not vector_store_dir.exists():
        raise FileNotFoundError(
            f"No vector store found at {vector_store_dir}. Run the ingestion "
            "pipeline (pipeline.py) first."
        )

    embeddings = get_embeddings(model_name)
    return FAISS.load_local(
        str(vector_store_dir),
        embeddings,
        allow_dangerous_deserialization=True,
    )


class ChunkedVectorStore:
    """Build a FAISS vector store from chunked parquet data."""

    def __init__(
        self,
        chunk_source: Path | str = CHUNKS_DATA_DIR,
        vector_store_dir: Path | str = VECTOR_STORE_DIR,
        model_name: str = EMBEDDING_MODEL,
    ):
        self.chunk_source = Path(chunk_source)
        self.vector_store_dir = Path(vector_store_dir)
        self.model_name = model_name
        self.embeddings = get_embeddings(model_name)

    def load_chunks(self) -> pd.DataFrame:
        return load_chunk_dataframe(self.chunk_source)

    @staticmethod
    def build_documents(df: pd.DataFrame) -> list[Document]:
        documents: list[Document] = []

        for _, row in df.iterrows():
            metadata = {
                "chunk_id": row.get("chunk_id"),
                "doc_id": row.get("doc_id"),
                "title": row.get("title"),
                "court": row.get("court"),
                "decision_year": row.get("decision_year"),
                "judge": row.get("judge"),
                "chunk_text": row.get("chunk_text", ""),
            }

            documents.append(
                Document(
                    page_content=row.get("contextual_text") or row.get("chunk_text", ""),
                    metadata={k: _to_native(v) for k, v in metadata.items() if pd.notna(v)},
                )
            )

        return documents

    def build(self, save: bool = True) -> FAISS:
        df = self.load_chunks()
        logger.info(f"Building vector store from {len(df)} chunks using {self.model_name}")
        documents = self.build_documents(df)
        vector_store = FAISS.from_documents(documents, embedding=self.embeddings)

        if save:
            self.save(vector_store)

        return vector_store

    def save(self, vector_store: FAISS) -> None:
        self.vector_store_dir.mkdir(parents=True, exist_ok=True)
        vector_store.save_local(str(self.vector_store_dir))
        logger.info(f"Saved FAISS vector store to {self.vector_store_dir}")


def main() -> None:
    builder = ChunkedVectorStore()
    builder.build(save=True)
    print(f"Saved FAISS vector store to {builder.vector_store_dir}")


if __name__ == "__main__":
    main()
