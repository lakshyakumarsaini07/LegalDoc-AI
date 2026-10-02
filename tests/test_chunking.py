import pandas as pd

from src.embeddings.chunking import ChunkProcessor, build_contextual_chunk


def test_build_contextual_chunk_includes_metadata():
    row = {"title": "State v. Doe", "court": "High Court", "judge": "Justice X", "decision_year": 2020}
    text = build_contextual_chunk(row, "The defendant was found not liable.")

    assert "State v. Doe" in text
    assert "High Court" in text
    assert "Justice X" in text
    assert "2020" in text
    assert "The defendant was found not liable." in text


def test_chunk_processor_produces_expected_schema(tmp_path):
    pdf_path = tmp_path / "pdf_text.parquet"
    meta_path = tmp_path / "processed.parquet"
    output_dir = tmp_path / "chunks_out"

    pdf_df = pd.DataFrame(
        [{"doc_id": "doc1", "pdf_text": "Sentence one. " * 200}]
    )
    meta_df = pd.DataFrame(
        [
            {
                "doc_id": "doc1",
                "title": "Sample Case",
                "court": "High Court",
                "judge": "Justice X",
                "decision_year": 2021,
            }
        ]
    )
    pdf_df.to_parquet(pdf_path)
    meta_df.to_parquet(meta_path)

    processor = ChunkProcessor(
        pdf_parquet_path=str(pdf_path),
        metadata_parquet_path=str(meta_path),
        output_dir=str(output_dir),
        chunk_size=200,
        chunk_overlap=20,
        batch_size=1000,
    )
    processor.process()

    result = pd.read_parquet(output_dir)

    assert len(result) > 1  # long text should split into multiple chunks
    assert set(["doc_id", "chunk_id", "contextual_text", "chunk_text", "title", "court", "judge"]).issubset(
        result.columns
    )
    assert (result["doc_id"] == "doc1").all()
    assert result["chunk_id"].iloc[0] == "doc1_0"
    assert "Sample Case" in result["contextual_text"].iloc[0]
