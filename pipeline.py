"""End-to-end ingestion pipeline: PDFs + JSON metadata -> chunks -> FAISS vector store.

PDF text extraction and metadata processing are idempotent (skip doc_ids
already present in their output datasets) unless --force is passed.
Chunking and the vector store are always rebuilt from the latest
extracted/processed data, since that step is cheap relative to PDF
extraction and embedding needs to reflect the full corpus.
"""
import argparse
import shutil

from process import MetadataProcessor
from src.config import (
    CHUNKS_DATA_DIR,
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    EMBEDDING_MODEL,
    JSON_DIR,
    PDF_DIR,
    PDF_TEXT_DIR,
    PROCESSED_DATA_DIR,
    VECTOR_STORE_BACKEND,
    VECTOR_STORE_DIR,
)
from src.embeddings.chunking import ChunkProcessor
from src.preprocessing.pdf_parser import PDFParser
from src.utils.logger import logger


def run_full_pipeline(force: bool = False) -> None:
    logger.info("Step 1/4: Extract PDF text")
    pdf_parser = PDFParser(pdf_dir=str(PDF_DIR), output_dir=str(PDF_TEXT_DIR), batch_size=50)
    pdf_parser.run(force=force)

    logger.info("Step 2/4: Process JSON metadata")
    metadata_processor = MetadataProcessor(
        json_dir=str(JSON_DIR),
        pdf_dir=str(PDF_DIR),
        batch_size=2000,
        output_dir=str(PROCESSED_DATA_DIR),
    )
    metadata_processor.process(force=force)

    logger.info("Step 3/4: Chunk text for embeddings (full rebuild)")
    if CHUNKS_DATA_DIR.exists():
        shutil.rmtree(CHUNKS_DATA_DIR)
    chunk_processor = ChunkProcessor(
        pdf_parquet_path=str(PDF_TEXT_DIR),
        metadata_parquet_path=str(PROCESSED_DATA_DIR),
        output_dir=str(CHUNKS_DATA_DIR),
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        batch_size=1000,
    )
    chunk_processor.process()

    logger.info(f"Step 4/4: Build vector store (backend={VECTOR_STORE_BACKEND})")
    if VECTOR_STORE_BACKEND == "supabase":
        from src.embeddings.supabase_store import SupabaseVectorStore

        SupabaseVectorStore().build(chunk_source=CHUNKS_DATA_DIR)
    else:
        from src.embeddings.vector_store import ChunkedVectorStore

        ChunkedVectorStore(
            chunk_source=CHUNKS_DATA_DIR,
            vector_store_dir=VECTOR_STORE_DIR,
            model_name=EMBEDDING_MODEL,
        ).build(save=True)

    logger.info("Pipeline complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the LegalDoc AI ingestion pipeline.")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Reprocess all PDFs/metadata even if already present in the output datasets.",
    )
    args = parser.parse_args()
    run_full_pipeline(force=args.force)
