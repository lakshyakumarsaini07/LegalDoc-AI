"""PDF Parser for extracting text and storing in Parquet.

- Reads PDFs
- Extracts text
- Stores in a partitioned parquet dataset
- Idempotent: skips doc_ids already present in the output dataset
"""

import logging
from pathlib import Path
from typing import Dict, List, Set

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from pypdf import PdfReader
from tqdm import tqdm

from src.utils.logger import logger

# pypdf logs a warning per malformed cross-reference entry, which floods
# stdout for scraped/imperfectly-generated legal PDFs without indicating a
# real failure (extraction still succeeds).
logging.getLogger("pypdf").setLevel(logging.ERROR)


class PDFParser:

    def __init__(
        self,
        pdf_dir: str,
        output_dir: str = "pdf_text_data",
        batch_size: int = 100,
    ):
        self.pdf_dir = Path(pdf_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.batch_size = batch_size
        self.buffer: List[Dict] = []

        self.processed = 0
        self.failed = 0
        self.skipped = 0

    # ----------------------------
    def get_pdf_files(self):
        return sorted(self.pdf_dir.glob("**/*.pdf"))

    # ----------------------------
    def get_existing_doc_ids(self) -> Set[str]:
        """Doc ids already present in the output dataset, for idempotent reruns."""
        try:
            existing = pd.read_parquet(self.output_dir, columns=["doc_id"])
            return set(existing["doc_id"].unique())
        except Exception:
            return set()

    # ----------------------------
    def extract_text(self, pdf_path: Path):
        try:
            reader = PdfReader(str(pdf_path))
            text = []
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text.append(page_text)

            return "\n".join(text), len(reader.pages), "success"

        except Exception as e:
            return "", 0, f"error: {str(e)}"

    # ----------------------------
    def process_pdf(self, pdf_path: Path) -> Dict:

        text, num_pages, status = self.extract_text(pdf_path)

        return {
            "doc_id": pdf_path.stem,  # IMPORTANT: matches metadata
            "pdf_path": str(pdf_path),
            "pdf_text": text,
            "num_pages": num_pages,
            "extraction_status": status,
        }

    # ----------------------------
    def add_record(self, record: Dict):
        self.buffer.append(record)

        if len(self.buffer) >= self.batch_size:
            self.write_batch()

    # ----------------------------
    def write_batch(self):

        if not self.buffer:
            return

        df = pd.DataFrame(self.buffer)

        table = pa.Table.from_pandas(df, preserve_index=False)

        pq.write_to_dataset(
            table,
            root_path=str(self.output_dir),
        )

        self.processed += len(df)
        self.buffer = []

    # ----------------------------
    def run(self, force: bool = False):

        pdf_files = list(self.get_pdf_files())
        existing_ids = set() if force else self.get_existing_doc_ids()

        if existing_ids:
            logger.info(f"Found {len(existing_ids)} already-processed docs — skipping them.")
            pdf_files = [p for p in pdf_files if p.stem not in existing_ids]
            self.skipped = len(existing_ids)

        logger.info(f"Found {len(pdf_files)} PDFs to process")

        for pdf in tqdm(pdf_files, desc="Extracting PDF text"):

            try:
                record = self.process_pdf(pdf)
                self.add_record(record)

            except Exception as e:
                logger.error(f"Error processing {pdf}: {e}")
                self.failed += 1

        if self.buffer:
            self.write_batch()

        logger.info(
            f"PDF extraction done. Processed: {self.processed}, "
            f"Skipped (already done): {self.skipped}, Failed: {self.failed}"
        )


# ----------------------------
def main():

    parser = PDFParser(
        pdf_dir="data/pdfs",
        output_dir="pdf_text_data",
        batch_size=50,
    )

    parser.run()


if __name__ == "__main__":
    main()
