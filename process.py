"""
Metadata Processor for Court Case Data (Parquet Version)

- Outputs Parquet dataset (partitioned)
- Removes raw_html, keeps cleaned_html
- Adds text_for_embedding field
- Optimized for large-scale processing
"""

from pathlib import Path
import json
import re
from datetime import datetime
from typing import Optional, Dict, Set

import lxml.html as LH
from lxml import html
import pandas as pd
from tqdm import tqdm
import pyarrow as pa
import pyarrow.parquet as pq

from src.utils.logger import logger


# ----------------------------
# HTML CLEANER
# ----------------------------
class HTMLCleaner:
    @staticmethod
    def clean_html(raw_html: str) -> str:
        if not raw_html:
            return ""

        try:
            tree = html.fromstring(raw_html)
            text = tree.text_content()
            text = re.sub(r'\s+', ' ', text).strip()
            return text
        except Exception:
            return raw_html

    @staticmethod
    def decode_html_entities(text: str) -> str:
        if not text:
            return ""

        try:
            tree = html.fromstring(f"<div>{text}</div>")
            return tree.text_content()
        except:
            return text


# ----------------------------
# METADATA EXTRACTOR
# ----------------------------
class MetadataExtractor:
    @staticmethod
    def extract_case_details(raw_html: str) -> Dict[str, Optional[str]]:
        case_details = {
            "cnr": None,
            "date_of_registration": None,
            "decision_date": None,
            "decision_year": None,
            "disposal_nature": None,
            "court": None,
        }

        if not raw_html:
            return case_details

        try:
            tree = html.fromstring(raw_html)
            element = tree.xpath('//strong[@class="caseDetailsTD"]')

            if not element:
                return case_details

            element = element[0]

            def extract(label):
                try:
                    return element.xpath(
                        f'.//span[contains(text(), "{label}")]/following-sibling::font/text()'
                    )[0].strip()
                except:
                    return None

            case_details["cnr"] = extract("CNR")
            case_details["date_of_registration"] = extract("Date of registration")

            date_str = extract("Decision Date")
            if date_str:
                case_details["decision_date"] = date_str
                for fmt in ("%d-%m-%Y", "%Y-%m-%d", "%d/%m/%Y"):
                    try:
                        case_details["decision_year"] = datetime.strptime(date_str, fmt).year
                        break
                    except:
                        continue

            case_details["disposal_nature"] = extract("Disposal Nature")

            try:
                court_text = element.xpath('.//span[contains(text(), "Court")]/text()')[0]
                case_details["court"] = court_text.split(":")[1].strip()
            except:
                pass

        except Exception:
            pass

        return case_details


# ----------------------------
# MAIN PROCESSOR
# ----------------------------
class MetadataProcessor:

    def __init__(
        self,
        json_dir: str,
        pdf_dir: str,
        batch_size: int = 2000,
        output_dir: str = "processed_data",
        clean_html: bool = True,
    ):
        self.json_dir = Path(json_dir)
        self.pdf_dir = Path(pdf_dir)
        self.batch_size = batch_size
        self.clean_html = clean_html

        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.record_buffer = []
        self.record_count = 0
        self.skipped_count = 0

        self.html_cleaner = HTMLCleaner()
        self.metadata_extractor = MetadataExtractor()

        # 🔥 Improved schema
        self.output_fields = [
            "doc_id",
            "court_code",
            "court",
            "title",
            "description",
            "judge",
            "cnr",
            "date_of_registration",
            "decision_date",
            "decision_year",
            "disposal_nature",
            "pdf_link",
            "pdf_exists",
            "cleaned_html",
            "text_for_embedding",
        ]

    # ----------------------------
    def get_metadata_files(self):
        return sorted(self.json_dir.glob("**/*.json"))

    def get_existing_doc_ids(self) -> Set[str]:
        """Doc ids already present in the output dataset, for idempotent reruns."""
        try:
            existing = pd.read_parquet(self.output_dir, columns=["doc_id"])
            return set(existing["doc_id"].unique())
        except Exception:
            return set()

    def get_pdf_path(self, json_file: Path) -> Path:
        return self.pdf_dir / (json_file.stem + ".pdf")

    def load_metadata(self, file: Path) -> dict:
        with open(file, 'r', encoding='utf-8') as f:
            return json.load(f)

    # ----------------------------
    def process_metadata(self, metadata: dict, file: Path) -> Optional[dict]:

        if "raw_html" not in metadata:
            return None

        raw_html = metadata["raw_html"]

        try:
            html_element = LH.fromstring(raw_html)
        except:
            return None

        # Extract fields
        title = ""
        description = ""
        judge_name = ""

        try:
            title = html_element.xpath("./button//text()")[0].strip()
            title = self.html_cleaner.decode_html_entities(title)
        except:
            pass

        try:
            desc = html_element.xpath("./text()")
            description = desc[0].strip() if desc else ""
            description = self.html_cleaner.decode_html_entities(description)
        except:
            pass

        try:
            judge_txt = html_element.xpath("./strong/text()")
            if judge_txt and ":" in judge_txt[0]:
                judge_name = judge_txt[0].split(":", 1)[1].strip()
        except:
            pass

        cleaned_html = self.html_cleaner.clean_html(raw_html)

        # Extract structured metadata
        parsed = self.metadata_extractor.extract_case_details(raw_html)

        pdf_filename = Path(metadata.get("pdf_link", "")).name
        pdf_path = self.get_pdf_path(file)

        # 🔥 Create embedding text
        text_for_embedding = " ".join([
            title or "",
            description or "",
            cleaned_html or ""
        ]).strip()

        return {
            "doc_id": file.stem,
            "court_code": metadata.get("court_code", ""),
            "court": parsed.get("court"),
            "title": title,
            "description": description,
            "judge": judge_name,
            "cnr": parsed.get("cnr"),
            "date_of_registration": parsed.get("date_of_registration"),
            "decision_date": parsed.get("decision_date"),
            "decision_year": parsed.get("decision_year"),
            "disposal_nature": parsed.get("disposal_nature"),
            "pdf_link": pdf_filename,
            "pdf_exists": pdf_path.exists(),
            "cleaned_html": cleaned_html,
            "text_for_embedding": text_for_embedding,
        }

    # ----------------------------
    def add_record(self, record: dict):
        self.record_buffer.append(record)

        if len(self.record_buffer) >= self.batch_size:
            self.write_batch()

    # ----------------------------
    def write_batch(self):

        if not self.record_buffer:
            return

        df = pd.DataFrame(self.record_buffer)

        for field in self.output_fields:
            if field not in df.columns:
                df[field] = None

        df = df[self.output_fields]

        # Convert date
        df["decision_date"] = pd.to_datetime(
            df["decision_date"],
            errors="coerce",
            dayfirst=True
        )

        table = pa.Table.from_pandas(df, preserve_index=False)

        # 🔥 Partitioned Parquet dataset
        pq.write_to_dataset(
            table,
            root_path=str(self.output_dir),
            partition_cols=["decision_year"]
        )

        self.record_count += len(df)
        self.record_buffer = []

    # ----------------------------
    def process(self, force: bool = False):

        logger.info(f"Processing from: {self.json_dir}")
        logger.info(f"Saving to: {self.output_dir}")

        files = self.get_metadata_files()
        existing_ids = set() if force else self.get_existing_doc_ids()

        if existing_ids:
            files = [f for f in files if f.stem not in existing_ids]
            self.skipped_count = len(existing_ids)
            logger.info(f"Found {len(existing_ids)} already-processed docs — skipping them.")

        for file in tqdm(files, desc="Processing metadata"):

            try:
                metadata = self.load_metadata(file)
                processed = self.process_metadata(metadata, file)

                if not processed:
                    self.skipped_count += 1
                    continue

                self.add_record(processed)

            except Exception as e:
                logger.error(f"Error: {file} -> {e}")
                self.skipped_count += 1

        if self.record_buffer:
            self.write_batch()

        logger.info(f"Metadata processing done. Processed: {self.record_count}, Skipped: {self.skipped_count}")


# ----------------------------
# ENTRY POINT
# ----------------------------
def main():
    processor = MetadataProcessor(
        json_dir="data/json",
        pdf_dir="data/pdfs",
        batch_size=2000,
        output_dir="processed_data",
    )

    processor.process()


if __name__ == "__main__":
    main()