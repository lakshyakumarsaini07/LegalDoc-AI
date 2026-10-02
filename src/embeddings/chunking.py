from pathlib import Path
import pandas as pd 
import pyarrow as pa
import pyarrow.parquet as pq 
from tqdm import tqdm 

from langchain_text_splitters import RecursiveCharacterTextSplitter

def build_contextual_chunk(row, chunk):
    return f"""
    Case Title: {row.get('title', '')}
    Court: {row.get('court', '')}
    Judge: {row.get('judge', '')} 
    Decision Year: {row.get('decision_year', '')}
    Content:{chunk}""".strip()

class ChunkProcessor:

    def __init__(
        self,
        pdf_parquet_path: str,
        metadata_parquet_path: str,
        output_dir: str = "chunks_data",
        chunk_size: int = 800,
        chunk_overlap: int = 150,
        batch_size: int = 1000,
    ):
        self.pdf_path = pdf_parquet_path
        self.meta_path = metadata_parquet_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.batch_size = batch_size

        self.buffer = []

        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ".", " "],
        )

    def load_data(self):
        pdf_df = pd.read_parquet(self.pdf_path)
        meta_df = pd.read_parquet(self.meta_path)

        return pdf_df.merge(meta_df, on="doc_id")

    def process(self):

        df = self.load_data()

        for _, row in tqdm(df.iterrows(), total=len(df)):

            text = row.get("pdf_text", "")
            if not text:
                continue

            chunks = self.splitter.split_text(text)

            for i, chunk in enumerate(chunks):
                self.buffer.append({
                    "doc_id": row["doc_id"],
                    "chunk_id": f"{row['doc_id']}_{i}",
                    "contextual_text": build_contextual_chunk(row, chunk),
                    "chunk_text": chunk,  # keep raw too (important for display)
                    "title": row.get("title"),
                    "court": row.get("court"),
                    "decision_year": row.get("decision_year"),
                    "judge": row.get("judge"),
                })

                if len(self.buffer) >= self.batch_size:
                    self.write_batch()
                                                        
        if self.buffer:
            self.write_batch()

    def write_batch(self):

        df = pd.DataFrame(self.buffer)
        table = pa.Table.from_pandas(df)

        pq.write_to_dataset(
            table,
            root_path=str(self.output_dir),
            partition_cols=["decision_year"]
        )

        self.buffer = []