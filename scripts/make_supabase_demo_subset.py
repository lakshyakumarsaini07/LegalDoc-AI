"""Build a small subset of chunks_data/ for ingestion into the Supabase backend.

Gemini's free-tier embedding quota is 1000 requests/day, and each chunk costs
one request to embed — the full corpus (5000+ chunks) doesn't fit in a single
day's quota. This selects the first N documents' worth of chunks (a few
hundred, well under the quota) so the Supabase/Vercel-chat demo path works
without needing a paid Gemini tier. The FAISS backend is unaffected and still
indexes the full local corpus.

Usage: python scripts/make_supabase_demo_subset.py [n_docs] [output_path]
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import CHUNKS_DATA_DIR  # noqa: E402


def main() -> None:
    n_docs = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    output_path = sys.argv[2] if len(sys.argv) > 2 else "chunks_data_demo_subset.parquet"

    df = pd.read_parquet(CHUNKS_DATA_DIR)
    doc_counts = df.groupby("doc_id").size()
    subset_docs = set(doc_counts.index[:n_docs])
    subset = df[df["doc_id"].isin(subset_docs)].reset_index(drop=True)
    subset.to_parquet(output_path)

    print(f"{len(subset)} chunks across {subset['doc_id'].nunique()} documents -> {output_path}")


if __name__ == "__main__":
    main()
