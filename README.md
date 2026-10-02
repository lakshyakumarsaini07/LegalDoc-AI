# LegalDoc AI — Smart Legal Document Assistant

An AI-powered assistant for law professionals: RAG-powered search over a
corpus of court judgments, legal document summarization, and AI-assisted
legal document generation with iterative refinement, backed by Gemini.

## Architecture

```
data/pdfs, data/json
      │
      ▼
pdf_parser.py ──► pdf_text_data/        process.py ──► processed_data/
      │                                                       │
      └──────────────────────┬────────────────────────────────┘
                              ▼
                    chunking.py ──► chunks_data/
                              │
                              ▼
        vector_store.py (FAISS)  or  supabase_store.py (pgvector)
                              │
                              ▼
          retriever.py (vector search + BM25, Reciprocal Rank Fusion)
                              │
                              ▼
                query_pipeline.py ──► Gemini ──► answer + citations
                              │
                     evaluate.py (optional judge + feedback loop)

summarize.py / doc_generator.py — standalone Gemini pipelines for
user-uploaded documents, independent of the corpus above.

src/ui/app.py (Streamlit) and src/api/main.py (FastAPI) both sit on top
of the same pipeline modules.
```

## Setup

```bash
python -m venv venv
venv\Scripts\activate        # Windows
pip install -r requirements.txt
copy .env.example .env       # then set GOOGLE_API_KEY (aistudio.google.com/apikey)
```

Without `GOOGLE_API_KEY` set, every LLM-backed feature (chat answers,
summarization, document generation, evaluation) degrades gracefully to a
retrieval-only / template-only fallback instead of failing, so the app
is usable before a key is provisioned.

## Build the index

Place PDFs in `data/pdfs/` and matching metadata JSON in `data/json/`
(filenames must share the same stem), then run:

```bash
python pipeline.py            # incremental: skips docs already processed
python pipeline.py --force    # reprocess everything from scratch
```

This extracts PDF text, parses metadata, chunks the text (with
case-context prefixed onto each chunk for better retrieval), and builds
the vector store.

**Vector store backend** — set via `VECTOR_STORE_BACKEND` in `.env`:

- `faiss` (default): a local index written to `vector_store/`. No external
  service, works offline.
- `supabase`: embeddings + chunk metadata live in a Supabase Postgres table
  (pgvector), so retrieval works across multiple app instances/deployments
  instead of a single local file. Apply `supabase/schema.sql` to your
  Supabase project first (SQL Editor, or `supabase db push`), then set
  `SUPABASE_URL` and `SUPABASE_KEY` (service role key) in `.env` before
  running `pipeline.py`. BM25 (for hybrid search) still runs in-process,
  fetching the corpus from the `legal_chunks` table at retriever load time.

## Run it

**Streamlit UI** (chat, summarize, generate — all in one app):

```bash
streamlit run src/ui/app.py
```

**FastAPI** (for programmatic/integration use):

```bash
uvicorn src.api.main:app --reload
```

Endpoints: `GET /health`, `POST /chat`, `POST /summarize`, `POST /generate`, `POST /refine`.

**Docker**:

```bash
docker compose up --build
```

## Tests

```bash
pytest
```

The test suite runs entirely offline (LLM calls are mocked), so it
exercises retrieval fusion logic, the data pipeline's idempotency and
schema, and every pipeline's LLM-available / LLM-unavailable code paths.

## Project layout

- `src/preprocessing/pdf_parser.py` — PDF → text (idempotent)
- `process.py` — JSON metadata → cleaned, structured records (idempotent)
- `src/embeddings/chunking.py` — text → context-aware chunks
- `src/embeddings/vector_store.py` — chunks → FAISS index (local backend)
- `src/embeddings/supabase_store.py` — chunks → Supabase pgvector (cloud backend)
- `supabase/schema.sql` — pgvector table + similarity-search RPC for the Supabase backend
- `src/rag/retriever.py` — hybrid vector + BM25 retrieval (RRF), backend-agnostic
- `src/rag/query_pipeline.py` — retrieval → Gemini → cited answer
- `src/summarization/summarize.py` — map-reduce document summarization
- `src/generation/doc_generator.py` — document drafting + refinement
- `src/evaluation/evaluate.py` — Gemini-as-judge scoring + feedback loop
- `src/ui/app.py` — Streamlit UI
- `src/api/main.py` — FastAPI wrapper
- `src/config.py` — central configuration (env-driven)
- `pipeline.py` — orchestrates the full ingestion pipeline
- `tests/` — pytest suite

## Configuration

All tunables (chunk size, retriever candidates, Gemini model/temperature,
evaluation thresholds, API host/port) are environment variables — see
`.env.example`.
