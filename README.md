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
pip install -r requirements-local.txt
copy .env.example .env       # then set GOOGLE_API_KEY (aistudio.google.com/apikey)
```

(Named `requirements-local.txt` rather than `requirements.txt` deliberately —
Vercel's Python builder only reads a `requirements.txt` at the project root,
and the root `requirements.txt` in this repo is a trimmed set for the Vercel
deployment. This file has the full local-dev set, including
torch/sentence-transformers/faiss-cpu, which alone exceed Vercel's 500MB
function size limit by more than 10x.)

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
  instead of a single local file, and is the only backend light enough to
  run as a Vercel serverless function (see **Deployment** below). Apply
  `supabase/schema.sql` to your Supabase project first (SQL Editor, or
  `supabase db push`), then set `SUPABASE_URL` and `SUPABASE_KEY` (service
  role key) in `.env` before running `pipeline.py`. BM25 (for hybrid search)
  still runs in-process, fetching the corpus from the `legal_chunks` table at
  retriever load time. This backend embeds with **Gemini's hosted embedding
  API** (`GEMINI_EMBEDDING_MODEL`, 768-dim by default) rather than local
  sentence-transformers/torch — deliberately, to keep the backend's
  dependency footprint small. The FAISS backend always uses local MiniLM
  embeddings (384-dim); the two are not interchangeable without re-ingesting.

  **Free-tier quota note**: Gemini's free embedding quota is 1000
  requests/day (one chunk = one request — there's no true batch endpoint).
  The full corpus (5000+ chunks) doesn't fit in a day's quota on the free
  tier. For a demo/personal-project deployment, build a smaller subset first:
  ```bash
  python scripts/make_supabase_demo_subset.py 100   # ~400-500 chunks
  python -c "from src.embeddings.supabase_store import SupabaseVectorStore; SupabaseVectorStore().build(chunk_source='chunks_data_demo_subset.parquet')"
  ```
  `SupabaseVectorStore.build()` is resumable (skips chunk_ids already
  ingested) and rate-limit-aware (paces requests, backs off on 429s), so it's
  safe to re-run if interrupted. To ingest the full corpus, either spread it
  across ~6 days of free-tier quota or enable billing on the Gemini API key.

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

## Deployment

The backend (FastAPI) and frontend (Streamlit) deploy separately — **Streamlit
cannot run on Vercel** (it needs a persistent server with a live WebSocket
connection; Vercel only runs stateless serverless functions), so the backend
goes to Vercel and the UI goes to a host that supports long-running processes.

### Backend → Vercel

Requires `VECTOR_STORE_BACKEND=supabase` (the FAISS backend needs
sentence-transformers/torch, which are far too large for a serverless
function — the root `requirements.txt` deliberately excludes them; local
full-stack dev uses `requirements-local.txt` instead). `api/index.py` is the
serverless entrypoint; `vercel.json` routes every path to it.

```bash
npm i -g vercel          # if you don't already have the CLI
cd "d:/LegalDoc AI"
vercel login
vercel link              # creates/links a Vercel project for this repo

vercel env add GOOGLE_API_KEY production
vercel env add VECTOR_STORE_BACKEND production   # value: supabase
vercel env add SUPABASE_URL production
vercel env add SUPABASE_KEY production           # service_role key
# optional: API_KEY (protect the endpoints), ALLOWED_ORIGINS (your Streamlit URL)

vercel --prod
```

Cold starts rebuild the in-process BM25 index from the full `legal_chunks`
table on every cold start (Supabase has no built-in BM25), which costs a few
seconds of latency under low traffic — a known tradeoff of this architecture,
not a bug.

### Frontend → Streamlit Community Cloud (recommended)

Free, official, zero code changes needed.

1. Push this repo to GitHub.
2. [share.streamlit.io](https://share.streamlit.io) → New app → point at your
   repo, branch, and `src/ui/app.py`.
3. App settings → Secrets → paste (TOML format):
   ```toml
   GOOGLE_API_KEY = "..."
   VECTOR_STORE_BACKEND = "supabase"
   SUPABASE_URL = "..."
   SUPABASE_KEY = "..."
   ```
4. Deploy. (`src/ui/app.py` mirrors `st.secrets` into `os.environ` at
   startup, so the same `src.config` env-var reads work unchanged.)

Render, Railway, Fly.io, or Hugging Face Spaces work equally well if you'd
rather not use Streamlit Community Cloud — any host that runs a persistent
process works; the constraint is specifically about Vercel's serverless model.

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
- `src/embeddings/vector_store.py` — chunks → FAISS index (local MiniLM embeddings)
- `src/embeddings/supabase_store.py` — chunks → Supabase pgvector (Gemini embeddings)
- `src/embeddings/gemini_embeddings.py` — Gemini embedding API wrapper (no torch dependency)
- `supabase/schema.sql` — pgvector table + similarity-search RPC for the Supabase backend
- `api/index.py`, `requirements.txt` (root, trimmed), `vercel.json` — Vercel serverless deployment
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
