-- LegalDoc AI — Supabase pgvector schema.
--
-- Apply this in two phases (SQL Editor, `supabase db push`, or the Supabase
-- MCP's apply_migration):
--
-- Phase 1 (before ingestion) — everything below up to the "PHASE 2" marker.
-- Phase 2 (after `python pipeline.py` has populated the table) — the ivfflat
-- index at the bottom. Building an IVF index on an empty table produces
-- degenerate clusters (poor recall), since it picks centroids from whatever
-- rows exist at CREATE INDEX time — so it must come after the data load.
--
-- The embedding dimension (384) matches the default EMBEDDING_MODEL
-- (all-MiniLM-L6-v2). If you change EMBEDDING_MODEL, update the `vector(384)`
-- dimensions below (and EMBEDDING_DIM in .env) to match its output size.

create extension if not exists vector;

create table if not exists legal_chunks (
    id bigserial primary key,
    chunk_id text not null unique,
    doc_id text not null,
    title text,
    court text,
    judge text,
    decision_year int,
    content text not null,     -- contextual_text: what gets embedded
    chunk_text text not null,  -- raw chunk: what gets shown/cited
    embedding vector(384) not null,
    created_at timestamptz not null default now()
);

-- `chunk_id` is the upsert conflict target (idempotent re-ingestion: rerunning
-- pipeline.py against the same chunk_id replaces that row instead of duplicating it).
create index if not exists legal_chunks_doc_id_idx on legal_chunks (doc_id);

-- RPC used by src/embeddings/supabase_store.py for similarity search.
create or replace function match_legal_chunks (
    query_embedding vector(384),
    match_count int default 5
) returns table (
    id bigint,
    chunk_id text,
    doc_id text,
    title text,
    court text,
    judge text,
    decision_year int,
    content text,
    chunk_text text,
    similarity float
)
language sql stable
as $$
    select
        legal_chunks.id,
        legal_chunks.chunk_id,
        legal_chunks.doc_id,
        legal_chunks.title,
        legal_chunks.court,
        legal_chunks.judge,
        legal_chunks.decision_year,
        legal_chunks.content,
        legal_chunks.chunk_text,
        1 - (legal_chunks.embedding <=> query_embedding) as similarity
    from legal_chunks
    order by legal_chunks.embedding <=> query_embedding
    limit match_count;
$$;

-- ============================================================
-- PHASE 2 — run only after the table has been populated
-- ============================================================

-- Approximate nearest-neighbor index for cosine similarity search.
-- `lists` should be roughly sqrt(row_count); 100 is a reasonable default
-- for a corpus in the tens of thousands of chunks — rebuild with a larger
-- value (`reindex`) as the table grows well past that.
create index if not exists legal_chunks_embedding_idx
    on legal_chunks using ivfflat (embedding vector_cosine_ops) with (lists = 100);
