-- One-time migration: switch legal_chunks from local MiniLM embeddings (384-dim)
-- to Gemini's hosted embedding API (768-dim), so the Supabase-backed query path
-- has no torch/sentence-transformers dependency and can run as a Vercel
-- serverless function.
--
-- Run this in the Supabase dashboard: project "legaldoc-ai" -> SQL Editor -> paste -> Run.
-- This truncates legal_chunks (5,648 rows) — safe, since it's fully regenerated
-- by `python pipeline.py` from your local chunks_data/ right after this runs.

truncate table legal_chunks;

drop index if exists legal_chunks_embedding_idx;

alter table legal_chunks alter column embedding type vector(768);

drop function if exists match_legal_chunks(vector, int);

create or replace function match_legal_chunks (
    query_embedding vector(768),
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
set search_path = public
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
