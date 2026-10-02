LegalDoc AI — Build Plan (Gemini LLM, full-quality pipeline, eval designed in)
Target architecture
data/ → pdf_parser + process → parquet → chunking → chunks parquet
                                                     │
                                                     ▼
                                      vector_store (FAISS, all-MiniLM-L6-v2)
                                                     │
User → Streamlit UI → query_pipeline (retriever + Gemini) → answer + citations
            │
            ├─ summarize (Gemini, structured sections)
            ├─ doc_generator (Gemini, contract drafting + refine)
            └─ evaluation (scorer on accuracy/completeness/relevance → feedback loop)

Phase 0 — Data foundation (fix before anything)

- De-duplicate + refresh: make pipeline idempotent (track processed doc_ids), then wipe pdf_text_data/, processed_data/, chunks_data/ and rerun pipeline.py once → clean 1171 docs, no duplication.
- Fix stale chunks: after rerun, verify chunks_data has chunk_text + contextual_text + decision_year columns (current stored chunks are from an old schema and lack contextual_text).
- Consolidate vector_store.py: keep one class (ChunkedVectorStore, all-MiniLM-L6-v2 — cached), delete the redundant FAISSIndexer (different model = latent retrieval bug).
- git init, add .env.example, commit Phase 0.

Phase 1 — Vector store (the missing backbone)
- Finish ChunkedVectorStore.build() → write vector_store/.
- Add vector_store/load_store() helper (currently no loader exists — retriever can't read it back).
- Verify: unit test that a sample query returns chunks with correct metadata (court, judge, year).

Phase 2 — Retriever (src/rag/retriever.py)
- Fix the syntax-error file → from langchain_community.vectorstores import FAISS etc.
- Load persisted store → as_retriever(k=5).
- Add hybrid BM25 rerank (rank-bm25) so exact legal terms ("indemnity", "jurisdiction") aren't drowned by vector similarity.

Phase 3 — RAG query pipeline (src/rag/query_pipeline.py)
- The "real chatbot" milestone: retrieve → assemble prompt with citations → Gemini answer → return {answer, sources[{chunk, court, judge, year}], used_gemini}.
- Gemini integration: add google-generativeai + langchain-google-genai to requirements; ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.2); key from GOOGLE_API_KEY in .env.
- Structure LLM calls behind a small interface so eval can reuse them.

Phase 4 — Summarization (src/summarization/summarize.py)
- Rewrite (current file imports nonexistent langchain_summarize and uses invalid gpt-5-turbo).
- Use load_summarize_chain with stuff/map_reduce for long docs → structured sections (obligations / risks / key clauses / deadlines), with chunk-level sources.

Phase 5 — Document generation (src/generation/doc_generator.py)
- Prompt-engineered drafting (NDA, freelance contract, agreement) from user requirements + optionally retrieved precedent clauses from the vector store.
- Iterative refine: "revise/expand section X" endpoint.

Phase 6 — UI (src/ui/app.py)
- Streamlit: upload PDF → auto-ingest; chat tab (RAG w/ citations); summarize tab; generate tab (draft + refine); export to DOCX (python-docx already in requirements).

Phase 7 — Evaluation loop (designed in now)
- src/evaluation/evaluate.py: Gemini-as-judge scoring responses on accuracy / completeness / relevance (1–5).
- Feedback loop: if score < threshold → re-prompt with specific improvement notes (cap iterations).
- Wire it as a post-processing step in the query pipeline so it's not bolted on later.

Phase 8 — Production hardening
- FastAPI wrapper + Streamlit front (FastAPI already in requirements), pytest suite, loguru logging, Dockerfile, .env.example with GOOGLE_API_KEY.

Suggested first working session (executable now)
1. Phase 0 (de-dup + rebuild data) → 2. Phase 1 (build + load vector store, verify retrieval) → 3. Phase 2 (retriever, pytest on retrieval quality).