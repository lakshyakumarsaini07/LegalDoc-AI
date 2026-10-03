"""Streamlit UI for LegalDoc AI: chat (RAG w/ citations), summarize,
generate (draft + refine), and DOCX export.

Run with: streamlit run src/ui/app.py
"""
from __future__ import annotations

import io
import os

import streamlit as st

# On Streamlit Community Cloud, config is set via the "Secrets" UI (TOML) and
# exposed through st.secrets; mirror it into os.environ before src.config
# reads env vars, so the same code works there, locally, and via Docker.
try:
    for _key, _value in st.secrets.items():
        os.environ.setdefault(_key, str(_value))
except Exception:
    pass

from docx import Document as DocxDocument

from src.config import VECTOR_STORE_BACKEND, has_llm_configured, vector_store_ready
from src.generation.doc_generator import DocumentGenerator
from src.summarization.summarize import SummarizationService

st.set_page_config(page_title="LegalDoc AI", page_icon="⚖️", layout="wide")


@st.cache_resource
def _get_query_pipeline():
    from src.rag.query_pipeline import RAGQueryPipeline

    return RAGQueryPipeline()


def _docx_bytes(text: str) -> bytes:
    doc = DocxDocument()
    for line in text.split("\n"):
        doc.add_paragraph(line)
    buffer = io.BytesIO()
    doc.save(buffer)
    return buffer.getvalue()


def _llm_status_banner() -> None:
    if not has_llm_configured():
        st.warning(
            "GOOGLE_API_KEY is not set — running in retrieval-only fallback mode. "
            "Add a key to .env and restart to enable AI answers, summaries, and drafting.",
            icon="⚠️",
        )


def render_sidebar() -> None:
    with st.sidebar:
        st.subheader("System status")
        st.markdown(f"**LLM (Gemini):** {'✅ configured' if has_llm_configured() else '⚠️ not configured'}")
        st.markdown(f"**Vector store backend:** `{VECTOR_STORE_BACKEND}`")

        if vector_store_ready():
            st.markdown("**Vector store:** ✅ ready")
            try:
                pipeline = _get_query_pipeline()
                docs = pipeline.retriever.docs
                unique_doc_ids = {d.metadata.get("doc_id") for d in docs if d.metadata.get("doc_id")}
                st.caption(f"{len(docs)} indexed chunks across {len(unique_doc_ids)} documents")
            except Exception as exc:  # noqa: BLE001
                st.caption(f"Could not load corpus stats: {exc}")
        else:
            st.markdown("**Vector store:** ❌ not built")
            st.caption("Run `python pipeline.py` to ingest documents.")


def render_chat_tab() -> None:
    st.subheader("Ask questions across the case law corpus")

    if not vector_store_ready():
        st.error(
            f"Vector store backend `{VECTOR_STORE_BACKEND}` is not ready. Run `python pipeline.py` "
            "first to ingest documents and build the index (and set SUPABASE_URL/SUPABASE_KEY if "
            "using the supabase backend)."
        )
        return

    try:
        _get_query_pipeline()
    except Exception as exc:  # noqa: BLE001
        st.error(f"Failed to load the vector store/retriever: {exc}")
        return

    col1, col2 = st.columns([3, 1])
    with col1:
        enable_eval = st.checkbox(
            "Enable evaluation loop (scores + auto-refines the answer; slower)", value=False
        )
    with col2:
        if st.button("Clear chat"):
            st.session_state.chat_history = []
            st.rerun()

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    for entry in st.session_state.chat_history:
        with st.chat_message(entry["role"]):
            st.markdown(entry["content"])

    query = st.chat_input("Ask about a case, clause, or precedent...")
    if query:
        st.session_state.chat_history.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)

        with st.chat_message("assistant"):
            with st.spinner("Retrieving and answering..."):
                pipeline = _get_query_pipeline()
                result = pipeline.answer(query, enable_eval=enable_eval)

            st.markdown(result["answer"])

            if result["eval"]:
                st.caption(
                    f"Eval score — accuracy: {result['eval']['accuracy']}, "
                    f"completeness: {result['eval']['completeness']}, "
                    f"relevance: {result['eval']['relevance']}"
                )

            if result["sources"]:
                with st.expander(f"Sources ({len(result['sources'])})"):
                    for src in result["sources"]:
                        st.markdown(
                            f"**[{src['index']}] {src.get('title') or src.get('doc_id')}** "
                            f"— {src.get('court') or 'Unknown court'}, {src.get('decision_year') or 'n/a'}"
                        )
                        st.caption(src["excerpt"])

        st.session_state.chat_history.append({"role": "assistant", "content": result["answer"]})


def render_summarize_tab() -> None:
    st.subheader("Summarize a legal document")

    uploaded = st.file_uploader("Upload a .txt or .pdf file", type=["txt", "pdf"])
    pasted = st.text_area("...or paste the document text", height=200)

    text = ""
    if uploaded is not None:
        if uploaded.name.endswith(".pdf"):
            from pypdf import PdfReader

            reader = PdfReader(uploaded)
            text = "\n".join(page.extract_text() or "" for page in reader.pages)
        else:
            text = uploaded.read().decode("utf-8", errors="ignore")
    elif pasted.strip():
        text = pasted

    if st.button("Summarize", type="primary", disabled=not text.strip()):
        with st.spinner("Summarizing..."):
            result = SummarizationService().summarize(text)

        st.session_state["last_summary"] = result["summary"]
        st.markdown(result["summary"])

    if st.session_state.get("last_summary"):
        st.download_button(
            "Download as DOCX",
            data=_docx_bytes(st.session_state["last_summary"]),
            file_name="summary.docx",
            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        )


def render_generate_tab() -> None:
    st.subheader("Generate a legal document")

    document_type = st.selectbox(
        "Document type",
        ["Non-Disclosure Agreement", "Freelance/Service Contract", "Employment Agreement", "Other"],
    )
    if document_type == "Other":
        document_type = st.text_input("Specify document type")

    requirements = st.text_area(
        "Describe the requirements",
        placeholder="e.g. Between a startup and a freelance developer, mutual confidentiality, 2 year term.",
        height=120,
    )

    if st.button("Generate draft", type="primary", disabled=not requirements.strip()):
        with st.spinner("Drafting..."):
            result = DocumentGenerator().generate(document_type, requirements)
        st.session_state["draft"] = result["document"]

    if st.session_state.get("draft"):
        st.text_area("Draft", value=st.session_state["draft"], height=400, key="draft_view")

        instruction = st.text_input("Refine instruction", placeholder="e.g. Add a governing law clause for Delaware.")
        if st.button("Apply refinement", disabled=not instruction.strip()):
            with st.spinner("Refining..."):
                result = DocumentGenerator().refine(st.session_state["draft"], instruction)
            st.session_state["draft"] = result["document"]
            st.rerun()

        st.download_button(
            "Download as DOCX",
            data=_docx_bytes(st.session_state["draft"]),
            file_name=f"{document_type.replace(' ', '_').lower()}.docx",
            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        )


def main() -> None:
    st.title("⚖️ LegalDoc AI")
    st.caption("Smart Legal Document Assistant — RAG search, summarization, and document generation.")

    render_sidebar()
    _llm_status_banner()

    tab_chat, tab_summarize, tab_generate = st.tabs(["Chat / Search", "Summarize", "Generate"])
    with tab_chat:
        render_chat_tab()
    with tab_summarize:
        render_summarize_tab()
    with tab_generate:
        render_generate_tab()


main()
