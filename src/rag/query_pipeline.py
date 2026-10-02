"""RAG query pipeline: retrieve relevant chunks, assemble a cited prompt,
call Gemini, and return a structured answer with sources.

Falls back to an extractive (LLM-free) answer when no API key is
configured, so retrieval can be exercised and demoed without Gemini.
"""
from __future__ import annotations

from langchain_core.documents import Document

from src.evaluation.evaluate import evaluate_with_feedback_loop
from src.llm import invoke_text
from src.rag.retriever import HybridRetriever, load_retriever
from src.utils.logger import logger

SYSTEM_PROMPT = """You are a legal assistant. Answer the question using ONLY the provided \
context, which consists of excerpts from court judgments. If the context does not contain \
enough information to answer, say so plainly instead of guessing.

Cite the source of each claim inline using the bracketed number of the excerpt it came from, \
e.g. [1]. Answer clearly and precisely, in the tone of a professional legal assistant.

Context:
{context}

Question:
{question}

Answer:"""


def _format_source(doc: Document, index: int) -> str:
    meta = doc.metadata
    header = " | ".join(
        str(v)
        for v in [meta.get("title"), meta.get("court"), meta.get("judge"), meta.get("decision_year")]
        if v
    )
    text = meta.get("chunk_text") or doc.page_content
    return f"[{index}] ({header})\n{text}"


def _to_source_dict(doc: Document, index: int) -> dict:
    meta = doc.metadata
    return {
        "index": index,
        "doc_id": meta.get("doc_id"),
        "chunk_id": meta.get("chunk_id"),
        "title": meta.get("title"),
        "court": meta.get("court"),
        "judge": meta.get("judge"),
        "decision_year": meta.get("decision_year"),
        "excerpt": (meta.get("chunk_text") or doc.page_content)[:500],
    }


class RAGQueryPipeline:
    def __init__(self, retriever: HybridRetriever | None = None):
        self.retriever = retriever or load_retriever()

    def assemble_prompt(self, query: str, docs: list[Document]) -> str:
        context = "\n\n".join(_format_source(doc, i + 1) for i, doc in enumerate(docs))
        return SYSTEM_PROMPT.format(context=context, question=query)

    def answer(self, query: str, k: int | None = None, enable_eval: bool = False) -> dict:
        docs = self.retriever.retrieve(query, k=k)
        sources = [_to_source_dict(doc, i + 1) for i, doc in enumerate(docs)]

        if not docs:
            return {
                "answer": "No relevant documents were found in the corpus for this question.",
                "sources": [],
                "used_gemini": False,
                "eval": None,
            }

        context = "\n\n".join(_format_source(doc, i + 1) for i, doc in enumerate(docs))
        llm_answer = invoke_text(self.assemble_prompt(query, docs))

        if llm_answer is None:
            logger.info("Gemini unavailable — returning extractive fallback answer.")
            fallback = (
                "LLM answer generation is unavailable (no GOOGLE_API_KEY configured). "
                "Showing the most relevant excerpts instead:\n\n" + context
            )
            return {"answer": fallback, "sources": sources, "used_gemini": False, "eval": None}

        if not enable_eval:
            return {"answer": llm_answer, "sources": sources, "used_gemini": True, "eval": None}

        def regenerate(_query: str, previous_response: str, feedback: str) -> str:
            from src.evaluation.evaluate import default_regenerate

            return default_regenerate(_query, context, previous_response, feedback)

        eval_result = evaluate_with_feedback_loop(query, context, llm_answer, regenerate)
        return {
            "answer": eval_result["response"],
            "sources": sources,
            "used_gemini": True,
            "eval": eval_result["score"],
        }


def main() -> None:
    pipeline = RAGQueryPipeline()
    query = "Tell me about the case on ARVIND SINGH SANGWAN. What was the final decision?"
    result = pipeline.answer(query)
    print(result["answer"])
    print("\nSources:")
    for src in result["sources"]:
        print(src)


if __name__ == "__main__":
    main()
