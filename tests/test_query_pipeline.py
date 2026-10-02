from langchain_core.documents import Document

from src.rag import query_pipeline as query_pipeline_module
from src.rag.query_pipeline import RAGQueryPipeline


class FakeRetriever:
    def __init__(self, docs):
        self.docs = docs

    def retrieve(self, query, k=None):
        return self.docs[:k] if k else self.docs


def _sample_docs():
    return [
        Document(
            page_content="contextual text about indemnity",
            metadata={
                "chunk_id": "doc1_0",
                "doc_id": "doc1",
                "chunk_text": "The indemnity clause requires Party A to compensate Party B.",
                "title": "Sample Case",
                "court": "High Court",
                "judge": "Justice X",
                "decision_year": 2023,
            },
        )
    ]


def test_answer_with_no_retrieved_docs_short_circuits(monkeypatch):
    pipeline = RAGQueryPipeline(retriever=FakeRetriever([]))
    result = pipeline.answer("What is indemnity?")

    assert result["sources"] == []
    assert result["used_gemini"] is False
    assert "no relevant documents" in result["answer"].lower()


def test_answer_falls_back_without_llm(monkeypatch):
    monkeypatch.setattr(query_pipeline_module, "invoke_text", lambda *a, **k: None)
    pipeline = RAGQueryPipeline(retriever=FakeRetriever(_sample_docs()))

    result = pipeline.answer("What is indemnity?")

    assert result["used_gemini"] is False
    assert len(result["sources"]) == 1
    assert result["sources"][0]["doc_id"] == "doc1"


def test_answer_uses_llm_when_available(monkeypatch):
    monkeypatch.setattr(query_pipeline_module, "invoke_text", lambda *a, **k: "Party A must indemnify Party B [1].")
    pipeline = RAGQueryPipeline(retriever=FakeRetriever(_sample_docs()))

    result = pipeline.answer("What is indemnity?")

    assert result["used_gemini"] is True
    assert result["answer"] == "Party A must indemnify Party B [1]."
    assert result["eval"] is None


def test_answer_with_eval_enabled_runs_feedback_loop(monkeypatch):
    monkeypatch.setattr(query_pipeline_module, "invoke_text", lambda *a, **k: "initial answer [1].")
    monkeypatch.setattr(
        query_pipeline_module,
        "evaluate_with_feedback_loop",
        lambda query, context, response, regenerate: {
            "response": response,
            "score": {"accuracy": 5, "completeness": 5, "relevance": 5, "average": 5.0, "feedback": ""},
            "history": [],
            "evaluated": True,
        },
    )
    pipeline = RAGQueryPipeline(retriever=FakeRetriever(_sample_docs()))

    result = pipeline.answer("What is indemnity?", enable_eval=True)

    assert result["eval"]["average"] == 5.0
