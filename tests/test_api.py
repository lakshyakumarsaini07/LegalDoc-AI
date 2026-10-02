from fastapi.testclient import TestClient

from src.api import main as api_main


def test_health_reports_llm_and_vector_store_status(monkeypatch):
    monkeypatch.setattr(api_main, "vector_store_ready", lambda: False)
    monkeypatch.setattr(api_main, "has_llm_configured", lambda: False)

    client = TestClient(api_main.app)
    response = client.get("/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["llm_configured"] is False
    assert body["vector_store_ready"] is False


def test_chat_returns_503_when_vector_store_missing(monkeypatch):
    monkeypatch.setattr(api_main, "vector_store_ready", lambda: False)

    client = TestClient(api_main.app)
    response = client.post("/chat", json={"query": "hello"})

    assert response.status_code == 503


def test_chat_returns_pipeline_answer_when_vector_store_present(monkeypatch):
    monkeypatch.setattr(api_main, "vector_store_ready", lambda: True)

    class FakePipeline:
        def answer(self, query, k=None, enable_eval=False):
            return {"answer": "fake answer", "sources": [], "used_gemini": False, "eval": None}

    monkeypatch.setattr(api_main, "get_query_pipeline", lambda: FakePipeline())

    client = TestClient(api_main.app)
    response = client.post("/chat", json={"query": "hello"})

    assert response.status_code == 200
    assert response.json()["answer"] == "fake answer"


def test_summarize_endpoint(monkeypatch):
    from src.summarization import summarize as summarize_module

    monkeypatch.setattr(summarize_module, "invoke_text", lambda *a, **k: None)

    client = TestClient(api_main.app)
    response = client.post("/summarize", json={"text": "Some legal text."})

    assert response.status_code == 200
    assert response.json()["used_gemini"] is False


def test_generate_endpoint(monkeypatch):
    from src.generation import doc_generator as doc_generator_module

    monkeypatch.setattr(doc_generator_module, "invoke_text", lambda *a, **k: "Drafted document.")

    client = TestClient(api_main.app)
    response = client.post("/generate", json={"document_type": "NDA", "requirements": "test"})

    assert response.status_code == 200
    assert response.json()["document"] == "Drafted document."


def test_chat_rejects_empty_query():
    client = TestClient(api_main.app)
    response = client.post("/chat", json={"query": ""})
    assert response.status_code == 422


def test_stats_endpoint_reports_corpus_size(monkeypatch):
    from langchain_core.documents import Document

    monkeypatch.setattr(api_main, "vector_store_ready", lambda: True)

    class FakeRetriever:
        docs = [
            Document(page_content="a", metadata={"doc_id": "doc1"}),
            Document(page_content="b", metadata={"doc_id": "doc1"}),
            Document(page_content="c", metadata={"doc_id": "doc2"}),
        ]

    class FakePipeline:
        retriever = FakeRetriever()

    monkeypatch.setattr(api_main, "get_query_pipeline", lambda: FakePipeline())

    client = TestClient(api_main.app)
    response = client.get("/stats")

    assert response.status_code == 200
    assert response.json() == {"indexed_chunks": 3, "indexed_documents": 2}


def test_stats_returns_503_when_vector_store_missing(monkeypatch):
    monkeypatch.setattr(api_main, "vector_store_ready", lambda: False)

    client = TestClient(api_main.app)
    response = client.get("/stats")

    assert response.status_code == 503


def test_protected_endpoints_require_api_key_when_configured(monkeypatch):
    monkeypatch.setattr(api_main, "API_KEY", "secret123")

    client = TestClient(api_main.app)

    no_key_response = client.post("/summarize", json={"text": "hello"})
    assert no_key_response.status_code == 401

    wrong_key_response = client.post(
        "/summarize", json={"text": "hello"}, headers={"X-API-Key": "wrong"}
    )
    assert wrong_key_response.status_code == 401

    from src.summarization import summarize as summarize_module

    monkeypatch.setattr(summarize_module, "invoke_text", lambda *a, **k: None)

    correct_key_response = client.post(
        "/summarize", json={"text": "hello"}, headers={"X-API-Key": "secret123"}
    )
    assert correct_key_response.status_code == 200


def test_health_does_not_require_api_key(monkeypatch):
    monkeypatch.setattr(api_main, "API_KEY", "secret123")

    client = TestClient(api_main.app)
    response = client.get("/health")

    assert response.status_code == 200
