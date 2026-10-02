from src.summarization import summarize as summarize_module
from src.summarization.summarize import summarize_document


def test_summarize_empty_text_returns_empty():
    result = summarize_document("")
    assert result == {"summary": "", "used_gemini": False, "chunks": 0}


def test_summarize_falls_back_without_llm(monkeypatch):
    monkeypatch.setattr(summarize_module, "invoke_text", lambda *a, **k: None)
    result = summarize_document("Some short legal text about an NDA.")
    assert result["used_gemini"] is False
    assert "unavailable" in result["summary"].lower()


def test_summarize_uses_llm_when_available(monkeypatch):
    monkeypatch.setattr(summarize_module, "invoke_text", lambda *a, **k: "## Key Obligations\n- none")
    result = summarize_document("Some short legal text about an NDA.")
    assert result["used_gemini"] is True
    assert "Key Obligations" in result["summary"]


def test_summarize_splits_long_documents(monkeypatch):
    calls = []

    def fake_invoke(prompt, **kwargs):
        calls.append(prompt)
        return "partial or final"

    monkeypatch.setattr(summarize_module, "invoke_text", fake_invoke)
    long_text = "This is a sentence about legal obligations. " * 2000

    result = summarize_document(long_text)

    assert result["chunks"] > 1
    # one map call per chunk, plus one reduce call
    assert len(calls) == result["chunks"] + 1
