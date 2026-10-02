from src.generation import doc_generator as doc_generator_module
from src.generation.doc_generator import DocumentGenerator


def test_generate_falls_back_without_llm(monkeypatch):
    monkeypatch.setattr(doc_generator_module, "invoke_text", lambda *a, **k: None)
    result = DocumentGenerator().generate("NDA", "Between two startups")
    assert result["used_gemini"] is False
    assert "unavailable" in result["document"].lower()


def test_generate_uses_llm_when_available(monkeypatch):
    monkeypatch.setattr(doc_generator_module, "invoke_text", lambda *a, **k: "1. Confidentiality...")
    result = DocumentGenerator().generate("NDA", "Between two startups")
    assert result["used_gemini"] is True
    assert result["document"] == "1. Confidentiality..."


def test_generate_includes_precedent_clauses_in_prompt(monkeypatch):
    captured_prompts = []
    monkeypatch.setattr(
        doc_generator_module, "invoke_text", lambda prompt, **k: captured_prompts.append(prompt) or "draft"
    )

    DocumentGenerator().generate("NDA", "Between two startups", precedent_clauses=["Indemnity clause example"])

    assert "Indemnity clause example" in captured_prompts[0]


def test_refine_returns_original_document_without_llm(monkeypatch):
    monkeypatch.setattr(doc_generator_module, "invoke_text", lambda *a, **k: None)
    result = DocumentGenerator().refine("original document text", "add a clause")
    assert result["used_gemini"] is False
    assert result["document"] == "original document text"
