"""AI-assisted legal document generation with iterative refinement.

Optionally grounds drafts in precedent clauses retrieved from the
corpus vector store (e.g. "draft an indemnity clause similar to what's
used in these judgments").
"""
from __future__ import annotations

from src.llm import invoke_text
from src.utils.logger import logger

DRAFT_PROMPT = """You are a professional legal expert.

Draft a detailed and formal {document_type}.

Requirements:
{requirements}
{precedent_block}
Ensure:
- Proper legal structure with numbered clauses
- Clear, unambiguous language
- Professional tone
- Standard legal protections appropriate for this document type

Output the complete document only, with no preamble or commentary."""

REFINE_PROMPT = """You are revising a legal document you previously drafted.

Current document:
{document}

Requested change:
{instruction}

Apply the requested change and output the COMPLETE revised document (not just the changed \
section), preserving the existing structure and clauses that were not asked to change."""

_UNAVAILABLE_MSG = (
    "Document generation is unavailable (no GOOGLE_API_KEY configured). "
    "Set GOOGLE_API_KEY in .env to enable AI drafting."
)


def _format_precedents(precedent_clauses: list[str] | None) -> str:
    if not precedent_clauses:
        return ""
    joined = "\n\n".join(f"- {clause}" for clause in precedent_clauses)
    return f"\nReference precedent clauses from similar judgments (adapt, do not copy verbatim):\n{joined}\n"


class DocumentGenerator:
    def generate(
        self,
        document_type: str,
        requirements: str,
        precedent_clauses: list[str] | None = None,
    ) -> dict:
        prompt = DRAFT_PROMPT.format(
            document_type=document_type,
            requirements=requirements,
            precedent_block=_format_precedents(precedent_clauses),
        )
        draft = invoke_text(prompt, temperature=0.4)

        if draft is None:
            logger.info("Gemini unavailable — cannot generate document draft.")
            return {"document": _UNAVAILABLE_MSG, "used_gemini": False}

        return {"document": draft, "used_gemini": True}

    def refine(self, document: str, instruction: str) -> dict:
        prompt = REFINE_PROMPT.format(document=document, instruction=instruction)
        revised = invoke_text(prompt, temperature=0.4)

        if revised is None:
            return {"document": document, "used_gemini": False}

        return {"document": revised, "used_gemini": True}


def generate_document(
    document_type: str,
    requirements: str,
    precedent_clauses: list[str] | None = None,
) -> dict:
    return DocumentGenerator().generate(document_type, requirements, precedent_clauses)


def refine_document(document: str, instruction: str) -> dict:
    return DocumentGenerator().refine(document, instruction)


def main() -> None:
    result = generate_document(
        document_type="Non-Disclosure Agreement",
        requirements="Between a startup and a freelance developer, mutual confidentiality, 2 year term.",
    )
    print(result["document"])


if __name__ == "__main__":
    main()
