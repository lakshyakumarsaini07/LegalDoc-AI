"""Legal document summarization.

Splits long documents into chunks, summarizes each chunk (map), then
combines those partial summaries into one structured summary (reduce)
covering key obligations, risks, important clauses, and deadlines.
"""
from __future__ import annotations

from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.llm import invoke_text
from src.utils.logger import logger

MAP_PROMPT = """Extract the key legal points from the following excerpt of a longer document. \
List only: obligations, risks, important clauses, and deadlines/dates you find. Be concise and \
use bullet points. If a category has nothing relevant, omit it.

Excerpt:
{text}

Key points:"""

REDUCE_PROMPT = """You are a legal assistant. Combine the following partial notes taken from \
different sections of the same legal document into ONE structured summary with exactly these \
sections:

## Key Obligations
## Risks
## Important Clauses
## Deadlines

Deduplicate overlapping points across notes. If a section has no relevant content, write \
"None identified."

Partial notes:
{notes}

Structured summary:"""

CHUNK_SIZE = 6000
CHUNK_OVERLAP = 400


def _split(text: str) -> list[str]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ".", " "],
    )
    return splitter.split_text(text)


class SummarizationService:
    """Map-reduce summarization over an arbitrary (possibly long) document."""

    def summarize(self, text: str) -> dict:
        if not text or not text.strip():
            return {"summary": "", "used_gemini": False, "chunks": 0}

        chunks = _split(text)
        logger.info(f"Summarizing document split into {len(chunks)} chunk(s)")

        if len(chunks) == 1:
            notes = [chunks[0]]
        else:
            notes = []
            for chunk in chunks:
                partial = invoke_text(MAP_PROMPT.format(text=chunk))
                notes.append(partial if partial is not None else chunk[:1000])

        combined_notes = "\n\n---\n\n".join(notes)
        summary = invoke_text(REDUCE_PROMPT.format(notes=combined_notes))
        used_gemini = summary is not None

        if summary is None:
            summary = (
                "LLM summarization is unavailable (no GOOGLE_API_KEY configured). "
                "Raw extracted notes from each section:\n\n" + combined_notes
            )

        return {"summary": summary, "used_gemini": used_gemini, "chunks": len(chunks)}


def summarize_document(text: str) -> dict:
    return SummarizationService().summarize(text)


def main() -> None:
    sample = (
        "This Non-Disclosure Agreement is entered into between Party A and Party B. "
        "Party B shall not disclose confidential information for a period of 3 years. "
        "Either party may terminate this agreement with 30 days written notice. "
        "In case of breach, the breaching party shall indemnify the other party for damages."
    )
    result = summarize_document(sample)
    print(result["summary"])


if __name__ == "__main__":
    main()
