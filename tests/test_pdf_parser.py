import pandas as pd

from src.preprocessing.pdf_parser import PDFParser


def _make_pdf_parser(tmp_path, existing_doc_ids=None):
    pdf_dir = tmp_path / "pdfs"
    pdf_dir.mkdir()
    for doc_id in ["doc1", "doc2", "doc3"]:
        (pdf_dir / f"{doc_id}.pdf").write_bytes(b"%PDF-1.4 fake content")

    output_dir = tmp_path / "pdf_text_data"
    if existing_doc_ids:
        output_dir.mkdir()
        pd.DataFrame({"doc_id": list(existing_doc_ids)}).to_parquet(output_dir / "existing.parquet")

    return PDFParser(pdf_dir=str(pdf_dir), output_dir=str(output_dir), batch_size=10)


def test_get_existing_doc_ids_empty_when_no_output(tmp_path):
    parser = _make_pdf_parser(tmp_path)
    assert parser.get_existing_doc_ids() == set()


def test_get_existing_doc_ids_reads_prior_run(tmp_path):
    parser = _make_pdf_parser(tmp_path, existing_doc_ids={"doc1", "doc2"})
    assert parser.get_existing_doc_ids() == {"doc1", "doc2"}


def test_run_skips_already_processed_docs(tmp_path, monkeypatch):
    parser = _make_pdf_parser(tmp_path, existing_doc_ids={"doc1"})
    monkeypatch.setattr(parser, "extract_text", lambda path: ("some text", 1, "success"))

    parser.run(force=False)

    assert parser.skipped == 1
    assert parser.processed == 2  # doc2 and doc3 only


def test_run_force_reprocesses_everything(tmp_path, monkeypatch):
    parser = _make_pdf_parser(tmp_path, existing_doc_ids={"doc1"})
    monkeypatch.setattr(parser, "extract_text", lambda path: ("some text", 1, "success"))

    parser.run(force=True)

    assert parser.skipped == 0
    assert parser.processed == 3
