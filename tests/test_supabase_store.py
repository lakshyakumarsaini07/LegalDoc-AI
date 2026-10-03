import pandas as pd
import pytest

from src.embeddings import supabase_store as supabase_store_module
from src.embeddings.supabase_store import SupabaseVectorStore, _retry_delay_seconds


class FakeExecuteResult:
    def __init__(self, data):
        self.data = data

    def execute(self):
        return self


class FakeTableQuery:
    def __init__(self, table):
        self.table = table
        self._select = None
        self._range = None

    def upsert(self, rows, on_conflict=None):
        assert on_conflict == "chunk_id"
        self.table.upserted.extend(rows)
        return self

    def select(self, _fields):
        self._select = _fields
        return self

    def range(self, start, end):
        self._range = (start, end)
        return self

    def execute(self):
        if self._range is not None:
            start, end = self._range
            return FakeExecuteResult(self.table.rows[start : end + 1])
        return FakeExecuteResult([])


class FakeTable:
    def __init__(self, rows=None):
        self.rows = rows or []
        self.upserted = []

    def __call__(self):
        return FakeTableQuery(self)


class FakeClient:
    def __init__(self, table_rows=None, rpc_response=None):
        self._table = FakeTable(table_rows or [])
        self._rpc_response = rpc_response or []
        self.rpc_calls = []

    def table(self, _name):
        return self._table()

    def rpc(self, function_name, params):
        self.rpc_calls.append((function_name, params))
        return FakeExecuteResult(self._rpc_response)


class FakeEmbeddings:
    def embed_documents(self, texts):
        return [[0.1, 0.2, 0.3] for _ in texts]

    def embed_query(self, text):
        return [0.1, 0.2, 0.3]


def _make_store(monkeypatch, client):
    monkeypatch.setattr(supabase_store_module, "_get_client", lambda: client)
    monkeypatch.setattr(supabase_store_module, "get_gemini_embeddings", lambda: FakeEmbeddings())
    return SupabaseVectorStore()


def test_build_upserts_rows_with_chunk_id_conflict_target(monkeypatch, tmp_path):
    chunk_path = tmp_path / "chunks.parquet"
    pd.DataFrame(
        [
            {
                "chunk_id": "doc1_0",
                "doc_id": "doc1",
                "title": "Sample Case",
                "court": "High Court",
                "judge": "Justice X",
                "decision_year": 2022,
                "contextual_text": "Case Title: Sample Case\nContent: indemnity text",
                "chunk_text": "indemnity text",
            }
        ]
    ).to_parquet(chunk_path)

    client = FakeClient()
    store = _make_store(monkeypatch, client)

    store.build(chunk_source=chunk_path)

    assert len(client._table.upserted) == 1
    row = client._table.upserted[0]
    assert row["chunk_id"] == "doc1_0"
    assert row["doc_id"] == "doc1"
    assert row["decision_year"] == 2022
    assert isinstance(row["decision_year"], int)
    assert row["content"] == "Case Title: Sample Case\nContent: indemnity text"
    assert row["embedding"] == [0.1, 0.2, 0.3]


def test_build_skips_chunks_already_present_in_supabase(monkeypatch, tmp_path):
    chunk_path = tmp_path / "chunks.parquet"
    pd.DataFrame(
        [
            {"chunk_id": "doc1_0", "doc_id": "doc1", "chunk_text": "already ingested"},
            {"chunk_id": "doc1_1", "doc_id": "doc1", "chunk_text": "still needs embedding"},
        ]
    ).to_parquet(chunk_path)

    client = FakeClient(table_rows=[{"chunk_id": "doc1_0"}])
    store = _make_store(monkeypatch, client)

    store.build(chunk_source=chunk_path)

    assert len(client._table.upserted) == 1
    assert client._table.upserted[0]["chunk_id"] == "doc1_1"


def test_build_force_reprocesses_everything(monkeypatch, tmp_path):
    chunk_path = tmp_path / "chunks.parquet"
    pd.DataFrame(
        [{"chunk_id": "doc1_0", "doc_id": "doc1", "chunk_text": "already ingested"}]
    ).to_parquet(chunk_path)

    client = FakeClient(table_rows=[{"chunk_id": "doc1_0"}])
    store = _make_store(monkeypatch, client)

    store.build(chunk_source=chunk_path, force=True)

    assert len(client._table.upserted) == 1


def test_build_handles_missing_optional_metadata(monkeypatch, tmp_path):
    chunk_path = tmp_path / "chunks.parquet"
    pd.DataFrame(
        [{"chunk_id": "doc1_0", "doc_id": "doc1", "chunk_text": "text only, no contextual_text"}]
    ).to_parquet(chunk_path)

    client = FakeClient()
    store = _make_store(monkeypatch, client)

    store.build(chunk_source=chunk_path)

    row = client._table.upserted[0]
    assert row["title"] is None
    assert row["decision_year"] is None
    assert row["content"] == "text only, no contextual_text"


def test_similarity_search_maps_rpc_rows_to_documents(monkeypatch):
    client = FakeClient(
        rpc_response=[
            {
                "chunk_id": "doc1_0",
                "doc_id": "doc1",
                "title": "Sample Case",
                "court": "High Court",
                "judge": None,
                "decision_year": 2022,
                "content": "contextual text",
                "chunk_text": "raw chunk text",
                "similarity": 0.9,
            }
        ]
    )
    store = _make_store(monkeypatch, client)

    results = store.similarity_search("indemnity", k=3)

    assert client.rpc_calls == [("match_legal_chunks", {"query_embedding": [0.1, 0.2, 0.3], "match_count": 3})]
    assert len(results) == 1
    doc = results[0]
    assert doc.page_content == "contextual text"
    assert doc.metadata["chunk_text"] == "raw chunk text"
    assert doc.metadata["doc_id"] == "doc1"
    assert "judge" not in doc.metadata  # None fields are omitted


def test_get_all_documents_paginates_until_short_page(monkeypatch):
    rows = [
        {
            "chunk_id": f"doc1_{i}",
            "doc_id": "doc1",
            "title": None,
            "court": None,
            "judge": None,
            "decision_year": None,
            "content": f"content {i}",
            "chunk_text": f"chunk {i}",
        }
        for i in range(3)
    ]
    client = FakeClient(table_rows=rows)
    store = _make_store(monkeypatch, client)

    docs = store.get_all_documents(page_size=2)

    assert len(docs) == 3
    assert [d.metadata["chunk_id"] for d in docs] == ["doc1_0", "doc1_1", "doc1_2"]


def test_get_client_raises_when_not_configured(monkeypatch):
    monkeypatch.setattr(supabase_store_module, "has_supabase_configured", lambda: False)
    with pytest.raises(RuntimeError, match="SUPABASE_URL"):
        supabase_store_module._get_client()


def test_retry_delay_parses_suggested_wait():
    exc = Exception("RESOURCE_EXHAUSTED ... Please retry in 27.19s")
    assert _retry_delay_seconds(exc) == pytest.approx(30.19)


def test_retry_delay_falls_back_to_default_when_unparseable():
    assert _retry_delay_seconds(Exception("some other error"), default=65.0) == 65.0


def test_embed_with_retry_recovers_from_rate_limit(monkeypatch):
    client = FakeClient()
    store = _make_store(monkeypatch, client)

    monkeypatch.setattr(supabase_store_module.time, "sleep", lambda _seconds: None)

    calls = {"count": 0}

    class FlakyEmbeddings(FakeEmbeddings):
        def embed_documents(self, texts):
            calls["count"] += 1
            if calls["count"] < 3:
                raise Exception("429 RESOURCE_EXHAUSTED ... Please retry in 1s")
            return super().embed_documents(texts)

    store.embeddings = FlakyEmbeddings()

    result = store._embed_with_retry(["a", "b"])

    assert calls["count"] == 3
    assert result == [[0.1, 0.2, 0.3], [0.1, 0.2, 0.3]]


def test_embed_with_retry_raises_after_exhausting_attempts(monkeypatch):
    client = FakeClient()
    store = _make_store(monkeypatch, client)
    monkeypatch.setattr(supabase_store_module.time, "sleep", lambda _seconds: None)

    class AlwaysFailsEmbeddings(FakeEmbeddings):
        def embed_documents(self, texts):
            raise Exception("429 RESOURCE_EXHAUSTED ... Please retry in 1s")

    store.embeddings = AlwaysFailsEmbeddings()

    with pytest.raises(Exception, match="RESOURCE_EXHAUSTED"):
        store._embed_with_retry(["a"], max_retries=2)
