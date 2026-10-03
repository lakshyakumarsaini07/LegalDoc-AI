from src.embeddings.gemini_embeddings import to_native


def test_to_native_converts_numpy_scalar():
    import numpy as np

    assert to_native(np.int64(5)) == 5
    assert isinstance(to_native(np.int64(5)), int)


def test_to_native_passes_through_plain_values():
    assert to_native("hello") == "hello"
    assert to_native(None) is None
    assert to_native(42) == 42
