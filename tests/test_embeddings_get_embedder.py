"""Tests for lore.embeddings: get_embedder with a mocked SentenceTransformer.

Round 2 already covered the "semantic disabled" path. This file covers:
- get_embedder: returns cached embedder when model is same (fast path)
- get_embedder: reloads embedder when model name changes
- get_embedder: backend=onnx path (kwargs injected)
- get_embedder: TypeError/ValueError fallback (retry without backend kwarg)
- encode_batch: empty list returns []
- encode_batch: delegates to get_embedder().encode(...)
- encode_text: delegates to encode_batch

All tests mock SentenceTransformer — no heavy ML deps required.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

import lore.embeddings as emb


@pytest.fixture(autouse=True)
def _reset_embedder():
    """Reset the cached embedder before and after each test."""
    emb.reset_for_tests()
    yield
    emb.reset_for_tests()


# ---------------------------------------------------------------------------
# encode_batch: empty input
# ---------------------------------------------------------------------------


def test_encode_batch_empty_list_returns_empty():
    """encode_batch([]) must short-circuit without touching the model."""
    result = emb.encode_batch([])
    assert result == []


# ---------------------------------------------------------------------------
# get_embedder: happy-path (mock SentenceTransformer)
# ---------------------------------------------------------------------------


def _make_row(values: list) -> MagicMock:
    """Return a mock with .tolist() returning *values* (avoids numpy dependency)."""
    row = MagicMock()
    row.tolist.return_value = values
    return row


def _make_fake_model(n_rows: int = 2, dim: int = 384):
    """Return a MagicMock that quacks like a SentenceTransformer.

    encode() returns a list of row-mocks each with a .tolist() method so that
    encode_batch()'s ``[v.tolist() for v in vectors]`` works without numpy.
    """
    model = MagicMock()
    rows = [_make_row([float(i) / dim] * dim) for i in range(n_rows)]
    model.encode.return_value = rows
    return model


def test_get_embedder_loads_model_when_semantic_enabled(monkeypatch):
    """get_embedder() returns a model when LORE_SEMANTIC_SEARCH=true."""
    monkeypatch.setenv("LORE_SEMANTIC_SEARCH", "true")
    fake_model = _make_fake_model()

    with patch("sentence_transformers.SentenceTransformer", return_value=fake_model) as mock_st:
        result = emb.get_embedder()

    assert result is fake_model
    mock_st.assert_called_once()


def test_get_embedder_returns_cached_on_second_call(monkeypatch):
    """Second call to get_embedder() with the same model name returns cached instance."""
    monkeypatch.setenv("LORE_SEMANTIC_SEARCH", "true")
    monkeypatch.delenv("LORE_EMBEDDING_MODEL", raising=False)
    fake_model = _make_fake_model()

    with patch("sentence_transformers.SentenceTransformer", return_value=fake_model) as mock_st:
        first = emb.get_embedder()
        second = emb.get_embedder()

    # SentenceTransformer should only be constructed once
    assert mock_st.call_count == 1
    assert first is second


def test_get_embedder_reloads_when_model_name_changes(monkeypatch):
    """get_embedder() reloads when the env var model name changes."""
    monkeypatch.setenv("LORE_SEMANTIC_SEARCH", "true")
    monkeypatch.setenv("LORE_EMBEDDING_MODEL", "sentence-transformers/model-a")

    model_a = _make_fake_model()
    model_b = _make_fake_model()

    call_count = {"n": 0}

    def _side_effect(name, **kwargs):
        call_count["n"] += 1
        if call_count["n"] == 1:
            return model_a
        return model_b

    with patch("sentence_transformers.SentenceTransformer", side_effect=_side_effect):
        first = emb.get_embedder()
        assert first is model_a

        # Change model name — should cause a reload
        monkeypatch.setenv("LORE_EMBEDDING_MODEL", "sentence-transformers/model-b")
        emb._embedder_model_name = "sentence-transformers/model-a"  # force stale cache

        second = emb.get_embedder()

    assert second is model_b


def test_get_embedder_passes_onnx_backend_kwarg(monkeypatch):
    """When LORE_EMBEDDING_BACKEND=onnx, SentenceTransformer is called with backend='onnx'."""
    monkeypatch.setenv("LORE_SEMANTIC_SEARCH", "true")
    monkeypatch.setenv("LORE_EMBEDDING_BACKEND", "onnx")
    monkeypatch.delenv("LORE_EMBEDDING_MODEL", raising=False)
    fake_model = _make_fake_model()

    with patch("sentence_transformers.SentenceTransformer", return_value=fake_model) as mock_st:
        emb.get_embedder()

    _, kwargs = mock_st.call_args
    assert kwargs.get("backend") == "onnx"


def test_get_embedder_fallback_when_backend_kwarg_raises_typeerror(monkeypatch):
    """When SentenceTransformer raises TypeError on backend=, it retries without it."""
    monkeypatch.setenv("LORE_SEMANTIC_SEARCH", "true")
    monkeypatch.setenv("LORE_EMBEDDING_BACKEND", "onnx")
    monkeypatch.delenv("LORE_EMBEDDING_MODEL", raising=False)

    fake_model = _make_fake_model()
    call_count = {"n": 0}

    def _side_effect(name, **kwargs):
        call_count["n"] += 1
        if "backend" in kwargs:
            raise TypeError("unexpected keyword argument 'backend'")
        return fake_model

    with patch("sentence_transformers.SentenceTransformer", side_effect=_side_effect):
        result = emb.get_embedder()

    assert result is fake_model
    assert call_count["n"] == 2  # first call failed, second succeeded


def test_get_embedder_fallback_when_backend_kwarg_raises_valueerror(monkeypatch):
    """When SentenceTransformer raises ValueError on backend=, it retries without it."""
    monkeypatch.setenv("LORE_SEMANTIC_SEARCH", "true")
    monkeypatch.setenv("LORE_EMBEDDING_BACKEND", "onnx")
    monkeypatch.delenv("LORE_EMBEDDING_MODEL", raising=False)

    fake_model = _make_fake_model()

    def _side_effect(name, **kwargs):
        if "backend" in kwargs:
            raise ValueError("backend not supported in this version")
        return fake_model

    with patch("sentence_transformers.SentenceTransformer", side_effect=_side_effect):
        result = emb.get_embedder()

    assert result is fake_model


def test_get_embedder_with_cache_dir(monkeypatch):
    """When LORE_EMBEDDING_CACHE_DIR is set, cache_folder is passed to SentenceTransformer."""
    monkeypatch.setenv("LORE_SEMANTIC_SEARCH", "true")
    monkeypatch.setenv("LORE_EMBEDDING_CACHE_DIR", "/tmp/hf_cache")
    monkeypatch.setenv("LORE_EMBEDDING_BACKEND", "torch")  # avoid onnx path
    monkeypatch.delenv("LORE_EMBEDDING_MODEL", raising=False)

    fake_model = _make_fake_model()

    with patch("sentence_transformers.SentenceTransformer", return_value=fake_model) as mock_st:
        emb.get_embedder()

    _, kwargs = mock_st.call_args
    assert kwargs.get("cache_folder") == "/tmp/hf_cache"


# ---------------------------------------------------------------------------
# encode_batch: delegates to model.encode()
# ---------------------------------------------------------------------------


def test_encode_batch_returns_float_vectors(monkeypatch):
    """encode_batch returns a list of float lists matching the model output."""
    monkeypatch.setenv("LORE_SEMANTIC_SEARCH", "true")
    monkeypatch.delenv("LORE_EMBEDDING_MODEL", raising=False)

    fake_model = _make_fake_model(n_rows=1)
    # Override encode return to a single row whose .tolist() gives 384 floats
    fake_model.encode.return_value = [_make_row([0.1] * 384)]

    with patch("sentence_transformers.SentenceTransformer", return_value=fake_model):
        vectors = emb.encode_batch(["hello world"])

    assert isinstance(vectors, list)
    assert len(vectors) == 1
    assert len(vectors[0]) == 384
    assert all(isinstance(v, float) for v in vectors[0])


def test_encode_batch_passes_correct_texts(monkeypatch):
    """encode_batch passes the text list verbatim to model.encode()."""
    monkeypatch.setenv("LORE_SEMANTIC_SEARCH", "true")
    monkeypatch.delenv("LORE_EMBEDDING_MODEL", raising=False)

    fake_model = _make_fake_model(n_rows=3)
    texts = ["foo", "bar", "baz"]
    fake_model.encode.return_value = [_make_row([0.0] * 384) for _ in texts]

    with patch("sentence_transformers.SentenceTransformer", return_value=fake_model):
        emb.encode_batch(texts)

    call_args = fake_model.encode.call_args
    assert call_args[0][0] == texts


# ---------------------------------------------------------------------------
# encode_text: single-item wrapper
# ---------------------------------------------------------------------------


def test_encode_text_returns_single_vector(monkeypatch):
    """encode_text is a convenience wrapper that returns the first vector."""
    monkeypatch.setenv("LORE_SEMANTIC_SEARCH", "true")
    monkeypatch.delenv("LORE_EMBEDDING_MODEL", raising=False)

    fake_model = _make_fake_model(n_rows=1)
    fake_model.encode.return_value = [_make_row([0.5] * 384)]

    with patch("sentence_transformers.SentenceTransformer", return_value=fake_model):
        vec = emb.encode_text("hello")

    assert isinstance(vec, list)
    assert len(vec) == 384
