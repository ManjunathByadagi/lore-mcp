"""Unit tests for lore.embeddings module.

Covers the helpers and public API that don't require a live model:
_model_name, _model_backend, _cache_dir, compute_content_hash, reset_for_tests,
get_model_name, and the EmbeddingUnavailableError raise path of get_embedder
(when LORE_SEMANTIC_SEARCH is disabled, no import of sentence-transformers occurs).
"""

from __future__ import annotations

import pytest

import lore.embeddings as emb

# ---------------------------------------------------------------------------
# _model_name
# ---------------------------------------------------------------------------


def test_model_name_default():
    """Default model name is all-MiniLM-L6-v2."""
    name = emb._model_name()
    assert name == emb.DEFAULT_MODEL


def test_model_name_env_override_with_slash(monkeypatch):
    monkeypatch.setenv("LORE_EMBEDDING_MODEL", "org/my-model")
    assert emb._model_name() == "org/my-model"


def test_model_name_short_name_gets_prefix(monkeypatch):
    """Short names without '/' are prefixed with 'sentence-transformers/'."""
    monkeypatch.setenv("LORE_EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    assert emb._model_name() == "sentence-transformers/all-MiniLM-L6-v2"


def test_model_name_empty_env_falls_back_to_default(monkeypatch):
    monkeypatch.setenv("LORE_EMBEDDING_MODEL", "")
    assert emb._model_name() == emb.DEFAULT_MODEL


def test_model_name_whitespace_only_falls_back(monkeypatch):
    monkeypatch.setenv("LORE_EMBEDDING_MODEL", "   ")
    assert emb._model_name() == emb.DEFAULT_MODEL


# ---------------------------------------------------------------------------
# get_model_name (public wrapper)
# ---------------------------------------------------------------------------


def test_get_model_name_returns_string():
    name = emb.get_model_name()
    assert isinstance(name, str)
    assert len(name) > 0


def test_get_model_name_matches_model_name():
    assert emb.get_model_name() == emb._model_name()


# ---------------------------------------------------------------------------
# _model_backend
# ---------------------------------------------------------------------------


def test_model_backend_default_is_onnx():
    assert emb._model_backend() == "onnx"


def test_model_backend_env_override(monkeypatch):
    monkeypatch.setenv("LORE_EMBEDDING_BACKEND", "torch")
    assert emb._model_backend() == "torch"


def test_model_backend_empty_env_falls_back_to_onnx(monkeypatch):
    monkeypatch.setenv("LORE_EMBEDDING_BACKEND", "")
    assert emb._model_backend() == "onnx"


# ---------------------------------------------------------------------------
# _cache_dir
# ---------------------------------------------------------------------------


def test_cache_dir_default_is_none():
    assert emb._cache_dir() is None


def test_cache_dir_env_override(monkeypatch):
    monkeypatch.setenv("LORE_EMBEDDING_CACHE_DIR", "/tmp/hf_cache")
    assert emb._cache_dir() == "/tmp/hf_cache"


def test_cache_dir_empty_env_is_none(monkeypatch):
    monkeypatch.setenv("LORE_EMBEDDING_CACHE_DIR", "")
    assert emb._cache_dir() is None


def test_cache_dir_whitespace_env_is_none(monkeypatch):
    monkeypatch.setenv("LORE_EMBEDDING_CACHE_DIR", "   ")
    assert emb._cache_dir() is None


# ---------------------------------------------------------------------------
# compute_content_hash
# ---------------------------------------------------------------------------


def test_compute_content_hash_returns_hex_string():
    h = emb.compute_content_hash("hello", "world")
    assert isinstance(h, str)
    assert all(c in "0123456789abcdef" for c in h)
    assert len(h) == 64  # SHA-256 hex digest


def test_compute_content_hash_deterministic():
    h1 = emb.compute_content_hash("title", "content")
    h2 = emb.compute_content_hash("title", "content")
    assert h1 == h2


def test_compute_content_hash_different_inputs_differ():
    h1 = emb.compute_content_hash("title A", "content A")
    h2 = emb.compute_content_hash("title B", "content B")
    assert h1 != h2


def test_compute_content_hash_order_matters():
    h1 = emb.compute_content_hash("a", "b")
    h2 = emb.compute_content_hash("b", "a")
    assert h1 != h2


def test_compute_content_hash_none_parts_are_skipped():
    """None parts don't crash — they're skipped in the hash computation."""
    h = emb.compute_content_hash("hello", None, "world")
    assert isinstance(h, str)
    assert len(h) == 64


def test_compute_content_hash_all_none():
    h = emb.compute_content_hash(None, None)
    assert isinstance(h, str)


def test_compute_content_hash_empty_string():
    h = emb.compute_content_hash("")
    assert isinstance(h, str)


# ---------------------------------------------------------------------------
# reset_for_tests
# ---------------------------------------------------------------------------


def test_reset_for_tests_clears_state():
    """reset_for_tests() must zero out the cached embedder state."""
    # First ensure the module-level cache is cleared
    emb.reset_for_tests()
    assert emb._embedder is None
    assert emb._embedder_model_name is None


def test_reset_for_tests_is_idempotent():
    emb.reset_for_tests()
    emb.reset_for_tests()  # should not raise
    assert emb._embedder is None


# ---------------------------------------------------------------------------
# get_embedder: raises EmbeddingUnavailableError when semantic is disabled
# ---------------------------------------------------------------------------


def test_get_embedder_raises_when_semantic_disabled(monkeypatch):
    """get_embedder() must raise EmbeddingUnavailableError when LORE_SEMANTIC_SEARCH
    is not 'true' — this is the normal, no-heavy-deps code path.
    """
    monkeypatch.setenv("LORE_SEMANTIC_SEARCH", "false")
    emb.reset_for_tests()
    with pytest.raises(emb.EmbeddingUnavailableError, match="not enabled"):
        emb.get_embedder()


def test_get_embedder_raises_when_semantic_unset(monkeypatch):
    monkeypatch.delenv("LORE_SEMANTIC_SEARCH", raising=False)
    emb.reset_for_tests()
    with pytest.raises(emb.EmbeddingUnavailableError):
        emb.get_embedder()


def test_get_embedder_raises_when_semantic_off_case_variants(monkeypatch):
    for value in ("FALSE", "0", "no", ""):
        monkeypatch.setenv("LORE_SEMANTIC_SEARCH", value)
        emb.reset_for_tests()
        with pytest.raises(emb.EmbeddingUnavailableError):
            emb.get_embedder()


# ---------------------------------------------------------------------------
# encode_text / encode_batch: raises when semantic disabled (no model loaded)
# ---------------------------------------------------------------------------


def test_encode_batch_empty_list_returns_empty(monkeypatch):
    """encode_batch([]) must return [] without touching the embedder."""
    # This doesn't call get_embedder at all for empty input.
    result = emb.encode_batch([])
    assert result == []


def test_encode_text_raises_when_semantic_disabled(monkeypatch):
    """encode_text propagates EmbeddingUnavailableError from get_embedder."""
    monkeypatch.setenv("LORE_SEMANTIC_SEARCH", "false")
    emb.reset_for_tests()
    with pytest.raises(emb.EmbeddingUnavailableError):
        emb.encode_text("hello world")


# ---------------------------------------------------------------------------
# EMBEDDING_DIM constant
# ---------------------------------------------------------------------------


def test_embedding_dim_is_384():
    assert emb.EMBEDDING_DIM == 384


def test_default_model_is_mini_lm():
    assert "MiniLM" in emb.DEFAULT_MODEL
