"""Local embedding model loader and helpers for semantic search.

Gated by ``LORE_SEMANTIC_SEARCH`` — this module imports heavy dependencies
(``sentence-transformers``, ``onnxruntime``) lazily inside ``get_embedder()``
so the rest of the server can keep starting without them.

PostgreSQL semantic path: deferred to Phase 2. This module focuses on the
producer side (encode text -> list[float]); SQLite ``vec0`` consumes those
floats in ``lore.search``.
"""

from __future__ import annotations

import hashlib
import logging
import os
import threading
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - typing only
    from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


# Embedding dimension is fixed by the model family. all-MiniLM-L6-v2 and the
# multilingual paraphrase-multilingual-MiniLM-L12-v2 both emit 384-d vectors.
EMBEDDING_DIM = 384

# Default model. Override with LORE_EMBEDDING_MODEL.
DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Module-private singleton state.
_embedder: Any | None = None
_embedder_model_name: str | None = None
_embed_lock = threading.Lock()


def _semantic_enabled() -> bool:
    """Whether semantic search is enabled via env var."""
    return os.getenv("LORE_SEMANTIC_SEARCH", "false").strip().lower() == "true"


def _model_name() -> str:
    """Resolve which embedding model to load."""
    raw = os.getenv("LORE_EMBEDDING_MODEL", DEFAULT_MODEL).strip()
    if not raw:
        return DEFAULT_MODEL
    # Allow short names like "all-MiniLM-L6-v2" by prefixing the HF org.
    if "/" not in raw:
        return f"sentence-transformers/{raw}"
    return raw


def get_model_name() -> str:
    """Public accessor for the resolved embedding model name."""
    return _model_name()


def _model_backend() -> str:
    """Resolve sentence-transformers backend: 'onnx' (default) or 'torch'."""
    return os.getenv("LORE_EMBEDDING_BACKEND", "onnx").strip().lower() or "onnx"


def _cache_dir() -> str | None:
    """Hugging Face cache directory override."""
    raw = os.getenv("LORE_EMBEDDING_CACHE_DIR", "").strip()
    return raw or None


class EmbeddingUnavailableError(RuntimeError):
    """Raised when semantic search is requested but the model cannot be loaded."""


def get_embedder() -> Any:
    """Return the cached SentenceTransformer instance, loading it on first call.

    Thread-safe: protected by ``_embed_lock`` so concurrent first calls don't
    double-load the model. Subsequent calls are fast (no lock contention).

    Raises:
        EmbeddingUnavailableError: if semantic search is disabled or the
            ``sentence-transformers`` package is missing.
    """
    global _embedder, _embedder_model_name

    if not _semantic_enabled():
        raise EmbeddingUnavailableError(
            "LORE_SEMANTIC_SEARCH is not enabled; semantic search is disabled."
        )

    if _embedder is not None and _embedder_model_name == _model_name():
        return _embedder

    with _embed_lock:
        # Double-checked locking: another thread may have loaded it while we waited.
        if _embedder is not None and _embedder_model_name == _model_name():
            return _embedder

        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover - exercised via degradation test
            raise EmbeddingUnavailableError(
                "sentence-transformers is not installed. Install the [semantic] "
                "extra: pip install -e '.[semantic]'"
            ) from exc

        model_name = _model_name()
        backend = _model_backend()
        cache_dir = _cache_dir()

        logger.info(
            "Loading embedding model %s (backend=%s, cache_dir=%s)",
            model_name,
            backend,
            cache_dir or "<default>",
        )

        # sentence-transformers>=2.7 supports backend="onnx" without pulling
        # full PyTorch at runtime. Fall back to torch if onnx fails to load.
        try:
            kwargs: dict[str, Any] = {}
            if backend == "onnx":
                kwargs["backend"] = "onnx"
            if cache_dir:
                kwargs["cache_folder"] = cache_dir
            model = SentenceTransformer(model_name, **kwargs)
        except (TypeError, ValueError) as exc:
            # Older sentence-transformers without backend= kwarg; retry without it.
            logger.warning(
                "Failed to load with backend=%s (%s); retrying with default backend",
                backend,
                exc,
            )
            kwargs = {}
            if cache_dir:
                kwargs["cache_folder"] = cache_dir
            model = SentenceTransformer(model_name, **kwargs)

        _embedder = model
        _embedder_model_name = model_name
        return model


def encode_text(text: str) -> list[float]:
    """Encode a single string into a 384-d embedding."""
    return encode_batch([text])[0]


def encode_batch(texts: list[str]) -> list[list[float]]:
    """Encode a batch of strings. Returns a list of float vectors."""
    if not texts:
        return []
    model = get_embedder()
    # convert_to_numpy=True returns float32 ndarray; tolist gives plain Python floats.
    vectors = model.encode(
        list(texts),
        batch_size=int(os.getenv("LORE_EMBEDDING_BATCH_SIZE", "32") or "32"),
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    return [v.tolist() for v in vectors]


def compute_content_hash(*parts: str | None) -> str:
    """SHA-256 of joined parts. Used to detect whether re-embedding is needed."""
    h = hashlib.sha256()
    for part in parts:
        if part is None:
            continue
        h.update(part.encode("utf-8"))
        h.update(b"\x1f")  # unit separator
    return h.hexdigest()


def reset_for_tests() -> None:
    """Reset the cached embedder. Tests only — do not call from production code."""
    global _embedder, _embedder_model_name
    with _embed_lock:
        _embedder = None
        _embedder_model_name = None
