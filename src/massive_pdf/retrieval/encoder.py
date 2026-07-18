"""Embedding encoders (slice 5, issue #5).

Default:  `HashBagEncoder` — deterministic token-bag with fixed dim,
no network, no model download. Used by tests + offline runs.

Real model:  gated by the `MASSIVE_PDF_EMBEDDING_MODEL` env var. Any
non-empty value selects `BgeM3Encoder`, which is constructed lazily on
first call and delegates to `sentence-transformers` (raises a clear
`RuntimeError` if the package is not installed).

Embedding storage:  a numpy float32 vector is packed via `tobytes()` and
persisted into `rule_cards.embedding` (BLOB). Round-trip is via
`decode_embedding`. Vectors are L2-normalised so cosine similarity
reduces to a dot product (slice 6 will exploit this).
"""
from __future__ import annotations
import hashlib
import os
import re
from typing import Iterable, Protocol, runtime_checkable

import numpy as np


_TOKEN_RE = re.compile(r"[\w]+", re.UNICODE)


def tokenize(text: str) -> list[str]:
    """Lowercased alphanumeric tokens; Vietnamese diacritics pass through."""
    return _TOKEN_RE.findall(text.lower())


@runtime_checkable
class Encoder(Protocol):
    """Pluggable text encoder. Implementations must be deterministic per input."""

    @property
    def dim(self) -> int: ...

    def encode(self, text: str) -> np.ndarray: ...

    def encode_batch(self, texts: list[str]) -> np.ndarray: ...


def normalize(v: np.ndarray) -> np.ndarray:
    """L2-normalise a 1-D or 2-D vector (rows). Returns a copy."""
    if v.ndim == 1:
        n = float(np.linalg.norm(v))
        if n == 0.0:
            return v.copy()
        return v / n
    if v.ndim == 2:
        n = np.linalg.norm(v, axis=1, keepdims=True)
        n[n == 0.0] = 1.0
        return v / n
    raise ValueError(f"normalize supports 1-D or 2-D arrays, got shape {v.shape}")


class HashBagEncoder:
    """Deterministic hashed-token bag-of-words encoder.

    Each token hashes into a bucket in `[0, dim)`. Counts are L2-normalised.
    Same input -> same vector, no randomness, no model load. Useful as
    a smoke-test encoder and for offline CI.
    """

    DEFAULT_DIM = 128

    def __init__(self, dim: int = DEFAULT_DIM):
        if dim <= 0:
            raise ValueError(f"dim must be positive, got {dim}")
        self._dim = int(dim)

    @property
    def dim(self) -> int:
        return self._dim

    def _bucket(self, token: str) -> int:
        h = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
        return int.from_bytes(h, "big") % self._dim

    def encode(self, text: str) -> np.ndarray:
        v = np.zeros(self._dim, dtype=np.float32)
        for tok in tokenize(text):
            v[self._bucket(tok)] += 1.0
        return normalize(v)

    def encode_batch(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self._dim), dtype=np.float32)
        return np.stack([self.encode(t) for t in texts])


class BgeM3Encoder:
    """BGE-M3 encoder via `sentence-transformers`.

    Constructed lazily on first `encode()` so importing this module never
    touches the network. Activated when `MASSIVE_PDF_EMBEDDING_MODEL` is
    set in the environment.

    Raises a clear `RuntimeError` if `sentence-transformers` is missing
    so offline CI gets a readable message instead of a stack trace from
    the model loader.
    """

    MODEL_NAME = "BAAI/bge-m3"
    DEFAULT_DIM = 1024

    def __init__(self, model_name: str | None = None):
        self._model_name = model_name or self.MODEL_NAME
        self._model = None  # lazy
        self._dim: int | None = None

    @property
    def dim(self) -> int:
        if self._dim is None:
            # Trigger lazy load + dim probe.
            self._ensure_model()
        return self._dim  # type: ignore[return-value]

    def _ensure_model(self) -> None:
        if self._model is not None:
            return
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:
            raise RuntimeError(
                "BgeM3Encoder requires `sentence-transformers`; "
                "install it (`pip install sentence-transformers`) "
                "or unset MASSIVE_PDF_EMBEDDING_MODEL to fall back "
                "to HashBagEncoder."
            ) from e
        self._model = SentenceTransformer(self._model_name)
        d = self._model.get_sentence_embedding_dimension()
        self._dim = int(d) if d is not None else self.DEFAULT_DIM

    def encode(self, text: str) -> np.ndarray:
        self._ensure_model()
        v = self._model.encode(text, normalize_embeddings=True)  # type: ignore[union-attr]
        return np.asarray(v, dtype=np.float32)

    def encode_batch(self, texts: list[str]) -> np.ndarray:
        self._ensure_model()
        if not texts:
            return np.zeros((0, self._dim or self.DEFAULT_DIM), dtype=np.float32)
        v = self._model.encode(texts, normalize_embeddings=True)  # type: ignore[union-attr]
        return np.asarray(v, dtype=np.float32)


def get_default_encoder(dim: int = HashBagEncoder.DEFAULT_DIM) -> Encoder:
    """Resolve an encoder based on the `MASSIVE_PDF_EMBEDDING_MODEL` env var.

    When the env var is set (any non-empty string), returns a
    `BgeM3Encoder`; otherwise returns a `HashBagEncoder(dim=dim)`.
    """
    model = os.environ.get("MASSIVE_PDF_EMBEDDING_MODEL")
    if model:
        return BgeM3Encoder(model_name=model)
    return HashBagEncoder(dim=dim)


# ----- Serialisation helpers ----------------------------------------------

def encode_embedding(vector: np.ndarray) -> bytes:
    """Pack a 1-D vector into BLOB bytes (float32, native byte order)."""
    if vector.ndim != 1:
        raise ValueError(f"expected 1-D vector, got shape {vector.shape}")
    arr = np.asarray(vector, dtype=np.float32)
    return np.ascontiguousarray(arr).tobytes()


def decode_embedding(blob: bytes, dim: int | None = None) -> np.ndarray:
    """Inverse of `encode_embedding`. Optionally checks dimension."""
    arr = np.frombuffer(blob, dtype=np.float32)
    if dim is not None and arr.shape[0] != dim:
        raise ValueError(
            f"embedding dim mismatch: blob has {arr.shape[0]}, expected {dim}"
        )
    return arr
