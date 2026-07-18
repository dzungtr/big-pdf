"""Encoder tests: deterministic HashBag + BLOB serialisation round-trip."""
import os

import numpy as np
import pytest

from massive_pdf.retrieval.encoder import (
    BgeM3Encoder,
    Encoder,
    HashBagEncoder,
    decode_embedding,
    encode_embedding,
    get_default_encoder,
    normalize,
    tokenize,
)


def test_tokenize_lowercases_and_strips_punctuation():
    toks = tokenize("Hộ kinh doanh, kê khai THUẾ!")
    assert toks == ["hộ", "kinh", "doanh", "kê", "khai", "thuế"]


def test_hashbag_is_deterministic():
    e1 = HashBagEncoder(dim=64)
    e2 = HashBagEncoder(dim=64)
    v1 = e1.encode("hộ kinh doanh")
    v2 = e2.encode("hộ kinh doanh")
    assert np.array_equal(v1, v2)


def test_hashbag_normalised_to_unit_norm():
    e = HashBagEncoder(dim=64)
    v = e.encode("a b c d e")
    assert abs(float(np.linalg.norm(v)) - 1.0) < 1e-5


def test_hashbag_dim_can_be_configured():
    e16 = HashBagEncoder(dim=16)
    e256 = HashBagEncoder(dim=256)
    assert e16.encode("x").shape == (16,)
    assert e256.encode("x").shape == (256,)


def test_hashbag_invalid_dim_rejected():
    with pytest.raises(ValueError):
        HashBagEncoder(dim=0)


def test_similar_texts_have_higher_similarity_than_unrelated():
    e = HashBagEncoder(dim=256)
    a = e.encode("hộ kinh doanh kê khai thuế theo quý")
    b = e.encode("hộ kinh doanh kê khai thuế hàng quý")
    c = e.encode("âm nhạc dân gian việt nam")
    sim_ab = float(a @ b)
    sim_ac = float(a @ c)
    assert sim_ab > sim_ac


def test_embedding_bytes_roundtrip():
    e = HashBagEncoder(dim=64)
    v = e.encode("hello world")
    blob = encode_embedding(v)
    assert isinstance(blob, bytes)
    assert len(blob) == 64 * 4  # float32 = 4 bytes per element
    v2 = decode_embedding(blob, dim=64)
    assert np.allclose(v, v2)


def test_encode_embedding_rejects_non_1d():
    with pytest.raises(ValueError, match="1-D"):
        encode_embedding(np.zeros((3, 3), dtype=np.float32))


def test_decode_embedding_dim_mismatch_raises():
    e = HashBagEncoder(dim=64)
    blob = encode_embedding(e.encode("x"))
    with pytest.raises(ValueError, match="dim mismatch"):
        decode_embedding(blob, dim=32)


def test_encoder_protocol_satisfied_by_hashbag():
    e = HashBagEncoder(dim=16)
    assert isinstance(e, Encoder)


def test_default_encoder_returns_hashbag_without_env():
    e = get_default_encoder(dim=32)
    assert isinstance(e, HashBagEncoder)
    assert e.dim == 32


def test_default_encoder_returns_bge_when_env_set(monkeypatch):
    monkeypatch.setenv("MASSIVE_PDF_EMBEDDING_MODEL", "bge-m3")
    e = get_default_encoder()
    assert isinstance(e, BgeM3Encoder)


def test_bge_lazy_loads_on_first_encode_when_deps_missing():
    """If sentence-transformers is not installed, BgeM3Encoder.encode raises
    a clear RuntimeError instead of an opaque ImportError."""
    e = BgeM3Encoder(model_name="bge-m3")
    # Don't install sentence-transformers in tests; if it IS available,
    # skip this assertion.
    try:
        import sentence_transformers  # noqa: F401
    except ImportError:
        with pytest.raises(RuntimeError, match="sentence-transformers"):
            e.encode("hello")
