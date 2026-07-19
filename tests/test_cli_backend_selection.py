"""Tests for slice-3 CLI backend selection (issue #22).

Covers the `ocr --backend {stub,vlm}` flag and `default_backend()` env
selection. No live network: the VLM backend is asserted on by capturing
the instance handed to `run_ocr_stage`, not by hitting a server.
"""
from __future__ import annotations

import fitz
import pytest

from massive_pdf.__main__ import main
from massive_pdf.ingest.ocr import StubBackend, default_backend
from massive_pdf.ingest.vlm import UnlimitedOcrBackend


def _make_pdf(path, num_pages: int = 2) -> None:
    doc = fitz.open()
    for i in range(num_pages):
        page = doc.new_page(width=595, height=842)
        page.insert_text((72, 72), f"Page {i + 1}")
    doc.save(str(path))
    doc.close()


def _register_with_pages(tmp_path, capsys, num_pages: int = 2) -> int:
    """Run pages stage; return the registered doc_id."""
    db = tmp_path / "t.sqlite"
    pdf = tmp_path / "doc.pdf"
    _make_pdf(pdf, num_pages=num_pages)
    out = tmp_path / "imgs"
    rc = main(["--db", str(db), "pages", str(pdf), "--out", str(out),
               "--dpi", "150", "--title", "demo"])
    assert rc == 0
    capsys.readouterr()
    rc = main(["--db", str(db), "list"])
    assert rc == 0
    list_out = capsys.readouterr().out
    return int(list_out.splitlines()[0].split("id=")[1].split()[0]), db, out


def _captured_backends(monkeypatch):
    """Capture the `backend=` passed to run_ocr_stage."""
    captured: list = []

    def _fake_run_ocr_stage(db_path, document_id, out_dir, backend=None):
        captured.append(backend)
        return 0

    import massive_pdf.__main__ as cli_mod
    monkeypatch.setattr(cli_mod, "run_ocr_stage", _fake_run_ocr_stage)
    return captured


def test_ocr_backend_stub_uses_stub_backend(tmp_path, capsys, monkeypatch):
    captured = _captured_backends(monkeypatch)
    doc_id, db, _ = _register_with_pages(tmp_path, capsys)
    out = tmp_path / "artifacts"
    rc = main(["--db", str(db), "ocr", str(doc_id), "--out", str(out),
               "--backend", "stub"])
    assert rc == 0
    assert len(captured) == 1
    assert isinstance(captured[0], StubBackend)
    assert not isinstance(captured[0], UnlimitedOcrBackend)


def test_ocr_backend_vlm_constructs_unlimited_backend(tmp_path, capsys, monkeypatch):
    captured = _captured_backends(monkeypatch)
    doc_id, db, _ = _register_with_pages(tmp_path, capsys)
    out = tmp_path / "artifacts"
    rc = main(["--db", str(db), "ocr", str(doc_id), "--out", str(out),
               "--backend", "vlm"])
    assert rc == 0
    assert len(captured) == 1
    assert isinstance(captured[0], UnlimitedOcrBackend)


def test_ocr_backend_vlm_default_endpoint_env_overridden(
    tmp_path, capsys, monkeypatch,
):
    """--backend vlm with MASSIVE_PDF_VLM_ENDPOINT set honors the env value."""
    captured = _captured_backends(monkeypatch)
    doc_id, db, _ = _register_with_pages(tmp_path, capsys)
    out = tmp_path / "artifacts"
    monkeypatch.setenv("MASSIVE_PDF_VLM_ENDPOINT", "http://example.test:9999/v1")
    rc = main(["--db", str(db), "ocr", str(doc_id), "--out", str(out),
               "--backend", "vlm"])
    assert rc == 0
    assert captured[0].endpoint == "http://example.test:9999/v1"


def test_ocr_backend_default_is_stub(tmp_path, capsys, monkeypatch):
    """With no --backend and no env var, the stub backend is selected."""
    monkeypatch.delenv("MASSIVE_PDF_VLM_ENDPOINT", raising=False)
    captured = _captured_backends(monkeypatch)
    doc_id, db, _ = _register_with_pages(tmp_path, capsys)
    out = tmp_path / "artifacts"
    rc = main(["--db", str(db), "ocr", str(doc_id), "--out", str(out)])
    assert rc == 0
    assert isinstance(captured[0], StubBackend)


def test_ocr_help_documents_backend_and_env_var(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["ocr", "--help"])
    assert exc.value.code == 0
    out = capsys.readouterr().out
    assert "--backend" in out
    assert "stub" in out
    assert "vlm" in out
    assert "MASSIVE_PDF_VLM_ENDPOINT" in out
    # argparse may wrap the runbook path across a line break; check the
    # unwrapped form too so a rewrap doesn't flake the test.
    assert "unlimited-ocr.md" in out.replace("\n", "").replace(" ", "")


def test_default_backend_returns_stub_when_env_unset(monkeypatch):
    monkeypatch.delenv("MASSIVE_PDF_VLM_ENDPOINT", raising=False)
    assert isinstance(default_backend(), StubBackend)


def test_default_backend_returns_vlm_when_env_set(monkeypatch):
    monkeypatch.setenv("MASSIVE_PDF_VLM_ENDPOINT", "http://example.test:9999/v1")
    backend = default_backend()
    assert isinstance(backend, UnlimitedOcrBackend)
    assert backend.endpoint == "http://example.test:9999/v1"
