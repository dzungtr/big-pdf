"""Tests for the `ocr` ingest stage (stub backend; no GPU needed)."""
import json
from pathlib import Path

import fitz
import pytest

from massive_pdf.ingest.ocr import (
    OcrBlock,
    OcrPage,
    OCR_STAGE,
    StubBackend,
    ocr_artifact_path,
    run_ocr_stage,
)
from massive_pdf.ingest.pages import run_pages_stage
from massive_pdf.store import (
    completed_page_ordinals,
    connect,
    get_checkpoint,
    init_db,
    insert_document,
)


def _make_pdf(path: Path, num_pages: int = 3) -> None:
    doc = fitz.open()
    for i in range(num_pages):
        page = doc.new_page(width=595, height=842)
        page.insert_text((72, 72), f"Page {i + 1}")
    doc.save(str(path))
    doc.close()


@pytest.fixture
def doc_with_pages(tmp_path):
    pdf = tmp_path / "tiny.pdf"
    _make_pdf(pdf, num_pages=3)
    db = tmp_path / "t.sqlite"
    out = tmp_path / "images"
    init_db(db)
    with connect(db) as c:
        doc_id = insert_document(c, str(pdf), title="tiny")
        c.commit()
    run_pages_stage(db, doc_id, pdf, out, dpi=150)
    return db, doc_id, out, pdf


def test_ocr_block_roundtrip():
    b = OcrBlock(kind="marker", bbox=(10.0, 20.0, 30.0, 40.0), text="Điều 5.")
    d = b.to_dict()
    b2 = OcrBlock.from_dict(d)
    assert b2 == b


def test_ocr_page_roundtrip():
    p = OcrPage(page_ordinal=7, image_path="/x.png", blocks=[
        OcrBlock(kind="text", bbox=(0, 0, 1, 1), text="hello"),
        OcrBlock(kind="header", bbox=(0, 0, 1, 1), text="Thông tư 89"),
    ], raw_text="hello\nThông tư 89")
    d = p.to_dict()
    p2 = OcrPage.from_dict(d)
    assert p2 == p


def test_stub_backend_is_deterministic():
    s = StubBackend()
    p1 = s.transcribe("/nonexistent/a.png", 1)
    p2 = s.transcribe("/nonexistent/a.png", 1)
    assert p1.raw_text == p2.raw_text
    assert p1.page_ordinal == 1
    assert "[stub OCR]" in p1.raw_text


def test_run_ocr_stage_writes_per_page_artifact(doc_with_pages):
    db, doc_id, out, _ = doc_with_pages
    n = run_ocr_stage(db, doc_id, out, backend=StubBackend())
    assert n == 3
    for ordinal in (1, 2, 3):
        art = ocr_artifact_path(out, doc_id, ordinal)
        assert art.exists()
        data = json.loads(art.read_text())
        assert data["page_ordinal"] == ordinal
        assert data["blocks"], "stub backend should produce at least one block"


def test_run_ocr_stage_is_idempotent_and_resumable(doc_with_pages):
    db, doc_id, out, _ = doc_with_pages
    run_ocr_stage(db, doc_id, out, backend=StubBackend())
    # Second run: all pages are already 'done'; no new files written, count stays.
    n2 = run_ocr_stage(db, doc_id, out, backend=StubBackend())
    assert n2 == 3
    with connect(db) as c:
        done = completed_page_ordinals(c, doc_id, OCR_STAGE)
        assert done == {1, 2, 3}
        for ordinal in (1, 2, 3):
            ck = get_checkpoint(c, doc_id, OCR_STAGE, ordinal)
            assert ck["status"] == "done"
            assert ck["artifact_path"].endswith(f"page{ordinal:04d}.json")
