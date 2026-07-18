"""Tests for the `pages` ingest stage (PDF -> page images)."""
from pathlib import Path

import fitz
import pytest

from massive_pdf.ingest.pages import (
    DEFAULT_DPI,
    PAGES_STAGE,
    page_image_path,
    render_pages_to_dir,
    run_pages_stage,
)
from massive_pdf.store import (
    completed_page_ordinals,
    connect,
    get_page,
    init_db,
    insert_document,
    list_pages,
)


def _make_pdf(path: Path, num_pages: int = 3) -> None:
    """Write a tiny multi-page PDF using PyMuPDF directly."""
    doc = fitz.open()
    for i in range(num_pages):
        page = doc.new_page(width=595, height=842)  # A4 portrait
        page.insert_text((72, 72), f"Page {i + 1}")
    doc.save(str(path))
    doc.close()


def test_page_image_path_layout(tmp_path):
    p = page_image_path(tmp_path, document_id=42, page_ordinal=7)
    assert p == tmp_path / "doc42" / "page0007.png"


def test_render_pages_to_dir_writes_pngs(tmp_path):
    pdf = tmp_path / "tiny.pdf"
    _make_pdf(pdf, num_pages=3)
    out = tmp_path / "images"
    pages = render_pages_to_dir(pdf, out, document_id=1, dpi=200)
    assert len(pages) == 3
    assert [p.page_ordinal for p in pages] == [1, 2, 3]
    for p in pages:
        assert Path(p.image_path).exists()
        assert p.width > 0 and p.height > 0
        assert p.dpi == 200


def test_run_pages_stage_is_idempotent(tmp_path):
    pdf = tmp_path / "tiny.pdf"
    _make_pdf(pdf, num_pages=4)
    db = tmp_path / "t.sqlite"
    out = tmp_path / "images"
    init_db(db)
    with connect(db) as c:
        doc_id = insert_document(c, str(pdf), title="tiny")
        c.commit()

    n1 = run_pages_stage(db, doc_id, pdf, out, dpi=150)
    assert n1 == 4
    # Second run should see all 4 already done in checkpoint table.
    n2 = run_pages_stage(db, doc_id, pdf, out, dpi=150)
    assert n2 == 4

    with connect(db) as c:
        rows = list_pages(c, doc_id)
        assert [r["page_ordinal"] for r in rows] == [1, 2, 3, 4]
        for r in rows:
            assert Path(r["image_path"]).exists()
        done = completed_page_ordinals(c, doc_id, PAGES_STAGE)
        assert done == {1, 2, 3, 4}
