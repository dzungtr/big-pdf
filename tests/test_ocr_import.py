"""Tests for the `ocr-import` stage: Unlimited-OCR markdown -> OcrPage JSON."""
import json
from pathlib import Path

import fitz
import pytest

from massive_pdf.ingest.ocr import OCR_STAGE, ocr_artifact_path
from massive_pdf.ingest.ocr_import import (
    _split_batches,
    _split_batch_into_pages,
    import_ocr_markdown,
)
from massive_pdf.ingest.pages import run_pages_stage
from massive_pdf.store import (
    completed_page_ordinals,
    connect,
    init_db,
    insert_document,
    list_pages,
)


def _make_pdf(path: Path, num_pages: int = 3) -> None:
    doc = fitz.open()
    for _ in range(num_pages):
        doc.new_page(width=595, height=842)
    doc.save(str(path))
    doc.close()


@pytest.fixture
def doc_with_pages(tmp_path):
    pdf = tmp_path / "tiny.pdf"
    _make_pdf(pdf, num_pages=6)
    db = tmp_path / "t.sqlite"
    out = tmp_path / "imgs"
    init_db(db)
    with connect(db) as c:
        doc_id = insert_document(c, str(pdf), title="tiny")
        c.commit()
    run_pages_stage(db, doc_id, pdf, out, dpi=72)
    return db, doc_id, out, pdf


# --- pure unit tests on the splitter helpers --------------------------------

def test_split_batches_returns_in_order():
    md = "head\n<!-- page 1-3 -->\nbody1\n<!-- page 4-4 -->\nbody2\n<!-- page 5-6 -->\nbody3\n"
    batches = _split_batches(md)
    assert [(s, e) for s, e, _ in batches] == [(1, 3), (4, 4), (5, 6)]
    assert "body1" in batches[0][2]
    assert "body2" in batches[1][2]
    assert "body3" in batches[2][2]


def test_split_batches_no_markers_raises():
    with pytest.raises(ValueError, match="batch markers"):
        _split_batches("no markers here")


def test_split_batch_into_pages_assigns_ordinals_in_order():
    body = "<PAGE>page1text\nmore\n<PAGE>page2text\n"
    mapping = _split_batch_into_pages(1, 3, body)
    assert set(mapping) == {1, 2, 3}
    assert "page1text" in mapping[1]
    assert "page2text" in mapping[2]
    # third page had no <PAGE> marker -> blank
    assert mapping[3] == ""


def test_split_batch_into_pages_drops_preamble():
    body = "preamble-before-first-page\n<PAGE>page1\n<PAGE>page2\n"
    mapping = _split_batch_into_pages(1, 2, body)
    assert "preamble" not in mapping[1]
    assert mapping[1].startswith("page1")


def test_split_batch_into_pages_no_markers_all_blank():
    body = "just prose, no <PAGE> markers at all"
    mapping = _split_batch_into_pages(10, 12, body)
    assert mapping == {10: "", 11: "", 12: ""}


# --- end-to-end import_ocr_markdown ------------------------------------------

def _md_for(pages: list[tuple[int, int]]) -> str:
    """Build a minimal markdown with batch markers covering `pages`."""
    parts = ["# header\n"]
    for s, e in pages:
        parts.append(f"\n<!-- page {s}-{e} -->\n")
        for ordinal in range(s, e + 1):
            parts.append(f"<PAGE>page-{ordinal}-content\n")
    return "".join(parts)


def test_import_writes_one_artifact_per_page_and_marks_done(doc_with_pages):
    db, doc_id, out, _ = doc_with_pages
    md = tmp_path_local = Path(str(db)).parent / "ocr.md"
    md.write_text(_md_for([(1, 3), (4, 6)]), encoding="utf-8")

    n = import_ocr_markdown(db, doc_id, md, out)
    assert n == 6

    with connect(db) as conn:
        done = completed_page_ordinals(conn, doc_id, OCR_STAGE)
    assert done == {1, 2, 3, 4, 5, 6}

    for ordinal in range(1, 7):
        art = ocr_artifact_path(out, doc_id, ordinal)
        assert art.exists(), f"missing {art}"
        data = json.loads(art.read_text(encoding="utf-8"))
        assert data["page_ordinal"] == ordinal
        assert f"page-{ordinal}-content" in data["raw_text"]


def test_import_resumes_skipping_done_pages(doc_with_pages):
    db, doc_id, out, _ = doc_with_pages
    md = Path(str(db)).parent / "ocr.md"
    md.write_text(_md_for([(1, 6)]), encoding="utf-8")

    import_ocr_markdown(db, doc_id, md, out)  # first pass: all 6
    # corrupt one artifact; second pass must NOT rewrite it (it's 'done')
    art3 = ocr_artifact_path(out, doc_id, 3)
    art3.write_text("CORRUPTED", encoding="utf-8")

    n = import_ocr_markdown(db, doc_id, md, out)
    assert n == 6  # all pages counted
    assert art3.read_text() == "CORRUPTED"  # skipped, not rewritten


def test_import_blank_pages_get_empty_raw_text(doc_with_pages):
    db, doc_id, out, _ = doc_with_pages
    md = Path(str(db)).parent / "ocr.md"
    # batch covers 1..6 but only 4 <PAGE> markers -> pages 5,6 blank
    body = "<PAGE>a\n<PAGE>b\n<PAGE>c\n<PAGE>d\n"
    md.write_text(f"<!-- page 1-6 -->\n{body}", encoding="utf-8")

    import_ocr_markdown(db, doc_id, md, out)
    for ordinal in (5, 6):
        art = ocr_artifact_path(out, doc_id, ordinal)
        data = json.loads(art.read_text(encoding="utf-8"))
        assert data["raw_text"] == ""


def test_import_raises_when_document_has_no_pages(tmp_path):
    db = tmp_path / "t.sqlite"
    init_db(db)
    with connect(db) as c:
        doc_id = insert_document(c, "ghost.pdf", title="ghost")
        c.commit()
    md = tmp_path / "x.md"
    md.write_text("<!-- page 1-1 -->\n<PAGE>hi\n", encoding="utf-8")
    with pytest.raises(ValueError, match="no pages"):
        import_ocr_markdown(db, doc_id, md, tmp_path)


# --- CLI --------------------------------------------------------------------

def test_cli_ocr_import_subcommand(doc_with_pages, capsys):
    from massive_pdf.__main__ import main
    db, doc_id, out, _ = doc_with_pages
    md = Path(str(db)).parent / "ocr.md"
    md.write_text(_md_for([(1, 6)]), encoding="utf-8")
    rc = main(["--db", str(db), "ocr-import", str(doc_id), str(md), "--out", str(out)])
    assert rc == 0
    captured = capsys.readouterr().out
    assert "ocr-import" in captured and "pages=6" in captured


def test_cli_ocr_import_unknown_doc_returns_2(tmp_path, capsys):
    from massive_pdf.__main__ import main
    db = tmp_path / "t.sqlite"
    init_db(db)
    md = tmp_path / "x.md"
    md.write_text("<!-- page 1-1 -->\n", encoding="utf-8")
    rc = main(["--db", str(db), "ocr-import", "999", str(md)])
    assert rc == 2
    err = capsys.readouterr().err
    assert "id=999" in err


def test_cli_help_lists_ocr_import(capsys):
    from massive_pdf.__main__ import main
    with pytest.raises(SystemExit):
        main(["--help"])
    out = capsys.readouterr().out
    assert "ocr-import" in out
