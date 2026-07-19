"""Tests for the structure ingest stage (end-to-end with stub OCR)."""
import json
from pathlib import Path

import pytest

from massive_pdf.ingest.ocr import OcrBlock, OcrPage, _serialize_page
from massive_pdf.structure.stage import (
    STRUCTURE_STAGE,
    check_invariants,
    run_structure_stage,
)
from massive_pdf.store import (
    connect,
    count_clauses_by_kind,
    dangling_references,
    get_checkpoint,
    init_db,
    insert_document,
    list_clause_references,
    list_clauses,
)


def _seed_ocr(out_dir: Path, document_id: int, pages: list[tuple[int, str]]) -> None:
    ocr_dir = out_dir / "ocr" / f"doc{document_id}"
    ocr_dir.mkdir(parents=True, exist_ok=True)
    for ordinal, text in pages:
        page = OcrPage(
            page_ordinal=ordinal,
            image_path=f"/tmp/p{ordinal}.png",
            blocks=[OcrBlock(kind="text", bbox=(0.0, 0.0, 1.0, 1.0), text=text)],
            raw_text=text,
        )
        _serialize_page(page, ocr_dir / f"page{ordinal:04d}.json")


@pytest.fixture
def doc_with_ocr(tmp_path):
    db = tmp_path / "t.sqlite"
    out_dir = tmp_path / "artifacts"
    init_db(db)
    pages = [
        (1, """THÔNG TƯ 89/2015/TT-BTC

Điều 1. Phạm vi điều chỉnh.
Thông tư này quy định về quản lý thuế cho hộ kinh doanh.

Điều 2. Đối tượng áp dụng.
1. Cá nhân;
2. Hộ kinh doanh có đăng ký;
3. Doanh nghiệp tư nhân.
Việc này thực hiện theo khoản 1 Điều 5 Luật Thương mại.
"""),
        (2, """Điều 3. Nguyên tắc kê khai.
Căn cứ Điều 5 và khoản 2 Điều này, kê khai như sau:
    a) Kê khai theo quý;
    b) Kê khai theo tháng.
Theo Thông tư số 39/2014/TT-BTC thì mức thuế suất là 1%.

Điều 4. Điều khoản thi hành.
Bãi bỏ Điều 5 Thông tư số 78/2014/TT-BTC.
"""),
    ]
    _seed_ocr(out_dir, document_id=1, pages=pages)
    with connect(db) as c:
        doc_id = insert_document(c, "/tmp/dummy.pdf", title="TT 89")
        c.commit()
    return db, doc_id, out_dir


def test_run_structure_stage_returns_summary(doc_with_ocr):
    db, doc_id, out_dir = doc_with_ocr
    result = run_structure_stage(db, doc_id, out_dir)
    assert result["ok"] is True
    # 4 Điều + 2 Điểm under Điều 3 = 6 clauses total.
    assert result["clauses"] == 6
    assert result["pages"] == 2
    # Artifact file exists.
    assert Path(result["artifact_path"]).exists()


def test_run_structure_stage_persists_clauses(doc_with_ocr):
    db, doc_id, out_dir = doc_with_ocr
    run_structure_stage(db, doc_id, out_dir)
    with connect(db) as c:
        clauses = list_clauses(c, doc_id)
        assert [cl["citation"] for cl in clauses] == [
            "Điều 1", "Điều 2", "Điều 3",
            "Điểm a Điều 3", "Điểm b Điều 3",
            "Điều 4",
        ]
        # Điểm a/b should have parent_id pointing at Điều 3.
        dieu_3 = next(cl for cl in clauses if cl["citation"] == "Điều 3")
        diem_a = next(cl for cl in clauses if cl["citation"] == "Điểm a Điều 3")
        assert diem_a["parent_id"] == dieu_3["id"]
        kinds = count_clauses_by_kind(c, doc_id)
        assert kinds == {"dieu": 4, "diem": 2}


def test_run_structure_stage_records_checkpoint(doc_with_ocr):
    db, doc_id, out_dir = doc_with_ocr
    run_structure_stage(db, doc_id, out_dir)
    with connect(db) as c:
        ck = get_checkpoint(c, doc_id, STRUCTURE_STAGE, 0)
        assert ck["status"] == "done"


def test_cross_references_persisted(doc_with_ocr):
    db, doc_id, out_dir = doc_with_ocr
    run_structure_stage(db, doc_id, out_dir)
    with connect(db) as c:
        clauses = list_clauses(c, doc_id)
        # Pick Điều 3 (which has 'khoản 2 Điều này' + 'Điều 5' refs).
        d3 = next(cl for cl in clauses if cl["citation"] == "Điều 3")
        out_refs = list_clause_references(c, d3["id"], direction="out")
        cites = [r["target_citation"] for r in out_refs]
        assert "Điều 5" in cites
        # Self-ref rewritten to the source clause's ancestor Điều.
        assert "Khoản 2 Điều 3" in cites


def test_external_refs_dangle(doc_with_ocr):
    db, doc_id, out_dir = doc_with_ocr
    run_structure_stage(db, doc_id, out_dir)
    with connect(db) as c:
        dangles = dangling_references(c, doc_id)
        ext_cites = [d["target_citation"] for d in dangles if d["kind"] == "external"]
        assert "Thông tư số 39/2014/TT-BTC" in ext_cites
        assert "Thông tư số 78/2014/TT-BTC" in ext_cites


def test_run_structure_stage_idempotent(doc_with_ocr):
    db, doc_id, out_dir = doc_with_ocr
    run_structure_stage(db, doc_id, out_dir)
    # Re-run; clause count must be unchanged (upsert key).
    run_structure_stage(db, doc_id, out_dir)
    with connect(db) as c:
        clauses = list_clauses(c, doc_id)
        assert len(clauses) == 6


def test_rebuild_flag_drops_existing_clauses(doc_with_ocr):
    db, doc_id, out_dir = doc_with_ocr
    run_structure_stage(db, doc_id, out_dir)
    with connect(db) as c:
        before = len(list_clauses(c, doc_id))
    assert before == 6
    run_structure_stage(db, doc_id, out_dir, rebuild=True)
    with connect(db) as c:
        after = len(list_clauses(c, doc_id))
    assert after == 6  # same content re-parsed


def test_no_ocr_no_clauses(tmp_path):
    db = tmp_path / "t.sqlite"
    out_dir = tmp_path / "artifacts"
    init_db(db)
    with connect(db) as c:
        doc_id = insert_document(c, "/tmp/x.pdf", title="empty")
        c.commit()
    result = run_structure_stage(db, doc_id, out_dir)
    assert result["clauses"] == 0
    # Still records a checkpoint so callers can see we ran.
    with connect(db) as c:
        ck = get_checkpoint(c, doc_id, STRUCTURE_STAGE, 0)
        assert ck["status"] == "done"


def test_invariants_pass_on_well_formed_doc(doc_with_ocr):
    db, doc_id, out_dir = doc_with_ocr
    run_structure_stage(db, doc_id, out_dir)
    report = check_invariants(db, doc_id)
    assert report.ok, f"unexpected issues: {report.issues}"


def test_invariants_gap_in_dieu_numbering_is_advisory(tmp_path):
    """A mid-chapter numbering gap is reported as an advisory note, not a
    failure — Vietnamese regulations legitimately skip article numbers."""
    db = tmp_path / "t.sqlite"
    out_dir = tmp_path / "artifacts"
    init_db(db)
    pages = [
        (1, "Điều 1. A.\nĐiều 3. C."),  # gap: no Điều 2, no Chương marker
    ]
    _seed_ocr(out_dir, document_id=1, pages=pages)
    with connect(db) as c:
        doc_id = insert_document(c, "/tmp/x.pdf", title="gaps")
        c.commit()
    run_structure_stage(db, doc_id, out_dir)
    report = check_invariants(db, doc_id, out_dir=out_dir)
    # Gap is advisory (a note), not a hard failure.
    assert report.ok, f"gap should be advisory, not a failure: {report.issues}"
    assert any("gaps" in n for n in report.notes)


def test_invariants_flag_empty_body(tmp_path):
    db = tmp_path / "t.sqlite"
    out_dir = tmp_path / "artifacts"
    init_db(db)
    pages = [
        (1, "Điều 1. "),  # body trimmed is empty
    ]
    _seed_ocr(out_dir, document_id=1, pages=pages)
    with connect(db) as c:
        doc_id = insert_document(c, "/tmp/x.pdf", title="empty body")
        c.commit()
    run_structure_stage(db, doc_id, out_dir)
    report = check_invariants(db, doc_id)
    assert not report.ok
    assert any("empty body" in s for s in report.issues)


# --- chapter-aware numbering invariant (per-chapter, not global) -----------

def _seed_ocr_with_dir(out_dir: Path, document_id: int, pages: list[tuple[int, str]]) -> None:
    """Like _seed_ocr but returns nothing; used for chapter-aware tests."""
    _seed_ocr(out_dir, document_id, pages)


def test_invariants_chapter_gap_not_flagged(tmp_path):
    """A numbering gap that coincides with a Chương marker is legitimate."""
    db = tmp_path / "t.sqlite"
    out_dir = tmp_path / "artifacts"
    init_db(db)
    # Page 1: Điều 13 ends, Chương III begins, then Điều 17 — mirroring
    # the real Thông tư 89 structure where 14-16 don't exist.
    pages = [
        (1, "Điều 13. Tail of chapter II.\nChương III\nKHAI THUẾ\nĐiều 17. Start of chapter III."),
    ]
    _seed_ocr(out_dir, document_id=1, pages=pages)
    with connect(db) as c:
        doc_id = insert_document(c, "/tmp/x.pdf", title="chapters")
        c.commit()
    run_structure_stage(db, doc_id, out_dir)
    report = check_invariants(db, doc_id, out_dir=out_dir)
    assert report.ok, f"chapter-boundary gap should not be flagged: {report.issues}"


def test_invariants_gap_within_chapter_reported_as_note(tmp_path):
    """A numbering gap with no Chương marker is reported as an advisory note."""
    db = tmp_path / "t.sqlite"
    out_dir = tmp_path / "artifacts"
    init_db(db)
    pages = [
        (1, "Điều 1. A.\nĐiều 3. C."),  # gap: no Điều 2, no Chương marker
    ]
    _seed_ocr(out_dir, document_id=1, pages=pages)
    with connect(db) as c:
        doc_id = insert_document(c, "/tmp/x.pdf", title="gap no chapter")
        c.commit()
    run_structure_stage(db, doc_id, out_dir)
    report = check_invariants(db, doc_id, out_dir=out_dir)
    assert report.ok
    assert any("gaps" in n for n in report.notes)


def test_invariants_out_dir_omitted_falls_back_to_global_gap_note(tmp_path):
    """When out_dir is omitted and no OCR dir is found, chapter detection
    can't run; the gap is still reported as an advisory note (not a failure)."""
    db = tmp_path / "t.sqlite"
    out_dir = tmp_path / "artifacts"
    init_db(db)
    pages = [(1, "Điều 1. A.\nĐiều 3. C.")]
    _seed_ocr(out_dir, document_id=1, pages=pages)
    with connect(db) as c:
        doc_id = insert_document(c, "/tmp/x.pdf", title="default out")
        c.commit()
    run_structure_stage(db, doc_id, out_dir)
    import os
    os.chdir(tmp_path)
    report = check_invariants(db, doc_id)  # out_dir defaults to ".massive_pdf"
    assert report.ok
    assert any("gaps" in n for n in report.notes)
