"""End-to-end tests for the slice-5 `cards` ingest stage."""
import json
from pathlib import Path

import pytest

from massive_pdf.ingest.ocr import OcrBlock, OcrPage, _serialize_page
from massive_pdf.retrieval.stage import (
    CARDS_STAGE,
    check_card_dimensions,
    run_cards_stage,
)
from massive_pdf.structure.stage import run_structure_stage
from massive_pdf.store import (
    connect,
    count_rule_cards_by_document,
    get_checkpoint,
    init_db,
    insert_document,
    list_rule_cards_for_document,
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
def doc_with_clauses(tmp_path):
    """A document whose structure stage has already run."""
    db = tmp_path / "t.sqlite"
    out_dir = tmp_path / "artifacts"
    init_db(db)
    pages = [
        (1, """Điều 1. Phạm vi điều chỉnh.
Hộ kinh doanh kê khai thuế theo quý. Phạt cảnh cáo nếu vi phạm.

Điều 2. Đối tượng áp dụng.
Cá nhân kinh doanh có đăng ký.
"""),
        (2, """Điều 3. Nguyên tắc kê khai.
Căn cứ Điều 1, kê khai thuế như sau.

Điều 4. Tổ chức thực hiện.
Doanh nghiệp thực hiện nộp thuế điện tử.
"""),
    ]
    _seed_ocr(out_dir, document_id=1, pages=pages)
    with connect(db) as c:
        doc_id = insert_document(c, "/tmp/dummy.pdf", title="test-doc")
        c.commit()
    # Run structure stage to populate the clauses table.
    run_structure_stage(db, doc_id, out_dir)
    return db, doc_id, out_dir


def test_cards_stage_inserts_one_card_per_dieu(doc_with_clauses):
    db, doc_id, out_dir = doc_with_clauses
    result = run_cards_stage(db, doc_id, out_dir)
    assert result["ok"] is True
    assert result["clauses_scanned"] == 4
    assert result["cards_inserted"] == 4  # one card per Điều
    assert result["skipped"] == 0

    with connect(db) as c:
        cards = list_rule_cards_for_document(c, doc_id)
        citations = sorted({c["statement"].split(":")[0] for c in cards})
        assert citations == ["Điều 1", "Điều 2", "Điều 3", "Điều 4"]


def test_cards_stage_records_checkpoints(doc_with_clauses):
    db, doc_id, out_dir = doc_with_clauses
    run_cards_stage(db, doc_id, out_dir)
    with connect(db) as c:
        # One per-clause 'done' marker.
        rows = c.execute(
            "SELECT page_ordinal, status FROM ingest_checkpoints "
            "WHERE document_id=? AND stage=? AND page_ordinal > 0",
            (doc_id, CARDS_STAGE),
        ).fetchall()
        ordinals = {r[0] for r in rows}
        # one marker per clause (4 clauses).
        assert len(ordinals) == 4
        # doc-level marker too.
        ck = get_checkpoint(c, doc_id, CARDS_STAGE, 0)
        assert ck["status"] == "done"


def test_cards_stage_is_idempotent(doc_with_clauses):
    db, doc_id, out_dir = doc_with_clauses
    run_cards_stage(db, doc_id, out_dir)
    result = run_cards_stage(db, doc_id, out_dir)
    with connect(db) as c:
        n = count_rule_cards_by_document(c, doc_id)
    # Second run should skip every clause (already-done markers) and not
    # duplicate rows.
    assert result["skipped"] == 4
    assert result["cards_inserted"] == 0
    assert n == 4


def test_cards_stage_rebuild_drops_existing_cards(doc_with_clauses):
    db, doc_id, out_dir = doc_with_clauses
    run_cards_stage(db, doc_id, out_dir)
    with connect(db) as c:
        before = count_rule_cards_by_document(c, doc_id)
    assert before == 4
    run_cards_stage(db, doc_id, out_dir, rebuild=True)
    with connect(db) as c:
        after = count_rule_cards_by_document(c, doc_id)
    assert after == 4  # regenerated, same count


def test_cards_stage_writes_artifact(doc_with_clauses):
    db, doc_id, out_dir = doc_with_clauses
    result = run_cards_stage(db, doc_id, out_dir)
    assert result["artifact_path"]
    artifact = json.loads(Path(result["artifact_path"]).read_text(encoding="utf-8"))
    assert artifact["document_id"] == doc_id
    assert artifact["card_count"] == 4


def test_cards_stage_no_clauses_still_records_checkpoint(tmp_path):
    db = tmp_path / "t.sqlite"
    out_dir = tmp_path / "artifacts"
    init_db(db)
    with connect(db) as c:
        did = insert_document(c, "/tmp/empty.pdf", title="empty")
        c.commit()
    result = run_cards_stage(db, did, out_dir)
    assert result["clauses_scanned"] == 0
    assert result["cards_inserted"] == 0
    with connect(db) as c:
        ck = get_checkpoint(c, did, CARDS_STAGE, 0)
        assert ck["status"] == "done"


def test_check_card_dimensions_with_no_embeddings(doc_with_clauses):
    db, doc_id, out_dir = doc_with_clauses
    run_cards_stage(db, doc_id, out_dir)
    report = check_card_dimensions(db, doc_id, expected_dim=128)
    # No embeddings yet.
    assert report["checked"] == 0
    assert report["ok"] is True
