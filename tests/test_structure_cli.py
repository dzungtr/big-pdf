"""Tests for the slice-3 CLI subcommands: structure, invariants."""
import json
from pathlib import Path

import pytest

from massive_pdf.__main__ import main


def _seed_ocr(out_dir: Path, document_id: int, pages: list[tuple[int, str]]) -> None:
    ocr_dir = out_dir / "ocr" / f"doc{document_id}"
    ocr_dir.mkdir(parents=True, exist_ok=True)
    for ordinal, text in pages:
        data = {
            "page_ordinal": ordinal,
            "image_path": f"/tmp/p{ordinal}.png",
            "blocks": [{"kind": "text", "bbox": [0, 0, 1, 1], "text": text}],
            "raw_text": text,
        }
        (ocr_dir / f"page{ordinal:04d}.json").write_text(
            json.dumps(data, ensure_ascii=False), encoding="utf-8",
        )


def test_help_includes_new_subcommands(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["--help"])
    assert exc.value.code == 0
    out = capsys.readouterr().out
    assert "structure" in out
    assert "invariants" in out


def test_structure_subcommand_runs(tmp_path, capsys):
    db = tmp_path / "t.sqlite"
    out_dir = tmp_path / "artifacts"
    _seed_ocr(out_dir, 1, [
        (1, "Điều 1. Phạm vi.\nĐiều 2. Đối tượng."),
        (2, "Điều 3. Nguyên tắc.\nTheo Thông tư số 39/2014/TT-BTC."),
    ])

    rc = main(["--db", str(db), "register", "/tmp/x.pdf", "--title", "demo"])
    assert rc == 0
    capsys.readouterr()

    rc = main(["--db", str(db), "list"])
    assert rc == 0
    doc_id = int(capsys.readouterr().out.splitlines()[0].split("id=")[1].split()[0])

    rc = main(["--db", str(db), "structure", str(doc_id), "--out", str(out_dir)])
    assert rc == 0
    out = capsys.readouterr().out
    assert "structure stage:" in out
    assert f"doc_id={doc_id}" in out
    assert "clauses=3" in out  # 3 Điều


def test_invariants_subcommand_passes(tmp_path, capsys):
    db = tmp_path / "t.sqlite"
    out_dir = tmp_path / "artifacts"
    _seed_ocr(out_dir, 1, [
        (1, "Điều 1. A.\nĐiều 2. B."),
    ])

    main(["--db", str(db), "register", "/tmp/x.pdf", "--title", "demo"])
    capsys.readouterr()
    main(["--db", str(db), "list"])
    doc_id = int(capsys.readouterr().out.splitlines()[0].split("id=")[1].split()[0])

    main(["--db", str(db), "structure", str(doc_id), "--out", str(out_dir)])
    capsys.readouterr()

    rc = main(["--db", str(db), "invariants", str(doc_id)])
    assert rc == 0
    out = capsys.readouterr().out
    assert "ok=True" in out


def test_invariants_subcommand_fails_on_bad_graph(tmp_path, capsys):
    """The invariants subcommand exits 1 on a hard failure (empty clause body).

    Note: numbering gaps are advisory (see check_invariants), so this test
    uses an empty body — a genuine structural defect — to exercise the
    non-zero exit path.
    """
    db = tmp_path / "t.sqlite"
    out_dir = tmp_path / "artifacts"
    # Điều 1 with an empty body after the header.
    _seed_ocr(out_dir, 1, [
        (1, "Điều 1. "),
    ])

    main(["--db", str(db), "register", "/tmp/x.pdf", "--title", "demo"])
    capsys.readouterr()
    main(["--db", str(db), "list"])
    doc_id = int(capsys.readouterr().out.splitlines()[0].split("id=")[1].split()[0])

    main(["--db", str(db), "structure", str(doc_id), "--out", str(out_dir)])
    capsys.readouterr()

    rc = main(["--db", str(db), "invariants", str(doc_id)])
    assert rc == 1
    out = capsys.readouterr().out
    assert "ok=False" in out
    assert "empty body" in out


def test_structure_unknown_doc_returns_error(tmp_path, capsys):
    db = tmp_path / "t.sqlite"
    rc = main(["--db", str(db), "structure", "999"])
    assert rc == 2
    err = capsys.readouterr().err
    assert "no document with id=999" in err


def test_invariants_unknown_doc_returns_error(tmp_path, capsys):
    db = tmp_path / "t.sqlite"
    rc = main(["--db", str(db), "invariants", "999"])
    assert rc == 2
    err = capsys.readouterr().err
    assert "no document with id=999" in err
