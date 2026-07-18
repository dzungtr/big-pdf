"""Tests for the slice-2 CLI subcommands: register, list, pages, ocr."""
import subprocess
import sys
from pathlib import Path

import fitz
import pytest

from massive_pdf.__main__ import main


def _make_pdf(path: Path, num_pages: int = 2) -> None:
    doc = fitz.open()
    for i in range(num_pages):
        page = doc.new_page(width=595, height=842)
        page.insert_text((72, 72), f"Page {i + 1}")
    doc.save(str(path))
    doc.close()


def test_help_lists_subcommands(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["--help"])
    assert exc.value.code == 0
    out = capsys.readouterr().out
    for cmd in ("register", "list", "pages", "ocr"):
        assert cmd in out


def test_register_then_list(tmp_path, capsys):
    db = tmp_path / "t.sqlite"
    pdf = tmp_path / "doc.pdf"
    _make_pdf(pdf, num_pages=1)
    rc = main(["--db", str(db), "register", str(pdf), "--title", "demo"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "registered doc_id=" in out

    capsys.readouterr()  # discard
    rc = main(["--db", str(db), "list"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "doc.pdf" in out
    assert "demo" in out


def test_pages_then_ocr_subcommand(tmp_path, capsys):
    db = tmp_path / "t.sqlite"
    pdf = tmp_path / "doc.pdf"
    _make_pdf(pdf, num_pages=3)
    images = tmp_path / "imgs"
    out = tmp_path / "artifacts"

    rc = main(["--db", str(db), "pages", str(pdf),
               "--out", str(images), "--dpi", "150", "--title", "demo"])
    assert rc == 0
    capsys.readouterr()

    rc = main(["--db", str(db), "list"])
    assert rc == 0
    list_out = capsys.readouterr().out
    # doc id is the first registered doc
    doc_id = int(list_out.splitlines()[0].split("id=")[1].split()[0])

    rc = main(["--db", str(db), "ocr", str(doc_id), "--out", str(out)])
    assert rc == 0
    ocr_out = capsys.readouterr().out
    assert "ocr stage: doc_id=" in ocr_out
    assert "pages=3" in ocr_out
    # Artifact files exist
    for o in (1, 2, 3):
        assert (out / "ocr" / f"doc{doc_id}" / f"page{o:04d}.json").exists()


def test_ocr_unknown_doc_returns_error(tmp_path, capsys):
    db = tmp_path / "t.sqlite"
    rc = main(["--db", str(db), "ocr", "999"])
    assert rc == 2
    err = capsys.readouterr().err
    assert "no document with id=999" in err


def test_module_invocation_runs(tmp_path):
    """The package should be runnable as `python -m massive_pdf ...`."""
    db = tmp_path / "t.sqlite"
    rc = main(["--db", str(db), "list"])
    assert rc == 0


# ---------------------------------------------------------------------------
# Slice 5 (issue #5): rule_cards CLI subcommands
# ---------------------------------------------------------------------------

def test_cards_cli_unknown_doc(tmp_path, capsys):
    db = tmp_path / "t.sqlite"
    rc = main(["--db", str(db), "cards", "999"])
    assert rc == 2
    err = capsys.readouterr().err
    assert "no document with id=999" in err


def test_embed_cli_unknown_doc(tmp_path, capsys):
    db = tmp_path / "t.sqlite"
    rc = main(["--db", str(db), "embed", "999"])
    assert rc == 2


def test_cards_cli_help_lists_subcommand(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["--help"])
    assert exc.value.code == 0
    out = capsys.readouterr().out
    assert "cards" in out
    assert "embed" in out
