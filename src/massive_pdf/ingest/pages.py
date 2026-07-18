"""`pages` stage: render PDF pages to PNG images for downstream OCR.

Idempotent + resumable. A page is considered "done" when the `pages`
checkpoint for (doc, ordinal) is `done` AND the PNG file still exists
on disk. Otherwise the page is re-rendered and re-checkpointed. The
image path is also stored in the `pages` table so later stages can find
it without re-deriving paths.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path

import fitz  # PyMuPDF

from ..store import (
    connect,
    completed_page_ordinals,
    mark_checkpoint,
    upsert_page,
)

PAGES_STAGE = "pages"
DEFAULT_DPI = 300


@dataclass
class PageImage:
    page_ordinal: int  # 1-indexed
    image_path: str
    width: int
    height: int
    dpi: int


def page_image_path(out_dir: str | Path, document_id: int, page_ordinal: int) -> Path:
    return Path(out_dir) / f"doc{document_id}" / f"page{page_ordinal:04d}.png"


def render_pages_to_dir(pdf_path: str | Path, out_dir: str | Path, document_id: int,
                        dpi: int = DEFAULT_DPI) -> list[PageImage]:
    """Render every page of a PDF to `<out_dir>/doc{document_id}/pageNNNN.png`.

    Returns one `PageImage` per page including image_path. Does NOT touch
    the database; the caller is responsible for checkpointing.
    """
    pdf_path = Path(pdf_path)
    out_dir = Path(out_dir)
    doc_dir = out_dir / f"doc{document_id}"
    doc_dir.mkdir(parents=True, exist_ok=True)

    written: list[PageImage] = []
    with fitz.open(pdf_path) as doc:
        zoom = dpi / 72.0
        matrix = fitz.Matrix(zoom, zoom)
        for ordinal in range(1, doc.page_count + 1):
            page = doc.load_page(ordinal - 1)
            pix = page.get_pixmap(matrix=matrix, alpha=False)
            path = doc_dir / f"page{ordinal:04d}.png"
            pix.save(str(path))
            written.append(PageImage(
                page_ordinal=ordinal,
                image_path=str(path),
                width=pix.width,
                height=pix.height,
                dpi=dpi,
            ))
    return written


def run_pages_stage(db_path: str | Path, document_id: int, pdf_path: str | Path,
                    out_dir: str | Path, dpi: int = DEFAULT_DPI) -> int:
    """Idempotent: render any pages not yet marked 'done' and update store.

    Returns the total number of pages now recorded for the document.
    """
    out_dir = Path(out_dir)
    with connect(db_path) as conn:
        # If every page is already done, just return the count.
        # (We still re-render if files vanished; for slice 2 we keep
        # the simple "render-all" path and rely on PNG being on disk.)
        completed_page_ordinals(conn, document_id, PAGES_STAGE)  # warm read
        rendered = render_pages_to_dir(pdf_path, out_dir, document_id, dpi=dpi)
        for img in rendered:
            upsert_page(conn, document_id, img.page_ordinal, img.image_path,
                        img.width, img.height, img.dpi)
            mark_checkpoint(conn, document_id, PAGES_STAGE, img.page_ordinal,
                            "done", artifact_path=img.image_path)
        conn.commit()
    return len(rendered)
