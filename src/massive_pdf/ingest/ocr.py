"""`ocr` stage: page image -> structured Vietnamese text + layout blocks.

This commit ships the *scaffold* (issue #3 strategy: "scaffold + stub +
tests is the goal for this dispatch"):

  - A clear `OcrBackend` protocol and a JSON-friendly result schema
    (`OcrBlock`, `OcrPage`) covering text, headers, Điều/Khoản markers
    and tables — the layout categories the brief calls out.
  - A `StubBackend` that returns deterministic placeholder text from
    the image file's basename. Lets the rest of the pipeline run
    end-to-end without a GPU.
  - `run_ocr_stage`: idempotent / resumable. Skips pages whose
    `ocr` checkpoint is `done`; per-page result is serialized to
    `<out_dir>/ocr/doc<id>/pageNNNN.json` and recorded in the
    checkpoint's `artifact_path`.

The real Baidu Unlimited-OCR backend (3B MoE VLM served via
vLLM/SGLang) will be added as a follow-up commit behind the same
`OcrBackend` interface.
"""
from __future__ import annotations
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable, Protocol

from .pages import PAGES_STAGE  # noqa: F401  (re-exported convenience)
from ..store import (
    completed_page_ordinals,
    connect,
    get_page,
    list_pages,
    mark_checkpoint,
)

OCR_STAGE = "ocr"


@dataclass
class OcrBlock:
    """A single layout-annotated block from a page."""
    kind: str          # 'text' | 'header' | 'marker' | 'table' | 'figure'
    bbox: tuple[float, float, float, float]  # x0, y0, x1, y1 in image pixels
    text: str

    def to_dict(self) -> dict:
        return {"kind": self.kind, "bbox": list(self.bbox), "text": self.text}

    @classmethod
    def from_dict(cls, d: dict) -> "OcrBlock":
        return cls(kind=d["kind"], bbox=tuple(d["bbox"]), text=d["text"])


@dataclass
class OcrPage:
    page_ordinal: int
    image_path: str
    blocks: list[OcrBlock] = field(default_factory=list)
    raw_text: str = ""

    def to_dict(self) -> dict:
        return {
            "page_ordinal": self.page_ordinal,
            "image_path": self.image_path,
            "blocks": [b.to_dict() for b in self.blocks],
            "raw_text": self.raw_text,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "OcrPage":
        return cls(
            page_ordinal=d["page_ordinal"],
            image_path=d["image_path"],
            blocks=[OcrBlock.from_dict(b) for b in d.get("blocks", [])],
            raw_text=d.get("raw_text", ""),
        )


class OcrBackend(Protocol):
    """Pluggable OCR backend. Implementations must be deterministic per
    (image_path, model) so reruns are stable.
    """
    name: str

    def transcribe(self, image_path: str, page_ordinal: int) -> OcrPage: ...


class StubBackend:
    """Deterministic offline backend. Returns a placeholder page so the
    pipeline runs end-to-end without a GPU. Useful for tests and CI.
    """
    name = "stub"

    def transcribe(self, image_path: str, page_ordinal: int) -> OcrPage:
        text = f"[stub OCR] page={page_ordinal} image={Path(image_path).name}"
        return OcrPage(
            page_ordinal=page_ordinal,
            image_path=image_path,
            blocks=[OcrBlock(kind="text", bbox=(0.0, 0.0, 0.0, 0.0), text=text)],
            raw_text=text,
        )


def default_backend() -> OcrBackend:
    """Select the default OCR backend based on the environment.

    If ``MASSIVE_PDF_VLM_ENDPOINT`` is explicitly set, return an
    ``UnlimitedOcrBackend`` pointed at it (which fails loudly with the
    SGLang launch command if the server is unreachable — that is the
    contract documented in ``docs/runbooks/sglang-unlimited-ocr.md``).
    Otherwise return ``StubBackend``, the offline CI default (no GPU, no
    network) so the existing stub tests pass without any env juggling.

    Use the ``--backend`` CLI flag to override this unconditionally.
    """
    import os

    if os.environ.get("MASSIVE_PDF_VLM_ENDPOINT"):
        from .vlm import UnlimitedOcrBackend
        return UnlimitedOcrBackend()
    return StubBackend()


def ocr_artifact_path(out_dir: str | Path, document_id: int, page_ordinal: int) -> Path:
    return Path(out_dir) / "ocr" / f"doc{document_id}" / f"page{page_ordinal:04d}.json"


def _serialize_page(page: OcrPage, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(page.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")


def _deserialize_page(path: Path, page_ordinal: int) -> OcrPage:
    data = json.loads(path.read_text(encoding="utf-8"))
    data["page_ordinal"] = page_ordinal
    return OcrPage.from_dict(data)


def run_ocr_stage(db_path: str | Path, document_id: int, out_dir: str | Path,
                  backend: OcrBackend | None = None) -> int:
    """Run the OCR stage for every page the `pages` stage produced.

    Resumable: pages already marked `done` for the `ocr` stage are
    skipped. Failed pages are recorded with status `failed` and
    skipped on subsequent runs (caller can manually re-mark them).
    Returns the number of pages whose OCR artifact now exists.
    """
    backend = backend or default_backend()
    out_dir = Path(out_dir)
    written = 0
    with connect(db_path) as conn:
        done = completed_page_ordinals(conn, document_id, OCR_STAGE)
        pages = list_pages(conn, document_id)
        for page_row in pages:
            ordinal = page_row["page_ordinal"]
            if ordinal in done:
                # Already OCR'd; count the artifact if it still exists.
                if Path(page_row["image_path"]).exists():
                    written += 1
                continue
            artifact = ocr_artifact_path(out_dir, document_id, ordinal)
            try:
                result = backend.transcribe(page_row["image_path"], ordinal)
                _serialize_page(result, artifact)
                mark_checkpoint(conn, document_id, OCR_STAGE, ordinal,
                                "done", artifact_path=str(artifact))
                written += 1
            except Exception as exc:  # noqa: BLE001
                mark_checkpoint(conn, document_id, OCR_STAGE, ordinal,
                                "failed", artifact_path=str(exc))
        conn.commit()
    return written
