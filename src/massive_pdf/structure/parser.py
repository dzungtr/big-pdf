"""Vietnamese clause parser: OCR text -> clause graph (Điều/Khoản/Điểm).

This is a deliberately conservative, regex-driven parser. It handles the
canonical Thông-tư shapes that show up in Circular 89 and similar MoF
instruments:

    Điều 1. Phạm vi điều chỉnh.
    Điều 2. Đối tượng áp dụng
        1. ...
        2. ...
        Khoản 2 Điều này: ...   (referenced in body)
    a) ...
    b) ...

Hierarchy rule: a new `Điều N` starts a top-level article; the most
recent `Khoản` (if any) belongs to the most recent `Điều`; a bare
`letter)` is a `Điểm` whose parent is the most recent `Khoản` (or `Điều`
if no Khoản is open).

What this parser is NOT (yet):
  - It does not handle tables, schedules, or appendices.
  - It does not repair OCR artefacts (merged headers, missing diacritics).
    Those land in `errata` per slice 2; the parser is honest about
    what it sees.
  - It does not attempt fuzzy matching for the document preamble or
    signature blocks. Those live "above" the first `Điều` and are
    captured implicitly as the document's leading body if needed.

The output is a `ClauseGraph`: ordered list of `ParsedClause`s with
canonical citations and page spans, ready to be upserted into the
`clauses` table.
"""
from __future__ import annotations
import re
from dataclasses import dataclass, field


# --- Vietnamese header regexes -------------------------------------------------
# Anchored to line start (after optional whitespace) so they don't trigger
# on the word "Điều" appearing in body prose.
#
# Numbered Điều/Khoản accept ASCII digits and full-width digits (０-９) because
# OCR sometimes emits the latter.

_DIEU_RE = re.compile(
    r"(?m)^[ \t]*Điều[ \t]+(\d+|[０-９]+)[ \t]*[\.:\-\)]"
)
_KHOAN_RE = re.compile(
    r"(?m)^[ \t]+Khoản[ \t]+(\d+|[０-９]+)[ \t]*[\.:\-\)]"
    # Khoản headers are commonly indented under Điều body
)
# Điểm: either "Điểm a)" or a bare "a)" / "a." at line start. Vietnamese
# letter alphabet omits several Latin letters; we accept any a-z + diacritics.
_DIEM_RE = re.compile(
    r"(?m)^[ \t]+(?:Điểm[ \t]+)?([a-zăâđêôơưăằẳẵặắấầẩẫậéèẻẽẹếềểễệíìỉĩịóòỏõọốồổỗộớờởỡợúùủũụứừửữựýỳỷỹỵ])[ \t]*[\)\.]"
)


def _to_ascii_digit(s: str) -> str:
    """Convert full-width digit to ASCII; pass-through otherwise."""
    fw = "０１２３４５６７８９"
    if s and s in fw:
        return str(fw.index(s))
    return s


@dataclass
class ParsedClause:
    """One node in the clause graph, ready to be persisted."""
    kind: str                # 'dieu' | 'khoan' | 'diem'
    number: str              # '1', '2', 'a', 'b' (canonical)
    citation: str            # 'Điều 1', 'Khoản 2 Điều 1', 'Điểm a Khoản 2 Điều 1'
    page_start: int          # 1-indexed
    page_end: int            # 1-indexed, inclusive
    body: str                # raw text from this header to the next header
    parent_number: tuple[str, ...] | None  # tuple of ancestor numbers (no kind)
    ord: int                 # document-wide ordering, set by `parse_pages`

    def to_db_row(self) -> dict:
        return {
            "kind": self.kind,
            "number": self.number,
            "citation": self.citation,
            "page_start": self.page_start,
            "page_end": self.page_end,
            "body": self.body,
            "parent_number": list(self.parent_number) if self.parent_number else None,
            "ord": self.ord,
        }


@dataclass
class ClauseGraph:
    """A parsed clause graph for one document."""
    document_id: int
    page_count: int
    clauses: list[ParsedClause] = field(default_factory=list)

    def dieu_numbers(self) -> list[str]:
        return [c.number for c in self.clauses if c.kind == "dieu"]

    def clauses_by_citation(self) -> dict[str, ParsedClause]:
        return {c.citation: c for c in self.clauses}


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def _char_to_page(char_offset: int, page_offsets: list[tuple[int, int]]) -> int:
    """Map a character offset (in concatenated text) to its 1-indexed page.

    `page_offsets` is a list of `(page_ordinal, char_start)` plus a sentinel
    `(last_page_ordinal + 1, total_len)`. Offsets at or beyond the sentinel
    clamp to the last real page.
    """
    last_real_page = page_offsets[-2][0] if len(page_offsets) >= 2 else page_offsets[-1][0]
    for ordinal, start in page_offsets:
        if start > char_offset:
            return ordinal - 1  # previous page
    return last_real_page


def _slice_body(text: str, page_offsets: list[tuple[int, int]],
                start: int, end: int) -> tuple[int, int, str]:
    """Return (page_start, page_end, body_text) for a slice of the concatenated
    text. page_end is the page of the *next* header, minus one (so the
    span is inclusive on both ends).
    """
    body_text = text[start:end].strip()
    next_ordinal = _char_to_page(end, page_offsets)
    this_ordinal = _char_to_page(start, page_offsets)
    # End is the page *before* the next header's page (or last page if no more).
    page_end = max(this_ordinal, next_ordinal - 1) if next_ordinal > this_ordinal else this_ordinal
    # Also keep the last page when end is at document end.
    if end >= page_offsets[-1][1] - 1 and next_ordinal == page_offsets[-1][0]:
        page_end = next_ordinal
    return this_ordinal, page_end, body_text


def _cite(kind: str, number: str, parents: dict[str, str]) -> str:
    """Build the canonical Vietnamese citation string for a clause."""
    if kind == "dieu":
        return f"Điều {number}"
    if kind == "khoan":
        d = parents.get("dieu", "?")
        return f"Khoản {number} Điều {d}"
    if kind == "diem":
        d = parents.get("dieu", "?")
        k = parents.get("khoan")
        if k is not None:
            return f"Điểm {number} Khoản {k} Điều {d}"
        return f"Điểm {number} Điều {d}"
    raise ValueError(f"unknown kind: {kind}")


def parse_pages(document_id: int, pages: list[tuple[int, str]]) -> ClauseGraph:
    """Parse a list of `(page_ordinal, raw_text)` into a clause graph.

    `page_ordinal` is 1-indexed; `raw_text` is the OCR text for that page.
    Pages may be provided out of order; they are sorted internally.
    """
    if not pages:
        return ClauseGraph(document_id=document_id, page_count=0)

    pages_sorted = sorted(pages, key=lambda p: p[0])
    page_count = pages_sorted[-1][0]

    # Build concatenated text + per-page offset table.
    parts: list[str] = []
    page_offsets: list[tuple[int, int]] = []  # (ordinal, char_start)
    cursor = 0
    for ordinal, text in pages_sorted:
        page_offsets.append((ordinal, cursor))
        parts.append(text)
        cursor += len(text) + 1  # +1 for the joining newline
        parts.append("\n")
    text = "".join(parts)
    last_page_ordinal = pages_sorted[-1][0]
    page_offsets.append((last_page_ordinal + 1, len(text)))  # sentinel

    # Find all header matches.
    headers: list[tuple[int, str, str]] = []  # (offset, kind, number)
    for m in _DIEU_RE.finditer(text):
        headers.append((m.start(), "dieu", _to_ascii_digit(m.group(1))))
    for m in _KHOAN_RE.finditer(text):
        headers.append((m.start(), "khoan", _to_ascii_digit(m.group(1))))
    for m in _DIEM_RE.finditer(text):
        headers.append((m.start(), "diem", m.group(1)))

    # Sort by offset; when two headers start at the same offset, prefer
    # the more specific (deeper) kind. E.g. "Khoản 2 Điều 3." on the same
    # line as "Điều 3." should be parsed as one Điều (no Khoản in that
    # case — same offset means it's a header that mentions a Khoản in
    # its body, not a new clause).
    kind_order = {"dieu": 0, "khoan": 1, "diem": 2}
    headers.sort(key=lambda h: (h[0], kind_order[h[1]]))

    # Dedup headers at the same offset, keeping only the topmost kind.
    deduped: list[tuple[int, str, str]] = []
    for h in headers:
        if deduped and deduped[-1][0] == h[0]:
            # Same position: keep the more "general" kind (lower kind_order).
            if kind_order[h[1]] < kind_order[deduped[-1][1]]:
                deduped[-1] = h
            continue
        deduped.append(h)

    # Walk the headers, building a hierarchy via running parents.
    clauses: list[ParsedClause] = []
    current_dieu: str | None = None
    current_khoan: str | None = None
    parents: dict[str, str] = {}

    last_offset = 0
    last_kind: str | None = None
    last_number: str | None = None

    # Sentinel end position: end of concatenated text.
    text_end = len(text)

    for offset, kind, number in deduped:
        # Close the previous clause: its body ends at this header's offset.
        if last_kind is not None:
            ps, pe, body = _slice_body(text, page_offsets, last_offset, offset)
            # Set ord on the previous clause (later overwritten once list built).
            clauses.append(ParsedClause(
                kind=last_kind,
                number=last_number,  # type: ignore[arg-type]
                citation=_cite(last_kind, last_number, parents_at(last_kind, parents)),  # type: ignore[arg-type]
                page_start=ps,
                page_end=pe,
                body=body,
                parent_number=_parent_tuple(last_kind, parents),  # type: ignore[arg-type]
                ord=0,  # filled below
            ))

        # Update running parents.
        if kind == "dieu":
            current_dieu = number
            current_khoan = None  # Khoản doesn't survive across Điều boundaries
        elif kind == "khoan":
            current_khoan = number
        # Điểm doesn't change parents.
        parents = {
            "dieu": current_dieu,
            "khoan": current_khoan,
        }

        last_offset = offset
        last_kind = kind
        last_number = number

    # Close the final clause (or sentinel document body).
    if last_kind is not None:
        ps, pe, body = _slice_body(text, page_offsets, last_offset, text_end)
        clauses.append(ParsedClause(
            kind=last_kind,
            number=last_number,  # type: ignore[arg-type]
            citation=_cite(last_kind, last_number, parents),  # type: ignore[arg-type]
            page_start=ps,
            page_end=pe,
            body=body,
            parent_number=_parent_tuple(last_kind, parents),  # type: ignore[arg-type]
            ord=0,
        ))

    # Assign document-wide ordering.
    for i, c in enumerate(clauses, start=1):
        c.ord = i

    return ClauseGraph(document_id=document_id, page_count=page_count, clauses=clauses)


def _parent_tuple(kind: str, parents: dict[str, str]) -> tuple[str, ...] | None:
    if kind == "dieu":
        return None
    if kind == "khoan":
        return (parents["dieu"],) if parents.get("dieu") else None
    if kind == "diem":
        d = parents.get("dieu")
        k = parents.get("khoan")
        if k is not None and d is not None:
            return (d, k)
        if d is not None:
            return (d,)
        return None
    return None


def parents_at(kind: str, parents: dict[str, str]) -> dict[str, str]:
    """Snapshot of running parents for citation generation."""
    return dict(parents)
