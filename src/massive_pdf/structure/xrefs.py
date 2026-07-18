"""Cross-reference extractor (Vietnamese legal patterns).

Issue #4 deliverable: parse out pointers from a clause's body to:

  - Internal clauses: `Điều N`, `khoản N Điều M`, `Khoản N Điều này`,
    `Khoản N Khoản M Điều K`, `điểm a khoản 2 Điều 3`, etc.
  - External instruments: `Thông tư số N/YYYY/TT-BTC`, `Luật số ...`,
    `Nghị định số ...`, `Quyết định số ...`, ...

External refs are recorded as dangling (dst_clause_id NULL, target_document_id
NULL) with the parsed citation string preserved for later auto-ingest
(slice #7).

The extractor is regex-driven and conservative — false positives on
external refs are surfaced as dangling nodes (cheap to dedupe later),
while false positives on internal refs would create bogus edges, so
internal patterns are tighter.

Returned `ParsedReference` records are then mapped onto `clause_references`
rows in the store: the stage resolves internal pointers against the
already-parsed `clauses` table.
"""
from __future__ import annotations
import re
from dataclasses import dataclass


# --- Internal reference patterns ---------------------------------------------
# These all resolve to clauses *within the same document* unless the
# surrounding text names a different instrument (handled by the external
# patterns below).
#
# Each pattern is `(<regex>, <builder>)` where the builder takes the match
# and returns (target_citation, raw_text).

# "Điều N" alone (most common internal ref).
# Match in any position; the parser's line-anchored header regex prevents
# double-counting at header positions (it consumes those matches first
# when assembling the graph). In body text we want to catch "Điều 7."
# at sentence end, so no negative-anchor.
_RE_DIEU = re.compile(r"(?<![a-zA-ZăâđêôơưĂÂĐÊÔƠƯ])(Điều|điều)\s+(\d+)")
# "khoản N Điều M" — full path
_RE_KHOAN_DIEU = re.compile(
    r"(Khoản|khoản)\s+(\d+)\s+(Điều|điều)\s+(\d+)(?!\s*[\.:\-\)])"
)
# "khoản N Điều này" — self-reference inside the same Điều
_RE_KHOAN_NAY = re.compile(r"(Khoản|khoản)\s+(\d+)\s+(Điều này|điều này)")
# "điểm a khoản 2 Điều 3" — full leaf path
_RE_DIEM_FULL = re.compile(
    r"(Điểm|điểm)\s+([a-zăâđêôơưà-ỹ])\s+(Khoản|khoản)\s+(\d+)\s+(Điều|điều)\s+(\d+)(?!\s*[\.:\-\)])"
)
# "điểm a khoản N Điều này" — self-reference leaf
_RE_DIEM_KHOAN_NAY = re.compile(
    r"(Điểm|điểm)\s+([a-zăâđêôơưà-ỹ])\s+(Khoản|khoản)\s+(\d+)\s+(Điều này|điều này)"
)
# "điểm a Điều N" — leaf without Khoản (when a Điều has no Khoản)
_RE_DIEM_DIEU = re.compile(
    r"(Điểm|điểm)\s+([a-zăâđêôơưà-ỹ])\s+(Điều|điều)\s+(\d+)(?!\s*[\.:\-\)])"
)


# --- External reference patterns ---------------------------------------------
# Vietnamese instruments are cited by type + number. We capture the type
# phrase and the citation string; the canonical form goes into the
# `target_citation` column.

_EXTERNAL_TYPES = [
    # Order matters: more specific patterns first so they win on overlap.
    "Thông tư", "Nghị định", "Quyết định", "Luật", "Pháp lệnh",
    "Nghị quyết", "Chỉ thị", "Thông báo", "Tờ trình",
]

# Build one big regex: e.g.  (Thông tư|Nghị định|...) số ...
_RE_EXTERNAL = re.compile(
    r"(" + "|".join(_EXTERNAL_TYPES) + r")"
    r"\s+số\s+"
    r"([0-9０-９]+(?:[/-][0-9０-９A-Za-zĐƠƯ]+)*)"
)


@dataclass
class ParsedReference:
    """A cross-reference detected inside a clause's body."""
    kind: str            # 'internal' | 'external'
    raw_text: str        # the matched substring
    target_citation: str # parsed target citation (e.g. 'Điều 5', 'Thông tư số 39/2014/TT-BTC')

    def dedup_key(self) -> tuple[str, str]:
        return (self.raw_text.lower(), self.target_citation.lower())


def _find_all(pattern: re.Pattern, text: str, builder):
    """Find all non-overlapping matches and yield ParsedReference objects."""
    out = []
    for m in pattern.finditer(text):
        ref = builder(m)
        if ref is not None:
            out.append(ref)
    return out


def _build_dieu(m: re.Match) -> ParsedReference:
    n = m.group(2)
    return ParsedReference(kind="internal", raw_text=m.group(0).strip(),
                           target_citation=f"Điều {n}")


def _build_khoan_dieu(m: re.Match) -> ParsedReference:
    k, d = m.group(2), m.group(4)
    return ParsedReference(kind="internal", raw_text=m.group(0).strip(),
                           target_citation=f"Khoản {k} Điều {d}")


def _build_khoan_nay(m: re.Match) -> ParsedReference:
    k = m.group(2)
    return ParsedReference(kind="internal_self", raw_text=m.group(0).strip(),
                           target_citation=f"Khoản {k} Điều này")


def _build_diem_full(m: re.Match) -> ParsedReference:
    letter, k, d = m.group(2), m.group(4), m.group(6)
    return ParsedReference(kind="internal", raw_text=m.group(0).strip(),
                           target_citation=f"Điểm {letter} Khoản {k} Điều {d}")


def _build_diem_khoan_nay(m: re.Match) -> ParsedReference:
    letter, k = m.group(2), m.group(4)
    return ParsedReference(kind="internal_self", raw_text=m.group(0).strip(),
                           target_citation=f"Điểm {letter} Khoản {k} Điều này")


def _build_diem_dieu(m: re.Match) -> ParsedReference:
    letter, d = m.group(2), m.group(4)
    return ParsedReference(kind="internal", raw_text=m.group(0).strip(),
                           target_citation=f"Điểm {letter} Điều {d}")


def _build_external(m: re.Match) -> ParsedReference:
    typ = m.group(1)
    cite = m.group(2).replace("０", "0").replace("１", "1").replace("２", "2") \
                     .replace("３", "3").replace("４", "4").replace("５", "5") \
                     .replace("６", "6").replace("７", "7").replace("８", "8") \
                     .replace("９", "9")
    return ParsedReference(kind="external", raw_text=m.group(0).strip(),
                           target_citation=f"{typ} số {cite}")


def extract_references(body: str) -> list[ParsedReference]:
    """Extract all cross-references from a clause body.

    Returns a list of `ParsedReference` records in document order.
    Self-references ("Khoản N Điều này") are tagged `internal_self` so
    the stage can resolve them against the clause's own ancestor Điều.
    """
    refs: list[ParsedReference] = []
    refs.extend(_find_all(_RE_DIEM_FULL, body, _build_diem_full))
    refs.extend(_find_all(_RE_DIEM_KHOAN_NAY, body, _build_diem_khoan_nay))
    refs.extend(_find_all(_RE_KHOAN_DIEU, body, _build_khoan_dieu))
    refs.extend(_find_all(_RE_KHOAN_NAY, body, _build_khoan_nay))
    refs.extend(_find_all(_RE_DIEM_DIEU, body, _build_diem_dieu))
    refs.extend(_find_all(_RE_DIEU, body, _build_dieu))
    refs.extend(_find_all(_RE_EXTERNAL, body, _build_external))

    # Sort by first occurrence position, dedup by (raw_text, target_citation).
    refs.sort(key=lambda r: body.find(r.raw_text))
    seen: set[tuple[str, str]] = set()
    deduped: list[ParsedReference] = []
    for r in refs:
        key = r.dedup_key()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(r)
    return deduped
