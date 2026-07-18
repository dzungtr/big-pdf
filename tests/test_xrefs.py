"""Tests for the Vietnamese cross-reference extractor."""
from massive_pdf.structure.xrefs import extract_references


def test_empty_body():
    assert extract_references("") == []


def test_internal_dieu_only():
    refs = extract_references("Căn cứ Điều 5 của Thông tư này.")
    assert len(refs) == 1
    assert refs[0].kind == "internal"
    assert refs[0].target_citation == "Điều 5"
    assert refs[0].raw_text == "Điều 5"


def test_internal_khoan_dieu():
    refs = extract_references("theo khoản 2 Điều 3")
    # Both the specific Khoản-level and the general Điều-level refs are
    # captured (dedup removes only identical raw+target pairs).
    cites = {r.target_citation for r in refs}
    assert "Khoản 2 Điều 3" in cites
    assert "Điều 3" in cites
    assert all(r.kind == "internal" for r in refs)


def test_internal_self_khoan_dieu_nay():
    refs = extract_references("khoản 2 Điều này")
    assert len(refs) == 1
    assert refs[0].kind == "internal_self"
    assert refs[0].target_citation == "Khoản 2 Điều này"


def test_internal_diem_full():
    refs = extract_references("điểm a khoản 2 Điều 10")
    # Order of returned refs: most specific first.
    cite_set = {r.target_citation for r in refs}
    assert "Điểm a Khoản 2 Điều 10" in cite_set
    assert "Khoản 2 Điều 10" in cite_set
    assert "Điều 10" in cite_set


def test_internal_self_diem_khoan_nay():
    refs = extract_references("điểm b khoản 1 Điều này")
    kinds = {r.kind for r in refs}
    assert "internal_self" in kinds
    cites = {r.target_citation for r in refs}
    assert "Điểm b Khoản 1 Điều này" in cites
    assert "Khoản 1 Điều này" in cites


def test_internal_diem_dieu_no_khoan():
    refs = extract_references("điểm a Điều 7")
    cite_set = {r.target_citation for r in refs}
    assert "Điểm a Điều 7" in cite_set
    assert "Điều 7" in cite_set


def test_external_thong_tu():
    refs = extract_references("Theo Thông tư số 39/2014/TT-BTC thì ...")
    assert any(r.kind == "external" and
               r.target_citation == "Thông tư số 39/2014/TT-BTC"
               for r in refs)


def test_external_nghi_dinh():
    refs = extract_references("Căn cứ Nghị định số 12/2020/NĐ-CP.")
    assert any(r.target_citation == "Nghị định số 12/2020/NĐ-CP" for r in refs)


def test_external_quyet_dinh():
    refs = extract_references("Căn cứ Quyết định số 15/QĐ-BTC.")
    assert any(r.target_citation == "Quyết định số 15/QĐ-BTC" for r in refs)


def test_external_luat():
    refs = extract_references("Căn cứ Luật số 38/2019/QH14.")
    assert any(r.target_citation == "Luật số 38/2019/QH14" for r in refs)


def test_mixed_internal_and_external():
    body = ("Căn cứ khoản 1 Điều 5 Luật Thương mại, "
            "Thông tư số 39/2014/TT-BTC quy định Điều 7.")
    refs = extract_references(body)
    cites = [r.target_citation for r in refs]
    # Internal: "Khoản 1 Điều 5", "Điều 7"
    assert "Khoản 1 Điều 5" in cites
    assert "Điều 7" in cites
    # External: "Thông tư số 39/2014/TT-BTC"
    assert "Thông tư số 39/2014/TT-BTC" in cites


def test_dedup_by_raw_and_citation():
    body = "Điều 5 và Điều 5 cùng nội dung."
    refs = extract_references(body)
    # Two occurrences of "Điều 5" should dedup to one record.
    targets = [r for r in refs if r.target_citation == "Điều 5"]
    assert len(targets) == 1


def test_no_header_lookalikes_in_body():
    """The 'Điều' inside regular prose without a number must not trigger."""
    refs = extract_references("Theo điều khoản chung thì ...")
    # No digit following "điều" → no match.
    assert refs == []


def test_dieu_after_letter_does_not_match():
    """The lookbehind ensures 'Điều' doesn't fire in the middle of words
    like 'Điều' as part of a longer identifier or in 'điều khoản'.
    """
    # 'điều khoản' has no digit after 'điều' → no ref.
    assert extract_references("theo điều khoản này") == []
    # A real Điều reference still works.
    refs = extract_references("Căn cứ điều 12.")
    assert any(r.target_citation == "Điều 12" for r in refs)
