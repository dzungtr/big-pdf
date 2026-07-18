"""Tests for the Vietnamese clause parser (Điều/Khoản/Điểm hierarchy)."""
from massive_pdf.structure.parser import parse_pages, ParsedClause


def test_empty_input_returns_empty_graph():
    g = parse_pages(document_id=1, pages=[])
    assert g.document_id == 1
    assert g.page_count == 0
    assert g.clauses == []


def test_single_dieu_single_page():
    g = parse_pages(document_id=1, pages=[(1, "Điều 1. Phạm vi điều chỉnh.\nNội dung.")])
    assert g.page_count == 1
    assert len(g.clauses) == 1
    c = g.clauses[0]
    assert c.kind == "dieu"
    assert c.number == "1"
    assert c.citation == "Điều 1"
    assert c.parent_number is None
    assert c.page_start == 1
    assert c.page_end == 1
    assert "Phạm vi" in c.body


def test_dieu_numbering_sequential_across_pages():
    g = parse_pages(document_id=1, pages=[
        (1, "Điều 1. A.\nĐiều 2. B.\nĐiều 3. C."),
    ])
    nums = g.dieu_numbers()
    assert nums == ["1", "2", "3"]
    assert [c.ord for c in g.clauses] == [1, 2, 3]


def test_khoan_under_dieu():
    g = parse_pages(document_id=1, pages=[(
        1,
        "Điều 1. Phạm vi.\n"
        "Nội dung mở đầu.\n"
        "Khoản 1. Quy định chi tiết.\n"
        "Khoản 2. Cụ thể hơn.\n"
    )])
    assert len(g.clauses) == 3
    kinds = [c.kind for c in g.clauses]
    assert kinds == ["dieu", "khoan", "khoan"]
    assert g.clauses[1].citation == "Khoản 1 Điều 1"
    assert g.clauses[1].parent_number == ("1",)
    assert g.clauses[2].citation == "Khoản 2 Điều 1"
    assert g.clauses[2].parent_number == ("1",)


def test_diem_under_dieu_when_no_khoan():
    g = parse_pages(document_id=1, pages=[(
        1,
        "Điều 1. Các hành vi:\n"
        "    a) Hành vi một;\n"
        "    b) Hành vi hai;\n"
        "    c) Hành vi ba.\n"
    )])
    # Expect 1 Điều + 3 Điểm.
    assert len(g.clauses) == 4
    diem = [c for c in g.clauses if c.kind == "diem"]
    assert [c.citation for c in diem] == [
        "Điểm a Điều 1", "Điểm b Điều 1", "Điểm c Điều 1",
    ]
    for c in diem:
        assert c.parent_number == ("1",)


def test_diem_with_diem_prefix():
    g = parse_pages(document_id=1, pages=[(
        1,
        "Điều 1. A.\n"
        "    Điểm a. Một;\n"
        "    Điểm b. Hai.\n"
    )])
    assert len(g.clauses) == 3
    assert [c.citation for c in g.clauses] == [
        "Điều 1", "Điểm a Điều 1", "Điểm b Điều 1",
    ]


def test_page_spans_across_pages():
    g = parse_pages(document_id=1, pages=[
        (1, "Điều 1. Bắt đầu tại đây."),
        (2, "Nội dung tiếp theo trên trang 2.\nĐiều 2. Bắt đầu mới."),
        (3, "Trang 3 nội dung Điều 2."),
    ])
    assert g.page_count == 3
    # Điều 1 starts on page 1 and its body continues into page 2
    # (page 2's preamble "Nội dung tiếp theo..." is Điều 1 content).
    assert g.clauses[0].page_start == 1
    assert g.clauses[0].page_end == 2
    assert g.clauses[1].page_start == 2
    assert g.clauses[1].page_end == 3  # extends until end (no more headers)


def test_khoan_does_not_leak_across_dieu():
    g = parse_pages(document_id=1, pages=[(
        1,
        "Điều 1.\nKhoản 1. Trong Điều 1.\nĐiều 2.\nKhoản 1. Trong Điều 2.",
    )])
    cites = [c.citation for c in g.clauses]
    assert cites == [
        "Điều 1", "Khoản 1 Điều 1", "Điều 2", "Khoản 1 Điều 2",
    ]


def test_full_width_digits_normalized():
    g = parse_pages(document_id=1, pages=[(1, "Điều １. Test.")])
    assert g.clauses[0].number == "1"
    assert g.clauses[0].citation == "Điều 1"


def test_page_ordinals_unsorted_input():
    g = parse_pages(document_id=1, pages=[
        (3, "Trang 3 tiếp nối.\nĐiều 2. Trên trang 3."),
        (1, "Điều 1. Trên trang 1."),
    ])
    # Điều 1 should come first (ord=1), then Điều 2 (ord=2).
    assert [c.ord for c in g.clauses] == [1, 2]
    assert g.clauses[0].page_start == 1
    assert g.clauses[1].page_start == 3


def test_no_headers_no_clauses():
    g = parse_pages(document_id=1, pages=[(1, "Chỉ có văn bản thường không có tiêu đề.")])
    assert g.clauses == []
    assert g.page_count == 1


def test_clause_graph_lookup():
    g = parse_pages(document_id=1, pages=[(
        1, "Điều 1. A.\nĐiều 2. B.",
    )])
    lookup = g.clauses_by_citation()
    assert "Điều 1" in lookup
    assert "Điều 2" in lookup
    assert lookup["Điều 2"].number == "2"
