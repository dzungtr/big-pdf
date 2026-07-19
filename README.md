# big-pdf — `massive-pdf-agent`

> Ingestion + retrieval layer for Vietnamese regulatory PDFs. Turns scanned image-only
> circulars, decrees, and thông tư into a queryable clause graph with hybrid retrieval,
> exposed as a local API + CLI so external LLM agents (Codex, Claude Code, etc.) can
> reason over ground-truth regulatory text.

---

## TL;DR

```bash
git clone https://github.com/dzungtr/big-pdf.git
cd big-pdf

python3 -m venv .venv
.venv/bin/pip install -e .

# Sanity check the CLI and run the test suite.
.venv/bin/python -m massive_pdf --help
.venv/bin/python -m pytest -q
```

A fresh checkout should report **134 tests passing** in a few seconds. If it doesn't, file an issue.

---

## What this is

Vietnamese regulatory PDFs are deeply nested (Điều → Khoản → Điểm), cross-reference each other
heavily, and frequently arrive as image-only scans where the text layer was never embedded.
Manually searching them is slow, brittle, and risky — a misplaced diacritic in a citation
changes legal meaning.

`massive-pdf-agent` is the **corpus brain** behind regulatory reasoning. It does ingestion
and retrieval only — no chat, no verdict engine, no agent loop. A conversational agent
(Codex, Claude Code, anything) drives interrogation and judgment by calling this layer.

The product surface, once slice #6 lands, is two endpoints:

- `retrieve(query, structural_filters)` — hybrid semantic + structural search.
- `get_clause(id)` + traversal — graph walk from a matched node to its context.

---

## Pipeline

The CLI exposes one subcommand per stage. Each stage reads its inputs from the previous
stage's outputs in the SQLite store and is runnable independently:

```
register  →  pages  →  ocr  →  structure  →  cards  →  embed
   │           │        │         │             │         │
 insert     PDF →     page      OCR →       clause   rule card  card →
record    PNGs    images →     graph       extract    embeddings
            text+layout
```

```bash
# Register a PDF, then walk it through the pipeline.
.venv/bin/python -m massive_pdf register Thong-tu-89-BTC.pdf --title "Thông tư 89/2015"
.venv/bin/python -m massive_pdf list

.venv/bin/python -m massive_pdf pages  Thong-tu-89-BTC.pdf --out work/pages
.venv/bin/python -m massive_pdf ocr    <doc_id> --backend stub --out work/ocr
.venv/bin/python -m massive_pdf ocr    <doc_id> --backend vlm  --out work/ocr
.venv/bin/python -m massive_pdf structure <doc_id>       --out work/structure
.venv/bin/python -m massive_pdf cards   <doc_id>         --out work/cards
.venv/bin/python -m massive_pdf embed   <doc_id>         --out work/embed
```

Stages 4–6 (structure, cards, embed) run on the SQLite store and don't need `--out`.
See `python -m massive_pdf <stage> --help` for full options.

**OCR backend.** `ocr --backend` selects the transcription engine:
`stub` (offline placeholder, CI default) or `vlm` (the Baidu Unlimited-OCR client added by
epic [#19](https://github.com/dzungtr/big-pdf/issues/19)). The VLM backend talks to a local
SGLang server and reads its endpoint from `MASSIVE_PDF_VLM_ENDPOINT` (default
`http://127.0.0.1:10000/v1`); the blessed launch recipe lives in
[docs/runbooks/sglang-unlimited-ocr.md](docs/runbooks/sglang-unlimited-ocr.md).

---

## Architecture (the five-minute tour)

The full decision record lives in [docs/adr/](docs/adr/). The short version:

- **ADR 0001 — Local VLM OCR via Baidu Unlimited-OCR.** Vietnamese diacritics are load-bearing;
  cloud OCR sends the corpus off-machine and costs scale. Local VLM (3B MoE, SAM+CLIP encoders)
  on a 16 GB GPU is the sweet spot for ~20–30 min / 1000-page doc at zero marginal cost.
- **ADR 0002 — Corpus brain, not agent.** This system supplies ground truth (original clause
  text + page refs); the calling LLM owns judgment. No chat, no verdict, no session state.
- **ADR 0003 — Clause graph + rule cards + hybrid retrieval.** Pure vector search drops
  exact-structural queries ("Điều 12 Khoản 3", "mẫu 01/GTGT"). The graph carries nodes for
  every clause and edges for every cross-reference; rule cards are non-authoritative glosses
  regenerable from the graph.
- **ADR 0004 — SQLite + `sqlite-vec`.** Corpus scale is ~10⁴–10⁵ clauses total. A single
  inspectable file (`.massive_pdf.sqlite`) is the right shape for MVP; the migration to
  Postgres + pgvector is mechanical when the time comes.

For the domain vocabulary and the slice decomposition rationale, see [CONTEXT.md](CONTEXT.md).

---

## Repo layout

```
big-pdf/
├── pyproject.toml           # Python 3.11+, deps: pymupdf, numpy, pytest
├── README.md                # ← you are here
├── CONTEXT.md               # domain vocabulary + slice rationale
├── docs/adr/                # architecture decision records
├── docs/runbooks/           # operator runbooks (e.g. SGLang Unlimited-OCR bring-up)
├── src/massive_pdf/         # package (the agent)
│   ├── __main__.py          # CLI entry point
│   ├── store.py             # SQLite schema + helpers
│   ├── ingest/              # pages + ocr + vlm backends (vlm.py = UnlimitedOcrBackend)
│   ├── structure/           # clause-graph parser + stage
│   └── retrieval/           # rule cards + encoder + embed stage
└── tests/                   # one test module per stage (134 tests)
```

---

## Status

Built and merged:

| Stage | Slice | Status |
|---|---|---|
| Project scaffold + SQLite store | [#2](https://github.com/dzungtr/big-pdf/issues/2) | ✅ merged ([PR #9](https://github.com/dzungtr/big-pdf/pull/9)) |
| OCR stage (local Unlimited-OCR) | [#3](https://github.com/dzungtr/big-pdf/issues/3) | ✅ merged ([PR #10](https://github.com/dzungtr/big-pdf/pull/10)) |
| Clause-graph structure stage | [#4](https://github.com/dzungtr/big-pdf/issues/4) | ✅ merged ([PR #11](https://github.com/dzungtr/big-pdf/pull/11)) |
| Rule cards + embeddings | [#5](https://github.com/dzungtr/big-pdf/issues/5) | ✅ merged ([PR #12](https://github.com/dzungtr/big-pdf/pull/12)) |

Human-owned (`ready-for-human`):

| Stage | Slice | Why human |
|---|---|---|
| Hybrid retrieval + graph traversal API | [#6](https://github.com/dzungtr/big-pdf/issues/6) | API surface + golden-query retrieval suite are judgment calls |
| Auto-ingest external references | [#7](https://github.com/dzungtr/big-pdf/issues/7) | Vietnamese legal sources are unofficial/fragmented; version-currency is the riskiest part |

The parent epic is [issue #1](https://github.com/dzungtr/big-pdf/issues/1) and tracks the
live dashboard of every slice.

### VLM OcrBackend initiative — epic [#19](https://github.com/dzungtr/big-pdf/issues/19)

Wires a real VLM-backed `OcrBackend` (Baidu Unlimited-OCR via SGLang) behind the existing
`OcrBackend` protocol so `massive-pdf ocr --backend vlm` produces actual Vietnamese text
instead of `StubBackend` placeholders.

| Slice | Issue | Status |
|---|---|---|
| SGLang bring-up + runbook | [#20](https://github.com/dzungtr/big-pdf/issues/20) | ✅ merged ([PR #24](https://github.com/dzungtr/big-pdf/pull/24)) |
| `UnlimitedOcrBackend` HTTP client | [#21](https://github.com/dzungtr/big-pdf/issues/21) | ✅ merged ([PR #25](https://github.com/dzungtr/big-pdf/pull/25)) |
| CLI wiring (`--backend {stub,vlm}`) | [#22](https://github.com/dzungtr/big-pdf/issues/22) | ✅ merged ([PR #26](https://github.com/dzungtr/big-pdf/pull/26)) |
| Request-body contract fix | [#28](https://github.com/dzungtr/big-pdf/issues/28) | ✅ merged ([PR #31](https://github.com/dzungtr/big-pdf/pull/31)) |
| Runbook RoPE-JIT correction | [#29](https://github.com/dzungtr/big-pdf/issues/29) | ✅ merged ([PR #30](https://github.com/dzungtr/big-pdf/pull/30)) |

Acceptance run + ADR-0001 "Measured results" capture ([#23](https://github.com/dzungtr/big-pdf/issues/23))
is **blocked** on a GCC 15 / RoPE JIT toolchain failure (`ready-for-human`); the request-body
fix above is in, but the live measurement pass can't complete until the toolchain is resolved.
See epic [#19](https://github.com/dzungtr/big-pdf/issues/19) for the diagnosis and options.

---

## Development

**Run tests:**

```bash
.venv/bin/python -m pytest -q
```

**Add a new stage.** Each stage is one module under `src/massive_pdf/<stage>/` exposing
`run_<stage>_stage(...)`, one CLI subcommand in `__main__.py`, and one test module under
`tests/`. The SQLite store (`store.py`) is the contract between stages — read prior-stage
rows, write new-stage rows, no direct file handoff between stages.

**Conventions.** Python 3.11+, src-layout, `pathlib.Path` for filesystem, SQLite for state,
type hints on every public function. PRs squash-merge into `main`; branch name
`feat/slice-N-<slug>` or `feat/<short-slug>` for doc/infra work. One logical change per PR.

**Worktrees.** Per project convention, every PR is developed in
`<repo-root>/.worktrees/<branch>/`, not in the main checkout. The main checkout stays clean.

---

## License

No license has been chosen yet. Treat the source as proprietary unless a `LICENSE` file says
otherwise.
