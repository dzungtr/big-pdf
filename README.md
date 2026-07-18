# big-pdf — massive-pdf-agent

An ingestion + retrieval layer for massive scanned Vietnamese regulatory documents (Ministry of Finance circulars). Turns image-only PDFs into a queryable clause graph with hybrid retrieval, exposed as a local API + CLI so external LLM agents (Codex, Claude Code) can interrogate and reason over ground-truth regulatory text.

## Design

- **PRD + coordination home:** [issue #1](https://github.com/dzungtr/big-pdf/issues/1)
- **Domain model & glossary:** [CONTEXT.md](CONTEXT.md)
- **Architecture decisions:** [docs/adr/](docs/adr/)
  - [0001 — Local VLM OCR via Baidu Unlimited-OCR](docs/adr/0001-local-vlm-ocr-unlimited-ocr.md)
  - [0002 — Corpus brain, not agent](docs/adr/0002-corpus-brain-not-agent.md)
  - [0003 — Clause graph + rule cards + hybrid retrieval](docs/adr/0003-clause-graph-rule-cards-hybrid-retrieval.md)
  - [0004 — SQLite + sqlite-vec store](docs/adr/0004-sqlite-store.md)

## Slices (dependency order)

1. [#2 — project scaffold + SQLite store](https://github.com/dzungtr/big-pdf/issues/2)
2. [#3 — OCR stage (local Unlimited-OCR)](https://github.com/dzungtr/big-pdf/issues/3)
3. [#4 — clause-graph structure stage](https://github.com/dzungtr/big-pdf/issues/4)
4. [#5 — rule cards + embeddings](https://github.com/dzungtr/big-pdf/issues/5)
5. [#6 — hybrid retrieval + graph traversal API](https://github.com/dzungtr/big-pdf/issues/6)
6. [#7 — auto-ingest external references](https://github.com/dzungtr/big-pdf/issues/7) *(sequenced last; ready-for-human)*

## Status

Design complete. Implementation dispatched as slices above.
