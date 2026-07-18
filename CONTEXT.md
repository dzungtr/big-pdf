# CONTEXT — massive-pdf-agent

An ingestion + retrieval layer for massive scanned Vietnamese regulatory documents (starting with Ministry of Finance circulars, e.g. Thong-tu-89-BTC.pdf — 839 scanned pages, no text layer). Exposed as a local API + CLI so that external agents (Codex, Claude Code) do the conversational reasoning; this system owns the corpus, the OCR, the structure, and retrieval.

## What this system is / is not

- **IS**: OCR, clause-graph parsing, rule-card derivation, embedding + retrieval, citation resolution. The corpus brain.
- **IS NOT**: a chat agent. No interrogation loop, no verdict engine, no web UI. External agents (Codex/Claude Code) drive interrogation and verdicts by calling retrieval.

## Glossary

- **Regulatory corpus** — the library of ingested Vietnamese finance/tax regulations, same category (MoF family), sharing structure (Điều/Khoản/Điểm).
- **Document** — one ingested regulation: canonical citation (e.g. "Thông tư 89/2026/TT-BTC"), page images, extracted structure.
- **Clause** — the atomic citable unit (Điều/Khoản/Điểm): original Vietnamese text + page refs.
- **Clause graph** — parsed native hierarchy of a Document with citations + cross-references.
- **Rule card** — derived, non-authoritative distillate of a clause/cluster: gloss, bound actor, topic tags, required-facts checklist, penalty if any. Regenerable. Retrieval and external-agent interrogation use cards; grounding always cites original Clause text.
- **OCR pass** — local transcription via Baidu Unlimited-OCR (3B MoE VLM, runs on 16GB VRAM, SGLang/vLLM).
- **Errata** — per-document hand corrections to OCR, applied without forking the pipeline.
- **Retrieval API** — the external surface: query → ranked clauses + rule cards with citations, in Vietnamese, designed for consumption by an LLM agent.

## Decisions

| # | Decision | Choice | Rationale |
|---|----------|--------|-----------|
| 1 | Use case | Regulatory reasoning: explain rules, test scenarios, classify actions | Application-close, not doc search |
| 2 | Language | Vietnamese throughout; cite original text | Legal fidelity |
| 3 | Corpus | Growing MoF library, same category | Generic pipeline; Circular 89 is stress-test |
| 4 | Verdict engine | None here — external LLM agent judges over retrieved clauses | This layer supplies ground truth |
| 5 | Missing facts | Handled by the calling agent (interrogation), fed by rule-card required-facts | Rule cards expose the checklist; agent asks |
| 6 | OCR | Local Baidu Unlimited-OCR | Privacy, zero marginal cost, strong doc parsing |
| 7 | Ingest structure | Clause graph + rule cards | Sharp retrieval + interrogation-ready checklists |
| 8 | Interface | Local API + CLI; external agents (Codex/Claude) consume it | Don't rebuild the agent; own the corpus brain |

## Measured results

(Filled at initiative close — OCR throughput, Vietnamese diacritic accuracy, ingestion wall-clock, retrieval precision.)

## Glossary (cont.)

- **Cross-reference** — a pointer from one Clause to another, internal ("khoản 2 Điều này", "Điều 5") or external ("theo Thông tư số 39/..."). Extracted at ingest; external refs resolve when the target Document exists in the corpus, else dangle.
- **Retrieval contract** — hybrid: `retrieve(query, filters)` (semantic + structural: document, Điều/Khoản number, actor, topic, penalty) **plus** full graph traversal (`get_clause(id)`, `children/parents`, `references/referenced-by`).

## Decisions (cont.)

| # | Decision | Choice | Rationale |
|---|----------|--------|-----------|
| 9 | Retrieval | Hybrid retrieval + full graph traversal | Exact-reference legal queries demand structure; cross-doc refs need the graph |

## Decisions (cont.)

| # | Decision | Choice | Rationale |
|---|----------|--------|-----------|
| 10 | External references | Auto-ingest on demand (recursive corpus growth) | Complete graph for traversal; risk: version/source-of-truth for fetched docs must be tracked |

## Open risk

- **Fetch source for auto-ingest** — need a reliable source of Vietnamese legal instruments (vanban.chinhphu.vn, thuvienphapluat.vn, etc.) and a way to know which *version/amendment* is current. Deferred to implementation; flagged.

## Decisions (cont.)

| # | Decision | Choice | Rationale |
|---|----------|--------|-----------|
| 11 | Store | SQLite + sqlite-vec (MVP) | Single-file, zero services; corpus is small, write-rarely; graph via recursive CTEs |

## Decisions (cont.)

| # | Decision | Choice | Rationale |
|---|----------|--------|-----------|
| 12 | Embeddings | BGE-M3, local, behind config flag | Strong multilingual + long-text + dense/sparse; A/B against Vietnamese-specialized later |

## Glossary (cont.)

- **Ingestion stage** — one of: `pages` (PDF → page images) → `ocr` (Unlimited-OCR → text+layout) → `structure` (clause graph) → `cards` (rule cards) → `embed` (BGE-M3 vectors). Each stage is idempotent and checkpoints per page/clause in SQLite; reruns skip completed work. Stages runnable individually via CLI.

## Decisions (cont.)

| # | Decision | Choice | Rationale |
|---|----------|--------|-----------|
| 13 | Pipeline | Resumable stages, checkpointed in SQLite | Failures mid-OCR cost nothing; iterate on late stages without re-running OCR |

## Decisions (cont.)

| # | Decision | Choice | Rationale |
|---|----------|--------|-----------|
| 14 | Testing | Three layers: golden-query retrieval set (primary gate, 20–30 queries, precision@k), structural invariants (cheap corruption tripwires), 3–5 golden OCR pages (CER budget) | Retrieval quality is the product; invariants catch silent corruption; OCR fixtures catch diacritic regressions |

## Decisions (cont.)

| # | Decision | Choice | Rationale |
|---|----------|--------|-----------|
| 15 | Scaffolding | Single Python package `bigpdf/` (ingest/ocr/store/api/cli), pyproject, `bigpdf` CLI + `bigpdf serve` | One tool, one package; CLI subcommands mirror ingestion stages |
