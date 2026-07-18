# ADR 0001: Local VLM OCR via Baidu Unlimited-OCR

## Status
Accepted

## Context
Source documents are scanned Vietnamese legal PDFs with no text layer (Circular 89: 838/839 pages image-only). Vietnamese diacritics are load-bearing — OCR errors change legal meaning. Ingestion is a rare batch operation; corpus grows to dozens of ~1000-page documents.

## Decision
Use Baidu **Unlimited-OCR** (3B MoE vision-language model, SAM+CLIP encoders, DeepSeek-style MoE decoder) served locally via SGLang/vLLM on a 16GB GPU for page transcription + layout structure.

## Alternatives
- **Tesseract/PaddleOCR/VietOCR** — free, local, but historically mediocre Vietnamese diacritic accuracy on scans.
- **Cloud OCR (Google/Azure/AWS)** — strong, ~$1.5/1k pages, but documents leave the machine; recurring cost; no semantic layout.
- **Frontier vision-LLM OCR (GPT/Gemini)** — best quality, ~$10–25/doc, but API dependency + cost scale with corpus growth.

## Consequences
- Zero marginal cost per page; full privacy.
- 8GB+ VRAM footprint on the dev machine; ingestion is GPU-bound (est. ~1–2s/page → ~20–30 min per 1000-page doc).
- Quality on Vietnamese legal scans must be validated against OmniDocBench-class expectations; **errata files** are the correction valve.

## Measured results
(To be filled at initiative close: observed pages/sec, character-error rate on sampled pages vs. manual transcription, vLLM vs SGLang throughput.)
