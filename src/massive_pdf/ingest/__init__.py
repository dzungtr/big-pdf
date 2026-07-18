"""Ingestion stages: pages -> ocr -> structure -> cards -> embed.

Each stage is idempotent and checkpoints per page in the SQLite store so
that long runs can be resumed after a crash without redoing work. Stages
are callable both from the CLI (see `massive_pdf.__main__`) and as
library functions.
"""
