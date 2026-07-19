"""Unit tests for UnlimitedOcrBackend (httpx MockTransport; no GPU/server)."""
from __future__ import annotations

import json

import httpx
import pytest

from massive_pdf.ingest.ocr import OcrPage
from massive_pdf.ingest.vlm import (
    UnlimitedOcrBackend,
    SGLANG_LAUNCH_CMD,
    VLM_MODEL,
    VLM_PROMPT,
)


def _sse_stream(chunks: list[str]) -> bytes:
    """Render an SSE stream of chat.completion chunks."""
    out = b""
    for piece in chunks:
        payload = {"choices": [{"delta": {"content": piece}, "index": 0}]}
        out += b"data: " + json.dumps(payload).encode() + b"\n\n"
    out += b"data: [DONE]\n\n"
    return out


def _backend_with(handler, endpoint="http://127.0.0.1:10000/v1") -> UnlimitedOcrBackend:
    transport = httpx.MockTransport(handler)
    client = httpx.Client(transport=transport)
    return UnlimitedOcrBackend(endpoint=endpoint, client=client)


def _happy_handler(request: httpx.Request) -> httpx.Response:
    body = json.loads(request.content)
    assert request.url.path == "/v1/chat/completions"
    assert body["model"] == VLM_MODEL
    assert body["temperature"] == 0
    assert body["stream"] is True
    return httpx.Response(
        200,
        content=_sse_stream(["Hello ", "Điều 1.", " Khoản 2."]),
        headers={"content-type": "text/event-stream"},
    )


def test_unlimited_backend_implements_protocol():
    backend = UnlimitedOcrBackend(endpoint="http://127.0.0.1:10000/v1")
    # Structural (duck-typed) protocol check — OcrBackend is not runtime_checkable.
    assert backend.name == "unlimited-ocr"
    assert callable(getattr(backend, "transcribe", None))


def test_transcribe_happy_path_accumulates_raw_text(tmp_path):
    image = tmp_path / "page.png"
    image.write_bytes(b"\x89PNG\r\n\x1a\n")
    backend = _backend_with(_happy_handler)
    page = backend.transcribe(str(image), page_ordinal=5)
    assert isinstance(page, OcrPage)
    assert page.page_ordinal == 5
    assert page.image_path == str(image)
    assert page.raw_text == "Hello Điều 1. Khoản 2."
    # The VLM emits text only — no bboxes (ADR 0001 schema mapping).
    assert page.blocks == []


def test_request_body_contains_prompt_and_logit_processor(tmp_path):
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            content=_sse_stream(["ok"]),
            headers={"content-type": "text/event-stream"},
        )

    image = tmp_path / "page.png"
    image.write_bytes(b"\x89PNG\r\n\x1a\n")
    _backend_with(handler).transcribe(str(image), page_ordinal=1)

    body = captured["body"]
    # Prompt is the grilling-decided "document parsing.".
    user_msg = body["messages"][0]["content"]
    texts = [part["text"] for part in user_msg if part["type"] == "text"]
    assert VLM_PROMPT in texts
    # Non-standard anti-repeat logit processor must travel in the body.
    proc = body.get("custom_logit_processor")
    assert proc, "custom_logit_processor must be present in the request body"
    assert proc[0]["class"] == "DeepseekOCRNoRepeatNGramLogitProcessor"
    assert proc[0]["args"]["ngram_size"] == 35
    # Image is a base64 data URL.
    img_parts = [part for part in user_msg if part["type"] == "image_url"]
    assert img_parts
    url = img_parts[0]["image_url"]["url"]
    assert url.startswith("data:image/png;base64,")


def test_connect_error_raises_runtime_error_with_launch_command(tmp_path):
    image = tmp_path / "page.png"
    image.write_bytes(b"\x89PNG\r\n\x1a\n")

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    backend = _backend_with(handler)
    with pytest.raises(RuntimeError) as exc_info:
        backend.transcribe(str(image), page_ordinal=1)
    msg = str(exc_info.value)
    assert "unreachable" in msg.lower()
    assert SGLANG_LAUNCH_CMD in msg


def test_http_status_error_raises_runtime_error_with_launch_command(tmp_path):
    image = tmp_path / "page.png"
    image.write_bytes(b"\x89PNG\r\n\x1a\n")

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, content=b"server not ready")

    backend = _backend_with(handler)
    with pytest.raises(RuntimeError) as exc_info:
        backend.transcribe(str(image), page_ordinal=1)
    assert SGLANG_LAUNCH_CMD in str(exc_info.value)
    assert "503" in str(exc_info.value)
