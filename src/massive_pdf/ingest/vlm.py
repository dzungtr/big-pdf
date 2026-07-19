"""VLM OCR backend: HTTP client of the long-running SGLang server.

Implements the `OcrBackend` protocol defined in `ocr.py`. The backend is a
pure HTTP consumer of the externally-managed SGLang server (epic #19 /
ADR 0001); it never spawns or manages the server itself. See the launch
runbook for how to stand the server up — when it is unreachable this
backend fails loudly with the canonical launch command.
"""
from __future__ import annotations

import base64
import json
import os
from pathlib import Path

import httpx

from .ocr import OcrBackend, OcrPage

VLM_ENDPOINT_ENV = "MASSIVE_PDF_VLM_ENDPOINT"
DEFAULT_VLM_ENDPOINT = "http://127.0.0.1:10000/v1"

# Model name the blessed SGLang launch registers via --served-model-name.
VLM_MODEL = "Unlimited-OCR"
VLM_PROMPT = "document parsing."

# The blessed launch recipe (epic #19). Embedded verbatim in the error
# message so a missing server is self-diagnosing.
SGLANG_LAUNCH_CMD = (
    "python -m sglang.launch_server --model baidu/Unlimited-OCR "
    "--served-model-name Unlimited-OCR --attention-backend fa3 "
    "--page-size 1 --mem-fraction-static 0.8 --context-length 32768 "
    "--enable-custom-logit-processor --disable-overlap-schedule "
    "--skip-server-warmup --host 0.0.0.0 --port 10000"
)


def _image_data_url(image_path: str) -> str:
    """Read the page image and return a base64 data URL."""
    mime = "image/png" if image_path.lower().endswith(".png") else "image/jpeg"
    data = base64.b64encode(Path(image_path).read_bytes()).decode("ascii")
    return f"data:{mime};base64,{data}"


def _no_repeat_ngram_processor() -> dict:
    """The non-standard logit-processor config the blessed launch requires.

    `--enable-custom-logit-processor` + `DeepseekOCRNoRepeatNGramLogitProcessor`
    (ngram_size=35) is mandatory for sane anti-repetition output (runbook /
    epic #19). It travels in the request body because this SGLang endpoint is
    OpenAI-compatible only up to the chat-completions surface.
    """
    return {
        "custom_logit_processor": [
            {
                "class": "DeepseekOCRNoRepeatNGramLogitProcessor",
                "args": {"ngram_size": 35},
            }
        ],
    }


class UnlimitedOcrBackend:
    """Real OCR backend: streams transcription from the SGLang VLM server.

    Reads the server endpoint from `MASSIVE_PDF_VLM_ENDPOINT` (default
    `http://127.0.0.1:10000/v1`) at construction time. Each call to
    `transcribe` POSTs the page image to `<endpoint>/chat/completions` as a
    base64 data URL, accumulates the streamed `delta["content"]` chunks, and
    returns an `OcrPage` whose `raw_text` is the concatenation.
    """

    name = "unlimited-ocr"

    def __init__(self, endpoint: str | None = None, client: httpx.Client | None = None):
        self.endpoint = (endpoint if endpoint is not None
                         else os.environ.get(VLM_ENDPOINT_ENV, DEFAULT_VLM_ENDPOINT))
        # A caller may inject an httpx.Client (e.g. with MockTransport) for tests.
        self._client = client

    def _request_body(self, image_path: str) -> dict:
        body = {
            "model": VLM_MODEL,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": VLM_PROMPT},
                        {"type": "image_url", "image_url": {"url": _image_data_url(image_path)}},
                    ],
                }
            ],
            "temperature": 0,
            "stream": True,
        }
        body.update(_no_repeat_ngram_processor())
        return body

    def transcribe(self, image_path: str, page_ordinal: int) -> OcrPage:
        url = f"{self.endpoint.rstrip('/')}/chat/completions"
        body = self._request_body(image_path)
        text_parts: list[str] = []
        own_client = self._client is None
        client = self._client or httpx.Client(timeout=httpx.Timeout(300.0, connect=10.0))
        try:
            try:
                response = client.post(url, json=body)
                response.raise_for_status()
            except httpx.ConnectError as exc:
                raise RuntimeError(
                    f"VLM OCR endpoint unreachable at {self.endpoint}: {exc}. "
                    f"Start the SGLang server first:\n    {SGLANG_LAUNCH_CMD}"
                ) from exc
            except httpx.HTTPStatusError as exc:
                raise RuntimeError(
                    f"VLM OCR endpoint {self.endpoint} returned HTTP "
                    f"{exc.response.status_code}. Ensure the SGLang server is "
                    f"running:\n    {SGLANG_LAUNCH_CMD}"
                ) from exc

            for line in response.iter_lines():
                if not line:
                    continue
                if line.startswith("data: "):
                    line = line[len("data: "):]
                if line == "[DONE]":
                    break
                try:
                    chunk = json.loads(line)
                except json.JSONDecodeError:
                    continue
                choices = chunk.get("choices") or []
                if not choices:
                    continue
                delta = choices[0].get("delta") or {}
                piece = delta.get("content")
                if piece:
                    text_parts.append(piece)
        finally:
            if own_client:
                client.close()

        raw_text = "".join(text_parts)
        # blocks is empty: the VLM emits text only, no bboxes (ADR 0001 schema mapping).
        return OcrPage(
            page_ordinal=page_ordinal,
            image_path=image_path,
            blocks=[],
            raw_text=raw_text,
        )


__all__ = ["UnlimitedOcrBackend", "VLM_ENDPOINT_ENV", "DEFAULT_VLM_ENDPOINT"]
