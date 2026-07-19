# Acceptance run results: SGLang `baidu/Unlimited-OCR` (slice 4 / issue #23)

> **Status: BLOCKED — no OCR text was produced.** The SGLang server starts and
> answers `GET /v1/models`, but every transcription request crashes the server's
> scheduler before a single token is generated. This document is the
> hand-off artifact for the ADR-0001 "Measured results" promotion step (autobot
> step 7); it records what was attempted, the two distinct failure modes found,
> and the unblock path. It does **not** modify ADR 0001 or `CONTEXT.md`.

## Run context

| Item | Value |
|---|---|
| Issue | #23 (slice 4 — acceptance run + ADR-0001 measurement capture) |
| Epic / spec home | #19 |
| Source PDF | `Thong-tu-89-BTC.pdf` (repo root, gitignored) — 839 scanned pages, no text layer |
| Planned sample | pages 1, 100, 300, 500, 700, 839 (spread across the document) |
| Page cache | All 839 PNGs already rendered at `.massive_pdf/pages/doc1/` (300 DPI); the `pages` table `image_path` column points at `work/pages/doc1/` and those files also exist — reused as-is |
| Doc id | 1 (`documents.source_path = 'Thong-tu-89-BTC.pdf'`) |
| GPU | NVIDIA RTX 4070 Ti SUPER, 16376 MiB (≈14.9 GB free at launch) |
| SGLang wheel | bundled `sglang-0.0.0.dev11416+g92e8bb79e` (installed per the runbook; `sglang.__version__` confirms) |
| CUDA / host GCC | CUDA 12.8 (`nvcc V12.8.93`), GCC 15.2.1 (Red Hat 15.2.1-7) |
| Launch command | the "full command that was verified on this box" from `docs/runbooks/sglang-unlimited-ocr.md` (i.e. the blessed recipe plus `SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1` and `--disable-cuda-graph`) |
| Server readiness | confirmed — `curl http://127.0.0.1:10000/v1/models` returned `{"id":"Unlimited-OCR",...,"max_model_len":32768}` on every launch |

## Outcome

**No sampled page was transcribed.** Throughput (pages/sec), the CER spot-check,
and the throughput-suitability extrapolation could not be measured because no
OCR text was ever returned. The two blockers below are both reproducible on
this machine; each crashes the scheduler with `SIGQUIT` and drops the HTTP
connection without a response body.

## Blocker 1 (root cause for `UnlimitedOcrBackend`): wrong request shape

The `UnlimitedOcrBackend` shipped in slice #21 (`src/massive_pdf/ingest/vlm.py`)
POSTs a request body whose `custom_logit_processor` field is a **list of
`{class, args}` dicts**:

```python
"custom_logit_processor": [
    {"class": "DeepseekOCRNoRepeatNGramLogitProcessor", "args": {"ngram_size": 35}}
]
```

The installed SGLang wheel rejects this at the HTTP layer:

```
HTTP 400 — 2 validation errors:
  custom_logit_processor: list[nullable[str]] -> Input should be a valid string
  custom_logit_processor: str            -> Input should be a valid string
```

`sglang/srt/managers/io_struct.py` declares the field as
`Optional[Union[List[Optional[str]], str]]`, and
`sglang/srt/sampling/sampling_batch_info.py` then treats each per-request value
as a **hashable string** that it passes to `CustomLogitProcessor.from_str()` —
which `orjson.loads`-decodes a JSON document of shape `{"callable": "<dill-hex>"}`
(see `sglang/srt/sampling/custom_logit_processor.py`, `to_str`/`from_str`).

The authoritative client in the model's own README
(`baidu/Unlimited-OCR` HF snapshot, `README.md` lines ~230-245) is unambiguous
about the correct shape:

```python
from sglang.srt.sampling.custom_logit_processor import DeepseekOCRNoRepeatNGramLogitProcessor

payload = {
    "model": "Unlimited-OCR",
    "messages": [...],
    "temperature": 0,
    "skip_special_tokens": False,
    "images_config": {"image_mode": "gundam"},          # for single-page
    "custom_logit_processor": DeepseekOCRNoRepeatNGramLogitProcessor.to_str(),  # a JSON STRING, not a list
    "custom_params": {"ngram_size": 35, "window_size": 128},  # separate field
    "stream": True,
}
```

I verified `DeepseekOCRNoRepeatNGramLogitProcessor.to_str()` is importable and
returns a 216-char `{"callable": "..."}` JSON string. So the backend in
`vlm.py` is missing three things versus the real server contract:

1. `custom_logit_processor` must be `…NoRepeatNGramLogitProcessor.to_str()`
   (a serialized string), not `[{class, args}]`.
2. A separate top-level `custom_params` field
   (`{"ngram_size": 35, "window_size": <int>}`) is required — the n-gram size
   does not travel inside the processor spec.
3. `images_config: {"image_mode": "gundam"}` (single page) / `"base"` (multi
   page) and `skip_special_tokens: False` are part of the documented payload
   and are absent from `vlm.py`.

This slice is scoped to a results report and **must not edit `src/`**, so the
backend is left as-is and this finding is handed off as a required fix for a
follow-up slice. With the backend unmodified, `massive-pdf ocr --backend vlm`
cannot succeed against the real server.

## Blocker 2 (root cause for the server crash): RoPE JIT compile fails under GCC 15

To still capture the acceptance numbers for this report, I exercised the VLM
via the README's authoritative client (a measurement harness, not a source
change). Using the correct request shape, the request passes HTTP validation
and enters the model forward pass — and the scheduler then crashes on the
**first** request while JIT-compiling the fused-RoPE CUDA kernel:

```
sglang/srt/layers/rotary_embedding/base.py -> apply_rope_with_cos_sin_cache_inplace
  -> sglang/jit_kernel/rope.py -> _jit_fused_rope_module -> load_jit/load_inline
  -> tvm_ffi build_ninja
RuntimeError: ninja exited with status 1
  /usr/local/cuda/.../crt/host_config.h:143: #error -- unsupported GNU version!
  gcc versions later than 14 are not supported!
```

`--disable-cuda-graph` (the runbook's documented flag for this box) disables
CUDA **graph capture** but does **not** skip the RoPE JIT — `rope.py` calls
`_jit_fused_rope_module` unconditionally on the first decode, so the JIT is
attempted regardless.

Per the runbook's troubleshooting, the documented workaround for the nvcc
version check is `NVCC_PREPEND_FLAGS=--allow-unsupported-compiler`. I retried
the launch with that flag set (the one retry this slice allows). nvcc's
version check is bypassed, but the kernel then fails at the **C++ standard
library** level under libstdc++15:

```
/usr/include/c++/15/bits/alloc_traits.h(676): error: no instance of function
  template "std::construct_at" matches the argument list
    argument types are: (std::string_view *, std::string_view)
... triggered from tvm-ffi/include/tvm/ffi/string.h and function.h
100 errors detected in the compilation of cuda.cu. Compilation terminated.
SIGQUIT received.
```

This is exactly the "may still fail at the C++ standard-library level
(`std::construct_at` / `tvm-ffi::String` deduction errors under libstdc++15)"
case the runbook warns about. On this machine neither documented workaround
yields a working server. The runbook's preferred long-term fix — "a GCC ≤ 14
toolchain for CUDA compilation" — is an environment change outside this slice's
scope and retry budget, so it is recorded here, not applied.

## Evidence collected

- `GET /v1/models` (every launch): `{"object":"list","data":[{"id":"Unlimited-OCR","object":"model","created":1784439854,"owned_by":"sglang","root":"Unlimited-OCR","parent":null,"max_model_len":32768}]}` → server process is healthy until a transcription request arrives.
- Launch-1 log (`/tmp/sglang.log`, 15:36-15:39): `The server is fired up and ready to roll!` + `Uvicorn running on http://0.0.0.0:10000`; then the `vlm.py` request → HTTP 400 (validation error quoted above).
- Launch-2 log (15:43-15:45): README-shape request → scheduler `TypeError: unhashable type: 'list'` is **not** hit once the shape is corrected; instead the RoPE JIT `ninja exited with status 1` (`unsupported GNU version! gcc versions later than 14 are not supported!`) → `SIGQUIT`.
- Launch-3 log (15:46): same request with `NVCC_PREPEND_FLAGS=--allow-unsupported-compiler` → nvcc version check bypassed, compile fails at `std::construct_at` / `tvm::ffi::String` deduction under `/usr/include/c++/15` → `SIGQUIT`.
- `gcc --version`: `gcc (GCC) 15.2.1 20260123 (Red Hat 15.2.1-7)`; `nvcc --version`: `Cuda compilation tools, release 12.8, V12.8.93`.
- No cached compiled kernel exists under `~/.cache/tvm-ffi/` (only the failed build dir for `sgl_kernel_jit_fused_rope_true_128_false_bf16_t_…`), so the JIT is attempted on every first request.

## Measurements

| Metric | Result |
|---|---|
| Observed pages/sec | **Not measured** — no page produced text. |
| CER spot-check vs. manual transcription | **Not measured** — no VLM output to compare. |
| Throughput suitability vs. ADR-0001 ~20–30 min/doc | **Cannot assess from this run.** The model never completed a forward pass on this box, so no empirical page-time exists to extrapolate to the 839-page corpus. ADR 0001's ~1–2 s/page → ~20–30 min/doc estimate remains **unvalidated** by this slice. |
| Anomalies | (a) backend request shape incompatible with installed SGLang wheel (blocker 1); (b) RoPE JIT kernel unbuildable under GCC 15 / libstdc++ 15 / CUDA 12.8 (blocker 2). No repetition/truncation/layout-loss observations are possible without output. |

## Recommended unblocks (for a follow-up slice, not this one)

1. **Fix `UnlimitedOcrBackend._request_body`** to match the model README
   contract: `custom_logit_processor = DeepseekOCRNoRepeatNGramLogitProcessor.to_str()`,
   plus a top-level `custom_params = {"ngram_size": 35, "window_size": 128}`
   (single page) / `1024` (multi page), `images_config = {"image_mode": "gundam"|"base"}`,
   and `skip_special_tokens = False`. Add a mocked-transport test asserting that
   shape. This unblocks the CLI path (`massive-pdf ocr --backend vlm`).
2. **Make the RoPE JIT buildable on this dev box.** Either (a) install a
   GCC ≤ 14 toolchain and point nvcc at it (`-ccbin gcc-14`), or (b) pre-compile
   the `sgl_kernel_jit_fused_rope` kernel once on a compatible toolchain and
   populate `~/.cache/tvm-ffi/`, or (c) upgrade the bundled SGLang wheel to a
   build that ships a pre-compiled RoPE kernel or a Triton fallback for this
   model. Update `docs/runbooks/sglang-unlimited-ocr.md` accordingly: the
   current note that `--disable-cuda-graph` "skips the JIT entirely" is
   inaccurate for this wheel — it only disables CUDA graph capture, not the
   RoPE JIT.
3. Once (1) and (2) land, re-run this slice's sampled acceptance pass
   (pages 1, 100, 300, 500, 700, 839) and promote the numbers into ADR 0001's
   "Measured results" section via the dedicated docs PR.

## What this slice changes

Docs only: adds this file. No changes to `src/`, `docs/adr/`,
`CONTEXT.md`, tests, or `pyproject.toml`. No weights, caches,
`.massive_pdf.sqlite`, or `.massive_pdf/` artifacts are committed.
