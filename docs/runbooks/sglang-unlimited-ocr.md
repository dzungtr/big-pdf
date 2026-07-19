# Runbook: SGLang server for `baidu/Unlimited-OCR`

This is the blessed launch recipe for the long-running SGLang server that backs
massive-pdf's VLM `OcrBackend` (see epic #19). `massive-pdf` is a pure HTTP
consumer of this server; it never launches or manages the server itself.

## Prerequisites

- A CUDA GPU with >= 16 GB VRAM (developed on an RTX 4070 Ti SUPER, 16 GB).
- The `baidu/Unlimited-OCR` weights present in the local Hugging Face cache
  (`~/.cache/huggingface/hub/models--baidu--Unlimited-OCR/`). Fetch them once:

  ```bash
  huggingface-cli download baidu/Unlimited-OCR
  # or, equivalently:
  python -c "from huggingface_hub import snapshot_download; snapshot_download('baidu/Unlimited-OCR')"
  ```

  This is a ~6 GB download. Only `config.json` ships with the model metadata;
  the safetensors must be fetched separately. Verify the snapshot is complete
  (`model-00001-of-000001.safetensors` should be ~6.67 GB; no `.incomplete`
  blobs should remain under `blobs/`).

- **The model's bundled SGLang wheel installed in the venv.** `baidu/Unlimited-OCR`
  ships a pinned SGLang build under `wheel/` in its HF repo
  (`sglang-0.0.0.dev11416+g92e8bb79e-py3-none-any.whl`). The released
  `sglang==0.5.9` on PyPI **cannot** load this model — its `ModelConfig` does not
  understand `UnlimitedOCRConfig`'s nested `language_config` shape and fails with
  `AttributeError: 'dict' object has no attribute 'hidden_size'`. Install the
  bundled wheel plus the README's pinned extras into the project venv:

  ```bash
  # path inside the downloaded HF snapshot
  WHEEL=~/.cache/huggingface/hub/models--baidu--Unlimited-OCR/snapshots/*/wheel/sglang-*.whl
  pip install --force-reinstall --no-deps $WHEEL
  pip install "kernels==0.11.7" "pymupdf==1.27.2.2"
  # remote-code imports required by the custom architecture
  pip install addict easydict matplotlib
  ```

  Confirm: `python -c "import sglang; print(sglang.__version__)"` should print
  `0.0.0.dev11416+g92e8bb79e` (not `0.5.9`).

## Launch command (blessed — from epic #19, verbatim)

```bash
python -m sglang.launch_server --model baidu/Unlimited-OCR --served-model-name Unlimited-OCR \
  --attention-backend fa3 --page-size 1 --mem-fraction-static 0.8 --context-length 32768 \
  --enable-custom-logit-processor --disable-overlap-schedule --skip-server-warmup \
  --host 0.0.0.0 --port 10000
```

Run it in a foreground terminal, a `tmux`/`screen` session, or backgrounded with
`nohup` — whichever survives your shell. The server binds `0.0.0.0:10000`; the
client contract below targets `127.0.0.1:10000`.

### Environment-specific additions (this machine)

The blessed command above is the verified recipe. On this dev box two
environment adjustments were required to make it come up; both are documented
here so the run is reproducible:

1. **`SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1`** — the model's config
   derives a max context of 2048, but the blessed `--context-length 32768`
   overrides it. SGLang refuses the override unless this env var is set. Export
   it before launching:

   ```bash
   export SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1
   ```

2. **`--disable-cuda-graph`** — the CUDA-graph JIT kernel (`sgl_kernel_jit_*`)
   failed to compile because the system GCC (15.2.1) is newer than the CUDA
   toolkit's supported host compiler (GCC <= 14), and the `tvm-ffi` headers
   don't compile under libstdc++15. Note: for the installed SGLang
   `0.0.0.dev11416+g92e8bb79e` (PyPI `0.5.9`-line) wheel, `--disable-cuda-graph`
   toggles **only** `server_args.disable_cuda_graph`; it does **NOT** prevent the
   RoPE JIT build. `sglang/jit_kernel/rope.py:30` (`_jit_fused_rope_module()`)
   is called on the decode path regardless of this flag, so the RoPE JIT compile
   can still fail under GCC 15 / libstdc++15 even with `--disable-cuda-graph`
   set. Passing the flag avoids the CUDA-graph capture JIT, so there is a
   decode-throughput penalty, but it is not a complete JIT skip. See
   troubleshooting for the full error and the real RoPE-JIT mitigations.

So the full command that was verified on this box is:

```bash
SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1 \
python -m sglang.launch_server --model baidu/Unlimited-OCR --served-model-name Unlimited-OCR \
  --attention-backend fa3 --page-size 1 --mem-fraction-static 0.8 --context-length 32768 \
  --enable-custom-logit-processor --disable-overlap-schedule --skip-server-warmup \
  --disable-cuda-graph --host 0.0.0.0 --port 10000
```

## `MASSIVE_PDF_VLM_ENDPOINT` contract

- **Default:** `http://127.0.0.1:10000/v1`
- **Meaning:** the OpenAI-compatible base URL that massive-pdf's VLM
  `OcrBackend` POSTs chat-completion requests to. The backend reads it from the
  `MASSIVE_PDF_VLM_ENDPOINT` environment variable; if unset, it falls back to
  the default above and fails loudly (with the canonical launch command) when
  the server is unreachable.
- **Override:** export the variable before running `massive-pdf ocr`:

  ```bash
  export MASSIVE_PDF_VLM_ENDPOINT=http://127.0.0.1:10000/v1   # default
  # or, e.g. a remote box:
  export MASSIVE_PDF_VLM_ENDPOINT=http://10.0.0.5:10000/v1
  ```

- The path is `/v1`; the backend appends `/chat/completions`. The served model
  name (`--served-model-name Unlimited-OCR`) is what the backend sends in the
  `model` field of the request body.

## Verify the server is up

Once the server logs show it is ready (it prints
`The server is fired up and ready to roll!` and
`Uvicorn running on http://0.0.0.0:10000`), check the model list:

```bash
curl http://127.0.0.1:10000/v1/models
```

Expected: a JSON object whose `data` array contains an entry with
`"id": "Unlimited-OCR"`, e.g.:

```json
{
  "object": "list",
  "data": [
    { "id": "Unlimited-OCR", "object": "model", "created": 1784437004, "owned_by": "sglang", "root": "Unlimited-OCR", "parent": null, "max_model_len": 32768 }
  ]
}
```

Verified output captured on this dev box (server PID 2633743, served model
`Unlimited-OCR`, `max_model_len: 32768`, access log line
`127.0.0.1:52574 - "GET /v1/models HTTP/1.1" 200 OK`):

```json
{"object":"list","data":[{"id":"Unlimited-OCR","object":"model","created":1784437004,"owned_by":"sglang","root":"Unlimited-OCR","parent":null,"max_model_len":32768}]}
```

If the response is empty or the curl fails to connect, the server is not up —
see troubleshooting.

## Troubleshooting

- **`--enable-custom-logit-processor` is required.** `baidu/Unlimited-OCR` is a
  DeepSeek-OCR-style model whose decoder degenerates into repetition without an
  n-gram suppression filter. The request body must include a
  `DeepseekOCRNoRepeatNGramLogitProcessor` with `ngram_size=35` (a non-standard
  OpenAI API field consumed by SGLang's custom-logit-processor path). Omitting
  the server flag disables that path entirely and produces garbage (repeated
  tokens) output. Do not drop this flag. (Import path:
  `sglang.srt.sampling.custom_logit_processor.DeepseekOCRNoRepeatNGramLogitProcessor`.)
- **`--trust-remote-code` and the bundled SGLang wheel.** `baidu/Unlimited-OCR`
  uses a custom `auto_map` architecture (`UnlimitedOCRForCausalLM` ->
  `modeling_unlimitedocr.py`), plus `configuration_deepseek_v2.py`,
  `modeling_deepseekv2.py`, `deepencoder.py`, and `conversation.py`. The
  **bundled** SGLang wheel (`sglang-0.0.0.dev11416+g92e8bb79e`, in the repo's
  `wheel/` dir) knows how to instantiate `UnlimitedOCRConfig`; the released
  PyPI `sglang` does not. Install the bundled wheel (see Prerequisites) and
  `pip install addict easydict matplotlib` — the remote code imports those.
  If your `sglang` build still refuses to load the model with a "trust remote
  code" error, add `--trust-remote-code` to the launch command.
- **`--page-size 1` and `--attention-backend fa3`** are tuned for the OCR
  workload (single-page, short-context image transcription). Do not change them.
- **`--mem-fraction-static 0.8`** targets a 16 GB GPU (model weights ~6.24 GB,
  KV cache ~5.04 GB; ~2.5 GB headroom). On larger GPUs raise it; on smaller
  ones lower it, but the model needs ~8 GB+ to load.
- **`--context-length 32768`** is the upper bound the OCR client uses; the
  model's own derived max is 2048, so set `SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1`
  or SGLang will refuse to start.
- **`--disable-overlap-schedule` and `--skip-server-warmup`** are part of the
  blessed recipe — keep them.
- **CUDA-graph JIT compile failure (`ninja exited with status 1`,
  `unsupported GNU version! gcc versions later than 14 are not supported`).**
  This happens when the system GCC is newer than the CUDA toolkit supports.
  `NVCC_PREPEND_FLAGS=--allow-unsupported-compiler` bypasses nvcc's version
  check but may still fail at the C++ standard-library level
  (`std::construct_at` / `tvm-ffi::String` deduction errors under libstdc++15).
  **`--disable-cuda-graph` does NOT skip the RoPE JIT** — for the installed
  SGLang `0.0.0.dev11416+g92e8bb79e` (PyPI `0.5.9`-line) wheel, that flag toggles
  only `server_args.disable_cuda_graph`; `sglang/jit_kernel/rope.py:30`
  (`_jit_fused_rope_module()`) still runs on the decode path and its build can
  fail under GCC 15 / libstdc++15. `--disable-cuda-graph` avoids only the
  CUDA-graph capture JIT (at a decode-throughput cost), not the RoPE JIT.
  Real mitigations for the RoPE JIT failure (environment-owner decisions — see
  epic #19's `ready-for-human` note for blocker #2): (a) install a GCC <= 14
  toolchain and pass `-ccbin gcc-14` to nvcc; (b) pre-compile the RoPE kernel on
  a compatible box and populate `~/.cache/tvm-ffi/`; (c) rebuild/reinstall
  SGLang with a Triton fallback for the RoPE kernel. SGLang also suggests
  `--cuda-graph-max-bs 16` or lowering `--mem-fraction-static`.
- **Server not reachable on `127.0.0.1:10000`:** confirm the process is alive
  (`ps aux | grep sglang`), confirm the port is bound
  (`ss -ltnp | grep 10000`), and re-run `curl http://127.0.0.1:10000/v1/models`.
  If `MASSIVE_PDF_VLM_ENDPOINT` is overridden, ensure it points at the same
  `host:port/v1` the server is actually listening on.
- **Weights missing / 0-byte `.incomplete` blobs:** if `huggingface-cli download`
  exits early (e.g. the shell that backgrounded it died), re-run it — it
  resumes. Confirm `model-00001-of-000001.safetensors` is ~6.67 GB before
  launching; SGLang fails at weight-loading time otherwise.
