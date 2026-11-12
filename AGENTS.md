# AGENTS.md

## Setup
- `uv sync` installs runtime + dev deps (`[dependency-groups] dev`: pytest, pytest-asyncio, ruff, httpx, fastapi,
  uvicorn). `uv.lock` is committed; use `uv add`/`uv sync`. Optional extras: `onnx`, `server`
  (fastapi+uvicorn for `mezzo-guard serve`), `client` (httpx for `mezzoguard.remote`).
- `torch` is intentionally not a dependency (only `transformers`, `huggingface_hub`). Model loading raises with a
  "pip install torch" hint; the ONNX backend needs the `onnx` extra. The fast test suite needs neither torch nor network.

## Commands
- Fast offline suite: `uv run pytest tests/test_*.py`
- Single test: `uv run pytest tests/test_guards.py::test_scan_returns_scores`
- Lint: `uv run ruff check .` — Format: `uv run ruff format .` (line-length 120, target py311)
- Do not run bare `uv run pytest`: it collects `tests/cache_test.py` (matches pytest's `*_test.py` default), which
  executes at import, downloads real HF models, and calls `sys.exit(1)` if its checks fail.
- `tests/compile_test.py` and `tests/onnx_benchmark.py` are manual benchmark scripts (`uv run python tests/...`),
  not pytest tests; they need real model downloads plus torch/onnx.
- No CI or pre-commit: run ruff and the test suite locally.

## Architecture
- `src/mezzoguard/model.py`: `Model` owns the local inference machinery — HF pipeline loading, tokenizer chunk offsets
  (`_chunk_text`), LRU cache (`_predict`), async via `run_in_executor`. It knows nothing about categories or policies.
- `guard/` is the single guard package (`config`, `guard`, `policy`, `result`). `Guard(GuardBase, Model)` auto-selects
  a preset (or a `.mezzoguard` config), always calls the pipeline with `top_k=None`, and exposes one score per
  configured category. Prompt models therefore yield `{"unsafe": score}`, content models yield per-category scores.
  `GuardConfig` normalizes labels and drops any category named `safe`; there is no prompt/content class split. Shared
  helpers live in `base_classes.py` (`ModelConfig`, `GuardBase`, `_redact_text_with_spans`, decorator factories,
  `_warn_if_deprecated`).
- `GuardBase` is the shared ABC for local and remote guards. Only `batch_scan`/`async_batch_scan`/`batch_redact`/
  `async_batch_redact` are abstract; the single-text `scan`/`redact` adapters and the `scan_before_exec`/
  `redact_before_exec` factories are concrete and shared. `Guard` and `APIGuard` both inherit it.
- Version is single-sourced from `__version__` in `src/mezzoguard/__init__.py` (hatch `[tool.hatch.version]`).
- `pyproject.toml` force-includes `src/mezzoguard/py.typed` (empty marker file, present); console script
  `mezzo-guard = mezzoguard.cli:main`.

### Server, CLI, remote clients
- `cli.py` (`mezzo-guard serve <model>`) is argparse-only and imports `server`/guards lazily so `--help` works
  without fastapi or torch. There is no guard-type flag; `create_guard(model, **kwargs)` always builds the unified
  `mezzoguard.guard.Guard`. `create_guard`/`run_server` are module-level for tests to monkeypatch.
- `server/` (extra `server`): `create_app(guard, *, max_batch_texts=..., max_text_chars=...)` wraps an already-loaded
  `Guard` in FastAPI with `GET /healthz`, `GET /v1/info`, `POST /v1/scan`, `POST /v1/redact`. Endpoints call
  `async_batch_scan`/`async_batch_redact`; the lifespan ejects the model on shutdown. Policy wire format is
  `{category: threshold}` plus optional `default_threshold` (falls back to `confidence`); `/v1/info` lists the
  categories declared by the loaded guard's config. `dtype`/`device` are metadata only (no request-time overrides).
- `remote.py` (extra `client`): `APIGuard(url)` mirrors the local Guard API (including `async_*` and `*_before_exec`)
  and rebuilds real `Result`/`Span` objects; `guard.info` lazily GETs `/v1/info`. httpx is imported lazily;
  `MezzoGuardServerError` wraps non-2xx responses.
- `Policy.to_dict()` and the `Result.chunks` properties exist for (de)serialization across the wire.

### Model loading, precision, "quantization", ONNX
- `load_model()` is lazy and idempotent (`if self.pipeline: return`); a missing torch raises `RuntimeError` with the
  install hint. `eject_model()` (or the context manager) drops the pipeline and empties the CUDA cache.
- `dtype="auto"` resolves to `torch.bfloat16` when `torch.cuda.is_bf16_supported()` else `torch.float16` (CPU
  included). There is no calibration or quantization code anywhere: the only precision knobs are `dtype`,
  `torch_compile`, and `use_onnx` (which are mutually exclusive).
- Default path calls `transformers.pipeline(task, model=self.name, dtype=...)` with the model id as a string, so
  transformers can retry the load in fp32 when the requested dtype fails on the device. The `torch_compile` path loads
  `AutoModelForSequenceClassification` itself and sets `model.forward = torch.compile(..., mode=..., dynamic=True)`,
  so it does not get that fallback.
- ONNX (`use_onnx=True`, requires the `onnx` extra): exports/reuses `ORTModelForSequenceClassification` artifacts under
  `~/.cache/mezzoguard/onnx/<model-name>` (`_resolve_onnx_dir`), loading with `export=False` afterwards. `dtype` is
  never passed to the ORT model's `from_pretrained`, and `transformers.pipeline()` drops `dtype` for pre-instantiated
  models, so `dtype="int8"` (as in `tests/onnx_benchmark.py`) is a silent no-op — implementing real int8 means calling
  `optimum.ORTQuantizer` during export, not passing `dtype`.
- `_strip_token_type_ids()` wraps `forward` for models whose signature lacks `token_type_ids`; `load_model` imports
  `torch`/`transformers` lazily so importing `mezzoguard` works without them.
- `device` (`None` default = current behavior, `"auto"` = CUDA when available) is passed to `transformers.pipeline`
  on the torch paths and mapped to an ONNX execution provider (`CPUExecutionProvider`/`CUDAExecutionProvider`) on
  the ONNX path; ONNX only supports `cpu`/`cuda`.

### Scan / redact internals
- `_chunk_text` uses the pipeline tokenizer's `offset_mapping` to return character `(start, end)` spans that cover the
  text losslessly; `overlap >= max_seq_length` raises `ValueError`, and consecutive spans intentionally overlap.
- `batch_scan` flattens the chunks of every input text into one `_predict` call, then regroups them per text. Empty or
  whitespace-only text yields no chunks and therefore zero/empty scores.
- `Guard._from_prediction` maps each chunk label through `config.get_category_for_label`, ignores labels that are not
  configured (including `"safe"`), takes the per-category max across chunks, and keeps normalized `chunks` (one list of
  label/score dicts per chunk).
  `_predict` always passes `top_k=None` (all labels per chunk). Scores are initialized to `0.0` for every configured
  category, so empty text still returns a complete score mapping.
- `_predict` de-duplicates repeated chunk texts within a batch and caches results keyed by exact chunk text.
  `mezzoguard/config.py` `cache_enabled` is checked on every call, but `cache_maxsize` is bound as a default argument
  in `TextLRUCache.__init__` at import time, so changing it at runtime does not resize existing or new caches.
- `_redact_text_with_spans` (base_classes.py) walks spans in order, merges consecutive flagged chunks into one
  replacement, and skips text already covered by overlap, so output preserves the original text exactly except flagged
  runs. `replace` may be a per-category dict keyed by raw category strings; unknown categories fall back to `[REDACTED]`.
- When `policy=None`, a default `Policy(default_threshold=confidence)` is used; thresholds are inclusive
  (`score >= threshold`). An explicit policy without `default_threshold` keeps the backwards-compatible `0.0` fallback.
  Returned `Span.start` is the cursor before the flagged run, so leading overlap is included.

### Config and categories
- `guard/config.py` `MODELS_CONFIG` is the single source of truth for supported model ids. A preset wins over a
  `.mezzoguard` JSON fetched from the HF repo (`ModelConfig.from_model_name`, also works on a local model dir); the
  JSON is used only when no preset exists. Neither → warning, `guard.config is None`.
- `GuardConfig` normalizes labels (strip, lowercase, `_`/space → `-`) and drops any category named `safe`. Categories
  are raw strings; safe labels are simply not listed as scored categories.
- `ModelConfig` no longer carries a model type.
- A repo README's YAML frontmatter `new_version` marks the model deprecated and emits a `DeprecationWarning`
  (`ModelConfig._parse_readme`); `from_model_name` and `get_deprecation_from_readme` are `lru_cache`d.

## Tests
- Use the `prompt_guard` / `content_guard` fixtures from `tests/conftest.py` (both are the same unified `Guard`, just
  with different fake pipelines); they monkeypatch `ModelConfig` and `Model.load_model` with `FakePipeline`.
  Constructing a real `Guard` hits the network.
- `FakePipeline` flags marker words: prompt guard → "kill"/"weapon"/"steal"; content guard → "kill" (violence/toxic),
  "fuck" (sexual), "stupid" (hate-speech), "hurt"/"suicide" (self-harm), "fight" (violence).
- `asyncio_mode = "auto"` (pyproject), so async tests need no `@pytest.mark.asyncio`.
- `test_server.py` uses `fastapi.testclient.TestClient` over fake-pipeline guards; `test_remote.py` starts a
  `uvicorn.Server` in a daemon thread on port 0 with the same fakes. Both stay offline and fast.
