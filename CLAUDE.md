# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

Single-file Gradio app (`app.py`) that batch-captions a folder of images/videos using Qwen VL models from Hugging Face. There is no library/package layer — everything (model loading, inference, UI wiring, generator handlers) lives in `app.py`.

## Commands

This fork uses **uv** with a CUDA 13 PyTorch index (the upstream README still describes the older `pip install -r requirements.txt` flow against CUDA 12.8 — `pyproject.toml` is the source of truth for this fork).

```bash
uv sync                 # install deps from pyproject.toml/uv.lock (resolves torch from pytorch-cu130 index)
uv run python app.py    # launch the Gradio UI (also: ./run_app.sh once a venv is active)
uv run ruff check .     # lint (config: line-length 120, selects C/E/F/I/W)
uv run ruff format .    # format
```

There is no test suite.

## Architecture notes that aren't obvious from one file

### Module-level mutable state

`model`, `processor`, `current_model_id`, `current_quant`, `should_abort`, and `ui_e` are module globals mutated by both UI callbacks and the inference generator. `unload_model()` / `load_selected_model()` rebind them; `generate_caption()` and `process_folder()` read them. Anything that touches model lifecycle must go through `load_selected_model()` so the fallback / cache-clear logic stays consistent. The model is **not** loaded at import — it loads lazily when the user clicks "Load / Reload Model" (or implicitly via UI start).

### Model class dispatch

`load_selected_model()` picks one of three concrete classes based on substring match against the model id, falling back to `AutoModelForImageTextToText`:

- `Qwen3.5` / `Qwen3_5` → `Qwen3_5ForConditionalGeneration` (text-only; uses thinking-mode prompt)
- `Qwen3-VL` → `Qwen3VLForConditionalGeneration`
- `Qwen2.5-VL` / `Qwen2_5-VL` → `Qwen2_5_VLForConditionalGeneration`

When adding a new model family, extend this dispatch *and* `is_qwen35_model()` if it shares Qwen3.5's thinking-mode quirk.

### Qwen3.5 thinking-mode handling

For Qwen3.5 models, `generate_caption()` appends a pre-seeded `{"role": "assistant", "content": "<think>\n\n</think>\n\n"}` message and uses `continue_final_message=True` instead of `add_generation_prompt`. Any `<think>…</think>` block that leaks into the output is stripped with `r_caption` (compiled at module top). VL models don't take this branch.

### Flash-attention fallback

`load_selected_model()` tries `flash_attention_2` first and silently retries with `eager` on any exception. `DEFAULT_ATTN` is `eager` on Windows (`os.name == "nt"`) and `flash_attention_2` elsewhere because flash-attn is unreliable on Windows. Don't tighten the `except Exception` — the underlying failures vary by torch/flash-attn version combo.

### `control_keys` and the generator yield contract

`process_folder()` is a Gradio generator. Each `yield` produces `(status, media, name_md, caption, progress, elapsed_str, *control_updates)` — the 6 leading values map to specific output components, and `*control_updates` must be a list **in the exact order of `control_keys`** (see the `start_button.click(...)` `outputs=` list). When you add a new interactive UI element:

1. Add it to `ui_e[...]` at construction time.
2. Add its key to `control_keys` (order matters — it's a positional contract).
3. The `disable_controls_dict()` / `enable_controls_dict()` / `start_process()` / `finish_process()` / `abort_process()` helpers will pick it up automatically.

Forgetting step 2 causes silent UI drift (wrong components get enabled/disabled).

### Resolution mode → processor kwargs

`generate_caption()` injects `min_pixels`/`max_pixels` or `resized_height`/`resized_width` directly into the message `content_block` based on `resolution_mode`. The top-level `PRESETS` dict is currently unused — the actual values are hardcoded in the if/elif chain. If you change one, change both or delete `PRESETS`.

### Abort flow

`should_abort` is checked at the top of each loop iteration in `process_folder()` and reset to `False` on consumption. The abort button has `queue=False` so it bypasses Gradio's queue and flips the flag immediately even while a generation is in-flight (the in-flight `model.generate()` call still runs to completion — abort takes effect on the *next* iteration).

### Output convention

For each input file `foo.jpg`, the caption is written as `foo.txt` next to it. `skip_existing` checks for that `.txt` before invoking the model. Subfolders are walked recursively via `os.walk`.

## Style

- Editorconfig: 4-space indent, LF, UTF-8. Ruff line length 120.
- Imports are grouped `# stdlib` / `# dependencies` by hand-written comments — preserve that style when adding imports.

## Rules

- When editing the code, do not forget to update your memories and this CLAUDE.md accordingly.
