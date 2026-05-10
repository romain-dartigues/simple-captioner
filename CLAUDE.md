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

`model`, `processor`, `current_model_id`, `current_quant`, `should_abort`, and `ui_e` are module globals mutated by both UI callbacks and the inference generator. `unload_model()` / `load_selected_model()` rebind them; the inference helpers (`preprocess_batch`, `run_generate`, `decode_batch`) read them. Anything that touches model lifecycle must go through `load_selected_model()` so the fallback / cache-clear / left-padding configuration stays consistent. The model is **not** loaded at import — it loads lazily when the user clicks "Load / Reload Model" (or implicitly via UI start).

### Inference decomposition: preprocess / run / decode

The captioning code is split into pure stages so the prefetcher and batched generation can compose them:

- `preprocess_batch(paths, ...)` — builds messages + runs the HF processor on CPU. Returns a `BatchFeature` on CPU. **Safe to call from worker threads** (touches only `processor`, not `model`).
- `run_generate(inputs, max_tokens)` — moves a CPU `BatchFeature` to GPU **in place** (`.to("cuda", non_blocking=True)` rebinds tensors), runs `model.generate`, returns `(generated_ids, input_ids)` on CPU. Single-threaded — only the GPU consumer thread calls this.
- `decode_batch(generated_ids, input_ids)` — `batch_decode` + Qwen3.5 `<think>…</think>` strip via the module-top `r_caption` regex.

`preprocess_one` / `decode_one` / `generate_caption` are thin single-sample wrappers preserved for back-compat; the active path is the batched pipeline below.

### Pipeline: dispatcher → ThreadPoolExecutor → consumer

`process_folder` is the Gradio handler; the pipeline lives in three pieces:

1. **`_prefetch_dispatcher` (thread)** — walks `media_files` in order, groups runs of non-skipped files into batches of up to `batch_size`, and submits each batch as a `Future` to a `ThreadPoolExecutor` of `prefetch_workers` threads. Skip detection happens inline (no executor work). Pushes ordered tagged items onto a bounded `Queue`:
   - `("skip",  idx,     path,  None)`
   - `("batch", indices, paths, Future[BatchFeature])`
   - `None` end-of-stream sentinel
2. **Executor pool** — preprocessing workers run `preprocess_batch` in parallel; the dispatcher's submit is non-blocking, queue back-pressure throttles work to in-flight ≈ `queue_size + workers`.
3. **`_captioning_loop` (consumer, the Gradio generator's body)** — pulls items from the queue in submission order, awaits the Future, runs one `model.generate` per batch, decodes N captions, writes the `.txt` files, and yields one progress update **per sample within the batch** (so progress bar is still per-sample granular even with `batch_size > 1`).

Cleanup on abort/finish (`finally` in `process_folder`): set the `cancel` event, drain the queue (unblocks dispatcher's `put`), `executor.shutdown(wait=False, cancel_futures=True)` cancels queued-but-unstarted preprocess tasks while running ones finish naturally, then `dispatcher.join(timeout=2.0)`.

### Left padding is required for batch generation

`load_selected_model` sets `processor.tokenizer.padding_side = "left"` after instantiating the processor. Causal-LM batch generation requires all sequences to share the same right edge so `model.generate` resumes each from its true last token. With right padding the shorter sequences would resume from a pad token. This was harmless when `batch_size` was hardcoded to 1; it's load-bearing now.

### Model class dispatch

`load_selected_model()` picks one of several concrete classes based on substring match against the model id, falling back to `AutoModelForImageTextToText`:

- `Qwen3.5` / `Qwen3_5` → `Qwen3_5ForConditionalGeneration` (text-only; uses thinking-mode prompt)
- `Qwen3-VL` → `Qwen3VLForConditionalGeneration`
- `Qwen2.5-VL` / `Qwen2_5-VL` → `Qwen2_5_VLForConditionalGeneration`
- `joycaption` (case-insensitive substring) → `LlavaForConditionalGeneration` — see "JoyCaption / LLaVA family" below

When adding a new model family, extend this dispatch *and* `is_qwen35_model()` if it shares Qwen3.5's thinking-mode quirk.

### Pre-quantized model IDs

`_is_prequantized()` matches IDs ending in `-4bit` / `-8bit` (and the dashed `-4-bit` / `-8-bit` variants) and suppresses our `BitsAndBytesConfig` kwarg in `load_selected_model()`. These checkpoints already carry a `quantization_config` in their `config.json` and HF will load them with the saved settings; layering our config on top can conflict. For pre-quantized variants the UI's quant radio is effectively ignored — the saved quant wins.

### JoyCaption / LLaVA family

JoyCaption Beta One (`fancyfeast/llama-joycaption-beta-one-hf-llava`) is a LLaVA model: Llama 3.1 8B + SigLIP vision tower. It does **not** share the Qwen path:

- **Messages**: `_preprocess_batch_joycaption()` builds a fixed `[{system: JOYCAPTION_SYSTEM_PROMPT}, {user: <augmented prompt>}]` chat. The system prompt is hard-coded — the user-facing prompt textbox drives only the user turn. No `process_vision_info`, no resolution kwargs (LLaVA's image processor handles sizing). Image is passed separately to `processor(text=..., images=...)`.
- **Image-only**: videos raise `ValueError` from preprocessing — the model has no video path.
- **Generation**: `_generation_kwargs()` adds `do_sample=True, temperature=0.6, top_p=0.9, suppress_tokens=None, use_cache=True` for JoyCaption. Other families stay greedy.
- **pixel_values dtype**: `run_generate()` casts `pixel_values` to the SigLIP vision tower's parameter dtype before generation. The processor returns fp32; the vision tower is in fp16/bf16 (it's deliberately not BnB-quantized — see next point), so without the cast SigLIP's forward errors on a dtype mismatch. The tower lives at `model.model.vision_tower` in transformers 5.x.
- **Vision tower is skipped from bnb quantization**: `load_selected_model` adds `llm_int8_skip_modules=["vision_tower"]` to `BitsAndBytesConfig` whenever `is_joycaption_model(...)` is true (`llm_int8_skip_modules` is honored for both 4-bit and 8-bit despite the name). Required because SigLIP's `nn.MultiheadAttention` calls `F.multi_head_attention_forward → F.linear` directly with the raw weight tensor, bypassing `Linear4bit.forward`'s dequantization. With a quantized vision-tower weight that path crashes with `self and mat2 must have the same dtype, but got BFloat16 and Byte`. Cost: ~0.4 GB extra VRAM for the fp16/bf16 vision tower — negligible vs. the LM weights.
- **Pre-quantized variants are not listed**: `heavlav/llama-joycaption-beta-one-hf-llava-4bit` baked the quant into the vision tower at save time, hits the same bug, and we can't undo it from our side. Stick with the un-quantized fancyfeast checkpoint + our own bnb config.
- **`DEFAULT_MAX_TOKENS = 256`**: JoyCaption captions are typically 200–400 tokens; the previous 128 default truncated.
- **`pad_token` fallback**: Llama-3.1's tokenizer ships without a `pad_token`, which breaks the processor's `padding=True`. `load_selected_model` aliases `pad_token = eos_token` whenever it's missing (no-op for Qwen tokenizers that already have one) and propagates the id to `model.generation_config.pad_token_id`.

### Qwen3.5 thinking-mode handling

For Qwen3.5 models, `_build_messages()` appends a pre-seeded `{"role": "assistant", "content": "<think>\n\n</think>\n\n"}` message and `preprocess_batch` uses `continue_final_message=True` instead of `add_generation_prompt`. Any `<think>…</think>` block that leaks into the output is stripped in `decode_batch` via `r_caption`. VL models don't take this branch.

### Flash-attention fallback

`load_selected_model()` tries `flash_attention_2` first and silently retries with `eager` on any exception. `DEFAULT_ATTN` is `eager` on Windows (`os.name == "nt"`) and `flash_attention_2` elsewhere because flash-attn is unreliable on Windows. Don't tighten the `except Exception` — the underlying failures vary by torch/flash-attn version combo.

### Compute dtype on Ampere+

`preferred_compute_dtype()` returns `bfloat16` on devices with compute capability ≥ 8.0 (Ampere+, including the RTX 3060) and `float16` otherwise. It's used as the 4-bit BnB compute dtype; bf16 has fp32's exponent range so it avoids activation overflows at no speed cost on supported hardware.

### `control_keys` and the generator yield contract

`process_folder()` is a Gradio generator. Each `yield` produces `(status, media, name_md, caption, progress, elapsed_str, *control_updates)` — the 6 leading values map to specific output components, and `*control_updates` must be a list **in the exact order of `control_keys`** (see the `start_button.click(...)` `outputs=` list). When you add a new interactive UI element:

1. Add it to `ui_e[...]` at construction time.
2. Add its key to `control_keys` (order matters — it's a positional contract).
3. The `disable_controls_dict()` / `enable_controls_dict()` / `start_process()` / `finish_process()` / `abort_process()` helpers will pick it up automatically.

Forgetting step 2 causes silent UI drift (wrong components get enabled/disabled).

### Resolution mode → processor kwargs

`_resolution_kwargs()` returns the `min_pixels`/`max_pixels` or `resized_height`/`resized_width` dict for a given mode; `_build_messages` injects it into the message content block (only for images — videos take the empty dict).

### Abort flow

`should_abort` is checked at the top of each `_captioning_loop` iteration **and** at the top of each `_prefetch_dispatcher` iteration, then reset to `False` on consumption by the consumer. The abort button has `queue=False` so it bypasses Gradio's queue and flips the flag immediately even while a generation is in-flight; the in-flight `model.generate()` call still runs to completion — abort takes effect on the *next* batch. The producer-side `cancel` `threading.Event` is a separate signal used only for thread cleanup in the `finally` block.

### Batch size suggestion

`suggest_batch_size()` runs after `load_selected_model` and writes a starting batch size into the slider via `gradio.update`. It buckets free post-load VRAM (≥12 GB → 8, ≥8 → 4, ≥4 → 2, else 1). Deliberately conservative because `max_tokens` and image resolution aren't known yet.

### Output convention

For each input file `foo.jpg`, the caption is written as `foo.<ext>` next to it (via `_caption_path_for`). The extension is taken from the **Caption File Extension** UI textbox (default `txt`); the user enters it without the leading dot, and `_sanitize_caption_extension()` strips stray dots/whitespace and falls back to `txt` if empty. `skip_existing` checks for the file at that exact extension before invoking the model — switching extensions effectively re-captions everything because the old `.txt` files no longer count as "already captioned". Subfolders are walked recursively via `os.walk`.

## Style

- Editorconfig: 4-space indent, LF, UTF-8. Ruff line length 120.
- Imports are grouped `# stdlib` / `# dependencies` by hand-written comments — preserve that style when adding imports.

## Rules

- When editing the code, do not forget to update your memories and this CLAUDE.md accordingly.
