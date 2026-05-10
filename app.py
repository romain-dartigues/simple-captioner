# stdlib
import re
from concurrent.futures import Future, ThreadPoolExecutor
from logging import basicConfig, getLogger
from os import cpu_count
from pathlib import Path
from queue import Empty, Full, Queue
from threading import Event, Thread
from time import monotonic
from typing import Any, cast

# dependencies
import gradio
import torch
from PIL import Image
from qwen_vl_utils import process_vision_info
from transformers.generation import GenerationMixin
from transformers.modeling_utils import SpecificPreTrainedModelType
from transformers.models.auto.modeling_auto import AutoModelForCausalLM, AutoModelForImageTextToText
from transformers.models.auto.processing_auto import AutoProcessor
from transformers.models.llava.modeling_llava import LlavaForConditionalGeneration
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLForConditionalGeneration
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForConditionalGeneration
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLForConditionalGeneration
from transformers.processing_utils import ProcessorMixin
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.utils.quantization_config import BitsAndBytesConfig

logger = getLogger(__name__)

r_caption = re.compile(r"<think>.*?</think>\s*", re.DOTALL)

IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".gif", ".webp")
VIDEO_EXTENSIONS = (".mp4", ".mov", ".avi", ".webm", ".mkv", ".gif", ".flv")
DEFAULT_PROMPT = (
    "In a concise way, describe this media, it's background, "
    "composition, lighting, camera, lens, style, ambiance, "
    "people, emotions, acts, racial traits, physical features, clothes, positions, "
    "objects, animals, fauna, flora, etc."
)
DEFAULT_MODEL_ID = "Qwen/Qwen3-VL-4B-Instruct"
DEFAULT_QUANT = "8-bit"  # "None" | "8-bit" | "4-bit"
DEFAULT_ATTN = "eager"
DEFAULT_MAX_TOKENS = 2048

JOYCAPTION_SYSTEM_PROMPT = "You are a helpful image captioner."

# Default sliders. batch_size starts at 1 (safe on any VRAM); a future
# commit may bump it after model load based on free VRAM.
# prefetch_workers default scales with the host CPU count but stays
# modest because preprocessing is much cheaper than GPU generation —
# extra workers mostly help once batch_size is high enough that the
# processor's per-batch CPU cost rivals model.generate.
DEFAULT_BATCH_SIZE = 1
DEFAULT_PREFETCH_WORKERS = max(1, min(4, (cpu_count() or 2) // 2))
MAX_BATCH_SIZE = 16
MAX_PREFETCH_WORKERS = 8


AVAILABLE_MODELS = [
    "Qwen/Qwen3.5-4B",
    "Qwen/Qwen3.5-9B",
    "Qwen/Qwen3-VL-4B-Instruct",
    "Qwen/Qwen3-VL-8B-Instruct",
    "Qwen/Qwen2.5-VL-3B-Instruct",
    "Qwen/Qwen2.5-VL-7B-Instruct",
    # Fastest and lightest. Good for rough captions, OCR, dense regions, and prepasses;
    # weaker for rich uncensored natural-language captions.
    "microsoft/Florence-2-large",
    "microsoft/Florence-2-large-ft",
    # JoyCaption Beta One (LLaVA: Llama 3.1 8B + SigLIP). Pick 4-bit quant
    # for 12 GB GPUs (~6 GB resident); 8-bit fits but is tight (~10 GB).
    # The pre-quantized `heavlav/...-4bit` mirror is intentionally NOT listed:
    # its vision tower is bnb-quantized and crashes inside SigLIP's
    # nn.MultiheadAttention (Byte vs BFloat16 dtype mismatch). Loading the
    # un-quantized fancyfeast checkpoint here lets us skip the vision tower
    # from quantization at load time, which sidesteps the bug.
    "fancyfeast/llama-joycaption-beta-one-hf-llava",
    "Custom...",
]


# Globals
processor: ProcessorMixin | None = None
current_model_id = DEFAULT_MODEL_ID
current_quant = DEFAULT_QUANT
model = None
should_abort = False
ui_e = {}


def preferred_compute_dtype():
    """bf16 on Ampere+ (sm_80+), fp16 elsewhere. bf16 has the same range
    as fp32 so it avoids the activation overflows fp16 can hit during
    matmul accumulation, at no speed cost on supported hardware."""
    if torch.cuda.is_available() and torch.cuda.get_device_capability(0) >= (8, 0):
        return torch.bfloat16
    return torch.float16


def build_bnb_config(quant_choice: str, skip_modules: list[str] | None = None):
    """`skip_modules` keeps the named submodules out of bnb quantization
    (despite the name, `llm_int8_skip_modules` is honored for both 4-bit and 8-bit).
    Useful for vision towers that internally call F.multi_head_attention_forward,
    which bypasses Linear4bit.forward and crashes on bnb-quantized weights with a dtype mismatch."""
    if quant_choice == "8-bit":
        return BitsAndBytesConfig(load_in_8bit=True, llm_int8_skip_modules=skip_modules)
    if quant_choice == "4-bit":
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=preferred_compute_dtype(),
            llm_int8_skip_modules=skip_modules,
        )
    return None


def is_qwen35_model(model_id: str) -> bool:
    """Qwen3.5 models use a different class and require thinking-mode handling."""
    return "Qwen3.5" in model_id or "Qwen3_5" in model_id


def is_joycaption_model(model_id: str) -> bool:
    """JoyCaption (LLaVA-based, Llama 3.1 8B + SigLIP) needs a different
    chat template, separate-image processor call, and sampling generation."""
    return "joycaption" in model_id.lower()


def _is_prequantized(model_id: str) -> bool:
    """Heuristic: model IDs ending in -4bit / -8bit are typically already
    bitsandbytes-quantized at save time. Re-applying a BnB config on top
    can conflict with the saved quantization_config — skip it and let HF
    load the baked-in settings."""
    return model_id.lower().endswith(("-4bit", "-8bit", "-4-bit", "-8-bit"))


def unload_model():
    global model, processor
    logger.debug("Unload currently loaded model: %r from %r", model, processor)
    model = None
    processor = None
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _configure_tokenizer_for_batching(processor_obj, model_obj) -> None:
    """Two batched-generation prerequisites that depend on the tokenizer:

    1. Left padding — required for correct batched causal-LM generation:
       all sequences must share the same right edge so model.generate
       continues each from its true last token. Right padding would make
       the model resume from pad tokens for the shorter sequences.

    2. pad_token fallback — Llama-family tokenizers (e.g. JoyCaption's
       Llama 3.1) ship without a pad_token, which breaks any processor
       call with padding=True. Alias to eos_token (standard Llama
       workaround; the attention mask still masks the pad positions) and
       propagate the id to model.generation_config so generate() doesn't
       fall back with a warning. No-op for Qwen-family tokenizers that
       already have a pad_token.
    """
    tokenizer: PreTrainedTokenizerBase | None = getattr(processor_obj, "tokenizer", None)
    if tokenizer is None:
        return
    if getattr(tokenizer, "padding_side", None) is not None:
        tokenizer.padding_side = "left"
    if getattr(tokenizer, "pad_token", None) is None:
        eos = getattr(tokenizer, "eos_token", None)
        if eos is not None:
            tokenizer.pad_token = eos
            gen_cfg = getattr(model_obj, "generation_config", None)
            if gen_cfg is not None:
                gen_cfg.pad_token_id = tokenizer.pad_token_id


def load_selected_model(model_id: str, quant_choice: str, attn_impl: str = DEFAULT_ATTN):
    """
    Loads (or reloads) the model + processor with chosen quantization and attention impl.
    Falls back to 'eager' if flash_attention_2 fails.
    """
    global model, processor, current_model_id, current_quant
    logger.debug("Loading selected model: %r", model_id)

    unload_model()

    kwargs: dict[str, Any] = {
        "dtype": torch.float16,
        "device_map": "auto",
        "attn_implementation": attn_impl,
    }
    # JoyCaption's SigLIP vision tower uses nn.MultiheadAttention internally
    # and crashes on bnb-quantized weights — skip it from quantization.
    # The tower is small (~0.4 GB at fp16) so the VRAM cost is negligible.
    skip_modules = ["vision_tower"] if is_joycaption_model(model_id) else None
    if not _is_prequantized(model_id) and (bnb := build_bnb_config(quant_choice, skip_modules)):
        kwargs["quantization_config"] = bnb

    if is_qwen35_model(model_id):
        model_cls = Qwen3_5ForConditionalGeneration
    elif "Qwen3-VL" in model_id:
        model_cls = Qwen3VLForConditionalGeneration
    elif "Qwen2.5-VL" in model_id or "Qwen2_5-VL" in model_id:
        model_cls = Qwen2_5_VLForConditionalGeneration
    elif is_joycaption_model(model_id):
        model_cls = LlavaForConditionalGeneration
    elif "Florence" in model_id:
        kwargs["trust_remote_code"] = True
        model_cls = AutoModelForCausalLM
    else:
        model_cls = AutoModelForImageTextToText

    try:
        model = model_cls.from_pretrained(model_id, **kwargs)
    except ImportError:
        if attn_impl == "flash_attention_2":
            kwargs["attn_implementation"] = "eager"
            model = model_cls.from_pretrained(model_id, **kwargs)
        else:
            raise

    processor = AutoProcessor.from_pretrained(model_id)
    _configure_tokenizer_for_batching(processor, model)

    current_model_id = model_id
    current_quant = quant_choice
    return get_model_info()


def toggle_controls(disabled=True):
    updates = {}
    for name, component in ui_e.items():
        if name == "abort_button":
            updates[name] = gradio.update(interactive=not disabled)
        else:
            updates[name] = gradio.update(interactive=not disabled if component.visible else False)
    return updates


def disable_controls_dict():
    return [toggle_controls(disabled=True)[k] for k in control_keys]


def enable_controls_dict():
    return [toggle_controls(disabled=False)[k] for k in control_keys]


control_keys = [
    "model_dropdown",
    "quant_dropdown",
    "attn_dropdown",
    "load_button",
    "reset_button",
    "start_button",
    "abort_button",
    "folder_input",
    "prompt_input",
    "skip_existing_checkbox",
    "caption_extension",
    "max_tokens_slider",
    "retain_preview_checkbox",
    "resolution_mode",
    "batch_size_slider",
    "prefetch_workers_slider",
    "status_output",
]


def finish_process():
    updates = enable_controls_dict()
    abort_index = control_keys.index("abort_button")
    updates[abort_index] = gradio.update(interactive=False)
    return updates


def abort_process():
    global should_abort
    should_abort = True
    updates = enable_controls_dict()
    abort_index = control_keys.index("abort_button")
    status_index = control_keys.index("status_output")
    updates[abort_index] = gradio.update(interactive=False)
    updates[status_index] = gradio.update(value="⛔ Aborting process...")
    return updates


def start_process():
    updates = disable_controls_dict()
    abort_index = control_keys.index("abort_button")
    updates[abort_index] = gradio.update(interactive=True)
    return updates


def serialize_for_debug(obj):
    if isinstance(obj, dict):
        return {k: serialize_for_debug(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [serialize_for_debug(i) for i in obj]
    elif isinstance(obj, Image.Image):
        return f"<Image {obj.size} {obj.mode}>"
    else:
        return obj


def get_model_info():
    global model
    if model is None:
        return "Model not loaded.", "N/A", "N/A", "N/A", "N/A"

    model = cast(SpecificPreTrainedModelType, model)
    model_name = getattr(model.config, "_name_or_path", "Unknown Model")
    device = "CUDA" if torch.cuda.is_available() else "CPU"

    if torch.cuda.is_available():
        vram_used = f"{torch.cuda.memory_allocated() / 1e9:.2f} GB"
        vram_total = f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB"
    else:
        vram_used, vram_total = "N/A", "N/A"

    dtype = str(next(model.parameters()).dtype)
    return model_name, device, f"{vram_used} / {vram_total}", dtype, str(model.config)


def suggest_batch_size() -> int:
    """Heuristic for the Batch Size slider after a model is loaded.
    Buckets free post-load VRAM into safe defaults — not a tight upper
    bound, just a reasonable starting point the user can raise.

    The free-VRAM number is what's left for activations + KV cache after
    the weights are resident. Per-sample cost grows with max_tokens and
    image resolution, both of which we don't know here, so we stay
    conservative and let the user climb."""
    if not torch.cuda.is_available() or model is None:
        return DEFAULT_BATCH_SIZE
    total = torch.cuda.get_device_properties(0).total_memory
    allocated = torch.cuda.memory_allocated(0)
    free_gb = max(0.0, (total - allocated) / 1e9)
    if free_gb >= 12:
        return 8
    if free_gb >= 8:
        return 4
    if free_gb >= 4:
        return 2
    return 1


logger.debug("CUDA availability: %s", torch.cuda.is_available())


def _resolution_kwargs(resolution_mode: str) -> dict[str, int]:
    if resolution_mode == "auto":
        return {"min_pixels": 256 * 28 * 28, "max_pixels": 896 * 28 * 28}
    if resolution_mode == "auto_high":
        return {"min_pixels": 256 * 28 * 28, "max_pixels": 1280 * 28 * 28}
    if resolution_mode == "fast":
        return {"resized_height": 392, "resized_width": 392}
    if resolution_mode == "high":
        return {"resized_height": 728, "resized_width": 728}
    return {}


def _build_messages(
    media_path: str,
    prompt: str,
    resolution_mode: str,
) -> list[dict[str, Any]]:
    is_video = is_video_file(media_path)
    content_type = "video" if is_video else "image"
    media_data = media_path if is_video else Image.open(media_path).convert("RGB")
    content_block: dict[str, Any] = {"type": content_type, content_type: media_data}
    if not is_video:
        content_block.update(_resolution_kwargs(resolution_mode))
    messages: list[dict[str, Any]] = [{"role": "user", "content": [content_block, {"type": "text", "text": prompt}]}]
    if is_qwen35_model(current_model_id):
        messages.append({"role": "assistant", "content": "<think>\n\n</think>\n\n"})
    return messages


def _preprocess_batch_joycaption(media_paths: list[str], prompt: str):
    """JoyCaption (LLaVA) preprocessing: hard-coded system prompt, the
    user-facing prompt drives the user turn, image is passed separately to
    the processor (no qwen_vl_utils, no resolution kwargs — the LLaVA image
    processor handles sizing internally). Image-only — videos are rejected."""
    global processor
    assert processor is not None
    messages = [
        {"role": "system", "content": JOYCAPTION_SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]
    convo_string = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    images: list[Image.Image] = []
    for path in media_paths:
        if is_video_file(path):
            raise ValueError(f"JoyCaption does not support video input: {path}")
        images.append(Image.open(path).convert("RGB"))
    return processor(
        text=[convo_string] * len(media_paths),
        images=images,
        padding=True,
        return_tensors="pt",
    )


def preprocess_batch(
    media_paths: list[str],
    prompt: str,
    resolution_mode: str = "auto",
):
    """Build a BatchFeature with batch dim len(media_paths) on CPU.
    Concatenates per-sample images/videos in positional order
    — the processor matches the i-th `<|image_pad|>` text token
    to the i-th image across the flat list, so order matters."""
    global processor
    assert processor is not None, "Processor must be loaded before preprocessing."
    if not media_paths:
        raise ValueError("preprocess_batch requires at least one media path")
    if is_joycaption_model(current_model_id):
        return _preprocess_batch_joycaption(media_paths, prompt)
    qwen35 = is_qwen35_model(current_model_id)
    texts: list[str] = []
    images_acc: list[Any] = []
    videos_acc: list[Any] = []
    for path in media_paths:
        messages = _build_messages(path, prompt, resolution_mode)
        texts.append(
            processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=not qwen35,
                continue_final_message=qwen35,
            )
        )
        imgs, vids = process_vision_info(messages)[:2]
        if imgs:
            images_acc.extend(imgs)
        if vids:
            videos_acc.extend(vids)
    return processor(
        text=texts,
        images=images_acc or None,
        videos=videos_acc or None,
        padding=True,
        return_tensors="pt",
    )


def preprocess_one(media_path: str, prompt: str, resolution_mode: str = "auto"):
    """Single-sample wrapper around preprocess_batch. Safe to call from
    worker threads while the GPU is busy (touches only the processor)."""
    return preprocess_batch([media_path], prompt, resolution_mode)


def _generation_kwargs(model_id: str, max_tokens: int) -> dict[str, Any]:
    """Per-family generation kwargs. JoyCaption upstream recommends
    sampling (temperature=0.6, top_p=0.9); other families stay greedy to
    preserve the existing per-sample behavior."""
    kwargs: dict[str, Any] = {"max_new_tokens": max_tokens}
    if is_joycaption_model(model_id):
        kwargs.update(
            {
                "do_sample": True,
                "temperature": 0.6,
                "top_p": 0.9,
                "suppress_tokens": None,
                "use_cache": True,
            }
        )
    return kwargs


def run_generate(inputs, max_tokens: int):
    """Move a CPU BatchFeature (any batch dim) to GPU in place, run
    model.generate, return (generated_ids, input_ids) on CPU. Mutates
    `inputs` (the .to() call rebinds its tensors to GPU); caller should
    not reuse the CPU view."""
    global model
    assert model is not None, "Model must be loaded before generating."
    model = cast(GenerationMixin, model)
    inputs_gpu = inputs.to("cuda", non_blocking=True)
    if is_joycaption_model(current_model_id) and "pixel_values" in inputs_gpu:
        # LLaVA's vision tower is not BnB-quantized; processor returns
        # pixel_values as fp32. Cast to the vision tower's dtype so the
        # SigLIP forward doesn't trip on a dtype mismatch.
        # In transformers 5.x, LlavaForConditionalGeneration nests it as
        # `.model.vision_tower` (the inner LlavaModel owns the tower).
        vt_dtype = next(model.model.vision_tower.parameters()).dtype
        inputs_gpu["pixel_values"] = inputs_gpu["pixel_values"].to(vt_dtype)
    gen_kwargs = _generation_kwargs(current_model_id, max_tokens)
    with torch.inference_mode():
        generated_ids = model.generate(**inputs_gpu, **gen_kwargs)
    return generated_ids.cpu(), inputs_gpu["input_ids"].cpu()


def decode_batch(generated_ids, input_ids) -> list[str]:
    """Decode a [B, L_out] generated_ids tensor into a list of B caption
    strings, trimming each by the corresponding input length and
    stripping Qwen3.5 think tags when present."""
    global processor
    assert processor is not None, "Processor must be loaded before decoding."
    trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(input_ids, generated_ids)]
    captions = processor.batch_decode(
        trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    if is_qwen35_model(current_model_id):
        captions = [r_caption.sub("", c) if "<think>" in c else c for c in captions]
    return [c.strip() for c in captions]


def decode_one(generated_ids, input_ids) -> str:
    """Single-sample wrapper around decode_batch."""
    return decode_batch(generated_ids, input_ids)[0]


def generate_caption(media_path, prompt, max_tokens, resolution_mode="auto"):
    """Single-sample orchestrator preserved for back-compat with callers
    that don't yet use the prefetched/batched pipeline."""
    t_start = monotonic()
    inputs = preprocess_one(media_path, prompt, resolution_mode)
    t_prep_end = monotonic()
    generated_ids, input_ids = run_generate(inputs, max_tokens)
    t_gen_end = monotonic()
    caption = decode_one(generated_ids, input_ids)
    t_dec_end = monotonic()

    logger.debug(
        "caption timings (s): prep=%.3f gen=%.3f dec=%.3f total=%.3f path=%s",
        t_prep_end - t_start,
        t_gen_end - t_prep_end,
        t_dec_end - t_gen_end,
        t_dec_end - t_start,
        media_path,
    )

    return caption


def is_image_file(filename):
    return filename.lower().endswith(IMAGE_EXTENSIONS)


def is_video_file(filename):
    return filename.lower().endswith(VIDEO_EXTENSIONS)


def _format_elapsed_str(start_time: float) -> str:
    elapsed = int(monotonic() - start_time)
    return f"{elapsed // 60:02d}:{elapsed % 60:02d}"


def _sanitize_caption_extension(ext: str) -> str:
    """Strip leading dots and whitespace; fall back to 'txt' if empty."""
    cleaned = (ext or "").strip().lstrip(".").strip()
    return cleaned or "txt"


def _caption_path_for(media_path: str, extension: str = "txt") -> Path:
    return Path(media_path).with_suffix(f".{extension}")


def _put_until_cancel(q: Queue, item: Any, cancel: Event) -> bool:
    """Block on q.put with periodic cancel checks. Returns False if cancelled."""
    while not cancel.is_set():
        try:
            q.put(item, timeout=0.25)
            return True
        except Full:
            continue
    return False


def _prefetch_dispatcher(
    media_files: list[str],
    batch_size: int,
    prompt: str,
    resolution_mode: str,
    skip_existing: bool,
    caption_extension: str,
    executor: ThreadPoolExecutor,
    out_queue: Queue,
    cancel: Event,
) -> None:
    """Walks media_files in order; emits queue items in submission order:
        ("skip",  idx,     path,    None)             — already captioned
        ("batch", indices, paths,   Future[BatchFeature])
    Skips don't go through the executor. Non-skipped runs are grouped
    into contiguous batches of up to batch_size and submitted as
    Futures. The consumer consumes in order; the executor's pool of
    workers drives parallel preprocessing of the in-flight Futures."""
    try:
        i = 0
        n = len(media_files)
        while i < n:
            if should_abort or cancel.is_set():
                return
            path_i = media_files[i]
            if skip_existing and _caption_path_for(path_i, caption_extension).exists():
                if not _put_until_cancel(out_queue, ("skip", i, path_i, None), cancel):
                    return
                i += 1
                continue
            indices = [i]
            paths = [path_i]
            j = i + 1
            while j < n and len(indices) < batch_size:
                pj = media_files[j]
                if skip_existing and _caption_path_for(pj, caption_extension).exists():
                    break
                indices.append(j)
                paths.append(pj)
                j += 1
            try:
                fut: Future = executor.submit(
                    preprocess_batch,
                    paths,
                    prompt,
                    resolution_mode,
                )
            except RuntimeError:
                # executor was shut down (abort path)
                return
            if not _put_until_cancel(out_queue, ("batch", indices, paths, fut), cancel):
                fut.cancel()
                return
            i = j
    finally:
        try:
            out_queue.put(None, timeout=0.25)
        except Full:
            pass


def _build_abort_yield():
    status_index = control_keys.index("status_output")
    abort_index = control_keys.index("abort_button")
    control_updates = enable_controls_dict()
    control_updates[status_index] = gradio.update(value="⛔ Aborted by user.")
    control_updates[abort_index] = gradio.update(interactive=False)
    return "⛔ Aborted by user.", None, None, "Aborted.", 0, "", *control_updates


def _resolve_batch(future: Future, max_tokens: int):
    """Await the preprocess Future, run the GPU step, decode. Returns
    (captions, n_samples, error_or_none). On error, captions is None."""
    try:
        inputs = future.result()
    except Exception as e:
        logger.exception("Preprocess failed: %s", e)
        return None, 0, e
    try:
        t_gen_start = monotonic()
        generated_ids, input_ids = run_generate(inputs, max_tokens)
        t_gen_end = monotonic()
        captions = decode_batch(generated_ids, input_ids)
        logger.debug(
            "batch timings (s): gen=%.3f dec=%.3f n=%d",
            t_gen_end - t_gen_start,
            monotonic() - t_gen_end,
            len(captions),
        )
        return captions, len(captions), None
    except Exception as e:
        logger.exception("Generation failed: %s", e)
        return None, 0, e


def _log_run_summary(action: str, processed: int, skipped: int, failed: int, start_time: float) -> None:
    total_elapsed = monotonic() - start_time
    avg = total_elapsed / processed if processed else 0.0
    logger.info(
        "processing %s: processed=%d skipped=%d failed=%d total=%.3fs avg=%.3fs/media",
        action, processed, skipped, failed, total_elapsed, avg,
    )


def _captioning_loop(
    folder_path, total_media, max_tokens, retain_preview, caption_extension, prefetch_queue, start_time
):
    """Consumer side of the prefetch pipeline. Yields Gradio update tuples.
    Reads `should_abort` (and resets it on consumption) so the abort
    button works mid-loop. A `try/finally` guarantees the run-summary
    log line always fires (clean completion, abort, or unexpected
    exception) — `completed_action` records which path won so the log
    label is accurate."""
    global should_abort
    processed_media = 0
    skipped_media = 0
    failed_media = 0
    last_media_to_show = None
    last_caption = ""
    last_name_md = ""
    elapsed_str = ""
    completed_action = "interrupted"

    try:
        while True:
            if should_abort:
                should_abort = False
                completed_action = "aborted"
                yield _build_abort_yield()
                return

            try:
                item = prefetch_queue.get(timeout=0.25)
            except Empty:
                continue
            if item is None:
                break

            kind = item[0]

            if kind == "skip":
                _, idx, media_path, _ = item
                rel_path = Path(media_path).relative_to(folder_path)
                skipped_media += 1
                elapsed_str = _format_elapsed_str(start_time)
                percent = int(((idx + 1) / total_media) * 100)
                logger.info(
                    "[%d/%d %3d%%] skipped %s (elapsed %s)",
                    idx + 1, total_media, percent, rel_path, elapsed_str,
                )
                yield (
                    f"⏭️ Skipped {idx + 1}/{total_media}: {rel_path} (already captioned)",
                    last_media_to_show if retain_preview else None,
                    last_name_md if retain_preview else None,
                    last_caption if retain_preview else "Skipped (already captioned)",
                    percent,
                    elapsed_str,
                    *start_process(),
                )
                continue

            # kind == "batch"
            _, indices, paths, future = item
            captions, _n, err = _resolve_batch(future, max_tokens)
            if err is not None:
                failed_media += len(indices)
                for idx, media_path in zip(indices, paths):
                    rel_path = Path(media_path).relative_to(folder_path)
                    logger.error(
                        "[%d/%d] error processing %s: %s",
                        idx + 1, total_media, rel_path, err,
                    )
                    yield (
                        f"⚠️ Error processing {media_path}: {err}",
                        None,
                        None,
                        "Error in captioning.",
                        0,
                        elapsed_str,
                        *start_process(),
                    )
                continue

            for idx, media_path, caption in zip(indices, paths, captions):
                try:
                    with open(_caption_path_for(media_path, caption_extension), "w", encoding="utf-8") as f:
                        f.write(caption)
                except OSError as e:
                    rel_path = Path(media_path).relative_to(folder_path)
                    logger.exception(
                        "[%d/%d] write failed for %s",
                        idx + 1, total_media, rel_path,
                    )
                    failed_media += 1
                    yield (
                        f"⚠️ Error writing {media_path}: {e}",
                        None,
                        None,
                        "Error in captioning.",
                        0,
                        elapsed_str,
                        *start_process(),
                    )
                    continue

                rel_path = Path(media_path).relative_to(folder_path)
                name_md = f"**File:** `{rel_path}`"
                media_to_show = Image.open(media_path) if is_image_file(media_path) else None
                elapsed_str = _format_elapsed_str(start_time)
                percent = int(((idx + 1) / total_media) * 100)
                last_media_to_show = media_to_show
                last_caption = caption
                last_name_md = name_md
                processed_media += 1
                logger.info(
                    "[%d/%d %3d%%] captioned %s (elapsed %s)",
                    idx + 1, total_media, percent, rel_path, elapsed_str,
                )
                yield (
                    f"🖼️ Processing {idx + 1}/{total_media}: {rel_path}",
                    media_to_show,
                    name_md,
                    caption,
                    percent,
                    elapsed_str,
                    *start_process(),
                )

        completed_action = "complete"
        yield (
            "✅ Processing complete!"
            f"processed {processed_media} media in {elapsed_str}, skipped {skipped_media} media."
            f"Failed to process {failed_media} media (inaccessible, unknown or broken file)",
            last_media_to_show,
            last_name_md,
            last_caption,
            None,
            None,
            *finish_process(),
        )
    finally:
        _log_run_summary(completed_action, processed_media, skipped_media, failed_media, start_time)


def process_folder(
    folder_path,
    prompt,
    skip_existing,
    caption_extension,
    max_tokens,
    retain_preview,
    resolution_mode,
    batch_size,
    prefetch_workers,
):
    batch_size = max(1, int(batch_size))
    prefetch_workers = max(1, int(prefetch_workers))
    caption_extension = _sanitize_caption_extension(caption_extension)

    logger.debug(
        "starting folder processing: model=%s quant=%s folder=%s skip_existing=%s "
        "caption_extension=%s max_tokens=%s retain_preview=%s "
        "resolution=%s batch_size=%s prefetch_workers=%s should_abort=%s",
        current_model_id,
        current_quant,
        folder_path,
        skip_existing,
        caption_extension,
        max_tokens,
        retain_preview,
        resolution_mode,
        batch_size,
        prefetch_workers,
        should_abort,
    )

    if not folder_path.strip():
        yield "⚠️ Please enter a valid folder path.", None, None, "No media to process.", 0, "", *finish_process()
        return

    folder = Path(folder_path)
    if not folder.exists():
        yield f"❌ Folder not found: {folder_path}", None, None, "No media to process.", 0, "", *finish_process()
        return

    media_files = [
        str(p)
        for p in folder.rglob("*")
        if p.is_file() and (is_image_file(p.name) or is_video_file(p.name))
    ]
    total_media = len(media_files)
    if not total_media:
        yield (
            "📂 No media found in the folder or subfolders.",
            None,
            None,
            "No media to process.",
            0,
            "",
            *finish_process(),
        )
        return

    start_time = monotonic()

    # Bound the in-flight Futures so we don't preprocess far ahead of
    # the GPU. Total in-flight ≈ queue_size + workers (queue holds
    # already-submitted, workers each may hold one running task).
    prefetch_queue: Queue = Queue(maxsize=max(2, prefetch_workers * 2))
    cancel = Event()
    executor = ThreadPoolExecutor(max_workers=prefetch_workers, thread_name_prefix="caption-prep")
    dispatcher = Thread(
        target=_prefetch_dispatcher,
        name="caption-dispatch",
        daemon=True,
        kwargs={
            "media_files": media_files,
            "batch_size": batch_size,
            "prompt": prompt,
            "resolution_mode": resolution_mode,
            "skip_existing": skip_existing,
            "caption_extension": caption_extension,
            "executor": executor,
            "out_queue": prefetch_queue,
            "cancel": cancel,
        },
    )
    dispatcher.start()

    try:
        yield from _captioning_loop(
            folder_path, total_media, max_tokens, retain_preview, caption_extension, prefetch_queue, start_time
        )
    finally:
        cancel.set()
        # Drain so a dispatcher blocked on .put(...) can wake.
        try:
            while True:
                prefetch_queue.get_nowait()
        except Empty:
            pass
        # Cancel queued (not-yet-running) preprocessing tasks; running ones
        # finish naturally. wait=False so we don't block the generator.
        executor.shutdown(wait=False, cancel_futures=True)
        dispatcher.join(timeout=2.0)


basicConfig(
    level="INFO",
)
logger.setLevel("DEBUG")

with gradio.Blocks() as iface:  # type: ignore
    gradio.Markdown("# Simple Captioner")
    gradio.Markdown(
        "A simple media caption generator for images and video using **[Qwen2.5/3/3.5 VL Instruct](https://huggingface.co/Qwen/)**"
        "Written by [Olli S.](https://github.com/o-l-l-i)"
    )

    with gradio.Accordion("⚙️ Model Settings", open=True):
        ui_e["model_dropdown"] = model_dropdown = gradio.Dropdown(
            label="Model",
            choices=AVAILABLE_MODELS,
            value=DEFAULT_MODEL_ID,
            allow_custom_value=True,
            interactive=True,
            info="Pick a model to use for captioning.",
        )
        custom_model_box = gradio.Textbox(
            label="Custom Model ID (Hugging Face)",
            placeholder="e.g. Qwen/Qwen3-VL-4B-Instruct or your-org/my-qwen3-checkpoint",
            visible=False,
        )
        ui_e["quant_dropdown"] = quant_dropdown = gradio.Radio(
            label="Quantization",
            choices=["None", "8-bit", "4-bit"],
            value=DEFAULT_QUANT,
            interactive=True,
            info="Lower-bit quantization reduces VRAM, may slightly affect quality.",
        )
        ui_e["attn_dropdown"] = attn_dropdown = gradio.Radio(
            label="Attention Implementation",
            choices=["flash_attention_2", "eager"],
            value=DEFAULT_ATTN,
            interactive=True,
            info="If FlashAttention isn't installed/working, choose 'eager'. Auto-fallback on load.",
        )
        ui_e["load_button"] = gradio.Button("📦 Load / Reload Model")

    def _toggle_custom(choice):
        return gradio.update(visible=(choice == "Custom..."))

    model_dropdown.change(_toggle_custom, inputs=[model_dropdown], outputs=[custom_model_box])

    def _ui_load_model(sel, custom_id, quant, attn):
        model_id = custom_id.strip() if sel == "Custom..." and custom_id and custom_id.strip() else sel
        name, device, vram, dtype, cfg = load_selected_model(model_id, quant, attn)
        suggested_bs = suggest_batch_size()
        status = f"✅ Loaded '{model_id}' with {quant} quantization ({attn}). Suggested batch size: {suggested_bs}."
        logger.debug("status: %s", status)
        return status, name, device, vram, dtype, cfg, gradio.update(value=suggested_bs)

    with gradio.Accordion("⚙️ Model Information", open=False):
        model_name_display = gradio.Textbox(label="Model Name", interactive=False)
        device_display = gradio.Textbox(label="Device", interactive=False)
        vram_display = gradio.Textbox(label="VRAM Usage", interactive=False)
        dtype_display = gradio.Textbox(label="Torch Dtype", interactive=False)
        config_display = gradio.Textbox(label="Model Config", interactive=False, lines=4)

    with gradio.Row():
        ui_e["folder_input"] = gradio.Textbox(
            label="📁 Folder Path",
            placeholder=r"e.g. C:\Users\you\Pictures\input_images",
        )
        ui_e["prompt_input"] = gradio.Textbox(label="Custom Prompt", value=DEFAULT_PROMPT)

    with gradio.Row():
        ui_e["skip_existing_checkbox"] = gradio.Checkbox(
            label="Skip already captioned media (caption file exists)", value=True
        )
        ui_e["caption_extension"] = gradio.Textbox(
            label="Caption File Extension",
            value="txt",
            info="Extension for the generated caption files, without the leading dot. e.g. 'txt', 'cap'.",
        )

    with gradio.Row():
        ui_e["max_tokens_slider"] = gradio.Slider(
            label="🧾 Max Tokens", minimum=32, maximum=1024 * 8, value=DEFAULT_MAX_TOKENS, step=16
        )
        ui_e["resolution_mode"] = gradio.Dropdown(
            label="Image Resolution",
            choices=["auto", "auto_high", "fast", "high"],
            value="auto",
            info="Choose the resolution mode for visual input.",
        )

    with gradio.Row():
        ui_e["batch_size_slider"] = gradio.Slider(
            label="📦 Batch Size",
            minimum=1,
            maximum=MAX_BATCH_SIZE,
            value=DEFAULT_BATCH_SIZE,
            step=1,
            info=(
                "Samples per model.generate() call. Higher = better GPU utilization, "
                "but VRAM scales roughly linearly. Start at 1 and raise until VRAM "
                "headroom shrinks."
            ),
        )
        ui_e["prefetch_workers_slider"] = gradio.Slider(
            label="🧵 Prefetch Workers",
            minimum=1,
            maximum=MAX_PREFETCH_WORKERS,
            value=DEFAULT_PREFETCH_WORKERS,
            step=1,
            info=(
                f"CPU threads preprocessing batches ahead of the GPU. "
                f"Default {DEFAULT_PREFETCH_WORKERS} guessed from "
                f"cpu_count={cpu_count() or '?'}."
            ),
        )

    with gradio.Row():
        ui_e["reset_button"] = gradio.Button("🔄 Reset to Default Prompt")
        ui_e["start_button"] = gradio.Button("🚀 Start Processing", interactive=True)
        ui_e["abort_button"] = gradio.Button("⛔ Abort", interactive=False)

    ui_e["status_output"] = gradio.Textbox(label="Status", interactive=False)
    progress_bar = gradio.Slider(minimum=0, maximum=100, label="Progress", interactive=False)
    time_display = gradio.Textbox(label="⏱️ Time Taken (s)", interactive=False)

    with gradio.Row():
        with gradio.Column(scale=1):
            media_output = gradio.Image(label="Current Image", interactive=False)
            media_name_markdown = gradio.Markdown()
            ui_e["retain_preview_checkbox"] = gradio.Checkbox(label="Retain preview on skip", value=True)
        with gradio.Column(scale=1):
            caption_output = gradio.Textbox(label="Generated Caption", interactive=False)

    ui_e["start_button"].click(start_process, inputs=[], outputs=[ui_e[k] for k in control_keys])

    ui_e["start_button"].click(
        process_folder,
        inputs=[
            ui_e["folder_input"],
            ui_e["prompt_input"],
            ui_e["skip_existing_checkbox"],
            ui_e["caption_extension"],
            ui_e["max_tokens_slider"],
            ui_e["retain_preview_checkbox"],
            ui_e["resolution_mode"],
            ui_e["batch_size_slider"],
            ui_e["prefetch_workers_slider"],
        ],
        outputs=[
            ui_e["status_output"],
            media_output,
            media_name_markdown,
            caption_output,
            progress_bar,
            time_display,
            *[ui_e[k] for k in control_keys],
        ],
    )

    ui_e["abort_button"].click(
        fn=abort_process,
        inputs=[],
        outputs=[ui_e[k] for k in control_keys],
        queue=False,
    )

    ui_e["reset_button"].click(lambda: DEFAULT_PROMPT, inputs=[], outputs=[ui_e["prompt_input"]])
    ui_e["start_button"].click(
        get_model_info,
        inputs=[],
        outputs=[
            model_name_display,
            device_display,
            vram_display,
            dtype_display,
            config_display,
        ],
    )

    ui_e["load_button"].click(
        _ui_load_model,
        inputs=[model_dropdown, custom_model_box, quant_dropdown, attn_dropdown],
        outputs=[
            ui_e["status_output"],
            model_name_display,
            device_display,
            vram_display,
            dtype_display,
            config_display,
            ui_e["batch_size_slider"],
        ],
    )
    gradio.Blocks.load(
        iface,
        get_model_info,
        inputs=[],
        outputs=[
            model_name_display,
            device_display,
            vram_display,
            dtype_display,
            config_display,
        ],
    )


iface.launch(
    share=False,
    theme=gradio.themes.Base(),
    css="""
.generating {
    border: none;
}
""",
)
