"""
Dependencies are slow to load, especially qwen, torch and transformers;
moved the minimal required dependencies to common
"""

# stdlib
import re
from collections.abc import Iterator
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from logging import getLogger
from pathlib import Path
from queue import Empty, Full, Queue
from threading import Event, Thread
from time import monotonic
from typing import Any, cast

# dependencies
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

# project
from .common import (
    DEFAULT_ATTN,
    DEFAULT_BATCH_SIZE,
    DEFAULT_MODEL_ID,
    DEFAULT_PREFETCH_WORKERS,
    DEFAULT_QUANT,
    IMAGE_EXTENSIONS,
    VIDEO_EXTENSIONS,
    abort_event,
)

logger = getLogger(__name__)

r_caption = re.compile(r"<think>.*?</think>\s*", re.DOTALL)

JOYCAPTION_SYSTEM_PROMPT = "You are a helpful image captioner."


# Module-level mutable state. Rebound via load_selected_model /
# unload_model; consumed by preprocess_batch / run_generate /
# decode_batch. The Gradio UI handlers and the CLI both share this
# state — there is only ever one loaded model per process.
processor: ProcessorMixin | None = None
current_model_id = DEFAULT_MODEL_ID
current_quant = DEFAULT_QUANT
model = None


@dataclass(frozen=True, kw_only=True)
class BaseEvent:
    """Fields common to every event yielded by caption_folder().
    kw_only=True keeps subclass fields positional-friendly: the inherited
    `total` / `elapsed_s` must always be passed as keywords, so subclasses
    are free to declare their own fields without colliding with the base
    field order."""

    total: int
    elapsed_s: float


@dataclass(frozen=True, kw_only=True)
class SkipEvent(BaseEvent):
    idx: int
    path: str


@dataclass(frozen=True, kw_only=True)
class CaptionedEvent(BaseEvent):
    idx: int
    path: str
    caption: str


@dataclass(frozen=True, kw_only=True)
class ErrorEvent(BaseEvent):
    idx: int
    path: str
    error: BaseException


@dataclass(frozen=True, kw_only=True)
class CompleteEvent(BaseEvent):
    processed: int
    skipped: int
    failed: int


@dataclass(frozen=True, kw_only=True)
class AbortedEvent(BaseEvent):
    processed: int
    skipped: int
    failed: int


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


def format_elapsed(elapsed_s: float) -> str:
    """Format seconds as MM:SS for display in status lines."""
    elapsed = int(elapsed_s)
    return f"{elapsed // 60:02d}:{elapsed % 60:02d}"


def sanitize_caption_extension(ext: str) -> str:
    """Strip leading dots and whitespace; fall back to 'txt' if empty."""
    cleaned = (ext or "").strip().lstrip(".").strip()
    return cleaned or "txt"


def caption_path_for(media_path: str, extension: str = "txt") -> Path:
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
            if abort_event.is_set() or cancel.is_set():
                return
            path_i = media_files[i]
            if skip_existing and caption_path_for(path_i, caption_extension).exists():
                if not _put_until_cancel(out_queue, ("skip", i, path_i, None), cancel):
                    return
                i += 1
                continue
            indices = [i]
            paths = [path_i]
            j = i + 1
            while j < n and len(indices) < batch_size:
                pj = media_files[j]
                if skip_existing and caption_path_for(pj, caption_extension).exists():
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


def _consume_batch_item(
    item: tuple,
    total: int,
    folder: Path,
    caption_extension: str,
    max_tokens: int,
    start: float,
) -> Iterator[BaseEvent]:
    """Resolve one 'batch' queue item: await the preprocess Future, run
    model.generate, write caption files, and yield one per-sample event
    (CaptionedEvent on success, ErrorEvent on preprocess/generate/write
    failure). The caller tallies events by type for the run stats."""
    _, indices, paths, future = item
    captions, _n, err = _resolve_batch(future, max_tokens)
    if err is not None:
        elapsed_s = monotonic() - start
        for idx, media_path in zip(indices, paths):
            rel_path = Path(media_path).relative_to(folder)
            logger.error(
                "[%d/%d] error processing %s: %s",
                idx + 1,
                total,
                rel_path,
                err,
            )
            yield ErrorEvent(total=total, elapsed_s=elapsed_s, idx=idx, path=media_path, error=err)
        return

    for idx, media_path, caption in zip(indices, paths, captions):
        try:
            with open(caption_path_for(media_path, caption_extension), "w", encoding="utf-8") as f:
                f.write(caption)
        except OSError as e:
            rel_path = Path(media_path).relative_to(folder)
            logger.exception(
                "[%d/%d] write failed for %s",
                idx + 1,
                total,
                rel_path,
            )
            yield ErrorEvent(total=total, elapsed_s=monotonic() - start, idx=idx, path=media_path, error=e)
            continue

        rel_path = Path(media_path).relative_to(folder)
        elapsed_s = monotonic() - start
        percent = int(((idx + 1) / total) * 100)
        logger.info(
            "[%d/%d %3d%%] captioned %s (elapsed %s)",
            idx + 1,
            total,
            percent,
            rel_path,
            format_elapsed(elapsed_s),
        )
        yield CaptionedEvent(total=total, elapsed_s=elapsed_s, idx=idx, path=media_path, caption=caption)


def _log_run_summary(action: str, processed: int, skipped: int, failed: int, start_time: float) -> None:
    total_elapsed = monotonic() - start_time
    avg = total_elapsed / processed if processed else 0.0
    logger.info(
        "processing %s: processed=%d skipped=%d failed=%d total=%.3fs avg=%.3fs/media",
        action,
        processed,
        skipped,
        failed,
        total_elapsed,
        avg,
    )


def caption_folder(  # noqa: C901  — coordinator: setup + consumer loop + teardown
    folder_path: str,
    prompt: str,
    skip_existing: bool,
    caption_extension: str,
    max_tokens: int,
    resolution_mode: str = "auto",
    batch_size: int = DEFAULT_BATCH_SIZE,
    prefetch_workers: int = DEFAULT_PREFETCH_WORKERS,
) -> Iterator[BaseEvent]:
    """Caption every image/video under `folder_path`, writing each
    caption next to its source file (sibling `.<caption_extension>`).
    Yields one event per sample (SkipEvent / CaptionedEvent /
    ErrorEvent) plus a terminal CompleteEvent or AbortedEvent.

    Cooperative cancellation: another thread calls request_abort();
    the in-flight batch finishes and the next iteration yields
    AbortedEvent. The abort_event is cleared at the start of each
    call, so leftover state from a previous run doesn't bleed in.

    Raises FileNotFoundError if `folder_path` doesn't exist (the
    caller is expected to validate input). Raises if no model is
    loaded — call load_selected_model() first."""
    batch_size = max(1, int(batch_size))
    prefetch_workers = max(1, int(prefetch_workers))
    caption_extension = sanitize_caption_extension(caption_extension)
    abort_event.clear()

    logger.info(
        "starting folder processing: model=%s quant=%s folder=%s skip_existing=%s "
        "caption_extension=%s max_tokens=%s resolution=%s batch_size=%s "
        "prefetch_workers=%s prompt=%r",
        current_model_id,
        current_quant,
        folder_path,
        skip_existing,
        caption_extension,
        max_tokens,
        resolution_mode,
        batch_size,
        prefetch_workers,
        prompt,
    )

    folder = Path(folder_path)
    if not folder.exists():
        raise FileNotFoundError(folder_path)

    media_files = [
        str(p) for p in folder.rglob("*") if p.is_file() and (is_image_file(p.name) or is_video_file(p.name))
    ]
    total = len(media_files)
    start = monotonic()

    if total == 0:
        logger.info("no media found under %s", folder_path)
        _log_run_summary("complete", 0, 0, 0, start)
        yield CompleteEvent(total=0, elapsed_s=monotonic() - start, processed=0, skipped=0, failed=0)
        return

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

    processed = 0
    skipped = 0
    failed = 0
    completed_action = "interrupted"

    try:
        while True:
            if abort_event.is_set():
                completed_action = "aborted"
                yield AbortedEvent(
                    total=total,
                    elapsed_s=monotonic() - start,
                    processed=processed,
                    skipped=skipped,
                    failed=failed,
                )
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
                rel_path = Path(media_path).relative_to(folder)
                skipped += 1
                elapsed_s = monotonic() - start
                percent = int(((idx + 1) / total) * 100)
                logger.info(
                    "[%d/%d %3d%%] skipped %s (elapsed %s)",
                    idx + 1,
                    total,
                    percent,
                    rel_path,
                    format_elapsed(elapsed_s),
                )
                yield SkipEvent(total=total, elapsed_s=elapsed_s, idx=idx, path=media_path)
                continue

            # kind == "batch"
            for event in _consume_batch_item(item, total, folder, caption_extension, max_tokens, start):
                if isinstance(event, CaptionedEvent):
                    processed += 1
                else:
                    failed += 1
                yield event

        completed_action = "complete"
        yield CompleteEvent(
            total=total,
            elapsed_s=monotonic() - start,
            processed=processed,
            skipped=skipped,
            failed=failed,
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
        _log_run_summary(completed_action, processed, skipped, failed, start)
