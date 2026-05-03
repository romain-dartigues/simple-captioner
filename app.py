# stdlib
import os
import re
import threading
from concurrent.futures import Future, ThreadPoolExecutor
from logging import basicConfig, getLogger
from queue import Empty, Full, Queue
from time import monotonic
from typing import Any

# dependencies
import gradio
import torch
from PIL import Image
from qwen_vl_utils import process_vision_info
from transformers.models.auto.modeling_auto import AutoModelForImageTextToText
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLForConditionalGeneration
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForConditionalGeneration
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLForConditionalGeneration
from transformers.utils.quantization_config import BitsAndBytesConfig

logger = getLogger(__name__)

r_caption = re.compile(r"<think>.*?</think>\s*", re.DOTALL)

IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".gif", ".webp")
VIDEO_EXTENSIONS = (".mp4", ".mov", ".avi", ".webm", ".mkv", ".gif", ".flv")
DEFAULT_PROMPT = (
    "In a concise way, describe this media, it's background,"
    "composition, style, people, acts, racial traits, physical features, clothes, positions, objects, etc."
)
DEFAULT_MODEL_ID = "Qwen/Qwen3-VL-4B-Instruct"
DEFAULT_QUANT = "8-bit"  # "None" | "8-bit" | "4-bit"
DEFAULT_ATTN = "eager" if os.name == "nt" else "flash_attention_2"

# Default sliders. batch_size starts at 1 (safe on any VRAM); a future
# commit may bump it after model load based on free VRAM.
# prefetch_workers default scales with the host CPU count but stays
# modest because preprocessing is much cheaper than GPU generation —
# extra workers mostly help once batch_size is high enough that the
# processor's per-batch CPU cost rivals model.generate.
DEFAULT_BATCH_SIZE = 1
DEFAULT_PREFETCH_WORKERS = max(1, min(4, (os.cpu_count() or 2) // 2))
MAX_BATCH_SIZE = 16
MAX_PREFETCH_WORKERS = 8


AVAILABLE_MODELS = [
    "Qwen/Qwen3.5-4B",
    "Qwen/Qwen3.5-9B",
    "Qwen/Qwen3-VL-4B-Instruct",
    "Qwen/Qwen3-VL-8B-Instruct",
    "Qwen/Qwen2.5-VL-3B-Instruct",
    "Qwen/Qwen2.5-VL-7B-Instruct",
    "Salesforce/blip-image-captioning-base",
    "Custom...",
]


# Globals
processor = None
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


def build_bnb_config(quant_choice: str):
    if quant_choice == "8-bit":
        return BitsAndBytesConfig(load_in_8bit=True)
    if quant_choice == "4-bit":
        return BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=preferred_compute_dtype())
    return None


def is_qwen35_model(model_id: str) -> bool:
    """Qwen3.5 models use a different class and require thinking-mode handling."""
    return "Qwen3.5" in model_id or "Qwen3_5" in model_id


def unload_model():
    global model, processor
    logger.debug("Unload currently loaded model: %r from %r", model, processor)
    model = None
    processor = None
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


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
    if bnb := build_bnb_config(quant_choice):
        kwargs["quantization_config"] = bnb

    if is_qwen35_model(model_id):
        model_cls = Qwen3_5ForConditionalGeneration
    elif "Qwen3-VL" in model_id:
        model_cls = Qwen3VLForConditionalGeneration
    elif "Qwen2.5-VL" in model_id or "Qwen2_5-VL" in model_id:
        model_cls = Qwen2_5_VLForConditionalGeneration
    else:
        model_cls = AutoModelForImageTextToText

    try:
        model = model_cls.from_pretrained(model_id, **kwargs)
    except Exception:
        if attn_impl == "flash_attention_2":
            kwargs["attn_implementation"] = "eager"
            model = model_cls.from_pretrained(model_id, **kwargs)
        else:
            raise
    from transformers import AutoProcessor as _AP

    processor = _AP.from_pretrained(model_id)

    # Left padding is required for correct batched causal-LM generation:
    # all sequences must share the same right edge so model.generate
    # continues each from its true last token. Right padding would make
    # the model resume from pad tokens for the shorter sequences.
    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is not None and getattr(tokenizer, "padding_side", None) is not None:
        tokenizer.padding_side = "left"

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
    "max_tokens_slider",
    "summary_mode",
    "one_sentence_mode",
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

    model_name = model.config._name_or_path if hasattr(model.config, "_name_or_path") else "Unknown Model"
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


def _augment_prompt(prompt: str, summary_mode: bool, one_sentence_mode: bool) -> str:
    if summary_mode and one_sentence_mode:
        return prompt + " Give a one-sentence summary of the scene."
    if summary_mode:
        return prompt + " Give a short summary of the scene."
    if one_sentence_mode:
        return prompt + " Describe this image in one sentence."
    return prompt


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
    summary_mode: bool,
    one_sentence_mode: bool,
    resolution_mode: str,
) -> list[dict[str, Any]]:
    is_video = is_video_file(media_path)
    content_type = "video" if is_video else "image"
    media_data = media_path if is_video else Image.open(media_path).convert("RGB")
    content_block: dict[str, Any] = {"type": content_type, content_type: media_data}
    if not is_video:
        content_block.update(_resolution_kwargs(resolution_mode))
    user_text = _augment_prompt(prompt, summary_mode, one_sentence_mode)
    messages: list[dict[str, Any]] = [{"role": "user", "content": [content_block, {"type": "text", "text": user_text}]}]
    if is_qwen35_model(current_model_id):
        messages.append({"role": "assistant", "content": "<think>\n\n</think>\n\n"})
    return messages


def preprocess_batch(
    media_paths: list[str],
    prompt: str,
    summary_mode: bool = False,
    one_sentence_mode: bool = False,
    resolution_mode: str = "auto",
):
    """Build a BatchFeature with batch dim len(media_paths) on CPU.
    Concatenates per-sample images/videos in positional order — the
    processor matches the i-th `<|image_pad|>` text token to the i-th
    image across the flat list, so order matters."""
    assert processor is not None, "Processor must be loaded before preprocessing."
    if not media_paths:
        raise ValueError("preprocess_batch requires at least one media path")
    qwen35 = is_qwen35_model(current_model_id)
    texts: list[str] = []
    images_acc: list[Any] = []
    videos_acc: list[Any] = []
    for path in media_paths:
        messages = _build_messages(path, prompt, summary_mode, one_sentence_mode, resolution_mode)
        texts.append(
            processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=not qwen35,
                continue_final_message=qwen35,
            )
        )
        imgs, vids = process_vision_info(messages)
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


def preprocess_one(
    media_path: str,
    prompt: str,
    summary_mode: bool = False,
    one_sentence_mode: bool = False,
    resolution_mode: str = "auto",
):
    """Single-sample wrapper around preprocess_batch. Safe to call from
    worker threads while the GPU is busy (touches only the processor)."""
    return preprocess_batch([media_path], prompt, summary_mode, one_sentence_mode, resolution_mode)


def run_generate(inputs, max_tokens: int):
    """Move a CPU BatchFeature (any batch dim) to GPU in place, run
    model.generate, return (generated_ids, input_ids) on CPU. Mutates
    `inputs` (the .to() call rebinds its tensors to GPU); caller should
    not reuse the CPU view."""
    assert model is not None, "Model must be loaded before generating."
    inputs_gpu = inputs.to("cuda", non_blocking=True)
    with torch.inference_mode():
        generated_ids = model.generate(**inputs_gpu, max_new_tokens=max_tokens)
    return generated_ids.cpu(), inputs_gpu["input_ids"].cpu()


def decode_batch(generated_ids, input_ids) -> list[str]:
    """Decode a [B, L_out] generated_ids tensor into a list of B caption
    strings, trimming each by the corresponding input length and
    stripping Qwen3.5 think tags when present."""
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


def generate_caption(
    media_path,
    prompt,
    max_tokens,
    summary_mode=False,
    one_sentence_mode=False,
    resolution_mode="auto",
):
    """Single-sample orchestrator preserved for back-compat with callers
    that don't yet use the prefetched/batched pipeline."""
    t_start = monotonic()
    inputs = preprocess_one(media_path, prompt, summary_mode, one_sentence_mode, resolution_mode)
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


def build_final_prompt(user_prompt, summary, one_sentence):
    parts = [user_prompt.strip()]
    if summary:
        parts.append("Please provide a short summary.")
    if one_sentence:
        parts.append("Keep the description to one sentence.")
    return " ".join(parts)


def _format_elapsed_str(start_time: float) -> str:
    elapsed = int(monotonic() - start_time)
    return f"{elapsed // 60:02d}:{elapsed % 60:02d}"


def _txt_path_for(media_path: str) -> str:
    return os.path.join(
        os.path.dirname(media_path),
        os.path.splitext(os.path.basename(media_path))[0] + ".txt",
    )


def _put_until_cancel(q: Queue, item: Any, cancel: threading.Event) -> bool:
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
    summary_mode: bool,
    one_sentence_mode: bool,
    resolution_mode: str,
    skip_existing: bool,
    executor: ThreadPoolExecutor,
    out_queue: Queue,
    cancel: threading.Event,
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
            if skip_existing and os.path.exists(_txt_path_for(path_i)):
                if not _put_until_cancel(out_queue, ("skip", i, path_i, None), cancel):
                    return
                i += 1
                continue
            indices = [i]
            paths = [path_i]
            j = i + 1
            while j < n and len(indices) < batch_size:
                pj = media_files[j]
                if skip_existing and os.path.exists(_txt_path_for(pj)):
                    break
                indices.append(j)
                paths.append(pj)
                j += 1
            try:
                fut: Future = executor.submit(
                    preprocess_batch,
                    paths,
                    prompt,
                    summary_mode,
                    one_sentence_mode,
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
    return ("⛔ Aborted by user.", None, None, "Aborted.", 0, "", *control_updates)


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


def _captioning_loop(folder_path, total_media, max_tokens, retain_preview, prefetch_queue, start_time):
    """Consumer side of the prefetch pipeline. Yields Gradio update tuples.
    Reads `should_abort` (and resets it on consumption) so the abort
    button works mid-loop. Emits the final summary on clean completion;
    returns early after the abort yield otherwise."""
    global should_abort
    processed_media = 0
    skipped_media = 0
    failed_media = 0
    last_media_to_show = None
    last_caption = ""
    last_name_md = ""
    elapsed_str = ""

    while True:
        if should_abort:
            should_abort = False
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
            rel_path = os.path.relpath(media_path, folder_path)
            skipped_media += 1
            elapsed_str = _format_elapsed_str(start_time)
            yield (
                f"⏭️ Skipped {idx + 1}/{total_media}: {rel_path} (already captioned)",
                last_media_to_show if retain_preview else None,
                last_name_md if retain_preview else None,
                last_caption if retain_preview else "Skipped (already captioned)",
                int(((idx + 1) / total_media) * 100),
                elapsed_str,
                *start_process(),
            )
            continue

        # kind == "batch"
        _, indices, paths, future = item
        captions, _n, err = _resolve_batch(future, max_tokens)
        if err is not None:
            failed_media += len(indices)
            for media_path in paths:
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
                with open(_txt_path_for(media_path), "w", encoding="utf-8") as f:
                    f.write(caption)
            except OSError as e:
                logger.exception("Write failed for %s", media_path)
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

            rel_path = os.path.relpath(media_path, folder_path)
            name_md = f"**File:** `{rel_path}`"
            media_to_show = Image.open(media_path) if is_image_file(media_path) else None
            elapsed_str = _format_elapsed_str(start_time)
            last_media_to_show = media_to_show
            last_caption = caption
            last_name_md = name_md
            processed_media += 1
            logger.info(f"🖼️ Processed {idx + 1}/{total_media}: {rel_path} in {elapsed_str}")
            yield (
                f"🖼️ Processing {idx + 1}/{total_media}: {rel_path}",
                media_to_show,
                name_md,
                caption,
                int(((idx + 1) / total_media) * 100),
                elapsed_str,
                *start_process(),
            )

    logger.info("processing complete: %r %.3f", processed_media, monotonic() - start_time)
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


def process_folder(
    folder_path,
    prompt,
    skip_existing,
    max_tokens,
    summary_mode,
    one_sentence_mode,
    retain_preview,
    resolution_mode,
    batch_size,
    prefetch_workers,
):
    batch_size = max(1, int(batch_size))
    prefetch_workers = max(1, int(prefetch_workers))

    logger.debug(
        "starting folder processing: model=%s quant=%s folder=%s skip_existing=%s "
        "max_tokens=%s summary=%s one_sentence=%s retain_preview=%s resolution=%s "
        "batch_size=%s prefetch_workers=%s should_abort=%s",
        current_model_id,
        current_quant,
        folder_path,
        skip_existing,
        max_tokens,
        summary_mode,
        one_sentence_mode,
        retain_preview,
        resolution_mode,
        batch_size,
        prefetch_workers,
        should_abort,
    )

    if not folder_path.strip():
        yield ("⚠️ Please enter a valid folder path.", None, None, "No media to process.", 0, "", *finish_process())
        return

    if not os.path.exists(folder_path):
        yield (f"❌ Folder not found: {folder_path}", None, None, "No media to process.", 0, "", *finish_process())
        return

    media_files = [
        os.path.join(root, file)
        for root, _, files in os.walk(folder_path)
        for file in files
        if is_image_file(file) or is_video_file(file)
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
    cancel = threading.Event()
    executor = ThreadPoolExecutor(max_workers=prefetch_workers, thread_name_prefix="caption-prep")
    dispatcher = threading.Thread(
        target=_prefetch_dispatcher,
        name="caption-dispatch",
        daemon=True,
        kwargs={
            "media_files": media_files,
            "batch_size": batch_size,
            "prompt": prompt,
            "summary_mode": summary_mode,
            "one_sentence_mode": one_sentence_mode,
            "resolution_mode": resolution_mode,
            "skip_existing": skip_existing,
            "executor": executor,
            "out_queue": prefetch_queue,
            "cancel": cancel,
        },
    )
    dispatcher.start()

    try:
        yield from _captioning_loop(folder_path, total_media, max_tokens, retain_preview, prefetch_queue, start_time)
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
    )
    gradio.Markdown("Supported image formats: png, jpg, jpeg, bmp, gif, webp")
    gradio.Markdown("Supported video formats: mp4, mov, avi, webm, mkv, gif, flv")
    gradio.Markdown("Written by [Olli S.](https://github.com/o-l-l-i)")

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
            placeholder="e.g. C:\\Users\\you\\Pictures\\input_images",
        )
        ui_e["prompt_input"] = gradio.Textbox(label="Custom Prompt", value=DEFAULT_PROMPT)

    ui_e["skip_existing_checkbox"] = gradio.Checkbox(label="Skip already captioned media (.txt exists)", value=True)

    with gradio.Row():
        gradio.Markdown("### Prompt Controls")
        gradio.Markdown("""
        - **Summary Mode**: Asks the model to summarize the media content briefly.
        - **One-Sentence Mode**: Instructs the model to keep the caption to a single concise sentence.
        """)
        ui_e["summary_mode"] = gradio.Checkbox(label="Summary Mode", value=False)
        ui_e["one_sentence_mode"] = gradio.Checkbox(label="One-Sentence Mode", value=False)

    prompt_preview = gradio.Textbox(label="Final Prompt Preview", lines=2, interactive=False)

    with gradio.Row():
        ui_e["max_tokens_slider"] = gradio.Slider(
            label="🧾 Max Tokens", minimum=32, maximum=1024 * 8, value=128, step=16
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
                f"cpu_count={os.cpu_count() or '?'}."
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

    ui_e["prompt_input"].change(
        fn=build_final_prompt,
        inputs=[ui_e["prompt_input"], ui_e["summary_mode"], ui_e["one_sentence_mode"]],
        outputs=[prompt_preview],
    )

    ui_e["summary_mode"].change(
        fn=build_final_prompt,
        inputs=[ui_e["prompt_input"], ui_e["summary_mode"], ui_e["one_sentence_mode"]],
        outputs=[prompt_preview],
    )

    ui_e["one_sentence_mode"].change(
        fn=build_final_prompt,
        inputs=[ui_e["prompt_input"], ui_e["summary_mode"], ui_e["one_sentence_mode"]],
        outputs=[prompt_preview],
    )

    ui_e["start_button"].click(start_process, inputs=[], outputs=[ui_e[k] for k in control_keys])

    ui_e["start_button"].click(
        process_folder,
        inputs=[
            ui_e["folder_input"],
            ui_e["prompt_input"],
            ui_e["skip_existing_checkbox"],
            ui_e["max_tokens_slider"],
            ui_e["summary_mode"],
            ui_e["one_sentence_mode"],
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
