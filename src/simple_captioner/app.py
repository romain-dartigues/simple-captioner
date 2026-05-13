# stdlib
from logging import basicConfig, getLogger
from pathlib import Path

# dependencies
import gradio
from PIL import Image

# project
from .captioner import (
    DEFAULT_ATTN,
    DEFAULT_BATCH_SIZE,
    DEFAULT_MODEL_ID,
    DEFAULT_PREFETCH_WORKERS,
    DEFAULT_QUANT,
    AbortedEvent,
    CaptionedEvent,
    CompleteEvent,
    ErrorEvent,
    SkipEvent,
    caption_folder,
    format_elapsed,
    get_model_info,
    is_image_file,
    load_selected_model,
    suggest_batch_size,
)
from .common import (
    AVAILABLE_MODELS,
    DEFAULT_MAX_TOKENS,
    DEFAULT_PROMPT,
    HELP_BATCH_SIZE,
    HELP_PREFETCH_WORKERS,
    HELP_RESOLUTION_MODE,
    MAX_BATCH_SIZE,
    MAX_PREFETCH_WORKERS,
    request_abort,
)

logger = getLogger(__name__)

ui_e: dict = {}


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
    request_abort()
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


def _build_abort_yield():
    status_index = control_keys.index("status_output")
    abort_index = control_keys.index("abort_button")
    control_updates = enable_controls_dict()
    control_updates[status_index] = gradio.update(value="⛔ Aborted by user.")
    control_updates[abort_index] = gradio.update(interactive=False)
    return "⛔ Aborted by user.", None, None, "Aborted.", 0, "", *control_updates


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
    """Gradio handler: thin adapter that maps caption_folder() events
    to the (status, media, name_md, caption, progress, elapsed, *control_updates)
    tuple the UI expects. The CLI consumes the same events directly."""
    if not folder_path.strip():
        yield "⚠️ Please enter a valid folder path.", None, None, "No media to process.", 0, "", *finish_process()
        return

    folder = Path(folder_path)
    if not folder.exists():
        yield f"❌ Folder not found: {folder_path}", None, None, "No media to process.", 0, "", *finish_process()
        return

    last_media_to_show = None
    last_caption = ""
    last_name_md = ""

    for event in caption_folder(
        folder_path=folder_path,
        prompt=prompt,
        skip_existing=skip_existing,
        caption_extension=caption_extension,
        max_tokens=max_tokens,
        resolution_mode=resolution_mode,
        batch_size=batch_size,
        prefetch_workers=prefetch_workers,
    ):
        if isinstance(event, SkipEvent):
            rel_path = Path(event.path).relative_to(folder)
            percent = int(((event.idx + 1) / event.total) * 100)
            elapsed_str = format_elapsed(event.elapsed_s)
            yield (
                f"⏭️ Skipped {event.idx + 1}/{event.total}: {rel_path} (already captioned)",
                last_media_to_show if retain_preview else None,
                last_name_md if retain_preview else None,
                last_caption if retain_preview else "Skipped (already captioned)",
                percent,
                elapsed_str,
                *start_process(),
            )
        elif isinstance(event, CaptionedEvent):
            rel_path = Path(event.path).relative_to(folder)
            percent = int(((event.idx + 1) / event.total) * 100)
            elapsed_str = format_elapsed(event.elapsed_s)
            name_md = f"**File:** `{rel_path}`"
            media = Image.open(event.path) if is_image_file(event.path) else None
            last_media_to_show = media
            last_caption = event.caption
            last_name_md = name_md
            yield (
                f"🖼️ Processing {event.idx + 1}/{event.total}: {rel_path}",
                media,
                name_md,
                event.caption,
                percent,
                elapsed_str,
                *start_process(),
            )
        elif isinstance(event, ErrorEvent):
            yield (
                f"⚠️ Error processing {event.path}: {event.error}",
                None,
                None,
                "Error in captioning.",
                0,
                format_elapsed(event.elapsed_s),
                *start_process(),
            )
        elif isinstance(event, AbortedEvent):
            yield _build_abort_yield()
        elif isinstance(event, CompleteEvent):
            if event.total == 0:
                yield (
                    "📂 No media found in the folder or subfolders.",
                    None,
                    None,
                    "No media to process.",
                    0,
                    "",
                    *finish_process(),
                )
            else:
                elapsed_str = format_elapsed(event.elapsed_s)
                yield (
                    "✅ Processing complete!"
                    f"processed {event.processed} media in {elapsed_str}, skipped {event.skipped} media."
                    f"Failed to process {event.failed} media (inaccessible, unknown or broken file)",
                    last_media_to_show,
                    last_name_md,
                    last_caption,
                    None,
                    None,
                    *finish_process(),
                )


def run_gradio():
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
                info=HELP_RESOLUTION_MODE,
            )

        with gradio.Row():
            ui_e["batch_size_slider"] = gradio.Slider(
                label="📦 Batch Size",
                minimum=1,
                maximum=MAX_BATCH_SIZE,
                value=DEFAULT_BATCH_SIZE,
                step=1,
                info=HELP_BATCH_SIZE,
            )
            ui_e["prefetch_workers_slider"] = gradio.Slider(
                label="🧵 Prefetch Workers",
                minimum=1,
                maximum=MAX_PREFETCH_WORKERS,
                value=DEFAULT_PREFETCH_WORKERS,
                step=1,
                info=HELP_PREFETCH_WORKERS,
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
        css=""".generating { border: none; }""",
    )


def main():
    basicConfig(
        level="INFO",
    )
    logger.setLevel("DEBUG")
    getLogger("captioner").setLevel("DEBUG")
    run_gradio()


if __name__ == "__main__":
    main()
