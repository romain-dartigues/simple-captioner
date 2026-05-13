"""common definitions which are loadable quickly"""

# stdlib
from os import cpu_count
from threading import Event
from importlib.metadata import Distribution

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
RESOLUTION_MODES = ("auto", "auto_high", "fast", "high")
QUANT_CHOICES = ("None", "8-bit", "4-bit")
ATTN_CHOICES = ("flash_attention_2", "eager")

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

HELP_BATCH_SIZE = (
    "Samples per model.generate() call. Higher = better GPU utilization, "
    "but VRAM scales roughly linearly. Start at 1 and raise until VRAM "
    "headroom shrinks."
)
HELP_PREFETCH_WORKERS = (
    f"CPU threads preprocessing batches ahead of the GPU. "
    f"Default {DEFAULT_PREFETCH_WORKERS} guessed from "
    f"cpu_count={cpu_count() or '?'}."
)

HELP_RESOLUTION_MODE = "Choose the resolution mode for visual input."

# Cooperative cancellation signal for caption_folder(). The Gradio
# abort button and the CLI's SIGINT handler both set this via
# request_abort(); caption_folder() clears it at the start of each
# run. In-flight model.generate() is not preemptible — abort takes
# effect at the next batch boundary.
abort_event: Event = Event()


def request_abort() -> None:
    """Signal the active caption_folder() generator to stop after the
    current batch completes."""
    abort_event.set()

DISTRIBUTION = Distribution.from_name(__package__)
