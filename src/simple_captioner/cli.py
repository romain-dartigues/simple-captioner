"""Headless captioner. Same backend as the Gradio UI; runs in the
terminal so it survives client disconnects."""

# stdlib
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter, BooleanOptionalAction
from logging import CRITICAL, DEBUG, basicConfig, getLogger

import logging
import signal
import sys

# project
from .common import (
    ATTN_CHOICES,
    AVAILABLE_MODELS,
    DEFAULT_ATTN,
    DEFAULT_BATCH_SIZE,
    DEFAULT_MAX_TOKENS,
    DEFAULT_MODEL_ID,
    DEFAULT_PREFETCH_WORKERS,
    DEFAULT_PROMPT,
    DEFAULT_QUANT,
    HELP_BATCH_SIZE,
    HELP_PREFETCH_WORKERS,
    HELP_RESOLUTION_MODE,
    QUANT_CHOICES,
    RESOLUTION_MODES,
    request_abort,
DISTRIBUTION,
)

logger = logging.getLogger("caption_cli")


def _build_parser() -> ArgumentParser:
    suggested = ", ".join(m for m in AVAILABLE_MODELS if m != "Custom...")
    p = ArgumentParser(
        prog="caption_cli",
        description="Caption every image/video in a folder. Headless; safe to run under nohup/tmux.",
        formatter_class=ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "-q",
        "--quiet",
        action="store_const",
        const=0,
        default=2,
        dest="verbose",
    )
    p.add_argument("-v", "--verbose", action="count")
    p.add_argument("-V", "--version", action="version", version=DISTRIBUTION.version)

    p.add_argument("folder", help="Folder to walk recursively for images/videos.")
    p.add_argument(
        "--model",
        default=DEFAULT_MODEL_ID,
        help=f"HuggingFace model ID. Suggested: {suggested}",
        metavar="NAME",
    )
    p.add_argument("--quant", default=DEFAULT_QUANT, choices=QUANT_CHOICES)
    p.add_argument("--attn", default=DEFAULT_ATTN, choices=ATTN_CHOICES)
    p.add_argument("--prompt", default=DEFAULT_PROMPT, help="User prompt sent to the model.", metavar="TEXT")
    p.add_argument(
        "--skip-existing",
        dest="skip_existing",
        action=BooleanOptionalAction,
        default=True,
        help="Skip files that already have a sibling caption file with the chosen extension.",
    )
    p.add_argument(
        "--caption-extension",
        default="txt",
        help="Extension for generated caption files.",
        metavar="EXT",
    )
    p.add_argument(
        "--max-tokens", type=int, default=DEFAULT_MAX_TOKENS, help="Max number of tokens to generate.", metavar="INT"
    )
    p.add_argument("--resolution-mode", default="auto", choices=RESOLUTION_MODES, help=HELP_RESOLUTION_MODE)
    p.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE, help=HELP_BATCH_SIZE, metavar="INT")
    p.add_argument(
        "--prefetch-workers", type=int, default=DEFAULT_PREFETCH_WORKERS, help=HELP_PREFETCH_WORKERS, metavar="INT"
    )
    return p


def _install_sigint_handler() -> None:
    """First Ctrl+C: request a graceful abort (in-flight batch finishes,
    then AbortedEvent yields and the loop exits cleanly). Second Ctrl+C:
    force-exit — useful if model.generate() is stuck."""
    pressed = {"count": 0}

    def handler(signum, frame):
        pressed["count"] += 1
        if pressed["count"] >= 2:
            print("\nForce exiting.", flush=True, file=sys.stderr)
            raise SystemExit(130)
        request_abort()
        print(
            "\nAbort requested; finishing the current batch then stopping. Press Ctrl+C again to force exit.",
            flush=True,
            file=sys.stderr,
        )

    signal.signal(signal.SIGINT, handler)


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    log_level = min(max(CRITICAL - args.verbose * 10, DEBUG), CRITICAL)
    basicConfig(
        format="%(asctime)s [%(levelname)s %(name)s] %(filename)s:%(lineno)d: %(message)s",
        datefmt="%F %T",
        level=log_level,
    )
    # captioner emits per-sample progress at INFO and timings at DEBUG;
    # match the requested level so --log-level DEBUG shows everything.
    logging.getLogger("captioner").setLevel(log_level)

    # Lazy loading
    from .captioner import (
        AbortedEvent,
        CompleteEvent,
        caption_folder,
        load_selected_model,
    )

    _install_sigint_handler()

    logger.info("loading model %s (quant=%s, attn=%s)", args.model, args.quant, args.attn)
    name, device, vram, dtype, _config = load_selected_model(args.model, args.quant, args.attn)
    logger.info("loaded %s on %s (vram %s, dtype %s)", name, device, vram, dtype)

    exit_code = 0
    for event in caption_folder(
        folder_path=args.folder,
        prompt=args.prompt,
        skip_existing=args.skip_existing,
        caption_extension=args.caption_extension.lstrip("."),
        max_tokens=args.max_tokens,
        resolution_mode=args.resolution_mode,
        batch_size=args.batch_size,
        prefetch_workers=args.prefetch_workers,
    ):
        if isinstance(event, AbortedEvent):
            exit_code = 130
        elif isinstance(event, CompleteEvent):
            exit_code = 0 if event.failed == 0 else 1
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
