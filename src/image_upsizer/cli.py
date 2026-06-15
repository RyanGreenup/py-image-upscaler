"""Command-line interface for image-upsizer."""

from __future__ import annotations

import time
from enum import StrEnum
from pathlib import Path
from typing import Annotated

import typer
from PIL import Image

from image_upsizer import upscaler

app = typer.Typer(
    help="Upscale images to a target width on the GPU.", no_args_is_help=True
)


class Method(StrEnum):
    """Upscaling backend."""

    ai = "ai"
    bicubic = "bicubic"


class Preset(StrEnum):
    """Convenience target-width presets."""

    p1080 = "1080p"
    k4 = "4k"
    k8 = "8k"


PRESET_WIDTHS = {Preset.p1080: 1920, Preset.k4: 3840, Preset.k8: 7680}


InputArg = Annotated[Path, typer.Argument(help="Path to the input image.")]
OutputArg = Annotated[
    Path | None,
    typer.Argument(help="Output path (default: <stem>_upscaled.png next to input)."),
]
WidthOpt = Annotated[
    int, typer.Option("--width", "-w", help="Target width in pixels (4K width = 3840).")
]
PresetOpt = Annotated[
    Preset | None,
    typer.Option("--preset", help="Target-width preset; overrides --width."),
]
MethodOpt = Annotated[Method, typer.Option("--method", help="Upscaling backend.")]
Fp16Opt = Annotated[
    bool, typer.Option("--fp16/--no-fp16", help="Use half precision for the AI model.")
]


@app.command()
def upscale(  # noqa: PLR0913
    input_path: InputArg,
    output_path: OutputArg = None,
    width: WidthOpt = 3840,
    preset: PresetOpt = None,
    method: MethodOpt = Method.ai,
    *,
    fp16: Fp16Opt = True,
) -> None:
    """Upscale INPUT to a target width, preserving aspect ratio."""
    if not input_path.exists():
        typer.echo(f"Input not found: {input_path}", err=True)
        raise typer.Exit(code=1)

    target_width = PRESET_WIDTHS[preset] if preset is not None else width
    if target_width < 1:
        typer.echo("Target width must be positive.", err=True)
        raise typer.Exit(code=1)

    out = output_path or input_path.with_name(f"{input_path.stem}_upscaled.png")

    img = Image.open(input_path)
    in_w, in_h = img.size

    upscaler.require_cuda()
    start = time.perf_counter()
    if method is Method.ai:
        model = upscaler.load_model(fp16=fp16)
        result = upscaler.upscale_ai(img, model, target_width)
    else:
        result = upscaler.upscale_bicubic(img, target_width)
    elapsed = time.perf_counter() - start

    out.parent.mkdir(parents=True, exist_ok=True)
    result.save(out)

    typer.echo(
        f"{in_w}x{in_h} -> {result.width}x{result.height} "
        f"({method.value}, {elapsed:.1f}s) -> {out}"
    )


def main() -> None:
    """Run the CLI application."""
    app()
