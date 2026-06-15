"""Tests for image-upsizer."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from PIL import Image
from typer.testing import CliRunner

from image_upsizer import upscaler
from image_upsizer.cli import app

runner = CliRunner()

FIXTURE = Path(__file__).parent / "fixtures" / "input.png"


def test_target_size_preserves_aspect_ratio() -> None:
    """1000x321 scaled to 4K width keeps the ultra-wide ratio."""
    assert upscaler.target_size(1000, 321, 3840) == (3840, 1233)


def test_target_size_rounds_height() -> None:
    """Height rounds to nearest pixel."""
    # 3840 * 321 / 1000 = 1232.64 -> 1233
    assert upscaler.target_size(1000, 321, 3840) == (3840, 1233)


def test_target_size_height_never_zero() -> None:
    """A height that would round to zero is clamped to one pixel."""
    # 100 * 1 / 1000 = 0.1 -> 0 -> clamped to 1
    assert upscaler.target_size(1000, 1, 100) == (100, 1)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_bicubic_cli_hits_target_width(tmp_path: Path) -> None:
    """The bicubic backend produces the requested width with the right ratio."""
    out = tmp_path / "out.png"
    result = runner.invoke(
        app,
        [str(FIXTURE), str(out), "--method", "bicubic", "--width", "1920"],
    )

    assert result.exit_code == 0, result.output
    with Image.open(out) as img:
        assert img.width == 1920
        # 321/1000 * 1920 = 616.32 -> 616
        assert img.height == 616
