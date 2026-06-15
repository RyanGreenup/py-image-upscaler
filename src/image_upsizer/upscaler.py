"""CUDA image upscaling: Real-ESRGAN super-resolution plus a bicubic fallback."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812
from PIL import Image
from spandrel import ImageModelDescriptor, ModelLoader

from image_upsizer.weights import ensure_weights

if TYPE_CHECKING:
    from pathlib import Path


def require_cuda() -> None:
    """Raise a clear error if CUDA is not available."""
    if not torch.cuda.is_available():
        msg = "CUDA is not available; this upscaler requires an NVIDIA GPU."
        raise RuntimeError(msg)


def target_size(width: int, height: int, target_width: int) -> tuple[int, int]:
    """Compute output (width, height) for a target width, preserving aspect ratio."""
    out_h = round(target_width * height / width)
    return target_width, max(1, out_h)


def load_model(
    weights: Path | None = None,
    *,
    device: str = "cuda",
    fp16: bool = True,
) -> ImageModelDescriptor[torch.nn.Module]:
    """Load the Real-ESRGAN model onto the GPU in eval mode."""
    path = weights if weights is not None else ensure_weights()
    model = ModelLoader(device=device).load_from_file(str(path))
    if not isinstance(model, ImageModelDescriptor):
        msg = "Loaded weights are not an image-to-image model."
        raise TypeError(msg)
    model.to(device).eval()
    if fp16 and model.supports_half:
        model.half()
    return model


def _to_tensor(img: Image.Image, *, device: str, dtype: torch.dtype) -> torch.Tensor:
    """Convert a PIL image to a (1, 3, H, W) tensor in [0, 1] on the given device."""
    rgb = img.convert("RGB")
    arr = np.array(rgb, dtype=np.uint8)  # (H, W, 3), writable copy
    t = torch.from_numpy(arr).permute(2, 0, 1)
    return t.unsqueeze(0).to(device=device, dtype=dtype).div(255.0)


def _to_pil(tensor: torch.Tensor) -> Image.Image:
    """Convert a (1, 3, H, W) tensor in [0, 1] back to a PIL RGB image."""
    arr = tensor.squeeze(0).clamp(0, 1).mul(255).round().to(torch.uint8)
    arr = arr.permute(1, 2, 0).contiguous().cpu().numpy()
    return Image.fromarray(arr, mode="RGB")


def _tiled_forward(
    model: ImageModelDescriptor[torch.nn.Module],
    tensor: torch.Tensor,
    *,
    tile: int = 512,
    overlap: int = 32,
) -> torch.Tensor:
    """Run the model over overlapping tiles and stitch, to bound VRAM use."""
    _, _, h, w = tensor.shape
    if h <= tile and w <= tile:
        return model(tensor)

    scale = model.scale
    out = tensor.new_zeros((1, model.output_channels, h * scale, w * scale))
    for y in range(0, h, tile):
        for x in range(0, w, tile):
            y1, x1 = min(y + tile, h), min(x + tile, w)
            # Pad each tile with overlap (clamped) to avoid seam artifacts.
            py0, px0 = max(0, y - overlap), max(0, x - overlap)
            py1, px1 = min(h, y1 + overlap), min(w, x1 + overlap)
            out_tile = model(tensor[:, :, py0:py1, px0:px1])
            top, left = (y - py0) * scale, (x - px0) * scale
            out[:, :, y * scale : y1 * scale, x * scale : x1 * scale] = out_tile[
                :, :, top : top + (y1 - y) * scale, left : left + (x1 - x) * scale
            ]
    return out


def upscale_ai(
    img: Image.Image,
    model: ImageModelDescriptor[torch.nn.Module],
    target_width: int,
    *,
    device: str = "cuda",
) -> Image.Image:
    """Upscale with Real-ESRGAN, looping passes until wide enough, then fit width."""
    require_cuda()
    out_w, out_h = target_size(img.width, img.height, target_width)

    dtype = next(model.model.parameters()).dtype
    tensor = _to_tensor(img, device=device, dtype=dtype)

    # Real-ESRGAN is a fixed Nx model; repeat passes until we exceed target width.
    # Each pass is tiled so large intermediate images stay within VRAM.
    with torch.no_grad():
        while tensor.shape[-1] < target_width:
            tensor = _tiled_forward(model, tensor)

    result = _to_pil(tensor.float())
    # High-quality Lanczos downscale to the exact target, preserving aspect ratio.
    if (result.width, result.height) != (out_w, out_h):
        result = result.resize((out_w, out_h), Image.Resampling.LANCZOS)
    return result


def upscale_bicubic(
    img: Image.Image,
    target_width: int,
    *,
    device: str = "cuda",
) -> Image.Image:
    """Fast GPU bicubic resize to the target width, preserving aspect ratio."""
    require_cuda()
    out_w, out_h = target_size(img.width, img.height, target_width)
    tensor = _to_tensor(img, device=device, dtype=torch.float32)
    with torch.no_grad():
        resized = F.interpolate(
            tensor,
            size=(out_h, out_w),
            mode="bicubic",
            align_corners=False,
        )
    return _to_pil(resized)
