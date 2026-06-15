"""Download and cache Real-ESRGAN model weights."""

from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

# Official Real-ESRGAN x4 general-purpose model (xinntao release).
WEIGHTS_URL = (
    "https://github.com/xinntao/Real-ESRGAN/releases/download/"
    "v0.1.0/RealESRGAN_x4plus.pth"
)
WEIGHTS_FILENAME = "RealESRGAN_x4plus.pth"
# Sanity floor: the real file is ~67 MB. Reject obvious truncations / HTML errors.
MIN_WEIGHTS_BYTES = 50 * 1024 * 1024
_COMPLETE_PCT = 100


def cache_dir() -> Path:
    """Return the directory where model weights are cached."""
    return Path.home() / ".cache" / "image-upsizer"


_last_pct = -1


def _report(block_num: int, block_size: int, total_size: int) -> None:
    """Print a simple download progress bar to stderr, once per percent."""
    global _last_pct  # noqa: PLW0603
    if total_size <= 0:
        return
    downloaded = block_num * block_size
    pct = min(100, downloaded * 100 // total_size)
    if pct == _last_pct:
        return
    _last_pct = pct
    mb = downloaded / 1024 / 1024
    total_mb = total_size / 1024 / 1024
    sys.stderr.write(f"\rDownloading weights: {pct:3d}% ({mb:.1f}/{total_mb:.1f} MB)")
    sys.stderr.flush()
    if pct >= _COMPLETE_PCT:
        sys.stderr.write("\n")


def ensure_weights() -> Path:
    """Return the path to the cached weights, downloading them if necessary."""
    target = cache_dir() / WEIGHTS_FILENAME
    if target.exists() and target.stat().st_size >= MIN_WEIGHTS_BYTES:
        return target

    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(".pth.part")
    urllib.request.urlretrieve(WEIGHTS_URL, tmp, _report)  # noqa: S310

    size = tmp.stat().st_size
    if size < MIN_WEIGHTS_BYTES:
        tmp.unlink(missing_ok=True)
        msg = f"Downloaded weights look truncated ({size} bytes); aborting."
        raise RuntimeError(msg)

    tmp.replace(target)
    return target
