# image-upsizer

A CUDA image upscaler CLI. It upsamples an image to a target **width**
(default 3840, i.e. 4K width), preserving aspect ratio. The default backend is
AI super-resolution with [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN)
(4x) run on the GPU via [spandrel](https://github.com/chainner-org/spandrel),
with a fast GPU bicubic fallback.

The project is a Python Typer CLI managed with [uv](https://docs.astral.sh/uv/),
linted/formatted with [ruff](https://docs.astral.sh/ruff/) (`select = ["ALL"]`),
type-checked with [pyright](https://microsoft.github.io/pyright/) in strict mode,
and tested with pytest. The package lives at `src/image_upsizer/`.

## Prerequisites

- An NVIDIA GPU with CUDA (the AI and bicubic backends both run on CUDA).
- **uv** ([install](https://docs.astral.sh/uv/getting-started/installation/))
- **just** ([install](https://github.com/casey/just)) for the task runner (optional)

## Quick start

```bash
uv sync                                              # create .venv, install deps
uv run image-upsizer tests/fixtures/input.png        # -> input_upscaled.png at 4K width
just check                                           # fmt + lint + type + test
```

The first AI run downloads the Real-ESRGAN weights (~64 MB) to
`~/.cache/image-upsizer/` and reuses them afterwards.

## CLI

```bash
uv run image-upsizer --help
# Upscale to 4K width (default), AI backend:
uv run image-upsizer input.png
# Custom width and explicit output path:
uv run image-upsizer input.png out.png --width 2560
# Presets (1080p / 4k / 8k) override --width:
uv run image-upsizer input.png --preset 8k
# Fast GPU bicubic instead of the AI model:
uv run image-upsizer input.png --method bicubic
# Disable half precision if you see artifacts:
uv run image-upsizer input.png --no-fp16
```

Options: `--width/-w` (target px width), `--preset {1080p,4k,8k}`,
`--method {ai,bicubic}`, `--fp16/--no-fp16`. Output defaults to
`<stem>_upscaled.png` next to the input. AI passes are tiled, so target widths
beyond a single 4x pass (e.g. 8K) stay within VRAM.

## Tasks

The `justfile` wraps the common loops:

| Command     | What it does                         |
| ----------- | ------------------------------------ |
| `just`      | runs `check` (fmt, lint, type, test) |
| `just fmt`  | `ruff format --check .`              |
| `just lint` | `ruff check .`                       |
| `just type` | `pyright`                            |
| `just test` | `pytest` with coverage               |

Run any underlying tool directly with `uv run <tool>` if you do not have `just`.

## Layout

- `src/image_upsizer/`
  - `cli.py` - Typer CLI and console entry point
  - `upscaler.py` - CUDA upscaling (Real-ESRGAN + bicubic, tiling)
  - `weights.py` - download/cache the Real-ESRGAN weights
- `tests/` - pytest suite
- `pyproject.toml` - project + ruff + pyright + pytest + coverage config
- `.pre-commit-config.yaml` - ruff + uv-lock hooks
- `.github/workflows/ci.yml` - CI: format check, lint, typecheck, test

## Notes

## Contributing

### Overview

The project is a Python Typer CLI managed with [uv](https://docs.astral.sh/uv/),
linted/formatted with [ruff](https://docs.astral.sh/ruff/) (`select = ["ALL"]`),
type-checked with [pyright](https://microsoft.github.io/pyright/) in strict mode,
and tested with pytest. The package is in `src/image_upsizer/`.

### Notes

- ruff is set to `select = ["ALL"]` with docstring rules (`D`) disabled by default.
- Tests relax a few rules (`S101`, `PLR2004`, `ANN`) via `per-file-ignores`.
- `uv.lock` is git tracked by default.
