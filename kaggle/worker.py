#!/usr/bin/env python3
"""Run the GPU-only STT stage and publish the stable subtitle contract."""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path


REPOSITORY_URL = "__REPOSITORY_URL__"
REPOSITORY_REF = "__REPOSITORY_REF__"
WHISPER_MODEL = "__WHISPER_MODEL_JSON__"
WORK_DIR = Path("/kaggle/working")
REPOSITORY_DIR = WORK_DIR / "pyvideotrans-src"
STT_OUTPUT_DIR = WORK_DIR / "pyvideotrans-stt"
GPU_OUTPUT_DIR = WORK_DIR / "gpu_output"
INPUT_ROOT = Path("/kaggle/input")


def run(command: list[str], *, cwd: Path | None = None) -> None:
    print(f"[run] {' '.join(command)}", flush=True)
    subprocess.run(command, cwd=cwd, check=True)


def find_input_video() -> Path:
    candidates = sorted(INPUT_ROOT.glob("*/input.mp4"))
    valid = [path for path in candidates if path.is_file() and path.stat().st_size > 0]
    if len(valid) != 1:
        raise RuntimeError(
            f"Expected exactly one non-empty /kaggle/input/*/input.mp4, found {len(valid)}"
        )
    return valid[0]


def find_subtitle(output_dir: Path) -> Path:
    candidates = sorted(
        path for path in output_dir.glob("*.srt") if path.is_file() and path.stat().st_size > 0
    )
    if len(candidates) != 1:
        raise RuntimeError(
            f"Expected exactly one non-empty SRT in {output_dir}, found {len(candidates)}"
        )
    subtitle = candidates[0]
    content = subtitle.read_text(encoding="utf-8-sig", errors="strict").strip()
    if not content or "-->" not in content:
        raise RuntimeError(f"Generated subtitle is not a valid SRT: {subtitle}")
    return subtitle


def checkout_repository() -> None:
    if (
        REPOSITORY_URL.startswith("__")
        or REPOSITORY_REF.startswith("__")
        or WHISPER_MODEL.startswith("__")
    ):
        raise RuntimeError("Workflow did not inject repository URL and revision")
    if REPOSITORY_DIR.exists():
        shutil.rmtree(REPOSITORY_DIR)
    REPOSITORY_DIR.mkdir(parents=True)
    run(["git", "init", "--quiet"], cwd=REPOSITORY_DIR)
    run(["git", "remote", "add", "origin", REPOSITORY_URL], cwd=REPOSITORY_DIR)
    run(["git", "fetch", "--depth", "1", "origin", REPOSITORY_REF], cwd=REPOSITORY_DIR)
    run(["git", "checkout", "--detach", "FETCH_HEAD"], cwd=REPOSITORY_DIR)


def main() -> int:
    print("=== pyVideoTrans Kaggle GPU worker ===", flush=True)
    run(["nvidia-smi"])
    input_video = find_input_video()
    print(f"Using input video: {input_video}", flush=True)

    checkout_repository()
    if STT_OUTPUT_DIR.exists():
        shutil.rmtree(STT_OUTPUT_DIR)
    STT_OUTPUT_DIR.mkdir(parents=True)
    GPU_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    run(["python", "-m", "pip", "install", "--upgrade", "uv"])
    run(["uv", "python", "install", "3.10"], cwd=REPOSITORY_DIR)
    run(["uv", "sync", "--locked", "--python", "3.10"], cwd=REPOSITORY_DIR)
    run(
        [
            "uv",
            "run",
            "--python",
            "3.10",
            "python",
            "-c",
            "import torch; assert torch.cuda.is_available(), 'torch.cuda.is_available() is false'",
        ],
        cwd=REPOSITORY_DIR,
    )
    run(
        [
            "uv",
            "run",
            "--python",
            "3.10",
            "cli.py",
            "--task",
            "stt",
            "--name",
            str(input_video),
            "--recogn_type",
            "0",
            "--detect_language",
            "auto",
            "--model_name",
            WHISPER_MODEL,
            "--cuda",
            "--output-dir",
            str(STT_OUTPUT_DIR),
        ],
        cwd=REPOSITORY_DIR,
    )

    subtitle = find_subtitle(STT_OUTPUT_DIR)
    target = GPU_OUTPUT_DIR / "subtitles.srt"
    shutil.copy2(subtitle, target)
    print(f"Stable output: {target}", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr, flush=True)
        raise
