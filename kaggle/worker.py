#!/usr/bin/env python3
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path


INPUT_CANDIDATES = [
    Path("/kaggle/input/pyvideotrans-input/input.mp4"),
    Path("/kaggle/input/input.mp4"),
]
REPO_URL = "https://github.com/canibeyours2001-ui/pyvideotrans.git"
WORK_DIR = Path("/kaggle/working")
REPO_DIR = WORK_DIR / "pyvideotrans-src"
RUN_OUTPUT_DIR = WORK_DIR / "pyvideotrans-run"
GPU_OUTPUT_DIR = WORK_DIR / "gpu_output"


def run(cmd: list[str], *, cwd: Path | None = None) -> None:
    print(f"[run] {' '.join(cmd)}")
    proc = subprocess.run(cmd, cwd=str(cwd) if cwd else None, check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed ({proc.returncode}): {' '.join(cmd)}")


def find_input_video() -> Path:
    for candidate in INPUT_CANDIDATES:
        if candidate.exists() and candidate.stat().st_size > 0:
            return candidate
    for path in Path("/kaggle/input").glob("**/*.mp4"):
        if path.stat().st_size > 0:
            return path
    raise FileNotFoundError("No non-empty input MP4 found under /kaggle/input")


def find_srt_file(root: Path) -> Path:
    direct = root / "input.srt"
    if direct.exists() and direct.stat().st_size > 0:
        return direct
    candidates = sorted([p for p in root.glob("**/*.srt") if p.stat().st_size > 0])
    if not candidates:
        raise FileNotFoundError(f"No SRT file found under {root}")
    return candidates[0]


def main() -> int:
    print("=== pyVideoTrans Kaggle GPU worker start ===")

    run(["nvidia-smi"])
    run(["python", "--version"])

    input_video = find_input_video()
    print(f"Using input video: {input_video}")

    if REPO_DIR.exists():
        shutil.rmtree(REPO_DIR)
    run(["git", "clone", "--depth", "1", REPO_URL, str(REPO_DIR)])

    if RUN_OUTPUT_DIR.exists():
        shutil.rmtree(RUN_OUTPUT_DIR)
    RUN_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    GPU_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    run(["python", "-m", "pip", "install", "--upgrade", "pip"])
    run(["python", "-m", "pip", "install", "uv"])

    run(["uv", "sync"], cwd=REPO_DIR)
    run(
        [
            "uv",
            "run",
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
            "large-v3",
            "--cuda",
            "--output-dir",
            str(RUN_OUTPUT_DIR),
        ],
        cwd=REPO_DIR,
    )

    srt_file = find_srt_file(RUN_OUTPUT_DIR)
    target_srt = GPU_OUTPUT_DIR / "subtitles.srt"
    shutil.copy2(srt_file, target_srt)
    print(f"Exported subtitle: {target_srt}")

    print("Generated files:")
    for path in sorted(WORK_DIR.glob("gpu_output/**/*")):
        if path.is_file():
            print(f"- {path}")

    print("=== pyVideoTrans Kaggle GPU worker done ===")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
