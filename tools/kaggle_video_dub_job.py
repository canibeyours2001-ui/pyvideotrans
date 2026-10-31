#!/usr/bin/env python3
"""Headless Kaggle pyVideoTrans job adapter for either Edge TTS or VoxCPM2.

Inputs are already staged as a private Kaggle Dataset or /kaggle/input file.
Do not embed Telegram bot tokens or user video data in the notebook source.

Usage:
 python tools/kaggle_video_dub_job.py --task vtv --input /kaggle/input/my-video/clip.mp4 \
   --source-language-code en --target-language-code my --tts voxcpm2 --voice-role Narrator
 python tools/kaggle_video_dub_job.py --task tts --input /kaggle/input/subtitles/translated.srt \
   --target-language-code en --tts edge --voice-role en-US-AriaNeural
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

PROVIDERS = {"edge": 0, "voxcpm2": 38}


def build_command(args, project_root):
    source = Path(args.input).resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Input file does not exist: {source}")
    if args.task == "tts" and source.suffix.lower() != ".srt":
        raise ValueError("Standalone TTS expects an existing translated .srt file")
    if args.task == "vtv" and source.suffix.lower() not in {".mp4", ".mkv", ".mov", ".webm"}:
        raise ValueError("Video translation expects a supported video file")
    if args.task == "vtv" and (not args.source_language_code or not args.target_language_code):
        raise ValueError("Video translation requires source and target language codes")
    if args.tts == "voxcpm2" and not args.cuda:
        raise ValueError("VoxCPM2 requires --cuda and an enabled Kaggle GPU")
    if args.tts == "voxcpm2" and args.reference_wav:
        ref = Path(args.reference_wav).resolve()
        if not ref.is_file():
            raise FileNotFoundError(f"Missing voice reference: {ref}")
    params = [
        sys.executable, str(project_root / "cli.py"),
        "--task", args.task,
        "--name", str(source),
        "--tts_type", str(PROVIDERS[args.tts]),
        "--voice_role", args.voice_role,
        "--output-dir", str(Path(args.temp_output).resolve()),
        "--target_language_code", args.target_language_code,
    ]
    if args.task == "vtv":
        params.extend([
            "--source_language_code", args.source_language_code,
            "--recogn_type", str(args.recogn_type),
            "--model_name", args.whisper_model,
            "--translate_type", str(args.translate_type),
        ])
    if args.cuda:
        params.append("--cuda")
    if args.voice_autorate:
        params.append("--voice_autorate")
    return params


def run(args):
    project_root = Path(__file__).resolve().parents[1]
    output_dir = Path(args.temp_output).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    cmd = build_command(args, project_root)

    env = os.environ.copy()
    env["QT_QPA_PLATFORM"] = "offscreen"
    if args.tts == "voxcpm2":
        env["VOXCPM_MODEL_DIR"] = args.voxcpm_model_dir
        if args.reference_wav:
            env["VOXCPM_REFERENCE_WAV"] = str(Path(args.reference_wav).resolve())
    print("PYVIDEOTRANS_ENGINE", args.tts, "PROVIDER_ID", PROVIDERS[args.tts], flush=True)
    print("PYVIDEOTRANS_MODE", args.task, flush=True)
    subprocess.run(cmd, check=True, env=env, cwd=str(project_root), timeout=args.timeout_seconds)
    suffix = ".wav" if args.task == "tts" else ".mp4"
    candidates = [
        p for p in output_dir.rglob("*")
        if p.is_file() and p.suffix.lower() == suffix and p.stat().st_size > 4096
    ]
    if not candidates:
        raise RuntimeError(f"pyVideoTrans completed but no {suffix} result was located")
    # Prefer the newest modified output if the pipeline generated auxiliary files.
    result = max(candidates, key=lambda p: p.stat().st_mtime)
    dest = Path(args.final_output).resolve()
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(result, dest)
    meta = {
        "status": "completed",
        "engine": args.tts,
        "provider_id": PROVIDERS[args.tts],
        "mode": args.task,
        "output_file": dest.name,
        "bytes": dest.stat().st_size,
    }
    (dest.parent / "pyvideotrans_result.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )
    print("PYVIDEOTRANS_RESULT_READY", json.dumps(meta), flush=True)


def parser():
    p = argparse.ArgumentParser(description="Kaggle pyVideoTrans dual-TTS batch job")
    p.add_argument("--task", choices=["tts", "vtv"], default="vtv")
    p.add_argument("--input", required=True)
    p.add_argument("--tts", choices=sorted(PROVIDERS), required=True)
    p.add_argument("--voice-role", required=True)
    p.add_argument("--source-language-code", default="")
    p.add_argument("--target-language-code", required=True)
    p.add_argument("--recogn-type", type=int, default=0)
    p.add_argument("--whisper-model", default="small")
    p.add_argument("--translate-type", type=int, default=0)
    p.add_argument("--voice-autorate", action="store_true")
    p.add_argument("--cuda", action="store_true")
    p.add_argument("--reference-wav", default="")
    p.add_argument("--voxcpm-model-dir", default="/kaggle/temp/VoxCPM2")
    p.add_argument("--temp-output", default="/kaggle/temp/pyvideotrans-output")
    p.add_argument("--final-output", default="/kaggle/working/pyvideotrans_result.mp4")
    p.add_argument("--timeout-seconds", type=int, default=3600)
    return p


if __name__ == "__main__":
    parsed = parser().parse_args()
    if parsed.task == "tts" and parsed.final_output.endswith(".mp4"):
        parsed.final_output = "/kaggle/working/pyvideotrans_result.wav"
    run(parsed)
