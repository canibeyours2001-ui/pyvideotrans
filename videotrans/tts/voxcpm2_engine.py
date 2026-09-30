from __future__ import annotations

import importlib
import inspect
import re
import shutil
import threading
from pathlib import Path
from typing import Callable, cast

import numpy as np
import soundfile as sf

from videotrans.configure.config import logger
from videotrans.tts.voxcpm2_profiles import VoxCPM2Profile
from videotrans.tts.voxcpm2_scheduler import VoxCPM2ChunkJob
from videotrans.tts.voxcpm2_voice import SelectedVoice, build_voxcpm2_generate_kwargs


def _torch():
    return importlib.import_module("torch")


def detect_voxcpm2_devices(max_workers: int | None = None) -> list[str]:
    try:
        torch = _torch()
    except ImportError:
        return ["cpu"]
    if torch.cuda.is_available() and torch.cuda.device_count() > 0:
        count = torch.cuda.device_count()
        if max_workers is not None:
            count = min(count, max_workers)
        return [f"cuda:{index}" for index in range(count)]
    return ["cpu"]


def split_long_text(text: str, max_chars: int = 4096) -> list[str]:
    if len(text) <= max_chars:
        return [text]
    sentences = [part.strip() for part in re.findall(r"[^.!?。！？]+[.!?。！？]?", text) if part.strip()]
    parts: list[str] = []
    current = ""
    for sentence in sentences:
        candidate = f"{current} {sentence}".strip() if current else sentence
        if len(candidate) <= max_chars:
            current = candidate
            continue
        if current:
            parts.append(current)
        if len(sentence) <= max_chars:
            current = sentence
            continue
        parts.extend(sentence[start : start + max_chars] for start in range(0, len(sentence), max_chars))
        current = ""
    if current:
        parts.append(current)
    return parts


class BuiltinVoxCPM2Worker:
    def __init__(self, worker_id: int, device: str, model_factory: Callable[[str], object] | None = None):
        self.worker_id = worker_id
        self.device = device
        self._model_factory = model_factory or self._default_model_factory
        self._model = None
        self._lock = threading.RLock()

    def _default_model_factory(self, device: str):
        voxcpm_module = importlib.import_module("voxcpm")
        return voxcpm_module.VoxCPM.from_pretrained(
            "openbmb/VoxCPM2",
            device=device,
            load_denoiser=False,
        )

    @property
    def model(self):
        with self._lock:
            if self._model is None:
                self._model = self._model_factory(self.device)
                logger.info("VoxCPM2 worker %s ready on %s", self.worker_id, self.device)
                print(f"VoxCPM2 worker {self.worker_id} ready on {self.device}", flush=True)
            return self._model

    def generate(self, job: VoxCPM2ChunkJob) -> str:
        if job.selected_voice is None:
            raise RuntimeError("VoxCPM2 selected voice is required")
        return str(self.generate_to_file(job.text, Path(job.output_path), job.selected_voice))

    def generate_to_file(self, text: str, output_path: str | Path, selected_voice: SelectedVoice) -> Path:
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        result = self._run_generate(build_voxcpm2_generate_kwargs(text, selected_voice))
        self._write_generation_result(result, output)
        return output

    def generate_profile_audition(self, profile: VoxCPM2Profile, text: str, output_path: str | Path) -> Path:
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        result = self._run_generate(
            {
                "text": f"({profile.instruction}){text}",
                "cfg_value": 2.0,
                "inference_timesteps": 10,
                "max_len": 4096,
                "retry_badcase": True,
                "retry_badcase_max_times": 3,
                "normalize": True,
                "denoise": False,
                "streaming": False,
            }
        )
        self._write_generation_result(result, output)
        return output

    def _run_generate(self, kwargs: dict[str, str | float | int | bool]) -> object:
        with self._lock:
            generate = getattr(self.model, "generate")
            call_kwargs = {key: value for key, value in kwargs.items() if key != "streaming"}
            signature = inspect.signature(generate)
            accepts_kwargs = any(param.kind == inspect.Parameter.VAR_KEYWORD for param in signature.parameters.values())
            filtered = call_kwargs if accepts_kwargs else {key: value for key, value in call_kwargs.items() if key in signature.parameters}
            return generate(**filtered)

    def _write_generation_result(self, result: object, output: Path) -> None:
        if isinstance(result, str | Path):
            source = Path(result)
            if not source.exists():
                raise RuntimeError(f"VoxCPM2 did not return a valid WAV: {source}")
            if source != output:
                shutil.copyfile(source, output)
            self._validate_wav(output)
            return
        waveform = self._waveform_from_result(result)
        sample_rate = int(getattr(getattr(self.model, "tts_model", None), "sample_rate", 24000))
        sf.write(str(output), waveform, sample_rate)
        self._validate_wav(output)

    @staticmethod
    def _validate_wav(path: Path) -> None:
        try:
            info = sf.info(str(path))
        except RuntimeError as exc:
            raise RuntimeError(f"VoxCPM2 did not return a valid WAV: {path}") from exc
        if info.format.upper() != "WAV" or info.frames <= 0 or info.samplerate <= 0:
            raise RuntimeError(f"VoxCPM2 did not return a valid WAV: {path}")

    @staticmethod
    def _waveform_from_result(result: object) -> np.ndarray:
        if inspect.isgenerator(result):
            chunks = [np.asarray(chunk, dtype=np.float32).reshape(-1) for chunk in result]
            if not chunks:
                raise RuntimeError("VoxCPM2 did not return audio data")
            return np.concatenate(chunks)
        waveform = np.asarray(result, dtype=np.float32)
        if waveform.ndim > 1:
            waveform = np.squeeze(waveform)
        if waveform.size == 0:
            raise RuntimeError("VoxCPM2 did not return audio data")
        return waveform


_workers: dict[tuple[type[BuiltinVoxCPM2Worker], str], BuiltinVoxCPM2Worker] = {}
_workers_lock = threading.Lock()


def get_voxcpm2_worker(
    device: str,
    worker_id: int = 0,
    worker_type: type[BuiltinVoxCPM2Worker] | None = None,
) -> BuiltinVoxCPM2Worker:
    worker_type = worker_type or BuiltinVoxCPM2Worker
    key = (worker_type, device)
    with _workers_lock:
        worker = _workers.get(key)
        if worker is None:
            worker = worker_type(worker_id=worker_id, device=device)
            _workers[key] = worker
        return cast(BuiltinVoxCPM2Worker, worker)


def generate_voxcpm2_audition(profile: VoxCPM2Profile, text: str, output_path: str | Path) -> Path:
    device = detect_voxcpm2_devices(max_workers=1)[0]
    worker = get_voxcpm2_worker(device)
    return worker.generate_profile_audition(profile, text, output_path)
