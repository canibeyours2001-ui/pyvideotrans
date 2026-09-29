from __future__ import annotations

import importlib
import inspect
import re
import shutil
import threading
from pathlib import Path
from typing import Callable

import numpy as np
import soundfile as sf

from videotrans.configure.config import logger
from videotrans.tts.voxcpm2_profiles import VoxCPM2Profile
from videotrans.tts.voxcpm2_scheduler import VoxCPM2ChunkJob
from videotrans.tts.voxcpm2_voice import (
    SelectedVoice,
    build_voxcpm2_generate_kwargs,
)


# ==========================================================
# Device detection
# ==========================================================

def _torch():
    return importlib.import_module("torch")


def detect_voxcpm2_devices(max_workers: int | None = None) -> list[str]:
    try:
        torch = _torch()
    except ImportError:
        return ["cpu"]

    if torch.cuda.is_available():
        count = torch.cuda.device_count()

        if max_workers:
            count = min(count, max_workers)

        return [
            f"cuda:{i}"
            for i in range(count)
        ]

    return ["cpu"]


# ==========================================================
# Text splitting
# ==========================================================

def split_long_text(
    text: str,
    max_chars: int = 4096
) -> list[str]:

    if len(text) <= max_chars:
        return [text]

    sentences = [
        x.strip()
        for x in re.findall(
            r"[^.!?。！？]+[.!?。！？]?",
            text
        )
        if x.strip()
    ]

    result = []
    current = ""

    for sentence in sentences:

        test = (
            f"{current} {sentence}"
            if current
            else sentence
        )

        if len(test) <= max_chars:
            current = test

        else:

            if current:
                result.append(current)

            current = sentence

    if current:
        result.append(current)

    return result


# ==========================================================
# Persistent worker cache
# ==========================================================

_WORKERS = {}
_WORKER_LOCK = threading.Lock()


def get_voxcpm2_worker(
    worker_id: int = 0,
    device: str = "cuda:0"
):

    key = f"{worker_id}:{device}"

    with _WORKER_LOCK:

        if key not in _WORKERS:

            logger.info(
                "Creating VoxCPM2 worker %s on %s",
                worker_id,
                device,
            )

            _WORKERS[key] = BuiltinVoxCPM2Worker(
                worker_id,
                device,
            )

        return _WORKERS[key]


# ==========================================================
# Worker
# ==========================================================

class BuiltinVoxCPM2Worker:

    def __init__(
        self,
        worker_id: int,
        device: str,
        model_factory: Callable[[str], object] | None = None,
    ):

        self.worker_id = worker_id
        self.device = device
        self._model_factory = (
            model_factory
            or self._default_model_factory
        )

        self._model = None


    def _default_model_factory(
        self,
        device: str
    ):

        voxcpm = importlib.import_module(
            "voxcpm"
        )

        return voxcpm.VoxCPM.from_pretrained(
            "openbmb/VoxCPM2",
            device=device,
            load_denoiser=False,
        )


    @property
    def model(self):

        if self._model is None:

            self._model = self._model_factory(
                self.device
            )

            logger.info(
                "VoxCPM2 worker %s ready on %s",
                self.worker_id,
                self.device,
            )

            print(
                f"VoxCPM2 worker {self.worker_id} ready on {self.device}",
                flush=True,
            )

        return self._model


    # ======================================================
    # Normal dubbing
    # ======================================================

    def generate(
        self,
        job: VoxCPM2ChunkJob
    ):

        if job.selected_voice is None:
            raise RuntimeError(
                "VoxCPM2 selected voice missing"
            )


        return str(
            self.generate_to_file(
                job.text,
                Path(job.output_path),
                job.selected_voice,
            )
        )


    def generate_to_file(
        self,
        text: str,
        output_path: str | Path,
        selected_voice: SelectedVoice,
    ):

        output = Path(output_path)

        output.parent.mkdir(
            parents=True,
            exist_ok=True,
        )


        result = self._run_generate(
            build_voxcpm2_generate_kwargs(
                text,
                selected_voice,
            )
        )


        self._write_generation_result(
            result,
            output,
        )


        return output


    # ======================================================
    # Voice audition
    # ======================================================

    def generate_profile_audition(
        self,
        profile: VoxCPM2Profile,
        text: str,
        output_path: str | Path,
    ):

        output = Path(output_path)

        output.parent.mkdir(
            parents=True,
            exist_ok=True,
        )


        result = self._run_generate(
            {
                "text":
                    f"({profile.instruction}) {text}",

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


        self._write_generation_result(
            result,
            output,
        )


        return output


    # ======================================================
    # VoxCPM generation wrapper
    # ======================================================

    def _run_generate(
        self,
        kwargs
    ):

        generate = getattr(
            self.model,
            "generate",
            None,
        )


        if generate is None:

            generate = getattr(
                self.model,
                "_generate",
            )


        signature = inspect.signature(
            generate
        )


        allowed = {}

        for key,value in kwargs.items():

            if key in signature.parameters:

                allowed[key] = value


        return generate(
            **allowed
        )


    # ======================================================
    # Save audio
    # ======================================================

    def _write_generation_result(
        self,
        result,
        output: Path,
    ):


        if isinstance(
            result,
            (str, Path)
        ):

            source = Path(result)

            if source.exists() and source != output:

                shutil.copyfile(
                    source,
                    output,
                )

            return


        waveform = self._waveform_from_result(
            result
        )


        sample_rate = int(
            getattr(
                getattr(
                    self.model,
                    "tts_model",
                    None,
                ),
                "sample_rate",
                24000,
            )
        )


        sf.write(
            str(output),
            waveform,
            sample_rate,
        )



    @staticmethod
    def _waveform_from_result(
        result
    ):


        if inspect.isgenerator(
            result
        ):

            chunks = [
                np.asarray(
                    x,
                    dtype=np.float32
                ).reshape(-1)

                for x in result
            ]


            if not chunks:
                raise RuntimeError(
                    "Empty VoxCPM2 audio"
                )


            return np.concatenate(
                chunks
            )


        audio = np.asarray(
            result,
            dtype=np.float32,
        )


        if audio.ndim > 1:
            audio = np.squeeze(audio)


        if audio.size == 0:

            raise RuntimeError(
                "Empty VoxCPM2 audio"
            )


        return audio



# ==========================================================
# Public audition API
# ==========================================================


def generate_voxcpm2_audition(
    profile: VoxCPM2Profile,
    text: str,
    output_path: str | Path,
):

    device = detect_voxcpm2_devices(
        max_workers=1
    )[0]


    worker = get_voxcpm2_worker(
        worker_id=0,
        device=device,
    )


    return worker.generate_profile_audition(
        profile,
        text,
        output_path,
    )
