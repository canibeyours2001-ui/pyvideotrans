"""Headless, single-GPU VoxCPM2 provider for pyVideoTrans batch jobs.

Unlike the existing VOXCPM_TTS (Gradio API), this runs the model locally in
the Kaggle notebook process. Each batch loads weights once and generates
subtitle segments sequentially to avoid concurrent model OOM.
"""
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from videotrans.tts._base import BaseTTS
from videotrans.util.help_misc import vail_file


@dataclass
class VoxCPM2KaggleTTS(BaseTTS):
    _model: Any = field(default=None, init=False, repr=False)

    def _download(self):
        from huggingface_hub import snapshot_download
        from voxcpm import VoxCPM

        model_dir = Path(os.getenv(
            "VOXCPM_MODEL_DIR", "/kaggle/temp/VoxCPM2"
            if Path("/kaggle/temp").is_dir() else
            str(Path.home() / ".cache" / "VoxCPM2")
        )).expanduser().resolve()
        model_dir.mkdir(parents=True, exist_ok=True)

        # The Kaggle working directory is exported as the notebook result.
        # Never download multi-GB model weights into /kaggle/working.
        if not (model_dir / "model.safetensors").is_file():
            snapshot_download(repo_id="openbmb/VoxCPM2", local_dir=str(model_dir))

        self._model = VoxCPM.from_pretrained(
            str(model_dir), load_denoiser=False, optimize=False, device="cuda"
        )

    def _exec(self):
        # Important: BaseTTS._exec() starts several threads. A single T4
        # model is intentionally reused sequentially for subtitle segments.
        for idx, item in enumerate(self.queue_tts):
            if self._exit():
                return
            if not item.get("text", "").strip() or vail_file(item.get("filename", "")):
                continue
            self._run(item, idx)
            self.signal(text=f"VoxCPM2 dubbing [{idx + 1}/{len(self.queue_tts)}]")
        self.signal(text="VoxCPM2 dubbing completed")

    def _run(self, data_item, idx=-1):
        if not data_item.get("text", "").strip() or vail_file(data_item.get("filename", "")):
            return
        if self._model is None:
            raise RuntimeError("VoxCPM2 not loaded; _download() must run first")

        import numpy as np
        import soundfile as sf

        ref_wav = os.getenv("VOXCPM_REFERENCE_WAV", "").strip()
        if ref_wav and not Path(ref_wav).is_file():
            raise FileNotFoundError("VOXCPM_REFERENCE_WAV does not exist")
        if not ref_wav:
            selected, _ = self.get_ref_wav(data_item)
            if selected:
                ref_wav = selected

        kwargs = {
            "text": data_item["text"].strip(),
            "cfg_value": float(os.getenv("VOXCPM_CFG_VALUE", "2.0")),
            "inference_timesteps": int(os.getenv("VOXCPM_INFERENCE_STEPS", "10")),
            "normalize": False,
        }
        if ref_wav:
            kwargs["reference_wav_path"] = ref_wav
        waveform = self._model.generate(**kwargs)
        if hasattr(waveform, "detach"):
            waveform = waveform.detach().cpu().numpy()
        pcm = np.asarray(waveform, dtype=np.float32).reshape(-1)
        if pcm.size == 0:
            raise RuntimeError(f"VoxCPM2 returned empty audio for subtitle {idx + 1}")

        path = Path(data_item["filename"])
        path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(path), pcm, int(self._model.tts_model.sample_rate), subtype="PCM_16")
