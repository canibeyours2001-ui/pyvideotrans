from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Final

from videotrans.configure.config import ROOT_DIR
from videotrans.tts.voxcpm2_profiles import get_profile

MODEL_ID: Final[str] = "openbmb/VoxCPM2"
MODEL_VERSION: Final[str] = "2.0.3"


@dataclass(frozen=True, slots=True)
class AuditionVoice:
    profile_id: str
    profile_name: str
    wav_path: Path
    test_script: str
    timestamp: str
    model_id: str = MODEL_ID
    model_version: str = MODEL_VERSION


@dataclass(frozen=True, slots=True)
class SelectedVoice:
    profile_id: str
    profile_name: str
    wav_path: Path
    test_script: str
    timestamp: str
    model_id: str = MODEL_ID
    model_version: str = MODEL_VERSION


class VoxCPM2VoiceStore:
    def __init__(self, root_dir: str | Path | None = None):
        self.root_dir = Path(root_dir) if root_dir is not None else Path(ROOT_DIR) / "storage" / "voxcpm2_voices"
        self.root_dir.mkdir(parents=True, exist_ok=True)

    def save_audition(
        self,
        profile_id: str,
        generated_wav: str | Path,
        test_script: str,
        model_id: str = MODEL_ID,
        model_version: str = MODEL_VERSION,
    ) -> AuditionVoice:
        profile = get_profile(profile_id)
        timestamp = datetime.now(timezone.utc).isoformat()
        profile_dir = self._profile_dir(profile.profile_id)
        audition_dir = profile_dir / "auditions"
        audition_dir.mkdir(parents=True, exist_ok=True)
        wav_path = audition_dir / f"{self._safe_timestamp(timestamp)}.wav"
        shutil.copyfile(generated_wav, wav_path)
        audition = AuditionVoice(profile.profile_id, profile.display_name, wav_path, test_script, timestamp, model_id, model_version)
        self._write_json(profile_dir / "latest_audition.json", self._voice_to_json(audition))
        return audition

    def select_latest_audition(self, profile_id: str) -> SelectedVoice:
        profile = get_profile(profile_id)
        latest_path = self._profile_dir(profile.profile_id) / "latest_audition.json"
        if not latest_path.exists():
            raise FileNotFoundError(f"No successful audition exists for {profile.display_name}")
        data = self._read_json(latest_path)
        selected_dir = self._profile_dir(profile.profile_id) / "selected"
        selected_dir.mkdir(parents=True, exist_ok=True)
        source_wav = Path(data["wav_path"])
        selected_wav = selected_dir / "reference.wav"
        shutil.copyfile(source_wav, selected_wav)
        selected = SelectedVoice(
            profile_id=profile.profile_id,
            profile_name=profile.display_name,
            wav_path=selected_wav,
            test_script=str(data["test_script"]),
            timestamp=datetime.now(timezone.utc).isoformat(),
            model_id=str(data.get("model_id", MODEL_ID)),
            model_version=str(data.get("model_version", MODEL_VERSION)),
        )
        self._write_json(selected_dir / "metadata.json", self._voice_to_json(selected))
        return selected

    def get_selected_voice(self, profile_id: str) -> SelectedVoice | None:
        profile = get_profile(profile_id)
        metadata = self._profile_dir(profile.profile_id) / "selected" / "metadata.json"
        if not metadata.exists():
            return None
        data = self._read_json(metadata)
        return SelectedVoice(
            profile_id=str(data["profile_id"]),
            profile_name=str(data["profile_name"]),
            wav_path=Path(data["wav_path"]),
            test_script=str(data["test_script"]),
            timestamp=str(data["timestamp"]),
            model_id=str(data.get("model_id", MODEL_ID)),
            model_version=str(data.get("model_version", MODEL_VERSION)),
        )

    def start_replace(self, profile_id: str) -> None:
        self._profile_dir(get_profile(profile_id).profile_id).mkdir(parents=True, exist_ok=True)

    def _profile_dir(self, profile_id: str) -> Path:
        path = self.root_dir / profile_id
        path.mkdir(parents=True, exist_ok=True)
        return path

    @staticmethod
    def _safe_timestamp(timestamp: str) -> str:
        return timestamp.replace(":", "-").replace("+", "_")

    @staticmethod
    def _voice_to_json(voice: AuditionVoice | SelectedVoice) -> dict[str, str]:
        data = asdict(voice)
        data["wav_path"] = str(voice.wav_path)
        return data

    @staticmethod
    def _write_json(path: Path, data: dict[str, str]) -> None:
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    @staticmethod
    def _read_json(path: Path) -> dict[str, str]:
        return json.loads(path.read_text(encoding="utf-8"))


def default_voice_store() -> VoxCPM2VoiceStore:
    return VoxCPM2VoiceStore()


def build_voxcpm2_generate_kwargs(text: str, selected_voice: SelectedVoice) -> dict[str, str | float | int | bool]:
    return {
        "text": text,
        "prompt_wav_path": str(selected_voice.wav_path),
        "prompt_text": selected_voice.test_script,
        "reference_wav_path": str(selected_voice.wav_path),
        "cfg_value": 2.0,
        "inference_timesteps": 10,
        "max_len": 4096,
        "retry_badcase": True,
        "retry_badcase_max_times": 3,
        "normalize": True,
        "denoise": True,
        "streaming": False,
    }


def main() -> int:
    from videotrans.tts.voxcpm2_profiles import main as profiles_main

    return profiles_main()


if __name__ == "__main__":
    raise SystemExit(main())
