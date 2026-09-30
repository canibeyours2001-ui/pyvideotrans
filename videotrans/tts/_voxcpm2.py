from __future__ import annotations

from pathlib import Path

from videotrans.configure.config import logger


def _torch():
    from videotrans.tts.voxcpm2_engine import _torch as engine_torch

    return engine_torch()


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
from videotrans.tts._base import BaseTTS
from videotrans.tts.voxcpm2_engine import (
    BuiltinVoxCPM2Worker,
    get_voxcpm2_worker,
    split_long_text,
)
from videotrans.tts.voxcpm2_profiles import get_profile
from videotrans.tts.voxcpm2_scheduler import VoxCPM2ChunkJob, VoxCPM2Scheduler
from videotrans.tts.voxcpm2_voice import default_voice_store
from videotrans.util.help_ffmpeg import concat_multi_audio, create_concat_txt
from videotrans.util.help_misc import vail_file


def _split_output_path(filename: Path, sub_index: int) -> Path:
    return filename if sub_index == 1 else filename.with_name(f"{filename.stem}_{sub_index}{filename.suffix}")


class VoxCPM2BuiltinTTS(BaseTTS):
    def _exec(self) -> None:
        if self._exit():
            return
        pending_items = [
            item
            for item in self.queue_tts
            if str(item.get("text", "")).strip() and not vail_file(item.get("filename"))
        ]
        if not pending_items:
            return
        devices = detect_voxcpm2_devices() if getattr(self, "is_cuda", True) else ["cpu"]
        workers = [
            get_voxcpm2_worker(device, index, BuiltinVoxCPM2Worker)
            for index, device in enumerate(devices)
        ]
        jobs: list[VoxCPM2ChunkJob] = []
        merge_groups: dict[int, tuple[Path, list[Path]]] = {}
        store = default_voice_store()
        for index, item in enumerate(pending_items, start=1):
            role = item.get("role") or "movie_recap_fast"
            profile = get_profile(str(role))
            selected = store.get_selected_voice(profile.profile_id)
            if selected is None:
                raise RuntimeError(f"No selected VoxCPM2 voice for {profile.display_name}; use Voice Studio first")
            filename = Path(str(item.get("filename")))
            parts = split_long_text(str(item.get("text", "")))
            outputs = [_split_output_path(filename, sub_index) for sub_index, _part in enumerate(parts, start=1)]
            if len(outputs) > 1:
                merge_groups[index] = (filename, outputs)
            for sub_index, part in enumerate(parts, start=1):
                jobs.append(VoxCPM2ChunkJob(index * 10_000 + sub_index, part, outputs[sub_index - 1], selected))
        results = VoxCPM2Scheduler(workers).run(jobs)
        for final_path, split_paths in merge_groups.values():
            concat_txt = final_path.with_suffix(".concat.txt")
            create_concat_txt(split_paths, concat_txt=concat_txt)
            if not concat_multi_audio(out=final_path.as_posix(), concat_txt=concat_txt.as_posix()):
                raise RuntimeError(f"Failed to merge VoxCPM2 audio chunks for {final_path}")
        logger.info("VoxCPM2 completed %s / %s chunks", len(results), len(jobs))


TTS = VoxCPM2BuiltinTTS
