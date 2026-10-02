from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

from videotrans.configure.config import TEMP_DIR, params
from videotrans.tts._base import BaseTTS


@dataclass
class VoxCPM2BuiltInTTS(BaseTTS):
    """Built-in VoxCPM2 provider: no external Gradio API required."""

    def __post_init__(self):
        super().__post_init__()
        self.model_name="openbmb/VoxCPM2"

    def _exec(self):
        queue_file=f"{TEMP_DIR}/{self.uuid}/voxcpm2-{time.time()}.json"
        logs_file=f"{TEMP_DIR}/{self.uuid}/voxcpm2-{time.time()}.log"
        Path(queue_file).write_text(json.dumps(self.queue_tts,ensure_ascii=False),encoding="utf-8")
        from videotrans.process.voxcpm2_tts import voxcpm2_fun
        # IMPORTANT: do not wrap this coordinator in pyVideoTrans's
        # multiprocessing.Pool. Pool workers are daemonic and Python forbids
        # them from spawning our GPU0/GPU1 model workers.
        ok, err = voxcpm2_fun(
            queue_tts_file=queue_file,
            logs_file=logs_file,
            cfg_value=float(params.get("voxcpm2_cfg",2.0) or 2.0),
            inference_timesteps=int(float(params.get("voxcpm2_steps",10) or 10)),
            delivery=str(params.get("voxcpm2_delivery","") or ""),
            custom_style=str(params.get("voxcpm2_custom_style","") or ""),
        )
        if not ok:
            from videotrans.configure.excepts import VideoTransError
            raise VideoTransError(err or "VoxCPM2 dubbing failed")
        return True
