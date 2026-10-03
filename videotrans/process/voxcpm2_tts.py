from __future__ import annotations

import json
import time
import traceback
from pathlib import Path

from videotrans.configure.config import logger
from videotrans.voxcpm2.engine import DualGPUVoxCPM, apply_speed, make_generation_kwargs, model_path_from_env
from videotrans.voxcpm2.library import resolve_role


def _log(path, text, type_="logs", callback=None):
    if callback:
        try: callback(text)
        except Exception: pass
    if path:
        try:
            Path(path).write_text(json.dumps({"type": type_, "text": text}, ensure_ascii=False), encoding="utf-8")
        except Exception:
            pass


def _speed(rate):
    try:
        value=float(str(rate or "0").replace("%",""))
    except Exception:
        value=0
    return max(.5,min(2.,1+value/100))


def voxcpm2_fun(queue_tts_file=None, logs_file=None, cfg_value=2.0, inference_timesteps=10,
                 delivery="", custom_style="", progress_callback=None, **kwargs):
    service=None
    try:
        queue=json.loads(Path(queue_tts_file).read_text(encoding="utf-8"))
        if not queue:
            return False,"VoxCPM2 queue is empty"
        _log(logs_file,"VoxCPM2: loading built-in model…",callback=progress_callback)
        service=DualGPUVoxCPM(model_dir=model_path_from_env(),status_callback=lambda s:_log(logs_file,s,callback=progress_callback))
        tasks=[]; post=[]
        common_control=" ".join(x.strip() for x in (delivery,custom_style) if str(x or "").strip())
        for i,item in enumerate(queue):
            text=str(item.get("text") or "").strip()
            filename=str(item.get("filename") or "")
            if not text or not filename:
                continue
            final=Path(filename)
            if final.is_file() and final.stat().st_size>128:
                continue
            role=resolve_role(str(item.get("role") or "No"),clone_wav=item.get("ref_wav"),clone_text=item.get("ref_text",""))
            direction=" ".join(x for x in (role.get("direction",""),common_control) if x).strip()
            raw=Path(str(final)+".voxcpm2-raw.wav")
            tasks.append({"chunk_index":i,"output_path":str(raw),"kwargs":make_generation_kwargs(
                text,reference_wav=role.get("wav"),reference_text=role.get("transcript",""),
                control_instruction=direction,cfg_value=cfg_value,inference_timesteps=inference_timesteps)})
            post.append((i,raw,final,_speed(item.get("rate"))))
        if not tasks:
            return True,None
        raw_by_index={}
        for n,result in enumerate(service.iter_results(tasks),1):
            raw_by_index[int(result["chunk_index"])]=Path(result["output_path"])
            _log(logs_file,f"VoxCPM2 dual-GPU dubbing {n}/{len(tasks)} • worker {result.get('gpu')}",callback=progress_callback)
        for i,raw,final,speed in post:
            generated=raw_by_index.get(i)
            if not generated or not generated.exists():
                continue
            final.parent.mkdir(parents=True,exist_ok=True)
            apply_speed(generated,final,speed)
            generated.unlink(missing_ok=True)
        ok=sum(1 for x in queue if Path(str(x.get("filename") or "")).is_file())
        if ok<1:
            return False,"VoxCPM2 did not generate any dubbing audio"
        _log(logs_file,f"VoxCPM2 dubbing complete: {ok}/{len(queue)}",callback=progress_callback)
        return True,None
    except BaseException as exc:
        msg=traceback.format_exc()
        logger.error(msg)
        _log(logs_file,f"VoxCPM2 error: {exc}","error",callback=progress_callback)
        return False,f"{exc}\n{msg}"
    finally:
        if service:
            try: service.shutdown()
            except Exception: pass
