from __future__ import annotations

import subprocess
import time
import uuid
from pathlib import Path

from videotrans.configure.config import TEMP_DIR
from videotrans.voxcpm2.engine import DualGPUVoxCPM, apply_speed, concat_wavs, make_generation_kwargs, model_path_from_env, split_long_text
from videotrans.voxcpm2.library import PROFILES, delete_voice, list_roles, list_saved_voices, resolve_role, save_voice
from videotrans.webui_media import download_remote_media


def profile_choices():
    return list(PROFILES.keys())

def saved_voice_choices():
    rows=list_saved_voices()
    return [x["name"] for x in rows] or ["No saved voices"]

def role_choices():
    return list_roles(include_no=True,include_clone=True)

def _folder(prefix):
    p=Path(TEMP_DIR)/"voxcpm2_studio"/f"{prefix}-{int(time.time())}-{uuid.uuid4().hex[:6]}"
    p.mkdir(parents=True,exist_ok=True)
    return p

def _extract_ref(source,dest,max_seconds=20,denoise=False):
    filt=["highpass=f=55","lowpass=f=14500"]
    if denoise: filt.append("afftdn=nf=-25")
    subprocess.run(["ffmpeg","-nostdin","-y","-loglevel","error","-i",str(source),"-vn","-ac","1","-ar","48000",
                    "-t",str(max(5,min(28,float(max_seconds)))),"-af",",".join(filt),"-c:a","pcm_s16le",str(dest)],check=True)
    return str(dest)

def prepare_clone_reference(source_mode,upload_path,remote_url,cookie_file=None,user_agent="",referer="",max_seconds=20,denoise=False):
    folder=_folder("clone")
    if str(source_mode).lower().startswith("remote"):
        source=download_remote_media(remote_url,cookie_file=cookie_file,user_agent=user_agent,referer=referer,output_dir=str(folder/"download"))
    else:
        if not upload_path or not Path(upload_path).is_file(): raise ValueError("Upload an audio/video reference first.")
        source=str(upload_path)
    ref=_extract_ref(source,folder/"reference.wav",max_seconds=max_seconds,denoise=denoise)
    return ref,ref,f"Reference ready: {Path(source).name}"

def _generate_one(text,out,role="No",clone_wav=None,clone_text="",delivery="",custom="",cfg=2.0,steps=10):
    info=resolve_role(role,clone_wav=clone_wav,clone_text=clone_text)
    direction=" ".join(x for x in (info.get("direction",""),delivery,custom) if str(x or "").strip()).strip()
    kwargs=make_generation_kwargs(text,reference_wav=info.get("wav"),reference_text=info.get("transcript",""),
                                   control_instruction=direction,cfg_value=cfg,inference_timesteps=steps)
    with DualGPUVoxCPM(model_dir=model_path_from_env()) as service:
        result=next(service.iter_results([{"chunk_index":0,"output_path":str(out),"kwargs":kwargs}]))
    return str(out)

def generate_profile_candidate(profile,text,cfg=2.0,steps=10):
    if not str(text or "").strip(): raise ValueError("Enter audition text.")
    out=_folder("candidate")/"candidate.wav"
    _generate_one(text,out,role=profile,cfg=cfg,steps=steps)
    return str(out),str(out),"Candidate ready. Listen, then save it if you like it."

def test_clone_voice(reference_wav,reference_text,test_text,cfg=2.0,steps=10):
    if not reference_wav or not Path(reference_wav).is_file(): raise ValueError("Prepare a clone reference first.")
    if not str(reference_text or "").strip(): raise ValueError("Reference transcript is required.")
    if not str(test_text or "").strip(): raise ValueError("Enter clone test text.")
    out=_folder("clone-test")/"test.wav"
    _generate_one(test_text,out,role="clone",clone_wav=reference_wav,clone_text=reference_text,cfg=cfg,steps=steps)
    return str(out),"Clone test ready."

def save_candidate_voice(name,candidate_path,transcript=""):
    item=save_voice(name,candidate_path,transcript=transcript,kind="designed")
    choices=saved_voice_choices()
    return choices,item["name"],f"Saved voice: {item['name']}"

def save_clone_voice(name,reference_wav,reference_text):
    item=save_voice(name,reference_wav,transcript=reference_text,kind="clone")
    choices=saved_voice_choices()
    return choices,item["name"],f"Saved cloned voice: {item['name']}"

def delete_saved_voice(name):
    ok=False if not name or name=="No saved voices" else delete_voice(name)
    choices=saved_voice_choices()
    return choices,choices[0],f"Deleted: {name}" if ok else "Nothing deleted."

def stream_voiceover(script,role,delivery,custom,generation_speed,cfg,steps,chunk_chars,join_pause_ms):
    """Gradio streaming generator. Each yield adds one ordered audio chunk."""
    import numpy as np
    import soundfile as sf
    chunks=split_long_text(script,max_chars=int(chunk_chars))
    if not chunks: raise ValueError("Enter a script with speakable text.")
    folder=_folder("long"); info=resolve_role(role)
    direction=" ".join(x for x in (info.get("direction",""),delivery,custom) if str(x or "").strip()).strip()
    tasks=[]
    for i,chunk in enumerate(chunks):
        tasks.append({"chunk_index":i,"output_path":str(folder/f"raw_{i:04d}.wav"),
                      "kwargs":make_generation_kwargs(chunk,reference_wav=info.get("wav"),reference_text=info.get("transcript",""),
                                                       control_instruction=direction,cfg_value=cfg,inference_timesteps=steps)})
    prepared=[]; service=DualGPUVoxCPM(model_dir=model_path_from_env())
    try:
        for n,result in enumerate(service.iter_results(tasks),1):
            raw=result["output_path"]; ready=folder/f"ready_{result['chunk_index']:04d}.wav"
            apply_speed(raw,ready,generation_speed); prepared.append(str(ready))
            audio,sr=sf.read(str(ready),dtype="float32",always_2d=False)
            if getattr(audio,"ndim",1)>1: audio=audio.mean(axis=1)
            if int(join_pause_ms)>0:
                audio=np.concatenate([audio,np.zeros(int(sr*int(join_pause_ms)/1000),dtype=np.float32)])
            yield (int(sr),audio),f"Live: {n}/{len(chunks)} chunks ready • GPU {result.get('gpu')}",None
        final=folder/f"voiceover_{int(time.time())}.wav"; concat_wavs(prepared,final,pause_ms=int(join_pause_ms))
        yield None,f"Finished: {len(chunks)}/{len(chunks)} chunks",str(final)
    finally:
        service.shutdown()
