from __future__ import annotations

import subprocess
import threading
import time
import uuid
from pathlib import Path

from videotrans.configure.config import TEMP_DIR
from videotrans.voxcpm2.engine import DualGPUVoxCPM, apply_speed, concat_wavs, make_generation_kwargs, model_path_from_env, split_long_text
from videotrans.voxcpm2.library import PROFILES, delete_voice, list_roles, list_saved_voices, resolve_role, save_voice
from videotrans.webui_media import download_remote_media


_STUDIO_SERVICE = None
_STUDIO_SERVICE_LOCK = threading.RLock()


def _get_studio_service():
    """Keep one GPU0 VoxCPM2 replica warm for auditions/clone tests."""
    global _STUDIO_SERVICE
    with _STUDIO_SERVICE_LOCK:
        if _STUDIO_SERVICE is None:
            _STUDIO_SERVICE = DualGPUVoxCPM(
                model_dir=model_path_from_env(),
                gpu_ids=(0,),
            )
        return _STUDIO_SERVICE


def release_studio_service():
    """Release the warm audition model before dual-GPU translation/streaming."""
    global _STUDIO_SERVICE
    with _STUDIO_SERVICE_LOCK:
        service = _STUDIO_SERVICE
        _STUDIO_SERVICE = None
    if service is not None:
        try:
            service.shutdown()
        except Exception:
            pass


def profile_choices():
    return list(PROFILES.keys())

def saved_voice_choices():
    rows=list_saved_voices()
    return [x["name"] for x in rows] or ["No saved voices"]

def role_choices():
    # The studio selects saved cloned voices, not the special per-subtitle
    # "clone" role used by pyVideoTrans's dubbing pipeline.
    return list_roles(include_no=True,include_clone=False)

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
    # A single warm studio worker is shared across UI callbacks. Serialize
    # auditions so concurrent clicks cannot consume each other's queue results.
    with _STUDIO_SERVICE_LOCK:
        service=_get_studio_service()
        result=next(service.iter_results([{"chunk_index":0,"output_path":str(out),"kwargs":kwargs}]))
    return str(out)

def generate_profile_candidate(profile,text,cfg=2.0,steps=10):
    if not str(text or "").strip(): raise ValueError("Enter audition text.")
    out=_folder("candidate")/"candidate.wav"
    _generate_one(text,out,role=profile,cfg=cfg,steps=steps)
    return str(out),str(out),"Candidate ready. It should autoplay. Use Try again for another variation, or choose another profile."



def _decode_audio_for_whisper(path, sampling_rate=16000):
    import numpy as np

    result = subprocess.run(
        [
            "ffmpeg", "-nostdin", "-v", "error",
            "-i", str(path),
            "-vn", "-ac", "1", "-ar", str(int(sampling_rate)),
            "-f", "f32le", "pipe:1",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    if result.returncode != 0:
        raise RuntimeError(
            result.stderr.decode("utf-8", errors="replace").strip()
            or "FFmpeg failed to decode the reference audio."
        )

    audio = np.frombuffer(result.stdout, dtype=np.float32).copy()
    if audio.size == 0:
        raise RuntimeError("No audio samples were decoded from the reference.")
    return audio


def transcribe_clone_reference(reference_wav,language="Burmese / Myanmar"):
    """Transcribe a clone reference.

    Burmese uses the compact real-world Conformer that performed well in our
    notebook tests. Other languages use pyVideoTrans's installed faster-whisper
    large-v3.
    """
    if not reference_wav or not Path(reference_wav).is_file():
        raise ValueError("Prepare a clone reference first.")
    if str(language).lower().startswith(("burmese","myanmar","my")):
        import importlib.util, re, sys, unicodedata
        from huggingface_hub import snapshot_download
        repo=snapshot_download(
            repo_id="freococo/myanmar_asr",token=False,
            allow_patterns=["model.safetensors","model.py","transcribe.py","config.json","vocab.json","cmvn.json","preprocessor_config.json"],
        )
        repo_path=Path(repo)
        if str(repo_path) not in sys.path: sys.path.insert(0,str(repo_path))
        # The published transcribe.py imports its sibling model.py as "model".
        spec=importlib.util.spec_from_file_location("model",str(repo_path/"model.py"))
        module=importlib.util.module_from_spec(spec); sys.modules["model"]=module; spec.loader.exec_module(module)
        tspec=importlib.util.spec_from_file_location("voxcpm2_burmese_asr",str(repo_path/"transcribe.py"))
        tm=importlib.util.module_from_spec(tspec); tspec.loader.exec_module(tm)
        try:
            import torch
            device="cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            device="cpu"
        asr=tm.BurmeseASR(model_dir=str(repo_path),device=device)
        raw=asr.transcribe(str(reference_wav))
        if isinstance(raw,dict): raw=raw.get("text") or raw.get("transcript") or ""
        text=unicodedata.normalize("NFC",str(raw or "").strip())
        myanmar=r"\u1000-\u109F\uA9E0-\uA9FF\uAA60-\uAA7F"
        for _ in range(4):
            text=re.sub(rf"(?<=[{myanmar}])\s+(?=[{myanmar}])","",text)
        return text,"Burmese real-world ASR transcript ready. Correct names/slang before saving."
    from faster_whisper import WhisperModel
    code={"English":"en","Thai":"th","Japanese":"ja","Chinese":"zh"}.get(str(language))
    device="cuda" if _cuda_available() else "cpu"
    model=WhisperModel("large-v3",device=device,compute_type="float16" if device=="cuda" else "int8")
    audio_input=_decode_audio_for_whisper(reference_wav,16000)
    segments,_=model.transcribe(audio_input,language=code,beam_size=5,vad_filter=True)
    text=" ".join(s.text.strip() for s in segments if s.text.strip()).strip()
    return text,"Whisper large-v3 transcript ready."


def _cuda_available():
    try:
        import torch
        return bool(torch.cuda.is_available())
    except Exception:
        return False

def test_clone_voice(reference_wav,reference_text,test_text,cfg=2.0,steps=10):
    if not reference_wav or not Path(reference_wav).is_file(): raise ValueError("Prepare a clone reference first.")
    if not str(reference_text or "").strip(): raise ValueError("Reference transcript is required.")
    if not str(test_text or "").strip(): raise ValueError("Enter clone test text.")
    out=_folder("clone-test")/"test.wav"
    _generate_one(test_text,out,role="clone",clone_wav=reference_wav,clone_text=reference_text,cfg=cfg,steps=steps)
    return str(out),"Clone test ready."

def use_profile_candidate(profile,candidate_path,transcript=""):
    """Promote the audition WAV into a stable reference voice for translation."""
    if not candidate_path or not Path(candidate_path).is_file():
        raise ValueError("Generate a voice candidate first.")

    profile_name = str(profile or "Designed Voice").strip() or "Designed Voice"
    # Keep one obvious active entry per profile instead of filling the library
    # with every audition attempt.
    selected_name = f"Selected • {profile_name}"
    delete_voice(selected_name)

    item = save_voice(
        selected_name,
        candidate_path,
        transcript=transcript,
        kind="designed",
    )
    return (
        role_choices(),
        item["name"],
        (
            f"✅ Using {item['name']} for video translation. "
            "The generated candidate is now the reference voice, so the same "
            "voice is reused across subtitle segments."
        ),
    )


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
    prepared=[]
    release_studio_service()
    service=DualGPUVoxCPM(model_dir=model_path_from_env())
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
