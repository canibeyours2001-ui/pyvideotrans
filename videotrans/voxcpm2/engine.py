from __future__ import annotations
import inspect, multiprocessing as mp, os, queue as pyqueue, random, re, shutil, subprocess, time, traceback, uuid
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence
import numpy as np
import soundfile as sf

DEFAULT_MODEL_ID="openbmb/VoxCPM2"


def _speakable(text: str) -> bool:
    return bool(text and any(ch.isalnum() for ch in text))


def split_long_text(text: str, max_chars: int = 230):
    """Unicode/Burmese-safe long-text splitter."""
    text = re.sub(r"\\r\\n?", "\\n", str(text or "")).strip()
    if not text:
        return []
    max_chars = max(60, int(max_chars))
    sentence_end = set(".!?。！？…၊။;；:\\n")
    chunks, buf = [], []
    def flush():
        value = "".join(buf).strip()
        buf.clear()
        if _speakable(value):
            chunks.append(value)
    for ch in text:
        buf.append(ch)
        if len(buf) >= max_chars:
            joined = "".join(buf)
            cut = -1
            lower = max(0, len(joined) - max(80, max_chars // 2))
            for i in range(len(joined)-1, lower-1, -1):
                if joined[i] in sentence_end or joined[i].isspace():
                    cut = i + 1
                    break
            if 0 < cut < len(joined):
                left, right = joined[:cut].strip(), joined[cut:]
                buf.clear(); buf.extend(right)
                if _speakable(left): chunks.append(left)
            else:
                flush()
        elif ch in sentence_end and len(buf) >= max(45, max_chars // 3):
            flush()
    flush()
    return chunks

def model_path_from_env():
    return os.environ.get("VOXCPM2_MODEL_DIR","").strip() or DEFAULT_MODEL_ID

def make_generation_kwargs(text, reference_wav=None, reference_text="", control_instruction="", cfg_value=2.0, inference_timesteps=10):
    text=str(text or "").strip(); control_instruction=str(control_instruction or "").strip()
    kw={"text":f"({control_instruction}){text}" if control_instruction else text,
        "cfg_value":float(cfg_value),"inference_timesteps":int(inference_timesteps),
        "retry_badcase":True,"normalize":False}
    if reference_wav:
        kw["reference_wav_path"]=str(reference_wav)
        if reference_text and not control_instruction:
            kw["prompt_wav_path"]=str(reference_wav); kw["prompt_text"]=str(reference_text).strip()
    return kw

def _worker(gpu, model_dir, task_q, result_q):
    if gpu is not None: os.environ["CUDA_VISIBLE_DEVICES"]=str(gpu)
    os.environ.setdefault("TOKENIZERS_PARALLELISM","false")
    try:
        import torch
        from voxcpm import VoxCPM
        device="cuda" if gpu is not None and torch.cuda.is_available() else "cpu"
        result_q.put({"status":"loading","gpu":gpu})
        model=VoxCPM.from_pretrained(model_dir,load_denoiser=False,optimize=False,device=device)
        try: sr=int(model.tts_model.sample_rate)
        except Exception: sr=48000
        try:
            sig=inspect.signature(model._generate); params=sig.parameters
            accepts_any=any(p.kind==inspect.Parameter.VAR_KEYWORD for p in params.values()); supported=set(params)
        except Exception: accepts_any=True; supported=set()
        result_q.put({"status":"ready","gpu":gpu,"sample_rate":sr})
    except Exception as e:
        result_q.put({"status":"startup_error","gpu":gpu,"error":repr(e),"traceback":traceback.format_exc()}); return
    while True:
        task=task_q.get()
        if task is None: break
        try:
            kw=dict(task["kwargs"])
            seed=kw.get("seed")
            if seed is not None and not accepts_any and "seed" not in supported:
                seed=int(seed); random.seed(seed); np.random.seed(seed%(2**32-1)); torch.manual_seed(seed); kw.pop("seed",None)
            if not accepts_any:
                for k in list(kw):
                    if k not in supported: kw.pop(k,None)
            wav=np.asarray(model.generate(**kw),dtype=np.float32).reshape(-1)
            out=Path(task["output_path"]); out.parent.mkdir(parents=True,exist_ok=True)
            sf.write(str(out),wav,sr,subtype="PCM_16")
            result_q.put({"status":"ok","batch_id":task["batch_id"],"task_id":task["task_id"],
                          "chunk_index":task["chunk_index"],"gpu":gpu,"output_path":str(out),
                          "duration":len(wav)/sr,"sample_rate":sr})
        except Exception as e:
            result_q.put({"status":"error","batch_id":task["batch_id"],"task_id":task["task_id"],
                          "chunk_index":task.get("chunk_index",0),"gpu":gpu,
                          "error":repr(e),"traceback":traceback.format_exc()})

class DualGPUVoxCPM:
    def __init__(self, model_dir=None, status_callback:Optional[Callable[[str],None]]=None, startup_timeout=900):
        self.model_dir=model_dir or model_path_from_env(); self.status_callback=status_callback
        try:
            import torch; n=torch.cuda.device_count() if torch.cuda.is_available() else 0
        except Exception: n=0
        self.gpu_ids=(0,1) if n>=2 else ((0,) if n==1 else (None,))
        self.ctx=mp.get_context("spawn"); self.task_q=self.ctx.Queue(); self.result_q=self.ctx.Queue()
        self.procs=[]; self.sample_rate=48000
        for gpu in self.gpu_ids: self._start(gpu,startup_timeout)
    def _signal(self,s):
        if self.status_callback:
            try:self.status_callback(s)
            except Exception:pass
    def _start(self,gpu,timeout):
        self._signal(f"Loading VoxCPM2 on {'CPU' if gpu is None else f'GPU {gpu}'}…")
        p=self.ctx.Process(target=_worker,args=(gpu,self.model_dir,self.task_q,self.result_q),daemon=True); p.start(); self.procs.append(p)
        deadline=time.time()+timeout
        while time.time()<deadline:
            if p.exitcode is not None: raise RuntimeError(f"VoxCPM2 worker {gpu} exited ({p.exitcode})")
            try:m=self.result_q.get(timeout=2)
            except pyqueue.Empty:continue
            if m.get("gpu")!=gpu:continue
            if m.get("status")=="ready":
                self.sample_rate=int(m.get("sample_rate",48000)); self._signal(f"VoxCPM2 worker {gpu} ready"); return
            if m.get("status")=="startup_error": raise RuntimeError(m.get("error","startup failed")+"\n"+m.get("traceback",""))
        raise TimeoutError(f"Timed out loading VoxCPM2 worker {gpu}")
    def iter_results(self,tasks:Sequence[Dict[str,Any]])->Iterator[Dict[str,Any]]:
        tasks=[dict(x) for x in tasks]; batch=uuid.uuid4().hex; pending={}; nxt=0; want=0; completed={}
        def submit(i):
            tid=f"{batch}_{i}"; item=dict(tasks[i]); item.update(batch_id=batch,task_id=tid); pending[tid]=i; self.task_q.put(item)
        for _ in range(min(len(tasks),len(self.gpu_ids))): submit(nxt); nxt+=1
        while pending:
            m=self.result_q.get()
            if m.get("batch_id")!=batch or m.get("task_id") not in pending: continue
            i=pending.pop(m["task_id"])
            if m.get("status")!="ok": raise RuntimeError(f"VoxCPM2 chunk {i+1} failed: {m.get('error')}\n{m.get('traceback','')}")
            completed[i]=m
            if nxt<len(tasks): submit(nxt); nxt+=1
            while want in completed:
                yield completed.pop(want); want+=1
    def shutdown(self):
        for _ in self.procs:
            try:self.task_q.put_nowait(None)
            except Exception:pass
        for p in self.procs:
            p.join(timeout=5)
            if p.is_alive(): p.terminate()
    def __enter__(self): return self
    def __exit__(self,*_): self.shutdown()

def _atempo(speed):
    speed=max(.25,min(4.,float(speed))); f=[]
    while speed>2:f.append(2.);speed/=2
    while speed<.5:f.append(.5);speed/=.5
    f.append(speed);return ",".join(f"atempo={x:.6f}" for x in f)

def apply_speed(src,dst,speed):
    if abs(float(speed)-1)<1e-4: shutil.copy2(src,dst); return str(dst)
    subprocess.run(["ffmpeg","-nostdin","-y","-loglevel","error","-i",str(src),"-af",_atempo(speed),"-c:a","pcm_s16le",str(dst)],check=True)
    return str(dst)

def concat_wavs(paths,out,pause_ms=0):
    arrays=[]; sr=None
    for i,p in enumerate(paths):
        a,s=sf.read(str(p),dtype="float32",always_2d=False)
        if getattr(a,"ndim",1)>1:a=a.mean(axis=1)
        sr=int(s) if sr is None else sr
        arrays.append(a)
        if pause_ms and i<len(paths)-1: arrays.append(np.zeros(int(sr*pause_ms/1000),dtype=np.float32))
    if sr is None: raise RuntimeError("No audio chunks")
    sf.write(str(out),np.concatenate(arrays),sr,subtype="PCM_16"); return str(out)
