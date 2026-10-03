from __future__ import annotations

import re
import shutil
import subprocess
import time
import urllib.parse
import urllib.request
from pathlib import Path

from videotrans.configure.config import ROOT_DIR, TEMP_DIR


def _file_path(value):
    if not value:
        return None
    if isinstance(value,str):
        return value
    if hasattr(value,"name"):
        return str(value.name)
    if isinstance(value,dict):
        return str(value.get("path") or value.get("name") or "") or None
    return None


def download_remote_media(url,cookie_file=None,user_agent="",referer="",output_dir=None):
    """Download one remote video/audio source.

    yt-dlp handles supported social/video platforms. Google Drive and MEGA
    have explicit adapters. Cookies/user-agent/referer are optional.
    """
    url=str(url or "").strip()
    if not re.match(r"^https?://",url,re.I):
        raise ValueError("Remote source must be an http/https URL.")
    folder=Path(output_dir or (Path(TEMP_DIR)/f"remote-{int(time.time())}"))
    folder.mkdir(parents=True,exist_ok=True)
    host=(urllib.parse.urlparse(url).hostname or "").lower()
    cookie=_file_path(cookie_file)

    if host in {"drive.google.com","docs.google.com"}:
        try: import gdown
        except ImportError as e: raise RuntimeError("Install gdown for Google Drive URLs.") from e
        result=gdown.download(url=url,output=str(folder/"remote-video"),fuzzy=True,quiet=False)
        if result and Path(result).is_file(): return str(Path(result))
        raise RuntimeError("Google Drive download failed.")

    if host=="mega.nz" or host.endswith(".mega.nz"):
        try: from mega import Mega
        except ImportError as e: raise RuntimeError("Install mega.py for MEGA URLs.") from e
        result=Mega().download_url(url,str(folder))
        if result and Path(result).is_file(): return str(Path(result))
        raise RuntimeError("MEGA download failed.")

    last_error=None
    try:
        import yt_dlp
        opts={"outtmpl":str(folder/"remote.%(ext)s"),"noplaylist":True,"restrictfilenames":True,
              "format":"bv*+ba/b","merge_output_format":"mp4","quiet":True}
        if cookie: opts["cookiefile"]=cookie
        headers={}
        if user_agent: headers["User-Agent"]=user_agent
        if referer: headers["Referer"]=referer
        if headers: opts["http_headers"]=headers
        with yt_dlp.YoutubeDL(opts) as ydl:
            ydl.extract_info(url,download=True)
        candidates=[p for p in folder.glob("remote.*") if p.is_file() and not p.name.endswith((".part",".ytdl"))]
        candidates.sort(key=lambda p:(p.suffix.lower() not in {".mp4",".mkv",".webm",".mov"},-p.stat().st_size))
        if candidates: return str(candidates[0])
    except Exception as e:
        last_error=e

    try:
        request=urllib.request.Request(url)
        if user_agent: request.add_header("User-Agent",user_agent)
        if referer: request.add_header("Referer",referer)
        suffix=Path(urllib.parse.urlparse(url).path).suffix or ".mp4"
        dest=folder/f"remote{suffix}"
        with urllib.request.urlopen(request,timeout=120) as response,dest.open("wb") as out:
            shutil.copyfileobj(response,out)
        if dest.stat().st_size>1024: return str(dest)
    except Exception as e:
        raise RuntimeError(f"Remote download failed. yt-dlp: {last_error}; direct: {e}") from e
    raise RuntimeError(f"Remote download failed: {last_error}")


def latest_output():
    root=Path(ROOT_DIR)/"output"
    if not root.exists(): return None,None
    videos=sorted([p for p in root.rglob("*") if p.suffix.lower() in {".mp4",".mkv",".mov",".webm"}],key=lambda p:p.stat().st_mtime,reverse=True)
    subs=sorted([p for p in root.rglob("*") if p.suffix.lower() in {".srt",".ass",".vtt"}],key=lambda p:p.stat().st_mtime,reverse=True)
    return (str(videos[0]) if videos else None,str(subs[0]) if subs else None)


def _escape_sub(path):
    return str(Path(path).resolve()).replace("\\","/").replace(":",r"\:").replace("'",r"\'").replace(",",r"\,")


def _width(path):
    p=subprocess.run(["ffprobe","-v","error","-select_streams","v:0","-show_entries","stream=width",
                      "-of","csv=p=0",str(path)],capture_output=True,text=True,check=True)
    return int(p.stdout.strip())


def render_edit_room(video_path,subtitle_path=None,blur_existing_subtitles=False,blur_height_percent=22,
                     blur_strength=10,burn_generated_subtitles=True,logo_path=None,logo_opacity=.75,
                     logo_width_percent=14,logo_position="Top right"):
    video=Path(video_path or "")
    if not video.is_file(): raise FileNotFoundError("Select a rendered video first.")
    sub=Path(subtitle_path) if subtitle_path else None
    if not sub or not sub.is_file(): burn_generated_subtitles=False
    logo=Path(logo_path) if logo_path else None
    if logo and not logo.is_file(): logo=None
    folder=Path(ROOT_DIR)/"output"/"editor"; folder.mkdir(parents=True,exist_ok=True)
    out=folder/f"edited_{int(time.time())}.mp4"
    args=["ffmpeg","-nostdin","-y","-loglevel","error","-i",str(video)]
    if logo: args+=["-i",str(logo)]
    filters=[]; current="0:v"; n=0
    if blur_existing_subtitles:
        frac=max(.05,min(.5,float(blur_height_percent)/100)); strength=max(1,min(40,float(blur_strength)))
        base=f"v{n}b"; crop=f"v{n}c"; blur=f"v{n}x"; outv=f"v{n}o"
        filters += [f"[{current}]split=2[{base}][{crop}]",
                    f"[{crop}]crop=iw:ih*{frac:.6f}:0:ih*(1-{frac:.6f}),boxblur=luma_radius={strength:.2f}:luma_power=1[{blur}]",
                    f"[{base}][{blur}]overlay=0:H-h[{outv}]"]
        current=outv; n+=1
    if burn_generated_subtitles:
        outv=f"v{n}o"
        filters.append(f"[{current}]subtitles='{_escape_sub(str(sub))}':force_style='FontName=Noto Sans Myanmar,FontSize=22,Outline=2,MarginV=24'[{outv}]")
        current=outv; n+=1
    if logo:
        w=max(32,int(_width(video)*max(.03,min(.5,float(logo_width_percent)/100))))
        alpha=max(0,min(1,float(logo_opacity))); lg=f"lg{n}"; outv=f"v{n}o"
        filters.append(f"[1:v]format=rgba,scale={w}:-1,colorchannelmixer=aa={alpha:.4f}[{lg}]")
        pos=str(logo_position).lower(); x="24" if "left" in pos else "W-w-24"; y="H-h-24" if "bottom" in pos else "24"
        filters.append(f"[{current}][{lg}]overlay={x}:{y}:format=auto[{outv}]"); current=outv
    if filters: args += ["-filter_complex",";".join(filters),"-map",f"[{current}]","-map","0:a?"]
    else: args += ["-map","0:v:0","-map","0:a?"]
    args += ["-c:v","libx264","-preset","fast","-crf","20","-c:a","aac","-b:a","192k","-movflags","+faststart",str(out)]
    subprocess.run(args,check=True)
    return str(out)
