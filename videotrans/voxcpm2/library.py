from __future__ import annotations

import json
import re
import shutil
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from videotrans.configure.config import ROOT_DIR


PROFILES: Dict[str, Dict[str, str]] = {
    "🎙 U Aung — Deep Calm Male": {
        "direction": "a deep calm mature Burmese male voice, steady, warm, authoritative, natural narration",
        "group": "Male",
    },
    "🧑 Ko Min — Friendly Young Male": {
        "direction": "a friendly young Burmese male voice, conversational, clear, relaxed and approachable",
        "group": "Male",
    },
    "👴 U Htet — Older Wise Male": {
        "direction": "an older wise Burmese male voice, measured pace, textured, thoughtful and dignified",
        "group": "Male",
    },
    "⚡ Ko Zaw — Energetic Presenter": {
        "direction": "an energetic Burmese male presenter, bright, punchy, confident and engaging",
        "group": "Male",
    },
    "👔 U Kyaw — Formal Professional Male": {
        "direction": "a formal professional Burmese male voice, articulate, composed, trustworthy and polished",
        "group": "Male",
    },
    "👩 Daw May — Warm Adult Female": {
        "direction": "a warm adult Burmese female voice, natural, caring, balanced and clear",
        "group": "Female",
    },
    "✨ Ma Su — Bright Young Female": {
        "direction": "a bright young Burmese female voice, fresh, cheerful, crisp and expressive",
        "group": "Female",
    },
    "🌺 Daw Thida — Elegant Mature Female": {
        "direction": "an elegant mature Burmese female voice, poised, smooth, refined and calm",
        "group": "Female",
    },
    "💗 Ma Nandar — Soft Emotional Female": {
        "direction": "a soft emotional Burmese female voice, intimate, gentle, sincere and expressive",
        "group": "Female",
    },
    "📰 Burmese News Presenter": {
        "direction": "a professional Burmese television news presenter, formal, clear, neutral, precise pacing",
        "group": "Narration",
    },
    "🎬 Deep Documentary Narrator": {
        "direction": "a deep cinematic documentary narrator, calm, serious, immersive, controlled pacing",
        "group": "Narration",
    },
    "🔥 Fast Movie Recap Narrator": {
        "direction": "a fast paced cinematic movie recap narrator, energetic, urgent, clear and dramatic",
        "group": "Narration",
    },
}


def library_root() -> Path:
    root = Path(ROOT_DIR) / "models" / "voxcpm2" / "voices"
    root.mkdir(parents=True, exist_ok=True)
    return root


def library_json() -> Path:
    return library_root() / "library.json"


def _load() -> Dict[str, Any]:
    path = library_json()
    if not path.exists():
        return {"version": 1, "voices": {}}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError()
        data.setdefault("version", 1)
        data.setdefault("voices", {})
        return data
    except Exception:
        return {"version": 1, "voices": {}}


def _save(data: Dict[str, Any]) -> None:
    path = library_json()
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def _safe_name(name: str) -> str:
    value = re.sub(r"[^\w\-. ]+", "_", str(name or "voice"), flags=re.UNICODE).strip(" ._")
    return value[:80] or "voice"


def list_saved_voices() -> List[Dict[str, Any]]:
    data = _load()
    rows = []
    for voice_id, item in data["voices"].items():
        row = dict(item)
        row["id"] = voice_id
        wav = library_root() / row.get("file", "")
        if wav.is_file():
            row["path"] = str(wav)
            rows.append(row)
    rows.sort(key=lambda x: float(x.get("created", 0)), reverse=True)
    return rows


def list_roles(include_no: bool = True, include_clone: bool = True) -> List[str]:
    out: List[str] = []
    if include_no:
        out.append("No")
    if include_clone:
        out.append("clone")
    out.extend(PROFILES.keys())
    out.extend(v["name"] for v in list_saved_voices())
    return list(dict.fromkeys(out))


def resolve_role(role: str, *, clone_wav: Optional[str] = None, clone_text: str = "") -> Dict[str, Any]:
    role = str(role or "No")
    if role == "clone":
        if not clone_wav or not Path(clone_wav).is_file():
            raise RuntimeError("VoxCPM2 clone role requires a valid reference WAV.")
        return {
            "kind": "clone",
            "name": "clone",
            "wav": str(clone_wav),
            "transcript": str(clone_text or "").strip(),
            "direction": "",
        }
    if role in PROFILES:
        return {
            "kind": "profile",
            "name": role,
            "wav": None,
            "transcript": "",
            "direction": PROFILES[role]["direction"],
        }
    for item in list_saved_voices():
        if role in (item.get("name"), item.get("id")):
            return {
                "kind": item.get("kind", "saved"),
                "name": item.get("name"),
                "id": item.get("id"),
                "wav": item.get("path"),
                "transcript": item.get("transcript", ""),
                "direction": item.get("direction", ""),
            }
    return {"kind": "neutral", "name": "No", "wav": None, "transcript": "", "direction": ""}


def save_voice(name: str, wav_path: str, transcript: str = "", *, kind: str = "saved", direction: str = "") -> Dict[str, Any]:
    source = Path(wav_path)
    if not source.is_file():
        raise FileNotFoundError(str(source))
    data = _load()
    voice_id = uuid.uuid4().hex[:12]
    ext = source.suffix.lower() if source.suffix else ".wav"
    dest_name = f"{voice_id}_{_safe_name(name)}{ext}"
    dest = library_root() / dest_name
    shutil.copy2(source, dest)
    entry = {
        "name": str(name or "My VoxCPM2 Voice").strip() or "My VoxCPM2 Voice",
        "file": dest_name,
        "transcript": str(transcript or "").strip(),
        "kind": str(kind or "saved"),
        "direction": str(direction or "").strip(),
        "created": time.time(),
    }
    data["voices"][voice_id] = entry
    _save(data)
    return {"id": voice_id, "path": str(dest), **entry}


def delete_voice(identifier: str) -> bool:
    data = _load()
    target_id = None
    for voice_id, item in data["voices"].items():
        if identifier in (voice_id, item.get("name")):
            target_id = voice_id
            break
    if not target_id:
        return False
    item = data["voices"].pop(target_id)
    try:
        (library_root() / item.get("file", "")).unlink(missing_ok=True)
    except Exception:
        pass
    _save(data)
    return True
