from __future__ import annotations

import argparse
import ipaddress
import re
import shutil
import socket
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Final
from urllib.parse import unquote, urlparse

import requests

from videotrans.configure.config import ROOT_DIR, logger

YTDLP_PROVIDERS: Final[set[str]] = {"youtube", "tiktok", "rednote", "douyin", "bilibili"}
SUPPORTED_PROVIDERS: Final[tuple[str, ...]] = (
    "local",
    "direct",
    "google_drive",
    "mega",
    "youtube",
    "tiktok",
    "rednote",
    "douyin",
    "bilibili",
)
LABEL_TO_PROVIDER: Final[dict[str, str]] = {
    "Local File": "local",
    "Direct Download URL": "direct",
    "Google Drive": "google_drive",
    "MEGA": "mega",
    "YouTube": "youtube",
    "TikTok": "tiktok",
    "RedNote / Xiaohongshu": "rednote",
    "Douyin": "douyin",
    "Bilibili": "bilibili",
}
SAFE_SUFFIXES: Final[set[str]] = {".mp4", ".mkv", ".mov", ".webm", ".mp3", ".wav", ".m4a", ".aac", ".flac", ".ogg"}


@dataclass(frozen=True, slots=True)
class ImportedMedia:
    local_path: Path
    source_provider: str
    original_url: str
    title: str = ""
    metadata: dict[str, str] = field(default_factory=dict)


class RemoteImportError(RuntimeError):
    """Raised when remote media import cannot produce a local media file."""


def _YoutubeDL():
    from yt_dlp import YoutubeDL

    return YoutubeDL


def _gdown():
    import gdown

    return gdown


def _Mega():
    from mega import Mega

    return Mega


def safe_filename(name: str) -> str:
    cleaned = unquote(name).replace("\\", "/").split("/")[-1].strip()
    cleaned = re.sub(r"[^A-Za-z0-9._ -]+", "", cleaned).strip(" .")
    return cleaned or "download.bin"


def provider_from_label(label: str) -> str:
    return LABEL_TO_PROVIDER.get(label, label)


def _filename_from_headers(url: str, headers: dict[str, str]) -> str:
    disposition = headers.get("content-disposition", "")
    match = re.search(r'filename\*?=(?:UTF-8\'\')?"?([^";]+)', disposition, flags=re.I)
    if match:
        return safe_filename(match.group(1))
    parsed_name = Path(urlparse(url).path).name
    return safe_filename(parsed_name)


def _is_private_address(hostname: str) -> bool:
    try:
        addresses = socket.getaddrinfo(hostname, None)
    except socket.gaierror as exc:
        raise RemoteImportError(f"Could not resolve host: {hostname}") from exc
    for addr in addresses:
        ip = ipaddress.ip_address(addr[4][0])
        if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_multicast or ip.is_reserved:
            return True
    return False


def _reject_private_network_url(url: str) -> None:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        raise RemoteImportError(f"Unsupported URL scheme: {parsed.scheme or 'missing'}")
    if not parsed.hostname:
        raise RemoteImportError("Invalid URL: missing host")
    if _is_private_address(parsed.hostname):
        raise RemoteImportError("Refusing to download from private or local network address")


class RemoteMediaImporter:
    def __init__(self, storage_dir: str | Path | None = None, cookies_file: str | Path | None = None):
        root = Path(storage_dir) if storage_dir is not None else Path(ROOT_DIR) / "storage" / "imports"
        self.storage_dir = root
        self.cookies_file = Path(cookies_file) if cookies_file else None

    def import_media(self, provider: str, source: str) -> ImportedMedia:
        normalized = provider_from_label(provider)
        if normalized not in SUPPORTED_PROVIDERS:
            raise RemoteImportError(f"Unsupported media source: {provider}")
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        match normalized:
            case "local":
                return self._import_local(source)
            case "direct":
                return self._import_direct(source)
            case "google_drive":
                return self._import_google_drive(source)
            case "mega":
                return self._import_mega(source)
            case "youtube" | "tiktok" | "rednote" | "douyin" | "bilibili":
                return self._import_ytdlp(normalized, source)
            case _:
                raise RemoteImportError(f"Unsupported media source: {provider}")

    def _provider_dir(self, provider: str) -> Path:
        path = self.storage_dir / provider / uuid.uuid4().hex
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _import_local(self, source: str) -> ImportedMedia:
        path = Path(source).expanduser().resolve()
        if not path.exists() or not path.is_file():
            raise RemoteImportError(f"Local media file does not exist: {source}")
        return ImportedMedia(local_path=path, source_provider="local", original_url=source, title=path.name, metadata={})

    def _import_direct(self, source: str) -> ImportedMedia:
        _reject_private_network_url(source)
        dest_dir = self._provider_dir("direct")
        try:
            with requests.get(source, stream=True, timeout=(10, 60), allow_redirects=True) as response:
                response.raise_for_status()
                filename = _filename_from_headers(source, dict(response.headers))
                suffix = Path(filename).suffix.lower()
                if suffix not in SAFE_SUFFIXES:
                    filename = f"{Path(filename).stem or 'download'}.mp4"
                final_path = dest_dir / safe_filename(filename)
                part_path = final_path.with_suffix(final_path.suffix + ".part")
                with part_path.open("wb") as handle:
                    for chunk in response.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            handle.write(chunk)
                part_path.replace(final_path)
        except requests.RequestException as exc:
            logger.exception("Remote direct download failed: %s", source, exc_info=True)
            raise RemoteImportError(f"Download failed: {exc}") from exc
        return ImportedMedia(local_path=final_path, source_provider="direct", original_url=source, title=final_path.name, metadata={})

    def _import_google_drive(self, source: str) -> ImportedMedia:
        dest_dir = self._provider_dir("google_drive")
        output = dest_dir / "google-drive-download"
        try:
            result = _gdown().download(url=source, output=str(output), quiet=False, fuzzy=True)
        except (OSError, RuntimeError) as exc:
            raise RemoteImportError("Google Drive download failed; check that the file is public") from exc
        if not result:
            raise RemoteImportError("Google Drive download failed; check that the file is public")
        local_path = Path(result)
        return ImportedMedia(local_path=local_path, source_provider="google_drive", original_url=source, title=local_path.name, metadata={})

    def _import_mega(self, source: str) -> ImportedMedia:
        dest_dir = self._provider_dir("mega")
        try:
            mega_factory = _Mega()
            mega = mega_factory() if isinstance(mega_factory, type) else mega_factory
            result = mega.login().download_url(source, dest_path=str(dest_dir))
        except (OSError, RuntimeError) as exc:
            raise RemoteImportError("MEGA download failed; check that the public link is accessible") from exc
        local_path = Path(result)
        return ImportedMedia(local_path=local_path, source_provider="mega", original_url=source, title=local_path.name, metadata={})

    def _import_ytdlp(self, provider: str, source: str) -> ImportedMedia:
        _reject_private_network_url(source)
        dest_dir = self._provider_dir(provider)
        outtmpl = str(dest_dir / "%(title).80s.%(ext)s")
        opts = {
            "outtmpl": outtmpl,
            "format": "bv*+ba/best",
            "merge_output_format": "mp4",
            "noplaylist": True,
            "quiet": False,
            "no_warnings": False,
        }
        if self.cookies_file:
            opts["cookiefile"] = str(self.cookies_file)
        try:
            with _YoutubeDL()(opts) as ydl:
                info = ydl.extract_info(source, download=True)
        except (OSError, RuntimeError) as exc:
            raise RemoteImportError(f"{provider} download failed: {exc}") from exc
        local_path = self._path_from_ytdlp_info(dest_dir, info)
        return ImportedMedia(
            local_path=local_path,
            source_provider=provider,
            original_url=source,
            title=str(info.get("title", local_path.stem)),
            metadata={"extractor": str(info.get("extractor", provider))},
        )

    @staticmethod
    def _path_from_ytdlp_info(dest_dir: Path, info: dict) -> Path:
        for item in info.get("requested_downloads", []):
            filepath = item.get("filepath")
            if filepath and Path(filepath).exists():
                return Path(filepath)
        candidates = [p for p in dest_dir.iterdir() if p.is_file() and not p.name.endswith(".part")]
        if not candidates:
            raise RemoteImportError("Download finished but no media file was produced")
        return max(candidates, key=lambda path: path.stat().st_mtime)


def _list_providers() -> None:
    print("\n".join(SUPPORTED_PROVIDERS))


def main() -> int:
    parser = argparse.ArgumentParser(description="pyVideoTrans remote media importer")
    parser.add_argument("--list-providers", action="store_true")
    args = parser.parse_args()
    if args.list_providers:
        _list_providers()
        return 0
    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
