from pathlib import Path

import pytest


def test_local_provider_returns_existing_path(tmp_path):
    from videotrans.remote_importer import RemoteMediaImporter

    media = tmp_path / "input.mp4"
    media.write_bytes(b"fake")

    result = RemoteMediaImporter(storage_dir=tmp_path / "imports").import_media("local", str(media))

    assert result.local_path == media
    assert result.source_provider == "local"
    assert result.original_url == str(media)


def test_direct_provider_rejects_unsafe_schemes(tmp_path):
    from videotrans.remote_importer import RemoteMediaImporter, RemoteImportError

    importer = RemoteMediaImporter(storage_dir=tmp_path / "imports")

    with pytest.raises(RemoteImportError, match="Unsupported URL scheme"):
        importer.import_media("direct", "file:///etc/passwd")

    assert not any((tmp_path / "imports").glob("**/*"))


def test_direct_provider_streams_part_then_atomic_completion(tmp_path, monkeypatch):
    from videotrans.remote_importer import RemoteMediaImporter

    class FakeResponse:
        headers = {"content-disposition": 'attachment; filename="clip.mp4"'}

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def raise_for_status(self):
            return None

        def iter_content(self, chunk_size=1024 * 1024):
            yield b"abc"
            yield b"def"

    calls = []

    def fake_get(url, stream, timeout, allow_redirects):
        calls.append((url, stream, timeout, allow_redirects))
        return FakeResponse()

    monkeypatch.setattr("videotrans.remote_importer.requests.get", fake_get)
    monkeypatch.setattr("videotrans.remote_importer._reject_private_network_url", lambda url: None)

    result = RemoteMediaImporter(storage_dir=tmp_path / "imports").import_media("direct", "https://example.com/video")

    assert result.local_path.read_bytes() == b"abcdef"
    assert result.local_path.name.endswith("clip.mp4")
    assert not result.local_path.with_suffix(result.local_path.suffix + ".part").exists()
    assert calls == [("https://example.com/video", True, (10, 60), True)]


def test_platform_routes_use_ytdlp_without_shell(tmp_path, monkeypatch):
    from videotrans.remote_importer import RemoteMediaImporter

    captured = {}

    class FakeYDL:
        def __init__(self, opts):
            captured["opts"] = opts

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def extract_info(self, url, download):
            captured["url"] = url
            captured["download"] = download
            output = Path(captured["opts"]["outtmpl"].replace("%(title).80s", "sample").replace("%(ext)s", "mp4"))
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_bytes(b"video")
            return {"title": "sample", "ext": "mp4", "requested_downloads": [{"filepath": str(output)}]}

    monkeypatch.setattr("videotrans.remote_importer._YoutubeDL", lambda: FakeYDL)

    result = RemoteMediaImporter(storage_dir=tmp_path / "imports").import_media("youtube", "https://youtube.com/watch?v=x")

    assert result.local_path.exists()
    assert result.source_provider == "youtube"
    assert captured["download"] is True
    assert captured["url"] == "https://youtube.com/watch?v=x"
    assert "cookiesfrombrowser" not in captured["opts"]
