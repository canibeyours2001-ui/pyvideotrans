import pytest


def test_unsupported_provider_has_clear_error(tmp_path):
    from videotrans.remote_importer import RemoteImportError, RemoteMediaImporter

    with pytest.raises(RemoteImportError, match="Unsupported media source"):
        RemoteMediaImporter(storage_dir=tmp_path).import_media("unknown", "https://example.com/x.mp4")


def test_optional_cookie_file_is_passed_to_ytdlp(tmp_path, monkeypatch):
    from videotrans.remote_importer import RemoteMediaImporter

    cookies = tmp_path / "cookies.txt"
    cookies.write_text("# Netscape HTTP Cookie File\n", encoding="utf-8")
    captured = {}

    class FakeYDL:
        def __init__(self, opts):
            captured.update(opts)

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def extract_info(self, url, download):
            output = tmp_path / "imports" / "yt" / "sample.mp4"
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_bytes(b"video")
            return {"title": "sample", "requested_downloads": [{"filepath": str(output)}]}

    monkeypatch.setattr("videotrans.remote_importer._YoutubeDL", lambda: FakeYDL)

    RemoteMediaImporter(storage_dir=tmp_path / "imports", cookies_file=cookies).import_media("tiktok", "https://tiktok.com/@x/video/1")

    assert captured["cookiefile"] == str(cookies)
