import pytest


def test_direct_provider_rejects_localhost_before_request(tmp_path, monkeypatch):
    from videotrans.remote_importer import RemoteImportError, RemoteMediaImporter

    def should_not_get(*args, **kwargs):
        raise AssertionError("request must not be issued")

    monkeypatch.setattr("videotrans.remote_importer.requests.get", should_not_get)

    with pytest.raises(RemoteImportError, match="private or local network"):
        RemoteMediaImporter(storage_dir=tmp_path).import_media("direct", "https://127.0.0.1/video.mp4")
