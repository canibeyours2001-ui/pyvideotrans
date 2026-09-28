def test_google_drive_uses_gdown_public_url(tmp_path, monkeypatch):
    from videotrans.remote_importer import RemoteMediaImporter

    captured = {}

    class FakeGdown:
        @staticmethod
        def download(url, output, quiet, fuzzy):
            captured.update({"url": url, "output": output, "quiet": quiet, "fuzzy": fuzzy})
            pathlib_output = __import__("pathlib").Path(output)
            pathlib_output.write_bytes(b"drive")
            return output

    monkeypatch.setattr("videotrans.remote_importer._gdown", lambda: FakeGdown)

    result = RemoteMediaImporter(storage_dir=tmp_path).import_media("google_drive", "https://drive.google.com/file/d/abc/view")

    assert result.local_path.read_bytes() == b"drive"
    assert captured["fuzzy"] is True


def test_mega_uses_public_link_without_credentials(tmp_path, monkeypatch):
    from videotrans.remote_importer import RemoteMediaImporter

    captured = {}

    class FakeMegaClient:
        def download_url(self, url, dest_path):
            captured.update({"url": url, "dest_path": dest_path})
            output = __import__("pathlib").Path(dest_path) / "mega.mp4"
            output.write_bytes(b"mega")
            return str(output)

    class FakeMega:
        def login(self):
            return FakeMegaClient()

    monkeypatch.setattr("videotrans.remote_importer._Mega", lambda: FakeMega)

    result = RemoteMediaImporter(storage_dir=tmp_path).import_media("mega", "https://mega.nz/file/example")

    assert result.local_path.read_bytes() == b"mega"
    assert captured["url"] == "https://mega.nz/file/example"
