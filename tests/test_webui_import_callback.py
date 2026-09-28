def test_import_media_for_webui_sets_video_path(tmp_path, monkeypatch):
    import webui
    from videotrans.remote_importer import ImportedMedia

    media = tmp_path / "clip.mp4"
    media.write_bytes(b"clip")

    class FakeImporter:
        def __init__(self, cookies_file=None):
            self.cookies_file = cookies_file

        def import_media(self, provider, source):
            assert provider == "direct"
            assert source == "https://example.com/clip.mp4"
            return ImportedMedia(local_path=media, source_provider=provider, original_url=source, title="clip", metadata={})

    monkeypatch.setattr(webui, "RemoteMediaImporter", FakeImporter)

    video_value, status = webui.import_media_for_webui("Direct Download URL", "https://example.com/clip.mp4", "")

    assert video_value == str(media)
    assert "Ready" in status
