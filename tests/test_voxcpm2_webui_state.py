def test_use_voice_for_webui_selects_latest_candidate(tmp_path, monkeypatch):
    import webui
    from videotrans.tts.voxcpm2_voice import VoxCPM2VoiceStore

    store = VoxCPM2VoiceStore(root_dir=tmp_path)
    wav1 = tmp_path / "one.wav"
    wav2 = tmp_path / "two.wav"
    wav1.write_bytes(b"one")
    wav2.write_bytes(b"two")
    store.save_audition("u_aung", wav1, "one")
    store.save_audition("u_aung", wav2, "two")

    monkeypatch.setattr(webui, "_voxcpm2_voice_store", lambda: store)

    status, audio = webui.use_voxcpm2_voice_for_webui("👨 U Aung — Deep Calm Male")

    assert "Voice selected for dubbing" in status
    assert audio.endswith(".wav")
    assert store.get_selected_voice("u_aung").test_script == "two"
