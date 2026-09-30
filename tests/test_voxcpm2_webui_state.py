from pathlib import Path


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
    selected = store.get_selected_voice("u_aung")
    assert selected is not None
    assert selected.test_script == "two"
    assert selected.wav_path == Path(audio)
    assert selected.wav_path.read_bytes() == b"two"


def test_repeated_webui_auditions_return_playable_latest_audio(tmp_path, monkeypatch):
    import webui
    from videotrans.tts import voxcpm2_engine
    from videotrans.tts.voxcpm2_voice import VoxCPM2VoiceStore

    store = VoxCPM2VoiceStore(root_dir=tmp_path / "voices")
    payloads = iter((b"first audition", b"latest audition"))

    def generate(_profile, _script, output_path):
        output_path.write_bytes(next(payloads))
        return output_path

    monkeypatch.setattr(webui, "_voxcpm2_voice_store", lambda: store)
    monkeypatch.setattr(voxcpm2_engine, "generate_voxcpm2_audition", generate)

    _first_status, first_audio = webui.test_voxcpm2_voice_for_webui("u_aung", "first")
    assert Path(first_audio).read_bytes() == b"first audition"

    latest_status, latest_audio = webui.test_voxcpm2_voice_for_webui("u_aung", "latest")

    assert "Generated new candidate" in latest_status
    assert Path(latest_audio).read_bytes() == b"latest audition"


def test_play_selected_voice_returns_persisted_audio(tmp_path, monkeypatch):
    import webui
    from videotrans.tts.voxcpm2_voice import VoxCPM2VoiceStore

    store = VoxCPM2VoiceStore(root_dir=tmp_path)
    candidate = tmp_path / "candidate.wav"
    candidate.write_bytes(b"selected voice")
    store.save_audition("u_aung", candidate, "selected script")
    selected = store.select_latest_audition("u_aung")
    monkeypatch.setattr(webui, "_voxcpm2_voice_store", lambda: store)

    status, audio = webui.selected_voxcpm2_voice_status("u_aung")

    assert "Voice selected for dubbing" in status
    assert audio == str(selected.wav_path)


def test_replace_voice_keeps_selection_until_latest_audition_is_used(tmp_path, monkeypatch):
    import webui
    from videotrans.tts.voxcpm2_voice import VoxCPM2VoiceStore

    store = VoxCPM2VoiceStore(root_dir=tmp_path)
    first = tmp_path / "first.wav"
    replacement = tmp_path / "replacement.wav"
    first.write_bytes(b"first")
    replacement.write_bytes(b"replacement")
    store.save_audition("u_aung", first, "first script")
    initially_selected = store.select_latest_audition("u_aung")
    monkeypatch.setattr(webui, "_voxcpm2_voice_store", lambda: store)

    replace_voice = getattr(webui, "replace_voxcpm2_voice")
    status, audio = replace_voice("u_aung")
    store.save_audition("u_aung", replacement, "replacement script")

    still_selected = store.get_selected_voice("u_aung")
    assert "Replacement mode" in status
    assert audio == str(initially_selected.wav_path)
    assert still_selected is not None
    assert still_selected.wav_path.read_bytes() == b"first"
