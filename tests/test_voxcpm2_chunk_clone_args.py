def test_provider_builds_same_selected_reference_for_every_chunk(tmp_path):
    from videotrans.tts.voxcpm2_voice import VoxCPM2VoiceStore, build_voxcpm2_generate_kwargs

    wav = tmp_path / "voice.wav"
    wav.write_bytes(b"voice")
    store = VoxCPM2VoiceStore(root_dir=tmp_path)
    store.save_audition("news_presenter", wav, "saved exact script")
    selected = store.select_latest_audition("news_presenter")

    kwargs1 = build_voxcpm2_generate_kwargs("chunk one", selected)
    kwargs2 = build_voxcpm2_generate_kwargs("chunk two", selected)

    assert kwargs1["reference_wav_path"] == kwargs2["reference_wav_path"] == str(selected.wav_path)
    assert kwargs1["prompt_wav_path"] == kwargs2["prompt_wav_path"] == str(selected.wav_path)
    assert kwargs1["prompt_text"] == kwargs2["prompt_text"] == "saved exact script"
    assert kwargs1["text"] == "chunk one"
    assert kwargs2["text"] == "chunk two"
