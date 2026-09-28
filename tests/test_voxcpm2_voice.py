from pathlib import Path


def _wav(path: Path, payload: bytes = b"RIFF....WAVEfmt ") -> Path:
    path.write_bytes(payload)
    return path


def test_profiles_include_all_required_burmese_voice_characters():
    from videotrans.tts.voxcpm2_profiles import VOXCPM2_PROFILES

    assert len(VOXCPM2_PROFILES) == 11
    assert {p.profile_id for p in VOXCPM2_PROFILES} == {
        "u_aung",
        "ko_min",
        "u_htet",
        "ko_zaw",
        "daw_may",
        "ma_su",
        "daw_thida",
        "ma_nandar",
        "news_presenter",
        "documentary_narrator",
        "movie_recap_fast",
    }
    assert all(p.instruction.strip() for p in VOXCPM2_PROFILES)


def test_latest_audition_can_be_selected_and_reloaded(tmp_path):
    from videotrans.tts.voxcpm2_voice import VoxCPM2VoiceStore

    store = VoxCPM2VoiceStore(root_dir=tmp_path)
    audition = store.save_audition(
        profile_id="movie_recap_fast",
        generated_wav=_wav(tmp_path / "candidate.wav", b"candidate-audio"),
        test_script="မင်္ဂလာပါ",
        model_id="openbmb/VoxCPM2",
        model_version="2.0.3",
    )

    selected = store.select_latest_audition("movie_recap_fast")
    reloaded = VoxCPM2VoiceStore(root_dir=tmp_path).get_selected_voice("movie_recap_fast")

    assert audition.wav_path.exists()
    assert selected.wav_path.read_bytes() == b"candidate-audio"
    assert reloaded is not None
    assert reloaded.profile_id == "movie_recap_fast"
    assert reloaded.test_script == "မင်္ဂလာပါ"
    assert reloaded.model_id == "openbmb/VoxCPM2"
    assert reloaded.model_version == "2.0.3"


def test_replace_mode_keeps_existing_selected_voice_until_new_selection(tmp_path):
    from videotrans.tts.voxcpm2_voice import VoxCPM2VoiceStore

    store = VoxCPM2VoiceStore(root_dir=tmp_path)
    store.save_audition("u_aung", _wav(tmp_path / "first.wav", b"first"), "first script")
    first = store.select_latest_audition("u_aung")

    store.start_replace("u_aung")
    store.save_audition("u_aung", _wav(tmp_path / "second.wav", b"second"), "second script")

    still_selected = store.get_selected_voice("u_aung")
    assert still_selected is not None
    assert still_selected.wav_path == first.wav_path
    assert still_selected.wav_path.read_bytes() == b"first"


def test_selected_voice_builds_clone_arguments_without_seed(tmp_path):
    from videotrans.tts.voxcpm2_voice import VoxCPM2VoiceStore, build_voxcpm2_generate_kwargs

    store = VoxCPM2VoiceStore(root_dir=tmp_path)
    store.save_audition("ko_min", _wav(tmp_path / "voice.wav", b"voice"), "exact transcript")
    selected = store.select_latest_audition("ko_min")

    kwargs = build_voxcpm2_generate_kwargs("dub text", selected)

    assert kwargs["text"] == "dub text"
    assert kwargs["reference_wav_path"] == str(selected.wav_path)
    assert kwargs["prompt_wav_path"] == str(selected.wav_path)
    assert kwargs["prompt_text"] == "exact transcript"
    assert kwargs["cfg_value"] == 2.0
    assert kwargs["inference_timesteps"] == 10
    assert kwargs["max_len"] == 4096
    assert kwargs["retry_badcase"] is True
    assert kwargs["retry_badcase_max_times"] == 3
    assert "seed" not in kwargs
