from pathlib import Path


def test_voxcpm2_provider_registered_as_builtin_without_api_url():
    from videotrans import tts

    provider = tts.ID_NAME_DICT[tts.VOXCPM_TTS]

    assert "VoxCPM2" in provider.name
    assert "Built-in" in provider.name
    assert provider.imp == "._voxcpm2"
    assert provider.key_name is None
    assert tts.is_input_api(tts.VOXCPM_TTS, return_str=True) is True


def test_voxcpm2_role_menu_returns_profiles():
    from videotrans import tts
    from videotrans.util.help_role import role_menu

    roles = role_menu(tts.VOXCPM_TTS, langcode="my")

    assert "🎬 Movie Recap — Fast Pace" in roles
    assert "📰 Professional Burmese News Presenter" in roles
    assert len(roles) == 11


def test_fake_voxcpm_worker_writes_output_with_selected_voice_args(tmp_path):
    from videotrans.tts.voxcpm2_voice import SelectedVoice
    from videotrans.tts._voxcpm2 import BuiltinVoxCPM2Worker

    captured = {}
    selected = SelectedVoice(
        profile_id="u_aung",
        profile_name="👨 U Aung — Deep Calm Male",
        wav_path=tmp_path / "selected.wav",
        test_script="exact prompt",
        timestamp="2026-09-28T00:00:00+00:00",
        model_id="openbmb/VoxCPM2",
        model_version="2.0.3",
    )
    selected.wav_path.write_bytes(b"voice")

    class FakeModel:
        def _generate(self, **kwargs):
            if "seed" in kwargs:
                raise AssertionError("seed must not be passed")
            captured.update(kwargs)
            out = tmp_path / "generated.wav"
            out.write_bytes(b"generated")
            return str(out)

    worker = BuiltinVoxCPM2Worker(worker_id=0, device="cpu", model_factory=lambda device: FakeModel())
    out_path = tmp_path / "chunk.wav"

    worker.generate_to_file("hello", out_path, selected)

    assert out_path.read_bytes() == b"generated"
    assert captured["text"] == "hello"
    assert captured["reference_wav_path"] == str(selected.wav_path)
    assert captured["prompt_wav_path"] == str(selected.wav_path)
    assert captured["prompt_text"] == "exact prompt"


def test_voxcpm_worker_writes_waveform_return_to_wav(tmp_path):
    import numpy as np
    import soundfile as sf

    from videotrans.tts._voxcpm2 import BuiltinVoxCPM2Worker

    class FakeModel:
        class TtsModel:
            sample_rate = 24000

        tts_model = TtsModel()

        def generate(self, **kwargs):
            return np.zeros(240, dtype=np.float32)

    worker = BuiltinVoxCPM2Worker(worker_id=0, device="cpu", model_factory=lambda device: FakeModel())
    out_path = tmp_path / "audition.wav"

    worker.generate_profile_audition(
        profile=__import__("videotrans.tts.voxcpm2_profiles", fromlist=["get_profile"]).get_profile("u_aung"),
        text="မင်္ဂလာပါ",
        output_path=out_path,
    )

    audio, sample_rate = sf.read(out_path)
    assert sample_rate == 24000
    assert audio.shape[0] == 240


def test_voxcpm_worker_writes_generator_return_to_wav(tmp_path):
    import numpy as np
    import soundfile as sf

    from videotrans.tts._voxcpm2 import BuiltinVoxCPM2Worker

    class FakeModel:
        class TtsModel:
            sample_rate = 22050

        tts_model = TtsModel()

        def _generate(self, **kwargs):
            yield np.zeros(120, dtype=np.float32)

    worker = BuiltinVoxCPM2Worker(worker_id=0, device="cpu", model_factory=lambda device: FakeModel())
    out_path = tmp_path / "audition.wav"

    worker.generate_profile_audition(
        profile=__import__("videotrans.tts.voxcpm2_profiles", fromlist=["get_profile"]).get_profile("u_aung"),
        text="မင်္ဂလာပါ",
        output_path=out_path,
    )

    audio, sample_rate = sf.read(out_path)
    assert sample_rate == 22050
    assert audio.shape[0] == 120
