def test_worker_loads_model_once_for_multiple_chunks(tmp_path):
    from videotrans.tts._voxcpm2 import BuiltinVoxCPM2Worker
    from videotrans.tts.voxcpm2_voice import SelectedVoice

    calls = {"loads": 0, "generates": 0}

    class FakeModel:
        def _generate(self, **kwargs):
            calls["generates"] += 1
            out = tmp_path / f"out-{calls['generates']}.wav"
            out.write_bytes(b"audio")
            return str(out)

    def factory(device):
        calls["loads"] += 1
        return FakeModel()

    voice = SelectedVoice("u_aung", "U Aung", tmp_path / "voice.wav", "prompt", "now", "openbmb/VoxCPM2", "2.0.3")
    voice.wav_path.write_bytes(b"voice")
    worker = BuiltinVoxCPM2Worker(0, "cpu", model_factory=factory)

    worker.generate_to_file("one", tmp_path / "one.wav", voice)
    worker.generate_to_file("two", tmp_path / "two.wav", voice)

    assert calls == {"loads": 1, "generates": 2}
