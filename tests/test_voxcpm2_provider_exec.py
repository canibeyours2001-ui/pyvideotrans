def test_provider_exec_schedules_queue_with_selected_voice(tmp_path, monkeypatch):
    from videotrans.tts import VOXCPM_TTS
    from videotrans.tts._voxcpm2 import VoxCPM2BuiltinTTS
    from videotrans.tts.voxcpm2_voice import VoxCPM2VoiceStore

    voice_root = tmp_path / "voices"
    wav = tmp_path / "voice.wav"
    wav.write_bytes(b"voice")
    store = VoxCPM2VoiceStore(root_dir=voice_root)
    store.save_audition("u_aung", wav, "prompt")
    store.select_latest_audition("u_aung")

    monkeypatch.setattr("videotrans.tts._voxcpm2.default_voice_store", lambda: store)
    monkeypatch.setattr("videotrans.tts._voxcpm2.detect_voxcpm2_devices", lambda max_workers=None: ["cpu"])

    class FakeWorker:
        def __init__(self, worker_id, device):
            self.worker_id = worker_id
            self.device = device

        def generate(self, job):
            job.output_path.write_bytes((job.text + ":" + job.selected_voice.test_script).encode())
            return str(job.output_path)

    monkeypatch.setattr("videotrans.tts._voxcpm2.BuiltinVoxCPM2Worker", FakeWorker)

    out1 = tmp_path / "1.wav"
    out2 = tmp_path / "2.wav"
    provider = VoxCPM2BuiltinTTS(
        queue_tts=[
            {"text": "hello", "filename": str(out1), "role": "👨 U Aung — Deep Calm Male"},
            {"text": "world", "filename": str(out2), "role": "u_aung"},
        ],
        tts_type=VOXCPM_TTS,
        uuid="test",
    )

    provider._exec()

    assert out1.read_text() == "hello:prompt"
    assert out2.read_text() == "world:prompt"
