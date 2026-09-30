from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf


def _write_wav(path):
    sf.write(str(path), np.zeros(240, dtype=np.float32), 24000)


def test_default_model_factory_disables_denoiser(monkeypatch):
    from videotrans.tts import voxcpm2_engine

    calls = []

    class FakeVoxCPM:
        @staticmethod
        def from_pretrained(model_id, **kwargs):
            calls.append((model_id, kwargs))
            return SimpleNamespace()

    monkeypatch.setattr(
        voxcpm2_engine.importlib,
        "import_module",
        lambda module_name: SimpleNamespace(VoxCPM=FakeVoxCPM),
    )
    worker = voxcpm2_engine.BuiltinVoxCPM2Worker(0, "cpu")

    worker._default_model_factory("cpu")

    assert calls == [
        ("openbmb/VoxCPM2", {"device": "cpu", "load_denoiser": False})
    ], "VoxCPM.from_pretrained must receive load_denoiser=False"


def test_worker_uses_public_generate_when_private_generate_exists(tmp_path):
    from videotrans.tts._voxcpm2 import BuiltinVoxCPM2Worker
    from videotrans.tts.voxcpm2_voice import SelectedVoice

    calls = []

    class FakeModel:
        def generate(self, **kwargs):
            calls.append(kwargs)
            out = tmp_path / "generated.wav"
            _write_wav(out)
            return str(out)

        def _generate(self, **kwargs):
            raise AssertionError("private _generate must never be called")

    voice = SelectedVoice("u_aung", "U Aung", tmp_path / "voice.wav", "prompt", "now", "model", "version")
    _write_wav(voice.wav_path)
    worker = BuiltinVoxCPM2Worker(0, "cpu", model_factory=lambda device: FakeModel())

    worker.generate_to_file("hello", tmp_path / "output.wav", voice)

    assert len(calls) == 1, "public model.generate must be called exactly once"


def test_worker_rejects_model_without_public_generate(tmp_path):
    from videotrans.tts._voxcpm2 import BuiltinVoxCPM2Worker
    from videotrans.tts.voxcpm2_voice import SelectedVoice

    class FakeModel:
        def _generate(self, **kwargs):
            raise AssertionError("private _generate must never be called")

    voice = SelectedVoice("u_aung", "U Aung", tmp_path / "voice.wav", "prompt", "now", "model", "version")
    _write_wav(voice.wav_path)
    worker = BuiltinVoxCPM2Worker(0, "cpu", model_factory=lambda device: FakeModel())

    with pytest.raises(AttributeError, match="generate"):
        worker.generate_to_file("hello", tmp_path / "output.wav", voice)


def test_clone_generation_disables_denoising(tmp_path):
    from videotrans.tts._voxcpm2 import BuiltinVoxCPM2Worker
    from videotrans.tts.voxcpm2_voice import SelectedVoice

    calls = []

    class FakeModel:
        def generate(self, **kwargs):
            calls.append(kwargs)
            out = tmp_path / "generated.wav"
            _write_wav(out)
            return str(out)

    voice = SelectedVoice("u_aung", "U Aung", tmp_path / "voice.wav", "prompt", "now", "model", "version")
    _write_wav(voice.wav_path)
    worker = BuiltinVoxCPM2Worker(0, "cpu", model_factory=lambda device: FakeModel())

    worker.generate_to_file("hello", tmp_path / "output.wav", voice)

    assert calls[0]["denoise"] is False, "clone generation must pass denoise=False"


def test_profile_audition_disables_denoising(tmp_path):
    from videotrans.tts._voxcpm2 import BuiltinVoxCPM2Worker
    from videotrans.tts.voxcpm2_profiles import VOXCPM2_PROFILES

    calls = []

    class FakeModel:
        def generate(self, **kwargs):
            calls.append(kwargs)
            out = tmp_path / "generated.wav"
            _write_wav(out)
            return str(out)

    worker = BuiltinVoxCPM2Worker(0, "cpu", model_factory=lambda device: FakeModel())

    worker.generate_profile_audition(VOXCPM2_PROFILES[0], "hello", tmp_path / "audition.wav")

    assert calls[0]["denoise"] is False, "audition generation must pass denoise=False"


def test_auditions_lazily_reuse_one_worker(tmp_path, monkeypatch):
    from videotrans.tts import voxcpm2_engine
    from videotrans.tts.voxcpm2_profiles import VOXCPM2_PROFILES

    created_devices = []

    class FakeWorker:
        def __init__(self, worker_id, device):
            created_devices.append(device)

        def generate_profile_audition(self, profile, text, output_path):
            output_path.write_bytes(text.encode())
            return output_path

    monkeypatch.setattr(voxcpm2_engine, "detect_voxcpm2_devices", lambda max_workers=None: ["cpu"])
    monkeypatch.setattr(voxcpm2_engine, "BuiltinVoxCPM2Worker", FakeWorker)
    profile = VOXCPM2_PROFILES[0]

    assert created_devices == [], "audition worker must be created lazily"
    voxcpm2_engine.generate_voxcpm2_audition(profile, "first", tmp_path / "first.wav")
    voxcpm2_engine.generate_voxcpm2_audition(profile, "second", tmp_path / "second.wav")

    assert created_devices == ["cpu"], "repeated auditions must reuse one persistent worker"


def test_worker_loads_model_once_for_multiple_chunks(tmp_path):
    from videotrans.tts._voxcpm2 import BuiltinVoxCPM2Worker
    from videotrans.tts.voxcpm2_voice import SelectedVoice

    calls = {"loads": 0, "generates": 0}

    class FakeModel:
        def generate(self, **kwargs):
            calls["generates"] += 1
            out = tmp_path / f"out-{calls['generates']}.wav"
            _write_wav(out)
            return str(out)

    def factory(device):
        calls["loads"] += 1
        return FakeModel()

    voice = SelectedVoice("u_aung", "U Aung", tmp_path / "voice.wav", "prompt", "now", "openbmb/VoxCPM2", "2.0.3")
    _write_wav(voice.wav_path)
    worker = BuiltinVoxCPM2Worker(0, "cpu", model_factory=factory)

    worker.generate_to_file("one", tmp_path / "one.wav", voice)
    worker.generate_to_file("two", tmp_path / "two.wav", voice)

    assert calls == {"loads": 1, "generates": 2}


def test_worker_rejects_missing_path_returned_by_generate(tmp_path):
    from videotrans.tts._voxcpm2 import BuiltinVoxCPM2Worker
    from videotrans.tts.voxcpm2_voice import SelectedVoice

    voice_path = tmp_path / "voice.wav"
    sf.write(str(voice_path), np.zeros(240, dtype=np.float32), 24000)

    class FakeModel:
        def generate(self, **kwargs):
            return tmp_path / "missing.wav"

    voice = SelectedVoice("u_aung", "U Aung", voice_path, "prompt", "now", "model", "version")
    worker = BuiltinVoxCPM2Worker(0, "cpu", model_factory=lambda device: FakeModel())

    with pytest.raises(RuntimeError, match="valid WAV"):
        worker.generate_to_file("hello", tmp_path / "output.wav", voice)


def test_worker_rejects_non_wav_path_returned_by_generate(tmp_path):
    from videotrans.tts._voxcpm2 import BuiltinVoxCPM2Worker
    from videotrans.tts.voxcpm2_voice import SelectedVoice

    voice_path = tmp_path / "voice.wav"
    sf.write(str(voice_path), np.zeros(240, dtype=np.float32), 24000)
    returned_path = tmp_path / "returned.wav"
    returned_path.write_bytes(b"not a wav")

    class FakeModel:
        def generate(self, **kwargs):
            return returned_path

    voice = SelectedVoice("u_aung", "U Aung", voice_path, "prompt", "now", "model", "version")
    worker = BuiltinVoxCPM2Worker(0, "cpu", model_factory=lambda device: FakeModel())

    with pytest.raises(RuntimeError, match="valid WAV"):
        worker.generate_to_file("hello", tmp_path / "output.wav", voice)
