def test_device_detection_uses_all_cuda_devices(monkeypatch):
    from videotrans.tts._voxcpm2 import detect_voxcpm2_devices

    class FakeCuda:
        @staticmethod
        def is_available():
            return True

        @staticmethod
        def device_count():
            return 2

    class FakeTorch:
        cuda = FakeCuda()

    monkeypatch.setattr("videotrans.tts._voxcpm2._torch", lambda: FakeTorch)

    assert detect_voxcpm2_devices() == ["cuda:0", "cuda:1"]


def test_device_detection_cpu_fallback(monkeypatch):
    from videotrans.tts._voxcpm2 import detect_voxcpm2_devices

    class FakeCuda:
        @staticmethod
        def is_available():
            return False

        @staticmethod
        def device_count():
            return 0

    class FakeTorch:
        cuda = FakeCuda()

    monkeypatch.setattr("videotrans.tts._voxcpm2._torch", lambda: FakeTorch)

    assert detect_voxcpm2_devices() == ["cpu"]
