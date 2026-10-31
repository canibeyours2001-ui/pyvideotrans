"""CPU-only static and argument-level validation; no GPU or API secrets."""
import ast
import importlib.util
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class KaggleDubbingTests(unittest.TestCase):
    def test_native_provider_is_syntax_valid_and_registered(self):
        provider = (ROOT / "videotrans/tts/_voxcpm2kaggle.py").read_text(encoding="utf-8")
        config = (ROOT / "videotrans/tts/_constants.py").read_text(encoding="utf-8")
        ast.parse(provider)
        ast.parse(config)
        self.assertIn("VOXCPM2_KAGGLE_TTS = 38", config)
        self.assertIn('EDGE_TTS = 0', config)
        self.assertIn('imp="._voxcpm2kaggle"', config)
        self.assertIn('snapshot_download(repo_id="openbmb/VoxCPM2"', provider)
        self.assertIn("reference_wav_path", provider)
        self.assertIn('device="cuda"', provider)
        self.assertIn("self._model.generate(**kwargs)", provider)
        self.assertIn("subtype=\"PCM_16\"", provider)
        self.assertNotIn("ThreadPoolExecutor", provider)

    def test_kaggle_wrapper_providers_and_arguments(self):
        path = ROOT / "tools/kaggle_video_dub_job.py"
        ast.parse(path.read_text(encoding="utf-8"))
        spec = importlib.util.spec_from_file_location("kaggle_video_dub_job", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.assertEqual(module.PROVIDERS, {"edge": 0, "voxcpm2": 38})
        with tempfile.TemporaryDirectory() as temp:
            file = Path(temp) / "test.srt"
            file.write_text("1\n00:00:00,000 --> 00:00:01,000\nHello world\n", encoding="utf-8")
            edge = module.parser().parse_args([
                "--task", "tts", "--input", str(file),
                "--tts", "edge", "--voice-role", "en-US-AriaNeural",
                "--target-language-code", "en",
            ])
            command = module.build_command(edge, ROOT)
            self.assertIn("0", command)
            self.assertIn("en-US-AriaNeural", command)
            vox = module.parser().parse_args([
                "--task", "tts", "--input", str(file),
                "--tts", "voxcpm2", "--voice-role", "Narrator",
                "--target-language-code", "en", "--cuda",
            ])
            self.assertIn("38", module.build_command(vox, ROOT))
            vox.cuda = False
            with self.assertRaisesRegex(ValueError, "--cuda"):
                module.build_command(vox, ROOT)


if __name__ == "__main__":
    unittest.main()
