import subprocess
import sys


def test_voxcpm2_voice_cli_lists_profiles():
    result = subprocess.run(
        [sys.executable, "-m", "videotrans.tts.voxcpm2_voice", "--list-profiles"],
        check=True,
        text=True,
        capture_output=True,
    )

    assert "movie_recap_fast" in result.stdout
    assert "Professional Burmese News Presenter" in result.stdout
