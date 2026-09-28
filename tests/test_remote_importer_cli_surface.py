import subprocess
import sys


def test_remote_importer_cli_lists_supported_providers():
    result = subprocess.run(
        [sys.executable, "-m", "videotrans.remote_importer", "--list-providers"],
        check=True,
        text=True,
        capture_output=True,
    )

    for provider in ["local", "direct", "google_drive", "mega", "youtube", "tiktok", "rednote", "douyin", "bilibili"]:
        assert provider in result.stdout
