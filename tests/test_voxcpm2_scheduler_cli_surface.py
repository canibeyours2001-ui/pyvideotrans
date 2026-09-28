import subprocess
import sys


def test_voxcpm2_scheduler_cli_demo_prints_cuda_activity():
    result = subprocess.run(
        [sys.executable, "-m", "videotrans.tts.voxcpm2_scheduler", "--demo"],
        check=True,
        text=True,
        capture_output=True,
    )

    assert "[cuda:0] chunk 1 started" in result.stdout
    assert "[cuda:1] chunk 2 started" in result.stdout
    assert "ordered: 1,2,3,4,5" in result.stdout
