import subprocess
import sys


def test_webui_build_ui_import_surface():
    result = subprocess.run(
        [sys.executable, "-c", "import webui; app=webui.build_ui(); print(app.__class__.__name__); print(webui.CLI_LANG)"],
        check=True,
        text=True,
        capture_output=True,
        timeout=30,
    )

    assert "Blocks" in result.stdout
    assert "en" in result.stdout
