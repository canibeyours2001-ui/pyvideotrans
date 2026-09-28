from pathlib import Path


def test_remote_and_voxcpm_dependencies_are_declared():
    pyproject = Path("pyproject.toml").read_text(encoding="utf-8")

    assert '"yt-dlp' in pyproject
    assert '"gdown' in pyproject
    assert '"mega.py' in pyproject
    assert '"voxcpm==2.0.3"' in pyproject


def test_generated_media_and_cookie_paths_are_gitignored():
    gitignore = Path(".gitignore").read_text(encoding="utf-8")

    assert "storage/" in gitignore
    assert "*.part" in gitignore
    assert "cookies.txt" in gitignore
    assert "*.cookies.txt" in gitignore
