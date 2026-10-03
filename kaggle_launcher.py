from __future__ import annotations

import argparse
import os
import signal
import sys
import time
from pathlib import Path
from typing import Optional


def _extract_urls(launch_result):
    """Return (local_url, share_url) from Gradio launch() across 5.x/6.x."""
    local_url = None
    share_url = None

    if isinstance(launch_result, tuple):
        # Gradio 6.x: (fastapi_app, local_url, share_url)
        if len(launch_result) >= 3:
            local_url = launch_result[-2]
            share_url = launch_result[-1]
        elif len(launch_result) == 2:
            local_url, share_url = launch_result
    else:
        # Defensive compatibility for any future named/object result.
        local_url = getattr(launch_result, "local_url", None)
        share_url = getattr(launch_result, "share_url", None)

    local_url = str(local_url or "").strip() or None
    share_url = str(share_url or "").strip() or None
    return local_url, share_url


def _write_atomic(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(value, encoding="utf-8")
    tmp.replace(path)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Kaggle-safe pyVideoTrans Gradio launcher")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share-url-file", default="/kaggle/working/pyvideotrans_share_url.txt")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args(argv)

    os.environ.setdefault("PYVIDEOTRANS_LANG", "en_US")
    os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")

    from webui import build_ui

    app = build_ui()
    if app is None:
        raise RuntimeError("webui.build_ui() returned None")

    if args.self_test:
        # Build the complete Gradio UI and test launch-result compatibility
        # without starting a network server/tunnel.
        a, b = _extract_urls((object(), "http://127.0.0.1:7860", "https://example.gradio.live"))
        assert a == "http://127.0.0.1:7860"
        assert b == "https://example.gradio.live"
        print("Kaggle launcher self-test passed", flush=True)
        return 0

    share_file = Path(args.share_url_file)
    share_file.unlink(missing_ok=True)

    print("Building Gradio server and requesting public tunnel…", flush=True)

    # Capture the URL directly from Gradio's return value. This is much more
    # reliable than scraping redirected stdout/stderr, which can be block-buffered
    # when a Python child process writes to a file.
    result = app.launch(
        server_name=args.host,
        server_port=args.port,
        share=True,
        inbrowser=False,
        prevent_thread_lock=True,
        show_error=True,
        quiet=False,
    )

    local_url, share_url = _extract_urls(result)

    print(f"PYVIDEOTRANS_LOCAL_URL={local_url or ''}", flush=True)
    print(f"PYVIDEOTRANS_SHARE_URL={share_url or ''}", flush=True)

    if not share_url:
        raise RuntimeError(
            "Gradio started but did not return a public share URL. "
            "Check the Gradio share-server/tunnel messages in this log."
        )

    _write_atomic(share_file, share_url)
    print(f"SHARE_URL_FILE={share_file}", flush=True)

    stop = False

    def _request_stop(_sig, _frame):
        nonlocal stop
        stop = True

    signal.signal(signal.SIGTERM, _request_stop)
    signal.signal(signal.SIGINT, _request_stop)

    try:
        # prevent_thread_lock=True returns immediately, so keep this process
        # alive to keep both the local Gradio server and FRP tunnel alive.
        while not stop:
            time.sleep(1)
    finally:
        try:
            app.close()
        except Exception:
            pass
        try:
            share_file.unlink(missing_ok=True)
        except Exception:
            pass

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
