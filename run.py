#!/usr/bin/env python3
import argparse
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

if getattr(sys, "frozen", False):
    _mp_dll_dir = Path(sys._MEIPASS) / "mediapipe" / "tasks" / "c"
    if _mp_dll_dir.is_dir():
        try:
            os.add_dll_directory(str(_mp_dll_dir))
        except Exception:
            pass
    try:
        os.add_dll_directory(str(Path(sys._MEIPASS)))
    except Exception:
        pass

from src.logger import setup_logging
from ui.app import HeadTimerUI

log = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="MisterTimer — pin OBS widget to streamer's forehead"
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--no-browser", action="store_true")
    parser.add_argument("--public", action="store_true")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    setup_logging(debug=args.debug)

    if getattr(sys, "frozen", False):
        if sys.stdout is None:
            sys.stdout = open(os.devnull, "w")
        if sys.stderr is None:
            sys.stderr = open(os.devnull, "w")

    host = "0.0.0.0" if args.public else args.host
    log.info("MisterTimer starting on %s:%s", host, args.port)

    application = HeadTimerUI()
    start_tray(args.port, application)
    try:
        application.run(host=host, port=args.port, show=not args.no_browser)
    finally:
        shutdown(application)


def shutdown(application: HeadTimerUI):
    try:
        application.tracker.stop()
    finally:
        application.obs.disconnect()


def start_tray(port: int, application: HeadTimerUI):
    try:
        from ui.tray import setup_tray

        setup_tray(port=port, on_quit=lambda: shutdown(application))
        log.debug("System tray icon started")
    except Exception:
        log.debug("System tray not available", exc_info=True)


if __name__ == "__main__":
    main()
