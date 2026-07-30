import os
import sys
import webbrowser
from collections.abc import Callable
from pathlib import Path

import pystray
from PIL import Image, ImageDraw


def setup_tray(port: int = 8080, on_quit: Callable[[], None] | None = None):
    if not getattr(sys, "frozen", False):
        return

    icon = _load_icon()

    def open_browser(_icon=None, _item=None):
        webbrowser.open(f"http://localhost:{port}")

    def quit_application(icon_obj=None, _item=None):
        if on_quit is not None:
            on_quit()
        if icon_obj is not None:
            icon_obj.stop()
        os._exit(0)

    menu = pystray.Menu(
        pystray.MenuItem("Open", open_browser, default=True),
        pystray.MenuItem("Quit", quit_application),
    )
    tray = pystray.Icon("MisterTimer", icon, "MisterTimer", menu)
    tray.run_detached()


def _load_icon():
    paths = [
        Path(sys._MEIPASS) / "icon.png",
        Path(sys._MEIPASS) / "icon.ico",
        Path("icon.png"),
        Path("icon.ico"),
    ]
    for path in paths:
        if path.exists():
            return Image.open(path)

    image = Image.new("RGBA", (64, 64), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    draw.ellipse([8, 8, 56, 56], fill=(0, 200, 100))
    draw.ellipse([20, 20, 44, 44], fill=(255, 255, 255, 100))
    return image
