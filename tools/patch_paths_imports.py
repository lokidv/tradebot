# -*- coding: utf-8 -*-
"""یک‌بار: «import paths» را از میانِ کتابخانه‌های استاندارد به گروهِ ماژول‌های خودِ ربات ببر."""
import os
import sys

BOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "bot")

# فایل → (خطی که paths پس از آن می‌آید، آیا پیش از آن یک خطِ خالی لازم است)
PLACE = {
    "autotrader.py": ("import paper", False),
    "fill_quality.py": ("from typing import Any", True),
    "market.py": ("import httpx", True),
    "log.py": ("import traceback", True),
    "research.py": ("import gates", False),
    "gates.py": ("import time", True),
    "main.py": ("import paper", False),
    "report.py": ("import journal", False),
    "paper.py": ("import log", False),
    "broker.py": ("import log", False),
    "candidates.py": ("import bracket", False),
    "shadow.py": ("import bracket", False),
    "calib.py": ("import market", False),
}


def main():
    for name, (anchor, blank) in PLACE.items():
        path = os.path.join(BOT, name)
        with open(path, "r", encoding="utf-8", newline="") as f:
            src = f.read()
        nl = "\r\n" if "\r\n" in src else "\n"
        wrong = f"{nl}import os{nl}import paths{nl}"
        if src.count(wrong) != 1:
            sys.exit(f"{name}: misplaced import count {src.count(wrong)}")
        src = src.replace(wrong, f"{nl}import os{nl}", 1)
        a = f"{nl}{anchor}{nl}"
        if src.count(a) != 1:
            sys.exit(f"{name}: anchor {anchor!r} count {src.count(a)}")
        src = src.replace(a, f"{nl}{anchor}{nl}{nl if blank else ''}import paths{nl}", 1)
        with open(path, "w", encoding="utf-8", newline="") as f:
            f.write(src)
        print("moved", name)


if __name__ == "__main__":
    main()
