# -*- coding: utf-8 -*-
"""نسخهٔ ایزولهٔ برنامه برای آزمایشِ سرتاسری: پورتِ جدا + کپیِ موقتِ داده‌ها.

نسخهٔ اصلی (پورتِ ۸۷۸۷، دادهٔ واقعی) دست نمی‌خورد: این‌جا هر چه باز و بسته شود در یک پوشهٔ موقت
می‌ماند و با بسته‌شدنِ برنامه دور ریخته می‌شود. ربات خاموش و بروکر «شبیه‌سازِ محلی» است؛ کلیدی کپی نمی‌شود.

اجرا:  python tools/run_isolated.py            (پورتِ ۸۷۸۸)
"""
import json
import os
import runpy
import shutil
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BOT = os.path.join(ROOT, "bot")
REAL = os.path.join(BOT, "data")
COPY_FILES = ("positions.json", "closed_trades.jsonl", "gates.json", "preregistration.json", "calib.json")
COPY_DIRS = ("research", "final_test")


def main():
    tmp = tempfile.mkdtemp(prefix="traderbot-isolated-")
    for name in COPY_FILES:
        src = os.path.join(REAL, name)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(tmp, name))
    for name in COPY_DIRS:
        src = os.path.join(REAL, name)
        if os.path.isdir(src):
            shutil.copytree(src, os.path.join(tmp, name))
    with open(os.path.join(tmp, "autobot.json"), "w", encoding="utf-8") as f:
        json.dump({"enabled": False, "level": 5}, f)                       # ربات خاموش
    os.environ["TRADERBOT_DATA_DIR"] = tmp
    os.environ.setdefault("TRADERBOT_PORT", os.environ.get("PORT") or "8788")
    print(f"[isolated] data dir: {tmp}  port: {os.environ['TRADERBOT_PORT']}", flush=True)
    sys.path.insert(0, BOT)
    os.chdir(BOT)
    try:
        runpy.run_path(os.path.join(BOT, "main.py"), run_name="__main__")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
