# -*- coding: utf-8 -*-
"""هر فایلِ تست این را **پیش از** هر ماژولِ ربات import می‌کند: کلِ پوشهٔ داده به یک
پوشهٔ موقت می‌رود، پس هیچ تستی نمی‌تواند به دفترِ واقعی دست بزند.

قبلاً هر تست یک فایل را جابه‌جا می‌کرد (مثلاً gates.json) و عوارضِ جانبی‌اش را
فراموش — نوشتنِ gates در ژورنال، هشدارِ لاگ — و آن‌ها به ``bot/data`` واقعی می‌رفت.
test_hermetic.py بررسی می‌کند که هر فایلِ تست این را اول import کند.
"""
import atexit
import os
import shutil
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BOT = os.path.join(ROOT, "bot")
REAL_DATA_DIR = os.path.normcase(os.path.join(BOT, "data"))


def _same(a, b):
    return os.path.normcase(os.path.abspath(a)) == os.path.normcase(os.path.abspath(b))


if not os.environ.get("TRADERBOT_DATA_DIR"):
    DATA_DIR = tempfile.mkdtemp(prefix="traderbot-test-")
    os.environ["TRADERBOT_DATA_DIR"] = DATA_DIR
    atexit.register(shutil.rmtree, DATA_DIR, True)
DATA_DIR = os.environ["TRADERBOT_DATA_DIR"]

if _same(DATA_DIR, REAL_DATA_DIR):
    raise RuntimeError("TRADERBOT_DATA_DIR به دادهٔ واقعی اشاره می‌کند — تست‌ها اجرا نمی‌شوند")

if BOT not in sys.path:
    sys.path.insert(0, BOT)

import paths  # noqa: E402

if not _same(paths.DATA_DIR, DATA_DIR):
    # یک ماژولِ ربات پیش از این فایل import شده و مسیرِ واقعی را گرفته است
    raise RuntimeError(f"paths.DATA_DIR={paths.DATA_DIR} — _hermetic باید اولین import باشد")
