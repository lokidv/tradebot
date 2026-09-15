# -*- coding: utf-8 -*-
"""یک پوشهٔ داده برای کلِ ربات — قابلِ جابه‌جایی با ``TRADERBOT_DATA_DIR``.

هر ماژول مسیرِ فایلش را از اینجا می‌گیرد، نه از ``__file__`` خودش. تست‌ها قبلاً
یک فایل را جابه‌جا می‌کردند (gates.json) و عوارضش را فراموش: ``gates._save``
یک GATE_CHANGE در ژورنال می‌نویسد و آن به **دفترِ واقعی** می‌رفت — ۱۵۵ رویدادِ
ساختگی («4h|zx|long مجاز شد»، «ریسک ۰٫۵٪ شد») کنارِ دو رویدادِ واقعی، و گزارش
آن‌ها را «تغییرِ گیت در دورهٔ اثبات» می‌شمرد و ساعتِ اثبات را صفر می‌کرد.

حالا تست‌ها (tests/_hermetic.py) این متغیر را پیش از importِ هر ماژولی به یک
پوشهٔ موقت می‌برند؛ پروسهٔ بازسازی هم آن را از والدش ارث می‌برد.
"""
import os

DEFAULT_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
DATA_DIR = os.path.abspath(os.environ.get("TRADERBOT_DATA_DIR") or DEFAULT_DATA_DIR)


def data(*parts):
    """مسیری داخلِ پوشهٔ داده."""
    return os.path.join(DATA_DIR, *parts)
