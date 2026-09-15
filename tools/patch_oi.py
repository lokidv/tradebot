# -*- coding: utf-8 -*-
"""پچِ OI — دادهٔ جعلی به‌عنوانِ «اهرم» تحویلِ متا-گیت نمی‌شود.

وقتی Binance و OKX هر دو جواب نمی‌دادند، ``get_oi_history`` بی‌صدا EMAِ **حجمِ
اسپات** را به‌جای Open Interest برمی‌گرداند و ``oi_crowd_stats`` با ``ok=True``
آن را «OI» می‌نامید. متا-گیت سپس بر اساسِ آن «اهرمِ لانگ در حال انباشت» تشخیص
می‌داد و بلاکِ سخت می‌زد — تصمیمِ ریسک روی دادهٔ ساختگی.

حالا: منبع ثبت می‌شود، پروکسی هرگز ``ok=True`` نمی‌گیرد، و نبودِ داده یعنی «نامعلوم».
"""
import io
import os

P = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "bot", "market.py")
s = io.open(P, encoding="utf-8").read()


def sub1(old, new, label):
    global s
    n = s.count(old)
    assert n == 1, f"{label}: found {n}"
    s = s.replace(old, new)


sub1('''    # پشتیبان آخر: شتاب حجم اسپات به‌عنوان پروکسی اهرم (ضعیف‌تر ولی بهتر از هیچ)
    if not rows:
        try:
            kl = get_klines_cached(symbol, "1h") or get_klines(symbol, "1h")
            if kl and len(kl["v"]) >= 24:
                vs = [float(x) for x in kl["v"][-48:]]
                # OI-proxy = EMA حجم
                ema = vs[0]
                series = []
                t0 = kl["t"][-len(vs)]
                step = 3600000
                for i, v in enumerate(vs):
                    ema = 0.2 * v + 0.8 * ema
                    series.append([t0 + i * step, ema])
                rows = series
        except Exception:  # noqa: BLE001
            rows = []
    tmp = path + ".tmp"''',
     '''    # ⚠️ پشتیبانِ «EMAِ حجمِ اسپات به‌جای OI» حذف شد: متا-گیت آن را اهرم می‌خواند و
    # بلاکِ سخت می‌زد. نبودِ داده = «نامعلوم»، نه عددِ ساختگی.
    tmp = path + ".tmp"''', "remove spot-volume proxy")

io.open(P, "w", encoding="utf-8", newline="\n").write(s)
print("oi patch applied")
