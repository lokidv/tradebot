# -*- coding: utf-8 -*-
"""آربیتراژِ فاندینگ (cash-and-carry): اسپات بخر، همان مقدار پرپچوال بفروش، فاندینگ را جمع کن.

سود از پیش‌بینیِ قیمت نمی‌آید؛ از نرخی می‌آید که لانگ‌های اهرمی به شورت‌ها می‌پردازند. پس این‌جا
پارامتری برای بهینه‌سازی نیست — فقط اندازه‌گیریِ تاریخی: فاندینگِ واقعیِ هر ۸ ساعت از آرشیوِ عمومیِ
بایننس (data.binance.vision؛ API فیوچرز از ایران 451 می‌دهد).

دو نسخه، هر دو پیش از دیدنِ نتیجه تعریف شده‌اند:
  always  — همیشه در پوزیشن (ساده‌ترین، بدون هیچ انتخابی)
  switch  — فقط وقتی میانگینِ فاندینگِ ۷ روزِ **گذشته** مثبت است (قاعدهٔ عمومی؛ بدونِ جست‌وجوی پارامتر)
هزینه: هر ورود/خروجِ کامل ۰٫۳٪ ارزشِ پوزیشن (اسپات ۰٫۱٪×۲ + پرپ ۰٫۰۵٪×۲).
بازده روی کلِ سرمایهٔ درگیر (نیمی اسپات، نیمی وثیقهٔ شورت با اهرمِ ۱) گزارش می‌شود.

اجرا:  python tools/funding_carry_study.py
"""
import csv
import io
import json
import os
import sys
import time
import zipfile

import httpx
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "bot"))
import paths  # noqa: E402

ARCHIVE = "https://data.binance.vision/data/futures/um/monthly/fundingRate/{s}/{s}-fundingRate-{y}-{m:02d}.zip"
OUT = paths.data("hist_research")
SYMBOLS = ("BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT", "DOGEUSDT")
ROUND_TRIP_PCT = 0.3
SWITCH_LOOKBACK = 21                        # ۲۱ تسویه = ۷ روز


def fetch(sym):
    path = os.path.join(OUT, f"funding_archive_{sym}.json")
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    rows, cli = [], httpx.Client(timeout=30)
    y, m = 2020, 1
    now = time.gmtime()
    while (y, m) < (now.tm_year, now.tm_mon):
        r = cli.get(ARCHIVE.format(s=sym, y=y, m=m))
        if r.status_code == 200:
            z = zipfile.ZipFile(io.BytesIO(r.content))
            for row in csv.reader(io.TextIOWrapper(z.open(z.namelist()[0]), encoding="utf-8")):
                if row and row[0].isdigit():
                    rows.append([int(row[0]), float(row[2])])
        m += 1
        if m > 12:
            y, m = y + 1, 1
    rows.sort()
    os.makedirs(OUT, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(rows, f)
    return rows


def simulate(rates, mode):
    """بازدهِ هر تسویه روی کلِ سرمایه (نصفِ نُشنال): فاندینگ/۲، منهای هزینهٔ ورود/خروج."""
    rates = np.asarray(rates, float)
    inpos = np.ones(len(rates), bool)
    if mode == "switch":
        csum = np.concatenate([[0.0], np.cumsum(rates)])
        for i in range(len(rates)):
            lo = max(0, i - SWITCH_LOOKBACK)
            inpos[i] = i >= SWITCH_LOOKBACK and (csum[i] - csum[lo]) > 0     # فقط گذشته
    ret = np.where(inpos, rates / 2.0, 0.0)
    switches = np.abs(np.diff(np.concatenate([[False], inpos, [False]]).astype(int))).sum() / 2
    ret = ret.copy()
    flips = np.flatnonzero(np.diff(np.concatenate([[False], inpos]).astype(int)) != 0)
    ret[flips] -= ROUND_TRIP_PCT / 100 / 2 / 2              # نیمِ رفت‌وبرگشت در هر تغییر، روی کلِ سرمایه
    return ret, inpos, int(switches)


def summarize(ts, ret, inpos, switches):
    ts = np.asarray(ts)
    eq = np.cumprod(1 + ret)
    years = (ts[-1] - ts[0]) / (365.25 * 86_400_000)
    dd = float(np.max(1 - eq / np.maximum.accumulate(eq)))
    by_year = {}
    for t, r in zip(ts, ret):
        by_year.setdefault(time.gmtime(t / 1000).tm_year, []).append(r)
    by_month = {}
    for t, r in zip(ts, ret):
        g = time.gmtime(t / 1000)
        by_month.setdefault((g.tm_year, g.tm_mon), 0.0)
        by_month[(g.tm_year, g.tm_mon)] += r
    months = np.asarray(list(by_month.values()))
    return {"cagr_pct": round((eq[-1] ** (1 / years) - 1) * 100, 2), "max_drawdown_pct": round(dd * 100, 2),
            "time_in_position_pct": round(float(inpos.mean()) * 100, 1), "round_trips": switches,
            "negative_months_pct": round(float((months < 0).mean()) * 100, 1),
            "worst_month_pct": round(float(months.min()) * 100, 2),
            "by_year_pct": {y: round(float(np.prod(1 + np.asarray(v)) - 1) * 100, 2) for y, v in sorted(by_year.items())}}


def main():
    out = {"generated_at": time.time(), "round_trip_pct": ROUND_TRIP_PCT, "symbols": {}}
    for sym in SYMBOLS:
        rows = fetch(sym)
        if len(rows) < 1000:
            print(sym, "insufficient", len(rows))
            continue
        ts, rates = [r[0] for r in rows], [r[1] for r in rows]
        res = {"first": time.strftime("%Y-%m-%d", time.gmtime(ts[0] / 1000)), "n_settlements": len(rates),
               "mean_rate_annual_pct": round(float(np.mean(rates)) * 3 * 365 * 100, 2),
               "negative_settlements_pct": round(float((np.asarray(rates) < 0).mean()) * 100, 1)}
        for mode in ("always", "switch"):
            ret, inpos, sw = simulate(rates, mode)
            res[mode] = summarize(ts, ret, inpos, sw)
        out["symbols"][sym] = res
        print(sym, json.dumps(res, ensure_ascii=False))
    path = paths.data("research", time.strftime("funding_carry_%Y%m%d_%H%M.json", time.gmtime()))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print("saved:", path)


if __name__ == "__main__":
    main()
