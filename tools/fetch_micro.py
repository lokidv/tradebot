# -*- coding: utf-8 -*-
"""دادهٔ ۱۵دقیقه‌ای + حجمِ خریدِ تهاجمی (taker buy) از آرشیوِ عمومیِ بایننس، برای پژوهشِ معاملهٔ ۱۵دقیقه‌ای.

قیمتِ پنج ارزِ بزرگ بینِ بایننس و کوکوین عملاً یکی است (آربیتراژ)، ولی جریانِ سفارش (taker buy) فقط
در آرشیوِ بایننس به‌صورتِ کامل و رایگان هست. خروجی: hist_research/micro/<market>_<SYM>_15m.npz
ستون‌ها: t (ms)، o، h، l، c، v، qv، n (تعدادِ معامله)، tbv (حجمِ خریدِ تهاجمیِ پایه).

اجرا:  python tools/fetch_micro.py
"""
import csv
import io
import os
import sys
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor

import httpx
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "bot"))
import paths  # noqa: E402
import watchlist  # noqa: E402

OUT = paths.data("hist_research", "micro")
URL = {"spot": "https://data.binance.vision/data/spot/monthly/klines/{s}/15m/{s}-15m-{y}-{m:02d}.zip",
       "um": "https://data.binance.vision/data/futures/um/monthly/klines/{s}/15m/{s}-15m-{y}-{m:02d}.zip"}
START = (2022, 1)
END = (2026, 8)                      # آخرین ماهِ کاملِ آرشیو


def months():
    y, m = START
    while (y, m) <= END:
        yield y, m
        m += 1
        if m > 12:
            y, m = y + 1, 1


def fetch_month(cli, market, sym, y, m):
    r = cli.get(URL[market].format(s=sym, y=y, m=m))
    if r.status_code != 200:
        return []
    z = zipfile.ZipFile(io.BytesIO(r.content))
    rows = []
    for row in csv.reader(io.TextIOWrapper(z.open(z.namelist()[0]), encoding="utf-8")):
        if not row or not row[0].isdigit():
            continue                                   # سرتیترِ آرشیوِ فیوچرز
        t = int(row[0])
        if t > 10**14:
            t //= 1000                                 # آرشیوِ اسپات از ۲۰۲۵ میکروثانیه است
        rows.append((t, float(row[1]), float(row[2]), float(row[3]), float(row[4]), float(row[5]),
                     float(row[7]), float(row[8]), float(row[9])))
    return rows


def fetch(market, sym):
    path = os.path.join(OUT, f"{market}_{sym}_15m.npz")
    if os.path.exists(path):
        return "cached"
    cli = httpx.Client(timeout=60)
    with ThreadPoolExecutor(max_workers=6) as pool:
        parts = list(pool.map(lambda ym: fetch_month(cli, market, sym, *ym), months()))
    rows = sorted({r[0]: r for part in parts for r in part}.values())
    if len(rows) < 10_000:
        return f"short ({len(rows)})"
    a = np.asarray(rows, float)
    np.savez_compressed(path, t=a[:, 0].astype(np.int64), o=a[:, 1], h=a[:, 2], l=a[:, 3], c=a[:, 4],
                        v=a[:, 5], qv=a[:, 6], n=a[:, 7], tbv=a[:, 8])
    return f"ok {len(rows)} bars {time.strftime('%Y-%m', time.gmtime(a[0, 0] / 1000))}..{time.strftime('%Y-%m', time.gmtime(a[-1, 0] / 1000))}"


def main():
    os.makedirs(OUT, exist_ok=True)
    for market in ("um", "spot"):
        for sym in watchlist.SYMBOLS:
            print(market, sym, fetch(market, sym), flush=True)


if __name__ == "__main__":
    main()
