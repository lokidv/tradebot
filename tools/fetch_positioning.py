# -*- coding: utf-8 -*-
"""دادهٔ موقعیت‌گیریِ معامله‌گرانِ فیوچرز از آرشیوِ عمومیِ بایننس، برای دورِ دومِ پژوهشِ ۱۵دقیقه‌ای.

- metrics (روزانه، هر ۵ دقیقه): open interest، نسبتِ لانگ/شورتِ حساب‌های برتر و کلِ حساب‌ها،
  نسبتِ حجمِ taker خرید/فروش.
- premiumIndexKlines (ماهانه، ۱۵ دقیقه): فاصلهٔ قیمتِ فیوچرز از شاخصِ اسپات (basis).

خروجی: hist_research/micro/pos_<SYM>.npz (ستون‌های metrics روی شبکهٔ ۵دقیقه) و prem_<SYM>_15m.npz
اجرا:  python tools/fetch_positioning.py
"""
import csv
import datetime as dt
import io
import os
import sys
import zipfile
from concurrent.futures import ThreadPoolExecutor

import httpx
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "bot"))
import paths  # noqa: E402
import watchlist  # noqa: E402

OUT = paths.data("hist_research", "micro")
METRICS = "https://data.binance.vision/data/futures/um/daily/metrics/{s}/{s}-metrics-{d}.zip"
PREM = "https://data.binance.vision/data/futures/um/monthly/premiumIndexKlines/{s}/15m/{s}-15m-{y}-{m:02d}.zip"
START = dt.date(2021, 12, 1)
END = dt.date(2026, 8, 31)
COLS = ("oi", "oi_value", "top_acct_ratio", "top_pos_ratio", "acct_ratio", "taker_ratio")


def _zip_rows(content):
    z = zipfile.ZipFile(io.BytesIO(content))
    return list(csv.reader(io.TextIOWrapper(z.open(z.namelist()[0]), encoding="utf-8")))


def _day(cli, sym, d):
    for _ in range(3):
        try:
            r = cli.get(METRICS.format(s=sym, d=d.isoformat()))
            if r.status_code == 404:
                return []
            if r.status_code == 200:
                break
        except httpx.HTTPError:
            continue
    else:
        return []
    out = []
    for row in _zip_rows(r.content):
        if not row or not row[0][:1].isdigit():
            continue
        t = int(dt.datetime.strptime(row[0], "%Y-%m-%d %H:%M:%S").replace(tzinfo=dt.timezone.utc).timestamp() * 1000)
        vals = []
        for x in row[2:8]:
            try:
                vals.append(float(x))
            except ValueError:
                vals.append(np.nan)
        out.append([t] + vals)
    return out


def metrics(sym):
    path = os.path.join(OUT, f"pos_{sym}.npz")
    if os.path.exists(path):
        return path
    days = [START + dt.timedelta(i) for i in range((END - START).days + 1)]
    with httpx.Client(timeout=30) as cli, ThreadPoolExecutor(16) as ex:
        rows = [r for chunk in ex.map(lambda d: _day(cli, sym, d), days) for r in chunk]
    a = np.asarray(sorted(rows), float)
    t = a[:, 0].astype(np.int64)
    keep = np.concatenate([[True], np.diff(t) > 0])
    np.savez_compressed(path, t=t[keep], **{c: a[keep, i + 1] for i, c in enumerate(COLS)})
    return path


def premium(sym):
    path = os.path.join(OUT, f"prem_{sym}_15m.npz")
    if os.path.exists(path):
        return path
    ym = [(y, m) for y in range(2021, 2027) for m in range(1, 13) if (2021, 12) <= (y, m) <= (2026, 8)]
    rows = []
    with httpx.Client(timeout=30) as cli:
        for y, m in ym:
            r = cli.get(PREM.format(s=sym, y=y, m=m))
            if r.status_code != 200:
                continue
            for row in _zip_rows(r.content):
                if row and row[0].isdigit():
                    rows.append((int(row[0]), float(row[4])))       # بستهٔ شاخصِ پرمیوم
    a = np.asarray(sorted(set(rows)), float)
    np.savez_compressed(path, t=a[:, 0].astype(np.int64), prem=a[:, 1])
    return path


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    for s in watchlist.SYMBOLS:
        d = np.load(metrics(s))
        p = np.load(premium(s))
        print(s, "metrics", len(d["t"]), "prem", len(p["t"]), flush=True)
