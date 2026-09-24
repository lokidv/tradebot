# -*- coding: utf-8 -*-
"""داده‌هایی که در نمودارِ قیمت نیستند — برای فرضیه‌های فاز ۲ (prereg_new_data_ensemble.json).

هر سری به شکلِ ``[[day_ms, value], ...]`` (روزِ UTC، مرتب) در ``hist_research/ext_<name>.json`` ذخیره
می‌شود. همه از منابعِ عمومیِ در دسترس از ایران؛ stooq به‌خاطرِ چالشِ ضدربات کنار گذاشته شد.
"""
import csv
import io
import json
import os
import time
from datetime import datetime, timezone

import httpx

import paths

DAY_MS = 86_400_000
OUT = paths.data("hist_research")
UA = {"User-Agent": "ctp-research/1.0"}
TTL_SEC = 6 * 3600


def _day(ms):
    return int(ms) // DAY_MS * DAY_MS


def _iso_day(s):
    return int(datetime.strptime(s[:10], "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp() * 1000)


def _get(url, params=None, timeout=60):
    r = httpx.get(url, params=params, headers=UA, timeout=timeout, follow_redirects=True)
    r.raise_for_status()
    return r


def stablecoin_supply():
    rows = _get("https://stablecoins.llama.fi/stablecoincharts/all").json()
    return [[_day(int(r["date"]) * 1000), float(sum((r.get("totalCirculatingUSD") or {}).values()))] for r in rows]


def mvrv(asset):
    out, url = [], "https://community-api.coinmetrics.io/v4/timeseries/asset-metrics"
    params = {"assets": asset, "metrics": "CapMVRVCur", "frequency": "1d", "page_size": 10000,
              "start_time": "2015-01-01"}
    while url:
        j = _get(url, params).json()
        out += [[_iso_day(r["time"]), float(r["CapMVRVCur"])] for r in j.get("data", []) if r.get("CapMVRVCur")]
        url, params = j.get("next_page_url"), None
    return out


def dvol(currency):
    out, end = {}, int(time.time() * 1000)
    start_floor = 1_609_459_200_000                         # 2021-01-01؛ DVOL از 2021-03 شروع می‌شود
    while end > start_floor:
        j = _get("https://www.deribit.com/api/v2/public/get_volatility_index_data",
                 {"currency": currency, "start_timestamp": start_floor, "end_timestamp": end,
                  "resolution": 86400}).json()["result"]
        data = j.get("data") or []
        for ts, _o, _h, _l, c in data:
            out[_day(ts)] = float(c)
        cont = j.get("continuation")
        if not data or not cont or int(cont) >= end:
            break
        end = int(cont)
        time.sleep(0.2)
    return sorted([k, v] for k, v in out.items())


def fred(series):
    text = _get("https://fred.stlouisfed.org/graph/fredgraph.csv", {"id": series}).text
    out = []
    for row in csv.reader(io.StringIO(text)):
        if len(row) == 2 and row[0][:1].isdigit() and row[1] not in ("", "."):
            out.append([_iso_day(row[0]), float(row[1])])
    return out


def fear_greed():
    rows = _get("https://api.alternative.me/fng/", {"limit": 0, "format": "json"}).json()["data"]
    return sorted([_day(int(r["timestamp"]) * 1000), float(r["value"])] for r in rows)


SOURCES = {
    "stablecoin_supply": stablecoin_supply,
    "mvrv_btc": lambda: mvrv("btc"), "mvrv_eth": lambda: mvrv("eth"),
    "dvol_btc": lambda: dvol("BTC"), "dvol_eth": lambda: dvol("ETH"),
    "spx": lambda: fred("SP500"), "dxy_broad": lambda: fred("DTWEXBGS"),
    "fear_greed": fear_greed,
}


def path(name):
    return os.path.join(OUT, f"ext_{name}.json")


def fetch(name, force=False):
    p = path(name)
    if not force and os.path.exists(p) and time.time() - os.path.getmtime(p) < TTL_SEC:
        return load(name)
    rows = SOURCES[name]()
    if len(rows) < 100:
        raise RuntimeError(f"{name}: only {len(rows)} rows")
    os.makedirs(OUT, exist_ok=True)
    with open(p + ".tmp", "w", encoding="utf-8") as f:
        json.dump(rows, f)
    os.replace(p + ".tmp", p)
    return rows


def load(name):
    with open(path(name), encoding="utf-8") as f:
        return json.load(f)


def asof(rows, day_ms, lag_days=0):
    """آخرین مقدار با تاریخِ ≤ ``day_ms − lag_days`` — بدونِ نگاه به آینده. ``None`` اگر نبود."""
    import bisect
    keys = [r[0] for r in rows]
    i = bisect.bisect_right(keys, day_ms - lag_days * DAY_MS) - 1
    return rows[i][1] if i >= 0 else None


if __name__ == "__main__":
    for n in SOURCES:
        try:
            rows = fetch(n, force=True)
            print(n, len(rows), time.strftime("%Y-%m-%d", time.gmtime(rows[0][0] / 1000)), "->",
                  time.strftime("%Y-%m-%d", time.gmtime(rows[-1][0] / 1000)))
        except Exception as e:  # noqa: BLE001
            print(n, "FAILED", str(e)[:120])
