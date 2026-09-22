# -*- coding: utf-8 -*-
"""تاریخچهٔ ۴h و ۱h برای پژوهشِ روند روی تایم‌فریم‌های درون‌روزی — در پوشهٔ **جدا** (hist_research).

کشِ اصلی (data/hist) را مدل‌ها برای ساخت می‌خوانند؛ تاریخچهٔ بلندترِ پژوهشی نباید دادهٔ آموزشِ
آن‌ها را عوض کند. فقط ارزهایی که در یک ماه از ۲۰۲۲-۰۱ تا ۲۰۲۵-۰۸ جزوِ ۵۰ ارزِ برترِ همان ماه
بوده‌اند (جهانِ نقطه‌-در-زمان از دادهٔ روزانه) گرفته می‌شوند.

اجرا:  python tools/fetch_history_intraday.py
"""
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "bot"))

import explore  # noqa: E402
import market  # noqa: E402
import paths  # noqa: E402
import universe  # noqa: E402

OUT = paths.data("hist_research")
START_MS = 1_640_995_200_000          # 2022-01-01
END_MS = 1_754_092_800_000            # 2025-08-02 — آغازِ پنجرهٔ منجمدِ ۴h؛ بعد از آن لازم نیست
TF_MS = {"4h": 14_400_000, "1h": 3_600_000}


def candidates():
    daily = explore.load_panel("1d", END_MS)
    snaps = universe.snapshots_from_histories(
        {s: {"t": k["t"].tolist(), "c": k["c"].tolist(), "v": k["v"].tolist()} for s, k in daily.items()},
        top_n=50)
    return sorted(set().union(*[u for ts, u in snaps if START_MS <= ts <= END_MS]))


def fetch(sym, tf):
    path = os.path.join(OUT, f"{sym}_{tf}.json")
    if os.path.exists(path):
        return "cached"
    rows, start = [], START_MS
    while start < END_MS:
        raw = market._binance_json("/api/v3/klines", {"symbol": sym, "interval": tf, "startTime": start,
                                                       "endTime": END_MS - 1, "limit": 1000})
        if not raw:
            break
        rows.extend((int(r[0]), float(r[1]), float(r[2]), float(r[3]), float(r[4]), float(r[5])) for r in raw)
        start = int(raw[-1][0]) + TF_MS[tf]
        if len(raw) < 1000:
            break
        time.sleep(0.05)
    if len(rows) < 500:
        return "short"
    out = {k: [r[i] for r in rows] for i, k in enumerate(("t", "o", "h", "l", "c", "v"))}
    with open(path + ".tmp", "w", encoding="utf-8") as f:
        json.dump(out, f)
    os.replace(path + ".tmp", path)
    return "ok"


def main():
    os.makedirs(OUT, exist_ok=True)
    syms = candidates()
    print(f"{len(syms)} coins were in a monthly top-50 between 2022-01 and 2025-08", flush=True)
    from concurrent.futures import ThreadPoolExecutor, as_completed

    def safe(s, tf):
        try:
            return fetch(s, tf)
        except Exception:  # noqa: BLE001 — نمادِ بی‌داده برای این پژوهش بی‌اثر است
            return "fail"

    # هر درخواست از data-api حدودِ ۲ ثانیه است؛ ۱۲ درخواستِ هم‌زمان ≈ ۱۴ وزن در ثانیه، زیرِ سقفِ بایننس
    for tf in ("4h", "1h"):
        stat = {}
        with ThreadPoolExecutor(max_workers=12) as pool:
            futs = [pool.submit(safe, s, tf) for s in syms]
            for i, f in enumerate(as_completed(futs), 1):
                r = f.result()
                stat[r] = stat.get(r, 0) + 1
                if i % 25 == 0:
                    print(f"  {tf} {i}/{len(syms)} {stat}", flush=True)
        print(f"{tf} done: {stat}", flush=True)


if __name__ == "__main__":
    main()
