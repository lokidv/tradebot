# -*- coding: utf-8 -*-
"""تاریخچهٔ روزانهٔ همهٔ جفت‌های USDTِ فعالِ اسپاتِ بایننس که در کش نیستند.

کشِ تاریخچه فقط ارزهای پرحجمِ **امروز** را داشت؛ برای جهانِ نقطه‌-در-زمانِ ۵۰تایی در
۲۰۱۸–۲۰۲۱، ارزهایی که آن زمان بزرگ بودند و بعد افت کردند (NEO، QTUM، VET، ...) غایب
بودند — یعنی سوگیریِ بقا به نفعِ لانگ. ارزهای کاملاً حذف‌شده (LUNA، FTT) از API در
دسترس نیستند؛ این سوگیریِ باقی‌مانده است.

اجرا:  python tools/fetch_history_universe.py
"""
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "bot"))

import explore  # noqa: E402
import market  # noqa: E402


def main():
    info = market._binance_json("/api/v3/exchangeInfo", None)
    syms = sorted(s["symbol"] for s in info["symbols"]
                  if s.get("quoteAsset") == "USDT" and s.get("status") == "TRADING")
    have = {f[:-len("_1d.json")] for f in os.listdir(market.HIST_DIR)
            if f.endswith("_1d.json") and not f.startswith("funding_")}
    todo = [s for s in syms if s not in have and not explore._excluded(s)]
    print(f"{len(syms)} USDT pairs trading; {len(have)} already cached; {len(todo)} to fetch", flush=True)
    ok = short = 0
    for i, s in enumerate(todo, 1):
        try:
            market.get_history(s, "1d", bars=3000)
            ok += 1
        except Exception:  # noqa: BLE001 — کمتر از ۴۰۰ کندل یا نمادِ بی‌داده: برای این پژوهش بی‌اثر
            short += 1
        if i % 25 == 0:
            print(f"  {i}/{len(todo)}  fetched={ok}  short/failed={short}", flush=True)
        time.sleep(0.15)
    print(f"done: fetched={ok}  short/failed={short}", flush=True)


if __name__ == "__main__":
    main()
