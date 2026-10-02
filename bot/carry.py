# -*- coding: utf-8 -*-
"""ردیاب سایهٔ فاندینگ‌کَری (cash-and-carry) برای BTC و ETH — فقط ثبتِ رو-به-جلو، بدونِ سفارش.

پژوهش: tools/funding_carry_study.py و گزارشِ funding_carry_*.json؛ پیش‌ثبت: prereg_carry_shadow.json.
قاعده: همیشه در پوزیشن (اسپات بخر + همان نُشنال پرپچوالِ USDT بفروش)، بدونِ هیچ پارامتر. سود از
فاندینگی می‌آید که لانگ‌های اهرمی می‌پردازند، نه از پیش‌بینیِ قیمت. این ماژول فقط تسویه‌های واقعیِ
فاندینگ را از ``TRACKING_START_MS`` به بعد در دفتر می‌نویسد و بازدهِ فرضیِ آن را می‌شمارد؛ سفارشی
نمی‌دهد و به gates دست نمی‌زند.
"""
import json
import os
import threading
import time

import numpy as np

import log
import market
import paths

SYMBOLS = ("BTCUSDT", "ETHUSDT")
DAY_MS = 86_400_000
TRACKING_START_MS = 1_790_553_600_000        # دوشنبه 2026-09-28 00:00 UTC — هم‌زمان با مومنتوم
LEDGER_PATH = paths.data("carry_ledger.jsonl")
ROUND_TRIP_PCT = 0.3                         # اسپات ۰٫۱٪×۲ + پرپ ۰٫۰۵٪×۲، مثلِ مطالعهٔ تاریخی
ENTRY_COST_CAPITAL = ROUND_TRIP_PCT / 100 / 2 / 2   # نیمِ رفت‌وبرگشت، روی نیمِ سرمایه
MIN_DAYS = 84
MIN_ANNUAL_PCT = 2.0
BRAKE_DD_PCT = 3.0
SETTLEMENTS_PER_YEAR = 3 * 365
REFRESH_SEC = 1800
_lock = threading.Lock()
_cache = {"ts": 0.0, "data": None}


def _read():
    if not os.path.exists(LEDGER_PATH):
        return []
    out = []
    with open(LEDGER_PATH, encoding="utf-8") as f:
        for line in f:
            try:
                out.append(json.loads(line))
            except ValueError:
                continue
    return out


def record(sym, rows, start_ms=TRACKING_START_MS):
    """هر تسویه از ``start_ms`` به بعد دقیقاً یک‌بار در دفتر؛ تعدادِ ردیفِ تازه برمی‌گردد."""
    with _lock:
        seen = {(r["sym"], r["ts"]) for r in _read()}
        new = [{"sym": sym, "ts": int(t), "rate": float(r), "logged_at": time.time()}
               for t, r in rows if int(t) >= start_ms and (sym, int(t)) not in seen]
        new.sort(key=lambda r: r["ts"])
        if new:
            os.makedirs(os.path.dirname(LEDGER_PATH), exist_ok=True)
            with open(LEDGER_PATH, "a", encoding="utf-8") as f:
                for r in new:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return len(new)


def stats(rows):
    """کارنامهٔ یک نماد از ردیف‌های دفتر؛ بازده روی کلِ سرمایه (نصفِ اسپات، نصفِ وثیقهٔ شورت)."""
    rows = sorted({r["ts"]: r for r in rows}.values(), key=lambda r: r["ts"])
    n = len(rows)
    out = {"settlements": n, "entry_cost_pct": round(ENTRY_COST_CAPITAL * 100, 3)}
    if n == 0:
        return {**out, "days": 0.0, "funding_pct": 0.0, "net_return_pct": round(-ENTRY_COST_CAPITAL * 100, 3),
                "max_dd_pct": 0.0, "negative_settlements_pct": None, "mean_rate_annual_pct": None,
                "last7d_rate_annual_pct": None, "verdict": "IN_PROGRESS",
                "verdict_reason": f"هنوز تسویه‌ای ثبت نشده — حکم بعد از {MIN_DAYS} روز"}
    rates = np.asarray([r["rate"] for r in rows], float)
    ret = rates / 2.0
    ret[0] -= ENTRY_COST_CAPITAL
    eq = np.concatenate([[1.0], np.cumprod(1 + ret)])
    dd = float(np.max(1 - eq / np.maximum.accumulate(eq))) * 100
    days = (rows[-1]["ts"] - TRACKING_START_MS) / DAY_MS
    net = (eq[-1] - 1) * 100
    last7 = rates[-21:]
    out.update({"days": round(days, 1), "funding_pct": round(float(rates.sum() / 2) * 100, 3),
                "net_return_pct": round(net, 3), "max_dd_pct": round(dd, 3),
                "negative_settlements_pct": round(float((rates < 0).mean()) * 100, 1),
                "mean_rate_annual_pct": round(float(rates.mean()) * SETTLEMENTS_PER_YEAR * 100, 2),
                "last7d_rate_annual_pct": round(float(last7.mean()) * SETTLEMENTS_PER_YEAR * 100, 2),
                "last_rate_pct": round(float(rates[-1]) * 100, 4), "last_ts": rows[-1]["ts"]})
    pace = float(eq[-1] ** (365.25 / days) - 1) * 100 if days >= 7 and eq[-1] > 0 else None
    out["annual_pace_pct"] = None if pace is None else round(pace, 2)
    if dd > BRAKE_DD_PCT:
        out["verdict"], out["verdict_reason"] = "BRAKE", f"افتِ زنده {dd:.1f}٪ از سقفِ {BRAKE_DD_PCT:.0f}٪ گذشت"
    elif days < MIN_DAYS:
        out["verdict"], out["verdict_reason"] = "IN_PROGRESS", f"{days:.0f} از {MIN_DAYS} روز"
    elif net > 0 and pace is not None and pace >= MIN_ANNUAL_PCT:
        out["verdict"], out["verdict_reason"] = "GO_SMALL", "معیارِ ازپیش‌ثبت‌شده برقرار شد: فقط با مبلغِ کم"
    else:
        out["verdict"], out["verdict_reason"] = "NO_GO", "معیارِ ازپیش‌ثبت‌شده برقرار نشد"
    return out


def snapshot(force=False, fetch=None):
    with _lock:
        if not force and fetch is None and _cache["data"] is not None and time.time() - _cache["ts"] < REFRESH_SEC:
            return _cache["data"]
    fetch = fetch or market.get_funding_history
    missing = []
    for sym in SYMBOLS:
        try:
            rows = fetch(sym)
            if not rows:
                missing.append(sym)
                continue
            record(sym, rows)
        except Exception:  # noqa: BLE001, silent-ok — در خروجی به‌عنوانِ بی‌داده گزارش می‌شود
            log.exc("carry ledger", symbol=sym)
            missing.append(sym)
    ledger = _read()
    data = {"rule": "always long spot + short perpetual, same notional; shadow only, no orders",
            "status": "backtest_only", "tracking_start_ms": TRACKING_START_MS, "missing_data": missing,
            "assets": {s: stats([r for r in ledger if r["sym"] == s]) for s in SYMBOLS},
            "not_modelled": "basis at entry/exit, margin top-ups, exchange risk — an upper bound",
            "generated_at": time.time()}
    with _lock:
        _cache.update(ts=time.time(), data=data)
    return data


def view(force=False):
    return snapshot(force=force)
