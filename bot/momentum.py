# -*- coding: utf-8 -*-
"""مومنتومِ ۲۸روزهٔ BTC و ETH — «در بازار یا نقد»، تصمیم فقط دوشنبه‌ها.

پژوهش: timing.py، پیش‌ثبت prereg_btc_eth_timing.json، گزارشِ btc_eth_timing_*.json. در هر دو نیمهٔ
داده و روی هر دو ارز مثبت بود («سازگار»)، ولی آستانهٔ معناداریِ ازپیش‌گفته (p<0.10) را نگذراند
(BTC p=0.11). مزیتش بیشتر دوری از بازارِ نزولی است، نه سودِ بیشتر در بازارِ صعودی.

قاعده (بدونِ هیچ پارامترِ بهینه‌شده): دوشنبه ۰۰:۰۰ UTC، اگر بستهٔ یکشنبه بالاتر از بستهٔ ۲۸ روز
قبلش باشد، برای یک هفته خرید؛ وگرنه نقد. این ماژول به gates دست نمی‌زند و سفارشی نمی‌دهد.
"""
import glob
import json
import os
import threading
import time

import numpy as np

import log
import market
import paths

SYMBOLS = ("BTCUSDT", "ETHUSDT")
LOOKBACK_DAYS = 28
DAY_MS = 86_400_000
WEEK_MS = 7 * DAY_MS
TRACKING_START_MS = 1_790_553_600_000        # دوشنبه 2026-09-28 00:00 UTC — اولین تصمیم پس از ثبت
LEDGER_PATH = paths.data("momentum_ledger.jsonl")
DEMO_STRATEGY = "tsmom28"
DEMO_ALLOC_RANGE = (5.0, 50.0)               # درصدِ موجودیِ دمو برای هر ارز
DEMO_DISASTER_STOP = 0.5                     # فقط محافظِ فاجعه (−۵۰٪)؛ خروجِ واقعی تصمیمِ دوشنبه است
REFRESH_SEC = 600
_lock = threading.Lock()
_cache = {"ts": 0.0, "data": None}


def monday_on_or_before(ms):
    day0 = ms // DAY_MS * DAY_MS
    wd = time.gmtime(day0 / 1000).tm_wday
    return day0 - wd * DAY_MS


def decision_at(t, c, monday_ms):
    """تصمیمِ دوشنبهٔ ``monday_ms``: بستهٔ یکشنبه (کندلی که ``monday − ۱ روز`` باز شد) در برابرِ
    بستهٔ ۲۸ روز قبلش. ``None`` یعنی داده ناکافی."""
    idx = {int(x): i for i, x in enumerate(t)}
    i = idx.get(int(monday_ms - DAY_MS))
    j = idx.get(int(monday_ms - DAY_MS - LOOKBACK_DAYS * DAY_MS))
    if i is None or j is None:
        return None
    return {"in_market": bool(c[i] > c[j]), "sunday_close": float(c[i]), "ref_close": float(c[j]),
            "change_pct": round((c[i] / c[j] - 1) * 100, 2), "monday_ms": int(monday_ms)}


def state(t, c, now_ms):
    """وضعیتِ این هفته، تصمیمِ هفتهٔ بعد اگر هفته همین حالا بسته می‌شد، و تاریخچهٔ تصمیم‌ها."""
    t = np.asarray(t, np.int64)
    c = np.asarray(c, float)
    this_monday = monday_on_or_before(now_ms)
    cur = decision_at(t, c, this_monday)
    history, m = [], this_monday
    while True:
        d = decision_at(t, c, m)
        if d is None:
            break
        history.append(d)
        m -= WEEK_MS
    history.reverse()
    since = None
    if cur is not None:
        for d in reversed(history):
            if d["in_market"] != cur["in_market"]:
                break
            since = d["monday_ms"]
    last = len(c) - 1                      # آخرین بستهٔ موجود
    ref = int(t[last]) - LOOKBACK_DAYS * DAY_MS
    k = int(np.searchsorted(t, ref))
    provisional = None
    if k < len(t) and int(t[k]) == ref:
        provisional = {"in_market": bool(c[last] > c[k]), "change_pct": round((c[last] / c[k] - 1) * 100, 2),
                       "threshold": float(c[k]), "as_of_ms": int(t[last])}
    switches = [d for a, d in zip(history, history[1:]) if a["in_market"] != d["in_market"]]
    return {"current": cur, "in_market_since_ms": since, "next_decision_ms": this_monday + WEEK_MS,
            "provisional_next": provisional, "recent_switches": switches[-6:],
            "switches_last_year": sum(1 for s in switches if s["monday_ms"] >= now_ms - 365 * DAY_MS)}


# ───────────────────────── دفترِ رو-به-جلو ─────────────────────────
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


def record(sym, t, c):
    """هر تصمیمِ دوشنبه از TRACKING_START به بعد یک‌بار؛ بازدهِ هفتهٔ قبل هم (بستهٔ یکشنبه به یکشنبه)."""
    seen = {(r["sym"], r["monday_ms"]) for r in _read()}
    t = np.asarray(t, np.int64)
    c = np.asarray(c, float)
    new, m = [], TRACKING_START_MS
    while True:
        d = decision_at(t, c, m)
        if d is None:
            break
        if (sym, m) not in seen:
            prev = decision_at(t, c, m - WEEK_MS)
            week_ret = (d["sunday_close"] / prev["sunday_close"] - 1) if prev else None
            new.append({"sym": sym, "monday_ms": m, "in_market": d["in_market"],
                        "prev_in_market": prev["in_market"] if prev else None,
                        "prev_week_market_ret": week_ret, "logged_at": time.time()})
        m += WEEK_MS
    if new:
        os.makedirs(os.path.dirname(LEDGER_PATH), exist_ok=True)
        with open(LEDGER_PATH, "a", encoding="utf-8") as f:
            for r in new:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return len(new)


def forward_stats(rows=None):
    """کارنامهٔ رو-به-جلو: بازدهِ هفته‌هایی که قاعده در بازار بود، منهای ۰٫۲٪ در هر جابه‌جایی."""
    rows = _read() if rows is None else rows
    out = {}
    for sym in SYMBOLS:
        mine = sorted((r for r in rows if r["sym"] == sym), key=lambda r: r["monday_ms"])
        rets, prev = [], None
        for r in mine:
            if r.get("prev_in_market") is not None and r.get("prev_week_market_ret") is not None:
                g = r["prev_week_market_ret"] if r["prev_in_market"] else 0.0
                rets.append(g)
            if prev is not None and prev != r["in_market"]:
                rets.append(-0.002)
            prev = r["in_market"]
        out[sym] = {"weeks": len(mine), "cum_return_pct": round((float(np.prod([1 + x for x in rets])) - 1) * 100, 2)
                    if rets else 0.0}
    return out


# ───────────────────────── شواهد ─────────────────────────
def evidence():
    files = sorted(glob.glob(paths.data("research", "btc_eth_timing_*.json")))
    if not files:
        return None
    with open(files[-1], encoding="utf-8") as f:
        rep = json.load(f)
    out = {"report": os.path.basename(files[-1])}
    for sym in SYMBOLS:
        v = (rep.get("results") or {}).get(f"{sym}:H3_TSMOM28") or {}
        conf, disc = v.get("confirmation") or {}, v.get("discovery") or {}
        out[sym] = {"verdict": v.get("verdict"), "p_adj": (v.get("romano_wolf_discovery") or {}).get("p_adj"),
                    "confirmation": {"strategy": conf.get("strategy"), "buy_and_hold": conf.get("buy_and_hold"),
                                     "time_in_market_pct": conf.get("time_in_market_pct")},
                    "discovery": {"strategy": disc.get("strategy"), "buy_and_hold": disc.get("buy_and_hold")}}
    return out


# ───────────────────────── دمو ─────────────────────────
def _positions():
    import paper
    return [p for p in paper.list_positions()["open"] if p.get("strategy") == DEMO_STRATEGY]


def open_demo(sym, alloc_pct=20.0, snap=None, price=None):
    """خرید در دمو فقط وقتی قاعده این هفته «در بازار» است. حجم = درصدی از موجودیِ دمو (بی‌اهرم)."""
    import paper
    snap = snap or snapshot()
    st = (snap.get("assets") or {}).get(sym) or {}
    cur = st.get("current") or {}
    if not cur.get("in_market"):
        raise ValueError(f"قاعده این هفته برای {sym} «نقد» است — خریدی در کار نیست")
    if any(p["symbol"] == sym for p in _positions()):
        raise ValueError(f"پوزیشنِ دموی مومنتوم روی {sym} از قبل باز است")
    price = float(price if price is not None else market.last_price(sym))
    alloc = min(max(float(alloc_pct), DEMO_ALLOC_RANGE[0]), DEMO_ALLOC_RANGE[1])
    w = paper.wallet_summary()
    equity = float(w.get("equity") or w.get("balance") or 0.0)
    if equity <= 0:
        raise ValueError("موجودیِ دمو صفر است")
    return paper.open_position(sym, "1d", "long", price, price * DEMO_DISASTER_STOP, None,
                               round(equity * alloc / 100.0, 2), 365 * 1440,
                               grade="مومنتوم ۲۸روزه", opened_by="user", cost_pct=0.4,
                               strategy=DEMO_STRATEGY, market="spot",
                               ref={"rule": "tsmom28", "monday_ms": cur.get("monday_ms")})


def sync_demo(snap=None, bar_fn=None):
    """وقتی تصمیمِ دوشنبه «نقد» شد، پوزیشنِ دمو در openِ همان دوشنبه بسته می‌شود (قیمتِ تصمیم)."""
    import paper
    snap = snap or snapshot()
    bar_fn = bar_fn or (lambda s: market.current_bar(s, "1d"))
    closed = 0
    for p in _positions():
        cur = ((snap.get("assets") or {}).get(p["symbol"]) or {}).get("current") or {}
        if cur and cur.get("in_market") is False:
            try:
                bar = bar_fn(p["symbol"])
                # اگر همان دوشنبه است، قیمتِ تصمیم = openِ دوشنبه؛ دیرتر = قیمتِ همین لحظه
                px = float(bar["o"]) if int(bar["t"]) == int(cur["monday_ms"]) else float(bar["c"])
            except Exception:  # noqa: BLE001
                log.exc("momentum demo exit price", symbol=p["symbol"])
                continue
            if paper.close_with(p["id"], px, "مومنتوم: تصمیمِ دوشنبه «نقد»"):
                closed += 1
    return {"closed": closed}


def snapshot(force=False, fetch=None, now_ms=None):
    with _lock:
        if not force and fetch is None and _cache["data"] is not None and time.time() - _cache["ts"] < REFRESH_SEC:
            return _cache["data"]
    fetch = fetch or (lambda s: market.get_klines(s, "1d", 420))
    now_ms = int(now_ms if now_ms is not None else time.time() * 1000)
    assets, missing = {}, []
    for sym in SYMBOLS:
        try:
            k = fetch(sym)
            assets[sym] = state(k["t"], k["c"], now_ms)
            try:
                record(sym, k["t"], k["c"])
            except OSError:
                log.exc("momentum ledger")
        except Exception:  # noqa: BLE001, silent-ok — در خروجی به‌عنوانِ بی‌داده گزارش می‌شود
            missing.append(sym)
    data = {"rule": "long for the week if Sunday close > close 28 days earlier; decided Mondays 00:00 UTC",
            "status": "consistent_not_proven", "tracking_start_ms": TRACKING_START_MS,
            "assets": assets, "missing_data": missing, "evidence": evidence(),
            "forward": forward_stats(), "generated_at": time.time()}
    with _lock:
        _cache.update(ts=time.time(), data=data)
    return data


def view(force=False):
    out = dict(snapshot(force=force))
    out["demo_open"] = sorted({p["symbol"] for p in _positions()})
    return out
