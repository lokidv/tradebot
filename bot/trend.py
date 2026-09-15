# -*- coding: utf-8 -*-
"""روندِ روزانه — نامزدِ پژوهشی، اثبات‌نشده، فقط کاغذی.

از ۱۵ خانواده‌ای که روی دادهٔ اکتشاف (۲۰۱۸-۰۶ تا ۲۰۲۴-۰۹، پیش از پنجره‌های منجمد)
آزموده شد (``explore.py``، گزارشِ ``data/research/explore_20260915_1110.json``) فقط روندِ
روزانهٔ لانگ لبهٔ مثبت با کرانِ پایینِ مثبت داشت: شکستِ ۲۰روزه +۰٫۳۶R در هر معامله،
زیرِ هزینهٔ سنگین‌تر +۰٫۲۸R، مثبت در هر سه زیرِدوره ولی کوچک‌شونده. پس از تصحیحِ
جست‌وجوی چندگانه **معنادار نیست** (p_adj ۰٫۱۱). پس این مجوز نیست؛ فرضیه است.

راهِ صادقانه برای سنجیدنش، شواهدِ تازه است. پیش از دیدنِ هر داده‌ای ثابت شده‌اند (همین
فایل در git):

* قاعده‌ها — همان ``explore.VARIANTS``، با همان حلقهٔ ``explore.trend_paths``؛
* جهان — ۲۰ ارزِ بزرگِ پیش‌ثبت‌شده (``research.MAJORS``)، ثابت؛
* شروع — ``TRACKING_START_MS``، بی‌پوزیشن؛ از آنجا کندل‌به‌کندل.

هر معامله‌ای که این قاعده‌ها از آن تاریخ به بعد بسازند — بی‌استثنا و بی‌انتخاب — نمونهٔ
رو-به-جلوِ پاک است. ورود و خروج در دفترِ فقط-افزودنی ثبت می‌شوند تا با چرخشِ پنجرهٔ
داده یا بازنگریِ کندل عوض نشوند. این ماژول به gates دست نمی‌زند و هیچ سفارشی نمی‌دهد.
"""
import glob
import json
import os
import threading
import time

import numpy as np

import explore
import log
import market
import paths
import research
import stats

LEDGER_PATH = paths.data("trend_ledger.jsonl")
RULE_KEYS = ("T_don20_long", "T_don55_long", "T_sma100_long")
RULES = {v["key"]: v for v in explore.VARIANTS if v["key"] in RULE_KEYS}
# پیش از دادهٔ رو-به-جلو برگزیده شد: کمترین p_adj، مثبت در هر سه زیرِدوره، کمترین افت
PRIMARY = "T_don20_long"
TRACKING_START_MS = 1_789_430_400_000          # 2026-09-15 00:00 UTC
SYMBOLS = tuple(research.MAJORS)
KLINE_LIMIT = 420
REFRESH_SEC = 1800
_FIXED_UNIVERSE = [(0, frozenset(SYMBOLS))]

_lock = threading.Lock()
_cache = {"ts": 0.0, "data": None}


def _arrays(k):
    return {key: np.asarray(k[key], np.int64 if key == "t" else float) for key in ("t", "o", "h", "l", "c", "v")}


def load_daily(fetch=None, symbols=SYMBOLS):
    """کندل‌های روزانهٔ **بسته‌شده**. دادهٔ غیرِ بایننس (کندلِ ۱۶:۰۰ از OKX) کنار می‌رود."""
    fetch = fetch or (lambda s: market.get_klines(s, "1d", KLINE_LIMIT))
    out, missing = {}, []
    for s in symbols:
        try:
            k = _arrays(fetch(s))
        except Exception:  # noqa: BLE001, silent-ok — در خروجی به‌عنوانِ «بی‌داده» گزارش می‌شود
            missing.append(s)
            continue
        if not len(k["t"]) or np.any(k["t"] % explore.DAY_MS):
            missing.append(s)
            continue
        out[s] = k
    return out, missing


def state_now(daily, key):
    """نمای امروزِ قاعده روی هر ارز — فقط برای نمایش، **نه** برای دفتر.

    پوزیشنی که پیش از شروعِ ردیابی باز شده هم نشان داده می‌شود (``counted=False``)،
    چون «قاعده الان چه می‌گوید» برای کاربر همان‌قدر مهم است که کارنامهٔ رو-به-جلو.
    برای ارزِ بی‌پوزیشن: سطحی که بستهٔ امروز باید از آن بگذرد تا سیگنال بدهد.
    """
    spec = RULES[key]
    rows = []
    for sym, k in daily.items():
        n = len(k["c"])
        if n < 30:
            continue
        last = float(k["c"][-1])
        row = {"sym": sym, "last": last, "in_position": False}
        for i, e, entry, R, res in explore.trend_paths(sym, k, spec, _FIXED_UNIVERSE):
            if res is not None and res["outcome"] == "end_of_data":
                row.update(in_position=True, entry_ts=int(k["t"][e]), entry_px=entry,
                           stop=float(res["stop"]), open_r=spec["side"] * (last - entry) / R,
                           counted=int(k["t"][i]) >= TRACKING_START_MS,
                           exiting_next_open=bool(res.get("exit_pending")))
            elif res is None:
                row.update(signal_today=True)
        if not row["in_position"] and spec["rule"] == "donchian":
            # سیگنالِ فردا: بستهٔ کندلِ جاری بالای سقفِ همین ۲۰ کندلِ بسته‌شده
            trigger = float(np.max(k["h"][n - spec["n"]:n]))
            row.update(trigger=trigger, distance_pct=(trigger / last - 1) * 100)
        rows.append(row)
    rows.sort(key=lambda r: (not r["in_position"], not r.get("signal_today", False),
                             r.get("distance_pct", 0.0)))
    return rows


def evaluate(daily, start_ms=TRACKING_START_MS):
    """برای هر قاعده: پوزیشن‌های باز، سیگنال‌های امروز و معامله‌های بسته — فقط رو-به-جلو."""
    out = {}
    for key, spec in RULES.items():
        rows = {"open": [], "signals": [], "closed": []}
        for sym, k in daily.items():
            start_idx = int(np.searchsorted(k["t"], start_ms))
            if start_idx >= len(k["t"]):
                continue
            for i, e, entry, R, res in explore.trend_paths(sym, k, spec, _FIXED_UNIVERSE, start_idx):
                sig_ts = int(k["t"][i])
                if res is None:
                    close = float(k["c"][i])
                    rows["signals"].append({"sym": sym, "signal_ts": sig_ts, "close": close,
                                            "stop_distance": R, "risk_pct": R / close * 100})
                    continue
                trade_id = f"{key}:{sym}:{sig_ts}"
                if res["outcome"] == "end_of_data":
                    last = float(k["c"][-1])
                    rows["open"].append({
                        "id": trade_id, "sym": sym, "signal_ts": sig_ts, "entry_ts": int(k["t"][e]),
                        "entry_px": entry, "stop": float(res["stop"]), "last": last,
                        "open_r": spec["side"] * (last - entry) / R, "risk_pct": R / entry * 100,
                        "exiting_next_open": bool(res.get("exit_pending"))})
                    continue
                tr = explore._trade(sym, "1d", spec["side"], i, e, res, entry, R, k)
                tr.update(id=trade_id, rule=key, entry_px=entry, exit_px=res["exit_px"])
                rows["closed"].append(tr)
        out[key] = rows
    return out


# ───────────────────────── دفترِ فقط-افزودنی ─────────────────────────
def _read():
    if not os.path.exists(LEDGER_PATH):
        return []
    rows = []
    with open(LEDGER_PATH, "r", encoding="utf-8") as f:
        for line in f:
            try:
                rows.append(json.loads(line))
            except ValueError:
                continue
    return rows


def record(state):
    """ورودِ تازه و خروجِ تازه را یک‌بار و برای همیشه ثبت کن. خروجی: تعدادِ ردیف‌های تازه."""
    seen = {(r.get("kind"), r.get("id")) for r in _read()}
    new = []
    for key, rows in state.items():
        for pos in rows["open"]:
            if ("entry", pos["id"]) not in seen:
                new.append({"kind": "entry", "rule": key, **{k: pos[k] for k in
                            ("id", "sym", "signal_ts", "entry_ts", "entry_px", "risk_pct")}})
        for tr in rows["closed"]:
            if ("entry", tr["id"]) not in seen:
                new.append({"kind": "entry", "rule": key, "id": tr["id"], "sym": tr["sym"],
                            "signal_ts": tr["ts"], "entry_ts": tr["entry_ts"],
                            "entry_px": tr["entry_px"], "risk_pct": tr["risk_pct"]})
            if ("exit", tr["id"]) not in seen:
                new.append({"kind": "exit", **tr})
    if new:
        os.makedirs(os.path.dirname(LEDGER_PATH), exist_ok=True)
        now = time.time()
        with open(LEDGER_PATH, "a", encoding="utf-8") as f:
            for r in new:
                f.write(json.dumps({**r, "logged_at": now}, ensure_ascii=False, default=float) + "\n")
    return len(new)


def forward_stats(rows=None):
    rows = _read() if rows is None else rows
    exits = [r for r in rows if r.get("kind") == "exit"]
    out = {}
    for key in RULES:
        mine = [r for r in exits if r.get("rule") == key]
        vals = [float(r["net_r"]) for r in mine]
        s = stats.summarize(vals, [int(r["entry_ts"]) for r in mine], explore.TREND_BLOCK_MS,
                            alpha=explore.ALPHA, B=500) if vals else {"n": 0}
        s["sum_r"] = round(float(np.sum(vals)), 3) if vals else 0.0
        out[key] = s
    return out


def dev_evidence():
    """آمارِ اکتشاف از آخرین گزارشِ متعهدشده — همان عددهایی که این قاعده‌ها را برگزید."""
    files = sorted(glob.glob(os.path.join(explore.OUT_DIR, "explore_*.json")))
    if not files:
        return None
    with open(files[-1], "r", encoding="utf-8") as f:
        rep = json.load(f)
    rules = {}
    for key in RULES:
        v = rep.get("variants", {}).get(key) or {}
        s, rb = v.get("stats") or {}, (rep.get("robustness") or {}).get(key) or {}
        mean, sd = s.get("mean"), s.get("sd")
        rules[key] = {
            "n": s.get("n"), "n_eff": s.get("n_eff"), "mean": mean, "lcb": s.get("lcb"),
            "profit_factor": s.get("profit_factor"), "p_adj": (v.get("romano_wolf") or {}).get("p_adj"),
            "verdict": v.get("verdict"), "stressed_mean": (rb.get("top20") or {}).get("mean_stressed"),
            "btc_eth_mean": (rb.get("btc_eth") or {}).get("mean"), "periods": rb.get("periods"),
            "account": rb.get("account_top20"),
            # چند معاملهٔ **مستقل** تا اثرِ دیده‌شده با توانِ ۸۰٪ تأیید شود
            "n_eff_to_confirm": stats.power_sample_size(mean, sd=sd) if mean and sd and mean > 0 else None,
        }
    return {"report": os.path.basename(files[-1]), "cutoff_ms": rep.get("cutoff_ms"),
            "btc_buy_and_hold": (rep.get("robustness") or {}).get("btc_buy_and_hold"), "rules": rules}


def snapshot(force=False, fetch=None):
    """برای ‎/api/trend‎ — کشِ ۳۰ دقیقه‌ای؛ کندلِ روزانه فقط روزی یک‌بار عوض می‌شود."""
    with _lock:
        if not force and _cache["data"] is not None and time.time() - _cache["ts"] < REFRESH_SEC:
            return _cache["data"]
    daily, missing = load_daily(fetch)
    state = evaluate(daily)
    try:
        added = record(state)
    except OSError:
        log.exc("trend ledger write")
        added = 0
    last_bar = max((int(k["t"][-1]) for k in daily.values()), default=None)
    data = {
        "status": "research_candidate",
        "authorized": False,
        "primary": PRIMARY,
        "tracking_start_ms": TRACKING_START_MS,
        "symbols": list(SYMBOLS),
        "missing_data": missing,
        "last_closed_bar_ms": last_bar,
        "rules": {key: {"spec": RULES[key], "open": rows["open"], "signals": rows["signals"],
                        "closed_recent": sorted(rows["closed"], key=lambda r: -r["exit_ts"])[:10]}
                  for key, rows in state.items()},
        "now": state_now(daily, PRIMARY),
        "forward": forward_stats(),
        "evidence": dev_evidence(),
        "ledger_rows_added": added,
        "generated_at": time.time(),
    }
    with _lock:
        _cache.update(ts=time.time(), data=data)
    return data
