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


# پیوستن به روندِ در جریان: روی دادهٔ اکتشاف، ۵ و ۱۰ روز پس از شکست همان‌قدر خوب بود که خودِ
# شکست (+۰٫۴۱ و +۰٫۴۴R)، ولی ۲۰ روز دیر دیگر نه (+۰٫۱۷R، کرانِ پایینِ منفی) — explore.join_report
JOIN_MAX_DAYS = 10
CATEGORY_ORDER = {"new": 0, "join": 1, "late": 2, "watch": 3}


def state_now(daily, key):
    """نمای امروزِ قاعده روی هر ارز — برای تصمیمِ امروز، **نه** برای دفتر.

    دسته‌ها: ``new`` شکست روی آخرین بستهٔ روزانه (ورود در openِ کندلِ جاری)؛ ``join`` روندی
    که حداکثر ``JOIN_MAX_DAYS`` روز پیش شروع شده — ورود در قیمتِ فعلی با حدضررِ فعلیِ
    قاعده؛ ``late`` روندِ قدیمی‌تر (فقط حدضررِ امروز برای کسی که از قبل دارد)؛ ``watch``
    بی‌پوزیشن، با سطحی که بستهٔ روزانه باید از آن بگذرد. پوزیشنی که پیش از شروعِ ردیابی
    باز شده ``counted=False`` است — در کارنامهٔ رو-به-جلو شمرده نمی‌شود.
    """
    spec = RULES[key]
    rows = []
    for sym, k in daily.items():
        n = len(k["c"])
        if n < 30:
            continue
        last = float(k["c"][-1])
        row = {"sym": sym, "last": last, "in_position": False, "category": "watch"}
        for i, e, entry, R, res in explore.trend_paths(sym, k, spec, _FIXED_UNIVERSE):
            if res is not None and res["outcome"] == "end_of_data":
                stop = float(res["stop"])
                days = n - e                   # پیوستن الان = openِ کندلِ جاری، n−e روز پس از ورودِ قاعده
                joinable = 1 <= days <= JOIN_MAX_DAYS and last > stop and not res.get("exit_pending")
                row.update(in_position=True, entry_ts=int(k["t"][e]), entry_px=entry, stop=stop,
                           open_r=spec["side"] * (last - entry) / R, days_in=days,
                           stop_distance_pct=(last - stop) / last * 100,
                           counted=int(k["t"][i]) >= TRACKING_START_MS,
                           exiting_next_open=bool(res.get("exit_pending")),
                           category="join" if joinable else "late")
            elif res is None:
                # شکست روی آخرین بسته: ورود در openِ کندلِ جاری (~بستهٔ دیروز)، حدضرر R پایین‌تر
                row.update(signal_today=True, category="new", stop=last - R,
                           stop_distance_pct=R / last * 100)
        if row["category"] == "watch" and spec["rule"] == "donchian":
            trigger = float(np.max(k["h"][n - spec["n"]:n]))   # سقفِ همین ۲۰ کندلِ بسته‌شده
            row.update(trigger=trigger, distance_pct=(trigger / last - 1) * 100)
        rows.append(row)
    rows.sort(key=lambda r: (CATEGORY_ORDER[r["category"]], r.get("days_in", 0),
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
            "win_rate": s.get("win_rate"),
            "profit_factor": s.get("profit_factor"), "p_adj": (v.get("romano_wolf") or {}).get("p_adj"),
            "verdict": v.get("verdict"), "stressed_mean": (rb.get("top20") or {}).get("mean_stressed"),
            "btc_eth_mean": (rb.get("btc_eth") or {}).get("mean"), "periods": rb.get("periods"),
            "account": rb.get("account_top20"),
            # چند معاملهٔ **مستقل** تا اثرِ دیده‌شده با توانِ ۸۰٪ تأیید شود
            "n_eff_to_confirm": stats.power_sample_size(mean, sd=sd) if mean and sd and mean > 0 else None,
        }
    joins = None
    jfiles = sorted(glob.glob(os.path.join(explore.OUT_DIR, "join_*.json")))
    if jfiles:
        with open(jfiles[-1], "r", encoding="utf-8") as f:
            jr = json.load(f)
        joins = {"report": os.path.basename(jfiles[-1]),
                 "breakout_spot_mean": (jr.get("breakout") or {}).get("mean_spot"),
                 "spot_round_trip_pct": jr.get("spot_round_trip_pct"),
                 "n_trials_project": jr.get("n_trials_project"),
                 # p_adj روی همین تعداد فرضیه که روی دادهٔ اکتشاف آزموده شد (بی ۸ فرضیهٔ آزمونِ منجمد)
                 "rw_family_size": (jr["n_trials_project"] - explore.PRIOR_HYPOTHESES
                                    if jr.get("n_trials_project") else None),
                 "offsets": {k: {"n": s.get("n"), "mean": s.get("mean"), "lcb": s.get("lcb"),
                                 "mean_spot": s.get("mean_spot"),
                                 "p_adj": (s.get("romano_wolf") or {}).get("p_adj")}
                             for k, s in (jr.get("joins") or {}).items()}}
    return {"report": os.path.basename(files[-1]), "cutoff_ms": rep.get("cutoff_ms"),
            "btc_buy_and_hold": (rep.get("robustness") or {}).get("btc_buy_and_hold"), "rules": rules,
            "joins": joins, "join_max_days": JOIN_MAX_DAYS}


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
