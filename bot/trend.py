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
# گسترش به ۵۰ ارز (explore.universe_report، data/research/universe_20260915_1657.json): رتبه‌های
# ۲۱ تا ۵۰ جداگانه معیارِ ازپیش‌گفته را داشتند (+۰٫۳۹R، کرانِ پایین +۰٫۱۴، مثبت در هر دو نیمه و زیرِ
# فشارِ هزینه). فهرست **ثابت** است، ۲۰۲۶-۰۹-۱۵ پیش از بسته‌شدنِ کندلِ آن روز فقط با حجمِ دلاریِ ۳۰
# روزهٔ اسپاتِ بایننس انتخاب شد: ۳۰ ارزِ بزرگ‌ترِ غیرِ اصلی با دستِ‌کم ۴۰۰ روز سابقه (سهامِ
# توکنی و ارزهای خیلی تازه بیرون می‌مانند)، بی‌استیبل/میخ‌شده/رپ‌شده، و بی‌طلا (PAXG: دارایی
# دیگری است). کارنامهٔ رو-به-جلوی این‌ها جدا از ۲۰ ارزِ اصلی شمرده می‌شود.
EXT_SYMBOLS = ("ZECUSDT", "SUIUSDT", "ENAUSDT", "TRUMPUSDT", "PEPEUSDT", "TAOUSDT", "WLDUSDT",
               "DASHUSDT", "AAVEUSDT", "BMTUSDT", "XLMUSDT", "ONDOUSDT", "PENGUUSDT", "FETUSDT",
               "INJUSDT", "HBARUSDT", "POLUSDT", "THEUSDT", "LSKUSDT", "ICPUSDT", "ZROUSDT",
               "RAYUSDT", "ETHFIUSDT", "ZENUSDT", "VIRTUALUSDT", "CAKEUSDT", "SHIBUSDT", "OPUSDT",
               "REZUSDT", "GPSUSDT")
ALL_SYMBOLS = SYMBOLS + EXT_SYMBOLS
KLINE_LIMIT = 420
REFRESH_SEC = 1800
_FIXED_UNIVERSE = [(0, frozenset(ALL_SYMBOLS))]


def universe_of(sym):
    return "majors" if sym in SYMBOLS else "top50"

_lock = threading.Lock()
_cache = {"ts": 0.0, "data": None}


def _arrays(k):
    return {key: np.asarray(k[key], np.int64 if key == "t" else float) for key in ("t", "o", "h", "l", "c", "v")}


def load_daily(fetch=None, symbols=ALL_SYMBOLS):
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
        atr = float(explore.engine.atr(k["h"], k["l"], k["c"], 20)[-1])
        row = {"sym": sym, "last": last, "in_position": False, "category": "watch",
               "universe": universe_of(sym), "atr": atr}
        exits = []
        for i, e, entry, R, res in explore.trend_paths(sym, k, spec, _FIXED_UNIVERSE):
            if res is not None and res["outcome"] == "end_of_data":
                stop = float(res["stop"])
                days = n - e                   # پیوستن الان = openِ کندلِ جاری، n−e روز پس از ورودِ قاعده
                joinable = 1 <= days <= JOIN_MAX_DAYS and last > stop and not res.get("exit_pending")
                row.update(in_position=True, entry_ts=int(k["t"][e]), rule_entry_ts=int(k["t"][e]),
                           entry_px=entry, stop=stop,
                           open_r=spec["side"] * (last - entry) / R, days_in=days,
                           stop_distance_pct=(last - stop) / last * 100,
                           counted=int(k["t"][i]) >= TRACKING_START_MS,
                           exiting_next_open=bool(res.get("exit_pending")),
                           category="join" if joinable else "late")
            elif res is None:
                # شکست روی آخرین بسته: ورود در openِ کندلِ جاری (~بستهٔ دیروز)، حدضرر R پایین‌تر
                row.update(signal_today=True, category="new", stop=last - R,
                           rule_entry_ts=int(k["t"][-1]) + explore.DAY_MS,
                           stop_distance_pct=R / last * 100)
            else:
                exits.append({"entry_ts": int(k["t"][e]), "exit_ts": int(k["t"][res["exit_idx"]]),
                              "exit_px": float(res["exit_px"]), "outcome": res["outcome"]})
        row["recent_exits"] = exits[-3:]       # برای خروجِ هم‌پای قاعده در پوزیشن‌های دمو
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
                uni = universe_of(sym)
                if res is None:
                    close = float(k["c"][i])
                    rows["signals"].append({"sym": sym, "signal_ts": sig_ts, "close": close,
                                            "stop_distance": R, "risk_pct": R / close * 100,
                                            "universe": uni})
                    continue
                trade_id = f"{key}:{sym}:{sig_ts}"
                if res["outcome"] == "end_of_data":
                    last = float(k["c"][-1])
                    rows["open"].append({
                        "id": trade_id, "sym": sym, "signal_ts": sig_ts, "entry_ts": int(k["t"][e]),
                        "entry_px": entry, "stop": float(res["stop"]), "last": last,
                        "open_r": spec["side"] * (last - entry) / R, "risk_pct": R / entry * 100,
                        "exiting_next_open": bool(res.get("exit_pending")), "universe": uni})
                    continue
                tr = explore._trade(sym, "1d", spec["side"], i, e, res, entry, R, k)
                tr.update(id=trade_id, rule=key, entry_px=entry, exit_px=res["exit_px"], universe=uni)
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
                new.append({"kind": "entry", "rule": key, "universe": universe_of(pos["sym"]),
                            **{k: pos[k] for k in ("id", "sym", "signal_ts", "entry_ts", "entry_px", "risk_pct")}})
        for tr in rows["closed"]:
            if ("entry", tr["id"]) not in seen:
                new.append({"kind": "entry", "rule": key, "id": tr["id"], "sym": tr["sym"],
                            "universe": universe_of(tr["sym"]),
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
    """کارنامهٔ رو-به-جلو. ``rule`` = ۲۰ ارزِ اصلیِ پیش‌ثبت‌شده؛ ``rule@top50`` = گسترش، جدا."""
    rows = _read() if rows is None else rows
    exits = [r for r in rows if r.get("kind") == "exit"]
    out = {}
    for key in RULES:
        for uni, name in (("majors", key), ("top50", f"{key}@top50")):
            mine = [r for r in exits if r.get("rule") == key
                    and (r.get("universe") or universe_of(r.get("sym", ""))) == uni]
            vals = [float(r["net_r"]) for r in mine]
            s = stats.summarize(vals, [int(r["entry_ts"]) for r in mine], explore.TREND_BLOCK_MS,
                                alpha=explore.ALPHA, B=500) if vals else {"n": 0}
            s["sum_r"] = round(float(np.sum(vals)), 3) if vals else 0.0
            out[name] = s
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
    uni = None
    ufiles = sorted(glob.glob(os.path.join(explore.OUT_DIR, "universe_*.json")))
    if ufiles:
        with open(ufiles[-1], "r", encoding="utf-8") as f:
            ur = json.load(f)
        pick = lambda s: {k: (s or {}).get(k) for k in ("n", "mean", "lcb", "mean_spot", "mean_stressed",
                                                        "first_half_mean", "second_half_mean", "win_rate")}
        uni = {"report": os.path.basename(ufiles[-1]), "top20": pick(ur.get("top20")),
               "top50": pick(ur.get("top50")), "ranks_21_50": pick(ur.get("ranks_21_50")),
               "p_adj_top50": ((ur.get("romano_wolf") or {}).get("U_top50") or {}).get("p_adj"),
               "rw_family_size": ur.get("rw_family_size"),
               "extend_to_top50": ur.get("extend_to_top50")}
    behaviour = None
    ffiles = sorted(glob.glob(os.path.join(explore.OUT_DIR, "filters_*.json")))
    if ffiles:
        with open(ffiles[-1], "r", encoding="utf-8") as f:
            fr = json.load(f)
        tg = fr.get("profit_targets_same_entries") or {}
        f2 = (fr.get("filters") or {}).get("F2_join_cushion_1_5_atr") or {}
        behaviour = {
            "report": os.path.basename(ffiles[-1]),
            "profile": fr.get("profile_entries"),
            "hold_mean": (fr.get("baseline_entries") or {}).get("mean"),
            "target_means": {k: (v or {}).get("mean") for k, v in tg.items()},
            "target_1r_second_half": (tg.get("target_1R") or {}).get("second_half_mean"),
            "filters_adopted": [k for k, v in (fr.get("filters") or {}).items() if v.get("adopt")],
            "filters_tested": list((fr.get("filters") or {}).keys()),
            "tight_win_rate": (f2.get("excluded") or {}).get("win_rate"),
            "normal_win_rate": (f2.get("kept") or {}).get("win_rate"),
            "account_top50": {k: {x: v.get(x) for x in ("cagr_pct", "max_drawdown_pct", "trades_taken")}
                              for k, v in (fr.get("account_top50") or {}).items()},
        }
    oos = None
    ofiles = sorted(glob.glob(os.path.join(explore.OUT_DIR, "trend_oos_*.json")))
    if ofiles:
        with open(ofiles[-1], "r", encoding="utf-8") as f:
            orep = json.load(f)
        g = lambda s: {k: (s or {}).get(k) for k in ("n", "mean", "lcb90", "win_rate", "first_half_mean",
                                                     "second_half_mean", "by_year")}
        oos = {"report": os.path.basename(ofiles[-1]), "verdict": orep.get("verdict"),
               "window_ms": orep.get("window_ms"), "top50": g(orep.get("primary_top50")),
               "top20": g(orep.get("top20")), "ranks_21_50": g(orep.get("ranks_21_50")),
               "account": orep.get("account_demo_rules"), "btc": orep.get("btc_buy_and_hold")}
    return {"report": os.path.basename(files[-1]), "cutoff_ms": rep.get("cutoff_ms"),
            "btc_buy_and_hold": (rep.get("robustness") or {}).get("btc_buy_and_hold"), "rules": rules,
            "joins": joins, "join_max_days": JOIN_MAX_DAYS, "universe": uni, "behaviour": behaviour,
            "oos": oos}


# ───────────────────────── پوزیشنِ دمو برای سیگنالِ امروز ─────────────────────────
DEMO_STRATEGY = "trend20"
DEMO_MAX_HOLD_MIN = 120 * 1440                 # همان حدِ زمانیِ ۱۲۰ روزهٔ قاعده
DEMO_RISK_RANGE = (0.1, 2.0)                   # درصدِ موجودی در هر معامله


DEMO_MAX_HEAT_PCT = 4.0                        # سقفِ مجموعِ ریسکِ بازِ پوزیشن‌های روند (٪ موجودی)
LIVE_TTL_SEC = 60
# پیوستن با فاصلهٔ کمتر از ۱٫۵ ATR تا حدضرر: روی دادهٔ اکتشاف فقط ~۲۴٪ برد (در برابرِ ~۴۰٪) و کرانِ
# پایینِ منفی — فیلترِ حذف از آزمونِ ازپیش‌گفته رد شد (explore.filter_study)، پس حذف نمی‌شود؛ فقط هشدار.
TIGHT_CUSHION_ATR = explore.MIN_CUSHION_ATR
_live_cache = {}


def live_bar(sym, bar_fn=None):
    """کندلِ روزانهٔ جاری (قیمتِ همین لحظه + کفِ امروز) با کشِ ۶۰ ثانیه‌ای."""
    now = time.time()
    hit = _live_cache.get(sym)
    if bar_fn is None and hit and now - hit[0] < LIVE_TTL_SEC:
        return hit[1]
    bar = (bar_fn or (lambda s: market.current_bar(s, "1d")))(sym)
    _live_cache[sym] = (now, bar)
    return bar


def with_live(row, bar):
    """سیگنالِ روزانه + واقعیتِ همین لحظه.

    کارت قبلاً «ورود ~بستهٔ دیروز» را نشان می‌داد؛ کاربری که ۱۷ ساعت بعد خرید، ۹٪ پایین‌تر و با نصفِ
    فاصله تا حدضرر وارد شد و هیچ‌جا این را ندید. و اگر کفِ همین امروز حدضررِ قاعده را زده باشد،
    قاعده عملاً بیرون است — سیگنال مرده، هرچند تا بسته‌شدنِ کندل در دادهٔ روزانه دیده نمی‌شود.
    """
    out = dict(row)
    stop, live = float(row["stop"]), float(bar["c"])
    day_open, day_low = float(bar["o"]), float(bar["l"])
    breached = day_low <= stop or live <= stop
    cushion = (live - stop) / row["atr"] if row.get("atr") else None
    out.update(live=live, day_open=day_open, day_low=day_low, live_bar_ts=int(bar["t"]),
               moved_from_open_pct=(live / day_open - 1) * 100 if day_open > 0 else None,
               live_stop_distance_pct=(live - stop) / live * 100 if live > 0 else None,
               live_cushion_atr=cushion, valid=not breached,
               invalid_reason="stop_breached_today" if breached else None,
               tight=bool(not breached and cushion is not None and cushion < TIGHT_CUSHION_ATR))
    return out


def actionable_live(rows, bar_fn=None):
    """ردیف‌های «ورودِ تازه» و «پیوستن» با قیمتِ زنده؛ بی‌قیمتِ زنده = نامعتبر (نه «فرض کن خوب است»)."""
    out = []
    for r in rows or []:
        if r.get("category") not in ("new", "join"):
            out.append(r)
            continue
        try:
            out.append(with_live(r, live_bar(r["sym"], bar_fn)))
        except Exception:  # noqa: BLE001, silent-ok — در خودِ ردیف گزارش می‌شود
            out.append(dict(r, valid=False, invalid_reason="no_live_price"))
    return out


def view(force=False, bar_fn=None):
    """آنچه ‎/api/trend‎ برمی‌گرداند: عکسِ روزانه (کش) + قیمتِ زندهٔ سیگنال‌های قابل‌اقدام (همین لحظه)."""
    snap = snapshot(force=force)
    rows = actionable_live(snap.get("now"), bar_fn)
    expected = (snap.get("last_closed_bar_ms") or 0) + explore.DAY_MS
    if not force and any(r.get("live_bar_ts") and r["live_bar_ts"] > expected for r in rows):
        snap = snapshot(force=True)              # کندلِ روزانهٔ تازه بسته شده؛ عکسِ کش‌شده کهنه است
        rows = actionable_live(snap.get("now"), bar_fn)
    out = dict(snap)
    out["now"] = rows
    out["demo_open"] = demo_open_symbols()
    out["demo_heat_pct"] = round(_demo_heat_pct(), 3)
    out["demo_max_heat_pct"] = DEMO_MAX_HEAT_PCT
    return out


def _trend_positions():
    import paper
    return [p for p in paper.list_positions()["open"] if p.get("strategy") == DEMO_STRATEGY]


def demo_open_symbols():
    return sorted({p["symbol"] for p in _trend_positions()})


def _demo_equity():
    import paper
    w = paper.wallet_summary()
    return float(w.get("equity") or w.get("balance") or 0.0)


def _demo_heat_pct():
    """مجموعِ ریسکِ بازِ پوزیشن‌های روند برحسبِ ٪ موجودی (ضررِ هم‌زمانِ همه تا حدضرر)."""
    equity = _demo_equity()
    if equity <= 0:
        return 0.0
    risk = sum(float(p["size_usdt"]) * max(float(p["entry"]) - float(p["sl"]), 0.0) / float(p["entry"])
               for p in _trend_positions())
    return risk / equity * 100.0


def open_demo(symbol, risk_pct=0.5, now_rows=None, bar=None):
    """پوزیشنِ دمو برای سیگنالِ قابل‌اقدامِ امروز: خرید در قیمتِ **زنده** با حدضررِ قاعده، بی‌هدف،
    هزینهٔ اسپات (بی‌فاندینگ)، حجم = موجودیِ دمو × ریسک٪ ÷ فاصلهٔ زنده تا حدضرر — و بی‌اهرم."""
    import paper
    rows = now_rows if now_rows is not None else (snapshot().get("now") or [])
    row = next((r for r in rows if r["sym"] == symbol), None)
    if row is None or row.get("category") not in ("new", "join"):
        raise ValueError(f"{symbol} الان سیگنالِ ورود یا پیوستن ندارد")
    if symbol in demo_open_symbols():
        raise ValueError(f"پوزیشنِ دموی روند روی {symbol} از قبل باز است")
    try:
        lv = with_live(row, bar if bar is not None else live_bar(symbol))
    except Exception as e:  # noqa: BLE001
        raise ValueError(f"قیمتِ زنده در دسترس نیست — بدونِ آن وارد نمی‌شوم ({e})")
    if not lv["valid"]:
        raise ValueError(f"سیگنالِ {symbol} باطل شده: قیمتِ امروز تا {lv['day_low']:g} پایین آمده و به حدضررِ "
                         f"قاعده ({lv['stop']:g}) رسیده است — قاعده عملاً از این روند بیرون است")
    price, stop = lv["live"], float(row["stop"])
    risk_pct = min(max(float(risk_pct), DEMO_RISK_RANGE[0]), DEMO_RISK_RANGE[1])
    equity = _demo_equity()
    if equity <= 0:
        raise ValueError("موجودیِ دمو صفر است")
    heat = _demo_heat_pct()
    if heat + risk_pct > DEMO_MAX_HEAT_PCT + 1e-9:
        raise ValueError(f"ریسکِ بازِ پوزیشن‌های روند {heat:.1f}٪ است؛ با این معامله از سقفِ {DEMO_MAX_HEAT_PCT:g}٪ "
                         "می‌گذرد — ارزها با هم حرکت می‌کنند و همه می‌توانند هم‌زمان حدضرر بخورند")
    size = min(equity * risk_pct / 100.0 / ((price - stop) / price), equity)
    label = "ورودِ تازه" if row["category"] == "new" else f"پیوستن — روزِ {row.get('days_in')}"
    return paper.open_position(symbol, "1d", "long", price, stop, None, round(size, 2), DEMO_MAX_HOLD_MIN,
                               grade=f"روند: {label}", opened_by="user",
                               cost_pct=explore.SPOT_ROUND_TRIP_PCT, strategy=DEMO_STRATEGY, market="spot",
                               ref={"rule": PRIMARY, "rule_entry_ts": row.get("rule_entry_ts")})


def sync_demo(now_rows=None, price_fn=None):
    """پوزیشن‌های دموی روند هم‌پای **همان معاملهٔ قاعده**: حدضرر فقط بالا می‌رود، و وقتی آن معامله
    بسته شد، دمو با **قیمتِ خروجِ خودِ قاعده** بسته می‌شود.

    نسخهٔ قبلی «قاعده بیرون است ⇒ با قیمتِ همین لحظه ببند» بود. FIL در ۲۰۲۶-۰۹-۱۶ ساعتِ ۱۲ UTC حدضررِ
    ۰٫۷۷۳ را زد، برنامه آن لحظه باز نبود، و دمو فردایش در ۰٫۷۹۶۷ بسته شد: ضررِ ۳۶ دلاری به‌جای ~۵۰
    دلاری که سفارشِ حدضررِ واقعی می‌ساخت. دمویی که خوش‌بینانه حساب کند، کاربر را گمراه می‌کند.
    """
    import paper
    rows = {r["sym"]: r for r in (now_rows if now_rows is not None else (snapshot().get("now") or []))}
    price_fn = price_fn or market.last_price
    moved = closed = 0
    for p in _trend_positions():
        r = rows.get(p["symbol"])
        if r is None:
            continue                                 # بی‌داده: دست نزن
        ref_ts = (p.get("ref") or {}).get("rule_entry_ts")
        active_ts = r.get("rule_entry_ts") if (r.get("in_position") or r.get("category") == "new") else None
        if active_ts is not None and (ref_ts is None or int(active_ts) == int(ref_ts)):
            if float(r["stop"]) > float(p["sl"]) * (1 + 1e-9) and paper.move_sl(p["id"], float(r["stop"])):
                moved += 1
            continue
        exits = r.get("recent_exits") or []
        ex = next((x for x in reversed(exits) if ref_ts is None or int(x["entry_ts"]) == int(ref_ts)), None)
        if ex is not None:
            stopped = ex["outcome"] in ("stop", "gap_stop")
            px = paper._apply_slippage(ex["exit_px"], p["side"], p.get("cost_pct")) if stopped else float(ex["exit_px"])
            done = paper.close_with(p["id"], px, "حدضررِ قاعدهٔ روند" if stopped else "خروجِ قاعدهٔ روند")
        else:                                        # معاملهٔ مرجع در پنجرهٔ داده پیدا نشد
            done = paper.close_with(p["id"], float(price_fn(p["symbol"])), "خروجِ قاعدهٔ روند")
        closed += 1 if done else 0
    return {"moved": moved, "closed": closed}


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
        "symbols": list(ALL_SYMBOLS),
        "majors": list(SYMBOLS),
        "extension": list(EXT_SYMBOLS),
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
