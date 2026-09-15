# -*- coding: utf-8 -*-
"""جست‌وجوی لبه روی دادهٔ اکتشاف — هرگز روی پنجره‌های منجمد.

حکمِ آزمونِ منجمدِ #۱: هیچ‌کدام از ۸ فرضیه لبه نداشت. آن خانواده فقط **یک** ایده
بود: ستاپِ کوتاه‌مدت (کراسِ z و شکستِ فشردگی) با خروجِ ثابتِ 1.8R/۴۰ کندل. اینجا
خانواده‌هایی آزموده می‌شوند که پشتوانهٔ اقتصادی و سابقهٔ منتشرشده در کریپتو دارند:
روند با خروجِ دنباله‌دار، همان ستاپ‌ها فقط هم‌جهتِ روندِ روزانه، مومنتومِ مقطعیِ
خنثی‌به‌بازار، و بازگشت پس از روزِ حدی.

قواعد پیش از دیدنِ هر عددی در کد ثابت شده‌اند (``VARIANTS``):

* **فقط پیش از پنجره‌های منجمد.** مرز = زودترین شروعِ پنجرهٔ منجمد در پیش‌ثبت
  (برای همهٔ تایم‌فریم‌ها یکی، تا هیچ پنجره‌ای حتی از راهِ تایم‌فریمِ دیگر دیده
  نشود). کندل‌های پس از مرز اصلاً بارگذاری نمی‌شوند؛ معاملهٔ باز در لبهٔ داده با
  بستهٔ آخرین کندل بسته و علامت‌گذاری می‌شود.
* **جهانِ نقطه‌-در-زمان.** فقط ۲۰ ارزِ پرحجمِ همان ماه — شاملِ ارزهای ۲۰۱۸ که بعداً
  سقوط کردند — نه برنده‌های امروز. ارزهای حذف‌شده از بورس (LUNA، FTT) در API نیستند؛
  این سوگیریِ باقی‌مانده است، پس BTC+ETH (که همیشه اول و دوم بوده‌اند) جدا هم
  گزارش می‌شود.
* **اجرای واقعی.** ورود در openِ کندلِ بعد؛ حدضررِ درون‌کندلی و گپ با قیمتِ بازِ
  مشاهده‌شده؛ هزینهٔ ردهٔ نقدشوندگیِ نقطه‌-در-زمان + فاندینگِ پیش‌فرضِ ۰٫۰۱٪ در هر
  تسویه که **همیشه هزینه** است (تاریخچهٔ فاندینگ فقط ~۳۳ روز است). حالتِ «فاندینگِ
  معمول» (لانگ می‌پردازد، شورت می‌گیرد) فقط برای حساسیت گزارش می‌شود.
* **بهای جست‌وجو.** Romano-Wolf روی **همهٔ** واریانت‌های اجراشده؛ آستانهٔ میانگین با
  شمارِ کلِ فرضیه‌های پروژه (۸ فرضیهٔ آزمایشِ #۱ + این‌ها) تنزیل می‌شود.

خروجی حکم نیست و به gates دست نمی‌زند. بهترین نامزدها فقط می‌توانند واردِ پیش‌ثبتِ
بعدی شوند و روی داده‌ای داوری شوند که اینجا دیده نشده است.
"""
import bisect
import json
import math
import os
import time

import numpy as np

import bracket
import costs
import engine
import paths
import stats
import universe

HIST_DIR = paths.data("hist")
OUT_DIR = paths.data("research")
DAY_MS = 86_400_000
TF_MS = costs.TF_MS
TOP_N = 20
ALPHA = 0.05
B = 1000
PRIOR_HYPOTHESES = 8                     # آزمایشِ #۱ — بخشی از بهای جست‌وجوی کلِ پروژه
PEG_MEDIAN_RANGE_PCT = 0.3               # میانهٔ بازهٔ روزانه؛ استیبل‌ها ~۰٫۰۲٪، BTC ~۳٪
# رپ‌شده/استیک‌شده‌ها همان BTC/ETH‌اند؛ دوباره‌شماری همبستگی را پنهان می‌کند
DUPLICATE_BASES = {"BETH", "WBETH", "WBTC", "STETH", "CBETH", "BTCB"}

TREND_BLOCK_MS = 30 * DAY_MS             # معامله‌های روندیِ هم‌زمان روی ارزها هم‌بسته‌اند
SETUP_BLOCK_MS = bracket.MAX_BARS * TF_MS["4h"]
XSEC_BLOCK_MS = 28 * DAY_MS
FAMILY_BLOCK_MS = 30 * DAY_MS            # Romano-Wolf: یک بلوکِ مشترک و محافظه‌کار

TREND_STOP_ATR = 3.0
TREND_TRAIL_ATR = 3.0
REV_SIGMA = 2.5
REV_STOP_ATR = 1.5
REV_HOLD = 3
XSEC_LEGS = 5
SETUP_TREND_SMA = 100

VARIANTS = (
    # روندِ روزانه: شکستِ کانالِ دانچیان؛ حدضررِ اولیه 3×ATR20 و خروجِ دنباله‌دارِ 3×ATR20
    {"key": "T_don20_long", "family": "trend", "tf": "1d", "rule": "donchian", "n": 20, "side": 1},
    {"key": "T_don20_short", "family": "trend", "tf": "1d", "rule": "donchian", "n": 20, "side": -1},
    {"key": "T_don55_long", "family": "trend", "tf": "1d", "rule": "donchian", "n": 55, "side": 1},
    {"key": "T_don55_short", "family": "trend", "tf": "1d", "rule": "donchian", "n": 55, "side": -1},
    # روندِ روزانه: عبور از SMA100؛ خروج با عبورِ معکوس یا حدضررِ اولیه
    {"key": "T_sma100_long", "family": "trend", "tf": "1d", "rule": "sma_cross", "n": 100, "side": 1},
    {"key": "T_sma100_short", "family": "trend", "tf": "1d", "rule": "sma_cross", "n": 100, "side": -1},
    # ستاپ‌های آزمایشِ #۱ روی 4h، فقط هم‌جهتِ روندِ روزانه (بسته نسبت به SMA100)، براکتِ واحد
    {"key": "S_zx_long_up", "family": "setup_trend", "tf": "4h", "setup": "zx", "side": 1},
    {"key": "S_zx_short_down", "family": "setup_trend", "tf": "4h", "setup": "zx", "side": -1},
    {"key": "S_sq_long_up", "family": "setup_trend", "tf": "4h", "setup": "sq", "side": 1},
    {"key": "S_sq_short_down", "family": "setup_trend", "tf": "4h", "setup": "sq", "side": -1},
    # مومنتومِ مقطعیِ هفتگی، خنثی‌به‌بازار: ۵ قوی‌تر لانگ و ۵ ضعیف‌تر شورت (rev7 برعکس)
    {"key": "X_mom14", "family": "xsec", "tf": "1d", "lookback": 14, "sign": 1},
    {"key": "X_mom28", "family": "xsec", "tf": "1d", "lookback": 28, "sign": 1},
    {"key": "X_rev7", "family": "xsec", "tf": "1d", "lookback": 7, "sign": -1},
    # بازگشت پس از روزِ حدی (> 2.5σ)، ۳ روز نگه‌داری، حدضرر 1.5×ATR20
    {"key": "V_rev_long", "family": "reversal", "tf": "1d", "side": 1},
    {"key": "V_rev_short", "family": "reversal", "tf": "1d", "side": -1},
)


# ───────────────────────── داده ─────────────────────────
def dev_cutoff_ms(doc=None):
    """زودترین شروعِ پنجرهٔ منجمد در پیش‌ثبت — مرزِ اکتشاف برای **همهٔ** تایم‌فریم‌ها.

    بی‌پیش‌ثبت = خطا، نه «همه‌چیز مجاز»: بدونِ مرز نمی‌دانیم کجا را نباید دید.
    """
    if doc is None:
        import research
        doc = research.load_prereg()
    starts = [int(w["start_ms"]) for w in ((doc or {}).get("final_windows") or {}).values()
              if w and w.get("start_ms")]
    if not starts:
        raise ValueError("پنجرهٔ منجمدی در پیش‌ثبت نیست — مرزِ اکتشاف نامعلوم است")
    return min(starts)


def _excluded(sym):
    import market
    base = sym[:-4] if sym.endswith("USDT") else sym
    return (not sym.endswith("USDT") or base in market.STABLE_BASES or base in DUPLICATE_BASES
            or any(base.endswith(s) for s in market.LEVERAGED_SUFFIX))


def load_panel(tf, cutoff_ms, min_bars=60, hist_dir=None):
    """کندل‌های کش برای یک تایم‌فریم، **بریده پیش از** ``cutoff_ms``.

    دارایی‌های میخ‌شده (میانهٔ بازهٔ روزانه زیرِ ۰٫۳٪) و رپ‌شده‌ها کنار می‌روند.
    """
    hist_dir = hist_dir or HIST_DIR
    out = {}
    suffix = f"_{tf}.json"
    for name in sorted(os.listdir(hist_dir)):
        if not name.endswith(suffix) or name.startswith("funding_"):
            continue
        sym = name[:-len(suffix)]
        if _excluded(sym):
            continue
        try:
            with open(os.path.join(hist_dir, name), "r", encoding="utf-8") as f:
                raw = json.load(f)
            t = np.asarray(raw["t"], dtype=np.int64)
            # کندلِ روزانهٔ ۱۶:۰۰ یعنی دادهٔ OKX (نمادی که در بایننس نیست): معامله‌پذیر
            # نیست و تقویمِ هفتگی را هم می‌شکست — مومنتومِ مقطعی پس از ۲۰۲۱ اصلاً اجرا نمی‌شد.
            if np.any(t % TF_MS[tf]):
                continue
            keep = t < int(cutoff_ms)
            if int(keep.sum()) < min_bars:
                continue
            k = {"t": t[keep]}
            for col in ("o", "h", "l", "c", "v"):
                k[col] = np.asarray(raw[col], dtype=np.float64)[keep]
        except (OSError, ValueError, KeyError):
            continue
        rng = (k["h"] - k["l"]) / np.maximum(k["c"], 1e-12) * 100.0
        if float(np.median(rng)) < PEG_MEDIAN_RANGE_PCT * (1.0 if tf == "1d" else 0.25):
            continue
        out[sym] = k
    return out


def sma(x, n):
    x = np.asarray(x, dtype=np.float64)
    out = np.full(x.size, np.nan)
    if x.size >= n:
        cs = np.cumsum(np.concatenate([[0.0], x]))
        out[n - 1:] = (cs[n:] - cs[:-n]) / n
    return out


# ───────────────────────── مسیرِ معامله ─────────────────────────
def walk(o, h, l, c, a, e, side, entry, R, trail_atr=None, max_hold=120, exit_fn=None):
    """یک معامله از کندلِ ورودِ ``e`` (ورود در openِ همان کندل) تا خروج.

    ترتیب در هر کندل: گپ پشتِ حدضرر (خروج با open) ← حدضررِ درون‌کندلی ← سیگنالِ
    خروج روی بسته (خروج در openِ کندلِ بعد) ← به‌روزرسانیِ حدضررِ دنباله‌دار با
    اطلاعاتِ همین بسته (از کندلِ بعد اعمال می‌شود). هیچ کندلی از آینده دیده نمی‌شود.
    """
    n = len(c)
    stop = entry - side * R
    best = entry
    last = min(e + max_hold - 1, n - 1)
    for j in range(e, last + 1):
        if (side == 1 and o[j] <= stop) or (side == -1 and o[j] >= stop):
            return {"exit_idx": j, "exit_px": float(o[j]), "outcome": "gap_stop"}
        if (side == 1 and l[j] <= stop) or (side == -1 and h[j] >= stop):
            return {"exit_idx": j, "exit_px": float(stop), "outcome": "stop"}
        if exit_fn is not None and exit_fn(j):
            if j + 1 < n:
                return {"exit_idx": j + 1, "exit_px": float(o[j + 1]), "outcome": "signal"}
            return {"exit_idx": j, "exit_px": float(c[j]), "outcome": "end_of_data"}
        if trail_atr:
            best = max(best, float(c[j])) if side == 1 else min(best, float(c[j]))
            cand = best - side * trail_atr * float(a[j])
            stop = max(stop, cand) if side == 1 else min(stop, cand)
    censored = last == n - 1 and last < e + max_hold - 1
    return {"exit_idx": last, "exit_px": float(c[last]),
            "outcome": "end_of_data" if censored else "timeout"}


def _typical_funding_pct(side, entry_ts, exit_ts):
    """حساسیت: لانگ ۰٫۰۱٪ در هر تسویه می‌پردازد و شورت همان را می‌گیرد."""
    return side * costs.settlements_between(entry_ts, exit_ts) * costs.DEFAULT_FUNDING_PER_SETTLEMENT_PCT


def _trade(sym, tf, side, i, e, res, entry, R, k):
    t, c, v = k["t"], k["c"], k["v"]
    entry_ts, exit_ts = int(t[e]), int(t[res["exit_idx"]])
    risk_pct = R / max(entry, 1e-12) * 100.0
    tier = costs.point_in_time_tier(c, v, i, tf)
    cost = tier + costs.funding_cost_pct(side, entry_ts, exit_ts, None)
    cost_typ = tier + _typical_funding_pct(side, entry_ts, exit_ts)
    gross_r = side * (res["exit_px"] - entry) / R
    return {"sym": sym, "side": side, "ts": int(t[i]), "entry_ts": entry_ts, "exit_ts": exit_ts,
            "gross_r": gross_r, "risk_pct": risk_pct, "cost_pct": cost,
            "net_r": bracket.net_r(gross_r, risk_pct, cost),
            "net_r_typical_funding": gross_r - cost_typ / max(risk_pct, 0.05),
            "bars": int(res["exit_idx"] - e + 1), "outcome": res["outcome"]}


# ───────────────────────── خانواده‌ها ─────────────────────────
def trend_trades(panel, snaps, spec):
    side, n_look = spec["side"], spec["n"]
    out = []
    for sym, k in panel.items():
        o, h, l, c, t = k["o"], k["h"], k["l"], k["c"], k["t"]
        n = len(c)
        a = engine.atr(h, l, c, 20)
        ma = sma(c, n_look) if spec["rule"] == "sma_cross" else None
        if spec["rule"] == "donchian":
            def entry_ok(i):
                window = h[i - n_look:i] if side == 1 else l[i - n_look:i]
                return c[i] > window.max() if side == 1 else c[i] < window.min()
            exit_fn, trail, max_hold = None, TREND_TRAIL_ATR, 120
        else:
            def entry_ok(i):
                if not (np.isfinite(ma[i]) and np.isfinite(ma[i - 1])):
                    return False
                return (c[i] > ma[i] and c[i - 1] <= ma[i - 1]) if side == 1 else \
                    (c[i] < ma[i] and c[i - 1] >= ma[i - 1])
            def exit_fn(j):
                return (c[j] < ma[j]) if side == 1 else (c[j] > ma[j])
            trail, max_hold = None, 250
        i = max(n_look, 21) + 1
        while i < n - 1:
            if not entry_ok(i) or not universe.in_universe(snaps, sym, int(t[i])):
                i += 1
                continue
            e = i + 1
            entry, R = float(o[e]), TREND_STOP_ATR * float(a[i])
            if R <= 0 or entry <= 0:
                i += 1
                continue
            res = walk(o, h, l, c, a, e, side, entry, R, trail_atr=trail, max_hold=max_hold,
                       exit_fn=exit_fn)
            out.append(_trade(sym, "1d", side, i, e, res, entry, R, k))
            i = max(res["exit_idx"], i + 1)       # یک پوزیشن در هر نماد؛ اسکن از خروج ادامه
    return out


def reversal_trades(panel, snaps, spec):
    side = spec["side"]
    out = []
    for sym, k in panel.items():
        o, h, l, c, t = k["o"], k["h"], k["l"], k["c"], k["t"]
        n = len(c)
        a = engine.atr(h, l, c, 20)
        ret = np.concatenate([[0.0], np.diff(c) / np.maximum(c[:-1], 1e-12)])
        i = 22
        while i < n - 1:
            sd = float(np.std(ret[i - 20:i], ddof=1))
            extreme = sd > 0 and (ret[i] < -REV_SIGMA * sd if side == 1 else ret[i] > REV_SIGMA * sd)
            if not extreme or not universe.in_universe(snaps, sym, int(t[i])):
                i += 1
                continue
            e = i + 1
            entry, R = float(o[e]), REV_STOP_ATR * float(a[i])
            if R <= 0 or entry <= 0:
                i += 1
                continue
            res = walk(o, h, l, c, a, e, side, entry, R, max_hold=REV_HOLD)
            out.append(_trade(sym, "1d", side, i, e, res, entry, R, k))
            i = max(res["exit_idx"], i + 1)
    return out


def setup_events(sym, k, tf="4h"):
    """رویدادهای ستاپ با همان حلقهٔ ``calib.extract_events`` (شروع ۲۶۰، پایان n−42،
    خنک‌شدنِ ۶ کندلی، براکتِ واحد، هزینهٔ رویداد) — بدونِ ساختنِ ویژگی‌ها."""
    o, h, l, c, v, t = k["o"], k["h"], k["l"], k["c"], k["v"], k["t"]
    n = len(c)
    if n < 310:
        return []
    cs = engine.component_series(o, h, l, c, v)
    events, cooldown = [], 0
    for i in range(260, n - 42):
        if cooldown > 0:
            cooldown -= 1
            continue
        sig, setup = engine.setup_signal(cs, o, h, l, c, i)
        if sig == 0:
            continue
        res = bracket.signal_trade(o, h, l, c, cs["a14"][i], i, sig)
        if res is None:
            continue
        entry_ts, exit_ts = int(t[i + 1]), int(t[min(res["exit_idx"], n - 1)])
        cost = costs.event_cost_pct(sig, c, v, i, tf, entry_ts, exit_ts, None)
        events.append({"ts": int(t[i]), "sym": sym, "dir": int(sig), "setup": setup,
                       "r": round(float(res["gross_r"]), 3), "risk_pct": round(float(res["risk_pct"]), 4),
                       "cost_pct": cost, "bars": int(res["bars_held"]), "outcome": res["outcome"],
                       "entry_ts": entry_ts, "exit_ts": exit_ts})
        cooldown = 6
    return events


def daily_trend_lookup(daily_panel, sma_n=SETUP_TREND_SMA):
    """``fn(sym, ts_signal_close) → +1/−1/0`` از آخرین کندلِ روزانهٔ **بسته‌شده**."""
    cache = {}
    for sym, k in daily_panel.items():
        cache[sym] = (k["t"], k["c"], sma(k["c"], sma_n))

    def trend(sym, close_ts):
        if sym not in cache:
            return 0
        t, c, m = cache[sym]
        d = bisect.bisect_right(t, close_ts - DAY_MS) - 1     # کندلی که تا close_ts بسته شده
        if d < 0 or not np.isfinite(m[d]):
            return 0
        return 1 if c[d] > m[d] else -1
    return trend


def setup_trend_trades(events, snaps, trend_fn, spec):
    out = []
    for ev in events:
        if ev["setup"] != spec["setup"] or ev["dir"] != spec["side"]:
            continue
        if not universe.in_universe(snaps, ev["sym"], ev["ts"]):
            continue
        close_ts = ev["ts"] + TF_MS["4h"]
        if trend_fn(ev["sym"], close_ts) != spec["side"]:
            continue
        risk = max(float(ev["risk_pct"]), 0.05)
        funding_default = costs.funding_cost_pct(ev["dir"], ev["entry_ts"], ev["exit_ts"], None)
        tier = float(ev["cost_pct"]) - funding_default
        out.append({"sym": ev["sym"], "side": ev["dir"], "ts": ev["ts"], "entry_ts": ev["entry_ts"],
                    "exit_ts": ev["exit_ts"], "gross_r": ev["r"], "risk_pct": risk,
                    "cost_pct": ev["cost_pct"],
                    "net_r": bracket.net_r(ev["r"], risk, ev["cost_pct"]),
                    "net_r_typical_funding": ev["r"] - (tier + _typical_funding_pct(
                        ev["dir"], ev["entry_ts"], ev["exit_ts"])) / risk,
                    "bars": ev["bars"], "outcome": ev["outcome"]})
    return out


def xsec_weeks(panel, snaps, spec):
    """هر دوشنبه: رتبه بر اساسِ بازدهِ ``lookback`` روزِ گذشته تا بستهٔ یکشنبه؛ ورود در
    openِ دوشنبه، خروج در openِ دوشنبهٔ بعد. خروجی به درصد (نه R)."""
    L, sign = spec["lookback"], spec["sign"]
    idx = {sym: {int(ts): i for i, ts in enumerate(k["t"])} for sym, k in panel.items()}
    all_ts = sorted({int(ts) for k in panel.values() for ts in k["t"]})
    mondays = [ts for ts in all_ts if ts % DAY_MS == 0 and time.gmtime(ts / 1000).tm_wday == 0]
    out = []
    for m0, m1 in zip(mondays, mondays[1:]):
        if m1 - m0 != 7 * DAY_MS:
            continue
        uni = universe.universe_at(snaps, m0)
        if not uni:
            continue
        rows = []
        for sym in uni:
            k, ix = panel.get(sym), idx.get(sym)
            if k is None or m0 not in ix or m1 not in ix:
                continue
            i0, i1 = ix[m0], ix[m1]
            s = i0 - 1                                   # بستهٔ یکشنبه
            if s - L < 0:
                continue
            past = k["c"][s] / max(k["c"][s - L], 1e-12) - 1.0
            fwd = (k["o"][i1] / max(k["o"][i0], 1e-12) - 1.0) * 100.0
            tier = costs.point_in_time_tier(k["c"], k["v"], s, "1d")
            rows.append((past, fwd, tier))
        if len(rows) < 2 * XSEC_LEGS:
            continue
        rows.sort(key=lambda r: r[0], reverse=(sign == 1))
        longs, shorts = rows[:XSEC_LEGS], rows[-XSEC_LEGS:]
        settle = costs.settlements_between(m0, m1)
        fund = settle * costs.DEFAULT_FUNDING_PER_SETTLEMENT_PCT
        gross = float(np.mean([r[1] for r in longs]) - np.mean([r[1] for r in shorts]))
        tiers = float(np.mean([r[2] for r in longs]) + np.mean([r[2] for r in shorts]))
        out.append({"ts": m0, "gross_pct": gross,
                    "net_pct": gross - tiers - 2 * fund,          # هر دو پا فاندینگ می‌پردازند
                    "net_pct_typical_funding": gross - tiers,     # پاها فاندینگ را خنثی می‌کنند
                    "n_universe": len(rows)})
    return out


# ───────────────────────── آمار ─────────────────────────
def _max_drawdown(values):
    if not len(values):
        return 0.0
    eq = np.cumsum(values)
    return float(np.max(np.maximum.accumulate(np.concatenate([[0.0], eq]))[1:] - eq))


def _by_year(values, ts):
    out = {}
    for v, t in zip(values, ts):
        y = time.gmtime(int(t) / 1000).tm_year
        out.setdefault(y, []).append(float(v))
    return {str(y): {"n": len(v), "mean": round(float(np.mean(v)), 4)} for y, v in sorted(out.items())}


def describe(values, ts, block_ms, rho=stats.DEFAULT_RHO, alt=None):
    values, ts = np.asarray(values, float), np.asarray(ts, np.int64)
    s = stats.summarize(values, ts, block_ms, alpha=ALPHA, rho=rho, B=B)
    if not len(values):
        return s
    order = np.argsort(ts)
    v_sorted = values[order]
    half = len(v_sorted) // 2
    s["first_half_mean"] = round(float(v_sorted[:half].mean()), 4) if half else None
    s["second_half_mean"] = round(float(v_sorted[half:].mean()), 4) if half else None
    s["max_drawdown"] = round(_max_drawdown(v_sorted), 3)
    s["by_year"] = _by_year(values, ts)
    years = s["by_year"].values()
    s["positive_years"] = f"{sum(1 for y in years if y['mean'] > 0)}/{len(s['by_year'])}"
    if alt is not None and len(alt):
        s["mean_typical_funding"] = round(float(np.mean(alt)), 4)
    return s


def classify(s, rw, deflated):
    """برچسبِ اکتشافی — نه مجوز."""
    if not s.get("n") or s.get("mean") is None or s["mean"] <= 0:
        return "no_edge"
    halves_ok = (s.get("first_half_mean") or 0) > 0 and (s.get("second_half_mean") or 0) > 0
    strong = (rw.get("p_adj") is not None and rw["p_adj"] < ALPHA and (s.get("lcb") or 0) > 0
              and deflated is not None and s["mean"] > deflated and halves_ok
              and (s.get("n_eff") or 0) >= 30)
    if strong:
        return "promising"
    if (s.get("lcb") or 0) > 0:
        return "weak"
    return "inconclusive"


def run(cutoff_ms=None, variants=VARIANTS, hist_dir=None, progress=print):
    cutoff_ms = int(cutoff_ms if cutoff_ms is not None else dev_cutoff_ms())
    progress(f"مرزِ اکتشاف: {time.strftime('%Y-%m-%d', time.gmtime(cutoff_ms / 1000))}")
    daily = load_panel("1d", cutoff_ms, hist_dir=hist_dir)
    snaps = universe.snapshots_from_histories(
        {s: {"t": k["t"].tolist(), "c": k["c"].tolist(), "v": k["v"].tolist()} for s, k in daily.items()},
        top_n=TOP_N)
    ever = set().union(*[s for _t, s in snaps]) if snaps else set()
    daily_uni = {s: k for s, k in daily.items() if s in ever}
    progress(f"روزانه: {len(daily)} نماد، {len(ever)} نماد تا کنون در جهانِ ۲۰تایی")
    need_4h = any(v["family"] == "setup_trend" for v in variants)
    setup_evs = []
    if need_4h:
        panel4 = {s: k for s, k in load_panel("4h", cutoff_ms, min_bars=310, hist_dir=hist_dir).items()
                  if s in ever}
        progress(f"4h: {len(panel4)} نماد برای ستاپ‌ها")
        for s, k in panel4.items():
            setup_evs.extend(setup_events(s, k, "4h"))
    trend_fn = daily_trend_lookup(daily)

    results, family_series = {}, {}
    for spec in variants:
        fam = spec["family"]
        if fam == "trend":
            rows = trend_trades(daily_uni, snaps, spec)
        elif fam == "reversal":
            rows = reversal_trades(daily_uni, snaps, spec)
        elif fam == "setup_trend":
            rows = setup_trend_trades(setup_evs, snaps, trend_fn, spec)
        else:
            rows = xsec_weeks(daily_uni, snaps, spec)
        if fam == "xsec":
            vals = [r["net_pct"] for r in rows]
            alt = [r["net_pct_typical_funding"] for r in rows]
            ts = [r["ts"] for r in rows]
            s = describe(vals, ts, XSEC_BLOCK_MS, rho=0.0, alt=alt)
            s["unit"] = "pct_per_week"
            if s.get("mean") is not None and s.get("sd"):
                s["sharpe_annual"] = round(s["mean"] / s["sd"] * math.sqrt(52), 3)
        else:
            vals = [r["net_r"] for r in rows]
            alt = [r["net_r_typical_funding"] for r in rows]
            ts = [r["entry_ts"] for r in rows]
            block = SETUP_BLOCK_MS if fam == "setup_trend" else TREND_BLOCK_MS
            s = describe(vals, ts, block, alt=alt)
            s["unit"] = "net_r"
            s["avg_bars"] = round(float(np.mean([r["bars"] for r in rows])), 1) if rows else None
            outcomes = {}
            for r in rows:
                outcomes[r["outcome"]] = outcomes.get(r["outcome"], 0) + 1
            s["outcomes"] = outcomes
            majors2 = [r["net_r"] for r in rows if r["sym"] in ("BTCUSDT", "ETHUSDT")]
            s["btc_eth"] = {"n": len(majors2),
                            "mean": round(float(np.mean(majors2)), 4) if majors2 else None}
        results[spec["key"]] = {"spec": dict(spec), "stats": s}
        if len(vals) >= 2:
            family_series[spec["key"]] = (np.asarray(vals, float), np.asarray(ts, np.int64))
        progress(f"  {spec['key']}: n={s.get('n')} mean={s.get('mean')} lcb={s.get('lcb')}")

    rw = stats.romano_wolf(family_series, FAMILY_BLOCK_MS, alpha=ALPHA, B=B) if family_series else {}
    n_trials = PRIOR_HYPOTHESES + len(variants)
    for key, blob in results.items():
        s = blob["stats"]
        blob["romano_wolf"] = rw.get(key, {})
        deflated = stats.deflated_mean_threshold(n_trials, s.get("sd") or 0, s.get("n_eff") or 0, ALPHA) \
            if s.get("n") else None
        blob["deflated_mean_threshold"] = deflated
        blob["verdict"] = classify(s, blob["romano_wolf"], deflated)
    return {
        "generated_at": time.time(),
        "cutoff_ms": cutoff_ms,
        "universe": universe.summary(snaps),
        "n_variants": len(variants),
        "n_trials_project": n_trials,
        "notes": [
            "exploration only — no gate is changed; survivors need a pre-registered test on unseen data",
            "funding history covers ~33 days; the default 0.01%/settlement is charged to both sides",
            "delisted coins (e.g. LUNA, FTT) are missing from the public API — residual survivorship bias",
        ],
        "variants": results,
    }


# ───────────────────────── حساسیت و نمای حساب ─────────────────────────
STRESS_LONG_FUNDING_PCT = 0.03          # بازارِ گاوی: لانگ‌ها اغلب ۰٫۰۳٪+ در هر تسویه می‌پردازند
STRESS_SLIPPAGE_PCT = 0.35              # ۰٫۱٪ ورود + ۰٫۲۵٪ حدضرر، رفت‌وبرگشت


def stressed_r(tr, long_funding_pct=STRESS_LONG_FUNDING_PCT, slippage_pct=STRESS_SLIPPAGE_PCT):
    """همان معامله با فاندینگِ سنگین‌ترِ لانگ و لغزشِ بدتر — سود باید زیرِ فشار بماند."""
    settle = costs.settlements_between(tr["entry_ts"], tr["exit_ts"])
    extra = slippage_pct
    if tr["side"] == 1:
        extra += settle * (long_funding_pct - costs.DEFAULT_FUNDING_PER_SETTLEMENT_PCT)
    return tr["net_r"] - extra / max(tr["risk_pct"], 0.05)


def portfolio(trades, risk_equity_pct=0.5, max_open=5, start=10_000.0):
    """حسابِ فرضی: هر معامله ``risk_equity_pct``٪ از موجودیِ لحظهٔ ورود را ریسک می‌کند؛
    اگر ``max_open`` پوزیشن باز باشد سیگنال رد می‌شود. منحنی روی معامله‌های **بسته**
    است، پس افتِ درونِ معامله را کمتر از واقع نشان می‌دهد."""
    events = sorted(trades, key=lambda t: (t["entry_ts"], t["sym"]))
    equity, peak, max_dd = start, start, 0.0
    open_pos, taken, by_year = [], 0, {}
    for tr in events:
        still = []
        for pos in sorted(open_pos, key=lambda p: p["exit_ts"]):
            if pos["exit_ts"] <= tr["entry_ts"]:
                pnl = pos["stake"] * pos["r"]
                equity += pnl
                y = time.gmtime(pos["exit_ts"] / 1000).tm_year
                by_year[y] = by_year.get(y, 0.0) + pnl
                peak = max(peak, equity)
                max_dd = max(max_dd, (peak - equity) / peak)
            else:
                still.append(pos)
        open_pos = still
        if len(open_pos) >= max_open:
            continue
        open_pos.append({"exit_ts": tr["exit_ts"], "r": tr["net_r"],
                         "stake": equity * risk_equity_pct / 100.0})
        taken += 1
    for pos in sorted(open_pos, key=lambda p: p["exit_ts"]):
        pnl = pos["stake"] * pos["r"]
        equity += pnl
        y = time.gmtime(pos["exit_ts"] / 1000).tm_year
        by_year[y] = by_year.get(y, 0.0) + pnl
        peak = max(peak, equity)
        max_dd = max(max_dd, (peak - equity) / peak)
    if not events:
        return {"trades_taken": 0}
    years = max((events[-1]["exit_ts"] - events[0]["entry_ts"]) / (365.25 * DAY_MS), 1e-9)
    return {"trades_taken": taken, "final_equity": round(equity, 2),
            "cagr_pct": round(((equity / start) ** (1 / years) - 1) * 100, 2),
            "max_drawdown_pct": round(max_dd * 100, 2),
            "pnl_by_year": {str(y): round(v, 2) for y, v in sorted(by_year.items())}}


def buy_and_hold(k, start_ts, end_ts):
    t, c = k["t"], k["c"]
    i0, i1 = int(np.searchsorted(t, start_ts)), int(np.searchsorted(t, end_ts, side="right")) - 1
    if i1 <= i0:
        return None
    px = c[i0:i1 + 1]
    dd = float(np.max(1 - px / np.maximum.accumulate(px)))
    years = (t[i1] - t[i0]) / (365.25 * DAY_MS)
    return {"cagr_pct": round(((px[-1] / px[0]) ** (1 / years) - 1) * 100, 2),
            "max_drawdown_pct": round(dd * 100, 2)}


def robustness(keys=("T_don20_long", "T_don55_long", "T_sma100_long"), cutoff_ms=None, hist_dir=None):
    """حساسیتِ خانوادهٔ روند: هزینهٔ سنگین‌تر، جهانِ کوچک‌تر، زیرِدوره‌ها، نمای حساب."""
    cutoff_ms = int(cutoff_ms if cutoff_ms is not None else dev_cutoff_ms())
    daily = load_panel("1d", cutoff_ms, hist_dir=hist_dir)
    hist = {s: {"t": k["t"].tolist(), "c": k["c"].tolist(), "v": k["v"].tolist()} for s, k in daily.items()}
    snaps20 = universe.snapshots_from_histories(hist, top_n=20)
    snaps10 = universe.snapshots_from_histories(hist, top_n=10)
    specs = {v["key"]: v for v in VARIANTS}
    out = {}
    for key in keys:
        spec = specs[key]
        tr20 = trend_trades(daily, snaps20, spec)
        tr10 = trend_trades(daily, snaps10, spec)
        both = [t for t in tr20 if t["sym"] in ("BTCUSDT", "ETHUSDT")]

        def mean(rows, f=lambda t: t["net_r"]):
            return round(float(np.mean([f(t) for t in rows])), 4) if rows else None

        periods = {}
        for name, lo, hi in (("2018-2020", 2018, 2020), ("2021-2022", 2021, 2022), ("2023-2024", 2023, 2024)):
            rows = [t for t in tr20 if lo <= time.gmtime(t["entry_ts"] / 1000).tm_year <= hi]
            periods[name] = {"n": len(rows), "mean": mean(rows)}
        out[key] = {
            "top20": {"n": len(tr20), "mean": mean(tr20), "mean_stressed": mean(tr20, stressed_r)},
            "top10": {"n": len(tr10), "mean": mean(tr10), "mean_stressed": mean(tr10, stressed_r)},
            "btc_eth": {"n": len(both), "mean": mean(both), "mean_stressed": mean(both, stressed_r)},
            "periods": periods,
            "account_top20": portfolio(tr20),
            "account_btc_eth": portfolio(both, max_open=2),
        }
    if "BTCUSDT" in daily and out:
        first = min(t["entry_ts"] for t in trend_trades(daily, snaps20, specs[keys[0]]))
        out["btc_buy_and_hold"] = buy_and_hold(daily["BTCUSDT"], first, cutoff_ms)
    return out


def write(report, out_dir=None):
    out_dir = out_dir or OUT_DIR
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, time.strftime("explore_%Y%m%d_%H%M.json", time.gmtime(report["generated_at"])))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=1, default=float)
    return path


def table(report):
    lines = [f"{'variant':18s} {'n':>5s} {'n_eff':>6s} {'mean':>8s} {'lcb':>8s} {'PF':>6s} "
             f"{'p_adj':>6s} {'halves':>15s} {'yrs+':>5s} {'fund≈':>8s}  verdict"]
    for key, blob in report["variants"].items():
        s, rw = blob["stats"], blob.get("romano_wolf") or {}
        fmt = lambda x, d=3: ("—" if x is None else f"{x:.{d}f}")
        halves = f"{fmt(s.get('first_half_mean'), 2)}/{fmt(s.get('second_half_mean'), 2)}"
        lines.append(f"{key:18s} {s.get('n', 0):5d} {fmt(s.get('n_eff'), 0):>6s} {fmt(s.get('mean')):>8s} "
                     f"{fmt(s.get('lcb')):>8s} {fmt(s.get('profit_factor'), 2):>6s} "
                     f"{fmt(rw.get('p_adj'), 3):>6s} {halves:>15s} {s.get('positive_years', '—'):>5s} "
                     f"{fmt(s.get('mean_typical_funding')):>8s}  {blob['verdict']}")
    return "\n".join(lines)


if __name__ == "__main__":
    rep = run()
    rep["robustness"] = robustness(cutoff_ms=rep["cutoff_ms"])
    print(table(rep))
    print("saved:", write(rep))
