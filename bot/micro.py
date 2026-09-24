# -*- coding: utf-8 -*-
"""سیگنال‌های ۱۵دقیقه‌ای برای فیوچرزِ کوکوین — پیش‌ثبت: data/research/prereg_15m_microstructure.json.

هر سیگنال در بسته‌شدنِ یک کندلِ ۱۵دقیقه‌ای فقط از همان کندل و قبل‌تر ساخته می‌شود؛ ورود در openِ
کندلِ بعد، خروج در بستهٔ کندلِ هشتم (۲ ساعت) یا حدضررِ محافظِ 3×ATR14. هر ارز یک پوزیشن.
"""
import os
import time

import numpy as np
import pandas as pd

import engine
import paths
import stats

BAR_MS = 900_000
DAY_MS = 86_400_000
LOOK = 2880                       # ۳۰ روز کندلِ ۱۵دقیقه‌ای
HOLD = 8                          # ۲ ساعت
STOP_ATR = 3.0
COST_TAKER = 0.14                 # ٪ رفت‌وبرگشت
COST_MAKER = 0.04
HYPOTHESES = ("M1_FLOW_MOMENTUM", "M2_ABSORPTION", "M3_CAPITULATION", "M4_BREAKOUT_FLOW", "M5_US_OPEN")
WINDOWS = {"discovery": ("2022-01-01", "2024-07-01"), "validation": ("2024-07-01", "2025-07-01"),
           "holdout": ("2025-07-01", "2026-09-01")}


def _ms(s):
    return int(pd.Timestamp(s, tz="UTC").timestamp() * 1000)


def load(sym, market="um"):
    d = np.load(paths.data("hist_research", "micro", f"{market}_{sym}_15m.npz"))
    return {k: d[k] for k in d.files}


def _prior_z(x, look=LOOK):
    """z-score نسبت به ``look`` مقدارِ **قبلی** (خودِ کندلِ جاری در میانگین و انحراف نیست)."""
    s = pd.Series(x)
    m = s.shift(1).rolling(look, min_periods=look).mean()
    sd = s.shift(1).rolling(look, min_periods=look).std()
    return ((s - m) / sd).to_numpy()


def features(k):
    c, v, tbv = k["c"], k["v"], k["tbv"]
    sv = pd.Series(v).rolling(4).sum().to_numpy()
    stb = pd.Series(tbv).rolling(4).sum().to_numpy()
    imb4 = np.where(sv > 0, stb / np.where(sv > 0, sv, 1), np.nan)
    r1 = np.concatenate([[np.nan], c[1:] / c[:-1] - 1])
    r4 = np.concatenate([[np.nan] * 4, c[4:] / c[:-4] - 1])
    sig1 = pd.Series(r1).shift(1).rolling(LOOK, min_periods=LOOK).std().to_numpy()
    sig4 = pd.Series(r4).shift(1).rolling(LOOK, min_periods=LOOK).std().to_numpy()
    vmed = pd.Series(v).shift(1).rolling(LOOK, min_periods=LOOK).median().to_numpy()
    hh = pd.Series(k["h"]).shift(1).rolling(96, min_periods=96).max().to_numpy()
    ll = pd.Series(k["l"]).shift(1).rolling(96, min_periods=96).min().to_numpy()
    return {"imb4": imb4, "zimb": _prior_z(imb4), "r1": r1, "r4": r4, "sig1": sig1, "sig4": sig4,
            "vmed": vmed, "hh": hh, "ll": ll, "atr": engine.atr(k["h"], k["l"], c, 14)}


def signals(hyp, k, f=None):
    """آرایهٔ −۱/۰/+۱ در بسته‌شدنِ هر کندل."""
    f = f or features(k)
    n = len(k["c"])
    s = np.zeros(n)
    with np.errstate(invalid="ignore"):
        if hyp == "M1_FLOW_MOMENTUM":
            s[f["zimb"] > 2] = 1
            s[f["zimb"] < -2] = -1
        elif hyp == "M2_ABSORPTION":
            s[(f["r4"] < -f["sig4"]) & (f["zimb"] > 1.5)] = 1
            s[(f["r4"] > f["sig4"]) & (f["zimb"] < -1.5)] = -1
        elif hyp == "M3_CAPITULATION":
            spike = k["v"] > 4 * f["vmed"]
            s[(f["r1"] < -4 * f["sig1"]) & spike] = 1
            s[(f["r1"] > 4 * f["sig1"]) & spike] = -1
        elif hyp == "M4_BREAKOUT_FLOW":
            s[(k["c"] > f["hh"]) & (f["zimb"] > 1)] = 1
            s[(k["c"] < f["ll"]) & (f["zimb"] < -1)] = -1
        elif hyp == "M5_US_OPEN":
            t = k["t"]
            tod = (t % DAY_MS) // BAR_MS                       # شمارهٔ کندل در روز (UTC)
            wd = ((t // DAY_MS) + 3) % 7                        # ۰ = دوشنبه (1970-01-01 پنجشنبه بود)
            idx = np.flatnonzero((tod == 55) & (wd < 5))        # کندلِ 13:45 که در 14:00 بسته می‌شود
            for i in idx:
                if i >= 2:
                    r = k["c"][i] / k["c"][i - 2] - 1           # بستهٔ 13:30 تا بستهٔ 14:00 ≈ 13:30→14:00
                    s[i] = 1 if r > 0 else (-1 if r < 0 else 0)
        else:
            raise ValueError(hyp)
    s[~np.isfinite(f["atr"])] = 0
    return s


def trades(k, sig, cost=COST_TAKER, hold=HOLD):
    o, h, l, c, t = k["o"], k["h"], k["l"], k["c"], k["t"]
    atr = engine.atr(h, l, c, 14)
    n = len(c)
    out, i = [], 0
    while i < n - hold - 1:
        side = sig[i]
        if side == 0:
            i += 1
            continue
        e = i + 1
        entry = o[e]
        stop = entry - side * STOP_ATR * atr[i]
        exit_px, j_exit = c[e + hold - 1], e + hold - 1
        for j in range(e, e + hold):
            if (side > 0 and o[j] <= stop) or (side < 0 and o[j] >= stop):
                exit_px, j_exit = o[j], j
                break
            if (side > 0 and l[j] <= stop) or (side < 0 and h[j] >= stop):
                exit_px, j_exit = stop, j
                break
        gross = side * (exit_px / entry - 1) * 100
        out.append((int(t[i]), int(side), gross, gross - cost))
        i = j_exit                                      # سیگنالِ بعدی از کندلِ خروج به بعد
    return np.asarray(out, float).reshape(-1, 4)


def window_stats(tr, win):
    lo, hi = _ms(win[0]), _ms(win[1])
    m = (tr[:, 0] >= lo) & (tr[:, 0] < hi) if len(tr) else np.zeros(0, bool)
    v, ts = tr[m, 3], tr[m, 0]
    if len(v) == 0:
        return {"n": 0, "mean": None}, v, ts
    s = stats.summarize(v, ts, DAY_MS, alpha=0.10, B=1000)
    s["mean_gross"] = round(float(tr[m, 2].mean()), 4)
    s["mean_maker_cost"] = round(float((tr[m, 2] - COST_MAKER).mean()), 4)
    s["long_share"] = round(float((tr[m, 1] > 0).mean()), 3)
    s["trades_per_day"] = round(len(v) / max((hi - lo) / DAY_MS, 1), 3)
    return s, v, ts


def study(symbols):
    res, family = {}, {}
    for sym in symbols:
        k = load(sym)
        f = features(k)
        for hyp in HYPOTHESES:
            tr = trades(k, signals(hyp, k, f))
            key = f"{sym}:{hyp}"
            res[key] = {}
            for name in ("discovery", "validation"):
                s, v, ts = window_stats(tr, WINDOWS[name])
                res[key][name] = s
                if name == "discovery" and len(v) >= 2:
                    family[key] = (v, ts)
            res[key]["_trades"] = tr
    rw = stats.romano_wolf(family, DAY_MS, alpha=0.10, B=1000)
    for key, v in res.items():
        d, val = v["discovery"], v["validation"]
        p = (rw.get(key) or {}).get("p_adj")
        v["p_adj"] = p
        d_ok = (d.get("n") or 0) >= 100 and (d.get("mean") or 0) > 0 and p is not None and p < 0.10
        v_ok = (val.get("n") or 0) >= 50 and (val.get("mean") or 0) > 0 and (val.get("lcb") or -1) > 0
        v["passed_discovery"], v["passed_validation"] = d_ok, d_ok and v_ok
    return res


def holdout(res):
    """آزمونِ یک‌باره روی ۱۴ ماهِ آخر — فقط برای آن‌هایی که هر دو مرحله را گذرانده‌اند."""
    out = {}
    for key, v in res.items():
        if v.get("passed_validation"):
            s, _v, _t = window_stats(v["_trades"], WINDOWS["holdout"])
            out[key] = {"stats": s, "pass": bool((s.get("mean") or 0) > 0)}
    return out
