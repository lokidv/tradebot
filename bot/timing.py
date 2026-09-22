# -*- coding: utf-8 -*-
"""زمان‌بندیِ «خرید یا نقد» روی BTC و ETH — پیش‌ثبت: data/research/prereg_btc_eth_timing.json.

کاربر فقط BTC و ETH را، اسپات و با پولِ خودش، معامله می‌کند. این ماژول چهار فرضیهٔ ازپیش‌ثبت‌شده
را می‌سنجد؛ دو تایشان از منبعی می‌آیند که پیش از این برای پیش‌بینی استفاده نشده بود: تاریخچهٔ
۶سالهٔ نرخِ فاندینگ (شلوغیِ لانگ‌ها و شورت‌های اهرمی).

قاعدهٔ زمان: موقعیتِ روزِ d (۰۰:۰۰ تا ۲۴:۰۰ UTC) فقط از داده‌های تا بسته‌شدنِ روزِ d−1 ساخته
می‌شود؛ تسویهٔ فاندینگ فقط اگر در یا پیش از همان لحظه باشد.
"""
import json
import os
import time

import numpy as np

import paths
import stats

DAY_MS = 86_400_000
ASSETS = ("BTCUSDT", "ETHUSDT")
HYPOTHESES = ("H1_FUND_LOW", "H2_FUND_NOT_HIGH", "H3_TSMOM28", "H4_SMA200")
SWITCH_COST = 0.002                        # ۰٫۲٪ ارزشِ پوزیشن در هر تغییر (۰٫۴٪ رفت‌وبرگشت)
DISCOVERY = (1_609_459_200_000, 1_688_169_600_000)        # 2021-01-01 .. 2023-07-01 (انحصاری)
CONFIRMATION = (1_688_169_600_000, 1_788_220_800_000)     # 2023-07-01 .. 2026-09-01 (انحصاری)
RW_ALPHA = 0.10
BLOCK_MS = 30 * DAY_MS
FUND_WINDOW_MS = 7 * DAY_MS
PCTL_DAYS = 365


def load(sym, hist_dir=None):
    hist_dir = hist_dir or paths.data("hist")
    with open(os.path.join(hist_dir, f"{sym}_1d.json"), encoding="utf-8") as f:
        d = json.load(f)
    with open(os.path.join(paths.data("hist_research"), f"funding_archive_{sym}.json"), encoding="utf-8") as f:
        fund = json.load(f)
    return np.asarray(d["t"], np.int64), np.asarray(d["c"], float), np.asarray(fund, float)


def funding_7d(t, fund):
    """میانگینِ فاندینگِ ۷ روزِ منتهی به لحظهٔ تصمیمِ هر روز (= openِ روز = بسته‌شدنِ دیروز)."""
    fts, rates = fund[:, 0], fund[:, 1]
    csum = np.concatenate([[0.0], np.cumsum(rates)])
    out = np.full(len(t), np.nan)
    for d in range(len(t)):
        hi = np.searchsorted(fts, t[d], side="right")              # تسویه‌های ≤ لحظهٔ تصمیم
        lo = np.searchsorted(fts, t[d] - FUND_WINDOW_MS, side="right")
        if hi - lo >= 15:                                           # دستِ‌کم ۵ روز داده
            out[d] = (csum[hi] - csum[lo]) / (hi - lo)
    return out


def _pctl_prior(x, d, q):
    past = x[max(0, d - PCTL_DAYS):d]
    past = past[np.isfinite(past)]
    return np.percentile(past, q) if len(past) >= 300 else None


def positions(hyp, t, c, f7):
    """آرایهٔ ۰/۱: آیا روزِ d در بازار هستیم. فقط از c[:d] و f7[d] (تصمیم‌گرفته‌شده در openِ d)."""
    n = len(c)
    pos = np.zeros(n)
    if hyp == "H1_FUND_LOW":
        hold = 0
        for d in range(1, n):
            p20 = _pctl_prior(f7, d, 20)
            if p20 is not None and np.isfinite(f7[d]) and f7[d] < p20:
                hold = 7
            pos[d] = 1.0 if hold > 0 else 0.0
            hold = max(hold - 1, 0)
    elif hyp == "H2_FUND_NOT_HIGH":
        for d in range(1, n):
            p80 = _pctl_prior(f7, d, 80)
            pos[d] = 0.0 if (p80 is None or not np.isfinite(f7[d]) or f7[d] > p80) else 1.0
    elif hyp == "H3_TSMOM28":
        cur = 0.0
        for d in range(29, n):
            if time.gmtime(t[d] / 1000).tm_wday == 0:                  # دوشنبه
                cur = 1.0 if c[d - 1] > c[d - 29] else 0.0
            pos[d] = cur
    elif hyp == "H4_SMA200":
        cs = np.concatenate([[0.0], np.cumsum(c)])
        for d in range(200, n):
            pos[d] = 1.0 if c[d - 1] > (cs[d] - cs[d - 200]) / 200 else 0.0
    else:
        raise ValueError(hyp)
    return pos


def daily_returns(c):
    r = np.zeros(len(c))
    r[1:] = c[1:] / c[:-1] - 1
    return r


def strategy_returns(pos, r, cost=SWITCH_COST):
    prev = np.concatenate([[0.0], pos[:-1]])
    return pos * r - cost * np.abs(pos - prev)


def edge_series(pos, r):
    """سری‌ای که میانگینش دقیقاً «میانگینِ بازده در بازار منهای بیرونِ بازار» است."""
    p = float(pos.mean())
    if p <= 0 or p >= 1:
        return None
    return (pos - p) * r / (p * (1 - p))


def perf(ret, t):
    eq = np.cumprod(1 + ret)
    years = max((t[-1] - t[0]) / (365.25 * DAY_MS), 1e-9)
    sd = ret.std()
    return {"cagr_pct": round((eq[-1] ** (1 / years) - 1) * 100, 2),
            "max_drawdown_pct": round(float(np.max(1 - eq / np.maximum.accumulate(eq))) * 100, 2),
            "sharpe": round(float(ret.mean() / sd * np.sqrt(365)) if sd > 0 else 0.0, 3)}


def window_stats(pos, r, t, win):
    m = (t >= win[0]) & (t < win[1])
    p, rr, tt = pos[m], r[m], t[m]
    e = edge_series(p, rr)
    strat = strategy_returns(pos, r)[m]
    return {"days": int(m.sum()), "time_in_market_pct": round(float(p.mean()) * 100, 1),
            "edge_daily_pct": round(float(e.mean()) * 100, 4) if e is not None else None,
            "mean_in_pct": round(float(rr[p > 0].mean()) * 100, 4) if (p > 0).any() else None,
            "mean_out_pct": round(float(rr[p == 0].mean()) * 100, 4) if (p == 0).any() else None,
            "strategy": perf(strat, tt), "buy_and_hold": perf(rr, tt),
            "switches": int(np.abs(np.diff(p)).sum())}, e, tt


def decide(disc, conf, p_adj):
    d_ok = disc["edge_daily_pct"] is not None and disc["edge_daily_pct"] > 0 and p_adj is not None and p_adj < RW_ALPHA
    c_ok = (conf["edge_daily_pct"] is not None and conf["edge_daily_pct"] > 0
            and conf["strategy"]["sharpe"] >= conf["buy_and_hold"]["sharpe"])
    if d_ok and c_ok:
        return "ADOPT"
    both_pos = (disc["edge_daily_pct"] or 0) > 0 and (conf["edge_daily_pct"] or 0) > 0
    return "CONSISTENT" if both_pos else "REJECT"


def study(hist_dir=None):
    res, family, series = {}, {}, {}
    for sym in ASSETS:
        t, c, fund = load(sym, hist_dir)
        f7 = funding_7d(t, fund)
        r = daily_returns(c)
        for hyp in HYPOTHESES:
            pos = positions(hyp, t, c, f7)
            disc, e_d, t_d = window_stats(pos, r, t, DISCOVERY)
            conf, _e, _t = window_stats(pos, r, t, CONFIRMATION)
            key = f"{sym}:{hyp}"
            res[key] = {"discovery": disc, "confirmation": conf}
            if e_d is not None:
                family[key] = (e_d, t_d)
            series[key] = {"t": t.tolist(), "pos": pos.tolist(), "c": c.tolist()}
    rw = stats.romano_wolf(family, BLOCK_MS, alpha=RW_ALPHA, B=2000)
    for key, v in res.items():
        v["romano_wolf_discovery"] = rw.get(key, {})
        v["verdict"] = decide(v["discovery"], v["confirmation"], (rw.get(key) or {}).get("p_adj"))
    return {"generated_at": time.time(), "prereg": "bot/data/research/prereg_btc_eth_timing.json",
            "results": res}, series


def funding_quintiles(sym, win, hist_dir=None, horizon=7):
    """توصیفی: بازدهِ ۷ روزِ بعد به تفکیکِ پنجکِ فاندینگِ ۷روزه (نسبت به ۳۶۵ روزِ قبل)."""
    t, c, fund = load(sym, hist_dir)
    f7 = funding_7d(t, fund)
    rows = []
    for d in range(1, len(c) - horizon):
        if not (win[0] <= t[d] < win[1]) or not np.isfinite(f7[d]):
            continue
        past = f7[max(0, d - PCTL_DAYS):d]
        past = past[np.isfinite(past)]
        if len(past) < 300:
            continue
        q = int(min(4, (past < f7[d]).mean() * 5))
        rows.append((q, c[d + horizon - 1] / c[d - 1] - 1))
    out = {}
    for q in range(5):
        v = [x for qq, x in rows if qq == q]
        out[q + 1] = {"n": len(v), "mean_fwd7_pct": round(float(np.mean(v)) * 100, 2) if v else None}
    return out
