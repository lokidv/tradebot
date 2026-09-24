# -*- coding: utf-8 -*-
"""سیگنال‌های ۱۵دقیقه‌ای، دورِ دوم — پیش‌ثبت: data/research/prereg_15m_positioning.json.

دادهٔ تازه نسبت به دورِ اول (micro.py): open interest، نسبت‌های لانگ/شورت، نسبتِ حجمِ taker و
شاخصِ پرمیومِ فیوچرز؛ به‌علاوهٔ پیشروبودنِ BTC و یک مدلِ خطیِ walk-forward. مکانیکِ معامله، هزینه،
پنجره‌ها و قاعدهٔ پذیرش همان دورِ اول است (micro.trades / micro.window_stats).
"""
import os

import numpy as np
import pandas as pd

import micro
import paths
import stats

BAR_MS = micro.BAR_MS
METRIC_LAG_MS = 5 * 60_000            # یک ردیفِ ۵دقیقه‌ایِ تأخیرِ اضافه (پیش‌ثبت)
STALE_MS = 60 * 60_000                # ردیفی که بیش از یک ساعت کهنه باشد = داده نداریم
COINS = ("BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "TRXUSDT")
HYPOTHESES = ("P1_OI_UNWIND_FADE", "P2_OI_BUILD_FOLLOW", "P3_CROWD_FADE", "P4_TOP_VS_CROWD",
              "P5_BASIS_FADE", "P6_BTC_LEAD", "P7_LINEAR_MODEL")
LAMBDA = 1.0
MODEL_START = "2022-07-01"
COST_SECONDARY = 0.10


def _z(x, look=micro.LOOK):
    """z نسبت به ``look`` مقدارِ قبلی؛ خانه‌های خالیِ داده نادیده، دست‌کم نیمی از پنجره لازم."""
    s = pd.Series(x)
    p = s.shift(1).rolling(look, min_periods=look // 2)
    return ((s - p.mean()) / p.std()).to_numpy()


def align(t_src, v_src, t_want):
    """آخرین مقدارِ منبع با زمانِ <= t_want؛ اگر بیش از STALE_MS کهنه باشد NaN."""
    idx = np.searchsorted(t_src, t_want, side="right") - 1
    ok = idx >= 0
    out = np.full(len(t_want), np.nan)
    out[ok] = v_src[idx[ok]]
    stale = ~ok.copy()
    stale[ok] = t_want[ok] - t_src[idx[ok]] > STALE_MS
    out[stale] = np.nan
    return out


def _exact(t_src, v_src, t_want):
    idx = np.searchsorted(t_src, t_want)
    idx = np.clip(idx, 0, len(t_src) - 1)
    return np.where(t_src[idx] == t_want, v_src[idx], np.nan)


def load_extra(sym, k):
    t = k["t"]
    ask = t + BAR_MS - METRIC_LAG_MS
    pos = np.load(paths.data("hist_research", "micro", f"pos_{sym}.npz"))
    prem = np.load(paths.data("hist_research", "micro", f"prem_{sym}_15m.npz"))
    return {"oi": align(pos["t"], pos["oi"], ask), "acct": align(pos["t"], pos["acct_ratio"], ask),
            "top": align(pos["t"], pos["top_pos_ratio"], ask), "taker": align(pos["t"], pos["taker_ratio"], ask),
            "prem": _exact(prem["t"], prem["prem"], t)}


def features(k, x, btc=None):
    """btc: kline‌های BTC هم‌تراز با زمان‌های این ارز (یا None برای خودِ BTC)."""
    f = micro.features(k)
    c = k["c"]
    with np.errstate(divide="ignore", invalid="ignore"):
        doi4 = np.log(x["oi"] / np.concatenate([[np.nan] * 4, x["oi"][:-4]]))
    r16 = np.concatenate([[np.nan] * 16, c[16:] / c[:-16] - 1])
    f["sig16"] = pd.Series(r16).shift(1).rolling(micro.LOOK, min_periods=micro.LOOK).std().to_numpy()
    f["r16"] = r16
    f["zdoi"] = _z(doi4)
    for key in ("acct", "top", "taker", "prem"):
        f["z_" + key] = _z(x[key])
    b = btc if btc is not None else k
    bc = b["c"]
    f["btc_r1"] = np.concatenate([[np.nan], bc[1:] / bc[:-1] - 1])
    f["btc_r4"] = np.concatenate([[np.nan] * 4, bc[4:] / bc[:-4] - 1])
    f["btc_sig1"] = pd.Series(f["btc_r1"]).shift(1).rolling(micro.LOOK, min_periods=micro.LOOK).std().to_numpy()
    f["btc_sig4"] = pd.Series(f["btc_r4"]).shift(1).rolling(micro.LOOK, min_periods=micro.LOOK).std().to_numpy()
    return f


def design(k, f):
    hour = (k["t"] % micro.DAY_MS) / 3_600_000
    with np.errstate(divide="ignore", invalid="ignore"):
        cols = [f["zimb"], f["r1"] / f["sig1"], f["r4"] / f["sig4"], f["r16"] / f["sig16"], f["zdoi"],
                f["z_acct"], f["z_top"], f["z_taker"], f["z_prem"], f["btc_r4"] / f["btc_sig4"],
                np.sin(2 * np.pi * hour / 24), np.cos(2 * np.pi * hour / 24)]
    return np.column_stack(cols)


def fit_logistic(X, y, lam=LAMBDA, iters=25):
    """لجستیک با جریمهٔ L2 (بدونِ جریمه روی عرض از مبدأ)، روشِ نیوتن."""
    Xb = np.column_stack([np.ones(len(X)), X])
    w = np.zeros(Xb.shape[1])
    pen = np.full(Xb.shape[1], lam)
    pen[0] = 0.0
    for _ in range(iters):
        p = 1 / (1 + np.exp(-np.clip(Xb @ w, -30, 30)))
        g = Xb.T @ (p - y) + pen * w
        H = (Xb * (p * (1 - p))[:, None]).T @ Xb + np.diag(pen)
        step = np.linalg.solve(H, g)
        w -= step
        if np.max(np.abs(step)) < 1e-8:
            break
    return w


def _predict(w, X):
    return 1 / (1 + np.exp(-np.clip(np.column_stack([np.ones(len(X)), X]) @ w, -30, 30)))


def model_signals(k, f):
    """walk-forward: بازآموزی اولِ هر ماه روی ردیف‌هایی که هدفشان پیش از آن ماه معلوم شده."""
    X = design(k, f)
    t, o, c = k["t"], k["o"], k["c"]
    n = len(c)
    y = np.full(n, np.nan)
    y[:n - 9] = (c[8:n - 1] > o[1:n - 8]).astype(float)
    valid = np.isfinite(X).all(axis=1)
    s = np.zeros(n)
    months = pd.date_range(MODEL_START, pd.Timestamp(int(t[-1]), unit="ms"), freq="MS", tz="UTC")
    for m0, m1 in zip(months, list(months[1:]) + [None]):
        lo = int(m0.timestamp() * 1000)
        hi = int(m1.timestamp() * 1000) if m1 is not None else int(t[-1]) + 1
        tr = valid & np.isfinite(y) & (t + 9 * BAR_MS <= lo)
        te = valid & (t >= lo) & (t < hi)
        if tr.sum() < 5000 or not te.any():
            continue
        mu, sd = X[tr].mean(0), X[tr].std(0)
        sd[sd == 0] = 1
        w = fit_logistic((X[tr] - mu) / sd, y[tr])
        p_in = _predict(w, (X[tr] - mu) / sd)
        hi_q, lo_q = np.quantile(p_in, 0.95), np.quantile(p_in, 0.05)
        p = _predict(w, (X[te] - mu) / sd)
        s[np.flatnonzero(te)] = np.where(p >= hi_q, 1, np.where(p <= lo_q, -1, 0))
    return s


def signals(hyp, k, f):
    n = len(k["c"])
    s = np.zeros(n)
    with np.errstate(invalid="ignore"):
        up, dn = f["r4"] > f["sig4"], f["r4"] < -f["sig4"]
        if hyp == "P1_OI_UNWIND_FADE":
            s[dn & (f["zdoi"] < -2)] = 1
            s[up & (f["zdoi"] < -2)] = -1
        elif hyp == "P2_OI_BUILD_FOLLOW":
            s[up & (f["zdoi"] > 2)] = 1
            s[dn & (f["zdoi"] > 2)] = -1
        elif hyp == "P3_CROWD_FADE":
            s[f["z_acct"] > 2] = -1
            s[f["z_acct"] < -2] = 1
        elif hyp == "P4_TOP_VS_CROWD":
            d = f["z_top"] - f["z_acct"]
            s[d > 2] = 1
            s[d < -2] = -1
        elif hyp == "P5_BASIS_FADE":
            s[f["z_prem"] > 2.5] = -1
            s[f["z_prem"] < -2.5] = 1
        elif hyp == "P6_BTC_LEAD":
            s[(f["btc_r1"] > 3 * f["btc_sig1"]) & (f["r1"] < f["sig1"])] = 1
            s[(f["btc_r1"] < -3 * f["btc_sig1"]) & (f["r1"] > -f["sig1"])] = -1
        elif hyp == "P7_LINEAR_MODEL":
            s = model_signals(k, f)
        else:
            raise ValueError(hyp)
    s[~np.isfinite(f["atr"])] = 0
    return s


def _btc_aligned(k, kb):
    idx = np.clip(np.searchsorted(kb["t"], k["t"]), 0, len(kb["t"]) - 1)
    hit = kb["t"][idx] == k["t"]
    return {"c": np.where(hit, kb["c"][idx], np.nan)}


def study(symbols=COINS):
    kb = micro.load("BTCUSDT")
    res, family = {}, {}
    for sym in symbols:
        k = micro.load(sym)
        f = features(k, load_extra(sym, k), None if sym == "BTCUSDT" else _btc_aligned(k, kb))
        for hyp in HYPOTHESES:
            if hyp == "P6_BTC_LEAD" and sym == "BTCUSDT":
                continue
            tr = micro.trades(k, signals(hyp, k, f))
            key = f"{sym}:{hyp}"
            res[key] = {}
            for name in ("discovery", "validation"):
                st, v, ts = micro.window_stats(tr, micro.WINDOWS[name])
                if len(v):
                    lo, hi = micro._ms(micro.WINDOWS[name][0]), micro._ms(micro.WINDOWS[name][1])
                    m = (tr[:, 0] >= lo) & (tr[:, 0] < hi)
                    st["mean_cost_0_10"] = round(float((tr[m, 2] - COST_SECONDARY).mean()), 4)
                res[key][name] = st
                if name == "discovery" and len(v) >= 2:
                    family[key] = (v, ts)
            res[key]["_trades"] = tr
    rw = stats.romano_wolf(family, micro.DAY_MS, alpha=0.10, B=1000)
    for key, v in res.items():
        d, val = v["discovery"], v["validation"]
        p = (rw.get(key) or {}).get("p_adj")
        v["p_adj"] = p
        d_ok = (d.get("n") or 0) >= 100 and (d.get("mean") or 0) > 0 and p is not None and p < 0.10
        v_ok = (val.get("n") or 0) >= 50 and (val.get("mean") or 0) > 0 and (val.get("lcb") or -1) > 0
        v["passed_discovery"], v["passed_validation"] = d_ok, d_ok and v_ok
    return res


def holdout(res):
    """یک‌باره، فقط برای آن‌هایی که هر دو مرحله را گذرانده‌اند."""
    out = {}
    for key, v in res.items():
        if v.get("passed_validation"):
            s, _v, _t = micro.window_stats(v["_trades"], micro.WINDOWS["holdout"])
            out[key] = {"stats": s, "pass": bool((s.get("n") or 0) >= 30 and (s.get("mean") or 0) > 0)}
    return out
