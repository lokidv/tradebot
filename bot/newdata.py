# -*- coding: utf-8 -*-
"""فاز ۲ (فرضیه‌های دادهٔ تازه) و فاز ۳ (مدلِ ترکیبیِ درصدِ حضور) — پیش‌ثبت:
data/research/prereg_new_data_ensemble.json (+ اصلاحیهٔ منبعِ S&P 500).

همان قراردادِ زمانیِ timing.py: موقعیتِ روزِ d فقط از داده‌های تا بسته‌شدنِ روزِ d−1.
"""
import time

import numpy as np

import extdata
import stats
import timing

DAY_MS = timing.DAY_MS
NEW_HYPOTHESES = ("N1_STABLE_GROWTH", "N2_MVRV_NOT_HOT", "N3_DVOL_CALM", "N4_SPX_RISK_ON",
                  "N5_DXY_WEAK", "N6_FNG_NOT_GREEDY")
MOM_LOOKBACKS = (14, 28, 56)


def _series(name):
    rows = extdata.load(name)
    return np.asarray([r[0] for r in rows], np.int64), np.asarray([r[1] for r in rows], float)


def _asof_idx(keys, ms):
    return int(np.searchsorted(keys, ms, side="right")) - 1


def new_positions(hyp, sym, t):
    """آرایهٔ ۰/۱ برای روزهای ``t``؛ ``nan`` یعنی دادهٔ کافی نبود (بیرون از بازار حساب می‌شود)."""
    n = len(t)
    pos = np.full(n, np.nan)
    asset = "btc" if sym.startswith("BTC") else "eth"
    if hyp == "N1_STABLE_GROWTH":
        k, v = _series("stablecoin_supply")
        for d in range(n):
            i = _asof_idx(k, t[d] - DAY_MS)
            j = _asof_idx(k, t[d] - DAY_MS - 30 * DAY_MS)
            if i >= 0 and j >= 0 and k[i] - k[j] >= 25 * DAY_MS:
                pos[d] = 1.0 if v[i] > v[j] else 0.0
    elif hyp == "N2_MVRV_NOT_HOT":
        k, v = _series(f"mvrv_{asset}")
        for d in range(n):
            i = _asof_idx(k, t[d] - DAY_MS)
            if i < 0:
                continue
            past = v[(k > k[i] - 730 * DAY_MS) & (k < k[i])]
            if len(past) >= 600:
                pos[d] = 0.0 if v[i] > np.percentile(past, 80) else 1.0
    elif hyp == "N3_DVOL_CALM":
        k, v = _series(f"dvol_{asset}")
        m7 = np.array([v[max(0, i - 6):i + 1].mean() for i in range(len(v))])
        for d in range(n):
            i = _asof_idx(k, t[d] - DAY_MS)
            if i < 6:
                continue
            past = m7[(k > k[i] - 365 * DAY_MS) & (k < k[i])]
            if len(past) >= 300:
                pos[d] = 1.0 if m7[i] < np.percentile(past, 70) else 0.0
    elif hyp == "N4_SPX_RISK_ON":
        k, v = _series("spx")
        cs = np.concatenate([[0.0], np.cumsum(v)])
        for d in range(n):
            i = _asof_idx(k, t[d] - DAY_MS)
            if i >= 199:
                pos[d] = 1.0 if v[i] > (cs[i + 1] - cs[i - 199]) / 200 else 0.0
    elif hyp == "N5_DXY_WEAK":
        k, v = _series("dxy_broad")
        cs = np.concatenate([[0.0], np.cumsum(v)])
        for d in range(n):
            i = _asof_idx(k, t[d] - 8 * DAY_MS)                    # d−1 منهای ۷ روز تأخیرِ انتشار
            if i >= 99:
                pos[d] = 1.0 if v[i] < (cs[i + 1] - cs[i - 99]) / 100 else 0.0
    elif hyp == "N6_FNG_NOT_GREEDY":
        k, v = _series("fear_greed")
        for d in range(n):
            i = _asof_idx(k, t[d] - DAY_MS)
            if i >= 6 and k[i] - k[i - 6] <= 8 * DAY_MS:
                pos[d] = 0.0 if v[i - 6:i + 1].mean() > 75 else 1.0
    else:
        raise ValueError(hyp)
    return pos


def momentum_vote(t, c, lookback):
    """رأیِ مومنتوم با تصمیمِ دوشنبه: بستهٔ یکشنبه > بستهٔ ``lookback`` روز قبلش."""
    n = len(c)
    pos = np.zeros(n)
    cur = 0.0
    for d in range(lookback + 1, n):
        if time.gmtime(t[d] / 1000).tm_wday == 0:
            cur = 1.0 if c[d - 1] > c[d - 1 - lookback] else 0.0
        pos[d] = cur
    return pos


def exposure_returns(expo, r, cost=timing.SWITCH_COST):
    prev = np.concatenate([[0.0], expo[:-1]])
    return expo * r - cost * np.abs(expo - prev)


def phase2(hist_dir=None):
    res, family = {}, {}
    for sym in timing.ASSETS:
        t, c, _f = timing.load(sym, hist_dir)
        r = timing.daily_returns(c)
        for hyp in NEW_HYPOTHESES:
            raw = new_positions(hyp, sym, t)
            pos = np.nan_to_num(raw, nan=0.0)
            disc, e_d, t_d = timing.window_stats(pos, r, t, timing.DISCOVERY)
            conf, _e, _t = timing.window_stats(pos, r, t, timing.CONFIRMATION)
            m = (t >= timing.DISCOVERY[0]) & (t < timing.CONFIRMATION[1])
            key = f"{sym}:{hyp}"
            res[key] = {"discovery": disc, "confirmation": conf,
                        "missing_data_days": int(np.isnan(raw[m]).sum())}
            if e_d is not None:
                family[key] = (e_d, t_d)
    rw = stats.romano_wolf(family, timing.BLOCK_MS, alpha=timing.RW_ALPHA, B=2000)
    for key, v in res.items():
        v["romano_wolf_discovery"] = rw.get(key, {})
        v["verdict"] = timing.decide(v["discovery"], v["confirmation"], (rw.get(key) or {}).get("p_adj"))
    return res


def exposure_stats(expo, r, t, win):
    m = (t >= win[0]) & (t < win[1])
    ret = exposure_returns(expo, r)[m]
    return {"strategy": timing.perf(ret, t[m]), "buy_and_hold": timing.perf(r[m], t[m]),
            "avg_exposure_pct": round(float(expo[m].mean()) * 100, 1),
            "turnover_per_year": round(float(np.abs(np.diff(expo[m])).sum()) / max(m.sum() / 365.25, 1e-9), 2)}


def ensemble(adopted_by_asset, hist_dir=None):
    """E1 = میانگینِ سه رأیِ مومنتوم؛ E2 = E1 + رأیِ فرضیه‌های پذیرفته‌شدهٔ فاز ۲ (اگر باشد)."""
    out = {}
    for sym in timing.ASSETS:
        t, c, f = timing.load(sym, hist_dir)
        r = timing.daily_returns(c)
        votes = [momentum_vote(t, c, L) for L in MOM_LOOKBACKS]
        e1 = np.mean(votes, axis=0)
        h3 = timing.positions("H3_TSMOM28", t, c, timing.funding_7d(t, f))
        models = {"H3_TSMOM28": h3, "E1_MOM3": e1}
        extra = [np.nan_to_num(new_positions(h, sym, t), nan=0.0) for h in adopted_by_asset.get(sym, [])]
        if extra:
            models["E2_MOM3_PLUS"] = np.mean(votes + extra, axis=0)
        res = {}
        for name, expo in models.items():
            res[name] = {"discovery": exposure_stats(expo, r, t, timing.DISCOVERY),
                         "confirmation": exposure_stats(expo, r, t, timing.CONFIRMATION)}
        h3c = res["H3_TSMOM28"]["confirmation"]["strategy"]
        chosen = "H3_TSMOM28"
        for name in ("E2_MOM3_PLUS", "E1_MOM3"):                  # ترجیحِ ازپیش‌گفته: E2 اگر هست، بعد E1
            if name in res:
                s = res[name]["confirmation"]["strategy"]
                if s["sharpe"] >= h3c["sharpe"] and s["max_drawdown_pct"] <= h3c["max_drawdown_pct"]:
                    chosen = name
                    break
        out[sym] = {"models": res, "chosen": chosen}
    return out


def run(hist_dir=None):
    p2 = phase2(hist_dir)
    adopted = {}
    for key, v in p2.items():
        if v["verdict"] == "ADOPT":
            sym, hyp = key.split(":")
            adopted.setdefault(sym, []).append(hyp)
    return {"generated_at": time.time(), "prereg": "bot/data/research/prereg_new_data_ensemble.json",
            "phase2": p2, "adopted": adopted, "phase3": ensemble(adopted, hist_dir)}
