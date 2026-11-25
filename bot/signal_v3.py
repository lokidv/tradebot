# -*- coding: utf-8 -*-
"""سیگنالِ v3 — قاعدهٔ ماتریس با دو تغییر، فقط روی 4h و 1d (``research/prereg_signal_v3.json``؛ الزام‌آور).

قاعدهٔ فعلی (``decision.side_series``) در 5m/15m/1h فقط کارمزد می‌پردازد و در 4h/1d سربه‌سر است. v3 همان
قاعده است با دو تفاوت، و فقط همان دو تایم‌فریم:

۱. دو ستاپِ «فیدِ روند» و «شکستِ فشردگی» انگار رخ نداده‌اند (اسکن به کندلِ قبل و سپس قاعدهٔ z/رأی می‌رود)؛
۲. هم‌جهت با حرکتِ ۲۸ روزهٔ همان ارز: خرید فقط وقتی بستهٔ کندلِ تصمیم بالاتر از بستهٔ ۲۸ روز پیش است،
   فروش فقط وقتی پایین‌تر.

شرطِ سومی که اول اعلام شد («دست‌کم ۴ رأی از ۵») در بررسیِ طراحی روی ۲۰۲۲ تا ۲۰۲۵ هیچ کمکی نکرد و در روزانه
معامله‌های بهتر را حذف می‌کرد؛ پیش از ثبت کنار گذاشته شد (``min_votes`` فقط برای مقایسهٔ توصیفی مانده).

براکت همان ``bracket.py`` است (حدضرر 1R، هدف 1.8R، سقفِ ۴۰ کندل) و هزینه همان ۰٫۱۴٪.

این دو تغییر از ممیزی و آزمونِ مومنتوم آمده‌اند و روی دادهٔ گذشته پیدا شده‌اند؛ مدرک نیستند. تنها داورِ v3
کارنامهٔ رو-به-جلوی آن از ``T0_MS`` است: بازپخشِ قطعیِ همین تابع روی کندل‌های بسته‌شدهٔ اسپاتِ بایننس، همان
کندل‌هایی که سیگنالِ زنده از آن‌ها ساخته می‌شود. نمایش، ترمز و آزمون هر سه از همین یک مسیر می‌آیند.

اختیار: v3 هرگز به gates.json نمی‌نویسد، autotrader آن را نمی‌خواند، و دفترِ قاعدهٔ اصلی
(``decision_ledger.jsonl``) و آزمونِ پیش‌ثبت‌شدهٔ گونه‌ها دست نمی‌خورند.
"""
import json
import math
import os
import threading
import time

import numpy as np

import bracket
import decision
import engine
import log
import paths

# ───────────────────────── ثابت‌های پیش‌ثبت (تغییر پس از دیدنِ نتیجه ممنوع) ─────────────────────────
SIGNAL_TFS = ("4h", "1d")                 # تنها تایم‌فریم‌هایی که سیگنال می‌دهند
SKIP_SETUPS = frozenset({"fd", "sq"})     # فیدِ روند و شکستِ فشردگی
MIN_VOTES = None                          # شرطِ رأی در مشخصاتِ ثبت‌شده نیست (بررسیِ طراحی: بی‌اثر در 4h، مضر در 1d)
MOM_DAYS = 28
DAY_MS = 86_400_000
T0_MS = 1_788_220_800_000                 # 2026-09-01 00:00 UTC — نخستین کندلِ سیگنالِ دورهٔ رو-به-جلو
SEALED_MS = (1_767_225_600_000, T0_MS)    # 2026-01-01..2026-08-31: holdoutِ مهروموم — هیچ سیگنالی در آن بازپخش نمی‌شود
WARMUP_BARS = 2000                        # کندل‌های پیش از T0 که فقط اندیکاتورها را گرم می‌کنند
COST_PCT = decision.COST_PCT              # 0.14
MAKER_COST_PCT = decision.MAKER_COST_PCT  # 0.10 — فقط توصیفی
MAX_BARS = bracket.MAX_BARS               # 40
WEEK_MS = 7 * DAY_MS
MONDAY0_MS = 4 * DAY_MS                   # 1970-01-05، نخستین دوشنبه پس از epoch


# ───────────────────────── ۱) تصمیمِ هر کندل ─────────────────────────
def momentum(t, c, days=MOM_DAYS):
    """بازدهِ بسته-به-بستهٔ ``days`` روزه در هر کندل: ``c[i] / c[k] − 1`` که کندلِ k دقیقاً ``days`` روز پیش از
    کندلِ i باز شده. اگر چنین کندلی در داده نیست (شکاف یا ابتدای تاریخچه) NaN — یعنی «نامعلوم ⇒ صبر»."""
    t = np.asarray(t, dtype=np.int64)
    c = np.asarray(c, dtype=float)
    n = len(c)
    if not n:
        return np.zeros(0)
    ref = t - int(days) * DAY_MS
    k = np.searchsorted(t, ref)
    kc = np.minimum(k, n - 1)
    has = (k < n) & (t[kc] == ref)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(has & (c[kc] > 0), c / np.where(c[kc] > 0, c[kc], 1.0) - 1.0, np.nan)


def evaluate(kl, first, knn_window=decision.LIVE_BARS, skip=SKIP_SETUPS, min_votes=MIN_VOTES, trend=True):
    """همهٔ اجزای تصمیمِ v3 در هر کندلِ ``i ≥ first``، فقط با کندل‌های ``≤ i``.

    ``base`` همان ``decision.side_series`` است با ستاپ‌های ``skip`` کنار گذاشته (با ``skip=()`` عیناً خودِ
    قاعده). ``votes`` رأی‌های هم‌جهتِ ``base`` از ۵ در کندلِ تصمیم است و ``mom`` بازدهِ ۲۸ روزه. ``side`` جهتِ
    نهایی است: ``base`` اگر شرطِ روند برقرار باشد، وگرنه ۰. پیش‌فرض‌ها همان مشخصاتِ ثبت‌شده‌اند؛ ``min_votes``
    (حداقلِ رأیِ هم‌جهت) و ``trend=False`` فقط برای مقایسه‌های توصیفی‌اند.
    """
    a = decision._arrays(kl)
    o, h, l, c, v, t = a["o"], a["h"], a["l"], a["c"], a["v"], a["t"]
    n = len(c)
    cs = engine.component_series(o, h, l, c, v)
    lb = decision.SETUP_LOOKBACK
    first = max(int(first), lb - 1)
    ssig = np.zeros(n, dtype=np.int8)
    sname = [None] * n
    for j in range(first - (lb - 1), n):
        sg, st = engine.setup_signal(cs, o, h, l, c, j)
        if sg and st not in skip:                   # ستاپِ کنارگذاشته = انگار رخ نداده (مثلِ ستاپِ معلق در analyze)
            ssig[j], sname[j] = sg, st
    base = np.zeros(n, dtype=np.int8)
    setup = [None] * n
    a14 = cs["a14"]
    for i in range(first, n):
        for back in range(lb):
            j = i - back
            if ssig[j]:
                if ssig[j] * (c[i] - c[j]) >= -1.2 * float(a14[i]):
                    base[i], setup[i] = ssig[j], sname[j]
                break
    b4 = ((cs["s_trend"] >= 0.30).astype(int) + (cs["s_mom"] >= 0.25) + (cs["s_vol"] >= 0.25)
          + (cs["s_struct"] >= 0.50))
    s4 = ((cs["s_trend"] <= -0.30).astype(int) + (cs["s_mom"] <= -0.25) + (cs["s_vol"] <= -0.25)
          + (cs["s_struct"] <= -0.50))
    z = cs["z"]
    live = np.arange(n) >= first
    free = live & (base == 0)
    # رأیِ kNN هم برای قاعدهٔ z لازم است و هم برای شمارشِ ۵ رأی روی کندل‌های ستاپ
    need = np.flatnonzero((free & (np.abs(z) >= decision.Z_ENTRY)) | (live & (base != 0)))
    ml = np.zeros(n)
    if len(need):
        ml[need] = decision.knn_series(h, l, c, cs, idx=need, window=knn_window)
    bull5 = b4 + (ml >= decision.AI_VOTE)
    bear5 = s4 + (ml <= -decision.AI_VOTE)
    base[free & (z >= decision.Z_ENTRY) & (bull5 >= 3) & (bear5 <= 1)] = 1
    base[free & (z <= -decision.Z_ENTRY) & (bear5 >= 3) & (bull5 <= 1)] = -1
    votes = np.where(base == 1, bull5, np.where(base == -1, bear5, 0)).astype(int)
    mom = momentum(t, c)
    ok_votes = (votes >= int(min_votes)) if min_votes is not None else np.ones(n, dtype=bool)
    with np.errstate(invalid="ignore"):
        ok_trend = (((base == 1) & (mom > 0)) | ((base == -1) & (mom < 0))) if trend else np.ones(n, dtype=bool)
    side = np.where((base != 0) & ok_votes & ok_trend, base, 0).astype(np.int8)
    return {"t": t, "o": o, "h": h, "l": l, "c": c, "first": first, "base": base, "setup": setup,
            "votes": votes, "bull5": bull5, "bear5": bear5, "mom": mom, "z": z, "a14": a14,
            "ok_votes": ok_votes, "ok_trend": ok_trend, "side": side}


# ───────────────────────── ۲) بازپخش: یک معامله در هر لحظه ─────────────────────────
def simulate(ev, side=None, cost_pct=COST_PCT, max_bars=MAX_BARS):
    """معامله‌های یک سریِ جهت با همان قراردادِ ``decision.replay``: سیگنالِ کندلِ i ⇒ ورود در بازِ i+1، و تا
    براکت بسته نشده سیگنالِ تازه نادیده گرفته می‌شود.

    خروجی: ``{"trades", "open", "fresh"}``. هر معاملهٔ بسته ``matured`` دارد: سقفِ ۴۰ کندلی‌اش کاملاً در
    داده گذشته است (``i + max_bars ≤ آخرین کندل``). کارنامه فقط معامله‌های matured را می‌شمرد؛ وگرنه معاملهٔ
    تازه‌ای که زود حدضرر خورده شمرده می‌شد و هم‌دورهٔ هنوز بازش نه. ``open`` معاملهٔ هنوز باز است و ``fresh``
    سیگنالِ آخرین کندلِ بسته که ورودش بازِ کندلِ بعد است.
    """
    o, h, l, c, t, a14 = ev["o"], ev["h"], ev["l"], ev["c"], ev["t"], ev["a14"]
    side = ev["side"] if side is None else np.asarray(side)
    n = len(c)
    first = int(ev["first"])
    trades, open_tr, fresh = [], None, None
    free_from = first
    for i in np.flatnonzero(side[first:]) + first:
        i = int(i)
        if i < free_from:
            continue
        sg = int(side[i])
        row = {"i": i, "t": int(t[i]), "side": "long" if sg == 1 else "short", "setup": ev["setup"][i],
               "votes": int(ev["votes"][i]), "mom": _finite(ev["mom"][i]), "z": float(ev["z"][i]),
               "atr": float(a14[i])}
        if i + 1 >= n:                              # سیگنالِ آخرین کندلِ بسته: کندلِ ورود هنوز باز نشده
            lv = bracket.levels(float(c[i]), float(a14[i]), sg)
            fresh = dict(row, entry=float(lv["entry"]), sl=float(lv["sl"]), tp=float(lv["tp"]),
                         risk_pct=float(lv["risk_pct"]))
            break
        res = bracket.signal_trade(o, h, l, c, float(a14[i]), i, sg, max_bars)
        row.update(entry=float(res["entry"]), sl=float(res["sl"]), tp=float(res["tp"]),
                   risk_pct=float(res["risk_pct"]), entry_t=int(t[i + 1]))
        if res["timed_out"] and i + max_bars > n - 1:
            r = abs(row["entry"] - row["sl"])
            open_tr = dict(row, bars_held=int(n - 1 - i), last=float(c[-1]),
                           unrealized_r=float(sg * (c[-1] - row["entry"]) / r) if r > 0 else None)
            break                                   # هنوز باز است — نتیجه‌اش معلوم نیست
        row.update(outcome=res["outcome"], gross_r=float(res["gross_r"]),
                   net_r=float(bracket.net_r(res["gross_r"], res["risk_pct"], cost_pct)),
                   exit_idx=int(res["exit_idx"]), exit_t=int(t[res["exit_idx"]]),
                   bars_held=int(res["bars_held"]), matured=bool(i + max_bars <= n - 1))
        trades.append(row)
        free_from = int(res["exit_idx"])
    return {"trades": trades, "open": open_tr, "fresh": fresh}


def _finite(x):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def window(kl, start_ts, warmup=WARMUP_BARS):
    """برشِ بازپخش: از ``warmup`` کندل پیش از نخستین کندلِ ``≥ start_ts`` تا انتها، و اندیسِ همان کندل در برش.

    شروعِ گرم‌شدن ثابت است (نه «هرچه داده هست»)، پس نتیجه به مقدارِ تاریخچهٔ بارشده بستگی ندارد و سرور و
    ابزارِ پژوهش روی همان کندل‌ها عیناً یک عدد می‌دهند.
    """
    a = decision._arrays(kl)
    i0 = int(np.searchsorted(a["t"], int(start_ts)))
    cut = max(0, i0 - int(warmup))
    return {k: v[cut:] for k, v in a.items()}, i0 - cut


def replay(kl, start_ts=T0_MS, cost_pct=COST_PCT, knn_window=decision.LIVE_BARS, **spec):
    """بازپخشِ v3 (یا گونهٔ توصیفیِ ``spec``) از ``start_ts``: ``{"ev", "trades", "open", "fresh"}``.

    شروع درونِ holdoutِ مهروموم رد می‌شود: کندل‌های آن دوره فقط اندیکاتورها را گرم می‌کنند.
    """
    if SEALED_MS[0] <= int(start_ts) < SEALED_MS[1]:
        raise ValueError("شروعِ بازپخش درونِ holdoutِ مهروموم (۲۰۲۶-۰۱..۲۰۲۶-۰۸) است — فقط از T0 به بعد یا پیش از ۲۰۲۶")
    sub, i0 = window(kl, start_ts)
    first = max(i0, decision.LIVE_BARS - 1)
    if first >= len(sub["t"]):
        return {"ev": None, "trades": [], "open": None, "fresh": None}
    ev = evaluate(sub, first, knn_window=knn_window, **spec)
    out = simulate(ev, cost_pct=cost_pct)
    out["ev"] = ev
    return out


# ───────────────────────── ۳) کارنامه، آزمون و ترمز (پیش‌ثبت) ─────────────────────────
BRAKE_MIN_N = 20              # ترمز (۱): دست‌کم این‌قدر معاملهٔ matured و میانگینِ منفی
BRAKE_RECENT_N = 100          # ترمز (۲): آخرین ۱۰۰ معاملهٔ matured …
BRAKE_RECENT_MEAN = -0.10     # … با میانگینِ زیرِ −۰٫۱۰R
BOOT_N = 10000
BOOT_SEED = 20261125
# داوریِ یک‌باره در دو نگاه: (نام، سقفِ انحصاریِ کندلِ سیگنال، نخستین زمانِ مجازِ داوری، آلفا، توقف برای بیهودگی؟)
LOOKS = (
    {"name": "look_1", "cutoff_ms": 1_819_756_800_000, "due_ms": 1_823_558_400_000, "alpha": 0.01, "futility": True},
    {"name": "look_2", "cutoff_ms": 1_851_379_200_000, "due_ms": 1_855_180_800_000, "alpha": 0.04, "futility": False},
)
PREREG = "prereg_signal_v3.json"


def week_of(t_ms):
    """شمارهٔ هفتهٔ تقویمیِ UTC (از دوشنبه ۰۰:۰۰) برای زمان(های) ``t_ms``."""
    return (np.asarray(t_ms, dtype=np.int64) - MONDAY0_MS) // WEEK_MS


def matured(trades):
    """فقط معامله‌هایی که کارنامه می‌شمرد (سقفِ ۴۰ کندلی‌شان کاملاً گذشته)، به ترتیبِ کندلِ سیگنال."""
    return sorted((x for x in trades or [] if x.get("matured")), key=lambda x: x["t"])


def stats(trades):
    """آمارِ R خالصِ یک فهرستِ معامله. خطای معیار دو جور است: ساده (``se``) و خوشه‌ایِ هفتگی (``se_week``)؛
    معامله‌های پنج ارز در همان هفته‌ها جمع می‌شوند، پس بازهٔ ۹۵٪ با خطای خوشه‌ای ساخته می‌شود."""
    rs = np.asarray([x["net_r"] for x in trades or []], dtype=float)
    n = len(rs)
    out = {"n": int(n), "mean": None, "win_pct": None, "total_r": 0.0, "se": None, "se_week": None,
           "ci_lo": None, "ci_hi": None, "weeks": 0, "avg_gross_r": None, "avg_cost_r": None,
           "long_n": 0, "short_n": 0, "long_mean": None, "short_mean": None}
    if not n:
        return out
    m = float(rs.mean())
    wk = week_of([x["t"] for x in trades])
    _, inv = np.unique(wk, return_inverse=True)
    S = np.bincount(inv, weights=rs)
    N = np.bincount(inv).astype(float)
    se = float(rs.std(ddof=1) / math.sqrt(n)) if n >= 2 else None
    se_w = float(math.sqrt(float(np.sum((S - m * N) ** 2))) / n) if len(S) >= 2 else None
    lo = [x["net_r"] for x in trades if x["side"] == "long"]
    sh = [x["net_r"] for x in trades if x["side"] == "short"]
    gross = [x["gross_r"] for x in trades if x.get("gross_r") is not None]
    out.update(mean=round(m, 4), win_pct=round(float((rs > 0).mean() * 100), 1), total_r=round(float(rs.sum()), 2),
               se=round(se, 4) if se is not None else None, se_week=round(se_w, 4) if se_w is not None else None,
               ci_lo=round(m - 1.96 * se_w, 4) if se_w is not None else None,
               ci_hi=round(m + 1.96 * se_w, 4) if se_w is not None else None, weeks=int(len(S)),
               avg_gross_r=round(float(np.mean(gross)), 4) if gross else None,
               avg_cost_r=round(float(np.mean([x["gross_r"] - x["net_r"] for x in trades
                                               if x.get("gross_r") is not None])), 4) if gross else None,
               long_n=len(lo), short_n=len(sh),
               long_mean=round(float(np.mean(lo)), 4) if lo else None,
               short_mean=round(float(np.mean(sh)), 4) if sh else None)
    return out


def _week_sums(trades, w0, W):
    S, N = np.zeros(W), np.zeros(W)
    if trades:
        k = week_of([x["t"] for x in trades]) - w0
        if k.min() < 0 or k.max() >= W:
            raise ValueError("معامله بیرون از هفته‌های آزمون")
        r = np.asarray([x["net_r"] for x in trades], dtype=float)
        S, N = np.bincount(k, weights=r, minlength=W), np.bincount(k, minlength=W).astype(float)
    return S, N


def bootstrap(trades, t_from, t_to, rule=None, n_boot=BOOT_N, seed=BOOT_SEED):
    """آزمونِ پیش‌ثبت‌شده: بوت‌استرپِ بلوکیِ هفته‌های تقویمیِ ``[t_from, t_to)`` (هفتهٔ بی‌معامله هم بلوک است).

    ``p`` = (۱ + #{میانگینِ تکرار ≤ ۰}) / (n_boot + ۱)؛ تکرارِ بی‌معامله ≤ ۰ حساب می‌شود. با ``rule`` (معامله‌های
    قاعدهٔ فعلی در همان دوره) تفاوتِ دو میانگین و p آن هم — فقط گزارشی — برمی‌گردد.
    """
    w0 = int(week_of([t_from])[0])
    W = int(week_of([int(t_to) - 1])[0]) - w0 + 1
    out = {"weeks": max(W, 0), "n_boot": int(n_boot), "seed": int(seed), "n": len(trades or []),
           "mean": None, "p": None, "ci_lo": None, "ci_hi": None}
    if W <= 0:
        return out
    S, N = _week_sums(trades, w0, W)
    draw = np.random.default_rng(seed).integers(0, W, size=(int(n_boot), W))
    bs, bn = S[draw].sum(axis=1), N[draw].sum(axis=1)
    ok = bn > 0
    with np.errstate(invalid="ignore", divide="ignore"):
        m = np.where(ok, bs / np.where(ok, bn, 1.0), -np.inf)     # بی‌معامله ⇒ به نفعِ فرضِ صفر
    if N.sum() > 0:
        out.update(mean=round(float(S.sum() / N.sum()), 4),
                   p=round(float((1 + np.count_nonzero(m <= 0)) / (int(n_boot) + 1)), 5))
        if ok.any():
            out.update(ci_lo=round(float(np.percentile(m[ok], 2.5)), 4),
                       ci_hi=round(float(np.percentile(m[ok], 97.5)), 4))
    if rule is not None:
        Sr, Nr = _week_sums(rule, w0, W)
        rs, rn = Sr[draw].sum(axis=1), Nr[draw].sum(axis=1)
        both = ok & (rn > 0)
        d = bs[both] / bn[both] - rs[both] / rn[both]
        mr = float(Sr.sum() / Nr.sum()) if Nr.sum() > 0 else None
        out["rule"] = {"n": len(rule), "mean": round(mr, 4) if mr is not None else None,
                       "diff": round(out["mean"] - mr, 4) if (out["mean"] is not None and mr is not None) else None,
                       "p_diff": round(float((1 + np.count_nonzero(d <= 0)) / (1 + int(both.sum()))), 5)
                       if both.any() else None, "valid": int(both.sum())}
    return out


def brake(trades):
    """ترمزِ یک تایم‌فریم از معامله‌های matured آن (پنج ارز با هم، از T0): ``{"muted", "code", ...}``.

    خاموش وقتی (۱) دست‌کم ۲۰ معامله و میانگینِ منفی، یا (۲) دست‌کم ۱۰۰ معامله و میانگینِ ۱۰۰ تای آخر زیرِ
    −۰٫۱۰R. بی‌حافظه است: هر بار از نو از روی کارنامه حساب می‌شود و با بهترشدنِ آن خودش برمی‌گردد.
    """
    tr = matured(trades)
    rs = np.asarray([x["net_r"] for x in tr], dtype=float)
    n = len(rs)
    mean = float(rs.mean()) if n else None
    recent = float(rs[-BRAKE_RECENT_N:].mean()) if n >= BRAKE_RECENT_N else None
    code = None
    if n >= BRAKE_MIN_N and mean < 0:
        code = "total_negative"
    elif recent is not None and recent < BRAKE_RECENT_MEAN:
        code = "recent_drop"
    return {"muted": code is not None, "code": code, "n": int(n),
            "mean": round(mean, 4) if mean is not None else None,
            "recent_mean": round(recent, 4) if recent is not None else None,
            "min_n": BRAKE_MIN_N, "recent_n": BRAKE_RECENT_N, "recent_floor": BRAKE_RECENT_MEAN,
            "armed": n >= BRAKE_MIN_N}


def judge(look, trades):
    """حکمِ یک نگاه از معامله‌های matured تجمیعیِ 4h و 1d با کندلِ سیگنال در ``[T0, cutoff)``."""
    tr = [x for x in matured(trades) if T0_MS <= x["t"] < look["cutoff_ms"]]
    bt = bootstrap(tr, T0_MS, look["cutoff_ms"])
    mean, p = bt["mean"], bt["p"]
    if mean is not None and mean > 0 and p is not None and p <= look["alpha"]:
        result = "pass"
    elif look["futility"] and mean is not None and mean > 0:
        result = "continue"
    else:
        result = "fail"
    return {"look": look["name"], "result": result, "alpha": look["alpha"], "cutoff_ms": look["cutoff_ms"],
            "n": bt["n"], "mean": mean, "p": p, "ci_lo": bt["ci_lo"], "ci_hi": bt["ci_hi"],
            "weeks": bt["weeks"], "n_boot": bt["n_boot"], "seed": bt["seed"]}


def check_prereg(path=None):
    """ثابت‌های این فایل باید عیناً همان پیش‌ثبت باشند (وگرنه خطا) — پیش از هر اجرای آزمون صدا زده می‌شود."""
    with open(path or paths.data("research", PREREG), "r", encoding="utf-8") as f:
        pr = json.load(f)
    sp, jd, bk = pr["spec"], pr["judgement"], pr["brake"]
    checks = [
        (tuple(sp["timeframes"]) == SIGNAL_TFS, "timeframes"),
        (tuple(sp["coins"]) == tuple(decision.SYMBOLS), "coins"),
        ("fd or sq" in sp["base"] and SKIP_SETUPS == frozenset({"fd", "sq"}), "skipped setups"),
        ("exactly 28 days" in sp["trend_condition"] and MOM_DAYS == 28, "trend condition"),
        (sp["votes_condition"] == "None." and MIN_VOTES is None, "votes condition"),
        ("0.14%" in sp["cost"] and abs(COST_PCT - 0.14) < 1e-12, "cost"),
        ("1788220800000" in pr["data"]["t0"] and T0_MS == 1_788_220_800_000, "t0"),
        ("2000 bars" in pr["data"]["window_of_bars"] and WARMUP_BARS == 2000, "warm-up"),
        ("i + 40 <=" in pr["forward_record"] and MAX_BARS == 40, "maturity"),
        ("seed 20261125" in pr["test"] and "10000 times" in pr["test"] and (BOOT_N, BOOT_SEED) == (10000, 20261125),
         "bootstrap"),
        ("2027-09-01" in jd["look_1"] and "2027-10-15" in jd["look_1"] and "p <= 0.01" in jd["look_1"]
         and LOOKS[0]["alpha"] == 0.01 and LOOKS[0]["futility"], "look 1"),
        ("2028-09-01" in jd["look_2"] and "2028-10-15" in jd["look_2"] and "p <= 0.04" in jd["look_2"]
         and LOOKS[1]["alpha"] == 0.04, "look 2"),
        ("at least 20 matured trades" in bk["rule"] and "latest 100 is below -0.10" in bk["rule"]
         and (BRAKE_MIN_N, BRAKE_RECENT_N, BRAKE_RECENT_MEAN) == (20, 100, -0.10), "brake"),
    ]
    bad = [name for ok, name in checks if not ok]
    if bad:
        raise ValueError(f"ثابت‌های signal_v3 با پیش‌ثبت یکی نیست: {bad}")
    return pr


# ───────────────────────── ۴) برنامهٔ معامله ─────────────────────────
ENTRY_TOL_R = 0.25            # راهنمای اجرا (بیرون از قراردادِ آزمون): ورود فقط تا این فاصله از قیمتِ برنامه
SIDE_FA = {"long": "خرید (لانگ)", "short": "فروش (شورت)"}
STATE_FA = {"signal": "سیگنالِ تازه", "in_trade": "معاملهٔ باز", "filtered": "خلافِ روندِ ۲۸ روزه — سیگنال نیست",
            "wait": "صبر"}


def bar_ms(tf):
    return decision.TF_MINUTES[tf] * 60_000


def plan(tf, sig):
    """برنامهٔ کاملِ یک سیگنالِ تازه یا معاملهٔ باز: ورود، حدضرر، هدف، اعتبار و سقفِ زمانی.

    سیگنالِ تازه: ورود = بازِ کندلِ بعد (≈ بستهٔ کندلِ سیگنال) و اعتبار تا بسته‌شدنِ همان کندلِ ورود؛ پس از آن
    کندل، موتور دوباره تصمیم می‌گیرد. محدودهٔ ورود (±۰٫۲۵R) فقط راهنمای اجراست؛ آزمون ورود را دقیقاً روی بازِ
    کندلِ بعد می‌سنجد.
    """
    step = bar_ms(tf)
    entry, sl, tp = float(sig["entry"]), float(sig["sl"]), float(sig["tp"])
    r = abs(entry - sl)
    sd = 1 if sig["side"] == "long" else -1
    entry_t = int(sig.get("entry_t") or (int(sig["t"]) + step))
    out = {"side": sig["side"], "side_fa": SIDE_FA[sig["side"]], "entry": engine.sig_round(entry),
           "sl": engine.sig_round(sl), "tp": engine.sig_round(tp), "r": engine.sig_round(r),
           "risk_pct": round(r / entry * 100, 3) if entry > 0 else None,
           "gain_pct": round(abs(tp - entry) / entry * 100, 3) if entry > 0 else None,
           "rr": bracket.TP_R, "signal_t": int(sig["t"]), "entry_t": entry_t,
           "time_stop_t": entry_t + MAX_BARS * step,            # بسته‌شدنِ چهلمین کندلِ معامله
           "max_bars": MAX_BARS, "cost_r": round(COST_PCT / max(r / entry * 100, 0.05), 3) if entry > 0 else None,
           "setup": sig.get("setup"), "setup_fa": engine.SETUP_FA.get(sig.get("setup")) if sig.get("setup") else None,
           "votes": sig.get("votes"), "z": round(float(sig["z"]), 2) if sig.get("z") is not None else None,
           "mom_pct": round(sig["mom"] * 100, 2) if sig.get("mom") is not None else None}
    if "bars_held" in sig:                                      # معاملهٔ باز
        out.update(bars_held=int(sig["bars_held"]), bars_left=max(MAX_BARS - int(sig["bars_held"]), 0),
                   last=engine.sig_round(float(sig["last"])) if sig.get("last") is not None else None,
                   unrealized_r=round(float(sig["unrealized_r"]), 3) if sig.get("unrealized_r") is not None else None)
    else:                                                       # سیگنالِ تازه
        out.update(valid_until=entry_t + step,
                   entry_lo=engine.sig_round(entry - ENTRY_TOL_R * r), entry_hi=engine.sig_round(entry + ENTRY_TOL_R * r),
                   chase_limit=engine.sig_round(entry + sd * ENTRY_TOL_R * r))
    return out


def last_bar(ev):
    """ارزیابیِ آخرین کندلِ بسته: جهتِ قاعده (بی fd/sq)، بازدهِ ۲۸ روزه و نتیجهٔ شرطِ روند."""
    if ev is None or not len(ev["t"]):
        return None
    i = len(ev["t"]) - 1
    base, side = int(ev["base"][i]), int(ev["side"][i])
    mom = _finite(ev["mom"][i])
    return {"t": int(ev["t"][i]), "close": float(ev["c"][i]),
            "base": {1: "long", -1: "short"}.get(base), "side": {1: "long", -1: "short"}.get(side),
            "setup": ev["setup"][i], "votes": int(ev["votes"][i]), "z": round(float(ev["z"][i]), 2),
            "mom_pct": round(mom * 100, 2) if mom is not None else None,
            "trend_ok": bool(ev["ok_trend"][i]) if base else None}


# ───────────────────────── ۵) کارنامهٔ رو-به-جلوی کامل ─────────────────────────
VERSION = 1
VENUE = decision.LIVE_VENUE               # binance-spot: همان کندل‌هایی که تصمیمِ زنده از آن‌ها ساخته می‌شود
RECENT_TRADES = 40                        # تعدادِ آخرین معامله‌هایی که برای نمایش نگه داشته می‌شود
_TRADE_KEYS = ("t", "side", "setup", "entry", "sl", "tp", "risk_pct", "outcome", "gross_r", "net_r",
               "exit_t", "bars_held", "mom", "votes")


def bars_needed(tf, now_ms):
    """کندل‌های لازم برای بازپخش از T0 با گرم‌شدنِ کامل، گردشده رو به بالا تا کشِ کندلِ بلند فقط دنباله بگیرد."""
    need = WARMUP_BARS + max(0, (int(now_ms) - T0_MS) // bar_ms(tf)) + 60
    return int(math.ceil(need / 500.0) * 500)


def _sha(kl, i_from):
    import hashlib
    hsh = hashlib.sha256()
    for k in ("t", "o", "h", "l", "c", "v"):
        hsh.update(np.ascontiguousarray(np.asarray(kl[k])[i_from:], dtype=np.int64 if k == "t" else np.float64).tobytes())
    return hsh.hexdigest()


def _compact(tr, sym, tf):
    row = {k: tr.get(k) for k in _TRADE_KEYS}
    row.update(sym=sym, tf=tf)
    for k in ("gross_r", "net_r", "risk_pct", "mom"):
        if row.get(k) is not None:
            row[k] = round(float(row[k]), 4)
    for k in ("entry", "sl", "tp"):
        if row.get(k) is not None:
            row[k] = engine.sig_round(float(row[k]))
    return row


def _maker(trades):
    return [dict(x, net_r=bracket.net_r(x["gross_r"], x["risk_pct"], MAKER_COST_PCT)) for x in trades]


def build_cell(kl, sym, tf, with_rule=True):
    """یک ارز × تایم‌فریم از T0: معامله‌های v3 (و قاعدهٔ فعلی روی همان برش)، معاملهٔ باز، سیگنالِ تازه."""
    a = decision._arrays(kl)
    sub, i0 = window(a, T0_MS)
    n = len(sub["t"])
    if n < decision.MIN_BARS or i0 >= n:
        raise ValueError(f"دادهٔ کافی برای بازپخش از T0 نیست ({n} کندل)")
    if i0 < min(WARMUP_BARS, decision.LIVE_BARS):
        raise ValueError(f"گرم‌شدنِ پیش از T0 ناقص است ({i0} کندل)")
    step = bar_ms(tf)
    gaps = int(np.count_nonzero(np.diff(sub["t"]) != step))
    rep = replay(a, T0_MS)
    ev = rep["ev"]
    out = {"sym": sym, "tf": tf, "bars": {"n": int(n), "from_t": int(sub["t"][0]), "last_t": int(sub["t"][-1]),
                                         "warmup": int(i0), "gaps": gaps, "sha256": _sha(sub, 0)},
           "trades": rep["trades"], "open": rep["open"], "fresh": rep["fresh"], "last": last_bar(ev)}
    if with_rule:
        rr = decision.replay(sub, start_ts=T0_MS)
        out["rule_trades"] = [dict(x, matured=bool(x["i"] + MAX_BARS <= n - 1)) for x in rr["trades"]]
    return out


def assemble(cells, now_ms, judgements=None, errors=None, symbols=None):
    """کارنامهٔ کامل از خانه‌های ``build_cell``: هر خانه، هر تایم‌فریم (با ترمز)، تجمیعی (با آزمون) و فهرستِ سیگنال‌ها."""
    symbols = tuple(symbols or decision.SYMBOLS)
    judgements = list(judgements or [])
    final = next((j for j in judgements if j.get("result") in ("pass", "fail")), None)
    by_tf = {tf: [] for tf in SIGNAL_TFS}
    rule_tf = {tf: [] for tf in SIGNAL_TFS}
    out_cells, signals, recent = {s: {} for s in symbols}, [], []
    horizon = {}
    for c in cells:
        sym, tf = c["sym"], c["tf"]
        m = matured(c["trades"])
        by_tf[tf] += m
        rule_tf[tf] += matured(c.get("rule_trades"))
        horizon[tf] = max(horizon.get(tf, 0), c["bars"]["last_t"] - MAX_BARS * bar_ms(tf))
        recent += [_compact(x, sym, tf) for x in m]
        pend = [x for x in c["trades"] if not x.get("matured")]
        out_cells[sym][tf] = {"stats": stats(m), "pending": len(pend), "bars": c["bars"], "last": c["last"],
                              "open": plan(tf, c["open"]) if c["open"] else None,
                              "fresh": plan(tf, c["fresh"]) if c["fresh"] else None}
    tfs = {}
    for tf in SIGNAL_TFS:
        bk = brake(by_tf[tf])
        if final is not None and final["result"] == "fail":
            bk = dict(bk, muted=True, code="failed_test")
        tfs[tf] = {"stats": stats(by_tf[tf]), "rule": stats(rule_tf[tf]), "maker": stats(_maker(by_tf[tf])),
                   "brake": bk, "matured_until": horizon.get(tf)}
    pooled = sorted((x for tf in SIGNAL_TFS for x in by_tf[tf]), key=lambda x: x["t"])
    pooled_rule = [x for tf in SIGNAL_TFS for x in rule_tf[tf]]
    # دورهٔ توصیفی: از T0 تا آخرین کندلی که سقفِ ۴۰ کندلی‌اش در یکی از دو تایم‌فریم گذشته
    t_to = max([T0_MS + 1] + [h + 1 for h in horizon.values() if h])
    for sym in symbols:
        for tf in SIGNAL_TFS:
            cell = out_cells.get(sym, {}).get(tf)
            if not cell:
                continue
            muted = tfs[tf]["brake"]["muted"]
            last = cell["last"] or {}
            if cell["open"]:
                state = "in_trade"
            elif cell["fresh"]:
                state = "signal"
            elif last.get("base") and last.get("trend_ok") is False:
                state = "filtered"
            else:
                state = "wait"
            cell.update(state=state, state_fa=STATE_FA[state], muted=bool(muted))
            if state in ("signal", "in_trade"):
                p = cell["fresh"] if state == "signal" else cell["open"]
                signals.append(dict(p, sym=sym, tf=tf, state=state, state_fa=STATE_FA[state], muted=bool(muted),
                                    expired=bool(state == "signal" and int(now_ms) >= p["valid_until"])))
    signals.sort(key=lambda s: (s["muted"], s["state"] != "signal", -s["signal_t"]))
    recent.sort(key=lambda x: -x["t"])
    status = "passed" if (final and final["result"] == "pass") else "failed" if final else "testing"
    return {"version": VERSION, "built_at": round(int(now_ms) / 1000, 3), "t0_ms": T0_MS, "venue": VENUE,
            "cost_pct": COST_PCT, "prereg": PREREG, "status": status, "signal_tfs": list(SIGNAL_TFS),
            "status_only_tfs": [tf for tf in decision.TFS if tf not in SIGNAL_TFS],
            "cells": out_cells, "tfs": tfs,
            "pooled": {"stats": stats(pooled), "rule": stats(pooled_rule), "maker": stats(_maker(pooled)),
                       "test": bootstrap(pooled, T0_MS, t_to, rule=pooled_rule), "matured_until": t_to - 1},
            "signals": signals, "recent_trades": recent[:RECENT_TRADES], "judgements": judgements,
            "looks": [dict(l) for l in LOOKS], "errors": dict(errors or {})}


def build(loader, now_ms, judgements=None, symbols=None, with_rule=True, keep_cells=False):
    """بازپخشِ همهٔ خانه‌ها با ``loader(sym, tf) -> کندل‌های بسته‌شده`` و سرهم‌کردنِ کارنامه.

    خانهٔ خراب (دادهٔ ناقص، خطای شبکه) در ``errors`` می‌آید و بقیه را نمی‌اندازد؛ ولی آزمون و ترمز فقط وقتی
    معتبرند که ``errors`` خالی باشد — ``complete`` همین را می‌گوید.
    """
    symbols = tuple(symbols or decision.SYMBOLS)
    cells, errors = [], {}
    for tf in SIGNAL_TFS:
        for sym in symbols:
            try:
                cells.append(build_cell(loader(sym, tf), sym, tf, with_rule=with_rule))
            except Exception as e:  # noqa: BLE001 — در خروجی گزارش می‌شود
                errors[f"{sym}|{tf}"] = str(e) or type(e).__name__
    out = assemble(cells, now_ms, judgements=judgements, errors=errors, symbols=symbols)
    out["complete"] = not errors and len(cells) == len(SIGNAL_TFS) * len(symbols)
    if keep_cells:
        out["_cells"] = cells
    return out


# ───────────────────────── ۶) مسیرِ زنده: کش، حکمِ یک‌باره، خروجیِ API ─────────────────────────
SNAPSHOT_PATH = paths.data("signal_v3_forward.json")       # آخرین کارنامه (زمانِ اجرا؛ در گیت نیست)
JUDGEMENT_PATH = paths.data("signal_v3_judgement.jsonl")   # حکم‌های یک‌باره (فقط افزودنی)
STALE_AFTER_SEC = 3 * 86400       # کارنامهٔ کهنه‌تر از این دیگر مبنای «بخر/بفروش» نیست
RETRY_SEC = 30.0                  # پس از ساختی که هنوز کندلِ تازه را نداشت (صرافیِ دیررس)، تا این‌قدر دوباره نمی‌سازد
GRACE_SEC = 180.0                 # درست پس از بسته‌شدنِ یک کندل، کارنامهٔ کندلِ قبل تا این‌قدر معتبر می‌ماند (در حالِ ساختِ تازه)
_lock = threading.Lock()
_build_lock = threading.Lock()
_cache = {"snap": None, "tried": 0.0}


def market_loader(sym, tf, now_ms=None):
    """کندل‌های بسته‌شدهٔ اسپاتِ بایننس برای بازپخش از T0.

    اول ``market.get_klines_long`` (کشِ دنباله‌دار روی میزبانِ دادهٔ عمومی). اگر آن میزبان در دسترس نبود، همان
    کندل‌ها از میزبان‌های دیگرِ **همان** بایننسِ اسپات صفحه‌به‌صفحه گرفته می‌شود — هرگز از ونیوی دیگر (حجمِ
    OKX/کوکوین فرق دارد و سیگنال را عوض می‌کند).
    """
    import market
    now_ms = int(time.time() * 1000) if now_ms is None else int(now_ms)
    bars = bars_needed(tf, now_ms)
    try:
        return market.get_klines_long(sym, tf, bars)
    except Exception as e:  # noqa: BLE001 — میزبانِ بعدیِ همان صرافی
        log.warn("signal_v3 long klines: primary host failed, trying the other Binance hosts",
                 symbol=sym, tf=tf, error=str(e) or type(e).__name__)
        return _binance_long(market, sym, tf, bars, now_ms)


def _binance_long(market, sym, tf, bars, now_ms):
    """دست‌کم ``bars`` کندلِ بسته‌شدهٔ اسپات با صفحه‌بندیِ رو به عقب روی همهٔ میزبان‌های بایننس (``_binance_json``)."""
    rows, end = {}, None
    for _ in range(bars // market.LONG_PAGE + 2):
        params = {"symbol": sym, "interval": market.TF_BINANCE[tf], "limit": market.LONG_PAGE}
        if end is not None:
            params["endTime"] = end
        chunk = [market._binance_row(r) for r in market._binance_json("/api/v3/klines", params)]
        if not chunk:
            break
        for r in chunk:
            rows[r[0]] = r
        end = chunk[0][0] - 1
        if len(chunk) < market.LONG_PAGE or len(rows) > bars:
            break
    step = bar_ms(tf)
    closed = [rows[t] for t in sorted(rows) if t + step <= now_ms]
    if not closed:
        raise RuntimeError(f"کندلِ بلندِ {sym} {tf} از هیچ میزبانِ بایننس دریافت نشد")
    return market._long_arrays(closed, "binance-spot")


def read_judgements(path=None):
    rows = []
    try:
        with open(path or JUDGEMENT_PATH, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    r = json.loads(line)
                except ValueError:
                    continue
                if isinstance(r, dict) and r.get("look"):
                    rows.append(r)
    except OSError:
        pass
    seen, out = set(), []
    for r in rows:                                    # هر نگاه فقط یک‌بار: نخستین ردیف معتبر است
        if r["look"] not in seen:
            seen.add(r["look"])
            out.append(r)
    return out


def due_look(judgements, now_ms):
    """نگاهی که موعدش رسیده و هنوز حکم ندارد؛ پس از pass/fail دیگر نگاهی نیست."""
    done = {j["look"]: j for j in judgements}
    for look in LOOKS:
        j = done.get(look["name"])
        if j is None:
            return look if int(now_ms) >= look["due_ms"] else None
        if j.get("result") != "continue":
            return None
    return None


def record_judgement(cells, now_ms, path=None):
    """اگر موعدِ نگاهی رسیده، حکمش را یک‌بار و برای همیشه می‌نویسد. خروجی: ردیفِ تازه یا ``None``."""
    path = path or JUDGEMENT_PATH
    look = due_look(read_judgements(path), now_ms)
    if look is None:
        return None
    pooled = [x for c in cells for x in c["trades"]]
    row = dict(judge(look, pooled), at=round(int(now_ms) / 1000, 3), prereg=PREREG, venue=VENUE)
    with _lock:                                       # چک + افزودن اتمی: حکمِ دوم ممکن نیست
        if any(j["look"] == look["name"] for j in read_judgements(path)):
            return None
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return row


def expected_last(tf, now_ms):
    """بازِ آخرین کندلِ بسته‌شدهٔ ``tf`` از روی ساعت."""
    step = bar_ms(tf)
    return (int(now_ms) // step) * step - step


def is_current(snap, now_ms, grace_ms=0):
    """کارنامه کامل است و آخرین کندلِ بستهٔ هر خانه را دارد (با ``grace_ms``: کندلی که همین حالا بسته شده
    هنوز الزامی نیست — سرور در حالِ ساختنِ کارنامهٔ تازه است)."""
    if not isinstance(snap, dict) or snap.get("version") != VERSION or not snap.get("complete"):
        return False
    for sym, row in (snap.get("cells") or {}).items():
        for tf, cell in row.items():
            if int(((cell or {}).get("bars") or {}).get("last_t") or 0) < expected_last(tf, int(now_ms) - int(grace_ms)):
                return False
    return True


def _persist(snap):
    try:
        os.makedirs(os.path.dirname(SNAPSHOT_PATH), exist_ok=True)
        tmp = SNAPSHOT_PATH + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(snap, f, ensure_ascii=False)
        os.replace(tmp, SNAPSHOT_PATH)
    except OSError:
        log.exc("signal_v3 persist")


def _load_persisted():
    try:
        with open(SNAPSHOT_PATH, "r", encoding="utf-8") as f:
            snap = json.load(f)
        return snap if isinstance(snap, dict) and snap.get("version") == VERSION else None
    except (OSError, ValueError):
        return None


def snapshot(force=False, loader=None, now=None):
    """کارنامهٔ رو-به-جلوی جاری. فقط وقتی کندلِ تازه‌ای بسته شده (یا ``force``) از نو ساخته می‌شود.

    ساختِ ناموفق (شبکه) آخرین کارنامهٔ موجود را با ``stale`` برمی‌گرداند؛ کارنامه‌ای که آخرین کندل را ندارد
    هرگز «تازه» خوانده نمی‌شود و ``view`` برایش هیچ «بخر/بفروش»ی نمی‌دهد.
    """
    now = time.time() if now is None else now
    now_ms = int(now * 1000)
    with _lock:
        snap, tried = _cache["snap"], _cache["tried"]
    if snap is None:
        snap = _load_persisted()
    if not force and snap is not None and (is_current(snap, now_ms) or abs(now - tried) < RETRY_SEC):
        with _lock:
            _cache["snap"] = snap
        return snap
    with _build_lock:
        with _lock:
            again, tried = _cache["snap"], _cache["tried"]
        if not force and again is not None and (is_current(again, now_ms) or abs(now - tried) < RETRY_SEC):
            return again                              # نخِ دیگری همین حالا ساخت (یا تلاش کرد)
        with _lock:
            _cache["tried"] = now
        ld = loader or (lambda sym, tf: market_loader(sym, tf, now_ms))
        try:
            fresh = build(ld, now_ms, judgements=read_judgements(), keep_cells=True)
            cells = fresh.pop("_cells")
            if fresh["complete"] and record_judgement(cells, now_ms) is not None:
                fresh = assemble(cells, now_ms, judgements=read_judgements())    # همین حالا حکمی ثبت شد
                fresh["complete"] = True
        except Exception as e:  # noqa: BLE001 — کارنامهٔ قبلی می‌ماند
            log.exc("signal_v3 build")
            if snap is None:
                raise
            return dict(snap, stale=True, build_error=str(e) or type(e).__name__)
        if not fresh["complete"] and snap is not None and snap.get("complete"):
            # ساختِ ناقص (یک خانه بی‌داده) جای کارنامهٔ کاملِ قبلی را نمی‌گیرد
            return dict(snap, stale=True, build_error="؛ ".join(f"{k}: {v}" for k, v in fresh["errors"].items()))
        with _lock:
            _cache["snap"] = fresh
        _persist(fresh)
        return fresh


def view(force=False, loader=None, now=None):
    """خروجیِ ``GET /api/signals``: کارنامه + اینکه الان می‌شود به آن تکیه کرد یا نه.

    ``actionable`` فقط وقتی درست است که کارنامه کامل و تازه باشد؛ وگرنه صفحه هیچ سیگنالی را «بخر/بفروش» نشان
    نمی‌دهد (ترمزی که داده ندارد باید بسته بماند، نه باز).
    """
    now = time.time() if now is None else now
    snap = dict(snapshot(force=force, loader=loader, now=now))
    age = now - float(snap.get("built_at") or 0)
    current = is_current(snap, int(now * 1000))
    usable = current or is_current(snap, int(now * 1000), grace_ms=int(GRACE_SEC * 1000))
    snap["age_sec"] = round(age, 1)
    snap["current"] = bool(current)
    snap["refreshing"] = bool(usable and not current)        # کندلی همین حالا بسته شده؛ کارنامهٔ تازه در راه است
    snap["actionable"] = bool(usable and not snap.get("stale") and age < STALE_AFTER_SEC)
    snap["now_ms"] = int(now * 1000)
    return snap
