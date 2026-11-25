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
import math

import numpy as np

import bracket
import decision
import engine

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
    import json
    import paths
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
