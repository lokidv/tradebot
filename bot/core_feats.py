# -*- coding: utf-8 -*-
"""ویژگی‌ها و برچسب‌های «هستهٔ v2» — یک مسیرِ numpy برای تاریخچه و زنده (پیش‌ثبت: prereg_core2.json).

پیش‌ثبت الزام‌آور است: نام‌ها، پنجره‌ها و قواعدِ این فایل همان‌اند که در
``bot/data/research/prereg_core2.json`` ثبت شد و پس از هیچ نتیجه‌ای عوض نمی‌شوند.

* ``FEATURES[tf]`` — فهرستِ مرتبِ نام‌ها برای 5m | 15m | 1h | 4h | 1d (گروه‌های G1, G2, G3, G5, G6, G7).
* ``compute(tf, bars, ctx)`` — ``{نام: آرایهٔ float32}`` هم‌ترازِ ``bars["t"]``. ``bars`` کندل‌های **اسپات**
  (t,o,h,l,c,v,qv,n,tbv؛ ``tbv`` می‌تواند NaN باشد ⇒ ویژگی‌های جریانِ سفارش NaN). ``ctx``:
  ``{"btc": کندل‌های اسپاتِ BTC، "eth": کندل‌های ETH، "closes5": {نماد: کندل‌ها} برای پهنای بازار،
  "kfund": {"t", "rate", "interval_h"} نرخِ فاندینگِ تسویه‌شدهٔ کوکوین}``. سری‌های بین‌دارایی فقط با
  **همان زمانِ بسته‌شدن** هم‌تراز می‌شوند؛ نبودِ داده ⇒ NaN (هرگز جایگزینِ بی‌صدا).
* ``labels(tf, fut_bars, cost)`` — برچسبِ براکت روی کندل‌های فیوچرز، دقیقاً ``bracket.signal_trade``.
* ``LIVE_BARS[tf]`` — تعدادِ کندلِ بستهٔ لازم تا سطرِ آخرِ پنجرهٔ زنده با تاریخچهٔ کامل برابر باشد
  (اندازه‌گیری‌شده روی دادهٔ محلی؛ آزمونِ برابریِ دُم در tests/test_core_feats.py).

قواعدِ علّی: هر پنجره پس‌رو است (کندل‌های ≤ t)؛ z-score فقط پنجرهٔ **قبلی** را می‌بیند (shift(1)) و
دستِ‌کم نیمی از آن باید موجود باشد؛ بازنمونه‌گیری فقط سطل‌های کامل را نگه می‌دارد. سطرهای پیش از
``CAUSAL_START`` همه NaN‌اند، چون ``engine.component_series`` نوزده سطرِ اول را با میانگینِ کلِ آرایه
پر می‌کند (یعنی از آینده). این ماژول هیچ‌وقت ``tradeable`` یا gates.json را لمس نمی‌کند.
"""
import hashlib
import math
import os

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view as _swv

import bracket
import engine

try:                                        # ثبتِ واحدِ تایم‌فریم‌ها، اگر موجود است
    import tf_spec as _tfs
    _MINUTES = dict(_tfs.MINUTES)
except Exception:                           # pragma: no cover — بدونِ ثبت، همان جدول
    _MINUTES = {"5m": 5, "15m": 15, "1h": 60, "4h": 240, "1d": 1440}

TFS = ("5m", "15m", "1h", "4h", "1d")
MINUTES = {tf: _MINUTES[tf] for tf in TFS}
BAR_MS = {tf: m * 60_000 for tf, m in MINUTES.items()}
DAY_MS = 86_400_000
HOUR_MS = 3_600_000
EPS = 1e-12

# ───────────────────────── پنجره‌های پیش‌ثبت‌شده (برحسبِ کندل) ─────────────────────────
ZW = {"5m": 2016, "15m": 960, "1h": 720, "4h": 360, "1d": 180}                 # پنجرهٔ z-score
ROCH = {"5m": (6, 24, 96), "15m": (4, 16, 64), "1h": (3, 12, 48), "4h": (3, 12, 42), "1d": (3, 10, 30)}
VWAP_N = {"5m": 288, "15m": 96, "1h": 24, "4h": 42, "1d": 20}
RV_SHORT = 12
RV_LONG = {"5m": 96, "15m": 96, "1h": 96, "4h": 96, "1d": 60}
FLOW_N = (4, 16, 64)
CVD_N = (16, 64)
BREADTH_EMA = 50
BREADTH_N = 5                # پهنای بازار روی پنج ارزِ فهرست (watchlist)
KFUND_Z_WIN = 90             # z روی ۹۰ تسویهٔ قبلی
KFUND_PERSIST_N = 9          # میانگینِ علامتِ ۹ تسویهٔ آخر
KFUND_STALE_INTERVALS = 2    # آخرین تسویه قدیمی‌تر از دو بازه ⇒ دادهٔ فاندینگ ناقص ⇒ NaN
KFUND_LIVE_SETTLEMENTS = 100  # زنده دستِ‌کم این‌قدر تسویه بگیرد (۹۰ قبلی + فعلی + حاشیه)
US_SESS_MIN = (13 * 60 + 30, 20 * 60)   # بازشدنِ کندل در ۱۳:۳۰ تا ۲۰:۰۰ UTC
CAUSAL_START = 320           # سطرهای قبل از این NaN — آلودگیِ پرکردنِ engine با میانگینِ کل (≤ ~۳۰۰ سطر)

SETUP_CODES = {None: 0, "zx": 1, "pb": 2, "sq": 3, "fd": 4, "rg": 5}

# ───────────────────────── نام‌ها — عیناً گروه‌های پیش‌ثبت ─────────────────────────
GROUPS = {
    "G1_engine": ["z", "s_trend", "s_mom", "s_vol", "s_struct", "votes_net", "kdist", "adx", "chop",
                  "rsi", "bbw_rank", "atr_rank", "vol_ts", "ema200_dist", "setup_code"],
    "G2_indicators": ["stochrsi", "cci20", "mfi14", "willr14", "roc_s", "roc_m", "roc_l", "bb_pctb",
                      "kelt_pos", "don20", "don55", "ichi_cloud", "ichi_tk", "vwap_dist", "svwap_dist",
                      "di_spread", "rv_ratio", "vol_z", "obv_slope", "clv", "clv4", "wick_up", "wick_dn"],
    "G3_spot_order_flow": ["sflow_4", "sflow_16", "sflow_64", "sflow_4_z", "sflow_16_z", "sflow_64_z",
                           "stbr1_z", "scvd_div_16", "scvd_div_64", "tsize_z", "ntrades_z"],
    "G5_kucoin_funding": ["kfund_bp", "kfund_z", "kfund_persist"],
    "G6_cross_asset": ["btc_roc_m", "btc_z", "btc_flow16_z", "rs_btc_m", "ethbtc_z", "breadth"],
    "G7_calendar": ["hour_sin", "hour_cos", "dow_sin", "dow_cos", "us_sess", "min_to_fund"],
}
TF_ONLY = {"svwap_dist": ("5m", "15m", "1h")}          # پیش‌ثبت: «svwap_dist (5m/15m/1h only)»
# حذف‌های پیش از اولین برازش — فقط به دلیلِ مجازِ پیش‌ثبت («ثابت» یا «زنده یکسان نیست»)
DROPPED = {
    "1d": {
        "hour_sin": "ثابت در 1d — بسته‌شدنِ کندلِ روزانه همیشه ۰۰:۰۰ UTC است",
        "hour_cos": "ثابت در 1d — بسته‌شدنِ کندلِ روزانه همیشه ۰۰:۰۰ UTC است",
        "us_sess": "ثابت در 1d — کندلِ روزانه همیشه ۰۰:۰۰ UTC باز می‌شود (بیرون از ۱۳:۳۰–۲۰:۰۰)",
        "min_to_fund": "ثابت در 1d — تا فاندینگِ بعدیِ کوکوین از ۰۰:۰۰ همیشه ۴ از ۸ ساعت (۰٫۵)",
    },
}
FEATURES = {tf: [f for g in GROUPS.values() for f in g
                 if tf in TF_ONLY.get(f, TFS) and f not in DROPPED.get(tf, {})] for tf in TFS}
GROUP_OF = {f: g for g, fs in GROUPS.items() for f in fs}

# کندلِ بستهٔ لازم برای برابریِ دقیقِ سطرِ آخر (اندازه‌گیری روی دادهٔ محلی + حاشیه؛ گلوگاه: EMA200
# با بذرِ x[0] در ema200_dist و پنجرهٔ z-score + گرمایشِ ATR). زنده باید دستِ‌کم این‌قدر بگیرد.
LIVE_BARS = {"5m": 3000, "15m": 2000, "1h": 1900, "4h": 1900, "1d": 1500}
WARMUP = LIVE_BARS           # سطرهای پیش از این در تاریخچه به نقطهٔ شروعِ داده وابسته‌اند (برابرِ زنده نیستند)

_G3 = set(GROUPS["G3_spot_order_flow"])


def _tf(tf):
    if tf not in BAR_MS:
        raise KeyError(f"تایم‌فریمِ ناشناخته برای هستهٔ v2: {tf!r} (مجاز: {', '.join(TFS)})")
    return tf


def source_hash():
    """sha256 همین فایل — در گزارشِ آموزش ثبت می‌شود (پیش‌ثبت: «file hash at training time»)."""
    with open(os.path.abspath(__file__), "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


# ───────────────────────── ابزارهای پنجره‌ای (numpy، علّی) ─────────────────────────
def _f(x):
    return np.asarray(x, dtype=np.float64)


def _win(x, n, fn):
    """``fn`` روی هر پنجرهٔ پس‌روِ کامل ``x[i-n+1..i]``؛ سطرهای < n-1 NaN. هر پنجره مستقل از
    تاریخچهٔ پیشین حساب می‌شود (نه با جمعِ تجمعی) تا تاریخچه و پنجرهٔ زنده بیت‌به‌بیت یکی باشند."""
    x = _f(x)
    m = len(x)
    out = np.full(m, np.nan)
    if n < 1 or m < n:
        return out
    W = _swv(x, n)
    step = max(1024, 4_000_000 // n)
    for s in range(0, W.shape[0], step):
        blk = W[s:s + step]
        out[n - 1 + s:n - 1 + s + len(blk)] = fn(blk)
    return out


def rsum(x, n):
    return _win(x, n, lambda W: W.sum(axis=1))


def sma(x, n):
    return _win(x, n, lambda W: W.mean(axis=1))


def rmax(x, n):
    return _win(x, n, lambda W: W.max(axis=1))


def rmin(x, n):
    return _win(x, n, lambda W: W.min(axis=1))


def rstd(x, n, ddof=1):
    return _win(x, n, lambda W: W.std(axis=1, ddof=ddof))


def lag(x, k):
    x = _f(x)
    if k <= 0:
        return x.copy()
    return np.r_[np.full(min(k, len(x)), np.nan), x[:-k]] if len(x) > k else np.full(len(x), np.nan)


def zs(x, w):
    """z-score روی پنجرهٔ **قبلی** ``x[i-w..i-1]`` (shift(1))، انحرافِ معیار با ddof=1؛ دستِ‌کم ``w//2``
    مقدارِ موجود لازم است، وگرنه NaN. پنجرهٔ ثابت (واریانسِ صفر) ⇒ NaN. هم‌معنا با
    ``pandas: (s - s.shift(1).rolling(w, min_periods=w//2).mean()) / ...std()``."""
    x = _f(x)
    n = len(x)
    out = np.full(n, np.nan)
    ok = np.isfinite(x)
    if not ok.any():
        return out
    ref = x[int(np.argmax(ok))]                     # مرکزکردن با اولین مقدارِ موجود (علّی) برای دقت
    xc = np.where(ok, x - ref, 0.0)
    c1 = np.concatenate(([0.0], np.cumsum(xc)))
    c2 = np.concatenate(([0.0], np.cumsum(xc * xc)))
    ck = np.concatenate(([0], np.cumsum(ok)))
    i = np.arange(n)
    lo = np.maximum(i - w, 0)
    k = (ck[i] - ck[lo]).astype(np.float64)
    s1 = c1[i] - c1[lo]
    s2 = c2[i] - c2[lo]
    good = ok & (k >= max(w // 2, 2))
    kk = np.where(good, k, 2.0)
    m = s1 / kk
    var = (s2 - kk * m * m) / (kk - 1.0)
    good &= var > 1e-12 * (s2 / kk)                 # پنجرهٔ ثابت: واریانس فقط نویزِ گردکردن است
    out[good] = (xc[good] - m[good]) / np.sqrt(var[good])
    return out


def _align(ts, vs, t):
    """مقدارِ سری در **همان** زمانِ کندل (هم‌زمانیِ دقیق)؛ نبود ⇒ NaN."""
    ts = np.asarray(ts, dtype=np.int64)
    out = np.full(len(t), np.nan)
    if len(ts) == 0 or len(t) == 0:
        return out
    idx = np.clip(np.searchsorted(ts, t), 0, len(ts) - 1)
    hit = ts[idx] == t
    out[hit] = _f(vs)[idx[hit]]
    return out


def _bars(b):
    """کندل‌ها ⇒ آرایه‌های float64 (t: int64). ستونِ غایب ⇒ NaN (ویژگی‌های وابسته NaN می‌شوند)."""
    if b is None:
        return None
    t = np.asarray(b["t"], dtype=np.int64)
    n = len(t)
    if n < 2:
        raise ValueError("دستِ‌کم دو کندل لازم است")
    if np.any(np.diff(t) <= 0):
        raise ValueError("زمانِ کندل‌ها باید اکیداً صعودی و بی‌تکرار باشد")
    out = {"t": t}
    for k in ("o", "h", "l", "c", "v"):
        a = _f(b[k])
        if not np.all(np.isfinite(a)):
            raise ValueError(f"ستونِ {k} مقدارِ نامعتبر دارد")
        out[k] = a
    for k in ("qv", "n", "tbv"):
        out[k] = _f(b[k]) if (k in b and b[k] is not None) else np.full(n, np.nan)
    return out


def _key(B):
    """اثرِ انگشتِ همهٔ ستون‌ها — کش فقط وقتی برمی‌گردد که ورودی بیت‌به‌بیت همان باشد."""
    h = hashlib.blake2b(digest_size=16)
    for k in ("t", "o", "h", "l", "c", "v", "qv", "n", "tbv"):
        h.update(np.ascontiguousarray(B[k]).tobytes())
    return h.hexdigest()


def _cached(ctx, name, tf, B, fn):
    cache = ctx.setdefault("_cache", {}) if isinstance(ctx, dict) else {}
    key = (name, tf, _key(B))
    if key not in cache:
        cache[key] = fn()
    return cache[key]


def resample(bars, tf_from, tf_to):
    """بازنمونه‌گیری به تایم‌فریمِ بزرگ‌تر — **فقط سطل‌های کامل** (تعدادِ کامل و پیوسته)؛
    o اول، c آخر، h بیشینه، l کمینه، v/qv/n/tbv جمع (NaN در هر جزء ⇒ NaN)."""
    fm, tm = BAR_MS[_tf(tf_from)], BAR_MS[_tf(tf_to)]
    if tm % fm:
        raise ValueError("تایم‌فریمِ مقصد باید مضربِ مبدأ باشد")
    t = np.asarray(bars["t"], dtype=np.int64)
    if tm == fm:
        return {k: (np.asarray(v).copy()) for k, v in bars.items()}
    per = tm // fm
    b = (t // tm) * tm
    st = np.flatnonzero(np.r_[True, b[1:] != b[:-1]])
    en = np.r_[st[1:], len(t)] - 1
    keep = ((en - st + 1) == per) & ((t[en] - t[st]) == (per - 1) * fm)
    out = {"t": b[st], "o": _f(bars["o"])[st], "c": _f(bars["c"])[en],
           "h": np.maximum.reduceat(_f(bars["h"]), st), "l": np.minimum.reduceat(_f(bars["l"]), st)}
    for k in ("v", "qv", "n", "tbv"):
        if k in bars and bars[k] is not None:
            out[k] = np.add.reduceat(_f(bars[k]), st)
    return {k: v[keep] for k, v in out.items()}


# ───────────────────────── G1 + G2 + G3 (یک دارایی) ─────────────────────────
def _setup_codes(cs, o, h, l, c):
    """کدِ ستاپ با خودِ ``engine.setup_signal`` در هر کندل (علّی؛ کندلِ ۰ ندارد چون z[i-1] لازم است)."""
    keys = ("z", "adx", "chop", "a14", "yhat", "s_trend", "s_mom", "s_vol", "s_struct", "bbwp",
            "bb_u", "bb_l", "rsi", "rng_lo", "rng_hi")
    L = {k: cs[k].tolist() for k in keys}               # لیستِ پایتونی: همان مقادیر، ~۵ برابر سریع‌تر
    ol, hl, ll, cl = o.tolist(), h.tolist(), l.tolist(), c.tolist()
    out = np.full(len(c), np.nan)
    code = SETUP_CODES
    sig = engine.setup_signal
    for i in range(1, len(c)):
        out[i] = code[sig(L, ol, hl, ll, cl, i)[1]]
    return out


def _ichimoku(h, l, c, a):
    def mid(n):
        return (rmax(h, n) + rmin(l, n)) / 2
    ten, kij, sb = mid(9), mid(26), mid(52)
    sa = (ten + kij) / 2
    A, B = lag(sa, 26), lag(sb, 26)
    top, bot = np.fmax(A, B), np.fmin(A, B)
    with np.errstate(invalid="ignore"):
        d = np.where(c > top, c - top, np.where(c < bot, c - bot, 0.0)) / a
    d[np.isnan(top)] = np.nan
    return np.clip(d, -5, 5), np.clip((ten - kij) / a, -5, 5)


def _svwap(t, c, qv, v, a):
    """VWAPِ لنگرشده در ۰۰:۰۰ UTC — جمعِ تجمعیِ هر روز جدا (مستقل از تاریخچهٔ قبلی)."""
    day = t // DAY_MS
    st = np.flatnonzero(np.r_[True, day[1:] != day[:-1]])
    en = np.r_[st[1:], len(t)]
    cq = np.empty(len(t))
    cv = np.empty(len(t))
    for s, e in zip(st.tolist(), en.tolist()):
        cq[s:e] = np.cumsum(qv[s:e])
        cv[s:e] = np.cumsum(v[s:e])
    return np.clip((c - cq / np.maximum(cv, EPS)) / a, -8, 8)


def _obv_slope(c, v, n=20):
    """شیبِ رگرسیونِ ۲۰تاییِ OBV تقسیم بر انحرافِ معیارِ ۲۰تایی‌اش — روی OBVِ محلیِ هر پنجره
    (نسبت به ثابتِ جمع‌شونده ناورداست، پس با OBVِ کلِ تاریخچه برابر است ولی به نقطهٔ شروع وابسته نیست)."""
    d = np.sign(np.diff(c, prepend=c[0])) * v
    ix = np.arange(n, dtype=np.float64)
    ix -= ix.mean()
    den = float((ix * ix).sum())

    def fn(W):
        L = np.cumsum(W, axis=1)
        return (L @ ix / den) / np.maximum(L.std(axis=1), EPS)
    return _win(d, n, fn)


def _single(tf, B):
    """G1، G2، G3 و سری‌های کمکی برای یک دارایی (float64)."""
    t, o, h, l, c, v = B["t"], B["o"], B["h"], B["l"], B["c"], B["v"]
    qv, nn, tbv = B["qv"], B["n"], B["tbv"]
    w = ZW[tf]
    rs, rm, rl = ROCH[tf]
    cs = engine.component_series(o, h, l, c, v)
    a = np.maximum(cs["a14"], EPS)
    F = {}
    # ── G1 موتور
    for f in ("z", "s_trend", "s_mom", "s_vol", "s_struct"):
        F[f] = _f(cs[f])
    b4 = ((cs["s_trend"] >= 0.30).astype(int) + (cs["s_mom"] >= 0.25) + (cs["s_vol"] >= 0.25)
          + (cs["s_struct"] >= 0.50))
    s4 = ((cs["s_trend"] <= -0.30).astype(int) + (cs["s_mom"] <= -0.25) + (cs["s_vol"] <= -0.25)
          + (cs["s_struct"] <= -0.50))
    F["votes_net"] = (b4 - s4).astype(np.float64)
    F["kdist"] = np.clip((c - cs["yhat"]) / a, -6, 6)
    F["adx"] = _f(cs["adx"])
    F["chop"] = _f(cs["chop"])
    F["rsi"] = cs["rsi"] - 50
    F["bbw_rank"] = cs["bbwp"] - 0.5
    F["atr_rank"] = cs["atrp"] - 0.5
    F["vol_ts"] = cs["a14"] / np.maximum(cs["a50"], EPS) - 1
    F["ema200_dist"] = np.clip((c - cs["ema200"]) / a, -15, 15)
    F["setup_code"] = _setup_codes(cs, o, h, l, c)
    # ── G2 اندیکاتورهای افزوده
    r14 = cs["rsi"]
    lo, hi = rmin(r14, 14), rmax(r14, 14)
    with np.errstate(invalid="ignore"):
        F["stochrsi"] = sma(np.where(hi - lo > EPS, (r14 - lo) / np.maximum(hi - lo, EPS), 0.5), 3) - 0.5
    tp = (h + l + c) / 3
    m20 = sma(tp, 20)
    md = _win(tp, 20, lambda W: np.abs(W - W.mean(axis=1, keepdims=True)).mean(axis=1))
    F["cci20"] = np.clip((tp - m20) / (0.015 * np.maximum(md, EPS)), -400, 400) / 100
    dtp = np.diff(tp, prepend=tp[0])
    mf = tp * v
    su, sd = rsum(np.where(dtp > 0, mf, 0.0), 14), rsum(np.where(dtp < 0, mf, 0.0), 14)
    F["mfi14"] = 100 * su / np.maximum(su + sd, EPS) - 50
    hh14, ll14 = rmax(h, 14), rmin(l, 14)
    F["willr14"] = (c - ll14) / np.maximum(hh14 - ll14, EPS) - 0.5
    for nm, k in (("roc_s", rs), ("roc_m", rm), ("roc_l", rl)):
        F[nm] = np.clip((c - lag(c, k)) / a / math.sqrt(k), -6, 6)
    bm, bsd = sma(c, 20), rstd(c, 20, ddof=0)
    F["bb_pctb"] = np.clip((c - (bm - 2 * bsd)) / np.maximum(4 * bsd, EPS), -1, 2) - 0.5
    F["kelt_pos"] = np.clip((c - engine.ema(c, 20)) / (2 * a), -3, 3)
    for k in (20, 55):
        hh, ll = rmax(h, k), rmin(l, k)
        F[f"don{k}"] = (c - ll) / np.maximum(hh - ll, EPS) - 0.5
    F["ichi_cloud"], F["ichi_tk"] = _ichimoku(h, l, c, a)
    nv = VWAP_N[tf]
    F["vwap_dist"] = np.clip((c - rsum(qv, nv) / np.maximum(rsum(v, nv), EPS)) / a, -8, 8)
    if "svwap_dist" in FEATURES[tf]:
        F["svwap_dist"] = _svwap(t, c, qv, v, a)
    up_ = np.diff(h, prepend=h[0])
    dn_ = -np.diff(l, prepend=l[0])
    pdi = engine.rma(np.where((up_ > dn_) & (up_ > 0), up_, 0.0), 14)
    mdi = engine.rma(np.where((dn_ > up_) & (dn_ > 0), dn_, 0.0), 14)
    F["di_spread"] = (pdi - mdi) / np.maximum(pdi + mdi, EPS)
    lr = np.diff(np.log(c), prepend=np.log(c[0]))
    F["rv_ratio"] = np.log(np.maximum(rstd(lr, RV_SHORT), EPS) / np.maximum(rstd(lr, RV_LONG[tf]), EPS))
    F["vol_z"] = zs(np.log(np.maximum(v, EPS)), w)
    F["obv_slope"] = _obv_slope(c, v)
    rng = np.maximum(h - l, EPS)
    F["clv"] = ((c - l) - (h - c)) / rng
    F["clv4"] = sma(F["clv"], 4)
    F["wick_up"] = np.clip((h - np.maximum(o, c)) / a, 0, 5)
    F["wick_dn"] = np.clip((np.minimum(o, c) - l) / a, 0, 5)
    # ── G3 جریانِ سفارشِ اسپات (حجمِ خریدِ تیکر در برابرِ کل)
    F["stbr1_z"] = zs(tbv / np.maximum(v, EPS), w)
    for k in FLOW_N:
        sv = rsum(v, k)
        fl = (2 * rsum(tbv, k) - sv) / np.maximum(sv, EPS)
        F[f"sflow_{k}"] = fl
        F[f"sflow_{k}_z"] = zs(fl, w)
    for k in CVD_N:
        F[f"scvd_div_{k}"] = F[f"sflow_{k}_z"] - zs((c - lag(c, k)) / a / math.sqrt(k), w)
    with np.errstate(invalid="ignore", divide="ignore"):
        F["tsize_z"] = zs(np.log(np.maximum(qv / np.maximum(nn, 1), EPS)), w)
        F["ntrades_z"] = zs(np.log(np.maximum(nn, 1)), w)
    miss = ~np.isfinite(tbv)                                  # بی تیکر ⇒ کلِ گروهِ جریان NaN (نه ۰٫۵)
    if miss.any():
        for f in _G3:
            F[f] = np.where(miss, np.nan, F[f])
    return F


def _cross(tf, B):
    """سه سریِ BTC که G6 لازم دارد (روی شبکهٔ زمانیِ خودِ BTC)."""
    c = B["c"]
    cs = engine.component_series(B["o"], B["h"], B["l"], c, B["v"])
    a = np.maximum(cs["a14"], EPS)
    rm = ROCH[tf][1]
    sv = rsum(B["v"], 16)
    fl = (2 * rsum(B["tbv"], 16) - sv) / np.maximum(sv, EPS)
    fz = zs(fl, ZW[tf])
    fz[~np.isfinite(B["tbv"])] = np.nan
    return {"roc_m": np.clip((c - lag(c, rm)) / a / math.sqrt(rm), -6, 6), "z": _f(cs["z"]),
            "flow16_z": fz}


def _ethbtc(tf, Bb, Be):
    """z-scoreِ تغییرِ m-کندلیِ ln(ETH/BTC) روی شبکهٔ BTC (برای همهٔ ارزها یکی)."""
    e = _align(Be["t"], Be["c"], Bb["t"])
    x = np.log(e / Bb["c"])
    return zs(x - lag(x, ROCH[tf][1]), ZW[tf])


def _above_ema(B):
    return (B["c"] > engine.ema(B["c"], BREADTH_EMA)).astype(np.float64)


# ───────────────────────── G5 فاندینگِ کوکوین + فاصله تا فاندینگ ─────────────────────────
def _kfund(tf, t, kf):
    """آخرین نرخِ **تسویه‌شده** با زمانِ تسویه ≤ بسته‌شدنِ کندل، نرمال‌شده به ۸ ساعت."""
    n = len(t)
    nan = np.full(n, np.nan)
    out = {"kfund_bp": nan, "kfund_z": nan.copy(), "kfund_persist": nan.copy(), "min_to_fund": nan.copy()}
    if not kf or kf.get("t") is None or len(kf["t"]) == 0:
        return out
    kt = (np.asarray(kf["t"], dtype=np.int64) // 60_000) * 60_000     # لرزشِ میلی‌ثانیه‌ای ⇒ دقیقه
    kr = _f(kf["rate"])
    ih = kf.get("interval_h", 8.0)
    ki = np.broadcast_to(_f(8.0 if ih is None else ih), kt.shape).astype(np.float64)
    o = np.argsort(kt, kind="stable")
    kt, kr, ki = kt[o], kr[o], ki[o]
    last = np.r_[kt[1:] != kt[:-1], True]                             # تکرارِ زمان ⇒ آخرین ردیف
    kt, kr, ki = kt[last], kr[last], ki[last]
    ki = np.where(np.isfinite(ki) & (ki > 0), ki, np.nan)
    r8 = kr * 8.0 / ki
    z = zs(r8, KFUND_Z_WIN)
    pers = _win(np.sign(r8), KFUND_PERSIST_N, lambda W: W.mean(axis=1))
    close = t + BAR_MS[tf]
    idx = np.searchsorted(kt, close, side="right") - 1
    ok = idx >= 0
    j = np.maximum(idx, 0)
    step = ki[j] * HOUR_MS
    age = (close - kt[j]).astype(np.float64)
    with np.errstate(invalid="ignore"):
        ok &= np.isfinite(step) & (age <= KFUND_STALE_INTERVALS * step)
    out["kfund_bp"] = np.where(ok, r8[j] * 1e4, np.nan)
    out["kfund_z"] = np.where(ok, z[j], np.nan)
    out["kfund_persist"] = np.where(ok, pers[j], np.nan)
    with np.errstate(invalid="ignore"):
        nxt = kt[j] + step * (np.floor(age / step) + 1)               # اولین تسویهٔ بعد از بسته‌شدن
        out["min_to_fund"] = np.where(ok, (nxt - close) / step, np.nan)
    return out


# ───────────────────────── API ─────────────────────────
def compute(tf, bars, ctx=None):
    """ویژگی‌های ``FEATURES[tf]`` برای هر کندلِ ``bars`` ⇒ ``{نام: float32}`` هم‌طول با ``bars["t"]``.

    ``bars`` فقط کندل‌های **بسته** (کندلِ بازِ آخرِ پاسخِ صرافی را فراخوان حذف کند) و زنده دستِ‌کم
    ``LIVE_BARS[tf]`` کندل. ``ctx`` (همه اختیاری؛ نبود ⇒ NaN): ``btc``، ``eth`` (کندل‌های اسپاتِ همان
    تایم‌فریم)، ``closes5`` (``{نماد: کندل‌ها}`` هر پنج ارز، شاملِ خودِ ارز)، ``kfund``
    (``{"t": زمانِ تسویه ms, "rate": نرخِ تسویه‌شده, "interval_h": بازهٔ فاندینگ}``، زنده دستِ‌کم
    ``KFUND_LIVE_SETTLEMENTS`` تسویهٔ آخر).
    اگر ``ctx`` دیکشنری باشد، سری‌های BTC/ETH/پهنا در ``ctx["_cache"]`` نگه داشته می‌شوند تا
    برای پنج ارز دوباره حساب نشوند.
    """
    tf = _tf(tf)
    ctx = ctx if ctx is not None else {}
    B = _bars(bars)
    t = B["t"]
    n = len(t)
    own = _key(B)
    F = _single(tf, B)
    # ── G6 بین‌دارایی
    Bb = _bars(ctx.get("btc"))
    if Bb is not None and _key(Bb) == own:                    # سطرِ خودِ BTC: همان مقادیرِ خودش
        X = {"roc_m": F["roc_m"], "z": F["z"], "flow16_z": F["sflow_16_z"]}
        F["btc_roc_m"], F["btc_z"], F["btc_flow16_z"] = X["roc_m"].copy(), X["z"].copy(), X["flow16_z"].copy()
    elif Bb is not None:
        X = _cached(ctx, "btc", tf, Bb, lambda: _cross(tf, Bb))
        F["btc_roc_m"] = _align(Bb["t"], X["roc_m"], t)
        F["btc_z"] = _align(Bb["t"], X["z"], t)
        F["btc_flow16_z"] = _align(Bb["t"], X["flow16_z"], t)
    else:
        F["btc_roc_m"] = F["btc_z"] = F["btc_flow16_z"] = np.full(n, np.nan)
    F["rs_btc_m"] = F["roc_m"] - F["btc_roc_m"]
    Be = _bars(ctx.get("eth"))
    if Bb is not None and Be is not None:
        eb = _cached(ctx, ("ethbtc", _key(Be)), tf, Bb, lambda: _ethbtc(tf, Bb, Be))
        F["ethbtc_z"] = _align(Bb["t"], eb, t)
    else:
        F["ethbtc_z"] = np.full(n, np.nan)
    c5 = ctx.get("closes5") or {}
    if len(c5) == BREADTH_N:                                   # تعریف: سهمِ **پنج** ارز؛ کمتر ⇒ NaN
        acc = np.zeros(n)
        for sym in sorted(c5):
            Bx = _bars(c5[sym])
            flag = _cached(ctx, "ema50", tf, Bx, lambda: _above_ema(Bx))
            acc += _align(Bx["t"], flag, t)                    # یکی غایب ⇒ NaN
        F["breadth"] = acc / len(c5) - 0.5
    else:
        F["breadth"] = np.full(n, np.nan)
    # ── G5 + فاصله تا فاندینگ
    F.update(_kfund(tf, t, ctx.get("kfund")))
    # ── G7 تقویم
    close = t + BAR_MS[tf]
    hr = (close // HOUR_MS) % 24 + ((close // 60_000) % 60) / 60.0
    F["hour_sin"], F["hour_cos"] = np.sin(2 * np.pi * hr / 24), np.cos(2 * np.pi * hr / 24)
    dow = ((t // DAY_MS) + 4) % 7
    F["dow_sin"], F["dow_cos"] = np.sin(2 * np.pi * dow / 7), np.cos(2 * np.pi * dow / 7)
    mins = (t // 60_000) % 1440
    F["us_sess"] = ((mins >= US_SESS_MIN[0]) & (mins < US_SESS_MIN[1])).astype(np.float64)
    out = {}
    for f in FEATURES[tf]:
        a = np.asarray(F[f], dtype=np.float32)
        a[:CAUSAL_START] = np.nan
        out[f] = a
    return out


def stack(tf, feats):
    """ماتریسِ ``(n, len(FEATURES[tf]))`` float32 به ترتیبِ ثابتِ ``FEATURES[tf]``."""
    return np.column_stack([np.asarray(feats[f], dtype=np.float32) for f in FEATURES[_tf(tf)]])


# ───────────────────────── برچسب‌ها ─────────────────────────
def _side_label(o, h, l, c, a14, sig, max_bars):
    """``bracket.signal_trade`` برداری‌شده برای همهٔ کندل‌ها در جهتِ ``sig`` — همان حساب، همان ترتیب."""
    n = len(c)
    m = n - 1
    i = np.arange(m)
    entry = o[1:]
    r = np.maximum(bracket.R_ATR_MULT * a14[:-1], bracket.R_MIN_PCT * np.abs(entry))
    sl = entry - sig * r
    tp = entry + sig * bracket.TP_R * r
    rr = np.abs(entry - sl)                                   # همان r که resolve_path دوباره می‌سازد
    last = np.minimum(i + max_bars, n - 1)
    ex_p = np.full(m, np.nan)
    ex_i = np.full(m, -1, dtype=np.int64)
    done = ~(np.isfinite(rr) & (rr > bracket.EPS))
    for k in range(max_bars):
        j = i + 1 + k
        act = ~done & (j <= last)
        if not act.any():
            break
        jj = np.minimum(j, n - 1)
        oj, hj, lj = o[jj], h[jj], l[jj]
        if sig == 1:
            gap, hs, ht = oj <= sl, lj <= sl, hj >= tp
        else:
            gap, hs, ht = oj >= sl, hj >= sl, lj <= tp
        g = act & gap
        s_ = act & ~gap & hs
        t_ = act & ~gap & ~hs & ht
        ex_p[g], ex_i[g] = oj[g], jj[g]
        ex_p[s_], ex_i[s_] = sl[s_], jj[s_]
        ex_p[t_], ex_i[t_] = tp[t_], jj[t_]
        done |= g | s_ | t_
    to = ~done
    trunc = i + max_bars > n - 1                               # سقفِ زمانی به انتهای داده خورده ⇒ نامعلوم
    ex_p[to], ex_i[to] = c[last[to]], last[to]
    bad = (to & trunc) | ~(np.isfinite(rr) & (rr > bracket.EPS))
    gross = sig * (ex_p - entry) / rr
    gross[bad] = np.nan
    ex_i[bad] = -1
    risk_pct = r / np.maximum(np.abs(entry), bracket.EPS) * 100.0
    return np.r_[gross, np.nan], np.r_[ex_i, -1], np.r_[risk_pct, np.nan]


def labels(tf, fut_bars, cost=0.14, max_bars=bracket.MAX_BARS):
    """برچسبِ هدفِ پیش‌ثبت برای هر کندلِ i و هر جهت روی کندل‌های **فیوچرز**: ``bracket.signal_trade``
    (ورود در بازِ کندلِ بعد، حدضرر 1R = max(1.3·ATR14, 0.25٪)، هدف 1.8R، سقفِ ۴۰ کندل، لمسِ هر دو =
    باخت، گپ روی قیمتِ باز)، خالص با ``bracket.net_r`` و هزینهٔ ``cost``٪، بریده به [−2, 1.8].

    خروجی ``(yL, yS, exit_idx, risk_pct)``: ``y`` ها float32 (NaN = نامعلوم: کندلِ آخر یا براکتی که تا
    انتهای داده بسته نشده)، ``exit_idx`` int64 = دیرترین کندلِ خروجِ دو جهت (برای پاک‌سازیِ هم‌پوشانی؛
    ‎-1 اگر هیچ‌کدام معلوم نیست)، ``risk_pct`` float64 (فاصلهٔ ۱R به درصد، برای هر دو جهت یکی).
    """
    _tf(tf)
    o, h, l, c = (_f(fut_bars[k]) for k in ("o", "h", "l", "c"))
    a14 = engine.atr(h, l, c, 14)
    gL, eL, rp = _side_label(o, h, l, c, a14, 1, max_bars)
    gS, eS, _ = _side_label(o, h, l, c, a14, -1, max_bars)
    pen = max(float(cost), 0.0) / np.maximum(rp, 0.05)
    yL = np.clip(gL - pen, -2.0, bracket.TP_R).astype(np.float32)
    yS = np.clip(gS - pen, -2.0, bracket.TP_R).astype(np.float32)
    return yL, yS, np.maximum(eL, eS), rp
