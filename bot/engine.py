# -*- coding: utf-8 -*-
"""موتور تحلیل CTP — پورت پایتونی منطق اندیکاتور CTP Pro:
روند + مومنتوم + حجم + ساختار + kNN لورنتزی → امتیاز ترکیبی و z-score →
درجه سیگنال، احتمال رشد، پیش‌بینی هدف با مخروط اطمینان، پیشنهاد ورود/حدضرر/هدف و بک‌تست سریع."""
import math
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

import gates

TF_MINUTES = {"15m": 15, "1h": 60, "4h": 240, "1d": 1440}
HORIZON = {"15m": 48, "1h": 36, "4h": 30, "1d": 30}
EPS = 1e-10


# ───────────────────────── اندیکاتورهای پایه ─────────────────────────
def ema(x, n):
    a = 2.0 / (n + 1)
    out = np.empty_like(x)
    out[0] = x[0]
    for i in range(1, len(x)):
        out[i] = a * x[i] + (1 - a) * out[i - 1]
    return out


def rma(x, n):
    a = 1.0 / n
    out = np.empty_like(x)
    out[0] = x[0]
    for i in range(1, len(x)):
        out[i] = a * x[i] + (1 - a) * out[i - 1]
    return out


def rsi(c, n=14):
    d = np.diff(c, prepend=c[0])
    up = rma(np.maximum(d, 0), n)
    dn = rma(np.maximum(-d, 0), n)
    return 100 - 100 / (1 + up / np.maximum(dn, EPS))


def macd_hist(c):
    line = ema(c, 12) - ema(c, 26)
    return line - ema(line, 9)


def true_range(h, l, c):
    pc = np.roll(c, 1)
    pc[0] = c[0]
    return np.maximum(h - l, np.maximum(np.abs(h - pc), np.abs(l - pc)))


def atr(h, l, c, n=14):
    return rma(true_range(h, l, c), n)


def adx_series(h, l, c, n=14):
    up = np.diff(h, prepend=h[0])
    dn = -np.diff(l, prepend=l[0])
    plus_dm = np.where((up > dn) & (up > 0), up, 0.0)
    minus_dm = np.where((dn > up) & (dn > 0), dn, 0.0)
    tr = rma(true_range(h, l, c), n)
    pdi = 100 * rma(plus_dm, n) / np.maximum(tr, EPS)
    mdi = 100 * rma(minus_dm, n) / np.maximum(tr, EPS)
    dx = 100 * np.abs(pdi - mdi) / np.maximum(pdi + mdi, EPS)
    return rma(dx, n)


def supertrend_dir(h, l, c, factor=3.0, period=10):
    a = atr(h, l, c, period)
    hl2 = (h + l) / 2
    ub, lb = hl2 + factor * a, hl2 - factor * a
    fu, fl = ub.copy(), lb.copy()
    d = np.ones(len(c))
    for i in range(1, len(c)):
        fu[i] = ub[i] if (ub[i] < fu[i - 1] or c[i - 1] > fu[i - 1]) else fu[i - 1]
        fl[i] = lb[i] if (lb[i] > fl[i - 1] or c[i - 1] < fl[i - 1]) else fl[i - 1]
        if d[i - 1] > 0:
            d[i] = -1 if c[i] < fl[i] else 1
        else:
            d[i] = 1 if c[i] > fu[i] else -1
    return d  # +۱ صعودی، −۱ نزولی


def kernel_rq(c, length=50, hh=8.0, alpha=1.0):
    """رگرسیون کرنلی گویا-درجه‌دو، فرم علّی (فقط داده گذشته) — برداری‌شده."""
    w = np.array([(1 + (k * k) / (2 * alpha * hh * hh)) ** (-alpha) for k in range(length)])
    out = np.empty_like(c)
    n = len(c)
    head = min(length - 1, n)
    for i in range(head):                       # فقط سطرهای ابتدایی با پنجره ناقص
        m = i + 1
        out[i] = np.dot(c[i - m + 1: i + 1][::-1], w[:m]) / w[:m].sum()
    if n >= length:
        win = sliding_window_view(c, length)    # پنجره کامل: ضرب ماتریسی یک‌جا
        out[length - 1:] = win @ w[::-1] / w.sum()
    return out


def roll_std(x, n):
    out = np.full_like(x, np.nan)
    if len(x) >= n:
        out[n - 1:] = sliding_window_view(x, n).std(axis=1)
    return out


def roll_mean(x, n):
    out = np.full_like(x, np.nan)
    cs = np.cumsum(np.insert(x, 0, 0.0))
    out[n - 1:] = (cs[n:] - cs[:-n]) / n
    return out


def linreg_slope(x, n):
    idx = np.arange(n, dtype=float)
    ix = idx - idx.mean()
    den = (ix * ix).sum()
    out = np.full_like(x, np.nan)
    if len(x) >= n:                              # ix جمعش صفر است؛ حذف میانگین لازم نیست
        out[n - 1:] = sliding_window_view(x, n) @ ix / den
    return out


def choppiness(h, l, c, n=14):
    tr = true_range(h, l, c)
    out = np.full_like(c, 50.0)
    if len(c) > n:
        trs = sliding_window_view(tr, n).sum(axis=1)          # پنجره منتهی به i از اندیس n-1
        hh = sliding_window_view(h, n).max(axis=1)
        ll = sliding_window_view(l, n).min(axis=1)
        rng = np.maximum(hh - ll, EPS)
        out[n:] = 100 * np.log10(np.maximum(trs[1:], EPS) / rng[1:]) / math.log10(n)
    return out


def confirmed_pivots(h, l, left=5, right=5):
    """آخرین پیوت تأییدشده تا هر کندل (با تأخیر right کندل — بدون نگاه به آینده)."""
    n = len(h)
    last_ph = np.full(n, np.nan)
    last_pl = np.full(n, np.nan)
    ph, pl = np.nan, np.nan
    for i in range(n):
        j = i - right                      # کاندیدای تازه‌تأییدشده
        if left <= j < n - right:
            win_h = h[j - left: j + right + 1]
            win_l = l[j - left: j + right + 1]
            if h[j] >= win_h.max():
                ph = h[j]
            if l[j] <= win_l.min():
                pl = l[j]
        last_ph[i], last_pl[i] = ph, pl
    return last_ph, last_pl


def trailing_pct_rank(x, win):
    """صدک مقدار فعلی نسبت به پنجره گذشته (۰ تا ۱) — برداری‌شده."""
    out = np.full_like(x, 0.5)
    n = len(x)
    if n > win:
        wins = sliding_window_view(x, win)[:n - win]          # پنجره x[i-win:i] برای i=win..n-1
        out[win:] = (wins < x[win:, None]).mean(axis=1)
    return out


def clamp(x, lo=-1.0, hi=1.0):
    return float(max(lo, min(hi, x)))


def pw_floor(stats, headwind=False):
    """کفِ احتمالِ بردِ نسبی: نرخِ پایهٔ مدل + لبهٔ لازم (۴ واحد؛ خلافِ باد ۷).
    کفِ سختِ ۳۸/۴۰ هم هست چون زیرِ ~۳۶٪ حتی با پاداشِ 1.8R سربه‌سر نمی‌شود."""
    base = stats.get("base") if stats else None
    if base:
        return max(40.0, base + 7.0) if headwind else max(38.0, base + 4.0)
    return 50.0 if headwind else 45.0                 # بدونِ نرخِ پایه: کفِ محافظه‌کارِ قدیمی


def sig_round(v, sig=8):
    """گردکردن به رقمِ معنادار — نه اعشارِ ثابت. round(4.29e-6, 6) حدها را نابود می‌کرد (باگِ SHIB)."""
    if not isinstance(v, float) or v == 0.0 or not math.isfinite(v):
        return v
    return round(v, max(6, sig - 1 - int(math.floor(math.log10(abs(v))))))


def norm_cdf(x):
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


# ───────────────────────── سری‌های امتیاز ترکیبی ─────────────────────────
def component_series(o, h, l, c, v):
    n = len(c)
    a14 = atr(h, l, c, 14)
    ema_f, ema_s = ema(c, 21), ema(c, 55)
    st = supertrend_dir(h, l, c)
    yhat = kernel_rq(c)
    kslope = np.zeros(n)
    kslope[2:] = (yhat[2:] - yhat[:-2]) / np.maximum(0.5 * a14[2:], EPS)
    stack = np.where((c > ema_f) & (ema_f > ema_s), 1.0,
             np.where((c < ema_f) & (ema_f < ema_s), -1.0,
             np.where(c > ema_s, 0.3, np.where(c < ema_s, -0.3, 0.0))))
    s_trend = np.clip(0.40 * stack + 0.30 * st + 0.30 * np.clip(kslope, -1, 1), -1, 1)

    r = rsi(c)
    hist = macd_hist(c)
    s_mom = np.clip(0.6 * np.clip((r - 50) / 25, -1, 1) + 0.4 * np.clip(hist / np.maximum(a14, EPS), -1, 1), -1, 1)

    obv = np.cumsum(np.sign(np.diff(c, prepend=c[0])) * v)
    obv_sl = linreg_slope(obv, 20)
    obv_sd = roll_std(obv, 20)
    obv_n = np.clip(np.nan_to_num(obv_sl / np.maximum(obv_sd, EPS)) * 3, -1, 1)
    rvol = v / np.maximum(np.nan_to_num(roll_mean(v, 20), nan=v.mean()), EPS)
    s_vol = np.clip(0.7 * obv_n + 0.3 * np.clip(rvol - 1, -1, 1) * np.sign(c - o), -1, 1)

    ph, pl = confirmed_pivots(h, l)
    mid_ok = ~(np.isnan(ph) | np.isnan(pl)) & (ph > pl)
    pos = np.where(mid_ok, (c - (ph + pl) / 2) / np.maximum((ph - pl) / 2, EPS), 0.0)
    s_struct = np.where(~np.isnan(ph) & (c > ph), 1.0,
                np.where(~np.isnan(pl) & (c < pl), -1.0, np.clip(0.4 * pos, -0.4, 0.4)))

    score = 0.30 * s_trend + 0.25 * s_mom + 0.20 * s_vol + 0.25 * s_struct
    score_s = ema(score, 3)
    mu, sd = roll_mean(score_s, 200), roll_std(score_s, 200)
    z = np.nan_to_num((score_s - mu) / np.maximum(sd, EPS))
    adx = adx_series(h, l, c)
    chop = choppiness(h, l, c)

    # باندهای بولینگر + صدک فشردگی پهنا (برای ستاپ شکست فشردگی)
    bb_mid = np.nan_to_num(roll_mean(c, 20), nan=c.mean())
    bb_sd = np.nan_to_num(roll_std(c, 20), nan=0.0)
    bb_u, bb_l = bb_mid + 2 * bb_sd, bb_mid - 2 * bb_sd
    bbw = (bb_u - bb_l) / np.maximum(bb_mid, EPS)
    bbwp = trailing_pct_rank(bbw, 120)
    atrp = trailing_pct_rank(a14 / np.maximum(c, EPS), 252)

    # لبه‌های محدوده ۴۰ کندلی (برای ستاپ فید رنج) — برداری‌شده
    rng_lo = np.full_like(c, np.nan)
    rng_hi = np.full_like(c, np.nan)
    if len(c) > 40:
        rng_lo[40:] = sliding_window_view(l, 40)[:len(c) - 40].min(axis=1)
        rng_hi[40:] = sliding_window_view(h, 40)[:len(c) - 40].max(axis=1)
    return dict(a14=a14, a50=atr(h, l, c, 50), s_trend=s_trend, s_mom=s_mom, s_vol=s_vol, s_struct=s_struct,
                score_s=score_s, z=z, adx=adx, chop=chop, rsi=r, yhat=yhat,
                last_ph=ph, last_pl=pl, ema200=ema(c, 200), rvol=rvol,
                bb_u=bb_u, bb_l=bb_l, bbwp=bbwp, atrp=atrp, rng_lo=rng_lo, rng_hi=rng_hi)


def votes_at(cs, i):
    b = int(cs["s_trend"][i] >= 0.30) + int(cs["s_mom"][i] >= 0.25) + int(cs["s_vol"][i] >= 0.25) + int(cs["s_struct"][i] >= 0.50)
    s = int(cs["s_trend"][i] <= -0.30) + int(cs["s_mom"][i] <= -0.25) + int(cs["s_vol"][i] <= -0.25) + int(cs["s_struct"][i] <= -0.50)
    return b, s


def regime_at(cs, c, i):
    """رژیم علّیِ همان کندل؛ فقط پنجرهٔ گذشته را می‌بیند."""
    lo = max(0, i - 251)
    atrp = np.asarray(cs["a14"][lo:i + 1], float) / np.maximum(
        np.asarray(c[lo:i + 1], float), EPS)
    atr_rank = float((atrp < atrp[-1]).mean() * 100) if len(atrp) else 50.0
    adx_i, chop_i = float(cs["adx"][i]), float(cs["chop"][i])
    if atr_rank > 92:
        return "volatile", atr_rank
    if adx_i >= 22 and chop_i < 55:
        return "trend", atr_rank
    if adx_i < 17 or chop_i > 61.8:
        return "range", atr_rank
    return "transition", atr_rank


# ───────────────────────── ستاپ‌های چهارگانه (مشترک آموزش/زنده) ─────────────────────────
SETUP_FA = {"zx": "کراس z", "pb": "پولبک روند", "sq": "شکست فشردگی", "fd": "فید روند",
            "rg": "فید رنج", "dm": "جهت‌یاب", "ap": "سیاست انتخاب عمل"}


def setup_signal(cs, o, h, l, c, i):
    """تشخیص ستاپ در کندل i (فقط داده بسته‌شده). خروجی: (جهت ±1/0، نوع ستاپ)."""
    b, s = votes_at(cs, i)
    z, zp = cs["z"][i], cs["z"][i - 1]
    adx, chop = cs["adx"][i], cs["chop"][i]
    a = max(cs["a14"][i], EPS)
    yh = cs["yhat"][i]
    # ۱) کراس z — عبور امتیاز نرمال‌شده از آستانه
    if z >= 1.5 and zp < 1.5 and b >= 3 and s <= 1 and adx >= 18:
        return 1, "zx"
    if z <= -1.5 and zp > -1.5 and s >= 3 and b <= 1 and adx >= 18:
        return -1, "zx"
    trending = adx >= 20 and chop < 58
    # ۲) پولبک روند — لمس کرنل در روند برقرار و ازسرگیری
    if trending and cs["s_trend"][i] >= 0.3 and z > 0.3 and b >= 2 and s == 0:
        touched = any((l[j] - cs["yhat"][j]) / max(cs["a14"][j], EPS) <= 0.15 for j in range(max(1, i - 2), i + 1))
        if touched and c[i] > o[i] and c[i] > yh and (c[i] - yh) / a <= 1.2:
            return 1, "pb"
    if trending and cs["s_trend"][i] <= -0.3 and z < -0.3 and s >= 2 and b == 0:
        touched = any((cs["yhat"][j] - h[j]) / max(cs["a14"][j], EPS) <= 0.15 for j in range(max(1, i - 2), i + 1))
        if touched and c[i] < o[i] and c[i] < yh and (yh - c[i]) / a <= 1.2:
            return -1, "pb"
    # ۳) شکست فشردگی — باند بولینگر فشرده و بسته‌شدن بیرون باند
    if cs["bbwp"][i - 1] < 0.15:
        if c[i] > cs["bb_u"][i - 1] and b >= 2 and s <= 1:
            return 1, "sq"
        if c[i] < cs["bb_l"][i - 1] and s >= 2 and b <= 1:
            return -1, "sq"
    # ۴) فید روند — جهشِ خلاف روند تا کرنل که رد می‌شود؛ سیگنال ادامه روند
    #    (z اینجا شرط نیست: در روند نزولی ممتد، z نزول را «عادی» می‌بیند و شورت بی‌صدا می‌ماند)
    kdist = (c[i] - yh) / a
    rsi_i = cs["rsi"][i]
    if adx >= 15:
        if cs["s_trend"][i] <= -0.25 and kdist >= 0.2 and 45 <= rsi_i <= 70 and c[i] < o[i] and s >= 1 and b <= 2:
            return -1, "fd"
        if cs["s_trend"][i] >= 0.25 and kdist <= -0.2 and 30 <= rsi_i <= 55 and c[i] > o[i] and b >= 1 and s <= 2:
            return 1, "fd"
    # ۵) فید رنج — بازگشت از لبه محدوده ۴۰کندلی در بازار رِنج (خلأ رژیم رنج را پر می‌کند)
    ranging = adx < 18 or chop > 61.8
    rlo, rhi = cs["rng_lo"][i], cs["rng_hi"][i]
    if ranging and not (math.isnan(rlo) or math.isnan(rhi)) and (rhi - rlo) >= 4 * a:
        if (c[i] - rlo) <= 0.5 * a and rsi_i < 40 and c[i] > o[i] and s <= 2:
            return 1, "rg"
        if (rhi - c[i]) <= 0.5 * a and rsi_i > 60 and c[i] < o[i] and b <= 2:
            return -1, "rg"
    return 0, None


FEATS = ["z_dir", "votes", "trend_w", "atr_pct", "rsi_dir", "kdist_dir", "funding_dir",
         "rs_dir", "htf_align", "btc_align", "setup_pb", "setup_sq", "setup_fd", "setup_rg",
         "hour_sin", "hour_cos", "gold_dir", "dxy_dir",
         "breadth_dir", "dom_dir", "ethbtc_dir", "vol_ts", "dow_sin", "dow_cos"]


def event_features(cs, c, i, sig, setup, ts_ms, funding_z=0.0, rs_rank=0.5, htf_sign=0, btc_align=True,
                   gold_m=0.0, dxy_m=0.0, breadth_m=0.0, dom_m=0.0, ethbtc_m=0.0):
    """بردار ویژگی جهت‌دار برای مدل متا-لیبلینگ — ترتیب دقیقاً مطابق FEATS."""
    b, s = votes_at(cs, i)
    votes = (b if sig == 1 else s) / 4.0
    tw = 0.5 * clamp((cs["adx"][i] - 15) / 25, 0, 1) + 0.5 * clamp((61.8 - cs["chop"][i]) / 23.6, 0, 1)
    kdist = (c[i] - cs["yhat"][i]) / max(cs["a14"][i], EPS)
    hour = (ts_ms // 3600000) % 24
    dow = (ts_ms // 86400000 + 4) % 7                       # مبدأ epoch پنجشنبه است
    vol_ts = clamp(cs["a14"][i] / max(cs["a50"][i], EPS) - 1.0)   # ساختار زمانی نوسان: انبساط/فشردگی
    return [
        clamp(cs["z"][i] * sig / 3.0, -2, 2),
        votes,
        tw,
        float(cs["atrp"][i]),
        clamp((cs["rsi"][i] - 50) / 50, -1, 1) * sig,
        clamp(kdist / 4.0, -1, 1) * sig,
        clamp(funding_z / 4.0, -1, 1) * sig,
        (rs_rank - 0.5) * 2 * sig,
        float(htf_sign) * sig,
        1.0 if btc_align else 0.0,
        1.0 if setup == "pb" else 0.0,
        1.0 if setup == "sq" else 0.0,
        1.0 if setup == "fd" else 0.0,
        1.0 if setup == "rg" else 0.0,
        math.sin(2 * math.pi * hour / 24),
        math.cos(2 * math.pi * hour / 24),
        clamp(gold_m) * sig,      # مومنتوم طلا (PAXG) هم‌جهت با معامله
        clamp(dxy_m) * sig,       # مومنتوم شاخص دلار هم‌جهت با معامله (رابطه معکوس را مدل یاد می‌گیرد)
        clamp(breadth_m) * sig,   # پهنای بازار: چند درصد ارزها بالای EMA50 خودشان‌اند (نرمال‌شده −۱..۱)
        clamp(dom_m) * sig,       # مومنتوم دامیننس BTC — دامیننسِ رو به رشد معمولاً آلت-نزولی است
        clamp(ethbtc_m) * sig,    # مومنتوم ETH/BTC — دماسنجِ اشتهای ریسکِ آلت‌ها
        vol_ts,
        math.sin(2 * math.pi * dow / 7),
        math.cos(2 * math.pi * dow / 7),
    ]


# ───────────────────────── kNN لورنتزی (نقطه آخر) ─────────────────────────
def knn_ml(h, l, c, cs, k=10, stride=2, max_samples=1500):
    n = len(c)
    hlc3 = (h + l + c) / 3
    cci_raw = hlc3 - np.nan_to_num(roll_mean(hlc3, 20), nan=hlc3.mean())
    cci_den = np.maximum(np.nan_to_num(roll_std(hlc3, 20), nan=1.0), EPS)
    f = np.stack([cs["rsi"] / 100.0,
                  np.clip(cs["adx"] / 50.0, 0, 1),
                  (np.clip(cci_raw / (2 * cci_den), -1, 1) + 1) / 2,
                  rsi(c, 9) / 100.0], axis=1)
    q = f[n - 1]
    dists, labels = [], []
    for i in range(60, n - 5, stride):
        lbl = np.sign(c[i + 4] - c[i])
        if lbl == 0:
            continue
        d = float(np.log1p(np.abs(f[i] - q)).sum())   # فاصله لورنتزی
        dists.append(d)
        labels.append(lbl)
    if len(dists) < 30:
        return 0.0
    order = np.argsort(dists[-max_samples:] if len(dists) > max_samples else dists)[:k]
    lab = np.array(labels[-max_samples:] if len(labels) > max_samples else labels)
    return float(np.mean(lab[order]))


# ───────────────────────── پیش‌بینی مسیر آینده ─────────────────────────
def forecast(c, cs, tf, votes_bull, votes_bear, dir_p=None, dir_lift=None):
    n = len(c)
    H = HORIZON[tf]
    logc = np.log(c)
    lr = np.diff(logc)
    sig1 = float(lr[-100:].std()) or 1e-6
    lin = float(np.nanmean([np.nan_to_num(linreg_slope(logc, 50)[-1]), np.nan_to_num(linreg_slope(logc, 150)[-1])]))
    kdr = float((cs["yhat"][-1] - cs["yhat"][-3]) / max(2 * cs["yhat"][-1], EPS))
    wlr = np.clip(lr[-20:], -4 * sig1, 4 * sig1)
    mom = float(wlr.mean())

    adx_l, chop_l = float(cs["adx"][-1]), float(cs["chop"][-1])
    tw = 0.5 * clamp((adx_l - 15) / 25, 0, 1) + 0.5 * clamp((61.8 - chop_l) / 23.6, 0, 1)
    w_l, w_m = 0.2 + 0.15 * tw, 0.15 + 0.2 * tw
    w_k = 1 - w_l - w_m
    dis = abs(lin - mom) + abs(lin - kdr) + abs(kdr - mom)
    agree = 1 / (1 + dis / (sig1 + EPS))
    drift0 = (w_l * lin + w_k * kdr + w_m * mom) * agree

    z_l = float(cs["z"][-1])
    cons_dir = 1 if z_l > 0.5 else -1 if z_l < -0.5 else 0
    cons_votes = votes_bull if cons_dir == 1 else votes_bear if cons_dir == -1 else 0
    cons_str = 0.0 if cons_dir == 0 else clamp(abs(z_l) / 1.8, 0, 1) * clamp(cons_votes / 3, 0, 1)
    score_l = float(cs["score_s"][-1])
    drift_f = drift0 * tw + 0.35 * clamp(score_l) * sig1
    # ادغام با مدل جهت‌یاب معتبر: درِیفت اکتشافی و مدل آماری یک‌صدا می‌شوند
    # (وزن مدل با لیفت برون‌نمونه‌ای‌اش مقیاس می‌شود — مدل قوی‌تر، حرف بیشتر)
    dir_tilt = 0.0
    w_m = 0.0
    if dir_p is not None:
        # وزن مدل = پایه + اعتبار OOS + قاطعیت پیش‌بینی (مدلِ مطمئن‌تر، صدای بلندتر)
        w_m = clamp(0.15 + (dir_lift or 0) / 80.0 + abs(dir_p - 50) / 50 * 0.15, 0.15, 0.50)
        mu_model_perbar = (dir_p - 50.0) / 50.0 * (sig1 * math.sqrt(H)) / H * 1.2
        drift_f = (1 - w_m) * drift_f + w_m * mu_model_perbar
        dir_tilt = (dir_p - 50.0) / 50.0

    phi = 0.90
    b0 = float(logc[-1])
    anchor = float(np.log(cs["ema200"][-1]))
    mr_k = 0.25 * (1 - 0.5 * tw)
    if cons_dir != 0 and math.copysign(1, anchor - b0) == -cons_dir:
        mr_k *= (1 - cons_str)
    r_l = float(cs["rsi"][-1])
    exc = (r_l - 70) / 30 if r_l > 70 else (r_l - 30) / 30 if r_l < 30 else 0.0

    mid = b0
    path = []
    for t in range(1, H + 1):
        damp = drift_f * phi * (1 - phi ** t) / (1 - phi)
        mr = mr_k * (anchor - b0) * (1 - 0.98 ** t)
        bend = -exc * 0.8 * sig1 * (1 - 0.85 ** t) / 0.15
        ctr = mr + bend
        if cons_dir != 0 and ctr * cons_dir < 0:
            cap = 0.40 * (1 - 0.6 * cons_str) * abs(damp)
            ctr = math.copysign(min(abs(ctr), cap), ctr)
        mid = b0 + damp + ctr
        path.append(mid)

    sig_h = sig1 * math.sqrt(H)
    mu = mid - b0
    conf = 0.4 + 0.6 * tw
    p_up = 0.5 + (norm_cdf(mu / max(sig_h, EPS)) - 0.5) * conf
    skew = 0.3 * ((1 - w_m) * clamp(score_l) + w_m * dir_tilt)   # کجی مخروط هم با دید ادغامی
    target = math.exp(mid)
    return {
        "target": target,
        "target_pct": (target / c[-1] - 1) * 100,
        "cone_hi": math.exp(mid + sig_h * (1 + skew)),
        "cone_lo": math.exp(mid - sig_h * (1 - skew)),
        "p_up": p_up * 100,
        "horizon_bars": H,
        "horizon_min": H * TF_MINUTES[tf],
        "trend_w": tw, "conf": conf, "path_log": path, "sig1": sig1,
    }


# ───────────────────────── پیشنهاد معامله ─────────────────────────
def trade_suggestion(c, cs, fc, tf, votes_bull, votes_bear, s_ml, force_side=None):
    entry = float(c[-1])
    a = float(cs["a14"][-1])
    z = float(cs["z"][-1])
    bull5 = votes_bull + (1 if s_ml >= 0.2 else 0)
    bear5 = votes_bear + (1 if s_ml <= -0.2 else 0)
    side = force_side or ("long" if (z >= 1.2 and bull5 >= 3 and bear5 <= 1) else "short" if (z <= -1.2 and bear5 >= 3 and bull5 <= 1) else None)
    reasons = []
    if side is None:
        if abs(z) < 1.2:
            reasons.append(f"z فعلی {z:+.1f} — برای سیگنال سطحی به ±1.2 نیاز است")
        if max(bull5, bear5) < 3:
            reasons.append(f"رأی‌ها {bull5}▲/{bear5}▼ — حداقل ۳ رأی هم‌جهت لازم است")
        if bull5 >= 2 and bear5 >= 2:
            reasons.append("دسته‌ها دوپاره‌اند")
        reasons.append("و هیچ‌یک از ۵ ستاپ رویدادی (کراس z، پولبک، شکست، فید روند، فید رنج) در ۴ کندل اخیر رخ نداده")
        # آمادگی: چند درصد شرایطِ نزدیک‌ترین سیگنال پر شده (برای نمایش پیشرفت در UI)
        long_prog = 0.55 * min(max(z, 0.0) / 1.2, 1.0) + 0.45 * min(bull5 / 3.0, 1.0)
        short_prog = 0.55 * min(max(-z, 0.0) / 1.2, 1.0) + 0.45 * min(bear5 / 3.0, 1.0)
        ready_side = "long" if long_prog >= short_prog else "short"
        return {"side": None, "status": "منتظر ستاپ", "reasons": reasons,
                "readiness": round(max(long_prog, short_prog) * 100),
                "ready_side": ready_side}

    votes = bull5 if side == "long" else bear5
    tw = fc["trend_w"]
    grade = "B"
    if votes >= 4 and abs(z) >= 1.6:
        grade = "A"
    if votes >= 5 and abs(z) >= 2.2 and tw >= 0.5:
        grade = "A+"

    d = 1 if side == "long" else -1
    min_sl = entry * 0.0025
    base = max(1.3 * a, min_sl)
    piv = float(cs["last_pl"][-1]) if side == "long" else float(cs["last_ph"][-1])
    sl = entry - d * base
    if not math.isnan(piv):
        cand = piv - 0.25 * a if side == "long" else piv + 0.25 * a
        if (d == 1 and entry > cand) or (d == -1 and entry < cand):
            sl = max(cand, sl) if d == 1 else min(cand, sl)
    sl = min(sl, entry - d * min_sl) if d == 1 else max(sl, entry - d * min_sl)
    sl = max(sl, entry - d * 3.5 * a) if d == 1 else min(sl, entry - d * 3.5 * a)
    r_dist = abs(entry - sl)

    tp = fc["target"]
    if (d == 1 and tp <= entry) or (d == -1 and tp >= entry):
        tp = entry + d * 1.8 * r_dist
    tp = min(max(tp, entry + d * 0.8 * r_dist), entry + d * 3.5 * r_dist) if d == 1 else max(min(tp, entry + d * 0.8 * r_dist), entry + d * 3.5 * r_dist)
    min_tp = entry * 0.004
    if abs(tp - entry) < min_tp:
        tp = entry + d * min_tp
    viable = abs(tp - entry) <= 8 * a
    rr = abs(tp - entry) / max(r_dist, EPS)
    return {
        "side": side, "grade": grade, "entry": entry, "sl": sl, "tp": tp,
        "rr": rr, "risk_pct": r_dist / entry * 100, "gain_pct": abs(tp - entry) / entry * 100,
        "time_stop_bars": 40, "time_stop_min": 40 * TF_MINUTES[tf],
        "viable": viable,
        "status": "ستاپ آماده" if viable else "نوسان برای هدف اقتصادی کافی نیست",
        "reasons": [],
    }


# ───────────────────────── بک‌تست سریع (سه‌مانعی) ─────────────────────────
def quick_backtest(o, h, l, c, cs, tf, cost_pct=0.15):
    """بک‌تست سریعِ همهٔ ستاپ‌ها با ورود کندل بعد، timeout و هزینه.

    نتیجهٔ هر رویداد حتماً شمرده می‌شود و برخورد هم‌زمان SL/TP به‌صورت
    محافظه‌کارانه باخت است.
    """
    n = len(c)
    wins = losses = timeouts = 0
    outcomes = []
    cooldown = 0
    for i in range(220, n - 42):
        if cooldown > 0:
            cooldown -= 1
            continue
        sig, _setup = setup_signal(cs, o, h, l, c, i)
        if sig == 0:
            continue
        entry = float(o[i + 1])
        r = max(1.3 * float(cs["a14"][i]), 0.0025 * entry)
        sl = entry - sig * r
        tp = entry + sig * 1.8 * r
        exit_price = None
        timed_out = True
        for j in range(i + 1, min(i + 41, n)):
            if (sig == 1 and o[j] <= sl) or (sig == -1 and o[j] >= sl):
                exit_price = float(o[j])              # گپِ زیان‌بار روی قیمت مشاهده‌شده
                timed_out = False
                break
            hit_sl = l[j] <= sl if sig == 1 else h[j] >= sl
            hit_tp = h[j] >= tp if sig == 1 else l[j] <= tp
            if hit_sl:                      # محافظه‌کار: اگر هر دو در یک کندل، ضرر
                exit_price = sl
                timed_out = False
                break
            if hit_tp:
                exit_price = tp
                timed_out = False
                break
        if exit_price is None:
            exit_price = float(c[min(i + 40, n - 1)])
        gross_r = sig * (exit_price - entry) / max(r, EPS)
        risk_pct = r / max(entry, EPS) * 100
        net_r = gross_r - max(float(cost_pct), 0.0) / max(risk_pct, EPS)
        outcomes.append(net_r)
        if timed_out:
            timeouts += 1
        if net_r > 0:
            wins += 1
        else:
            losses += 1
        cooldown = 10
    total = wins + losses
    gross_win = sum(x for x in outcomes if x > 0)
    gross_loss = -sum(x for x in outcomes if x <= 0)
    return {"n": total, "win_rate": (wins / total * 100) if total else None,
            "avg_r": (sum(outcomes) / total) if total else None,
            "timeouts": timeouts,
            "profit_factor": gross_win / gross_loss if gross_loss > 0 else None}


# ───────────────────────── تحلیل کامل یک نماد/تایم‌فریم ─────────────────────────
def analyze(kl, tf, btc_z=None, predict_fn=None, extras=None, dir_fn=None, action_fn=None):
    o = np.array(kl["o"], dtype=float)
    h = np.array(kl["h"], dtype=float)
    l = np.array(kl["l"], dtype=float)
    c = np.array(kl["c"], dtype=float)
    v = np.array(kl["v"], dtype=float)
    cs = component_series(o, h, l, c, v)
    n = len(c)
    vb, vs = votes_at(cs, n - 1)
    s_ml = knn_ml(h, l, c, cs)
    regime_code, atr_pct = regime_at(cs, c, n - 1)

    # ── مدل جهت‌یاب همیشه‌روشن (قبل از پیش‌بینی، تا مسیر پیش‌بینی با آن یک‌صدا شود) ──
    ex = extras or {}
    dir_stats = None
    action_stats = None
    if dir_fn or action_fn:
        feat_kwargs = dict(
            funding_z=ex.get("funding_z", 0.0),
            rs_rank=ex.get("rs_rank", 0.5),
            htf_sign=ex.get("htf_sign", 0),
            gold_m=ex.get("gold", 0.0), dxy_m=ex.get("dxy", 0.0),
            breadth_m=ex.get("breadth", 0.0), dom_m=ex.get("dom", 0.0),
            ethbtc_m=ex.get("ethbtc", 0.0),
        )
        dfeats = event_features(
            cs, c, n - 1, 1, None, kl["t"][-1],
            btc_align=(btc_z is None) or (btc_z > -0.3), **feat_kwargs)
        sfeats = event_features(
            cs, c, n - 1, -1, None, kl["t"][-1],
            btc_align=(btc_z is None) or (btc_z < 0.3), **feat_kwargs)
        if dir_fn:
            dir_stats = dir_fn(tf, dfeats)
        if action_fn:
            base_risk_pct = max(1.3 * float(cs["a14"][-1]), 0.0025 * float(c[-1])) / max(float(c[-1]), EPS) * 100
            action_stats = action_fn(
                tf, dfeats, sfeats, base_risk_pct,
                cost=ex.get("cost"), regime=regime_code)

    fc = forecast(c, cs, tf, vb, vs,
                  dir_p=dir_stats["p_up"] if dir_stats else None,
                  dir_lift=dir_stats.get("oos_lift") if dir_stats else None)

    # ستاپ زنده (همان منطق آموزش) — حداکثر کندل جاری/قبلی؛ سیگنال کهنه اجرا نمی‌شود
    suspended = ex.get("suspended") or {}            # P1: ستاپ‌های معلق‌شده با عملکردِ زندهٔ ضعیف
    live_sig, live_setup = 0, None
    for back in range(0, 2):
        j = n - 1 - back
        if j < 260:
            break
        sg, st_ = setup_signal(cs, o, h, l, c, j)
        if sg != 0:
            if st_ in suspended:
                break                                # ستاپِ معلق — انگار سیگنالی نبود
            # اعتبار: قیمت از کندل ستاپ بیش از ۱٫۲ ATR خلاف جهت نرفته باشد
            if sg * (c[n - 1] - c[j]) >= -1.2 * float(cs["a14"][n - 1]):
                live_sig, live_setup = sg, st_
            break
    force = ("long" if live_sig == 1 else "short") if live_sig != 0 else \
        (action_stats.get("side") if action_stats and action_stats.get("policy_pass") else None)
    action_used = bool(live_sig == 0 and force and action_stats and action_stats.get("policy_pass"))
    tr = trade_suggestion(c, cs, fc, tf, vb, vs, s_ml, force_side=force)
    bt = quick_backtest(o, h, l, c, cs, tf, cost_pct=ex.get("cost", 0.15))

    # ── ویژگی‌سازی زنده + احتمال از متا-مدل ──
    calibrated = False
    stats = None
    if tr.get("side"):
        d = 1 if tr["side"] == "long" else -1
        setup_used = "ap" if action_used else live_setup if (live_sig != 0 and live_sig == d) else "zx"
        policy_applicable = bool(action_used or (live_sig != 0 and live_sig == d and live_setup is not None))
        btc_align = True if btc_z is None else (btc_z * d > -0.3)   # همان آستانه آموزش
        trending_now = float(cs["adx"][-1]) >= 22 and float(cs["chop"][-1]) < 55
        tr["btc_align"] = btc_align
        tr["setup"] = setup_used
        tr["setup_fa"] = SETUP_FA.get(setup_used, setup_used)
        votes_now = vb if d == 1 else vs
        feats = event_features(cs, c, n - 1, d, setup_used, kl["t"][-1],
                               funding_z=ex.get("funding_z", 0.0),
                               rs_rank=ex.get("rs_rank", 0.5),
                               htf_sign=ex.get("htf_sign", 0),
                               btc_align=btc_align,
                               gold_m=ex.get("gold", 0.0), dxy_m=ex.get("dxy", 0.0),
                                breadth_m=ex.get("breadth", 0.0), dom_m=ex.get("dom", 0.0),
                                ethbtc_m=ex.get("ethbtc", 0.0))
        stats = action_stats if action_used else (
            predict_fn(tf, feats, tr["risk_pct"],
                       legacy={"direction": tr["side"], "z": float(cs["z"][-1]),
                               "votes": votes_now, "trending": trending_now,
                               "btc_align": btc_align},
                       cost=ex.get("cost"), regime=regime_code) if predict_fn else None
        )
        # مشاهدهٔ ستاپ ≠ مجوز معامله. side/grade فقط «چه ستاپی دیده شد» را نگه می‌دارند.
        tr["setup_observed"] = bool(policy_applicable and live_setup)
        if stats:
            # «کالیبره ✓» فقط وقتی مرجعِ احتمال واقعاً برای تصمیم فعال است — نه سیاستِ مردود دادگاه
            calibrated = bool(
                not stats.get("diagnostic_only")
                and (
                    (stats.get("source") in ("policy", "action_policy") and stats.get("policy_trusted"))
                    or stats.get("source") == "model"
                )
            )
            tr.update(p_win=stats.get("p_win"), n_hist=stats.get("n") or 0,
                      avg_r_hist=stats.get("avg_r"),
                      ev_pct=stats.get("ev_pct"), reliability=stats.get("reliability"),
                      model_source=stats.get("source"), oos_lift=stats.get("oos_lift"),
                      p_win_low=stats.get("p_win_low"), p_win_high=stats.get("p_win_high"),
                      p_uncertainty=stats.get("p_uncertainty"),
                      ev_lcb_pct=stats.get("ev_lcb_pct"),
                      edge_trusted=stats.get("edge_trusted", False),
                      edge_r=stats.get("edge_r"), edge_lcb_r=stats.get("edge_lcb_r"),
                      edge_uncertainty_r=stats.get("edge_uncertainty_r"),
                      edge_rank_ic=stats.get("edge_rank_ic"),
                      edge_lift_r=stats.get("edge_lift_r"),
                      feature_zmax=stats.get("feature_zmax"),
                      policy_trusted=bool(stats.get("policy_trusted", False)),
                      policy_pass=bool(stats.get("policy_pass", False)),
                      policy_score=stats.get("policy_score"),
                      policy_margin=stats.get("policy_margin"),
                      policy_test=stats.get("policy_test"),
                      market_rank_required=stats.get("market_rank_required", False),
                      select_quantile=stats.get("select_quantile"),
                      regime_veto=stats.get("regime_veto", False),
                      regime_ok=stats.get("regime_ok", False),
                      regime_stats=stats.get("regime_stats"),
                      authority=stats.get("authority"),
                      diagnostic_only=bool(stats.get("diagnostic_only")),
                      policy_court_failed=bool(stats.get("policy_court_failed")))
        else:
            tr.update(p_win=None, n_hist=0, avg_r_hist=None, ev_pct=None, reliability="کالیبره‌نشده")

        # ── مرجع واحد تصمیم (فاز ۰) ──
        # فقط «سیاست نهاییِ دادگاه‌قبول» مرجع است، و ورودِ نهایی هم باید ترکیبِ
        # پیش‌ثبت‌شده در gates.json باشد. دو مرجعِ قبلی حذف شدند:
        #   • setup_model — مدلِ ستاپ با EV-LCB مثبت، بدونِ آزمونِ منجمد
        #   • edge_pocket — انتخابِ پس از مشاهده روی ۱۲ سطلِ کوچک (n≥۱۸)؛ با لبهٔ صفر
        #     احتمالِ ساختِ جیبِ کاذب ۸۳–۹۹٪ بود و از همهٔ وتوها هم معاف می‌شد.
        # هر دو حالا فقط عددِ تشخیصی تولید می‌کنند.
        src = (stats or {}).get("source")
        auth = None
        pockets = ex.get("edge_pockets") or {}
        susp_combos = ex.get("suspended_combos") or {}
        combo = f"{tf}|{setup_used}"
        combo_side = f"{tf}|{setup_used}|{tr['side']}"
        pocket_hit = pockets.get(combo_side)
        if pocket_hit:                              # فقط نمایش — هرگز مجوزِ ورود
            tr["pocket_n"] = int(pocket_hit.get("n") or 0)
            tr["pocket_avg_r"] = float(pocket_hit.get("avg_r") or 0.0)
            tr["pocket_wr"] = pocket_hit.get("win_rate")
            tr["pocket_lcb_r"] = pocket_hit.get("lcb_r")
            tr["pocket_diagnostic_only"] = True

        if stats and src in ("policy", "action_policy") and stats.get("policy_trusted"):
            auth = "policy" if src == "policy" else "action_policy"

        if auth is None:
            tr["viable"] = False
            tr["policy_trusted"] = False
            tr["policy_pass"] = False
            tr["authority"] = None
            pst = (stats or {}).get("policy_test") or tr.get("policy_test") or {}
            if ex.get("toxic_tf"):
                tr["status"] = ("مشاهده — تایم‌فریم روزانه در کارنامهٔ زنده زیان‌ده است؛ "
                                "ورود فقط با سیاست نهاییِ دادگاه‌قبول مجاز است")
            elif stats and src == "policy" and not stats.get("policy_trusted"):
                avg = pst.get("avg_net_r")
                pf = pst.get("profit_factor")
                extra = ""
                if avg is not None:
                    extra = f" (آزمون: {float(avg):+.2f}R"
                    if pf is not None:
                        extra += f"، PF={float(pf):.2f}"
                    extra += ")"
                tr["status"] = ("مشاهده — سیاست نهایی این تایم‌فریم آزمون زمانی را پاس نکرده"
                                + extra + "؛ ستاپ دیده شد ولی مجوز ورود نیست")
            elif stats and stats.get("diagnostic_only"):
                tr["status"] = "مشاهده — آمار سطلی فقط تشخیصی است؛ مرجع تصمیم معتبر نیست"
            elif stats and src == "model":
                tr["status"] = ("مشاهده — مدل ستاپ فقط تشخیصی است؛ مجوز ورود تنها از "
                                "سیاست نهاییِ معتبر روی ترکیبِ پیش‌ثبت‌شده می‌آید")
            elif not policy_applicable:
                tr["status"] = "منتظر یکی از ستاپ‌های آموزش‌دیده؛ جهت یا امتیاز تکنیکال به‌تنهایی مجوز ورود نیست"
            else:
                tr["status"] = "مشاهده — هیچ مرجع تصمیم معتبری برای ورود فعال نیست"
        elif not policy_applicable:
            tr["viable"] = False
            tr["status"] = "منتظر یکی از ستاپ‌های آموزش‌دیده؛ جهت یا امتیاز تکنیکال به‌تنهایی مجوز ورود نیست"
        elif stats is not None and float(stats.get("feature_zmax") or 0) > 6.0 and auth in ("policy", "action_policy"):
            tr["viable"] = False
            tr["status"] = (f"مسدود — وضعیت فعلی خارج از محدودهٔ دادهٔ آموزش است "
                            f"(فاصلهٔ ویژگی {stats['feature_zmax']:.1f}σ)")
        elif (stats is not None and stats.get("regime_veto")
              and auth in ("policy", "action_policy")):
            # veto رژیم فقط به دادگاهِ سیاست مربوط است (تنها مرجعِ باقی‌مانده).
            tr["viable"] = False
            rs = stats.get("regime_stats") or {}
            tr["status"] = (f"مسدود — بخش آزمون مستقل در رژیم «{regime_code}» زیان ساختاری نشان داده "
                            f"(n={rs.get('n', 0)}، میانگین خالص={rs.get('avg_net_r') or 0:+.2f}R)")
        elif auth in ("policy", "action_policy") and stats is not None and not stats.get("policy_pass"):
            tr["viable"] = False
            tr["status"] = (f"ردِ سیاست نهایی — امتیاز ترکیبیِ احتمال/بازده "
                            f"{stats.get('policy_margin') or 0:+.2f} زیر آستانه است")
        elif ex.get("tf_suspended") is not None:
            tr["viable"] = False
            tr["status"] = (f"معلق — بازده خالصِ زندهٔ این تایم‌فریم منفی است "
                            f"({ex['tf_suspended']:+.2f}R) — تا بهبودِ آمار، سیگنال تازه پیشنهاد نمی‌شود")
        elif combo_side in susp_combos or combo in susp_combos:
            bad = susp_combos.get(combo_side, susp_combos.get(combo))
            tr["viable"] = False
            tr["status"] = (f"معلق — ترکیب زندهٔ {combo} ضعیف است "
                            f"(بازده خالص: {bad:+.2f}R)")
        elif setup_used in suspended:
            tr["viable"] = False
            tr["status"] = (f"معلق — عملکردِ زندهٔ ستاپ «{SETUP_FA.get(setup_used, setup_used)}» ضعیف است "
                            f"(بازده خالص: {suspended[setup_used]:+.2f}R) — تا بازآموزی بعدی خاموش")
        elif btc_z is not None and btc_z * d < -0.8:
            tr["viable"] = False
            tr["status"] = f"مسدود — مخالفتِ شدید با روند بیت‌کوین (z={btc_z:+.1f})"
        else:
            if dir_stats:
                dp = dir_stats["p_up"]
                dq = dir_stats.get("p_q") or {}
                lo_n, hi_n = dq.get("q40", 48), dq.get("q60", 52)
                against = (d == 1 and dp < lo_n) or (d == -1 and dp > hi_n)
                neutral = lo_n <= dp <= hi_n
                if against:
                    tr["direction_warning"] = f"جهت‌یاب کمکی مخالف است (احتمال رشد {dp:.0f}٪)"
                elif neutral:
                    tr["direction_warning"] = f"جهت‌یاب کمکی خنثی است ({dp:.0f}٪)"
        # ── قفلِ ایمنیِ سرمایه: تنها فهرستِ پیش‌ثبت‌شدهٔ gates.json اجازهٔ اجرا می‌دهد ──
        gate_ok = gates.is_combo_allowed(tf, setup_used, tr["side"])
        tr["gate_allowed"] = gate_ok
        if tr["viable"] and not gate_ok:
            tr["viable"] = False
            tr["status"] = (f"قفل ایمنی — ترکیب {combo_side} در فهرست مجاز gates.json نیست؛ "
                            "ورود فقط پس از پیش‌ثبت و عبور از آزمون منجمد")
        tr["tradeable"] = bool(tr["viable"])
        # ── متا-گیت (لایهٔ دوم تصمیم) — فقط روی سیگنال‌های مجوزدار ──
        if tr["tradeable"] and tr.get("side"):
            try:
                import meta_gate
                hyst = meta_gate.adx_hysteresis(cs["adx"], cs["chop"], need=3)
                gate_extras = dict(ex)
                gate_extras["adx_hysteresis"] = hyst
                gate_extras["setup"] = setup_used
                gate_extras["btc_z"] = btc_z
                meta = meta_gate.evaluate(tr["side"], gate_extras, cs)
                tr["meta_score"] = meta["score"]
                tr["meta_size_mult"] = meta["size_mult"]
                tr["meta_regime"] = meta.get("regime")
                tr["meta_reasons"] = meta.get("reasons") or []
                tr["meta_blocks"] = meta.get("blocks") or []
                tr["meta_confluence"] = meta.get("confluence")
                tr["meta_session"] = meta.get("session_tier")
                tr["oi_z"] = meta.get("oi_z")
                if not meta.get("approve"):
                    tr["viable"] = False
                    tr["tradeable"] = False
                    why = "؛ ".join(meta.get("blocks") or ["امتیاز متا پایین"])
                    tr["status"] = f"ردِ متا-گیت — {why}"
                else:
                    base = tr.get("status") or "مجوز"
                    tr["status"] = (f"{base} · متا {meta['score']:.0%} "
                                    f"· تلاقی {meta.get('confluence', 0)} "
                                    f"(حجم×{meta['size_mult']:.2f})")
            except Exception:  # noqa: BLE001
                tr["meta_score"] = None
                tr["meta_size_mult"] = 1.0
        if not tr["tradeable"]:
            tr["recommendation"] = None
            tr["observe_only"] = True
        else:
            tr["recommendation"] = tr["side"]
            tr["observe_only"] = False

    # ── جهت‌یاب فقط بایاس تشخیصی است — هرگز side/grade خرید نمی‌سازد ──
    if not tr.get("side") and dir_stats:
        pgo = dir_stats["p_up"]
        _thr = dir_stats.get("dm_thr") or [40.0, 60.0]
        dm_side = "long" if pgo >= _thr[1] else "short" if pgo <= _thr[0] else None
        score_l0 = float(cs["score_s"][-1])
        dm_disagree = (dm_side == "long" and score_l0 < -0.15) or (dm_side == "short" and score_l0 > 0.15)
        btc_bear = btc_z is not None and btc_z < -0.05
        knife = dm_disagree and (
            (dm_side == "long" and btc_bear)
            or (dm_side == "short" and btc_z is not None and btc_z > 0.05)
        )
        if knife:
            _mk = "نزولی" if dm_side == "long" else "صعودی"
            tr["status"] = f"مشاهده — بازگشتِ خلافِ روند در بازار {_mk} بیت‌کوین (چاقوی سقوط)"
        elif dm_side:
            tr["dir_bias"] = dm_side
            tr["status"] = (f"جهت‌یاب سمتِ احتمالی را {('صعودی' if dm_side == 'long' else 'نزولی')} "
                            f"می‌بیند ({pgo:.0f}٪) — منتظر ستاپ رویدادی و مرجع تصمیم")
        elif abs(pgo - 50) >= 5:
            tr["status"] = f"جهت {'صعودی' if pgo > 50 else 'نزولی'} ({pgo:.0f}٪) — منتظر نقطه ورود"

    # ── امتیاز فقط وقتی مرجع تصمیم روشن و ورود مجاز است ──
    # امتیاز ۱۲/۱۰۰ کنار «خرید A» همان ناهماهنگی بود: ستاپ دیده می‌شد ولی سیاست مردود بود.
    if tr.get("side") and tr.get("tradeable"):
        pst = tr.get("policy_test") or {}
        policy_base = 35
        margin_comp = clamp((tr.get("policy_margin") or 0.0) / 1.5, 0, 1) * 25
        test_avg_comp = clamp(float(pst.get("avg_net_r") or 0.0) / 0.40, 0, 1) * 10
        test_pf_comp = clamp((float(pst.get("profit_factor") or 1.0) - 1.0) / 0.80, 0, 1) * 10
        rel_comp = {"خوب": 10, "متوسط": 6, "کم": 2}.get(tr.get("reliability"), 0)
        grd_comp = {"A+": 10, "A": 7, "B": 4}.get(tr.get("grade"), 0)
        regime_comp = 5 if tr.get("regime_ok") else 0
        score100 = round(
            policy_base + margin_comp + test_avg_comp + test_pf_comp
            + rel_comp + grd_comp + regime_comp
        )
        d_tr = 1 if tr["side"] == "long" else -1
        score_now = float(cs["score_s"][-1])
        contrarian = (d_tr == 1 and score_now < -0.15) or (d_tr == -1 and score_now > 0.15)
        if contrarian:
            score100 = min(score100, 48)
            tr["contrarian"] = True
        if btc_z is not None and btc_z * d_tr < -0.05:
            score100 = round(score100 * 0.7)
        br_m = ex.get("breadth", 0.0) or 0.0
        if (d_tr == 1 and br_m < -0.3) or (d_tr == -1 and br_m > 0.3):
            score100 = round(score100 * 0.85)
            tr["market_headwind"] = True
        score100 = min(max(score100, 0), 100)
        tr["signal_score"] = score100
        uncertainty_penalty = clamp((tr.get("p_uncertainty") or 25.0) / 25.0, 0, 1)
        tr["model_confidence"] = round(clamp(score100 / 100.0, 0, 1) * (1.0 - 0.35 * uncertainty_penalty), 3)
        tr["signal_tier"] = ("عالی" if score100 >= 75 else "قوی" if score100 >= 60
                             else "متوسط" if score100 >= 45 else "ضعیف")
    elif tr.get("side"):
        tr["signal_score"] = None
        tr["signal_tier"] = None
        tr["model_confidence"] = 0.0

    adx_l, chop_l = float(cs["adx"][-1]), float(cs["chop"][-1])
    regime = {"volatile": "پرنوسان", "trend": "رونددار",
              "range": "رنج", "transition": "انتقالی"}[regime_code]

    p_up = fc["p_up"]
    if dir_stats:
        # اولویت با مدل جهت‌یاب همیشه‌روشن (آموزش‌دیده روی ده‌ها هزار نمونه، تأییدشده OOS)
        p_up = dir_stats["p_up"]
        calibrated = True
    elif calibrated and tr.get("p_win") is not None:
        # احتمال از فراوانی واقعی ستاپ‌های مشابه تاریخی (کالیبره)
        p_up = tr["p_win"] if tr["side"] == "long" else 100 - tr["p_win"]
    elif bt["win_rate"] is not None and bt["n"] >= 8:
        bias = bt["win_rate"] if float(cs["score_s"][-1]) >= 0 else 100 - bt["win_rate"]
        p_up = 0.65 * p_up + 0.35 * bias

    return {
        "tf": tf,
        "price": float(c[-1]),
        "score": round(float(cs["score_s"][-1]), 3),
        "z": round(float(cs["z"][-1]), 2),
        "z_prev": round(float(cs["z"][-2]), 3) if n >= 2 else 0.0,   # برای هم‌ترازی جهتِ تایم بالاتر (بدون نشتی)
        "zt": int(kl["t"][-1]),                                     # زمان بازِ آخرین کندل بسته‌شده
        "votes_bull": vb + (1 if s_ml >= 0.2 else 0),
        "votes_bear": vs + (1 if s_ml <= -0.2 else 0),
        "components": {
            "روند": round(float(cs["s_trend"][-1]), 2),
            "مومنتوم": round(float(cs["s_mom"][-1]), 2),
            "حجم": round(float(cs["s_vol"][-1]), 2),
            "ساختار": round(float(cs["s_struct"][-1]), 2),
            "هوش مصنوعی": round(s_ml, 2),
        },
        "regime": regime, "adx": round(adx_l, 1), "chop": round(chop_l, 1), "atr_pct": round(atr_pct, 0),
        "p_up": round(p_up, 1),
        "p_calibrated": calibrated,
        "forecast": {k: (sig_round(vv) if isinstance(vv, float) else vv)
                     for k, vv in fc.items() if k not in ("path_log", "trend_w", "conf", "sig1")},
        "trade": {k: (sig_round(vv) if isinstance(vv, float) else vv) for k, vv in tr.items()},
        "backtest": bt,
    }
