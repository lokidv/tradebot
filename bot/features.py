# -*- coding: utf-8 -*-
"""تنها مسیرِ ساختِ ورودیِ مدل — یکسان برای آموزش و اجرا.

تابعِ ویژگی (``engine.event_features``) از اول مشترک بود، ولی **ورودی‌هایش** نبود
و همان جا ناسازگاری می‌ساخت:

* ``funding_dir`` — آموزش از ``market.funding_z_map`` (z غلتان، نرخِ جاری بیرونِ
  پنجره) و اجرا از ``meta_gate.funding_crowd_stats`` (نرخِ جاری داخلِ پنجره)
  می‌خواند؛ |z| زندهٔ ۵ تا ۱۰ درصد کوچک‌تر بود.
* ``dxy_dir`` — منبعش خالی بود و ویژگی در سراسرِ آموزش ثابتِ ۰ ماند؛ حذف شد.

اینجا هم سازندهٔ زمینه (context) هست و هم گزارشِ سلامتِ ویژگی‌ها، تا ساختی که
ویژگیِ مرده یا کم‌پوشش دارد سر و صدا کند به‌جای اینکه بی‌صدا مدل را بی‌اطلاع کند.
"""
import bisect
import math

import engine
import market

FEATS = engine.FEATS
N_FEATS = len(FEATS)

MIN_COVERAGE = 0.50        # ویژگی‌ای که بیش از نیمی از ردیف‌ها صفر است، مرده حساب می‌شود
MIN_VARIANCE = 1e-9
PSI_WARN = 0.20            # جابه‌جاییِ توزیعِ آموزش/اجرا

# زمینه‌های بیرونی که «۰» در آن‌ها یعنی «داده نبود»، نه یک مقدارِ واقعی (htf_align: ساختِ تک‌تایم‌فریمی
# مثلِ --tfs 4h نقشهٔ تایم‌بالاتر ندارد و در آموزش همه‌جا ۰ است، ولی اجرا جهتِ 1d را دارد)
MISSING_AS_ZERO = ("funding_dir", "rs_dir", "breadth_dir", "gold_dir", "dom_dir", "ethbtc_dir",
                   "htf_align")
# عمداً در آموزش **و** اجرا خنثی (calib-F1): برنامه زنده فقط پنج ارز را می‌خواند (watchlist)،
# ولی تعریفِ آموزشیِ این دو رتبه/پهنا بینِ ۱۰۰ ارز با کفِ جمعیتِ ۴۰ بود؛ زنده همیشه ۰/۰٫۵
# می‌شدند. ساختنشان از پنج ارز هم متا-گیت را عوض می‌کرد (همان کلیدهای breadth/rs_rank با
# آستانه‌های تنظیم‌شده برای ۱۰۰ ارز) — تصمیمی محصولی، نه اصلاحِ ورودیِ مدل.
DISABLED = ("rs_dir", "breadth_dir")
# زیرِ این پوشش در آموزش، ویژگیِ «۰ = نبود» در اجرا هم ۰ می‌شود (calib-F4): با پوششِ ۰٫۵٪،
# انحراف‌معیارِ آموزش ~۰٫۰۲ است و مقدارِ زندهٔ عادی |z|≈۵-۱۱ می‌سازد (وتوی zmax>6 و پیش‌بینیِ پرت).
LIVE_MIN_COVERAGE = 0.10


def context(funding_z=0.0, rs_rank=0.5, htf_sign=0, btc_align=True,
            gold_m=0.0, breadth_m=0.0, dom_m=0.0, ethbtc_m=0.0):
    """زمینهٔ ویژگی‌ها با نام‌های صریح — همان کلیدهایی که ``event_features`` می‌خواهد."""
    return {
        "funding_z": float(funding_z or 0.0),
        "rs_rank": float(rs_rank if rs_rank is not None else 0.5),
        "htf_sign": int(htf_sign or 0),
        "btc_align": bool(btc_align),
        "gold_m": float(gold_m or 0.0),
        "breadth_m": float(breadth_m or 0.0),
        "dom_m": float(dom_m or 0.0),
        "ethbtc_m": float(ethbtc_m or 0.0),
    }


def context_from_extras(ex, btc_align=True):
    """زمینه از دیکشنریِ ``extras`` که ``main.get_analysis`` می‌سازد."""
    ex = ex or {}
    return context(
        funding_z=ex.get("funding_z", 0.0),
        rs_rank=ex.get("rs_rank", 0.5),
        htf_sign=ex.get("htf_sign", 0),
        btc_align=btc_align,
        gold_m=ex.get("gold", 0.0),
        breadth_m=ex.get("breadth", 0.0),
        dom_m=ex.get("dom", 0.0),
        ethbtc_m=ex.get("ethbtc", 0.0),
    )


def build(cs, c, i, sig, setup, ts_ms, ctx):
    """بردارِ ویژگی — تنها نقطهٔ ساخت، چه در آموزش چه در اجرا."""
    return engine.event_features(cs, c, i, sig, setup, ts_ms, **ctx)


def funding_z_at(fz_rows, bar_open_ms, bar_ms, keys=None):
    """z فاندینگِ آخرین تسویه‌ای که **پیش از بسته‌شدنِ** کندل رخ داده (fundingTime < باز + طول).

    یک قاعده برای آموزش و اجرا (calib-F3). قبلاً آموزش «تسویهٔ ≤ بازِ کندل» را می‌گرفت و اجرا
    «آخرین ردیفِ کش» (تا ۱۲ ساعت کهنه، بی‌فیلتر) را؛ روی 1d هر کندل تا سه تسویه جابه‌جا بود.
    تسویه‌ای که دقیقاً روی مرزِ بسته‌شدن است (بایننس با +۰..۴ms) کنار می‌ماند تا اجرا به
    انتشارِ لحظه‌ایِ آن وابسته نباشد. ``keys`` (زمان‌های ``fz_rows``) برای حلقه‌های آموزش یک‌بار ساخته می‌شود.
    """
    if not fz_rows:
        return 0.0
    keys = keys if keys is not None else [r[0] for r in fz_rows]
    i = bisect.bisect_left(keys, int(bar_open_ms) + int(bar_ms)) - 1
    return float(fz_rows[i][1]) if i >= 0 else 0.0


def live_funding_z(symbol, bar_open_ms=None, bar_ms=None):
    """``funding_dir`` زنده با **همان** فرمول و همان هم‌ترازیِ آموزش (``market.funding_z_map``).

    با ``bar_open_ms``/``bar_ms`` مقدارِ کندلِ تحلیل‌شده (``funding_z_at``)؛ بی آن‌ها آخرین ردیف.
    عمداً از ``meta_gate.funding_crowd_stats`` استفاده نمی‌شود؛ آن برای وتوی
    شلوغی است و نرخِ جاری را داخلِ پنجرهٔ خودش می‌آورد.
    """
    try:
        rows = market.funding_z_map(symbol)
    except Exception:  # noqa: BLE001
        return 0.0
    if not rows:
        return 0.0
    if bar_open_ms is None or not bar_ms:
        return float(rows[-1][1])
    return funding_z_at(rows, bar_open_ms, bar_ms)


def expected_constant(tf, dense=False):
    """ویژگی‌هایی که بی‌واریانس بودنشان در این ماتریس **طبیعی** است — «مرده» گزارش نمی‌شوند.

    خنثی‌های عمدی؛ ساعت روی 1d (همهٔ کندل‌ها ۰۰:۰۰ باز می‌شوند)؛ پرچم‌های ستاپ در نمونه‌های
    متراکم (بی ستاپ). وگرنه هشدارِ «مرده» هر ساخت تکرار می‌شد و هشدارِ واقعی گم می‌شد.
    """
    out = list(DISABLED)
    if tf == "1d":
        out += ["hour_sin", "hour_cos"]
    if dense:
        out += ["setup_pb", "setup_sq", "setup_fd", "setup_rg"]
    return tuple(out)


def live_zeroed(report):
    """از گزارشِ ``health``: ویژگی‌های «۰ = نبود» که پوششِ آموزشی‌شان زیرِ ``LIVE_MIN_COVERAGE`` است."""
    feats = (report or {}).get("features") or {}
    out = []
    for name in MISSING_AS_ZERO:
        rep = feats.get(name)
        if rep is not None and float(rep.get("coverage") or 0.0) < LIVE_MIN_COVERAGE:
            out.append(name)
    return out


def zero_features(vec, names):
    """کپیِ ``vec`` با ویژگی‌های ``names`` روی ۰ — همان مقداری که مدل در آموزش تقریباً همیشه دید."""
    if not names:
        return vec
    out = list(vec)
    for name in names:
        if name in FEATS:
            k = FEATS.index(name)
            if k < len(out):
                out[k] = 0.0
    return out


# ───────────────────────── سلامتِ ویژگی‌ها ─────────────────────────
def _coverage(col):
    n = len(col)
    return float(sum(1 for v in col if abs(float(v)) > 1e-12) / n) if n else 0.0


def _variance(col):
    n = len(col)
    if n < 2:
        return 0.0
    mu = sum(col) / n
    return float(sum((v - mu) ** 2 for v in col) / n)


def psi(train_col, live_col, bins=10):
    """Population Stability Index — چقدر توزیعِ اجرا از آموزش دور شده است."""
    if not train_col or not live_col:
        return None
    lo, hi = min(train_col), max(train_col)
    if hi - lo < 1e-12:
        return 0.0
    width = (hi - lo) / bins
    out = 0.0
    for b in range(bins):
        a, z = lo + b * width, lo + (b + 1) * width
        last = b == bins - 1
        t = sum(1 for v in train_col if (a <= v <= z if last else a <= v < z)) / len(train_col)
        l = sum(1 for v in live_col if (a <= v <= z if last else a <= v < z)) / len(live_col)
        t, l = max(t, 1e-4), max(l, 1e-4)
        out += (l - t) * math.log(l / t)
    return round(out, 4)


def health(X, names=None, expected=()):
    """گزارشِ پوشش و واریانسِ هر ویژگی روی ماتریسِ آموزش.

    سه سطحِ جدا:
    * ``dead`` — واریانسِ ~صفر: ویژگی هیچ اطلاعاتی ندارد و مدل نمی‌تواند یادش بگیرد.
      این وضعِ ``dxy_dir`` بود که سال‌ها ثابتِ ۰ ماند بدونِ اینکه کسی بفهمد. ``ok`` را
      خاموش می‌کند؛ اگر از خانوادهٔ «۰ = نبود» باشد، ``live_zeroed`` آن را در اجرا هم ۰ نگه
      می‌دارد تا مدل ورودیِ ندیده نگیرد.
    * ``expected_constant`` — بی‌واریانس ولی طبیعی (``expected``، مثلاً ساعت روی 1d).
    * ``low_coverage`` — بیش از نیمی از ردیف‌ها صفرند (مثل ``funding_dir`` که برای
      ۷۶٪ نمادها داده ندارد). ضعیف است ولی واقعی؛ فقط هشدار.
    """
    names = list(names or FEATS)
    expected = set(expected or ())
    rows = [list(map(float, r)) for r in X]
    if not rows:
        return {"n": 0, "n_feat": 0, "features": {}, "dead": list(names),
                "low_coverage": [], "expected_constant": [], "ok": False}
    cols = list(zip(*rows))
    report, dead, low, const = {}, [], [], []
    for k, name in enumerate(names):
        if k >= len(cols):
            dead.append(name)
            continue
        col = list(cols[k])
        cov, var = _coverage(col), _variance(col)
        report[name] = {"coverage": round(cov, 4), "variance": round(var, 8),
                        "mean": round(sum(col) / len(col), 6),
                        "min": round(min(col), 6), "max": round(max(col), 6)}
        if var < MIN_VARIANCE and name in expected:
            const.append(name)
            report[name]["expected_constant"] = True
        elif var < MIN_VARIANCE:
            dead.append(name)
            report[name]["dead"] = True
        elif cov < MIN_COVERAGE:
            low.append(name)
            report[name]["low_coverage"] = True
    return {"n": len(rows), "n_feat": len(cols), "features": report,
            "dead": dead, "low_coverage": low, "expected_constant": const, "ok": not dead}
