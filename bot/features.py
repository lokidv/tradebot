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
import math

import engine
import market

FEATS = engine.FEATS
N_FEATS = len(FEATS)

MIN_COVERAGE = 0.50        # ویژگی‌ای که بیش از نیمی از ردیف‌ها صفر است، مرده حساب می‌شود
MIN_VARIANCE = 1e-9
PSI_WARN = 0.20            # جابه‌جاییِ توزیعِ آموزش/اجرا


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


def live_funding_z(symbol):
    """``funding_dir`` زنده با **همان** فرمولِ آموزش (``market.funding_z_map``).

    عمداً از ``meta_gate.funding_crowd_stats`` استفاده نمی‌شود؛ آن برای وتوی
    شلوغی است و نرخِ جاری را داخلِ پنجرهٔ خودش می‌آورد.
    """
    try:
        rows = market.funding_z_map(symbol)
    except Exception:  # noqa: BLE001
        return 0.0
    return float(rows[-1][1]) if rows else 0.0


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


def health(X, names=None):
    """گزارشِ پوشش و واریانسِ هر ویژگی روی ماتریسِ آموزش.

    دو سطحِ جدا:
    * ``dead`` — واریانسِ ~صفر: ویژگی هیچ اطلاعاتی ندارد و مدل نمی‌تواند یادش بگیرد.
      این وضعِ ``dxy_dir`` بود که سال‌ها ثابتِ ۰ ماند بدونِ اینکه کسی بفهمد. **کشنده.**
    * ``low_coverage`` — بیش از نیمی از ردیف‌ها صفرند (مثل ``funding_dir`` که برای
      ۷۶٪ نمادها داده ندارد). ضعیف است ولی واقعی؛ فقط هشدار.
    """
    names = list(names or FEATS)
    rows = [list(map(float, r)) for r in X]
    if not rows:
        return {"n": 0, "n_feat": 0, "features": {}, "dead": list(names),
                "low_coverage": [], "ok": False}
    cols = list(zip(*rows))
    report, dead, low = {}, [], []
    for k, name in enumerate(names):
        if k >= len(cols):
            dead.append(name)
            continue
        col = list(cols[k])
        cov, var = _coverage(col), _variance(col)
        report[name] = {"coverage": round(cov, 4), "variance": round(var, 8),
                        "mean": round(sum(col) / len(col), 6),
                        "min": round(min(col), 6), "max": round(max(col), 6)}
        if var < MIN_VARIANCE:
            dead.append(name)
            report[name]["dead"] = True
        elif cov < MIN_COVERAGE:
            low.append(name)
            report[name]["low_coverage"] = True
    return {"n": len(rows), "n_feat": len(cols), "features": report,
            "dead": dead, "low_coverage": low, "ok": not dead}
