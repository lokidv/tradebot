# -*- coding: utf-8 -*-
"""دفتر جیب لبهٔ زنده — معیارهای سخت‌گیرانه برای مجوز ورود بدون سیاست OOS.

جیب فقط وقتی زنده است که:
- n کافی، میانگین مثبت، کران پایین (LCB) منفیِ عمیق نباشد
- نیمهٔ اخیر کارنامه فرو نریخته باشد
- ستاپ تمیز (zx/sq) و TF غیر از 1d
"""
from __future__ import annotations

import math
import time
from typing import Any

import shadow

# آستانه‌های ازپیش‌تعیین‌شده (قبل از دیدن جیب فعلی)
MIN_N = 18
MIN_AVG_R = 0.08
MIN_WR = 47.0
# درجه A: کران پایین محکم
MIN_LCB_A = -0.05
MIN_RECENT_A = 0.0
MIN_HALF1_A = -0.08
# درجه B (تحت‌نظر / لبهٔ نازک): هنوز میانگین مثبت، ولی نیمهٔ اخیر ضعیف‌تر
MIN_LCB_B = -0.25
MIN_RECENT_B = 0.0
MIN_HALF1_B = -0.35
SIZE_HINT = {"A": 1.0, "B": 0.65}
ALLOWED_SETUPS = ("zx", "sq")
ALLOWED_TFS = ("15m", "1h", "4h")
BLOCKED_TFS = ("1d",)

_cache = {"ts": 0.0, "pockets": {}, "suspended": {}, "health": {}}


def _lcb(avg, sd, n, z=1.28):
    """کران پایین تقریبی میانگین (یک‌طرفه ~۹۰٪)."""
    if n <= 1:
        return avg
    return avg - z * (sd / math.sqrt(n))


def _series_quality(rs: list[float]) -> dict[str, Any]:
    n = len(rs)
    if n == 0:
        return {"n": 0, "avg_r": None, "win_rate": None, "lcb_r": None,
                "sd": None, "half0": None, "half1": None, "recent": None}
    avg = sum(rs) / n
    wr = sum(1 for x in rs if x > 0) / n * 100
    var = sum((x - avg) ** 2 for x in rs) / n
    sd = math.sqrt(var)
    mid = n // 2
    h0 = sum(rs[:mid]) / mid if mid else None
    h1 = sum(rs[mid:]) / (n - mid) if n - mid else None
    # ۱۵تای آخر (یا نصف آخر اگر کمتر)
    tail = rs[-max(8, min(15, n)):]
    recent = sum(tail) / len(tail)
    return {
        "n": n,
        "avg_r": round(avg, 3),
        "win_rate": round(wr, 1),
        "lcb_r": round(_lcb(avg, sd, n), 3),
        "sd": round(sd, 3),
        "half0": round(h0, 3) if h0 is not None else None,
        "half1": round(h1, 3) if h1 is not None else None,
        "recent": round(recent, 3),
    }


def _combo_rows(days=30):
    """از signals_log ردیف‌های R را بر اساس tf|setup|side جمع می‌کند."""
    with shadow._lock:
        db = shadow._load()
    cutoff = (time.time() - days * 86400) * 1000
    buckets: dict[str, list[float]] = {}
    for r in db.get("resolved") or []:
        if r.get("ts", 0) < cutoff:
            continue
        rm = r.get("r_mult")
        if rm is None:
            continue
        key = f"{r.get('tf')}|{r.get('setup') or 'zx'}|{r.get('side') or '?'}"
        buckets.setdefault(key, []).append(float(rm))
        key2 = f"{r.get('tf')}|{r.get('setup') or 'zx'}"
        buckets.setdefault(key2, []).append(float(rm))
    return buckets


def refresh(force=False):
    """بازسازی جیب‌ها / تعلیق‌ها / سلامت."""
    if not force and time.time() - _cache["ts"] < 90:
        return _cache["pockets"], _cache["suspended"], _cache["health"]
    pockets, suspended = {}, {}
    buckets = _combo_rows(30)
    for key, rs in buckets.items():
        q = _series_quality(rs)
        n, avg, wr = q["n"], q["avg_r"], q["win_rate"]
        if avg is None or n < 12:
            continue
        parts = key.split("|")
        if len(parts) == 2:
            if (n >= 15 and avg <= -0.05) or (n >= 20 and avg <= 0.0):
                suspended[key] = avg
            continue
        if len(parts) < 3:
            continue
        tf_k, setup_k, side_k = parts[0], parts[1], parts[2]
        # تعلیق سمت‌دار
        if (n >= 15 and avg <= -0.05) or (n >= 20 and avg <= 0.0):
            suspended[key] = avg
        # جیب سودده
        if setup_k not in ALLOWED_SETUPS or tf_k not in ALLOWED_TFS or tf_k in BLOCKED_TFS:
            continue
        if side_k not in ("long", "short"):
            continue
        lcb = q["lcb_r"]
        recent = q["recent"]
        half1 = q["half1"]
        ok_a = (
            n >= MIN_N
            and avg >= 0.10
            and (wr is None or wr >= 48.0)
            and lcb is not None and lcb >= MIN_LCB_A
            and recent is not None and recent >= MIN_RECENT_A
            and (half1 is None or half1 >= MIN_HALF1_A)
        )
        ok_b = (
            n >= MIN_N
            and avg >= MIN_AVG_R
            and (wr is None or wr >= MIN_WR)
            and lcb is not None and lcb >= MIN_LCB_B
            and recent is not None and recent >= MIN_RECENT_B
            and (half1 is None or half1 >= MIN_HALF1_B)
        )
        # مسیر جایگزین: نمونهٔ زیاد با LCB غیرمنفی
        ok_alt = (
            n >= 28
            and avg >= 0.05
            and (wr is None or wr >= 48.0)
            and lcb is not None and lcb >= 0.0
            and recent is not None and recent >= 0.0
        )
        if ok_a or ok_alt or ok_b:
            grade = "A" if (ok_a or ok_alt) else "B"
            pockets[key] = {
                **q,
                "tf": tf_k,
                "setup": setup_k,
                "side": side_k,
                "alive": True,
                "grade": grade,
                "size_hint": SIZE_HINT.get(grade, 0.65),
            }
    # سلامت کلی
    health = {
        "updated": time.time(),
        "pocket_count": len(pockets),
        "pockets": pockets,
        "suspended_count": len(suspended),
        "verdict": (
            "healthy" if any(p.get("grade") == "A" for p in pockets.values())
            else "ok" if pockets
            else "frozen"
        ),
        "advice": (
            "جیب درجه A فعال — ورود با حجم عادی روی همان ترکیب‌ها"
            if any(p.get("grade") == "A" for p in pockets.values()) else
            "جیب درجه B (تحت‌نظر) — ورود با حجم کاهش‌یافته؛ نیمهٔ اخیر لبه ضعیف‌تر است"
            if pockets else
            "هیچ جیب زنده‌ای نیست — فقط مشاهده؛ ورود جدید توصیه نمی‌شود"
        ),
    }
    _cache.update(ts=time.time(), pockets=pockets, suspended=suspended, health=health)
    return pockets, suspended, health


def get_pockets():
    p, _, _ = refresh()
    return p


def get_suspended():
    _, s, _ = refresh()
    return s


def get_health():
    _, _, h = refresh()
    return h


def scan_tfs_for(level_scan: list[str]) -> list[str]:
    """اسکن را به TFهایی که جیب زنده دارند محدود کن (اگر جیب هست)."""
    pockets, _, _ = refresh()
    if not pockets:
        # بدون جیب: فقط 1h را نگه دار تا شکار جیب جدید ممکن باشد؛ 4h/1d را حذف کن
        return [tf for tf in level_scan if tf == "1h"] or ["1h"]
    alive_tfs = {p["tf"] for p in pockets.values()}
    ordered = [tf for tf in level_scan if tf in alive_tfs]
    # همیشه 1h را اگر جیب دارد اول بگذار
    if "1h" in ordered:
        ordered = ["1h"] + [t for t in ordered if t != "1h"]
    return ordered or ["1h"]
