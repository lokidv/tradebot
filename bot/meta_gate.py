# -*- coding: utf-8 -*-
"""متا-گیت زنده — لایهٔ دوم تصمیم (Lopez de Prado meta-labeling در عمل).

تحقیق ۲۰۲۵-۲۰۲۶ (AFML، regime-filter bots، funding crowding، OI leverage، breadth):
سیگنال اولیه جهت را می‌گوید؛ متا-گیت فقط می‌پرسد «آیا الان ارزش معامله دارد؟».

ویژگی‌ها عمداً متعامد با ستاپ قیمتی‌اند:
- رژیم کلان BTC (SMA100/200 + پهنا)
- شلوغی فاندینگ × Open Interest (اهرمی)
- هم‌راستایی پهنای بازار
- پایداری رژیم ADX (hysteresis)
- کیفیت نشست/نقدینگی نسبی
"""
from __future__ import annotations

import math
import time
from typing import Any

import numpy as np

import engine


def _sma(x, n):
    x = np.asarray(x, float)
    if len(x) < n:
        return float("nan")
    return float(np.mean(x[-n:]))


def btc_macro_regime(btc_closes, breadth01_signed=0.0):
    """رژیم کلان به سبک coinregime: bull / chop / bear."""
    c = np.asarray(btc_closes, float)
    if len(c) < 210:
        return {"regime": "chop", "btc_vs_sma100": 0.0, "btc_vs_sma200": 0.0, "breadth": breadth01_signed}
    sma100, sma200 = _sma(c, 100), _sma(c, 200)
    px = float(c[-1])
    vs100 = (px / sma100 - 1.0) if sma100 > 0 else 0.0
    vs200 = (px / sma200 - 1.0) if sma200 > 0 else 0.0
    pct_above = (breadth01_signed + 1.0) * 50.0
    if vs100 > 0 and vs200 > 0 and pct_above >= 50.0:
        regime = "bull"
    elif vs200 < 0 and pct_above <= 35.0:
        regime = "bear"
    else:
        regime = "chop"
    return {
        "regime": regime,
        "btc_vs_sma100": round(vs100 * 100, 2),
        "btc_vs_sma200": round(vs200 * 100, 2),
        "breadth_pct_above": round(pct_above, 1),
        "breadth": breadth01_signed,
    }


def funding_crowd_stats(funding_rows):
    """از تاریخچهٔ (ts, rate) شلوغی و پایداری علامت را می‌سازد."""
    if not funding_rows or len(funding_rows) < 8:
        return {"funding_z": 0.0, "persist": 0, "crowded_long": False, "crowded_short": False}
    rates = [float(r[1]) for r in funding_rows[-40:]]
    window = rates[-30:] if len(rates) >= 30 else rates
    mu = sum(window) / len(window)
    var = sum((x - mu) ** 2 for x in window) / max(len(window), 1)
    sd = math.sqrt(var) or 1e-9
    z = (rates[-1] - mu) / sd
    sign = 1 if rates[-1] >= 0 else -1
    persist = 0
    for r in reversed(rates):
        if (1 if r >= 0 else -1) != sign:
            break
        persist += 1
    return {
        "funding_z": round(z, 3),
        "persist": persist,
        "crowded_long": z >= 2.0 and persist >= 3,
        "crowded_short": z <= -2.0 and persist >= 3,
        "last_rate": rates[-1],
    }


def adx_hysteresis(adx_series, chop_series, need=3):
    """رژیم پایدار: حداقل need کندل اخیر هم‌رژیم باشند تا whipsaw کم شود."""
    adx = np.asarray(adx_series, float)
    chop = np.asarray(chop_series, float)
    if len(adx) < need or len(chop) < need:
        return {"stable": False, "local": "unknown", "streak": 0}

    def label(i):
        a, ch = float(adx[i]), float(chop[i])
        if a >= 25 and ch < 55:
            return "trend"
        if a < 20 or ch > 61.8:
            return "range"
        return "transition"

    recent = [label(i) for i in range(-need, 0)]
    stable = len(set(recent)) == 1
    streak = 1
    for i in range(2, min(len(adx), 20) + 1):
        if label(-i) != recent[-1]:
            break
        streak += 1
    return {"stable": stable, "local": recent[-1], "streak": streak, "recent": recent}


def session_quality(now_ts=None):
    """کیفیت تقریبی نقدینگی بر اساس ساعت UTC (کریپتو ۲۴/۷ است ولی عمق فرق دارد)."""
    h = time.gmtime(now_ts or time.time()).tm_hour
    # هم‌پوشانی لندن/نیویورک معمولاً عمیق‌تر؛ آسیای میانی رقیق‌تر
    if 12 <= h < 21:
        return {"tier": "high", "mult": 1.0, "hour_utc": h}
    if 7 <= h < 12 or 21 <= h < 24:
        return {"tier": "mid", "mult": 0.92, "hour_utc": h}
    return {"tier": "thin", "mult": 0.78, "hour_utc": h}


def evaluate(side: str, extras: dict, cs=None) -> dict[str, Any]:
    """خروجی: approve, score∈[0,1], reasons, size_mult, blocks."""
    d = 1 if side == "long" else -1
    reasons = []
    blocks = []
    score = 0.55

    macro = extras.get("btc_macro") or {}
    regime = macro.get("regime") or "chop"
    breadth = float(extras.get("breadth", macro.get("breadth", 0.0)) or 0.0)
    fz = float(extras.get("funding_z", 0.0) or 0.0)
    persist = int(extras.get("funding_persist", 0) or 0)
    crowded_long = bool(extras.get("crowded_long"))
    crowded_short = bool(extras.get("crowded_short"))
    if not crowded_long and not crowded_short:
        crowded_long = fz >= 2.0 and persist >= 3
        crowded_short = fz <= -2.0 and persist >= 3

    oi_z = float(extras.get("oi_z") or 0.0)
    oi_chg = float(extras.get("oi_chg_pct") or 0.0)
    oi_rising = bool(extras.get("oi_rising"))
    oi_falling = bool(extras.get("oi_falling"))
    oi_ok = bool(extras.get("oi_ok"))

    # ۱) رژیم کلان BTC
    extreme_bull = regime == "bull" and float(macro.get("btc_vs_sma100") or 0) > 8 and breadth > 0.45
    extreme_bear = regime == "bear" and float(macro.get("btc_vs_sma100") or 0) < -8 and breadth < -0.45
    if extreme_bear and d == 1:
        blocks.append("رژیم کلان نزولیِ شدید BTC — لانگ آلت رد شد")
        score -= 0.40
    elif extreme_bull and d == -1:
        blocks.append("رژیم کلان صعودیِ شدید BTC — شورت رد شد")
        score -= 0.35
    elif regime == "bear" and d == 1:
        reasons.append("احتیاط: رژیم کلان نزولی")
        score -= 0.18
    elif regime == "bull" and d == -1:
        reasons.append("احتیاط: رژیم کلان صعودی")
        score -= 0.12
    elif regime == "bull" and d == 1:
        reasons.append("رژیم کلان صعودی هم‌راستا")
        score += 0.12
    elif regime == "bear" and d == -1:
        reasons.append("رژیم کلان نزولی هم‌راستا")
        score += 0.12
    else:
        reasons.append("رژیم کلان خنثی/انتقالی")
        score += 0.02

    # ۲) پهنای بازار
    if d == 1 and breadth < -0.35:
        blocks.append(f"پهنای بازار ضعیف است ({breadth:+.2f}) — لانگ خلاف جریان")
        score -= 0.25
    elif d == -1 and breadth > 0.35:
        blocks.append(f"پهنای بازار قوی است ({breadth:+.2f}) — شورت خلاف جریان")
        score -= 0.25
    elif (d == 1 and breadth > 0.1) or (d == -1 and breadth < -0.1):
        reasons.append("پهنای بازار هم‌جهت")
        score += 0.10

    # ۳) crowding فاندینگ
    if d == 1 and crowded_long:
        blocks.append(f"لانگ‌ها شلوغ‌اند (funding z={fz:+.1f}, persist={persist})")
        score -= 0.30
    elif d == -1 and crowded_short:
        blocks.append(f"شورت‌ها شلوغ‌اند (funding z={fz:+.1f}, persist={persist})")
        score -= 0.30
    elif d == 1 and fz <= -1.0:
        reasons.append("فاندینگ به نفع لانگ")
        score += 0.08
    elif d == -1 and fz >= 1.0:
        reasons.append("فاندینگ به نفع شورت")
        score += 0.08

    # ۴) تعامل funding × OI — اهرم شلوغ = ریسک اسکوییز
    if oi_ok:
        # لانگ وقتی فاندینگ مثبتِ شدید + OI بالا می‌آید = crowded long squeeze risk
        if d == 1 and fz >= 1.2 and (oi_rising or oi_z >= 1.0):
            blocks.append(f"اهرم لانگ در حال انباشت (OI z={oi_z:+.1f}, Δ{oi_chg:+.1f}٪)")
            score -= 0.22
        elif d == -1 and fz <= -1.2 and (oi_rising or oi_z >= 1.0):
            blocks.append(f"اهرم شورت در حال انباشت (OI z={oi_z:+.1f}, Δ{oi_chg:+.1f}٪)")
            score -= 0.22
        # تأیید: حرکت قیمت هم‌جهت با کاهش OI (پوشش) یا افزایش OI در جهت سالم
        elif d == 1 and oi_falling and fz > 0.5:
            reasons.append("کاهش OI با فاندینگ مثبت — احتمالاً شورت‌کاور")
            score += 0.07
        elif d == -1 and oi_falling and fz < -0.5:
            reasons.append("کاهش OI با فاندینگ منفی — احتمالاً لانگ‌آنویند")
            score += 0.07
        elif abs(oi_z) < 0.5:
            score += 0.02

    # ۵) hysteresis رژیم محلی
    hyst = extras.get("adx_hysteresis") or {}
    if hyst:
        if not hyst.get("stable"):
            blocks.append(f"رژیم محلی ناپایدار ({'/'.join(hyst.get('recent') or [])})")
            score -= 0.15
        elif hyst.get("local") == "trend":
            reasons.append(f"روند پایدار ({hyst.get('streak', 0)} کندل)")
            score += 0.10
        elif hyst.get("local") == "range" and extras.get("setup") in ("zx",):
            score -= 0.08
            reasons.append("رنج پایدار — کراس z احتیاطی")

    # ۶) هم‌راستایی BTC z
    btc_z = extras.get("btc_z")
    if btc_z is not None:
        bz = float(btc_z)
        if bz * d > 0.3:
            reasons.append("هم‌راستا با تکانهٔ BTC")
            score += 0.08
        elif bz * d < -0.5:
            blocks.append(f"مخالف تکانهٔ BTC (z={bz:+.1f})")
            score -= 0.20

    # ۷) کیفیت نشست
    sess = extras.get("session") or session_quality()
    if sess.get("tier") == "thin":
        reasons.append("نشست کم‌عمق — حجم محتاط")
        score -= 0.06
    elif sess.get("tier") == "high":
        score += 0.04

    # ۸) قدرت نسبی (اگر باشد): لانگ روی ضعیف‌ترین‌ها / شورت روی قوی‌ترین‌ها جریمه
    rs = extras.get("rs_rank")
    if rs is not None:
        rs = float(rs)
        if d == 1 and rs < 0.25:
            score -= 0.08
            reasons.append("قدرت نسبی ضعیف برای لانگ")
        elif d == -1 and rs > 0.75:
            score -= 0.08
            reasons.append("قدرت نسبی قوی برای شورت")
        elif (d == 1 and rs > 0.6) or (d == -1 and rs < 0.4):
            score += 0.06
            reasons.append("قدرت نسبی هم‌جهت")

    score = float(engine.clamp(score, 0.0, 1.0))
    hard = any(
        "شدید" in x or "شلوغ" in x or "اهرم" in x or (x.startswith("پهنای بازار") and "خلاف" in x)
        for x in blocks
    )
    approve = (not hard) and score >= 0.45
    if hard:
        approve = False

    sess_mult = float(sess.get("mult") or 1.0)
    if not approve:
        size_mult = 0.0
    else:
        size_mult = round((0.50 + 0.65 * score) * sess_mult, 3)
        size_mult = float(engine.clamp(size_mult, 0.40, 1.15))

    return {
        "approve": approve,
        "score": round(score, 3),
        "size_mult": size_mult,
        "reasons": reasons[:5],
        "blocks": blocks[:4],
        "regime": regime,
        "hard_block": hard,
        "session_tier": sess.get("tier"),
        "oi_z": oi_z if oi_ok else None,
        "confluence": round(score * 100),
    }
