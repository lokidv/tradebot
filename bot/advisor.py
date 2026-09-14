# -*- coding: utf-8 -*-
"""🧭 مشاورِ پوزیشنِ باز — برای هر پوزیشنِ باز، با مدلِ کالیبره حساب می‌کند نگه‌داشتن هنوز می‌ارزد یا نه.

پایهٔ ریاضی: برای قیمتِ بینِ حدضرر و هدف، احتمالِ برخورد به هدف قبل از حدضرر در گشتِ تصادفی برابرِ
نسبتِ فاصله‌هاست: P = d_sl/(d_sl+d_tp) — «فاصلهٔ بیشتر تا حدضرر = شانسِ بیشترِ رسیدن به هدف».
این پایه با جهتِ مدلِ کالیبره (p_up) در فضای logit تیلت می‌شود؛ سپس ارزشِ انتظاریِ نگه‌داشتن (به واحد R)
محاسبه و به توصیهٔ شفاف با دلیل ترجمه می‌شود."""
import math
from datetime import datetime, timezone


def _sig(z):
    return 1.0 / (1.0 + math.exp(-max(min(z, 30), -30)))


def _logit(p):
    p = max(min(p, 1 - 1e-6), 1e-6)
    return math.log(p / (1 - p))


def advise(pos, a, btc_z=None):
    """pos: پوزیشنِ باز (paper). a: تحلیلِ زندهٔ همان نماد/تایم‌فریم (حداقل p_up و price) یا None.
    خروجی: {action, title, urgency ۰-۳, reasons[], p_hit_tp, ev_hold_r, pnl_r}"""
    d = 1 if pos["side"] == "long" else -1
    price = float(pos.get("last_price") or pos["entry"])
    entry, sl, tp = float(pos["entry"]), float(pos["sl"]), float(pos["tp"])
    r0 = float(pos.get("r0") or abs(entry - sl)) or 1e-9   # ریسکِ اولیه — حتی بعد از انتقالِ حدضرر به سربه‌سر
    pnl_r = d * (price - entry) / r0
    reasons = []

    # فاصله‌ها از قیمتِ فعلی (در جهتِ معامله)
    d_tp = d * (tp - price)
    d_sl = d * (price - sl)
    if d_tp <= 0:                                    # از هدف رد شده — هر لحظه ممکن است برگردد
        return {"action": "lock_profit", "title": "ببند — از هدف عبور کرده، سود را قفل کن",
                "urgency": 3, "reasons": ["قیمت فراتر از هدفِ برنامه است؛ ماندن یعنی قمار روی سودِ ساخته‌شده"],
                "p_hit_tp": None, "ev_hold_r": None, "pnl_r": round(pnl_r, 2)}
    if d_sl <= 0:
        return {"action": "close_now", "title": "ببند — زیرِ حدضرر معلق است",
                "urgency": 3, "reasons": ["قیمت از حدضرر گذشته و پوزیشن هنوز باز است"],
                "p_hit_tp": None, "ev_hold_r": None, "pnl_r": round(pnl_r, 2)}

    # احتمالِ برخورد به هدف قبل از حدضرر: پایهٔ گشتِ تصادفی + تیلتِ مدل
    base_p = d_sl / (d_sl + d_tp)
    p_up = a.get("p_up") if a else None
    tilt = 0.0
    if p_up is not None:
        tilt = d * (p_up - 50.0) / 50.0              # مثبت = مدل هم‌جهتِ من
        p_hit = _sig(_logit(base_p) + 1.2 * tilt)
        model_txt = f"مدل: احتمال حرکت هم‌جهت {p_up if d==1 else round(100-p_up,1)}٪"
    else:
        p_hit = base_p
        model_txt = "مدل در دسترس نیست — فقط هندسهٔ فاصله‌ها"

    ev_hold_r = p_hit * (d_tp / r0) - (1 - p_hit) * (d_sl / r0)

    # سنِ پوزیشن نسبت به حدِ زمانی
    age_frac = 0.0
    try:
        opened = datetime.fromisoformat(pos["opened_at"])
        age_min = (datetime.now(timezone.utc) - opened).total_seconds() / 60
        age_frac = age_min / max(float(pos.get("time_stop_min") or 1), 1)
    except Exception:  # noqa: BLE001
        pass

    btc_against = btc_z is not None and (btc_z * d < -0.5)
    if btc_against:
        reasons.append(f"بیت‌کوین خلافِ جهتِ پوزیشن چرخیده (z={btc_z:+.1f}) — ریسکِ سیستمیک")

    flip_hard = p_up is not None and ((d == 1 and p_up <= 40) or (d == -1 and p_up >= 60))
    flip_soft = p_up is not None and ((d == 1 and p_up <= 46) or (d == -1 and p_up >= 54))

    # ── قواعدِ تصمیم (به ترتیبِ اولویت) ──
    if flip_hard:
        reasons.insert(0, f"{model_txt} — جهتِ مدل کاملاً برگشته")
        reasons.append(f"ارزشِ انتظاریِ نگه‌داشتن: {ev_hold_r:+.2f}R")
        title = "ببند — احتمالِ برگشت پایین است" if pnl_r < 0 else "ببند — سود را نگه دار، مدل برگشته"
        return {"action": "close_now", "title": title, "urgency": 3, "reasons": reasons,
                "p_hit_tp": round(p_hit * 100, 1), "ev_hold_r": round(ev_hold_r, 2), "pnl_r": round(pnl_r, 2)}

    if ev_hold_r < -0.10:
        reasons.insert(0, f"ارزشِ انتظاریِ نگه‌داشتن منفی شده ({ev_hold_r:+.2f}R) — {model_txt}")
        reasons.append(f"احتمال رسیدن به هدف قبل از حدضرر: {p_hit*100:.0f}٪")
        return {"action": "close_now",
                "title": "ببند — از این نقطه، ماندن به ضررِ توست" if pnl_r < 0 else "ببند — ادامه نمی‌ارزد، سود را بردار",
                "urgency": 3 if btc_against else 2,
                "reasons": reasons,
                "p_hit_tp": round(p_hit * 100, 1), "ev_hold_r": round(ev_hold_r, 2), "pnl_r": round(pnl_r, 2)}

    if pnl_r >= 0.7 and (flip_soft or ev_hold_r < 0.15):
        reasons.insert(0, f"سودِ {pnl_r:.1f}R ساخته‌ای ولی ادامهٔ حرکت کم‌رمق است ({model_txt}، EV نگه‌داشتن {ev_hold_r:+.2f}R)")
        return {"action": "lock_profit", "title": "سود را قفل کن — ببند یا حدضرر را بالای سربه‌سر بیاور",
                "urgency": 2, "reasons": reasons,
                "p_hit_tp": round(p_hit * 100, 1), "ev_hold_r": round(ev_hold_r, 2), "pnl_r": round(pnl_r, 2)}

    if pnl_r <= -0.5 and flip_soft:
        reasons.insert(0, f"در ضررِ {pnl_r:.1f}R هستی و مدل هم دیگر هم‌جهتت نیست ({model_txt})")
        reasons.append(f"احتمال رسیدن به هدف: فقط {p_hit*100:.0f}٪ — امیدِ برگشت آماری ضعیف است")
        return {"action": "close_now", "title": "ضرر را ببند — برنمی‌گردد",
                "urgency": 2, "reasons": reasons,
                "p_hit_tp": round(p_hit * 100, 1), "ev_hold_r": round(ev_hold_r, 2), "pnl_r": round(pnl_r, 2)}

    if pnl_r >= 1.0 and d * (entry - sl) > 0:        # حدضرر هنوز زیرِ ورود است
        reasons.insert(0, f"سود {pnl_r:.1f}R — دیگر نباید این معامله به ضرر تبدیل شود")
        return {"action": "move_be", "title": "حدضرر را به سربه‌سر (یا بالاتر) منتقل کن",
                "urgency": 1, "reasons": reasons,
                "p_hit_tp": round(p_hit * 100, 1), "ev_hold_r": round(ev_hold_r, 2), "pnl_r": round(pnl_r, 2)}

    if age_frac >= 0.7 and abs(pnl_r) < 0.3 and (p_up is None or 45 <= p_up <= 55):
        reasons.insert(0, f"{age_frac*100:.0f}٪ از مهلتِ معامله گذشته و قیمت جایی نرفته — سرمایه را آزاد کن")
        return {"action": "time_exit", "title": "خروجِ زمانیِ زودهنگام را در نظر بگیر",
                "urgency": 1, "reasons": reasons,
                "p_hit_tp": round(p_hit * 100, 1), "ev_hold_r": round(ev_hold_r, 2), "pnl_r": round(pnl_r, 2)}

    if ev_hold_r < 0.03:                             # لبهٔ آماری عملاً تمام شده — «معتبر» خواندنش صادقانه نیست
        reasons.insert(0, f"{model_txt} — EV نگه‌داشتن {ev_hold_r:+.2f}R و احتمال هدف {p_hit*100:.0f}٪؛ "
                          f"لبهٔ آماری صفر شده")
        return {"action": "hold", "title": "نگه‌دار — ولی لبه تمام شده؛ در جهشِ بعدی خروج را جدی بگیر",
                "urgency": 1, "reasons": reasons,
                "p_hit_tp": round(p_hit * 100, 1), "ev_hold_r": round(ev_hold_r, 2), "pnl_r": round(pnl_r, 2)}
    reasons.insert(0, f"{model_txt} — EV نگه‌داشتن {ev_hold_r:+.2f}R، احتمال هدف {p_hit*100:.0f}٪")
    return {"action": "hold", "title": "نگه‌دار — برنامه هنوز معتبر است",
            "urgency": 0, "reasons": reasons,
            "p_hit_tp": round(p_hit * 100, 1), "ev_hold_r": round(ev_hold_r, 2), "pnl_r": round(pnl_r, 2)}
