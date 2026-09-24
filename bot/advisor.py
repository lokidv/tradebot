# -*- coding: utf-8 -*-
"""🧭 مشاورِ پوزیشنِ باز — برای هر پوزیشنِ باز حساب می‌کند نگه‌داشتن هنوز می‌ارزد یا نه.

پایهٔ ریاضی: برای قیمتِ بینِ حدضرر و هدف، احتمالِ برخورد به هدف قبل از حدضرر در گشتِ تصادفی برابرِ
نسبتِ فاصله‌هاست: P = d_sl/(d_sl+d_tp) — «فاصلهٔ بیشتر تا حدضرر = شانسِ بیشترِ رسیدن به هدف».
این پایه فقط وقتی با جهتِ مدل (p_up) در فضای logit تیلت می‌شود که p_up از **مدلِ جهت‌یابِ کالیبره**
آمده باشد (``p_calibrated=True``)؛ امتیازِ اکتشافیِ forecast() کالیبره نیست و روی معامله‌های واقعیِ
قاعده احتمالِ هدف را ~۵ واحد باد می‌کرد. بی‌مدل، خروجی «فقط هندسه» است: احتمالِ هدف همان پایه و
ارزشِ انتظاریِ نگه‌داشتن (پیش از هزینه) صفر است — پس قاعده‌های مبتنی بر EV آن‌جا فعال نمی‌شوند.
سپس ارزشِ انتظاریِ نگه‌داشتن (به واحد R) به توصیهٔ شفاف با دلیل ترجمه می‌شود."""
import math
from datetime import datetime, timezone

import log


def _sig(z):
    return 1.0 / (1.0 + math.exp(-max(min(z, 30), -30)))


def _logit(p):
    p = max(min(p, 1 - 1e-6), 1e-6)
    return math.log(p / (1 - p))


def model_p_up(a):
    """p_upِ قابلِ اتکا برای تیلت، یا None.

    فقط وقتی فراخواننده صریحاً گفته p_up از مدلِ جهت‌یابِ کالیبره است (``p_calibrated=True`` یا
    ``p_up_kind="model"``). امتیازِ اکتشافی (``p_up_kind="heuristic"``) یا p_upِ بی‌برچسب رد می‌شود —
    بسته‌به‌پیش‌فرض، تا مسیرِ تازه‌ای دوباره عددِ کالیبره‌نشده را «مدلِ کالیبره» نخواند.
    """
    if not a:
        return None
    p = a.get("p_up")
    if p is None or not (a.get("p_calibrated") is True or a.get("p_up_kind") == "model"):
        return None
    return p


def advise(pos, a, btc_z=None):
    """pos: پوزیشنِ باز (paper). a: تحلیلِ زندهٔ همان نماد/تایم‌فریم (p_up و p_calibrated) یا None.
    خروجی: {action, title, urgency ۰-۳, reasons[], p_hit_tp, ev_hold_r, pnl_r, basis}
    basis: "model" (تیلت با مدلِ کالیبره) | "geometry" (فقط هندسهٔ فاصله‌ها) | "rule" (قاعدهٔ ثابت)."""
    d = 1 if pos["side"] == "long" else -1
    price = float(pos.get("last_price") or pos["entry"])
    if pos.get("strategy") == "tsmom28":
        r0 = float(pos.get("r0") or 1e-9)
        return {"action": "hold_trend", "title": "مومنتوم — تا تصمیمِ دوشنبهٔ بعد نگه دارید",
                "urgency": 0, "reasons": ["خروج فقط وقتی تصمیمِ دوشنبه «نقد» شود؛ حدضررِ −۵۰٪ فقط محافظِ فاجعه است"],
                "p_hit_tp": None, "ev_hold_r": None, "pnl_r": round(d * (price - float(pos["entry"])) / r0, 2),
                "basis": "rule"}
    if pos.get("tp") is None:
        # پوزیشنِ روند: هدف ندارد، پس «احتمالِ رسیدن به هدف» بی‌معناست؛ قاعده خودش خروج را می‌گوید
        r0 = float(pos.get("r0") or abs(float(pos["entry"]) - float(pos["sl"]))) or 1e-9
        return {"action": "hold_trend", "title": "روند — تا خوردنِ حدضررِ دنباله‌دار نگه دارید",
                "urgency": 0, "reasons": ["قاعدهٔ روند هدفِ ثابت ندارد؛ حدضرر هر روز فقط رو به بالا می‌رود"],
                "p_hit_tp": None, "ev_hold_r": None,
                "pnl_r": round(d * (price - float(pos["entry"])) / r0, 2), "basis": "rule"}
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
                "p_hit_tp": None, "ev_hold_r": None, "pnl_r": round(pnl_r, 2),
                "basis": "rule"}
    if d_sl <= 0:
        return {"action": "close_now", "title": "ببند — زیرِ حدضرر معلق است",
                "urgency": 3, "reasons": ["قیمت از حدضرر گذشته و پوزیشن هنوز باز است"],
                "p_hit_tp": None, "ev_hold_r": None, "pnl_r": round(pnl_r, 2),
                "basis": "rule"}

    # احتمالِ برخورد به هدف قبل از حدضرر: پایهٔ گشتِ تصادفی + تیلتِ مدل (فقط مدلِ کالیبره)
    base_p = d_sl / (d_sl + d_tp)
    p_up = model_p_up(a)
    has_model = p_up is not None
    if has_model:
        tilt = d * (p_up - 50.0) / 50.0              # مثبت = مدل هم‌جهتِ من
        p_hit = _sig(_logit(base_p) + 1.2 * tilt)
        model_txt = f"مدلِ کالیبره: احتمال حرکت هم‌جهت {p_up if d==1 else round(100-p_up,1)}٪"
        ev_hold_r = p_hit * (d_tp / r0) - (1 - p_hit) * (d_sl / r0)
    else:
        # فقط هندسه: گشتِ تصادفی بازیِ منصفانه است، پس EV نگه‌داشتن (پیش از هزینه) دقیقاً صفر است
        p_hit = base_p
        model_txt = "فقط هندسهٔ فاصله‌ها — مدلِ جهتِ کالیبره در دسترس نیست"
        ev_hold_r = 0.0
    basis = "model" if has_model else "geometry"

    def res(action, title, urgency):
        return {"action": action, "title": title, "urgency": urgency, "reasons": reasons,
                "p_hit_tp": round(p_hit * 100, 1), "ev_hold_r": round(ev_hold_r, 2),
                "pnl_r": round(pnl_r, 2), "basis": basis}

    # سنِ پوزیشن نسبت به حدِ زمانی
    age_frac = 0.0
    try:
        opened = datetime.fromisoformat(pos["opened_at"])
        age_min = (datetime.now(timezone.utc) - opened).total_seconds() / 60
        age_frac = age_min / max(float(pos.get("time_stop_min") or 1), 1)
    except Exception:  # noqa: BLE001
        log.exc()

    btc_against = btc_z is not None and (btc_z * d < -0.5)
    if btc_against:
        reasons.append(f"بیت‌کوین خلافِ جهتِ پوزیشن چرخیده (z={btc_z:+.1f}) — ریسکِ سیستمیک")

    flip_hard = has_model and ((d == 1 and p_up <= 40) or (d == -1 and p_up >= 60))
    flip_soft = has_model and ((d == 1 and p_up <= 46) or (d == -1 and p_up >= 54))

    # ── قواعدِ تصمیم (به ترتیبِ اولویت) ──
    # قاعده‌های مبتنی بر EV فقط با مدلِ کالیبره: در حالتِ «فقط هندسه» EV همیشه صفر است و
    # «EV < ۰٫۱۵ ⇒ قفلِ سود» یا «EV < ۰٫۰۳ ⇒ لبه تمام شده» فقط مصنوعِ ساختِ فرمول می‌شد، نه اطلاع.
    if flip_hard:
        reasons.insert(0, f"{model_txt} — جهتِ مدل کاملاً برگشته")
        reasons.append(f"ارزشِ انتظاریِ نگه‌داشتن: {ev_hold_r:+.2f}R")
        title = "ببند — احتمالِ برگشت پایین است" if pnl_r < 0 else "ببند — سود را نگه دار، مدل برگشته"
        return res("close_now", title, 3)

    if has_model and ev_hold_r < -0.10:
        reasons.insert(0, f"ارزشِ انتظاریِ نگه‌داشتن منفی شده ({ev_hold_r:+.2f}R) — {model_txt}")
        reasons.append(f"احتمال رسیدن به هدف قبل از حدضرر: {p_hit*100:.0f}٪")
        return res("close_now",
                   "ببند — از این نقطه، ماندن به ضررِ توست" if pnl_r < 0 else "ببند — ادامه نمی‌ارزد، سود را بردار",
                   3 if btc_against else 2)

    if has_model and pnl_r >= 0.7 and (flip_soft or ev_hold_r < 0.15):
        reasons.insert(0, f"سودِ {pnl_r:.1f}R ساخته‌ای ولی ادامهٔ حرکت کم‌رمق است ({model_txt}، EV نگه‌داشتن {ev_hold_r:+.2f}R)")
        return res("lock_profit", "سود را قفل کن — ببند یا حدضرر را بالای سربه‌سر بیاور", 2)

    if pnl_r <= -0.5 and flip_soft:
        reasons.insert(0, f"در ضررِ {pnl_r:.1f}R هستی و مدل هم دیگر هم‌جهتت نیست ({model_txt})")
        reasons.append(f"احتمال رسیدن به هدف: فقط {p_hit*100:.0f}٪ — امیدِ برگشت آماری ضعیف است")
        return res("close_now", "ضرر را ببند — برنمی‌گردد", 2)

    if pnl_r >= 1.0 and d * (entry - sl) > 0:        # حدضرر هنوز زیرِ ورود است
        reasons.insert(0, f"سود {pnl_r:.1f}R — دیگر نباید این معامله به ضرر تبدیل شود")
        return res("move_be", "حدضرر را به سربه‌سر (یا بالاتر) منتقل کن", 1)

    if age_frac >= 0.7 and abs(pnl_r) < 0.3 and (p_up is None or 45 <= p_up <= 55):
        reasons.insert(0, f"{age_frac*100:.0f}٪ از مهلتِ معامله گذشته و قیمت جایی نرفته — سرمایه را آزاد کن")
        return res("time_exit", "خروجِ زمانیِ زودهنگام را در نظر بگیر", 1)

    if not has_model:
        reasons.insert(0, f"{model_txt}: احتمالِ رسیدن به هدف قبل از حدضرر {p_hit*100:.0f}٪ (گشتِ تصادفی) "
                          f"و EV نگه‌داشتن صفر (پیش از هزینه) — لبه‌ای اندازه‌گیری نمی‌شود")
        return res("hold", "طبقِ برنامه — فقط هندسه؛ حدضرر/هدف/مهلت خروج را می‌گویند", 0)

    if ev_hold_r < 0.03:                             # لبهٔ آماری عملاً تمام شده — «معتبر» خواندنش صادقانه نیست
        reasons.insert(0, f"{model_txt} — EV نگه‌داشتن {ev_hold_r:+.2f}R و احتمال هدف {p_hit*100:.0f}٪؛ "
                          f"لبهٔ آماری صفر شده")
        return res("hold", "نگه‌دار — ولی لبه تمام شده؛ در جهشِ بعدی خروج را جدی بگیر", 1)
    reasons.insert(0, f"{model_txt} — EV نگه‌داشتن {ev_hold_r:+.2f}R، احتمال هدف {p_hit*100:.0f}٪")
    return res("hold", "نگه‌دار — برنامه هنوز معتبر است", 0)
