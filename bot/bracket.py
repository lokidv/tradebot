# -*- coding: utf-8 -*-
"""قراردادِ واحدِ معامله — تنها تعریفِ «یک معامله» در کلِ سیستم.

پیش از این چهار پیاده‌سازیِ موازی از یک براکت وجود داشت (برچسبِ آموزش در
``calib.extract_events``، بک‌تستِ سریع در ``engine.quick_backtest``، پیشنهادِ زندهٔ
``engine.trade_suggestion`` و داوریِ ``shadow.resolve``) و با هم فرق داشتند: مدل روی
هدفِ ثابتِ 1.8R آموزش می‌دید ولی معاملهٔ زنده هدفش را از «پیش‌بینیِ مسیر» می‌گرفت و
بین 0.8R تا 3.5R کلمپ می‌شد، و حدضرر با پیوت سفت‌تر می‌شد. در نتیجه ``p_win`` و ``EV``
که به کاربر نشان داده می‌شد، توصیفِ معامله‌ای بود که هرگز اجرا نمی‌شد.

از این پس هر چهار مسیر از همین ماژول می‌خوانند. تغییرِ قاعده یعنی تغییرِ برچسب‌ها،
پس هر تغییری اینجا باید با بالا بردنِ ``calib.CALIB_VERSION`` همراه شود.

قواعدِ نتیجه‌گیری عمداً محافظه‌کارانه‌اند (از ``engine.quick_backtest`` آمده‌اند):
* اگر کندل با گپ پشتِ حدضرر باز شود، خروج روی همان **قیمتِ بازِ مشاهده‌شده** است، نه حدضرر.
* اگر حدضرر و هدف در یک کندل هر دو لمس شوند، نتیجه **باخت** فرض می‌شود.
* اگر تا سقفِ کندل‌ها هیچ‌کدام نخورد، خروج روی **بستهٔ** آخرین کندلِ مجاز است.
"""
import math

EPS = 1e-10

R_ATR_MULT = 1.3        # فاصلهٔ حدضرر برحسبِ ATR(14)
R_MIN_PCT = 0.0025      # کفِ فاصلهٔ حدضرر نسبت به قیمتِ ورود (۰٫۲۵٪)
TP_R = 1.8              # هدف برحسبِ R — ثابت؛ سربه‌سرِ این ساختار ۳۵٫۷٪ است
MAX_BARS = 40           # حدِ زمانی برحسبِ کندل

OUTCOME_STOP = "stop"
OUTCOME_GAP_STOP = "gap_stop"
OUTCOME_TARGET = "target"
OUTCOME_TIMEOUT = "timeout"


def breakeven_win_rate(tp_r=TP_R):
    """نرخِ بردِ لازم برای سربه‌سر شدن (پیش از هزینه)."""
    return 1.0 / (1.0 + float(tp_r))


def risk_unit(entry, atr14):
    """فاصلهٔ ۱R — همان فرمولی که برچسب‌های آموزش با آن ساخته می‌شوند."""
    return max(R_ATR_MULT * float(atr14), R_MIN_PCT * abs(float(entry)))


def levels(entry, atr14, sig):
    """سطوحِ براکت برای ورودِ ``entry`` در جهتِ ``sig`` (۱ لانگ، −۱ شورت)."""
    sig = 1 if sig >= 0 else -1
    entry = float(entry)
    r = risk_unit(entry, atr14)
    return {
        "entry": entry,
        "r": r,
        "sl": entry - sig * r,
        "tp": entry + sig * TP_R * r,
        "risk_pct": r / max(abs(entry), EPS) * 100.0,
        "max_bars": MAX_BARS,
        "rr": TP_R,
    }


def resolve_path(o, h, l, c, start_idx, sig, entry, sl, tp, max_bars=MAX_BARS):
    """نتیجهٔ یک معامله را از کندلِ ``start_idx`` به بعد حساب می‌کند.

    ``start_idx`` اولین کندلی است که معامله در آن **باز** است (یعنی کندلِ بعد از سیگنال).
    خروجی: ``{outcome, exit_price, exit_idx, gross_r, bars_held}`` که ``gross_r`` پیش از
    هزینه است؛ کم‌کردنِ هزینه/فاندینگ وظیفهٔ فراخوان است.
    """
    sig = 1 if sig >= 0 else -1
    n = len(c)
    entry, sl, tp = float(entry), float(sl), float(tp)
    r = abs(entry - sl)
    if r <= EPS or not math.isfinite(r):
        raise ValueError("فاصلهٔ حدضرر صفر یا نامعتبر است — این معامله قابل داوری نیست")
    last = min(start_idx + max_bars - 1, n - 1)
    outcome, exit_price, exit_idx = None, None, last
    for j in range(start_idx, last + 1):
        # ۱) گپِ زیان‌بار: کندل پشتِ حدضرر باز شده — روی قیمتِ واقعیِ مشاهده‌شده خارج شو
        if (sig == 1 and o[j] <= sl) or (sig == -1 and o[j] >= sl):
            outcome, exit_price, exit_idx = OUTCOME_GAP_STOP, float(o[j]), j
            break
        hit_sl = l[j] <= sl if sig == 1 else h[j] >= sl
        hit_tp = h[j] >= tp if sig == 1 else l[j] <= tp
        if hit_sl:                       # محافظه‌کار: اگر هر دو در یک کندل، باخت
            outcome, exit_price, exit_idx = OUTCOME_STOP, sl, j
            break
        if hit_tp:
            outcome, exit_price, exit_idx = OUTCOME_TARGET, tp, j
            break
    if outcome is None:
        outcome, exit_price, exit_idx = OUTCOME_TIMEOUT, float(c[last]), last
    return {
        "outcome": outcome,
        "exit_price": float(exit_price),
        "exit_idx": int(exit_idx),
        "gross_r": float(sig * (exit_price - entry) / r),
        "bars_held": int(exit_idx - start_idx + 1),
        "timed_out": outcome == OUTCOME_TIMEOUT,
    }


def signal_trade(o, h, l, c, atr14_at_signal, i, sig, max_bars=MAX_BARS):
    """براکتِ کاملِ یک سیگنال روی کندلِ ``i``: ورود در openِ کندلِ بعد، سپس داوری.

    مسیرِ مشترکِ ``calib.extract_events`` و ``engine.quick_backtest`` — هر دو باید
    برای یک کندل دقیقاً یک عدد بدهند.
    """
    if i + 1 >= len(c):
        return None
    lv = levels(float(o[i + 1]), atr14_at_signal, sig)
    res = resolve_path(o, h, l, c, i + 1, sig, lv["entry"], lv["sl"], lv["tp"], max_bars)
    res.update(entry=lv["entry"], sl=lv["sl"], tp=lv["tp"],
               r=lv["r"], risk_pct=lv["risk_pct"])
    return res


def net_r(gross_r, risk_pct, cost_pct):
    """بازدهٔ خالص برحسبِ R پس از کسرِ هزینهٔ رفت‌وبرگشت."""
    return float(gross_r) - max(float(cost_pct), 0.0) / max(float(risk_pct), 0.05)
