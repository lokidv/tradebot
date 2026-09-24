# -*- coding: utf-8 -*-
"""مدلِ هزینهٔ واقعی — داخلِ برچسبِ آموزش، نه فقط در شبیه‌ساز.

آموزش با یک هزینهٔ ثابتِ ۰٫۱۵٪ برای همهٔ نمادها و همهٔ مدت‌ها ساخته می‌شد، در
حالی که:

* هزینهٔ رفت‌وبرگشت بسته به نقدشوندگی از ۰٫۰۸٪ (BTC) تا ۰٫۳۰٪ (آلتِ کم‌عمق)
  است. همین تفاوت برای ریسکِ ۰٫۸٪ یعنی ۰٫۱ تا ۰٫۴R — بزرگ‌تر از کلِ لبهٔ ادعایی.
* فاندینگِ پرپچوال اصلاً حساب نمی‌شد. با حدِ زمانیِ ۴۰ کندل، یک معاملهٔ روزانه
  تا ۴۰ روز باز می‌ماند و با ~۰٫۰۱٪ در هر تسویهٔ ۸ساعته این ~۱٫۲٪ است —
  هم‌اندازهٔ کلِ پاداشِ 1.8R. «لبهٔ +۰٫۰۷R روزانه» پس از این عملاً صفر می‌شود.

ردهٔ هزینه **در لحظهٔ رویداد** از حجمِ دلاریِ ۲۴ ساعتهٔ همان کندل‌ها ساخته
می‌شود (نه از حجمِ امروز) تا نگاه به آینده نداشته باشد. فاندینگ اگر تاریخچه‌اش
موجود باشد دقیقاً از تسویه‌های واقعیِ بینِ ورود و خروج جمع می‌شود؛ وگرنه یک
برآوردِ محافظه‌کار **همیشه به‌عنوانِ هزینه** اعمال می‌شود.
"""
import math

import numpy as np

import tf_spec

# ردهٔ هزینهٔ رفت‌وبرگشت (٪) بر اساسِ حجمِ دلاریِ ۲۴ ساعته — همان جدولِ main._symbol_cost
TIERS = ((5e8, 0.08), (1e8, 0.11), (2e7, 0.18), (0.0, 0.30))
FUNDING_INTERVAL_MS = 8 * 3600 * 1000
# وقتی تاریخچهٔ فاندینگ نیست: ۰٫۰۱٪ در هر تسویه، همیشه به ضررِ ما (نه به نفع)
DEFAULT_FUNDING_PER_SETTLEMENT_PCT = 0.01
BARS_PER_DAY = tf_spec.BARS_PER_DAY     # ثبتِ واحد: bot/tf_spec.py
TF_MS = tf_spec.BAR_MS


def tier_cost(quote_vol_24h):
    """هزینهٔ رفت‌وبرگشت (٪) برای یک حجمِ دلاریِ ۲۴ ساعته."""
    q = float(quote_vol_24h or 0.0)
    for floor, cost in TIERS:
        if q >= floor:
            return cost
    return TIERS[-1][1]


def quote_volume_24h(c, v, i, tf):
    """حجمِ دلاریِ ۲۴ ساعتهٔ منتهی به کندلِ ``i`` — فقط از گذشته (بدونِ نگاه به آینده).

    ⚠️ حجمِ اسپات است؛ حجمِ فیوچرز معمولاً بیشتر است، پس این برآورد محافظه‌کار است.
    """
    k = tf_spec.bars_per_day(tf)          # تایم‌فریمِ ناشناخته خطا می‌دهد، نه «۲۴ کندل» بی‌صدا
    lo = max(0, i - k + 1)
    return float(np.sum(np.asarray(c[lo:i + 1], float) * np.asarray(v[lo:i + 1], float)))


def point_in_time_tier(c, v, i, tf):
    return tier_cost(quote_volume_24h(c, v, i, tf))


def settlements_between(entry_ts, exit_ts):
    """تعدادِ تسویه‌های ۸ساعته‌ای که پوزیشن در طولشان باز بوده."""
    if exit_ts <= entry_ts:
        return 0
    first = math.floor(entry_ts / FUNDING_INTERVAL_MS) + 1
    last = math.floor(exit_ts / FUNDING_INTERVAL_MS)
    return max(0, last - first + 1)


def funding_cost_pct(side, entry_ts, exit_ts, funding_rows=None):
    """هزینهٔ فاندینگ (٪ نُشنال) برای یک معامله. مثبت = پرداخت، منفی = دریافت.

    اگر تاریخچهٔ واقعی بازهٔ معامله را پوشش دهد، دقیقاً همان جمع می‌شود (لانگ نرخِ
    مثبت را می‌پردازد). وگرنه ``DEFAULT`` برای هر تسویه **همیشه** به‌عنوانِ هزینه
    اعمال می‌شود — ندانستن نباید به سودِ فرضی تبدیل شود.
    """
    d = 1 if side in ("long", 1) else -1
    if funding_rows:
        covered_from = float(funding_rows[0][0])
        # تاریخچه معامله را پوشش می‌دهد اگر از **اولین تسویهٔ پس از ورود** شروع شده باشد؛
        # تسویه‌ای پیش از آن اصلاً به این پوزیشن مربوط نیست.
        first_settle = (math.floor(entry_ts / FUNDING_INTERVAL_MS) + 1) * FUNDING_INTERVAL_MS
        if covered_from <= first_settle:
            return round(sum(d * float(rate) * 100.0 for ts, rate in funding_rows
                             if entry_ts < float(ts) <= exit_ts), 6)
    return round(settlements_between(entry_ts, exit_ts) * DEFAULT_FUNDING_PER_SETTLEMENT_PCT, 6)


def event_cost_pct(side, c, v, i, tf, entry_ts, exit_ts, funding_rows=None):
    """هزینهٔ کاملِ یک رویدادِ آموزشی (٪ نُشنال): ردهٔ نقدشوندگی + فاندینگ."""
    return round(point_in_time_tier(c, v, i, tf)
                 + funding_cost_pct(side, entry_ts, exit_ts, funding_rows), 6)


def expected_funding_pct(tf, bars_held):
    """برآوردِ محافظه‌کارِ فاندینگ برای نمونه‌های جهت‌یاب که مسیرشان ثبت نشده."""
    hold_ms = float(bars_held) * tf_spec.bar_ms(tf)
    return round((hold_ms / FUNDING_INTERVAL_MS) * DEFAULT_FUNDING_PER_SETTLEMENT_PCT, 6)
