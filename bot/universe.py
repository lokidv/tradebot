# -*- coding: utf-8 -*-
"""جهانِ نقطه‌-در-زمان (point-in-time) — کدام ارزها **در آن تاریخ** برتر بودند.

آموزش روی ۱۰۰ ارزِ پرحجمِ **امروز** ساخته می‌شد، با تا ۸۰۰۰ کندل تاریخچه. این
دو خطا را با هم وارد می‌کرد:

* **سوگیریِ بقا** — ارزهایی که در این مدت فروپاشیدند یا حذف شدند اصلاً در
  برچسب‌ها نیستند؛ لانگ‌ها روی آلت‌ها خوش‌بینانه به‌نظر می‌رسند.
* **نگاه به آینده در انتخاب** — تاریخچهٔ پیش از رشدِ برنده‌های امروز وارد
  آموزش می‌شد؛ مدل در عمل می‌دانست کدام ارزها بعداً «برنده» می‌شوند.

اینجا برای هر ماه، ارزهای برتر بر اساسِ حجمِ دلاریِ ۳۰ روزِ **پیش از همان
ماه** انتخاب می‌شوند، و هر رویداد فقط اگر ارزش در جهانِ همان لحظه بوده وارد
آموزش می‌شود. رتبهٔ قدرتِ نسبی و پهنای بازار هم فقط روی همان جهان حساب می‌شوند.

⚠️ محدودیتِ صادقانه: ارزهایی که پیش از شروعِ کشِ داده حذف شده‌اند از APIِ
عمومی قابلِ بازیابی نیستند. پس این روش سوگیری را **کم** می‌کند، صفر نمی‌کند؛
برای همین فرضیه‌های پیش‌ثبت‌شده فقط روی ارزهای بزرگ و پایدار تعریف می‌شوند.
"""
import bisect
import time

import numpy as np

DAY_MS = 86_400_000
WINDOW_DAYS = 30
DEFAULT_TOP_N = 100
MIN_POPULATION = 40        # کمتر از این، رتبهٔ مقطعی بی‌معناست ⇒ مقدارِ خنثی


def _month_starts(t_min, t_max):
    """مهرِ زمانیِ ابتدای هر ماهِ میلادی (UTC) در بازهٔ داده."""
    out = []
    y, m = time.gmtime(t_min / 1000)[:2]
    while True:
        ts = int(time.mktime((y, m, 1, 0, 0, 0, 0, 0, 0)) - time.timezone) * 1000
        if ts > t_max:
            break
        if ts >= t_min:
            out.append(ts)
        m += 1
        if m > 12:
            y, m = y + 1, 1
    return out


def snapshots_from_histories(hists, top_n=DEFAULT_TOP_N, window_days=WINDOW_DAYS):
    """فهرستِ ``[(ts, frozenset(symbols)), ...]`` — برترین‌های هر ماه بر اساسِ گذشته.

    ``hists``: ``{symbol: {"t": [...], "c": [...], "v": [...]}}`` در هر تایم‌فریمی.
    """
    series = {}
    t_min, t_max = None, None
    for sym, kl in (hists or {}).items():
        if not kl or not kl.get("t"):
            continue
        t = np.asarray(kl["t"], dtype=np.int64)
        dv = np.asarray(kl["c"], float) * np.asarray(kl["v"], float)
        series[sym] = (t, np.concatenate([[0.0], np.cumsum(dv)]))
        t_min = int(t[0]) if t_min is None else min(t_min, int(t[0]))
        t_max = int(t[-1]) if t_max is None else max(t_max, int(t[-1]))
    if not series:
        return []
    window = window_days * DAY_MS
    snaps = []
    for ms in _month_starts(t_min, t_max):
        vols = []
        for sym, (t, cum) in series.items():
            hi = bisect.bisect_left(t, ms)          # فقط کندل‌های **پیش از** ابتدای ماه
            lo = bisect.bisect_left(t, ms - window)
            if hi - lo < 3:
                continue                            # هنوز لیست نشده یا داده ندارد
            vols.append((cum[hi] - cum[lo], sym))
        vols.sort(reverse=True)
        snaps.append((ms, frozenset(sym for _v, sym in vols[:top_n])))
    return snaps


def universe_at(snapshots, ts):
    """جهانِ معتبر در لحظهٔ ``ts`` — آخرین عکسی که پیش از آن گرفته شده.

    پیش از اولین عکس ``None`` برمی‌گردد، یعنی «نامعلوم» — فراخوان باید
    رویداد را رد کند، نه اینکه همه را مجاز بداند.
    """
    if not snapshots:
        return None
    keys = [s[0] for s in snapshots]
    i = bisect.bisect_right(keys, ts) - 1
    return snapshots[i][1] if i >= 0 else None


def in_universe(snapshots, sym, ts):
    uni = universe_at(snapshots, ts)
    return uni is not None and sym in uni


def rank_within(values, universe=None, min_population=MIN_POPULATION):
    """رتبهٔ صدکی (۰..۱) فقط بینِ اعضای جهان. جمعیتِ کم ⇒ همه خنثی (۰٫۵).

    رتبهٔ زنده قبلاً بدونِ حداقلِ جمعیت ساخته می‌شد؛ روی کشِ نیمه‌خالی پس از
    ری‌استارت، چند ارزِ موجود رتبهٔ ۰ یا ۱ می‌گرفتند — ورودیِ کاملاً جعلی.
    """
    items = [(k, v) for k, v in values.items() if universe is None or k in universe]
    if len(items) < min_population:
        return {k: 0.5 for k, _ in items}
    items.sort(key=lambda kv: kv[1])
    m = len(items) - 1
    return {k: i / m for i, (k, _v) in enumerate(items)}


def summary(snapshots):
    """گزارشِ کوتاه برای calib.json: چند ارز در طولِ زمان وارد/خارج شدند."""
    if not snapshots:
        return {"months": 0}
    ever = set().union(*[s for _t, s in snapshots])
    churn = [len(a[1] ^ b[1]) for a, b in zip(snapshots, snapshots[1:])]
    return {
        "months": len(snapshots),
        "first": snapshots[0][0],
        "last": snapshots[-1][0],
        "symbols_ever": len(ever),
        "size_last": len(snapshots[-1][1]),
        "avg_monthly_churn": round(sum(churn) / len(churn), 1) if churn else 0.0,
    }
