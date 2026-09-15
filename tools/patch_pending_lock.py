# -*- coding: utf-8 -*-
"""پچِ هم‌زمانی — یک قفل برای سفارش‌های صبور؛ حذفِ شاخهٔ مردهٔ هزینه.

دو نخ ``_pending`` را دستکاری می‌کردند: ``_monitor`` هر ۱۲ ثانیه سفارش را پر
می‌کرد و ``_cycle`` هر ۴۵ ثانیه سفارش می‌گذاشت یا همه را لغو می‌کرد. اگر لغو بینِ
«خواندنِ سفارش» و «بازکردنِ پوزیشن» در نخِ دیگر رخ می‌داد، سفارشِ لغوشده باز هم پر
می‌شد. حالا هر دو زیرِ یک قفلِ بازگشت‌پذیرند.
"""
import io
import os

P = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "bot", "autotrader.py")
s = io.open(P, encoding="utf-8").read()


def sub1(old, new, label):
    global s
    n = s.count(old)
    assert n == 1, f"{label}: found {n}"
    s = s.replace(old, new)


sub1('''_pending = {}                      # symbol -> سفارشِ صبور (لیمیتِ بازگشتی — تعقیبِ قیمت ممنوع)''',
     '''_pending = {}                      # symbol -> سفارشِ صبور (لیمیتِ بازگشتی — تعقیبِ قیمت ممنوع)
_pending_lock = threading.RLock()  # _monitor پر می‌کند و _cycle می‌گذارد/لغو می‌کند — یک قفل برای هر دو''',
     "lock decl")

sub1('''def _clear_pending(why):
    """لغوِ همهٔ سفارش‌های صبور — با ثبتِ تک‌تک، تا replay آن‌ها را زنده نکند."""
    for sym, o in list(_pending.items()):
        _pending.pop(sym, None)
        fill_quality.log_event("cancelled", symbol=sym, tf=(o or {}).get("tf"),
                               side=(o or {}).get("side"), why=why,
                               authority=(o or {}).get("authority"))''',
     '''def _clear_pending(why):
    """لغوِ همهٔ سفارش‌های صبور — با ثبتِ تک‌تک، تا replay آن‌ها را زنده نکند."""
    with _pending_lock:
        for sym, o in list(_pending.items()):
            _pending.pop(sym, None)
            fill_quality.log_event("cancelled", symbol=sym, tf=(o or {}).get("tf"),
                                   side=(o or {}).get("side"), why=why,
                                   authority=(o or {}).get("authority"))''',
     "clear pending locked")

sub1('''def _process_pending():
    """🎯 سفارش‌های صبور: قیمت به لیمیت رسید → باز کن؛ مهلت گذشت → با دیسیپلین رد شو."""
    now = time.time()''',
     '''def _process_pending():
    """🎯 سفارش‌های صبور: قیمت به لیمیت رسید → باز کن؛ مهلت گذشت → با دیسیپلین رد شو.

    کلِ گذر زیرِ ``_pending_lock`` است تا لغوِ هم‌زمان از نخِ دیگر نتواند بینِ
    خواندنِ سفارش و بازکردنِ پوزیشن جا بیفتد.
    """
    with _pending_lock:
        return _process_pending_locked()


def _process_pending_locked():
    now = time.time()''',
     "process pending wrapper")

sub1('''        _pending[sym] = {"side": side, "target": target, "tf": tf,
                         "r_abs": abs(entry0 - sl0), "tp_abs": abs(tp0 - entry0),
                         "size": round(size, 1), "tstop": tstop, "grade": c.get("grade") or "",
                         "score": c.get("signal_score"), "cost": c.get("cost") or 0.15,
                         "signal_ts": c.get("zt"), "expires": time.time() + PENDING_TTL.get(tf, 1200),
                         "meta_size_mult": c.get("meta_size_mult"),
                         "authority": c.get("authority")}
        journal.append(journal.ORDER_PLACED, symbol=sym, order=_pending[sym])''',
     '''        order = {"side": side, "target": target, "tf": tf,
                 "r_abs": abs(entry0 - sl0), "tp_abs": abs(tp0 - entry0),
                 "size": round(size, 1), "tstop": tstop, "grade": c.get("grade") or "",
                 "score": c.get("signal_score"),
                 "cost": c["cost"] if c.get("cost") is not None else 0.15,
                 "signal_ts": c.get("zt"), "expires": time.time() + PENDING_TTL.get(tf, 1200),
                 "meta_size_mult": c.get("meta_size_mult"),
                 "authority": c.get("authority")}
        with _pending_lock:
            _pending[sym] = order
            journal.append(journal.ORDER_PLACED, symbol=sym, order=order)''',
     "placement locked")

# شاخهٔ مرده: شرطِ پرشدن همیشه slip_r ≤ 0 می‌دهد، پس این هرگز اجرا نمی‌شد
sub1('''            # هزینهٔ واقع‌گرایانه: اگر پر شدن بدتر از هدف بود، کمی هزینه اضافه
            cost = float(o.get("cost") or 0.15)
            if slip_r > 0.05:
                cost = min(cost + 0.03, 0.28)''',
     '''            # لغزشِ واقعی را شبیه‌ساز (paper.refresh با ویک و لغزشِ رده‌ای) حساب می‌کند؛
            # شاخهٔ قبلیِ «slip_r > 0.05 ⇒ +۰٫۰۳» هرگز اجرا نمی‌شد چون شرطِ پرشدن slip_r ≤ ۰ است.
            cost = float(o["cost"]) if o.get("cost") is not None else 0.15''',
     "dead slip branch")

io.open(P, "w", encoding="utf-8", newline="\n").write(s)
print("pending lock patch applied")
