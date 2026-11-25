# -*- coding: utf-8 -*-
"""تاریخچهٔ هسته برای آموزش و اعتبارسنجیِ ماتریسِ تصمیم روی همهٔ تایم‌فریم‌ها (۵ دقیقه تا روزانه).

همه از آرشیوِ عمومیِ فیوچرزِ بایننس (data.binance.vision) که از ایران باز است؛ بازهٔ پیش‌فرض ۲۰۲۲-۰۱ تا ۲۰۲۶-۰۸.
خروجی‌ها در hist_research/micro/ (در گیت نیست). فایلِ موجود دوباره گرفته نمی‌شود (مگر با --force).

  um_<SYM>_<iv>.npz   کندلِ فیوچرز برای iv در 5m, 1h, 4h, 1d (15m از قبل با tools/fetch_micro.py هست)
                      کلیدها مثلِ فایلِ ۱۵دقیقه: t (ms، زمانِ باز شدن), o, h, l, c, v, qv, n (تعدادِ معامله),
                      tbv (حجمِ پایهٔ خریدِ تهاجمی؛ فروشِ تهاجمی = v - tbv)
  prem_<SYM>_5m.npz   t, prem (بستهٔ شاخصِ پرمیوم در ۵ دقیقه؛ هم‌شکلِ prem_<SYM>_15m.npz)
  fund_<SYM>.npz      t (ms، calc_time), rate (نرخِ فاندینگِ تسویه‌شده), interval_h (فاصلهٔ فاندینگ به ساعت)
  depth_<SYM>.npz     عمقِ دفترِ سفارش روی شبکهٔ ۵دقیقه (آرشیو از 2023-01-01):
                      t (ms، شروعِ بازهٔ ۵دقیقه، هم‌تراز با t کندلِ ۵m), ts_last (ms، آخرین عکسِ دفتر در بازه),
                      ns (تعدادِ عکس‌ها در بازه), bid1..bid5 و ask1..ask5 (ارزشِ دلاریِ تجمعیِ سفارش‌ها تا x٪
                      زیر/بالای قیمت، از آخرین عکسِ بازه), bid1_avg, ask1_avg, bid2_avg, ask2_avg, bid5_avg, ask5_avg
                      (میانگینِ همان بازه). علّی: مقدارِ ردیفِ t تا پایانِ کندلِ t معلوم است (ts_last < t + 5m).
  spot_<SYM>_<iv>.npz --market spot: کندلِ اسپاتِ بایننس (data/spot در همان آرشیو) با همان کلیدها
                      (t,o,h,l,c,v,qv,n,tbv). 5m و 15m از --start (۲۰۲۲-۰۱)، 1h/4h/1d از --slow-start (۲۰۱۹-۰۱) برای
                      گرم‌شدن و آموزشِ بلندترِ تایم‌فریم‌های کند. آرشیوِ اسپات از ۲۰۲۵ میکروثانیه است → ms.
                      فایلِ موجود اگر سالم باشد (کلیدها، یکنواخت، بی‌تکرار، روی شبکهٔ زمانی، ms، تا آخرین کندلِ بازه)
                      نگه داشته و فقط روزهای ناقصش از آرشیوِ روزانه پر می‌شود؛ وگرنه از نو ساخته می‌شود.
  kfund_<SYM>.npz     --only kfund: فاندینگِ تسویه‌شدهٔ کوکوین فیوچرز (api-futures.kucoin.com
                      /api/v1/contract/funding-rates، XBTUSDTM..TRXUSDTM) از --kfund-start تا اکنون:
                      t (ms، timepoint = لحظهٔ تسویه)، rate، interval_h (از فاصلهٔ تسویه‌ها با market.infer_interval_h،
                      همان تابعِ مسیرِ زنده). فایلِ موجود فقط با ردیف‌های تازه‌تر ادامه می‌یابد (مگر با --force).
  core_history_audit.json   گزارشِ پوشش: تعداد، اول/آخر، یکنواختی، تکراری، واحدِ ms، شکاف‌های بیش از ۱ ساعت،
                            کندل‌های تخت و بی‌حجم (flat_zero_volume_bars/days)، و مقایسهٔ ۵m بازنمونه‌شده با فایل‌های
                            15m/1h/4h/1d (um و spot جداگانه، با mismatch_days).

روزِ «کامل ولی خراب»: آرشیوِ ماهانه گاهی کندل‌های یخ‌زده (o=h=l=c، v=0) دارد که فایلِ روزانهٔ همان روز ندارد
(BTCUSDT 5m/15m ۲۰۲۳-۱۱-۱۰). ساختِ فایل و --fill-gaps (و اسپات همیشه) چنین روزی را از آرشیوِ روزانه دوباره می‌گیرند
و گزارش می‌دهند: تعمیرشده، یا «روزانه هم تخت» (توقفِ واقعی/ناسازگاریِ خودِ آرشیو که از این‌جا تعمیر نمی‌شود).

liquidationSnapshot برای um در آرشیو وجود ندارد (فقط cm). قطعه‌های ماهانهٔ عمق در micro/_parts/depth/ می‌مانند
تا اجرای قطع‌شده از همان‌جا ادامه دهد.

اجرا:
  python tools/fetch_core_history.py                       # همه + گزارش
  python tools/fetch_core_history.py --only klines funding  # فقط بخشی
  python tools/fetch_core_history.py --only audit           # فقط گزارشِ پوشش
  python tools/fetch_core_history.py --depth-start 2024-01-01
  python tools/fetch_core_history.py --market spot           # کندلِ اسپات 5m..1d + گزارش
  python tools/fetch_core_history.py --only kfund audit      # فاندینگِ کوکوین + گزارش
"""
import argparse
import csv
import datetime as dt
import io
import json
import os
import sys
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor

import httpx
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "bot"))
import market  # noqa: E402  — صفحه‌بندی و interval_hِ فاندینگِ کوکوین: یک مسیر با زنده
import paths  # noqa: E402
import watchlist  # noqa: E402

OUT = paths.data("hist_research", "micro")
PARTS = os.path.join(OUT, "_parts", "depth")
BASE = "https://data.binance.vision/data/futures/um"
KLINE = BASE + "/monthly/klines/{s}/{iv}/{s}-{iv}-{y}-{m:02d}.zip"
PREM = BASE + "/monthly/premiumIndexKlines/{s}/{iv}/{s}-{iv}-{y}-{m:02d}.zip"
FUND = BASE + "/monthly/fundingRate/{s}/{s}-fundingRate-{y}-{m:02d}.zip"
KLINE_DAY = BASE + "/daily/klines/{s}/{iv}/{s}-{iv}-{d}.zip"         # روزهایی که در فایلِ ماهانه نیستند
PREM_DAY = BASE + "/daily/premiumIndexKlines/{s}/{iv}/{s}-{iv}-{d}.zip"
DAY = 86_400_000
DEPTH = BASE + "/daily/bookDepth/{s}/{s}-bookDepth-{d}.zip"
SPOT = "https://data.binance.vision/data/spot"
SPOT_KLINE = SPOT + "/monthly/klines/{s}/{iv}/{s}-{iv}-{y}-{m:02d}.zip"
SPOT_KLINE_DAY = SPOT + "/daily/klines/{s}/{iv}/{s}-{iv}-{d}.zip"
SLOW_IVS = ("1h", "4h", "1d")          # اسپاتِ این‌ها از --slow-start
KLINE_KEYS = ("t", "o", "h", "l", "c", "v", "qv", "n", "tbv")
STEP_MS = {"5m": 300_000, "15m": 900_000, "1h": 3_600_000, "4h": 14_400_000, "1d": 86_400_000}
DEPTH_AVG = (1, 2, 5)                  # سطوحی که میانگینِ بازه هم دارند
HOUR = 3_600_000

STATS = {"requests": 0, "bytes": 0, "missing": [], "failed": [],
         "repaired": [], "daily_also_flat": [], "daily_missing_rows": [], "no_daily": [],
         "daily_worse": []}                                                    # روزهای کندلِ تخت و بی‌حجم


# ---------------------------------------------------------------- شبکه

def _client(workers):
    return httpx.Client(timeout=60, limits=httpx.Limits(max_connections=workers, max_keepalive_connections=workers),
                        headers={"User-Agent": "ctp-research/1.0"})


def get(cli, url, tries=5):
    """محتوای zip یا None (۴۰۴). خطای شبکه و 5xx/429 با فاصلهٔ فزاینده دوباره امتحان می‌شود."""
    for k in range(tries):
        try:
            r = cli.get(url)
            STATS["requests"] += 1
            if r.status_code == 200:
                STATS["bytes"] += len(r.content)
                return r.content
            if r.status_code == 404:
                STATS["missing"].append(url.rsplit("/", 1)[-1])
                return None
            if r.status_code not in (429, 500, 502, 503, 504):
                break
        except httpx.HTTPError:
            pass
        time.sleep(min(2 ** k, 20))
    STATS["failed"].append(url.rsplit("/", 1)[-1])
    return None


def _csv_rows(content):
    z = zipfile.ZipFile(io.BytesIO(content))
    return csv.reader(io.TextIOWrapper(z.open(z.namelist()[0]), encoding="utf-8"))


def _ms(t):
    """آرشیوِ اسپات از ۲۰۲۵ میکروثانیه است؛ هر دو را به میلی‌ثانیه برمی‌گرداند."""
    return t // 1000 if t > 10**14 else t


def months(start, end):
    y, m = start
    while (y, m) <= end:
        yield y, m
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)


def _save(path, **cols):
    tmp = path + ".tmp.npz"
    np.savez_compressed(tmp, **cols)
    os.replace(tmp, path)


def _dedup_sorted(a):
    """مرتب بر حسبِ ستونِ اول و حذفِ زمان‌های تکراری (آخرین مقدار می‌ماند)."""
    a = a[np.argsort(a[:, 0], kind="stable")]
    keep = np.concatenate([a[1:, 0] != a[:-1, 0], [True]])
    return a[keep]


def _span_days(span):
    (y0, m0), (y1, m1) = span
    d0, d1 = dt.date(y0, m0, 1), dt.date(y1 + (m1 == 12), m1 % 12 + 1, 1)
    return [d0 + dt.timedelta(i) for i in range((d1 - d0).days)]


def _flat_mask(a):
    """کندلِ «تخت و بی‌حجم» (o=h=l=c و v=0) در آرایهٔ ستون‌های ``KLINE_KEYS``: یا توقفِ واقعیِ صرافی است یا کندلِ یخ‌زدهٔ
    آرشیوِ ماهانه (ممیزی DATA-3: BTCUSDT 5m/15m در ۲۰۲۳-۱۱-۱۰ که فایلِ روزانه‌اش سالم است)."""
    a = np.asarray(a, float)
    if a.ndim != 2 or a.shape[1] < 6 or not len(a):
        return np.zeros(len(a), bool)
    o, h, l, c, v = (a[:, i] for i in range(1, 6))
    return (o == h) & (h == l) & (l == c) & (v == 0)


def _fill(cli, pool, a, day_url, parse, step, span, check_flat=False):
    """روزهای ناقص/غایب در آرشیوِ ماهانه از آرشیوِ روزانه پر می‌شوند (مثلاً SOL و TRX در ۲۰۲۲-۰۲-۲۶..۲۸).

    ``check_flat`` (فقط کندل): روزی که ردیف‌هایش کامل است ولی کندلِ تخت و بی‌حجم دارد هم از آرشیوِ روزانه دوباره
    گرفته می‌شود (ممیزی DATA-3: پیش‌تر فقط روزِ کم‌ردیف دوباره گرفته می‌شد، پس کندل‌های یخ‌زدهٔ فایلِ ماهانه
    هرگز جایگزین نمی‌شدند). ردیف‌های روزانه جای ردیف‌های هم‌زمانِ ماهانه را می‌گیرند، مگر فایلِ روزانه کندلِ تختِ
    بیشتری داشته باشد. خروجی ``(a, ردیف‌های تازه, گزارش)``؛ گزارش = فهرستِ روزها با «تعدادِ تخت ماهانه→پس از ادغام»
    (شمارش روی ردیف‌های ادغام‌شدهٔ همان روز: کندلِ تختِ ماهانه‌ای که فایلِ روزانه ردیفی برایش ندارد می‌ماند):
    ``repaired`` (کندلِ تخت کم شد)، ``daily_also_flat`` (روزانه هم همان کندل‌های تخت را دارد: توقفِ واقعی یا
    ناسازگاریِ خودِ آرشیو — از این‌جا قابلِ تعمیر نیست)، ``daily_missing_rows`` (روزانه ردیفِ آن زمان‌ها را ندارد؛
    کندل‌های تخت ماندند)، ``daily_worse`` (ماهانه نگه داشته شد) و ``no_daily`` (فایلِ روزانه نیست).
    """
    fix = {"repaired": [], "daily_also_flat": [], "daily_missing_rows": [], "daily_worse": [], "no_daily": []}
    have = dict(zip(*np.unique(a[:, 0].astype(np.int64) // DAY, return_counts=True))) if len(a) else {}
    epoch = dt.date(1970, 1, 1)
    first = int(a[0, 0]) // DAY if len(a) else -1       # پیش از فهرست‌شدنِ نماد (SOL اسپات ۲۰۲۰-۰۸) روزی نیست
    days = _span_days(span)
    miss = [d for d in days if (d - epoch).days >= first and have.get((d - epoch).days, 0) < DAY // step]
    flat, fm = {}, None
    if check_flat and len(a):
        fm = _flat_mask(a)
        in_span = {(d - epoch).days: d for d in days}
        k, cnt = np.unique(a[fm, 0].astype(np.int64) // DAY, return_counts=True)
        flat = {in_span[int(x)]: int(n) for x, n in zip(k, cnt) if int(x) in in_span and in_span[int(x)] not in miss}
    todo = miss + sorted(flat)
    if not todo:
        return a, 0, fix
    n0 = len(a)
    parts = list(pool.map(lambda d: parse(cli, day_url.format(d=d.isoformat())), todo))
    rows = []
    for d, part in zip(todo, parts):
        if d in flat:
            if not part:
                fix["no_daily"].append(d.isoformat())
                continue
            pa = np.asarray(part, float)
            on_day = (a[:, 0].astype(np.int64) // DAY) == (d - epoch).days
            left = len(set(a[on_day & fm, 0].astype(np.int64).tolist()) - set(pa[:, 0].astype(np.int64).tolist()))
            after = int(_flat_mask(pa).sum()) + left            # ردیفِ روزانه جای ماهانهٔ هم‌زمان را می‌گیرد، نه بیشتر
            key = ("repaired" if after < flat[d] else "daily_worse" if after > flat[d]
                   else "daily_missing_rows" if left else "daily_also_flat")
            fix[key].append(f"{d.isoformat()} ({flat[d]}->{after} flat bars)")
            if key == "daily_worse":
                continue
        rows.extend(part)
    if rows:
        a = _dedup_sorted(np.vstack([a, np.asarray(rows, float)]) if len(a) else np.asarray(rows, float))
    return a, len(a) - n0, fix                           # فقط ردیف‌های واقعاً تازه (روزِ قطعیِ صرافی = ۰)


def _fix_note(fix):
    """متنِ کوتاهِ گزارشِ روزهای تخت برای خروجیِ کنسول (خالی اگر چیزی نبود)."""
    out = [f"{k} {', '.join(v)}" for k, v in fix.items() if v]
    return ("; flat zero-volume days: " + "; ".join(out)) if out else ""


# ---------------------------------------------------------------- کندل، پرمیوم، فاندینگ

def _kline_month(cli, url):
    content = get(cli, url)
    if content is None:
        return []
    rows = []
    for row in _csv_rows(content):
        if not row or not row[0].isdigit():
            continue                                        # سرتیتر
        rows.append((_ms(int(row[0])), float(row[1]), float(row[2]), float(row[3]), float(row[4]),
                     float(row[5]), float(row[7]), float(row[8]), float(row[9])))
    return rows


def _series(cli, pool, path, keys, month_url, day_url, parse, step, span, force, fill):
    """ماهانه + پر کردنِ روزهای غایب از روزانه. فایلِ موجود فقط با fill (پر کردنِ شکاف) یا force دست می‌خورد.
    برای کندل روزهای دارای کندلِ تخت و بی‌حجم هم از روزانه دوباره گرفته می‌شوند (``_fill``) و در STATS گزارش می‌شوند."""
    check_flat = tuple(keys) == KLINE_KEYS
    name = os.path.basename(path)
    if os.path.exists(path) and not force:
        if not fill:
            return "cached"
        with np.load(path) as d:                       # ویندوز: فایلِ باز جایگزین نمی‌شود
            a = np.column_stack([d[k].astype(float) for k in keys])
        a, added, fix = _fill(cli, pool, a, day_url, parse, step, span, check_flat)
        _record(name, fix)
        if added or fix["repaired"]:
            _save(path, **{k: (a[:, i].astype(np.int64) if k == "t" else a[:, i]) for i, k in enumerate(keys)})
        return f"cached, gap-fill +{added} rows -> {len(a)}" + _fix_note(fix)
    urls = [month_url.format(y=y, m=m) for y, m in months(*span)]
    rows = [r for part in pool.map(lambda u: parse(cli, u), urls) for r in part]
    a = _dedup_sorted(np.asarray(rows, float)) if rows else np.zeros((0, len(keys)))
    a, added, fix = _fill(cli, pool, a, day_url, parse, step, span, check_flat)
    _record(name, fix)
    if not len(a):
        return "empty"
    _save(path, **{k: (a[:, i].astype(np.int64) if k == "t" else a[:, i]) for i, k in enumerate(keys)})
    return f"{len(a)} rows (+{added} from daily files)" + _fix_note(fix)


def _record(name, fix):
    for k, v in fix.items():
        STATS[k].extend(f"{name} {x}" for x in v)


def klines(cli, pool, sym, iv, span, force, fill=False):
    return _series(cli, pool, os.path.join(OUT, f"um_{sym}_{iv}.npz"), KLINE_KEYS,
                   KLINE.replace("{s}", sym).replace("{iv}", iv), KLINE_DAY.replace("{s}", sym).replace("{iv}", iv),
                   _kline_month, STEP_MS[iv], span, force, fill)


def _prem_month(cli, url):
    content = get(cli, url)
    if content is None:
        return []
    return [(_ms(int(r[0])), float(r[4])) for r in _csv_rows(content) if r and r[0].isdigit()]


def premium(cli, pool, sym, iv, span, force, fill=False):
    return _series(cli, pool, os.path.join(OUT, f"prem_{sym}_{iv}.npz"), ("t", "prem"),
                   PREM.replace("{s}", sym).replace("{iv}", iv), PREM_DAY.replace("{s}", sym).replace("{iv}", iv),
                   _prem_month, STEP_MS[iv], span, force, fill)


def _fund_month(cli, url):
    content = get(cli, url)
    if content is None:
        return []
    out = []
    for r in _csv_rows(content):
        if r and r[0].isdigit():                  # calc_time, funding_interval_hours, last_funding_rate
            out.append((_ms(int(r[0])), float(r[2]), float(r[1])) if len(r) >= 3 else (_ms(int(r[0])), float(r[-1]), np.nan))
    return out


def funding(cli, pool, sym, span, force):
    path = os.path.join(OUT, f"fund_{sym}.npz")
    if os.path.exists(path) and not force:
        return "cached"
    urls = [FUND.format(s=sym, y=y, m=m) for y, m in months(*span)]
    rows = [r for part in pool.map(lambda u: _fund_month(cli, u), urls) for r in part]
    if not rows:
        return "empty"
    a = _dedup_sorted(np.asarray(rows, float))
    _save(path, t=a[:, 0].astype(np.int64), rate=a[:, 1], interval_h=a[:, 2])
    return f"{len(a)} rows"


# ---------------------------------------------------------------- کندلِ اسپات

def _span_last_bar(span, step):
    (y1, m1) = span[1]
    return int(dt.datetime(y1 + (m1 == 12), m1 % 12 + 1, 1, tzinfo=dt.timezone.utc).timestamp() * 1000) - step


def spot_check(path, step, span):
    """فایلِ اسپاتِ موجود سالم است؟ (دلیل یا None). شکافِ درونی این‌جا خرابی نیست — از آرشیوِ روزانه پر می‌شود."""
    try:
        with np.load(path) as d:
            if not set(KLINE_KEYS) <= set(d.files):
                return f"keys {sorted(d.files)}"
            t = d["t"].astype(np.int64)
            finite = all(np.isfinite(d[k]).all() for k in KLINE_KEYS if k != "t")
    except Exception as e:  # noqa: BLE001 — فایلِ خراب/نیمه‌نوشته
        return f"unreadable {e}"
    if not len(t):
        return "empty"
    dtt = np.diff(t)
    if not (dtt > 0).all():
        return "not strictly increasing"
    if not ((t > 1.5e12) & (t < 2.1e12)).all():
        return "not ms"
    if (t % step).any():
        return "off grid"
    if not finite:
        return "NaN/inf values"
    if t[-1] < _span_last_bar(span, step):
        return f"ends {_utc(t[-1])}"
    return None


def spot_klines(cli, pool, sym, iv, span, force):
    path = os.path.join(OUT, f"spot_{sym}_{iv}.npz")
    why = None
    if os.path.exists(path) and not force:
        why = spot_check(path, STEP_MS[iv], span)
        if why:
            print(f"  {os.path.basename(path)} rebuilt: {why}", flush=True)
    urls = (SPOT_KLINE.replace("{s}", sym).replace("{iv}", iv), SPOT_KLINE_DAY.replace("{s}", sym).replace("{iv}", iv))
    return _series(cli, pool, path, KLINE_KEYS, urls[0], urls[1], _kline_month, STEP_MS[iv], span,
                   force or bool(why), fill=True)


# ---------------------------------------------------------------- فاندینگِ کوکوین

def _kucoin_get(cli):
    """getter برای market.kucoin_funding_pages با تلاشِ دوباره روی خطای شبکه/429/5xx."""
    def g(url, params):
        for k in range(6):
            try:
                r = cli.get(url, params=params)
                STATS["requests"] += 1
                if r.status_code == 200:
                    STATS["bytes"] += len(r.content)
                    return r.json()
                if r.status_code not in (429, 500, 502, 503, 504):
                    r.raise_for_status()
            except httpx.TransportError:
                pass
            time.sleep(min(2 ** k, 20))
        raise RuntimeError(f"kucoin {params} failed")
    return g


def kfund(cli, sym, start_ms, force):
    """فاندینگِ تسویه‌شدهٔ کوکوین از start_ms تا اکنون؛ فایلِ موجود فقط از آخرین ردیفش ادامه می‌یابد."""
    path = os.path.join(OUT, f"kfund_{sym}.npz")
    old = []
    if os.path.exists(path) and not force:
        with np.load(path) as d:
            old = list(zip(d["t"].astype(np.int64).tolist(), d["rate"].tolist()))
    contract = market.kucoin_contract(sym)
    now = int(time.time() * 1000)
    since = old[-1][0] + 1 if old else start_ms
    new = market.kucoin_funding_pages(contract, since, now, get=_kucoin_get(cli))
    rows = sorted(dict(old + new).items())
    if not rows:
        return "empty"
    cols = market.kfund_arrays(rows)                   # interval_h با همان تابعِ مسیرِ زنده
    _save(path, **cols)
    iv = {float(x): int(n) for x, n in zip(*np.unique(cols["interval_h"], return_counts=True))}
    return (f"{contract}: {len(rows)} rows (+{len(new)}), {_utc(rows[0][0])} .. {_utc(rows[-1][0])}, "
            f"interval_h {iv}")


# ---------------------------------------------------------------- عمقِ دفترِ سفارش

def _depth_day(cli, sym, day):
    """ماتریسِ (عکس × ۱۰) از ارزشِ تجمعی: ستون‌های ۰..۴ = bid1..bid5، ۵..۹ = ask1..ask5. سطوحِ کسری (مثلِ ±0.2) کنار می‌روند."""
    content = get(cli, DEPTH.format(s=sym, d=day.isoformat()))
    if content is None:
        return None
    z = zipfile.ZipFile(io.BytesIO(content))
    df = pd.read_csv(z.open(z.namelist()[0]), header=None, names=["ts", "pct", "depth", "notional"], dtype=str)
    df = df[df["ts"].str[:1].str.isdigit().fillna(False)]
    if df.empty:
        return None
    stamp = pd.to_datetime(df["ts"], format="%Y-%m-%d %H:%M:%S", utc=True)
    ts = ((stamp - pd.Timestamp(0, tz="UTC")) // pd.Timedelta(milliseconds=1)).to_numpy(np.int64)
    pct = pd.to_numeric(df["pct"], errors="coerce").to_numpy()
    val = pd.to_numeric(df["notional"], errors="coerce").to_numpy()
    lvl = np.rint(np.abs(pct))
    ok = np.isfinite(pct) & (np.abs(np.abs(pct) - lvl) < 1e-9) & (lvl >= 1) & (lvl <= 5)
    col = (lvl[ok] - 1).astype(int) + np.where(pct[ok] > 0, 5, 0)
    snap, inv = np.unique(ts[ok], return_inverse=True)
    mat = np.full((len(snap), 10), np.nan)
    mat[inv, col] = val[ok]
    return snap, mat


def _bucket(snap, mat):
    """عکس‌ها به بازهٔ ۵دقیقه: آخرین عکس + میانگینِ سطوحِ DEPTH_AVG + تعداد."""
    b = snap // 300_000 * 300_000
    t, first, ns = np.unique(b, return_index=True, return_counts=True)
    last = first + ns - 1
    out = {"t": t.astype(np.int64), "ts_last": snap[last].astype(np.int64), "ns": ns.astype(np.int32)}
    for i in range(5):
        out[f"bid{i + 1}"] = mat[last, i]
        out[f"ask{i + 1}"] = mat[last, 5 + i]
    for k in DEPTH_AVG:
        for side, c in (("bid", k - 1), ("ask", 4 + k)):
            v = mat[:, c]
            s = np.add.reduceat(np.where(np.isfinite(v), v, 0.0), first)
            n = np.add.reduceat(np.isfinite(v).astype(float), first)
            with np.errstate(invalid="ignore", divide="ignore"):
                out[f"{side}{k}_avg"] = np.where(n > 0, s / n, np.nan)
    return out


def _depth_month(cli, pool, sym, y, m, d0, d1):
    part = os.path.join(PARTS, sym, f"{y}-{m:02d}.npz")
    if os.path.exists(part):
        return part
    first = max(dt.date(y, m, 1), d0)
    nxt = dt.date(y + (m == 12), m % 12 + 1, 1)
    days = [first + dt.timedelta(i) for i in range((min(nxt - dt.timedelta(1), d1) - first).days + 1)]
    got = [r for r in pool.map(lambda d: _depth_day(cli, sym, d), days) if r is not None]
    if got:
        snap = np.concatenate([g[0] for g in got])
        mat = np.concatenate([g[1] for g in got])
        o = np.argsort(snap, kind="stable")
        snap, mat = snap[o], mat[o]
        keep = np.concatenate([snap[1:] != snap[:-1], [True]])
        cols = _bucket(snap[keep], mat[keep])
    else:
        cols = {"t": np.zeros(0, np.int64)}
    os.makedirs(os.path.dirname(part), exist_ok=True)
    _save(part, days_ok=np.int32(len(got)), days_total=np.int32(len(days)), **cols)
    return part


def depth(cli, pool, sym, d0, d1, force):
    path = os.path.join(OUT, f"depth_{sym}.npz")
    if os.path.exists(path) and not force:
        return "cached"
    parts = [np.load(_depth_month(cli, pool, sym, y, m, d0, d1)) for y, m in months((d0.year, d0.month), (d1.year, d1.month))]
    parts = [p for p in parts if len(p["t"])]
    if not parts:
        return "empty"
    keys = [k for k in parts[0].files if k not in ("days_ok", "days_total")]
    cols = {k: np.concatenate([p[k] for p in parts]) for k in keys}
    o = np.argsort(cols["t"], kind="stable")
    cols = {k: v[o] for k, v in cols.items()}
    ok = sum(int(p["days_ok"]) for p in parts)
    _save(path, **cols)
    return f"{len(cols['t'])} rows from {ok} days"


# ---------------------------------------------------------------- گزارشِ پوشش

def _utc(ms):
    return time.strftime("%Y-%m-%d %H:%M", time.gmtime(int(ms) / 1000))


def audit_file(path, step, gap_ms=HOUR):
    d = np.load(path)
    t = d["t"].astype(np.int64)
    dtt = np.diff(t)
    rep = {"file": os.path.basename(path), "rows": int(len(t)), "mb": round(os.path.getsize(path) / 1e6, 2),
           "first": _utc(t[0]), "last": _utc(t[-1]), "monotonic": bool((dtt > 0).all()),
           "duplicates": int((dtt == 0).sum()), "ms_units": bool(((t > 1.5e12) & (t < 2.1e12)).all()),
           "keys": list(d.files)}
    if step:
        rep["on_grid"] = bool((t % step == 0).all())
        rep["expected_rows"] = int((t[-1] - t[0]) // step + 1)
        rep["missing_rows"] = rep["expected_rows"] - rep["rows"]
    if "interval_h" in d.files:                          # فاندینگ: شکاف یعنی بیش از فاصلهٔ اعلام‌شده + ۱ ساعت
        iv = np.nan_to_num(d["interval_h"][:-1], nan=8.0) * HOUR
        big = np.where(dtt > iv + HOUR)[0]
        rep["interval_h_values"] = sorted({float(x) for x in np.unique(d["interval_h"][np.isfinite(d["interval_h"])])})
    else:
        big = np.where(dtt > max(gap_ms, step or 0))[0]
    key = "gaps_over_interval" if "interval_h" in d.files else "gaps_over_1h" if step < HOUR else "gaps_over_1bar"
    rep[key] = int(len(big))
    order = big[np.argsort(-dtt[big])][:6]
    rep["largest_gaps"] = [f"{_utc(t[i])} -> {_utc(t[i + 1])} ({dtt[i] / HOUR:.2f}h)" for i in order]
    nan = {k: round(float(np.isnan(d[k]).mean() * 100), 2) for k in d.files
           if k not in ("t", "ts_last", "ns") and d[k].dtype.kind == "f" and np.isnan(d[k]).any()}
    if nan:
        rep["nan_pct"] = nan
    if "ns" in d.files:
        rep["median_snapshots_per_bucket"] = float(np.median(d["ns"]))
    if set(KLINE_KEYS) <= set(d.files):                  # کندلِ تخت و بی‌حجم: توقفِ صرافی یا خرابیِ آرشیو (DATA-3)
        fm = _flat_mask(np.column_stack([d[k].astype(float) for k in KLINE_KEYS]))
        days = sorted({time.strftime("%Y-%m-%d", time.gmtime(int(x) // 1000)) for x in t[fm]})
        rep["flat_zero_volume_bars"] = int(fm.sum())
        rep["flat_zero_volume_days"] = days[:40]
    return rep


def resample(d, step):
    """۵ دقیقه → تایم‌فریمِ بزرگ‌تر (فقط سطل‌های کامل)."""
    t = d["t"].astype(np.int64)
    g = t // step * step
    u, first, n = np.unique(g, return_index=True, return_counts=True)
    full = n == step // STEP_MS["5m"]
    last = first + n - 1
    out = {"t": u, "o": d["o"][first], "c": d["c"][last],
           "h": np.maximum.reduceat(d["h"], first), "l": np.minimum.reduceat(d["l"], first)}
    for k in ("v", "qv", "n", "tbv"):
        out[k] = np.add.reduceat(d[k], first)
    return {k: v[full] for k, v in out.items()}


def crosscheck(sym, market_="um"):
    """۵ دقیقهٔ بازنمونه‌شده در برابرِ فایل‌های 15m/1h/4h/1d. ``mismatch_days`` = روزهای ناسازگارِ OHLC (برای یافتنِ
    روزِ خرابِ آرشیوِ ماهانه؛ تعمیرش با ``--fill-gaps`` که روزِ دارای کندلِ تخت و بی‌حجم را از فایلِ روزانه می‌گیرد)."""
    p5 = os.path.join(OUT, f"{market_}_{sym}_5m.npz")
    if not os.path.exists(p5):
        return {}
    d5 = np.load(p5)
    res = {}
    for iv in ("15m", "1h", "4h", "1d"):
        p = os.path.join(OUT, f"{market_}_{sym}_{iv}.npz")
        if not os.path.exists(p):
            continue
        r, d = resample(d5, STEP_MS[iv]), np.load(p)
        common, i, j = np.intersect1d(r["t"], d["t"], return_indices=True)
        rel = {k: float(np.nanmax(np.abs(r[k][i] - d[k][j]) / np.maximum(np.abs(d[k][j]), 1e-12)))
               for k in ("o", "h", "l", "c", "v", "tbv")} if len(common) else {}
        mism = np.zeros(len(common), bool)
        bad = 0
        for k in ("o", "h", "l", "c"):
            m = np.abs(r[k][i] - d[k][j]) > 1e-6 * np.maximum(np.abs(d[k][j]), 1)
            bad += int(m.sum())
            mism |= m
        days =sorted({time.strftime("%Y-%m-%d", time.gmtime(int(x) // 1000)) for x in common[mism]})
        res[iv] = {"common_bars": int(len(common)), "bars_only_in_file": int(len(d["t"]) - len(common)),
                   "ohlc_mismatches": bad, "mismatch_days": days[:40],
                   "max_rel_diff": {k: round(v, 9) for k, v in rel.items()}}
    return res


def audit(symbols):
    rep = {"generated_utc": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()), "files": {}, "resample_check": {},
           "resample_check_spot": {}}
    specs = [(f"{m}_{{s}}_{iv}.npz", STEP_MS[iv]) for m in ("um", "spot") for iv in ("5m", "15m", "1h", "4h", "1d")]
    specs += [("pos_{s}.npz", 300_000), ("prem_{s}_5m.npz", 300_000), ("prem_{s}_15m.npz", 900_000),
              ("fund_{s}.npz", None), ("kfund_{s}.npz", None), ("depth_{s}.npz", 300_000)]
    for s in symbols:
        for pat, step in specs:
            p = os.path.join(OUT, pat.format(s=s))
            if os.path.exists(p):
                rep["files"][os.path.basename(p)] = audit_file(p, step)
        rep["resample_check"][s] = crosscheck(s)
        rep["resample_check_spot"][s] = crosscheck(s, "spot")
    with open(os.path.join(OUT, "core_history_audit.json"), "w", encoding="utf-8") as f:
        json.dump(rep, f, indent=1)
    for name, r in rep["files"].items():
        g = next(r[k] for k in ("gaps_over_1h", "gaps_over_1bar", "gaps_over_interval") if k in r)
        print(f"{name:24s} rows={r['rows']:>8d} {r['first']} .. {r['last']} mb={r['mb']:>6} mono={r['monotonic']} "
              f"dup={r['duplicates']} ms={r['ms_units']} missing={r.get('missing_rows', '-')} gaps={g}", flush=True)
    return rep


# ---------------------------------------------------------------- اجرا

def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--market", default="um", choices=["um", "spot"],
                    help="um = فیوچرزِ USDT-M (پیش‌فرض)؛ spot = کندلِ اسپات با tbv (فقط klines)")
    ap.add_argument("--only", nargs="+", default=None,
                    choices=["klines", "premium", "funding", "kfund", "depth", "audit"],
                    help="پیش‌فرض: um → klines premium funding depth audit؛ spot → klines audit")
    ap.add_argument("--symbols", nargs="+", default=list(watchlist.SYMBOLS))
    ap.add_argument("--intervals", nargs="+", default=None, help="پیش‌فرض: um → 5m 1h 4h 1d؛ spot → 5m 15m 1h 4h 1d")
    ap.add_argument("--start", default="2022-01", help="اولین ماه (YYYY-MM)")
    ap.add_argument("--slow-start", default="2019-01", help="اولین ماهِ اسپاتِ 1h/4h/1d (گرم‌شدن و آموزشِ بلندتر)")
    ap.add_argument("--kfund-start", default="2021-10-01",
                    help="اولین روزِ فاندینگِ کوکوین (سه ماه پیش از ۲۰۲۲ برای گرم‌شدنِ z روی ۹۰ تسویه)")
    ap.add_argument("--end", default="2026-08", help="آخرین ماهِ کامل (YYYY-MM)")
    ap.add_argument("--depth-start", default="2023-01-01", help="آرشیوِ bookDepth از 2023-01-01 شروع می‌شود")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--force", action="store_true", help="فایلِ موجود را دوباره بساز")
    ap.add_argument("--fill-gaps", action="store_true",
                    help="روزهای غایبِ فایل‌های موجودِ 5m/1h/4h/1d و روزهای دارای کندلِ تخت و بی‌حجم را از آرشیوِ روزانه بگیر")
    a = ap.parse_args()
    spot = a.market == "spot"
    a.only = a.only or (["klines", "audit"] if spot else ["klines", "premium", "funding", "depth", "audit"])
    a.intervals = a.intervals or (["5m", "15m", "1h", "4h", "1d"] if spot else ["5m", "1h", "4h", "1d"])
    if spot and set(a.only) & {"premium", "funding", "depth"}:
        print("--market spot: premium/funding/depth فقط فیوچرز است — رد شد", flush=True)
    span = (tuple(map(int, a.start.split("-"))), tuple(map(int, a.end.split("-"))))
    slow = (tuple(map(int, a.slow_start.split("-"))), span[1])
    kfund_ms = int(dt.datetime.fromisoformat(a.kfund_start).replace(tzinfo=dt.timezone.utc).timestamp() * 1000)
    ey, em = span[1]
    d1 = dt.date(ey + (em == 12), em % 12 + 1, 1) - dt.timedelta(1)
    d0 = dt.date.fromisoformat(a.depth_start)
    os.makedirs(OUT, exist_ok=True)
    t0 = time.time()
    with _client(a.workers) as cli, ThreadPoolExecutor(a.workers) as pool:
        for s in a.symbols:
            if "klines" in a.only:
                for iv in a.intervals:
                    if spot:
                        print(s, "spot", iv, spot_klines(cli, pool, s, iv, slow if iv in SLOW_IVS else span, a.force),
                              f"{time.time() - t0:.0f}s", flush=True)
                    else:
                        print(s, iv, klines(cli, pool, s, iv, span, a.force, a.fill_gaps), flush=True)
            if spot:
                pass
            elif "premium" in a.only:
                print(s, "prem 5m", premium(cli, pool, s, "5m", span, a.force, a.fill_gaps), flush=True)
            if "funding" in a.only and not spot:
                print(s, "funding", funding(cli, pool, s, span, a.force), flush=True)
            if "kfund" in a.only:
                print(s, "kfund", kfund(cli, s, kfund_ms, a.force), f"{time.time() - t0:.0f}s", flush=True)
            if "depth" in a.only and not spot:
                print(s, "depth", depth(cli, pool, s, d0, d1, a.force), f"{time.time() - t0:.0f}s", flush=True)
    print(f"http: {STATS['requests']} ok-or-404 responses, {STATS['bytes'] / 1e6:.1f} MB, "
          f"404={len(STATS['missing'])} failed={len(STATS['failed'])} in {time.time() - t0:.0f}s", flush=True)
    if STATS["missing"]:
        print("404:", STATS["missing"][:40], flush=True)
    if STATS["failed"]:
        print("FAILED (rerun to retry):", STATS["failed"][:40], flush=True)
    for k, what in (("repaired", "repaired from the daily file (flat zero-volume bars in the monthly file)"),
                    ("daily_also_flat", "flat zero-volume bars also in the daily file (real halt or upstream; kept)"),
                    ("daily_missing_rows", "daily file has no rows at the flat zero-volume bars (kept)"),
                    ("daily_worse", "daily file had more flat bars (monthly kept)"),
                    ("no_daily", "flat zero-volume day without a daily file (kept)")):
        if STATS[k]:
            print(f"{what}:", STATS[k][:40], flush=True)
    if "audit" in a.only:
        audit(a.symbols)


if __name__ == "__main__":
    main()
