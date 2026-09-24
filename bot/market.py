# -*- coding: utf-8 -*-
"""لایه داده بازار — دریافت ۱۵ ارز برتر و کندل‌ها از Binance (با چند هاست جایگزین) و OKX به‌عنوان پشتیبان."""
import json
import os
import time
import threading
import httpx
import numpy as np

import paths

BINANCE_HOSTS = [
    "https://data-api.binance.vision",   # هاست عمومی داده — در اکثر مناطق باز است
    "https://api.binance.com",
    "https://api1.binance.com",
    "https://api2.binance.com",
]
OKX = "https://www.okx.com"

STABLE_BASES = {"USDC", "FDUSD", "TUSD", "BUSD", "DAI", "USDP", "PYUSD", "EUR", "AEUR", "USD1", "XUSD", "EURI"}
LEVERAGED_SUFFIX = ("UP", "DOWN", "BULL", "BEAR")
# فهرستِ نام‌ها همیشه عقب است: استیبل‌کوینِ تازهٔ «U» با قیمتِ 1.0002 در جدول
# «ستاپ A» نشان می‌داد، و USDG در جمعیتِ آموزش و breadth/RS بود. دارایی‌ای که
# بازهٔ ۲۴ساعته‌اش کمتر از این درصدِ قیمت است میخ‌شده است، نه بازار.
# سنجیده روی ۲۹۹۹ روز: آرام‌ترین روزِ BTC ۰٫۳۴٪ و TRX ۰٫۳۹٪؛ میانهٔ USDG ۰٫۰۲٪.
# ۰٫۱٪ با هر دو طرف فاصلهٔ چندبرابری دارد.
PEG_MAX_RANGE_PCT = 0.1
STALE_TICKER_MS = 6 * 3600 * 1000       # تیکری که ۶ ساعت معامله نداشته، بازارِ زنده نیست


def _is_pegged(last, high, low):
    try:
        return last > 0 and (high - low) / last * 100 < PEG_MAX_RANGE_PCT
    except (TypeError, ZeroDivisionError):
        return False

TF_BINANCE = {"5m": "5m", "15m": "15m", "1h": "1h", "4h": "4h", "1d": "1d"}
TF_OKX = {"5m": "5m", "15m": "15m", "1h": "1H", "4h": "4H", "1d": "1D"}
TF_MINUTES = {"5m": 5, "15m": 15, "1h": 60, "4h": 240, "1d": 1440}
KLINE_TTL = {"5m": 30, "15m": 60, "1h": 120, "4h": 300, "1d": 900}
# دورهٔ OI در rubikِ OKX (هر پنج تایم‌فریم را مستقیم دارد؛ سنجیده ۲۰۲۶-۰۹-۲۴)
TF_OKX_OI = {"5m": "5m", "15m": "15m", "1h": "1H", "4h": "4H", "1d": "1D"}

# کلیدهای کندل، هم‌شکلِ آرشیوِ spot_/um_<SYM>_<tf>.npz: qv حجمِ دلاری (فیلدِ ۷)، n تعدادِ معامله (۸)،
# tbv حجمِ پایهٔ خریدِ تهاجمی (۹). پشتیبانِ OKX/کوکوین این سه را ندارد → NaN، هرگز عددِ ساختگی مثلِ نصف.
KLINE_KEYS = ("t", "o", "h", "l", "c", "v", "qv", "n", "tbv")
NAN = float("nan")
SPOT_DATA = "https://data-api.binance.vision"      # تنها میزبانِ کندلِ بلند (get_klines_long)
KUCOIN_FUT = "https://api-futures.kucoin.com"

# مهلتِ اتصال کوتاه‌تر از مهلتِ خواندن: میزبانِ فیلترشده SYN را بی‌پاسخ می‌گذارد و هر
# تلاش ۱۲ ثانیه می‌سوخت؛ ۴ میزبانِ fapi پشتِ هم یعنی ~۵۰ ثانیه برای یک فاندینگ.
_client = httpx.Client(timeout=httpx.Timeout(12.0, connect=5.0), headers={"User-Agent": "ctp-bot/1.0"})
_lock = threading.Lock()
_kline_cache: dict = {}          # (symbol, tf) -> (ts, data)
_top_cache = {"ts": 0.0, "symbols": [], "tickers": {}}
TOP_STORE = 200                  # فهرستِ برترها همیشه با این اندازه گرفته می‌شود (۵۰/۱۰۰/۲۰۰ از یک کش)

# ── کول‌داونِ میزبان: میزبانِ قطع/مسدود (مهلت، ۴۵۱، ۴۰۳، ۵xx، محدودیتِ نرخ) مدتی امتحان نمی‌شود ──
# قبلاً هر تماس دوباره از اولِ فهرست شروع می‌کرد و هر بار همان مهلت‌ها را می‌سوزاند.
HOST_COOLDOWN = 60.0
_host_dead: dict = {}            # host -> زمانِ پایانِ کول‌داون
_host_lock = threading.Lock()

# ── تک‌پروازی: درخواست‌های هم‌زمانِ یک کلید فقط یک‌بار به شبکه می‌روند ──
_flight_locks: dict = {}
_flight_guard = threading.Lock()


def _flight_lock(key):
    with _flight_guard:
        lk = _flight_locks.get(key)
        if lk is None:
            lk = _flight_locks[key] = threading.Lock()
        return lk


def _host_of(url):
    return url.split("://", 1)[-1].split("/", 1)[0]


def _host_failure(e):
    """خطا مربوط به میزبان است (نه به خودِ درخواست)؟ ۴۰۰/۴۰۴ از همهٔ میزبان‌ها یکسان است."""
    if isinstance(e, httpx.HTTPStatusError):
        code = e.response.status_code
        return code in (403, 418, 429, 451) or code >= 500
    return True                                    # قطع، مهلت، پاسخِ غیر JSON (صفحهٔ فیلتر)


def host_status(now=None):
    """میزبان‌های در کول‌داون و ثانیه‌های باقی‌مانده — برای پایش."""
    now = time.time() if now is None else now
    with _host_lock:
        return {h: round(t - now, 1) for h, t in _host_dead.items() if t > now}


def _get_json(url, params=None):
    host = _host_of(url)
    with _host_lock:
        until = _host_dead.get(host, 0.0)
    if until > time.time():
        raise RuntimeError(f"{host} در کول‌داون است (خطای اخیر)")
    try:
        r = _client.get(url, params=params)
        r.raise_for_status()
        return r.json()
    except Exception as e:  # noqa: BLE001 — ثبتِ خرابیِ میزبان و بالا دادنِ همان خطا
        if _host_failure(e):
            with _host_lock:
                _host_dead[host] = time.time() + HOST_COOLDOWN
        raise


def _binance_json(path, params=None):
    last_err = None
    for host in BINANCE_HOSTS:
        try:
            return _get_json(host + path, params)
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 400:
                raise                              # درخواستِ نامعتبر (مثلاً نمادِ ناموجود) — میزبانِ بعدی هم همین را می‌گوید
            last_err = e
        except Exception as e:  # noqa: BLE001 — هاست بعدی را امتحان کن
            last_err = e
    raise last_err


def _top_fresh(n):
    """کش تازه است و دست‌کم n نماد دارد — یا بازار کلاً کمتر از اندازهٔ گرفته‌شده نماد داشت."""
    return (time.time() - _top_cache["ts"] < 600
            and max(len(_top_cache["symbols"]), _top_cache.get("n_req", 0)) >= n
            and bool(_top_cache["symbols"]))


def get_top_symbols(n=15):
    """۱۵ جفت USDT برتر بر اساس حجم ۲۴ساعته + دیتای تیکر (قیمت و تغییر ۲۴س).

    همیشه دست‌کم ``TOP_STORE`` ردیف ذخیره می‌شود: قبلاً درخواستِ ۵۰ → ۱۰۰ → ۲۰۰ در یک
    overview سه بار کلِ تیکرِ بایننس را می‌گرفت (هر اندازهٔ بزرگ‌تر کش را باطل می‌کرد).
    """
    with _lock:
        if _top_fresh(n):
            return _top_cache["symbols"][:n], _top_cache["tickers"]
    with _flight_lock(("top",)):
        with _lock:                                # نخِ دیگری همین حالا گرفته باشد
            if _top_fresh(n):
                return _top_cache["symbols"][:n], _top_cache["tickers"]
        symbols, tickers = _fetch_top(max(n, TOP_STORE))
        return symbols[:n], tickers


def _fetch_top(n):
    symbols, tickers = [], {}
    try:
        data = _binance_json("/api/v3/ticker/24hr")
        rows = []
        for t in data:
            s = t.get("symbol", "")
            if not s.endswith("USDT"):
                continue
            base = s[:-4]
            if base in STABLE_BASES or any(base.endswith(sfx) for sfx in LEVERAGED_SUFFIX):
                continue
            try:
                # جفتِ متوقف‌شده (status=BREAK) در ticker/24hr با حجمِ **کهنه** می‌ماند؛ TON و RNDR و POLY
                # با closeTimeِ ۳۲۴ ساعت پیش واردِ ۲۰۰ ارزِ برتر شده بودند و هر تایم‌فریم ۸ خطا می‌داد
                if time.time() * 1000 - float(t.get("closeTime") or 0) > STALE_TICKER_MS:
                    continue
                last = float(t["lastPrice"])
                if _is_pegged(last, float(t["highPrice"]), float(t["lowPrice"])):
                    continue
                rows.append((float(t["quoteVolume"]), s, last, float(t["priceChangePercent"])))
            except (KeyError, ValueError):
                continue
        rows.sort(reverse=True)
        for qv, s, px, chg in rows[:n]:
            symbols.append(s)
            tickers[s] = {"price": px, "chg24h": chg, "quote_vol": qv}
    except Exception:
        try:
            data = _get_json(OKX + "/api/v5/market/tickers", {"instType": "SPOT"})["data"]
            rows = []
            for t in data:
                inst = t.get("instId", "")
                if not inst.endswith("-USDT"):
                    continue
                base = inst[:-5]
                if base in STABLE_BASES:
                    continue
                try:
                    last = float(t["last"])
                    if _is_pegged(last, float(t["high24h"]), float(t["low24h"])):
                        continue
                    open24 = float(t["open24h"]) or last
                    rows.append((float(t["volCcy24h"]), base + "USDT", last, (last / open24 - 1) * 100))
                except (KeyError, ValueError, ZeroDivisionError):
                    continue
            rows.sort(reverse=True)
            for qv, s, px, chg in rows[:n]:
                symbols.append(s)
                tickers[s] = {"price": px, "chg24h": chg, "quote_vol": qv}
        except Exception:
            # هر دو منبع در دسترس نیستند (قطعیِ شبکه/DNS) — به‌جای crash، آخرین کشِ موجود را برگردان
            with _lock:
                if _top_cache["symbols"]:
                    return _top_cache["symbols"][:n], _top_cache["tickers"]
            raise
    with _lock:
        _top_cache.update(ts=time.time(), symbols=symbols, tickers=tickers, n_req=n)
    return symbols, tickers


def cache_stats(now=None):
    """سنِ کشِ کندل‌ها به تفکیکِ تایم‌فریم — برای پایشِ سلامت (بدونِ تماسِ شبکه).

    ``behind_candles_min``: تازه‌ترین دادهٔ کش چند کندل پشتِ آخرین کندلِ بسته‌شده است.
    «سن از زمانِ دریافت» معیارِ کهنگی نیست: ``get_klines`` عمداً تا بسته‌شدنِ کندلِ بعد
    دوباره نمی‌گیرد، پس کندلِ روزانه‌ای که ظهر گرفته شده تا نیمه‌شب کاملاً به‌روز است.
    """
    now = time.time() if now is None else now
    out = {}
    with _lock:
        items = list(_kline_cache.items())
        top_age = now - _top_cache["ts"] if _top_cache["ts"] else None
    for (sym, tf), (ts, data) in items:
        row = out.setdefault(tf, {"symbols": 0, "newest_age_sec": None, "oldest_age_sec": None,
                                  "behind_candles_min": None})
        row["symbols"] += 1
        age = now - ts
        row["newest_age_sec"] = age if row["newest_age_sec"] is None else min(row["newest_age_sec"], age)
        row["oldest_age_sec"] = age if row["oldest_age_sec"] is None else max(row["oldest_age_sec"], age)
        t = (data or {}).get("t") or []
        if t:
            tf_ms = TF_MINUTES[tf] * 60000
            boundary = int(now * 1000) // tf_ms * tf_ms      # بسته‌شدنِ آخرین کندل = openِ کندلِ جاری
            behind = max(0, (boundary - (int(t[-1]) + tf_ms)) // tf_ms)
            prev = row["behind_candles_min"]
            row["behind_candles_min"] = behind if prev is None else min(prev, behind)
    for row in out.values():
        row["newest_age_sec"] = round(row["newest_age_sec"], 1)
        row["oldest_age_sec"] = round(row["oldest_age_sec"], 1)
    return {"klines_by_tf": out, "universe_age_sec": round(top_age, 1) if top_age else None,
            "universe_size": len(_top_cache["symbols"]),
            "hosts_cooling": host_status(now)}      # میزبان‌های قطع/مسدود که فعلاً دور زده می‌شوند


def current_bar(symbol, tf):
    """کندلِ **جاریِ** (هنوز باز) اسپاتِ بایننس: ``{t, o, h, l, c}`` — ``c`` آخرین قیمت است.
    ``get_klines`` کندلِ باز را عمداً حذف می‌کند؛ سیگنال‌های روزانه برای «قیمتِ همین لحظه» و
    «کفِ امروز تا این لحظه» به همین نیاز دارند."""
    raw = _binance_json("/api/v3/klines", {"symbol": symbol, "interval": TF_BINANCE[tf], "limit": 1})
    r = raw[-1]
    return {"t": int(r[0]), "o": float(r[1]), "h": float(r[2]), "l": float(r[3]), "c": float(r[4])}


def current_bar_open(symbol, tf):
    """قیمتِ بازِ کندلِ **جاری** — مرجعِ ورودِ قاعده‌های روزانه."""
    b = current_bar(symbol, tf)
    return b["t"], b["o"]


def spot_price(symbol):
    return float(_binance_json("/api/v3/ticker/price", {"symbol": symbol})["price"])


def get_klines_cached(symbol, tf):
    """فقط از کش حافظه — بدون هیچ فراخوانی شبکه (برای محاسبات جمعی مثل رتبه قدرت نسبی)."""
    with _lock:
        hit = _kline_cache.get((symbol, tf))
        return hit[1] if hit else None


def get_klines(symbol, tf, limit=420):
    """کندل‌های بسته‌شده. کش تا پایان کندل جاری معتبر است (تحلیل فقط با کندل جدید عوض می‌شود)."""
    key = (symbol, tf)
    now = time.time()
    boundary = now - (now % (TF_MINUTES[tf] * 60))
    with _lock:
        hit = _kline_cache.get(key)
        if hit and (hit[0] >= boundary or now - hit[0] < 45):
            return hit[1]
    with _flight_lock(("kl", symbol, tf)):         # هم‌زمان‌ها منتظرِ همین یک دریافت می‌مانند
        with _lock:
            hit = _kline_cache.get(key)
            if hit and (hit[0] >= boundary or now - hit[0] < 45):
                return hit[1]
        return _fetch_klines(symbol, tf, limit)


def _binance_row(r):
    """ردیفِ کندلِ بایننس → (t, o, h, l, c, v, qv, n, tbv). ردیفِ کوتاه (بی‌فیلدِ ۷..۹) NaN می‌گیرد."""
    ext = (float(r[7]), float(r[8]), float(r[9])) if len(r) > 9 else (NAN, NAN, NAN)
    return (int(r[0]), float(r[1]), float(r[2]), float(r[3]), float(r[4]), float(r[5])) + ext


def _fetch_klines(symbol, tf, limit):
    key = (symbol, tf)
    now = time.time()                              # پس از انتظار برای قفل — مرزِ کندلِ باز دقیق بماند
    data = None
    src = "binance"
    try:
        raw = _binance_json("/api/v3/klines", {"symbol": symbol, "interval": TF_BINANCE[tf], "limit": limit + 1})
        rows = [_binance_row(r) for r in raw]
        if rows and rows[-1][0] + TF_MINUTES[tf] * 60000 > now * 1000:
            rows = rows[:-1]                       # حذف کندل باز
        data = rows[-limit:]
    except Exception:
        src = "okx"                                # بی‌جریانِ خرید/فروش: qv و n و tbv = NaN
        inst = symbol[:-4] + "-USDT"
        raw = _get_json(OKX + "/api/v5/market/candles", {"instId": inst, "bar": TF_OKX[tf], "limit": str(min(limit + 1, 300))})["data"]
        rows = []
        for r in reversed(raw):                    # OKX جدیدترین اول است
            if len(r) >= 9 and r[8] == "0":
                continue                           # کندل تأییدنشده
            rows.append((int(r[0]), float(r[1]), float(r[2]), float(r[3]), float(r[4]), float(r[5]), NAN, NAN, NAN))
        data = rows[-limit:]
    if not data or len(data) < 220:
        raise RuntimeError(f"داده کافی برای {symbol} {tf} دریافت نشد ({len(data or [])} کندل)")
    out = {k: [r[i] for r in data] for i, k in enumerate(KLINE_KEYS)}
    out["src"] = src
    with _lock:
        _kline_cache[key] = (now, out)
    return out


# ── کندلِ بلندِ اسپات برای گرم‌شدنِ پنجره‌های هستهٔ v2 (z تا ۲۰۱۶ کندل در ۵m) ──
LONG_PAGE = 1000                 # سقفِ هر درخواستِ /api/v3/klines
_long_cache: dict = {}           # (symbol, tf) -> (ts شروعِ دریافت, data)


def _long_fresh(hit, tf, bars, now):
    """کش تا بسته‌شدنِ کندلِ بعد معتبر است، ولی نه بیش از KLINE_TTL[tf] ثانیه."""
    if not hit or len(hit[1]["t"]) < bars:
        return False
    step = TF_MINUTES[tf] * 60
    return hit[0] >= now - (now % step) and now - hit[0] < KLINE_TTL[tf]


def _long_arrays(rows, src):
    a = {k: np.array([r[i] for r in rows], dtype=np.int64 if k == "t" else float) for i, k in enumerate(KLINE_KEYS)}
    a["src"] = src
    return a


def _spot_page(symbol, tf, limit, end=None):
    params = {"symbol": symbol, "interval": TF_BINANCE[tf], "limit": limit}
    if end is not None:
        params["endTime"] = end
    return [_binance_row(r) for r in _get_json(SPOT_DATA + "/api/v3/klines", params)]


def get_klines_long(symbol, tf, bars):
    """دست‌کم ``bars`` کندلِ **بسته‌شدهٔ** اسپات از data-api.binance.vision (صفحه‌بندیِ ۱۰۰۰تایی رو به عقب).

    خروجی آرایهٔ numpy با کلیدهای آرشیوِ spot_<SYM>_<tf>.npz (t به ms و int64؛ o..tbv اعشاری) و ``src``.
    کندلِ در حالِ شکل‌گیری حذف می‌شود. بی‌پشتیبان: OKX/کوکوین جریانِ خرید/فروش ندارند و ویژگی‌ها را عوض
    می‌کنند — خطا بالا می‌رود تا فراخوان «نامعلوم» نشان دهد. نمادی که تاریخچه‌اش کوتاه‌تر است همهٔ موجودی را
    برمی‌گرداند. کش و تک‌پروازی به ازای (symbol, tf)؛ پس از انقضا فقط دنبالهٔ تازه گرفته و وصل می‌شود.
    """
    key = (symbol, tf)
    now = time.time()
    with _lock:
        hit = _long_cache.get(key)
    if _long_fresh(hit, tf, bars, now):
        return hit[1]
    with _flight_lock(("kll", symbol, tf)):
        now = time.time()
        with _lock:
            hit = _long_cache.get(key)
        if _long_fresh(hit, tf, bars, now):
            return hit[1]
        data = None
        if hit and len(hit[1]["t"]) >= bars:
            data = _extend_long(symbol, tf, hit[1], now)
        if data is None:
            data = _fetch_klines_long(symbol, tf, bars, now)
        with _lock:
            _long_cache[key] = (now, data)
        return data


def _closed(rows, tf, now):
    step = TF_MINUTES[tf] * 60000
    return [r for r in rows if r[0] + step <= now * 1000]


def _extend_long(symbol, tf, old, now):
    """فقط کندل‌های بسته‌شدهٔ پس از آخرین کندلِ کش؛ اگر صفحه به آخرین کندلِ کش نرسد None (دریافتِ کامل)."""
    step = TF_MINUTES[tf] * 60000
    last = int(old["t"][-1])
    missing = int((now * 1000 - last) // step) + 2
    if missing > LONG_PAGE:
        return None
    page = _spot_page(symbol, tf, missing)
    if not page or page[0][0] > last:
        return None
    new = [r for r in _closed(page, tf, now) if r[0] > last]
    if not new:
        return old
    add = _long_arrays(new, old["src"])
    keep = len(old["t"])                              # پنجرهٔ غلتان با همان طول
    return {k: (old[k] if k == "src" else np.concatenate([old[k], add[k]])[-keep:]) for k in old}


def _fetch_klines_long(symbol, tf, bars, now):
    rows, end = [], None
    for _ in range(bars // LONG_PAGE + 2):            # +۱ برای کندلِ باز، +۱ حاشیه
        chunk = _spot_page(symbol, tf, LONG_PAGE, end)
        if not chunk:
            break
        rows = chunk + rows
        end = chunk[0][0] - 1
        if len(chunk) < LONG_PAGE or len(_closed(rows, tf, now)) >= bars:
            break                                     # ابتدای تاریخچهٔ نماد یا به اندازهٔ کافی
    rows = _closed(rows, tf, now)
    t = [r[0] for r in rows]
    if any(b <= a for a, b in zip(t, t[1:])):          # صفحه‌ها نباید هم‌پوشانی/بی‌نظمی داشته باشند
        rows = sorted({r[0]: r for r in rows}.values())
    if not rows:
        raise RuntimeError(f"کندلِ بلندِ {symbol} {tf} دریافت نشد")
    return _long_arrays(rows, "binance-spot")


# ── فاندینگِ کوکوین فیوچرز (همان منبع در تاریخچه و زنده؛ ویژگی‌های kfund_* هسته) ──
KFUND_TTL = 600.0
KFUND_ROWS = 181                 # z روی ۹۰ تسویهٔ قبلی برای هر یک از ۹۰ تسویهٔ آخر + آخرین
KFUND_LOOKBACK_MS = 400 * 86_400_000
FUND_INTERVALS_H = (1.0, 2.0, 4.0, 8.0)
_kfund_cache: dict = {}          # symbol -> (ts, data)


def kucoin_contract(symbol):
    """BTCUSDT → XBTUSDTM (قراردادِ دائمیِ USDT کوکوین)."""
    base = symbol[:-4] if symbol.endswith("USDT") else symbol
    return ("XBT" if base == "BTC" else base) + "USDTM"


def kucoin_funding_pages(contract, start_ms, end_ms, max_pages=None, get=None):
    """نرخ‌های تسویه‌شدهٔ [start_ms, end_ms] به‌صورتِ [(timepoint_ms, rate)] صعودی.

    /api/v1/contract/funding-rates با from و to (هر دو شامل، ms) حداکثر ۱۰۰ ردیفِ **جدیدترِ** بازه را
    جدید→قدیم می‌دهد (بی‌این دو پارامتر ۴۰۰). پس to هر بار به قدیمی‌ترین timepoint منهای ۱ می‌رود تا پاسخِ خالی.
    خطای منطقی با HTTP 200 می‌آید (مثلاً 404000 برای قراردادِ ناموجود) و بالا می‌رود.
    """
    get = get or _get_json
    seen = {}
    to, pages = int(end_ms), 0
    while to >= start_ms and (max_pages is None or pages < max_pages):
        j = get(KUCOIN_FUT + "/api/v1/contract/funding-rates",
                {"symbol": contract, "from": int(start_ms), "to": to})
        pages += 1
        if str(j.get("code")) != "200000":
            raise RuntimeError(f"کوکوین فاندینگ {contract}: {j.get('code')} {j.get('msg')}")
        rows = j.get("data") or []
        if isinstance(rows, dict):
            rows = rows.get("dataList") or []
        if not rows:
            break
        for r in rows:
            seen[int(r["timepoint"])] = float(r["fundingRate"])
        oldest = min(int(r["timepoint"]) for r in rows)
        if oldest > to:
            break
        to = oldest - 1
    return sorted(seen.items())


def infer_interval_h(t):
    """فاصلهٔ فاندینگ (ساعت) از فاصلهٔ تسویه‌ها — یک مسیرِ مشترک برای آرشیوِ kfund و زنده.

    فاصله تا تسویهٔ قبلی اگر یکی از ۱/۲/۴/۸ ساعت باشد؛ وگرنه (اولین ردیف یا تسویهٔ غایب) فاصله تا بعدی؛
    وگرنه مقدارِ ردیفِ قبل؛ وگرنه ۸.
    """
    t = np.asarray(t, dtype=np.int64)
    n = len(t)
    d = np.rint(np.diff(t) / 3_600_000.0)
    d = np.where(np.isin(d, FUND_INTERVALS_H), d, np.nan)
    prev = np.r_[np.nan, d] if n else np.zeros(0)
    nxt = np.r_[d, np.nan] if n else np.zeros(0)
    iv = np.where(np.isfinite(prev), prev, nxt)
    for i in range(n):
        if not np.isfinite(iv[i]):
            iv[i] = iv[i - 1] if i else 8.0
    return iv.astype(float)


def kfund_arrays(rows):
    """[(t, rate)] → {t, rate, interval_h} هم‌شکلِ kfund_<SYM>.npz."""
    t = np.array([r[0] for r in rows], dtype=np.int64)
    return {"t": t, "rate": np.array([r[1] for r in rows], dtype=float), "interval_h": infer_interval_h(t)}


def _kfund_fresh(hit, rows):
    """کش تازه است و درخواستِ قبلی دست‌کم همین‌قدر ردیف خواسته بود (یا همین‌قدر دارد)."""
    return bool(hit) and time.time() - hit[0] < KFUND_TTL and (rows <= hit[2] or len(hit[1]["t"]) >= rows)


def get_kucoin_funding(symbol, rows=KFUND_ROWS):
    """تاریخچهٔ اخیرِ فاندینگِ تسویه‌شدهٔ کوکوین (دست‌کم ``rows`` تسویه اگر موجود باشد) — کش ۱۰ دقیقه، تک‌پروازی.

    خروجی: t (ms، زمانِ تسویه)، rate، interval_h، src. خطا بالا می‌رود (فراخوان «نامعلوم» نشان می‌دهد)."""
    with _lock:
        hit = _kfund_cache.get(symbol)
    if _kfund_fresh(hit, rows):
        return hit[1]
    with _flight_lock(("kfund", symbol)):
        with _lock:
            hit = _kfund_cache.get(symbol)
        if _kfund_fresh(hit, rows):
            return hit[1]
        now = time.time()
        end = int(now * 1000)
        got = kucoin_funding_pages(kucoin_contract(symbol), end - KFUND_LOOKBACK_MS, end,
                                   max_pages=-(-rows // 100))
        got = [r for r in got if r[0] <= end]
        if not got:
            raise RuntimeError(f"فاندینگِ کوکوین برای {symbol} خالی بود")
        data = kfund_arrays(got)
        data["src"] = "kucoin"
        with _lock:
            _kfund_cache[symbol] = (now, data, rows)
        return data


HIST_DIR = paths.data("hist")
HIST_TTL = 86400            # تاریخچه عمیق روزی یک‌بار تازه می‌شود


def get_history(symbol, tf, bars=3000):
    """تاریخچه عمیق (تا چند هزار کندل) برای کالیبراسیون — با صفحه‌بندی و کش دیسک ۲۴ساعته."""
    os.makedirs(HIST_DIR, exist_ok=True)
    path = os.path.join(HIST_DIR, f"{symbol}_{tf}.json")
    with _flight_lock(("hist", path)):             # چند تحلیلِ هم‌زمان = یک صفحه‌بندی، نه چند تا
        if os.path.exists(path) and time.time() - os.path.getmtime(path) < HIST_TTL:
            with open(path, "r", encoding="utf-8") as f:
                cached = json.load(f)
            if len(cached.get("c", [])) >= bars * 0.7:
                return cached
        return _fetch_history(symbol, tf, bars, path)


def _fetch_history(symbol, tf, bars, path):
    rows, end = [], None
    try:
        while len(rows) < bars:
            params = {"symbol": symbol, "interval": TF_BINANCE[tf], "limit": 1000}
            if end:
                params["endTime"] = end
            raw = _binance_json("/api/v3/klines", params)
            if not raw:
                break
            chunk = [(int(r[0]), float(r[1]), float(r[2]), float(r[3]), float(r[4]), float(r[5])) for r in raw]
            rows = chunk + rows
            end = chunk[0][0] - 1
            if len(raw) < 1000:
                break
    except Exception:
        rows, after = [], None
        inst = symbol[:-4] + "-USDT"
        while len(rows) < min(bars, 2000):
            params = {"instId": inst, "bar": TF_OKX[tf], "limit": "100"}
            if after:
                params["after"] = str(after)
            raw = _get_json(OKX + "/api/v5/market/history-candles", params)["data"]
            if not raw:
                break
            chunk = [(int(r[0]), float(r[1]), float(r[2]), float(r[3]), float(r[4]), float(r[5])) for r in reversed(raw)]
            rows = chunk + rows
            after = chunk[0][0]
            if len(raw) < 100:
                break
    if rows and rows[-1][0] + TF_MINUTES[tf] * 60000 > time.time() * 1000:
        rows = rows[:-1]
    rows = rows[-bars:]
    if len(rows) < 400:
        raise RuntimeError(f"تاریخچه کافی برای {symbol} {tf} به دست نیامد ({len(rows)})")
    out = {"t": [r[0] for r in rows], "o": [r[1] for r in rows], "h": [r[2] for r in rows],
           "l": [r[3] for r in rows], "c": [r[4] for r in rows], "v": [r[5] for r in rows]}
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(out, f)
    os.replace(tmp, path)
    return out


FAPI_HOSTS = [
    "https://fapi.binance.com",
    "https://fapi1.binance.com",
    "https://fapi2.binance.com",
    "https://fapi3.binance.com",
]


def _fapi_json(path, params=None):
    last_err = None
    for host in FAPI_HOSTS:
        try:
            return _get_json(host + path, params)
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 400:
                raise                              # نمادِ بی‌قرارداد — میزبانِ بعدی هم همین را می‌گوید
            last_err = e
        except Exception as e:  # noqa: BLE001
            last_err = e
            continue
    raise RuntimeError(str(last_err) if last_err else "fapi unavailable")


NEG_TTL = 600.0             # «منبع در دسترس نبود» این‌قدر در حافظه می‌ماند (نه ۱۲ ساعت روی دیسک)
_neg_until: dict = {}       # path -> تا این لحظه بدونِ تماسِ شبکه خالی برگردان


def _neg_hit(path):
    return _neg_until.get(path, 0.0) > time.time()


def _answered(e):
    """منبع پاسخ داد ولی داده‌ای نداشت (نمادِ نامعتبر) — برخلافِ قطع/مسدودی/کول‌داون."""
    return isinstance(e, httpx.HTTPStatusError) and e.response.status_code in (400, 404)


FUNDING_TTL = 43200             # سقفِ سنِ کشِ فاندینگ حتی اگر تسویهٔ تازه‌ای در راه نباشد
FUNDING_RECHECK_SEC = 300       # پس از موعدِ تسویه، تا انتشارش هر ۵ دقیقه یک‌بار
FUNDING_PUBLISH_GRACE_MS = 60_000
FUNDING_STEP_MS = 8 * 3600 * 1000
FUNDING_MAX_PAGES = 80          # سقفِ صفحه در هر واکشی (بایننس ۱۰۰۰ ردیف، کوکوین ۱۰۰ ردیف در صفحه)
FUNDING_PAGE_GAP = {"binance": 0.65, "kucoin": 0.12, "okx": 0.12}   # فاصلهٔ کمینهٔ درخواست‌ها (سقفِ نرخ)
_fund_pace_lock = threading.Lock()
_fund_pace_last: dict = {}


def _fund_pace(src):
    """صفحه‌بندیِ عمیقِ ۱۰۰ نماد نباید به سقفِ نرخ بخورد (بایننس ۵۰۰ درخواست در ۵ دقیقه).
    فقط صفحه‌های دوم به بعد صبر می‌کنند؛ به‌روزرسانیِ یک‌صفحه‌ایِ زنده معطل نمی‌شود."""
    gap = FUNDING_PAGE_GAP.get(src, 0.0)
    with _fund_pace_lock:
        wait = _fund_pace_last.get(src, 0.0) + gap - time.time()
        if wait > 0:
            time.sleep(wait)
        _fund_pace_last[src] = time.time()


def _funding_rows_binance(symbol, start_ms, end_ms):
    """[start, end] از /fapi/v1/fundingRate با startTime — صفحه‌به‌صفحه رو به جلو."""
    out, start, pages = {}, int(start_ms), 0
    while start <= end_ms and pages < FUNDING_MAX_PAGES:
        if pages:
            _fund_pace("binance")
        raw = _fapi_json("/fapi/v1/fundingRate",
                         {"symbol": symbol, "startTime": start, "endTime": int(end_ms), "limit": 1000})
        pages += 1
        if not raw:
            break
        for r in raw:
            out[int(r["fundingTime"])] = float(r["fundingRate"])
        last = max(int(r["fundingTime"]) for r in raw)
        if len(raw) < 1000 or last < start:
            break
        start = last + 1
    return sorted([t, v] for t, v in out.items())


def _funding_rows_kucoin(symbol, start_ms, end_ms):
    """[start, end] از کوکوین فیوچرز — تاریخچهٔ عمیق وقتی fapi بایننس مسدود است (۴۵۱)."""
    pages = []

    def get(url, params):
        if pages:
            _fund_pace("kucoin")
        pages.append(1)
        return _get_json(url, params)
    rows = kucoin_funding_pages(kucoin_contract(symbol), int(start_ms), int(end_ms),
                                max_pages=FUNDING_MAX_PAGES, get=get)
    return [[int(t), float(v)] for t, v in rows]


def _funding_rows_okx(symbol, start_ms, end_ms):
    """[start, end] از OKX — فقط حدودِ سه ماهِ اخیر را نگه می‌دارد (after = قدیمی‌تر از)."""
    inst = symbol.replace("USDT", "") + "-USDT-SWAP"
    out, after, pages = {}, None, 0
    while pages < FUNDING_MAX_PAGES:
        params = {"instId": inst, "limit": 100}
        if after is not None:
            params["after"] = str(after)
        if pages:
            _fund_pace("okx")
        raw = _get_json(OKX + "/api/v5/public/funding-rate-history", params)["data"]
        pages += 1
        if not raw:
            break
        for r in raw:
            t = int(r["fundingTime"])
            if start_ms <= t <= end_ms:
                out[t] = float(r["fundingRate"])
        oldest = min(int(r["fundingTime"]) for r in raw)
        if oldest <= start_ms or len(raw) < 100:
            break
        after = oldest
    return sorted([t, v] for t, v in out.items())


_FUNDING_SOURCES = {"binance": _funding_rows_binance, "kucoin": _funding_rows_kucoin,
                    "okx": _funding_rows_okx}
FUNDING_CHAIN = ("binance", "kucoin", "okx")   # عمیق‌ها اول؛ OKX فقط ~۳ ماه دارد


def _read_funding_doc(path):
    """فایلِ کش → {src, from_ms, rows}. قالبِ قدیمی (فهرستِ خالی) منبعِ نامعلوم دارد و از نو گرفته می‌شود."""
    with open(path, "r", encoding="utf-8") as f:
        doc = json.load(f)
    if isinstance(doc, list):
        return {"src": None, "from_ms": None, "rows": doc}
    doc.setdefault("rows", [])
    return doc


def _write_funding_doc(path, doc):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(doc, f)
    os.replace(tmp, path)


def _funding_doc_fresh(doc, mtime, now=None):
    """کش تا **تسویهٔ بعدی** تازه است، نه ۱۲ ساعتِ ثابت (calib-F3).

    موعدِ تسویهٔ بعد از فاصلهٔ دو ردیفِ آخر (۱/۴/۸ ساعت) درمی‌آید؛ پس از موعد تا انتشارش هر
    ``FUNDING_RECHECK_SEC`` یک‌بار سر می‌زند. کشِ قدیمی‌تر از ``FUNDING_TTL`` در هر حال کهنه است.
    """
    now = time.time() if now is None else now
    if doc.get("src") is None and doc.get("rows"):
        return False                               # قالبِ قدیمی: منبع نامعلوم ⇒ یک‌بار از نو
    if now - mtime >= FUNDING_TTL:
        return False
    rows = doc.get("rows") or []
    if len(rows) < 2:
        return True
    step_h = min(max(round((rows[-1][0] - rows[-2][0]) / 3_600_000), 1), 8)
    due = rows[-1][0] + step_h * 3_600_000 + FUNDING_PUBLISH_GRACE_MS
    if now * 1000 < due:
        return True
    return now - mtime < FUNDING_RECHECK_SEC


def get_funding_history(symbol, max_rows=1000, since_ms=None):
    """تاریخچه فاندینگ‌ریت فیوچرز ``[[fundingTime, rate], ...]`` صعودی — خالی اگر در دسترس نباشد.

    * بی ``since_ms`` (اجرا، متا-گیت، پوزیشن‌ها): آخرین ``max_rows`` ردیف، مثل قبل.
    * با ``since_ms`` (آموزش): **همهٔ** ردیف‌ها از ``since_ms``؛ تاریخچه صفحه‌به‌صفحه تا آن‌جا
      پر می‌شود (calib-F4). قبلاً فقط ۱۰۰۰ ردیفِ آخر (~۳۳۳ روز) بود و funding_dir در ۷۸-۸۹٪
      نمونه‌های 4h/1d صفر می‌ماند در حالی که زنده همیشه مقدار داشت.

    کشِ دیسک همهٔ ردیف‌ها را با منبعشان نگه می‌دارد؛ به‌روزرسانی فقط ردیف‌های تازه را از **همان**
    منبع می‌گیرد (آموزش و اجرا یک سری می‌بینند، نه بایننس این‌جا و OKX آن‌جا).
    """
    os.makedirs(HIST_DIR, exist_ok=True)
    path = os.path.join(HIST_DIR, f"funding_{symbol}.json")
    now = time.time()
    # با since: ۴۰ تسویهٔ پیش از آن هم، تا z غلتانِ ۳۰تایی از اولین ردیفِ لازم گرم باشد
    want_from = (int(since_ms) - 40 * FUNDING_STEP_MS if since_ms is not None
                 else int(now * 1000) - int(max_rows) * FUNDING_STEP_MS)
    # تحلیل (live_funding_z) و واکشیِ پس‌زمینه (_fetch_funding) هم‌زمان همین را می‌خواستند
    with _flight_lock(("funding", path)):
        doc = None
        if os.path.exists(path):
            try:
                doc = _read_funding_doc(path)
            except (OSError, ValueError):
                doc = None
        covered = bool(doc) and doc.get("from_ms") is not None and doc["from_ms"] <= want_from
        if not (covered and _funding_doc_fresh(doc, os.path.getmtime(path), now)):
            if _neg_hit(path):
                rows = (doc or {}).get("rows") or []
            else:
                doc = _fetch_funding_history(symbol, path, doc, want_from)
                rows = (doc or {}).get("rows") or []
        else:
            rows = doc.get("rows") or []
    if since_ms is not None:
        return [r for r in rows if r[0] >= want_from]
    return rows[-int(max_rows):] if max_rows else rows


def _fetch_funding_history(symbol, path, doc, want_from):
    """ردیف‌های تازه (و در صورتِ نیاز، پرکردنِ عقب تا ``want_from``) از منبعِ کش، وگرنه زنجیره."""
    end = int(time.time() * 1000)
    rows = {int(t): float(v) for t, v in ((doc or {}).get("rows") or [])}
    src = old_src = (doc or {}).get("src")
    from_ms = (doc or {}).get("from_ms")
    if src == "okx" and (from_ms is None or want_from < from_ms):
        src = None                     # OKX فقط ~۳ ماه دارد: برای پرکردنِ عقب اول منبعِ عمیق‌تر
    answered = False
    try:
        if src in _FUNDING_SOURCES:
            fetch = _FUNDING_SOURCES[src]
            if from_ms is None or want_from < from_ms:        # پرکردنِ عقب
                lo_end = (min(rows) - 1) if rows else end
                for t, v in fetch(symbol, want_from, lo_end):
                    rows[t] = v
                from_ms = want_from
            start = (max(rows) + 1) if rows else want_from
            for t, v in fetch(symbol, start, end):
                rows[t] = v
            answered = True
        else:
            src, got = _funding_chain(symbol, want_from, end)
            if src != old_src:
                rows = {}                  # منبعِ تازه: سری‌های دو صرافی آمیخته نمی‌شوند
            for t, v in got:
                rows[int(t)] = float(v)
            from_ms, answered = want_from, True
    except Exception as e:  # noqa: BLE001 — منبعِ کش در دسترس نیست: همان کشِ قبلی، بدونِ آمیختنِ منبع
        answered = _answered(e)
        if not answered:
            _neg_until[path] = time.time() + NEG_TTL
            return doc
    if not answered:
        # هیچ منبعی در دسترس نبود: «خالی» به‌جای ۱۲ ساعت روی دیسک فقط NEG_TTL در حافظه می‌ماند
        _neg_until[path] = time.time() + NEG_TTL
        return doc
    new = {"src": src, "from_ms": from_ms, "rows": sorted([t, v] for t, v in rows.items())}
    _write_funding_doc(path, new)
    return new


def _funding_chain(symbol, start_ms, end_ms):
    """اولین منبعی که پاسخ داد → (src, rows). پاسخِ ۴۰۰/۴۰۴ (نمادِ بی‌قرارداد) منبعِ بعدی را امتحان
    می‌کند؛ اگر همه «نیست» گفتند ``("none", [])`` — «پاسخ‌داده»، پس کشِ خالی روی دیسک می‌ماند.
    هیچ منبعی در دسترس نبود ⇒ خطا (فراخوان فقط کشِ منفیِ حافظه می‌سازد)."""
    any_answered = False
    for src in FUNDING_CHAIN:
        try:
            return src, _FUNDING_SOURCES[src](symbol, start_ms, end_ms)
        except Exception as e:  # noqa: BLE001 — منبعِ بعدی
            any_answered = any_answered or _answered(e)
    if any_answered:
        return "none", []
    raise RuntimeError("هیچ منبعِ فاندینگی در دسترس نبود")


def funding_z_map(symbol, window=30, since_ms=None):
    """نقشه ts→z-score فاندینگ (نسبت به ۳۰ فاندینگ قبلی) برای ویژگی‌سازی رویدادها.
    ``since_ms`` (آموزش): کلِ تاریخچه از آن لحظه؛ بی آن (اجرا) ۱۰۰۰ ردیفِ آخر — z هر ردیف فقط به
    ۳۰ ردیفِ قبلش بسته است، پس مقدارِ ردیف‌های اخیر در هر دو یکی است."""
    rows = get_funding_history(symbol, since_ms=since_ms)
    out = []
    vals = []
    for ts, rate in rows:
        if len(vals) >= 8:
            seg = vals[-window:]
            mu = sum(seg) / len(seg)
            sd = (sum((x - mu) ** 2 for x in seg) / len(seg)) ** 0.5 or 1e-9
            out.append((ts, (rate - mu) / sd))
        vals.append(rate)
    return out          # مرتب بر اساس زمان


def get_oi_history(symbol, period="1h", limit=48):
    """تاریخچهٔ Open Interest — اول Binance fapi، بعد OKX. کش ۱ساعته."""
    os.makedirs(HIST_DIR, exist_ok=True)
    path = os.path.join(HIST_DIR, f"oi_{symbol}_{period}.json")
    with _flight_lock(("oi", path)):
        if os.path.exists(path) and time.time() - os.path.getmtime(path) < 3600:
            with open(path, "r", encoding="utf-8") as f:
                cached = json.load(f)
            if cached:                             # «[]»ِ قدیمیِ روی دیسک (باگِ instId) کش نیست
                return cached
        if _neg_hit(path):
            return []
        return _fetch_oi_history(symbol, period, limit, path)


def _okx_oi_rows(symbol, period, limit):
    """OI از rubikِ OKX: instId مثلِ BTC-USDT-SWAP لازم است (بی‌آن ۴۰۰ «instId can't be empty»).
    ردیف‌ها [ts, oi (قرارداد), oiCcy (ارزِ پایه), oiUsd] جدید→قدیم؛ oiCcy هم‌واحدِ sumOpenInterestِ بایننس است."""
    inst = symbol[:-4] + "-USDT-SWAP"
    raw = _get_json(
        OKX + "/api/v5/rubik/stat/contracts/open-interest-history",
        {"instId": inst, "period": TF_OKX_OI.get(period, "1H"), "limit": str(min(int(limit), 100))},
    ).get("data") or []
    parsed = []
    for r in raw:
        if len(r) >= 2:
            parsed.append([int(r[0]), float(r[2] if len(r) >= 3 else r[1])])
    parsed.sort(key=lambda x: x[0])
    return parsed[-limit:]


def _fetch_oi_history(symbol, period, limit, path):
    rows = []
    try:
        raw = _fapi_json(
            "/futures/data/openInterestHist",
            {"symbol": symbol, "period": period, "limit": limit},
        )
        rows = [[int(r["timestamp"]), float(r["sumOpenInterest"])] for r in raw]
        rows.sort(key=lambda x: x[0])
    except Exception:  # noqa: BLE001
        rows = []
    if not rows:
        try:
            rows = _okx_oi_rows(symbol, period, limit)
        except Exception:  # noqa: BLE001
            rows = []
    if not rows:
        # ۴xx (مثلِ درخواستِ بی‌instId که ماه‌ها «[]» را یک ساعت روی دیسک نگه می‌داشت و وتوی OI را
        # بی‌صدا خاموش می‌کرد)، قطعی یا پاسخِ خالی: «خالی» هرگز روی دیسک نمی‌ماند، فقط NEG_TTL در حافظه
        _neg_until[path] = time.time() + NEG_TTL
        return rows
    # ⚠️ پشتیبانِ «EMAِ حجمِ اسپات به‌جای OI» حذف شد: متا-گیت آن را اهرم می‌خواند و
    # بلاکِ سخت می‌زد. نبودِ داده = «نامعلوم»، نه عددِ ساختگی.
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(rows, f)
    os.replace(tmp, path)
    return rows


def oi_crowd_stats(symbol, period="1h"):
    """آمار OI برای متا-گیت: تغییر نسبی، z-score، پرچم اهرمِ بالا."""
    rows = get_oi_history(symbol, period=period, limit=48)
    if len(rows) < 8:
        return {
            "oi": None, "oi_chg_pct": 0.0, "oi_z": 0.0,
            "oi_rising": False, "oi_falling": False, "ok": False,
        }
    vals = [float(r[1]) for r in rows]
    cur, prev = vals[-1], vals[-2]
    chg = (cur / prev - 1.0) * 100.0 if prev > 0 else 0.0
    rets = []
    for i in range(1, len(vals)):
        if vals[i - 1] > 0:
            rets.append(vals[i] / vals[i - 1] - 1.0)
    if len(rets) < 5:
        z = 0.0
    else:
        mu = sum(rets) / len(rets)
        var = sum((x - mu) ** 2 for x in rets) / len(rets)
        sd = var ** 0.5 or 1e-9
        z = (rets[-1] - mu) / sd
    return {
        "oi": cur,
        "oi_chg_pct": round(chg, 3),
        "oi_z": round(z, 3),
        "oi_rising": z >= 1.2 or chg >= 1.5,
        "oi_falling": z <= -1.2 or chg <= -1.5,
        "ok": True,
    }

def get_dxy_daily():
    """شاخص دلار (DXY) روزانه از stooq — کش دیسک ۲۴ساعته؛ در صورت عدم دسترسی خالی."""
    os.makedirs(HIST_DIR, exist_ok=True)
    path = os.path.join(HIST_DIR, "dxy_daily.json")
    if os.path.exists(path) and time.time() - os.path.getmtime(path) < 86400:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    rows = []
    try:
        r = _client.get("https://stooq.com/q/d/l/", params={"s": "dx.f", "i": "d"})
        r.raise_for_status()
        for line in r.text.strip().splitlines()[1:]:
            parts = line.split(",")
            if len(parts) >= 5 and parts[0] and parts[4]:
                y, m, d = parts[0].split("-")
                ts = int(time.mktime((int(y), int(m), int(d), 0, 0, 0, 0, 0, 0))) * 1000
                rows.append([ts, float(parts[4])])
        rows = rows[-600:]
    except Exception:  # noqa: BLE001
        rows = []
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(rows, f)
    os.replace(tmp, path)
    return rows


def last_price(symbol):
    """قیمت لحظه‌ای برای به‌روزرسانی پوزیشن‌ها."""
    try:
        return float(_binance_json("/api/v3/ticker/price", {"symbol": symbol})["price"])
    except Exception:
        inst = symbol[:-4] + "-USDT"
        return float(_get_json(OKX + "/api/v5/market/ticker", {"instId": inst})["data"][0]["last"])
