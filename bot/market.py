# -*- coding: utf-8 -*-
"""لایه داده بازار — دریافت ۱۵ ارز برتر و کندل‌ها از Binance (با چند هاست جایگزین) و OKX به‌عنوان پشتیبان."""
import json
import os
import time
import threading
import httpx

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


def _is_pegged(last, high, low):
    try:
        return last > 0 and (high - low) / last * 100 < PEG_MAX_RANGE_PCT
    except (TypeError, ZeroDivisionError):
        return False

TF_BINANCE = {"15m": "15m", "1h": "1h", "4h": "4h", "1d": "1d"}
TF_OKX = {"15m": "15m", "1h": "1H", "4h": "4H", "1d": "1D"}
TF_MINUTES = {"15m": 15, "1h": 60, "4h": 240, "1d": 1440}
KLINE_TTL = {"15m": 60, "1h": 120, "4h": 300, "1d": 900}

_client = httpx.Client(timeout=12.0, headers={"User-Agent": "ctp-bot/1.0"})
_lock = threading.Lock()
_kline_cache: dict = {}          # (symbol, tf) -> (ts, data)
_top_cache = {"ts": 0.0, "symbols": [], "tickers": {}}


def _get_json(url, params=None):
    r = _client.get(url, params=params)
    r.raise_for_status()
    return r.json()


def _binance_json(path, params=None):
    last_err = None
    for host in BINANCE_HOSTS:
        try:
            return _get_json(host + path, params)
        except Exception as e:  # noqa: BLE001 — هاست بعدی را امتحان کن
            last_err = e
    raise last_err


def get_top_symbols(n=15):
    """۱۵ جفت USDT برتر بر اساس حجم ۲۴ساعته + دیتای تیکر (قیمت و تغییر ۲۴س)."""
    with _lock:
        if time.time() - _top_cache["ts"] < 600 and len(_top_cache["symbols"]) >= n:
            return _top_cache["symbols"][:n], _top_cache["tickers"]
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
        _top_cache.update(ts=time.time(), symbols=symbols, tickers=tickers)
    return symbols, tickers


def cache_stats():
    """سنِ کشِ کندل‌ها به تفکیکِ تایم‌فریم — برای پایشِ سلامت (بدونِ تماسِ شبکه)."""
    now = time.time()
    out = {}
    with _lock:
        for (sym, tf), (ts, _data) in _kline_cache.items():
            row = out.setdefault(tf, {"symbols": 0, "newest_age_sec": None, "oldest_age_sec": None})
            row["symbols"] += 1
            age = now - ts
            row["newest_age_sec"] = age if row["newest_age_sec"] is None else min(row["newest_age_sec"], age)
            row["oldest_age_sec"] = age if row["oldest_age_sec"] is None else max(row["oldest_age_sec"], age)
        top_age = now - _top_cache["ts"] if _top_cache["ts"] else None
    for row in out.values():
        row["newest_age_sec"] = round(row["newest_age_sec"], 1)
        row["oldest_age_sec"] = round(row["oldest_age_sec"], 1)
    return {"klines_by_tf": out, "universe_age_sec": round(top_age, 1) if top_age else None,
            "universe_size": len(_top_cache["symbols"])}


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
    data = None
    try:
        raw = _binance_json("/api/v3/klines", {"symbol": symbol, "interval": TF_BINANCE[tf], "limit": limit + 1})
        rows = [(int(r[0]), float(r[1]), float(r[2]), float(r[3]), float(r[4]), float(r[5])) for r in raw]
        if rows and rows[-1][0] + TF_MINUTES[tf] * 60000 > now * 1000:
            rows = rows[:-1]                       # حذف کندل باز
        data = rows[-limit:]
    except Exception:
        inst = symbol[:-4] + "-USDT"
        raw = _get_json(OKX + "/api/v5/market/candles", {"instId": inst, "bar": TF_OKX[tf], "limit": str(min(limit + 1, 300))})["data"]
        rows = []
        for r in reversed(raw):                    # OKX جدیدترین اول است
            if len(r) >= 9 and r[8] == "0":
                continue                           # کندل تأییدنشده
            rows.append((int(r[0]), float(r[1]), float(r[2]), float(r[3]), float(r[4]), float(r[5])))
        data = rows[-limit:]
    if not data or len(data) < 220:
        raise RuntimeError(f"داده کافی برای {symbol} {tf} دریافت نشد ({len(data or [])} کندل)")
    out = {
        "t": [r[0] for r in data], "o": [r[1] for r in data], "h": [r[2] for r in data],
        "l": [r[3] for r in data], "c": [r[4] for r in data], "v": [r[5] for r in data],
    }
    with _lock:
        _kline_cache[key] = (now, out)
    return out


HIST_DIR = paths.data("hist")
HIST_TTL = 86400            # تاریخچه عمیق روزی یک‌بار تازه می‌شود


def get_history(symbol, tf, bars=3000):
    """تاریخچه عمیق (تا چند هزار کندل) برای کالیبراسیون — با صفحه‌بندی و کش دیسک ۲۴ساعته."""
    os.makedirs(HIST_DIR, exist_ok=True)
    path = os.path.join(HIST_DIR, f"{symbol}_{tf}.json")
    if os.path.exists(path) and time.time() - os.path.getmtime(path) < HIST_TTL:
        with open(path, "r", encoding="utf-8") as f:
            cached = json.load(f)
        if len(cached.get("c", [])) >= bars * 0.7:
            return cached
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
        except Exception as e:  # noqa: BLE001
            last_err = e
            continue
    raise RuntimeError(str(last_err) if last_err else "fapi unavailable")


def get_funding_history(symbol, max_rows=1000):
    """تاریخچه فاندینگ‌ریت فیوچرز (هر ۸ ساعت) — کش دیسک ۱۲ساعته. خالی اگر در دسترس نباشد."""
    os.makedirs(HIST_DIR, exist_ok=True)
    path = os.path.join(HIST_DIR, f"funding_{symbol}.json")
    if os.path.exists(path) and time.time() - os.path.getmtime(path) < 43200:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    rows = []
    try:
        raw = _fapi_json("/fapi/v1/fundingRate", {"symbol": symbol, "limit": max_rows})
        rows = [[int(r["fundingTime"]), float(r["fundingRate"])] for r in raw]
    except Exception:  # noqa: BLE001
        # پشتیبان OKX
        try:
            inst = symbol.replace("USDT", "") + "-USDT-SWAP"
            raw = _get_json(OKX + "/api/v5/public/funding-rate-history",
                            {"instId": inst, "limit": min(max_rows, 100)})["data"]
            rows = [[int(r["fundingTime"]), float(r["fundingRate"])] for r in reversed(raw)]
        except Exception:  # noqa: BLE001
            rows = []
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(rows, f)
    os.replace(tmp, path)
    return rows


def funding_z_map(symbol, window=30):
    """نقشه ts→z-score فاندینگ (نسبت به ۳۰ فاندینگ قبلی) برای ویژگی‌سازی رویدادها."""
    rows = get_funding_history(symbol)
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
    if os.path.exists(path) and time.time() - os.path.getmtime(path) < 3600:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
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
            # OKX: uly مثل BTC-USDT ، period مثل 1H
            base = symbol.replace("USDT", "")
            uly = f"{base}-USDT"
            okx_period = {"15m": "5m", "1h": "1H", "4h": "4H", "1d": "1D"}.get(period, "1H")
            raw = _get_json(
                OKX + "/api/v5/rubik/stat/contracts/open-interest-history",
                {"instType": "SWAP", "uly": uly, "period": okx_period},
            ).get("data") or []
            # data: [ts, oi, oiCcy] جدید→قدیم
            parsed = []
            for r in raw:
                if len(r) >= 2:
                    parsed.append([int(r[0]), float(r[1])])
            parsed.sort(key=lambda x: x[0])
            rows = parsed[-limit:]
        except Exception:  # noqa: BLE001
            rows = []
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
