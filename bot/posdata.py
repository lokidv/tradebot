# -*- coding: utf-8 -*-
"""وضعیتِ بازار برای هر ارز — **فقط نمایشی**، هرگز ورودیِ مدل.

پیش‌ثبتِ core v2 (bot/data/research/prereg_core2.json) می‌گوید موقعیت‌گیریِ فیوچرزِ بایننس از این
دستگاه فقط از راهِ دور زدنِ انسدادِ منطقه‌ای (۴۵۱) می‌آید و آن راه استفاده نمی‌شود؛ پس این‌جا فقط
منابعی هست که مستقیم باز است:

- اسپاتِ بایننس (data-api.binance.vision): حجمِ خریدِ تیکر در برابرِ فروشِ تیکر در ۱/۴/۲۴ ساعت و CVDِ ۲۴ ساعته
  از کندل‌های ۵ دقیقه‌ایِ بسته؛
- OKX rubik (عمومی): حجمِ خرید/فروشِ تیکرِ ۵ دقیقه‌ای (ردیفِ آخر ناقص است و بعداً اصلاح می‌شود — کنار
  گذاشته می‌شود)، نسبتِ حساب‌های لانگ به شورت، موقعیت‌های باز و حجم؛
- فیوچرزِ کوکوین (عمومی): موقعیت‌های باز، فاندینگِ جاری/پیش‌بینی، فاصلهٔ مارک از شاخص، عمقِ دفترِ
  سفارش در ±۰٫۵٪ و ±۱٪ و ۹ تسویهٔ اخیرِ فاندینگ.

این اعداد تاریخچه‌ای با همین تعریف ندارند و لبه‌شان آزموده نشده؛ ``record`` آن‌ها را هر ۵ دقیقه
در ``hist_research/live_pos/<SYM>.jsonl`` می‌نویسد تا اگر روزی خواستیم بسنجیم، تاریخچهٔ هم‌منبع داشته
باشیم. هیچ چیزی از این ماژول به ``tradeable``، گیت‌ها یا اتوتریدر نمی‌رسد.
"""
import json
import math
import os
import re
import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor

import httpx

import paths
import watchlist

SPOT = "https://data-api.binance.vision"
OKX = "https://www.okx.com"
KUCOIN_FUT = "https://api-futures.kucoin.com"
ALLOWED_HOSTS = (SPOT, OKX, KUCOIN_FUT)       # هیچ میزبانِ دیگری (fapi، /futures/data، بای‌بیت) صدا زده نمی‌شود

CONTRACT = {"BTCUSDT": "XBTUSDTM", "ETHUSDT": "ETHUSDTM", "BNBUSDT": "BNBUSDTM",
            "SOLUSDT": "SOLUSDTM", "TRXUSDT": "TRXUSDTM"}
# ضریبِ قرارداد (سکه به ازای هر لات) — دیده‌شده در ۲۰۲۶-۰۹-۲۴؛ اگر جزئیاتِ قرارداد زودتر رسیده باشد از آن استفاده می‌شود
MULT_FALLBACK = {"XBTUSDTM": 0.001, "ETHUSDTM": 0.01, "BNBUSDTM": 0.01, "SOLUSDTM": 0.1, "TRXUSDTM": 100.0}

NOTE_FA = "فقط برای دیدنِ وضعیتِ بازار؛ ورودیِ مدل نیست و لبه‌اش آزموده نشده"
TIMEOUT = 5.0                  # مهلتِ هر درخواست (ثانیه)
CACHE_TTL = 60.0               # snapshot_all این‌قدر کش می‌شود
RECORD_EVERY = 300.0           # هر ارز حداکثر یک خط در هر ۵ دقیقه
BAR_MS = 300_000               # کندل/ردیفِ ۵ دقیقه‌ای
HOUR_MS = 3_600_000
BOOK_BANDS = (0.005, 0.01)     # ±۰٫۵٪ و ±۱٪ از قیمتِ میانه
FUNDING_ROWS = 9
OKX_RATE = (5, 2.2)            # OKX rubik بالای ~۵ درخواست در ۲ ثانیه خطای 50011/429 می‌دهد
OKX_COOLDOWN = 10.0            # پس از 429 این‌قدر OKX صدا زده نمی‌شود
LIVE_POS_DIR = paths.data("hist_research", "live_pos")
SCHEMA = 1

SOURCES = {
    "binance_spot": "Binance spot · data-api.binance.vision /api/v3/klines 5m",
    "okx_taker": "OKX rubik · /api/v5/rubik/stat/taker-volume (CONTRACTS, 5m)",
    "okx_lsr": "OKX rubik · /api/v5/rubik/stat/contracts/long-short-account-ratio (5m)",
    "okx_oi": "OKX rubik · /api/v5/rubik/stat/contracts/open-interest-volume (5m)",
    "kucoin_contract": "KuCoin futures · /api/v1/contracts/{symbol}",
    "kucoin_book": "KuCoin futures · /api/v1/level2/snapshot",
    "kucoin_funding": "KuCoin futures · /api/v1/contract/funding-rates",
}

# نامِ فیلد -> (منبع، توضیحِ فارسی)
FIELDS = {
    "spot_taker_1h": ("binance_spot", "خریدِ تیکر در برابرِ فروشِ تیکر در اسپاتِ بایننس، ۱ ساعتِ اخیر (USDT، کندل‌های ۵ دقیقه‌ایِ بسته)"),
    "spot_taker_4h": ("binance_spot", "خریدِ تیکر در برابرِ فروشِ تیکر در اسپاتِ بایننس، ۴ ساعتِ اخیر (USDT)"),
    "spot_taker_24h": ("binance_spot", "خریدِ تیکر در برابرِ فروشِ تیکر در اسپاتِ بایننس، ۲۴ ساعتِ اخیر (USDT)"),
    "spot_cvd_24h": ("binance_spot", "دلتای حجمِ تجمعی (خریدِ تیکر منهای فروشِ تیکر، USDT) در ۲۴ ساعتِ اخیرِ اسپاتِ بایننس، با قیمتِ بسته‌شدن"),
    "okx_taker_1h": ("okx_taker", "خرید/فروشِ تیکرِ همهٔ قراردادهای OKX برای این ارز، ۱ ساعتِ اخیر (دلار)؛ ردیفِ آخرِ ناقص کنار گذاشته شد"),
    "okx_taker_4h": ("okx_taker", "خرید/فروشِ تیکرِ همهٔ قراردادهای OKX برای این ارز، ۴ ساعتِ اخیر (دلار)؛ ردیفِ آخرِ ناقص کنار گذاشته شد"),
    "okx_lsr": ("okx_lsr", "نسبتِ حساب‌های لانگ به شورت در قراردادهای OKX (همهٔ کاربران) و تغییرِ ۱ و ۲۴ ساعته"),
    "okx_oi": ("okx_oi", "موقعیت‌های باز و حجمِ معاملاتِ قراردادهای OKX (دلار)، تغییرِ ۱ و ۲۴ ساعته"),
    "kc_open_interest": ("kucoin_contract", "موقعیت‌های بازِ قراردادِ دائمیِ کوکوین (لات، سکه، USDT به قیمتِ مارک)"),
    "kc_funding": ("kucoin_contract", "فاندینگِ دورهٔ جاریِ کوکوین (در تسویهٔ بعدی پرداخت می‌شود) و نرخِ پیش‌بینی، بیسیس‌پوینت"),
    "kc_premium_bp": ("kucoin_contract", "فاصلهٔ قیمتِ مارک از شاخص در کوکوین (بیسیس‌پوینت)؛ مثبت یعنی فیوچرز گران‌تر از شاخص"),
    "kc_book": ("kucoin_book", "دفترِ سفارشِ کوکوین: ارزشِ سفارش‌های خرید و فروش در ±۰٫۵٪ و ±۱٪ قیمتِ میانه و عدمِ توازن (مثبت = سمتِ خرید سنگین‌تر)؛ فقط زنده، بی‌تاریخچه"),
    "kc_funding_hist": ("kucoin_funding", "۹ تسویهٔ اخیرِ فاندینگِ کوکوین (بیسیس‌پوینت)، میانگین و پایداریِ علامت"),
}

_client = httpx.Client(timeout=httpx.Timeout(TIMEOUT), headers={"User-Agent": "ctp-bot/1.0"})
_mult = {}                      # ضریبِ قرارداد از آخرین جزئیاتِ قراردادِ خوانده‌شده
_cache = {"ts": 0.0, "data": None}
_cache_lock = threading.Lock()
_flight = threading.Lock()      # snapshot_allهای هم‌زمان فقط یک‌بار به شبکه می‌روند
_rec_lock = threading.Lock()
_last_rec = {}                  # sym -> آخرین زمانِ ثبت (ثانیه)
_okx_dead = {"until": 0.0}


class _Limiter:
    """پنجرهٔ لغزان: حداکثر n درخواست در هر per ثانیه (برای OKX rubik)."""

    def __init__(self, n, per, clock=time.monotonic, sleep=time.sleep):
        self.n, self.per, self.clock, self.sleep = n, per, clock, sleep
        self.q = deque()
        self.lock = threading.Lock()

    def acquire(self, max_wait=10.0):
        deadline = self.clock() + max_wait
        while True:
            with self.lock:
                t = self.clock()
                while self.q and t - self.q[0] >= self.per:
                    self.q.popleft()
                if len(self.q) < self.n:
                    self.q.append(t)
                    return
                wait = self.per - (t - self.q[0]) + 0.01
            if self.clock() + wait > deadline:
                raise RuntimeError("صفِ درخواست‌های OKX پر است (محدودیتِ نرخ)")
            self.sleep(wait)


_okx_limiter = _Limiter(*OKX_RATE)


def _http_get(url, params=None):
    """GET با مهلتِ ۵ ثانیه، فقط به میزبان‌های مجاز؛ OKX از محدودکنندهٔ نرخ می‌گذرد."""
    if not url.startswith(tuple(h + "/" for h in ALLOWED_HOSTS)):
        raise ValueError(f"میزبانِ غیرمجاز: {url.split('?')[0]}")
    is_okx = url.startswith(OKX + "/")
    if is_okx:
        if time.time() < _okx_dead["until"]:
            raise RuntimeError("OKX در مکث پس از خطای محدودیتِ نرخ (429)")
        _okx_limiter.acquire()
    r = _client.get(url, params=params)
    if is_okx and r.status_code == 429:
        _okx_dead["until"] = time.time() + OKX_COOLDOWN
    r.raise_for_status()
    return r.json()


# ── کمکی‌ها ──

def _f(x):
    """عدد یا None (رشتهٔ خالی/None/NaN)."""
    if x is None or x == "":
        return None
    v = float(x)
    return v if math.isfinite(v) else None


def _r(x, nd=2):
    return None if x is None else round(float(x), nd)


def _sig(x, n=6):
    """گرد کردن به n رقمِ معنادار (قیمت‌ها از ۰٫۳ تا ۸۰ هزار)."""
    if x is None or x == 0:
        return x
    return round(float(x), max(0, n - 1 - int(math.floor(math.log10(abs(x))))))


def _flow(buy, sell, bars):
    tot = buy + sell
    return {"buy": _r(buy, 0), "sell": _r(sell, 0), "delta": _r(buy - sell, 0),
            "buy_pct": _r(100.0 * buy / tot, 2) if tot > 0 else None,
            "imb": _r((buy - sell) / tot, 4) if tot > 0 else None, "bars": bars}


def _window_flow(rows, end_ms, window_ms):
    """rows: [(t_open, buy, sell)] صعودی؛ ردیف‌هایی که شروعشان در [end-window, end) است."""
    sel = [r for r in rows if end_ms - window_ms <= r[0] < end_ms]
    return _flow(sum(r[1] for r in sel), sum(r[2] for r in sel), len(sel))


def _at_or_before(rows, t):
    """آخرین ردیفی که زمانش ≤ t است (rows صعودی)."""
    hit = None
    for r in rows:
        if r[0] <= t:
            hit = r
        else:
            break
    return hit


def _okx_data(get, path, params):
    j = get(OKX + path, params)
    if str(j.get("code")) != "0":
        if str(j.get("code")) == "50011":             # محدودیتِ نرخ با HTTP 200
            _okx_dead["until"] = time.time() + OKX_COOLDOWN
        raise RuntimeError(f"OKX code={j.get('code')} {j.get('msg') or ''}".strip())
    return j.get("data") or []


def _kc_data(get, path, params=None):
    j = get(KUCOIN_FUT + path, params)
    if str(j.get("code")) != "200000":
        raise RuntimeError(f"KuCoin code={j.get('code')} {j.get('msg') or ''}".strip())
    return j.get("data")


# ── منابع: هر کدام {field: (value, as_of_ms, note_extra)} برمی‌گرداند ──

def _src_spot(sym, get, now_ms):
    raw = get(SPOT + "/api/v3/klines", {"symbol": sym, "interval": "5m", "limit": 290})
    rows = []
    for k in raw:
        t = int(k[0])
        if t + BAR_MS > now_ms:
            continue                                  # کندلِ باز
        qv, tbq = float(k[7]), float(k[10])
        rows.append((t, tbq, qv - tbq, float(k[4])))
    rows.sort(key=lambda r: r[0])
    rows = rows[-288:]
    if not rows:
        raise RuntimeError("کندلِ بسته‌ای برنگشت")
    end = rows[-1][0] + BAR_MS
    out = {}
    for name, hours in (("spot_taker_1h", 1), ("spot_taker_4h", 4), ("spot_taker_24h", 24)):
        out[name] = (_window_flow(rows, end, hours * HOUR_MS), end, None)
    day = [r for r in rows if r[0] >= end - 24 * HOUR_MS]
    cvd, acc = [], 0.0
    for r in day:
        acc += r[1] - r[2]
        cvd.append(round(acc, 0))
    out["spot_cvd_24h"] = ({"t": [r[0] + BAR_MS for r in day], "cvd": cvd, "close": [_sig(r[3]) for r in day],
                            "last": cvd[-1] if cvd else None}, end, None)
    return out


def _src_okx_taker(sym, get, now_ms):
    data = _okx_data(get, "/api/v5/rubik/stat/taker-volume",
                     {"ccy": sym[:-4], "instType": "CONTRACTS", "period": "5m"})
    rows = sorted(((int(r[0]), float(r[2]), float(r[1])) for r in data), key=lambda r: r[0])   # [ts, sell, buy]
    rows = rows[:-1]                                  # جدیدترین ردیف ناقص است و بعداً اصلاح می‌شود
    rows = [r for r in rows if r[0] + BAR_MS <= now_ms]
    if not rows:
        raise RuntimeError("ردیفِ کاملی از حجمِ تیکرِ OKX نماند")
    end = rows[-1][0] + BAR_MS
    return {"okx_taker_1h": (_window_flow(rows, end, HOUR_MS), end, None),
            "okx_taker_4h": (_window_flow(rows, end, 4 * HOUR_MS), end, None)}


def _src_okx_lsr(sym, get, now_ms):
    data = _okx_data(get, "/api/v5/rubik/stat/contracts/long-short-account-ratio", {"ccy": sym[:-4], "period": "5m"})
    rows = sorted(((int(r[0]), float(r[1])) for r in data), key=lambda r: r[0])
    if not rows:
        raise RuntimeError("نسبتِ لانگ/شورتِ OKX خالی است")
    t, ratio = rows[-1]
    ago1, ago24 = _at_or_before(rows, t - HOUR_MS), _at_or_before(rows, t - 24 * HOUR_MS)
    val = {"ratio": _r(ratio, 4), "long_pct": _r(100.0 * ratio / (1.0 + ratio), 2) if ratio >= 0 else None,
           "chg_1h": _r(ratio - ago1[1], 4) if ago1 else None, "chg_24h": _r(ratio - ago24[1], 4) if ago24 else None}
    return {"okx_lsr": (val, t, None)}


def _src_okx_oi(sym, get, now_ms):
    data = _okx_data(get, "/api/v5/rubik/stat/contracts/open-interest-volume", {"ccy": sym[:-4], "period": "5m"})
    rows = sorted(((int(r[0]), float(r[1]), float(r[2])) for r in data), key=lambda r: r[0])   # [ts, oi, vol] دلار
    if not rows:
        raise RuntimeError("موقعیت‌های بازِ OKX خالی است")
    t, oi, _ = rows[-1]

    def chg(h):
        r = _at_or_before(rows, t - h * HOUR_MS)
        return _r(100.0 * (oi / r[1] - 1.0), 3) if r and r[1] > 0 else None

    done = [r for r in rows if r[0] + BAR_MS <= now_ms]       # حجمِ ردیفِ باز (هنوز بسته‌نشده) ناقص است
    vol24 = None
    if done:
        vend = done[-1][0] + BAR_MS
        vol24 = sum(r[2] for r in done if r[0] >= vend - 24 * HOUR_MS)
    val = {"oi_usd": _r(oi, 0), "chg_1h_pct": chg(1), "chg_24h_pct": chg(24), "vol_24h_usd": _r(vol24, 0)}
    return {"okx_oi": (val, t, None)}


def _src_kc_contract(sym, get, now_ms):
    csym = CONTRACT[sym]
    d = _kc_data(get, f"/api/v1/contracts/{csym}")
    mult = _f(d.get("multiplier")) or MULT_FALLBACK.get(csym)
    if d.get("multiplier"):
        _mult[csym] = mult
    mark, index = _f(d.get("markPrice")), _f(d.get("indexPrice"))
    oi = _f(d.get("openInterest"))
    oi_val = None
    if oi is not None:
        oi_val = {"lots": oi, "coin": _r(oi * mult, 4), "usdt": _r(oi * mult * mark, 0) if mark else None}
    rate, pred = _f(d.get("fundingFeeRate")), _f(d.get("predictedFundingFeeRate"))
    gran = _f(d.get("currentFundingRateGranularity")) or _f(d.get("fundingRateGranularity"))
    fund = {"current_bp": _r(rate * 1e4, 3) if rate is not None else None,
            "predicted_bp": _r(pred * 1e4, 3) if pred is not None else None,
            "interval_h": _r(gran / HOUR_MS, 2) if gran else None,
            "next_at": int(d["nextFundingRateDateTime"]) if d.get("nextFundingRateDateTime") else None}
    fund_note = "کوکوین نرخِ پیش‌بینی را خالی برگرداند" if pred is None else None
    prem = None
    if mark and index:
        prem = {"bp": _r((mark / index - 1.0) * 1e4, 2), "mark": mark, "index": index}
    return {"kc_open_interest": (oi_val, now_ms, None), "kc_funding": (fund, now_ms, fund_note),
            "kc_premium_bp": (prem, now_ms, None)}


def book_depth(bids, asks, mult, bands=BOOK_BANDS):
    """ارزشِ (USDT) سفارش‌های خرید/فروش در ±band از قیمتِ میانه و عدمِ توازنِ (bid−ask)/(bid+ask).

    bids/asks: [[price, size_lots], ...] با هر ترتیبی. ``covered_*`` نادرست یعنی دفترِ برگشتی به لبهٔ باند نرسید."""
    bids = [(float(p), float(s)) for p, s, *_ in bids]
    asks = [(float(p), float(s)) for p, s, *_ in asks]
    if not bids or not asks:
        raise RuntimeError("دفترِ سفارش یک سمتِ خالی دارد")
    best_bid, best_ask = max(p for p, _ in bids), min(p for p, _ in asks)
    mid = (best_bid + best_ask) / 2.0
    out = {"mid": _sig(mid, 8), "spread_bp": _r((best_ask - best_bid) / mid * 1e4, 3)}
    for b in bands:
        tag = f"{b * 100:g}".replace(".", "")          # 0.5% -> "05"، 1% -> "1"
        lo, hi = mid * (1 - b), mid * (1 + b)
        bid_n = sum(p * s * mult for p, s in bids if p >= lo)
        ask_n = sum(p * s * mult for p, s in asks if p <= hi)
        tot = bid_n + ask_n
        out[f"bid_{tag}"] = _r(bid_n, 0)
        out[f"ask_{tag}"] = _r(ask_n, 0)
        out[f"imb_{tag}"] = _r((bid_n - ask_n) / tot, 4) if tot > 0 else None
        out[f"covered_{tag}"] = bool(min(p for p, _ in bids) <= lo and max(p for p, _ in asks) >= hi)
    return out


def _src_kc_book(sym, get, now_ms):
    csym = CONTRACT[sym]
    d = _kc_data(get, "/api/v1/level2/snapshot", {"symbol": csym})
    mult = _mult.get(csym) or MULT_FALLBACK[csym]
    val = book_depth(d.get("bids") or [], d.get("asks") or [], mult)
    ts = d.get("ts")
    as_of = int(ts) // 1_000_000 if ts and int(ts) > 1e15 else (int(ts) if ts else now_ms)   # ts کوکوین نانوثانیه است
    note = None if val["covered_1"] else "دفترِ برگشتی به ±۱٪ نرسید؛ عدد کمتر از واقع است"
    return {"kc_book": (val, as_of, note)}


def _src_kc_funding(sym, get, now_ms):
    csym = CONTRACT[sym]
    data = _kc_data(get, "/api/v1/contract/funding-rates",
                    {"symbol": csym, "from": now_ms - 4 * 24 * HOUR_MS, "to": now_ms}) or []
    rows = sorted(((int(r["timepoint"]), float(r["fundingRate"])) for r in data), key=lambda r: -r[0])[:FUNDING_ROWS]
    if not rows:
        raise RuntimeError("تاریخچهٔ فاندینگِ کوکوین خالی است")
    bps = [r[1] * 1e4 for r in rows]
    val = {"rows": [{"t": t, "bp": _r(b, 3)} for (t, _), b in zip(rows, bps)],
           "last_bp": _r(bps[0], 3), "mean_bp": _r(sum(bps) / len(bps), 3),
           "persist": _r(sum((b > 0) - (b < 0) for b in bps) / len(bps), 3), "n": len(rows)}
    note = None if len(rows) == FUNDING_ROWS else f"فقط {len(rows)} تسویه در ۴ روزِ اخیر"
    return {"kc_funding_hist": (val, rows[0][0], note)}


SOURCE_FUNCS = {"binance_spot": _src_spot, "okx_taker": _src_okx_taker, "okx_lsr": _src_okx_lsr,
                "okx_oi": _src_okx_oi, "kucoin_contract": _src_kc_contract, "kucoin_book": _src_kc_book,
                "kucoin_funding": _src_kc_funding}
_OKX_SOURCES = ("okx_taker", "okx_lsr", "okx_oi")


def _err_text(e):
    return f"{type(e).__name__}: {e}"[:240]


def _run_source(src, sym, get, now_ms):
    """(src, {field: (value, as_of, note)} یا None، متنِ خطا یا None) — هرگز خطا بالا نمی‌دهد."""
    try:
        return src, SOURCE_FUNCS[src](sym, get, now_ms), None
    except Exception as e:  # noqa: BLE001 — خرابیِ یک منبع فقط فیلدهای همان منبع را خالی می‌کند
        return src, None, _err_text(e)


def _assemble(sym, results, now_ms, elapsed_ms=None):
    fields, errors = {}, {}
    got = {}
    for src, vals, err in results:
        if err:
            errors[src] = err
        got[src] = vals or {}
    for name, (src, desc) in FIELDS.items():
        val, as_of, extra = got.get(src, {}).get(name, (None, None, None))
        f = {"value": val, "source": SOURCES[src], "as_of": as_of, "note": desc + (f" — {extra}" if extra else "")}
        if src in errors:
            f["error"] = errors[src]
        fields[name] = f
    return {"symbol": sym, "display_only": True, "note": NOTE_FA, "as_of": now_ms,
            "fields": fields, "errors": errors, "elapsed_ms": elapsed_ms}


def _collect(symbols, get, now_ms):
    t0 = time.time()
    # منابعِ غیر-OKX اول، تا کارگرها پشتِ محدودکنندهٔ نرخِ OKX نمانند
    order = [s for s in SOURCE_FUNCS if s not in _OKX_SOURCES] + list(_OKX_SOURCES)
    tasks = [(src, sym) for src in order for sym in symbols]
    with ThreadPoolExecutor(max_workers=min(16, max(1, len(tasks)))) as ex:
        futs = [(sym, ex.submit(_run_source, src, sym, get, now_ms)) for src, sym in tasks]
        per = {sym: [] for sym in symbols}
        for sym, fu in futs:
            per[sym].append(fu.result())
    elapsed = int((time.time() - t0) * 1000)
    return {sym: _assemble(sym, per[sym], now_ms, elapsed) for sym in symbols}


def _empty(sym, now_ms, err):
    return _assemble(sym, [(src, None, err) for src in SOURCE_FUNCS], now_ms)


def snapshot(sym, get=None, now=None):
    """وضعیتِ بازارِ یک ارز (بی‌کش). هر فیلد: {value, source, as_of, note[, error]}؛ خطا بالا نمی‌رود."""
    now = time.time() if now is None else now
    now_ms = int(now * 1000)
    try:
        if sym not in CONTRACT:
            raise ValueError(f"ارزِ ناشناخته: {sym}")
        return _collect([sym], get or _http_get, now_ms)[sym]
    except Exception as e:  # noqa: BLE001
        return _empty(sym, now_ms, _err_text(e))


def snapshot_all(get=None, now=None, force=False):
    """{sym: snapshot} برای پنج ارز، موازی، ۶۰ ثانیه کش. هرگز خطا بالا نمی‌دهد."""
    now = time.time() if now is None else now
    with _cache_lock:
        if not force and _cache["data"] is not None and 0 <= now - _cache["ts"] < CACHE_TTL:
            return dict(_cache["data"])
    with _flight:
        with _cache_lock:
            if not force and _cache["data"] is not None and 0 <= now - _cache["ts"] < CACHE_TTL:
                return dict(_cache["data"])
        syms = [s for s in watchlist.SYMBOLS if s in CONTRACT]
        now_ms = int(now * 1000)
        try:
            data = _collect(syms, get or _http_get, now_ms)
        except Exception as e:  # noqa: BLE001
            data = {s: _empty(s, now_ms, _err_text(e)) for s in syms}
        with _cache_lock:
            _cache.update(ts=now, data=data)
        return dict(data)


def peek(now=None):
    """آخرین snapshot_all بدونِ رفتن به شبکه: (سنِ ثانیه، {sym: snap}) یا (None, None).

    اولین snapshot_all به‌خاطرِ محدودیتِ نرخِ OKX حدودِ ۷ ثانیه طول می‌کشد؛ endpoint می‌تواند این را برگرداند
    و تازه‌سازی را به حلقهٔ پس‌زمینه (tick) بسپارد."""
    now = time.time() if now is None else now
    with _cache_lock:
        if _cache["data"] is None:
            return None, None
        return round(now - _cache["ts"], 1), dict(_cache["data"])


# ── ثبت برای تاریخچهٔ هم‌منبع ──

def _compact(snap, now_ms):
    fields = {}
    for name, f in (snap.get("fields") or {}).items():
        val = f.get("value")
        if name == "spot_cvd_24h" and isinstance(val, dict):
            cvd = val.get("cvd") or []
            val = {"last": val.get("last"), "chg_1h": (cvd[-1] - cvd[-13]) if len(cvd) >= 13 else None}
        elif name == "kc_funding_hist" and isinstance(val, dict):
            val = {k: val.get(k) for k in ("last_bp", "mean_bp", "persist", "n")}
        fields[name] = {"value": val, "as_of": f.get("as_of")}
    return {"v": SCHEMA, "ts": now_ms, "sym": snap.get("symbol"), "display_only": True,
            "fields": fields, "errors": snap.get("errors") or {}}


def _path(sym):
    return os.path.join(LIVE_POS_DIR, f"{sym}.jsonl")


def _last_line_ts(path):
    """زمانِ (ثانیه) آخرین خطِ سالمِ فایل — تا پس از راه‌اندازیِ دوباره هم فاصلهٔ ۵ دقیقه رعایت شود."""
    try:
        with open(path, "rb") as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            f.seek(max(0, size - 8192))
            tail = f.read().decode("utf-8", "ignore").splitlines()
    except OSError:
        return None
    for line in reversed(tail):
        try:
            return float(json.loads(line)["ts"]) / 1000.0
        except Exception:  # noqa: BLE001 — خطِ نیمه‌کاره یا خراب
            continue
    return None


def record(snapshots, now=None):
    """برای هر ارز یک خطِ JSON در live_pos/<SYM>.jsonl — حداکثر یک‌بار در ۵ دقیقه. نمادهای نوشته‌شده را برمی‌گرداند.

    snapshots: خروجیِ snapshot_all ({sym: snap})، یک snapshot، یا فهرستی از آن‌ها."""
    now = time.time() if now is None else now
    if isinstance(snapshots, dict):
        items = [snapshots] if "fields" in snapshots else list(snapshots.values())
    else:
        items = list(snapshots or [])
    written = []
    with _rec_lock:
        for snap in items:
            sym = (snap or {}).get("symbol") if isinstance(snap, dict) else None
            if not sym or not re.fullmatch(r"[A-Z0-9]{2,20}", sym):
                continue
            if not any(f.get("value") is not None for f in (snap.get("fields") or {}).values()):
                continue                              # همهٔ منابع خطا دادند — چیزی برای ثبت نیست
            path = _path(sym)
            last = _last_rec.get(sym)
            if last is None:
                last = _last_line_ts(path)
            if last is not None and 0 <= now - last < RECORD_EVERY:
                continue
            try:
                os.makedirs(LIVE_POS_DIR, exist_ok=True)
                line = json.dumps(_compact(snap, int(now * 1000)), ensure_ascii=False, separators=(",", ":"))
                with open(path, "a", encoding="utf-8") as f:
                    f.write(line + "\n")
            except (OSError, TypeError, ValueError):
                continue
            _last_rec[sym] = now
            written.append(sym)
    return written


def tick(get=None, now=None):
    """snapshot_all و سپس record — برای حلقهٔ پس‌زمینه؛ خطا بالا نمی‌دهد."""
    snaps = snapshot_all(get=get, now=now)
    try:
        record(snaps, now=now)
    except Exception:  # noqa: BLE001
        pass
    return snaps
