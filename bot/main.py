# -*- coding: utf-8 -*-
"""سرور ربات پایش ۱۵ ارز برتر — FastAPI
اجرا:  python main.py   سپس   http://127.0.0.1:8787
⚠️ فقط تحلیل و پوزیشن دمو (شبیه‌سازی) — به هیچ صرافی واقعی وصل نمی‌شود."""
import os
import sys
import time
import threading
from concurrent.futures import ThreadPoolExecutor

# کنسول ویندوز پیش‌فرض cp1252 است — خروجی را UTF-8 کن تا متن فارسی لاگ‌ها خطا ندهد
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001, silent-ok — پیش از import log اجرا می‌شود؛ ارجاع به log اینجا NameError می‌داد
        pass

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

import numpy as np

import log
import market
import candidates
import engine
import features
import universe
import paper
import paths
import broker
import calib
import shadow
import autotrader
import advisor
import gates
import trend
import trend_exec
import momentum
import watchlist
import kucoin_desk
import notify
import decision
import tf_spec
from app_meta import APP_VERSION, AI_CORE_VERSION, RELEASE_DATE
from datetime import datetime, timezone

_mstate_cache: dict = {}   # (tf, tk) -> (ts, سبدِ کامل؟, {"breadth","dom","ethbtc"})
_susp_cache = {"ts": 0.0, "map": {}}

# ── تک‌پروازیِ کمک‌تابع‌ها: پنج نخِ overview روی کشِ سرد هر کدام همین‌ها را از نو می‌ساختند
# (تاریخچهٔ PAXG با صفحه‌بندی، دفترِ سایه، edge_book). حالا اولی می‌سازد و بقیه همان را می‌خوانند.
_once_locks: dict = {}
_once_guard = threading.Lock()


def _once(key):
    with _once_guard:
        lk = _once_locks.get(key)
        if lk is None:
            lk = _once_locks[key] = threading.Lock()
        return lk


def _last_closed_open(tf, now=None):
    """زمانِ بازِ آخرین کندلِ بسته‌شدهٔ ``tf`` از روی ساعت (بی شبکه)."""
    bar = tf_spec.bar_ms(tf)
    now_ms = int((time.time() if now is None else now) * 1000)
    return (now_ms // bar) * bar - bar


def _basket_klines(tf, tk):
    """کندل‌های سبدِ مرجع که کندلِ ``tk`` را دارند — با get_klines، نه کشِ خام.

    قبلاً ``get_klines_cached`` هر چه آخرین‌بار در کش بود را می‌داد: درونِ تحلیلِ BTC، کندل‌های
    BTC تازه و بقیه یک کندل کهنه بودند (نخ‌هایشان پشتِ همان تحلیلِ BTC منتظر بودند). اگر صرافی
    کندلِ ``tk`` را هنوز منتشر نکرده بود (کش دریافتِ بی آن را تا KLINE_TTL نگه می‌دارد)، یک‌بار مستقیم تازه می‌شود.
    """
    out = {}
    for s in calib.MARKET_BASKET:
        try:
            kl = _klines_at(s, tf, tk)
            if kl:
                out[s] = kl
        except Exception:  # noqa: BLE001 — ارزِ بی‌کندل از سبد کم می‌شود ⇒ دامیننس خنثی
            log.exc(f"market-state klines {s} {tf}")
    return out


def _klines_at(symbol, tf, tk):
    """کندل‌های بستهٔ ``symbol`` که کندلِ ``tk`` را دارند، وگرنه None.
    اگر صرافی کندلِ ``tk`` را هنوز منتشر نکرده بود و کش همان دریافتِ بی آن را بدهد، یک‌بار مستقیم تازه می‌شود."""
    kl = market.get_klines(symbol, tf)
    if kl and kl["t"] and int(kl["t"][-1]) < tk:
        kl = market._fetch_klines(symbol, tf, 420)
    return kl if kl and tk in kl["t"] else None


def _market_state(tf, tk=None):
    """وضعیتِ کلِ بازار (مومنتوم دامیننس BTC و ETH/BTC) در **کندلِ تحلیل‌شده** ``tk``.

    یک تعریف با آموزش (calib-F1): ``calib.market_state_at`` روی سبدِ ثابتِ پنج‌ارزی
    (``calib.MARKET_BASKET`` = watchlist)، فقط وقتی همهٔ سبد همان کندل را دارد. کش با کلیدِ
    (tf, tk) است، نه TTLِ ساعتی — حالتِ یک کندل برای همهٔ ارزهای همان کندل یکی است.
    پهنای بازار عمداً خنثی (۰) است: نه ویژگیِ مدل است (features.DISABLED) و نه ورودیِ متا-گیت
    تغییر می‌کند (با پنج ارز هم قبلاً همیشه ۰ بود).

    بی ``tk`` (رژیمِ کلان، overview، یادداشتِ پوزیشن‌ها) فقط پهنا خوانده می‌شود که همیشه ۰ است:
    حالتِ خنثی بی واکشیِ کندل‌های سبد. دامیننس و ETH/BTC فقط در کندلِ تحلیل‌شده ساخته می‌شوند.
    """
    if tf not in tf_spec.MODEL_TFS or tk is None:
        return {"breadth": 0.0, "dom": 0.0, "ethbtc": 0.0}      # 5m مدلی ندارد / بی کندلِ تحلیل
    tk = int(tk)
    key = (tf, tk)
    hit = _mstate_cache.get(key)
    if hit and (hit[1] or time.time() - hit[0] < 60):
        return hit[2]
    with _once(("mstate", tf)):
        hit = _mstate_cache.get(key)
        if hit and (hit[1] or time.time() - hit[0] < 60):
            return hit[2]
        return _market_state_build(tf, tk)


def _market_state_build(tf, tk):
    out = {"breadth": 0.0, "dom": 0.0, "ethbtc": 0.0}
    complete = False
    try:
        hists = _basket_klines(tf, tk)
        complete = len(hists) == len(calib.MARKET_BASKET)
        out = calib.market_state_at(hists, tk)
    except Exception:  # noqa: BLE001
        log.exc()
    # سبدِ ناقص (خطای شبکه) فقط ۶۰ ثانیه کش می‌شود تا در همان کندل دوباره ساخته شود
    for k in [k for k in list(_mstate_cache) if k[0] == tf and k[1] < tk]:   # کلیدهای کندل‌های قبل
        _mstate_cache.pop(k, None)
    _mstate_cache[(tf, tk)] = (time.time(), complete, out)
    return out


def _model_era(tf=None):
    """لحظهٔ تولدِ مدلِ فعلی (trained_ts) — آمارِ زنده فقط از این به بعد شمرده می‌شود تا
    گناهانِ مدلِ قبلی (مثلاً کالیبراسیونِ آلوده) به پای مدلِ تازه نوشته نشود."""
    try:
        tab = calib.load() or {}
        tss = []
        for tf_, d in (tab.get("tfs") or {}).items():
            if tf is not None and tf_ != tf:
                continue
            for k in ("model", "dir_model", "edge_model"):
                ts_ = (d.get(k) or {}).get("trained_ts")
                if ts_:
                    tss.append(float(ts_))
        return min(tss) if tss else None
    except Exception:  # noqa: BLE001
        return None


# ── بازخوردِ زنده: تعلیقِ خودکار (ستاپ/تایم‌فریم/ترکیب)، جیبِ لبه و هشدارِ drift ──
# همه از signals_log.json (shadow) خوانده می‌شدند، ولی از ۱۶ جولای هیچ کدِ تولیدی در آن نمی‌نویسد
# (shadow.log_signal فقط در تست‌ها صدا زده می‌شود؛ دفترِ جایگزین candidates.jsonl است). پس این «ترمزها»
# هرگز فعال نمی‌شدند، در حالی که UI ترمزِ خودکار را القا می‌کرد. احیایشان تصمیمِ زنده را از کارنامهٔ
# بازپخش (decision.replay، بی‌تعلیق) جدا می‌کرد؛ پس عمداً خاموش‌اند و API/UI صریحاً «غیرفعال» می‌گویند.
LIVE_FEEDBACK_ACTIVE = False
LIVE_FEEDBACK_NOTE_FA = ("تعلیقِ خودکارِ ستاپ/تایم‌فریم/ترکیب، جیبِ لبه و هشدارِ افتِ عملکرد غیرفعال‌اند: "
                         "دفترِ سایه‌ای که از آن تغذیه می‌شدند از ژوئیه ثبتی ندارد. تصمیم‌ها دقیقاً همان قاعدهٔ "
                         "کارنامه‌اند و هیچ ترمزِ خودکاری رویشان نیست؛ عملکردِ زنده در کارنامهٔ رو-به-جلوی "
                         "میزِ تصمیم است.")


def _live_feedback():
    """وضعیتِ بازخوردِ زنده برای API/UI — صریح، نه ضمنی."""
    state = "active" if LIVE_FEEDBACK_ACTIVE else "inactive"
    return {"active": LIVE_FEEDBACK_ACTIVE, "suspension": state, "edge_pockets": state, "drift": state,
            "source": "signals_log.json (shadow.log_signal) — بدونِ نویسندهٔ تولیدی از ۱۶ جولای",
            "live_ledger": "decision_ledger.jsonl (/api/decisions → live_overall)",
            "note_fa": None if LIVE_FEEDBACK_ACTIVE else LIVE_FEEDBACK_NOTE_FA}


def _suspended_setups():
    """تعلیق سراسری فقط برای ستاپ‌های واقعاً سمی.
    تعلیقِ خفیفِ سراسری (مثل zx با −۰٫۰۱R) جیب‌های سوددهٔ تایم‌فریم‌محور را می‌کشت."""
    if not LIVE_FEEDBACK_ACTIVE:
        return {}                                  # منبع تغذیه نمی‌شود — هیچ تعلیقی (تصمیم = کارنامه)
    if time.time() - _susp_cache["ts"] < 120:
        return _susp_cache["map"]
    with _once("susp"):
        if time.time() - _susp_cache["ts"] < 120:
            return _susp_cache["map"]
        return _suspended_setups_build()


def _suspended_setups_build():
    m = {}
    try:
        st = shadow.stats(None, days=30, since_ts=_model_era())
        for k, v in (st.get("by_setup") or {}).items():
            if v.get("n", 0) >= 20 and v.get("avg_r") is not None and v["avg_r"] < -0.12:
                m[k] = v.get("avg_r")
    except Exception:  # noqa: BLE001
        log.exc()
    _susp_cache.update(ts=time.time(), map=m)
    return m


_tf_susp_cache = {"ts": 0.0, "map": {}}
def _live_edge_book():
    """جیب‌های سودده و ترکیب‌های معلق — معیار LCB/نیمه‌ها در edge_book."""
    if not LIVE_FEEDBACK_ACTIVE:
        return {}, {}
    try:
        import edge_book
        with _once("edge_book"):                   # edge_book کشِ ۹۰ثانیه‌ای دارد ولی قفل ندارد
            pockets, suspended, _ = edge_book.refresh()
        return pockets, suspended
    except Exception:  # noqa: BLE001
        return {}, {}


def _suspended_tfs():
    """تایم‌فریم‌هایی که بازده خالص زندهٔ ۳۰روزه‌شان با n کافی مثبت نیست — معلق تا بهبود.
    اگر همان TF جیب سوددهٔ ترکیبی داشته باشد، تعلیقِ کل TF اعمال نمی‌شود."""
    if not LIVE_FEEDBACK_ACTIVE:
        return {}
    if time.time() - _tf_susp_cache["ts"] < 120:
        return _tf_susp_cache["map"]
    with _once("tf_susp"):
        if time.time() - _tf_susp_cache["ts"] < 120:
            return _tf_susp_cache["map"]
        return _suspended_tfs_build()


def _suspended_tfs_build():
    m = {}
    pockets, _ = _live_edge_book()
    pocket_tfs = {p["tf"] for p in pockets.values()}
    try:
        for tf in tf_spec.MODEL_TFS:                 # 5m سایه/جیب ندارد (فقط ماتریسِ تصمیم)
            if tf in pocket_tfs:
                continue
            st = shadow.stats(tf, days=30, since_ts=_model_era(tf))
            if st.get("n", 0) >= 30 and st.get("avg_r") is not None and st["avg_r"] <= 0.0:
                m[tf] = st["avg_r"]
    except Exception:  # noqa: BLE001
        log.exc()
    _tf_susp_cache.update(ts=time.time(), map=m)
    return m

HTF_OF = tf_spec.HTF_OF   # 5m→1h، 15m/1h→4h، 4h→1d
_fz_cache: dict = {}      # sym -> (ts, {"z", "persist", "crowded_long", "crowded_short"})
_fz_pending: set = set()
_oi_cache: dict = {}      # sym -> (ts, oi_stats)
_oi_pending: set = set()
_btc_macro_cache = {"ts": 0.0, "data": None}
_macro_cache: dict = {}   # (tf, tk) -> (ts, کندلِ tk بود؟, {"gold": x})


def _rs_rank(tf):
    """رتبهٔ قدرتِ نسبی — عمداً خنثی (۰٫۵ برای همه)، یک تعریف با آموزش (calib-F1).

    تعریفِ قبلی رتبه بینِ ۱۰۰ ارزِ برتر با کفِ جمعیتِ ۴۰ بود، ولی برنامه زنده فقط پنج ارزِ
    watchlist را می‌خواند؛ پس اجرا همیشه ۰٫۵ می‌گرفت و آموزش مقدارِ واقعی. حالا آموزش هم خنثی
    است (``features.DISABLED``). رتبه‌گرفتن بینِ پنج ارز بندِ «قدرت نسبی»ِ متا-گیت را هم روشن
    می‌کرد (آستانه‌هایش برای ۱۰۰ ارز تنظیم شده) — آن تصمیمِ محصولی جداست.
    """
    return {}


def _macro(tf, tk=None):
    """مومنتوم زندهٔ طلا (PAXG) در **کندلِ تحلیل‌شده** ``tk`` — همان منبع و نرمال‌سازیِ آموزش.

    قبلاً از ``get_history`` (کشِ دیسکِ ۲۴ساعته + کشِ حافظهٔ ۱۵دقیقه) آخرین مقدار برداشته می‌شد:
    طلای زنده تا یک روز کهنه بود و روی 1h همبستگی‌اش با تعریفِ آموزش ~۰٫۴ (calib-F7). حالا
    کندل‌های PAXG با همان تازگیِ کندلِ بسته‌ی تایم‌فریم (get_klines) و مقدار در خودِ ``tk``.

    شاخصِ دلار حذف شد: فایلِ dxy_daily.json هرگز پر نشد و ویژگی در تمامِ آموزش
    ثابتِ ۰ بود؛ زنده‌کردنش بعداً یعنی دادنِ ورودیِ ندیده به مدل.
    """
    if tf not in tf_spec.MODEL_TFS:
        # طلا فقط ورودیِ مدل‌های calib است و 5m مدلی ندارد — واکشیِ PAXG بی‌فایده است
        return {"gold": 0.0}
    tk = int(tk) if tk is not None else _last_closed_open(tf)
    key = (tf, tk)
    hit = _macro_cache.get(key)
    if hit and (hit[1] or time.time() - hit[0] < 60):
        return hit[2]
    with _once(("macro", tf)):
        hit = _macro_cache.get(key)
        if hit and (hit[1] or time.time() - hit[0] < 60):
            return hit[2]
        return _macro_build(tf, tk)


def _macro_build(tf, tk):
    out = {"gold": 0.0}
    found = False
    try:
        gk = _klines_at("PAXGUSDT", tf, tk)
        if gk:
            gmap = calib.mom_norm_map(gk["t"], gk["c"])
            found = tk in gmap
            out["gold"] = float(gmap.get(tk, 0.0))          # همان gold_map.get(ts, 0.0) ِ آموزش
    except Exception:  # noqa: BLE001
        log.exc()
    for k in [k for k in list(_macro_cache) if k[0] == tf and k[1] < tk]:   # کلیدهای کندل‌های قبل
        _macro_cache.pop(k, None)
    _macro_cache[(tf, tk)] = (time.time(), found, out)   # بی‌کندلِ tk فقط ۶۰ ثانیه
    return out


def _btc_macro():
    """رژیم کلان BTC روی روزانه + پهنای 1d — کش ۱۵ دقیقه‌ای."""
    if _btc_macro_cache["data"] and time.time() - _btc_macro_cache["ts"] < 900:
        return _btc_macro_cache["data"]
    with _once("btc_macro"):
        if _btc_macro_cache["data"] and time.time() - _btc_macro_cache["ts"] < 900:
            return _btc_macro_cache["data"]
        return _btc_macro_build()


def _btc_macro_build():
    import meta_gate
    now = time.time()
    data = {"regime": "chop", "btc_vs_sma100": 0.0, "btc_vs_sma200": 0.0, "breadth": 0.0}
    try:
        br = (_market_state("1d") or {}).get("breadth", 0.0)
        kl = market.get_klines_cached("BTCUSDT", "1d") or market.get_klines("BTCUSDT", "1d")
        data = meta_gate.btc_macro_regime(kl["c"], br)
    except Exception:  # noqa: BLE001
        log.exc()
    _btc_macro_cache.update(ts=now, data=data)
    return data


def _fetch_funding(symbol):
    blob = {"z": 0.0, "persist": 0, "crowded_long": False, "crowded_short": False}
    try:
        import meta_gate
        rows = market.get_funding_history(symbol)
        st = meta_gate.funding_crowd_stats(rows)
        blob = {
            "z": float(st.get("funding_z") or 0.0),
            "persist": int(st.get("persist") or 0),
            "crowded_long": bool(st.get("crowded_long")),
            "crowded_short": bool(st.get("crowded_short")),
        }
    except Exception:  # noqa: BLE001
        try:
            rows = market.funding_z_map(symbol)
            if rows:
                blob["z"] = float(rows[-1][1])
        except Exception:  # noqa: BLE001
            log.exc()
    _fz_cache[symbol] = (time.time(), blob)
    _fz_pending.discard(symbol)


def _symbol_cost(symbol):
    """هزینهٔ واقعیِ رفت‌وبرگشت (٪) بر اساس نقدینگی — با فرضِ سبکِ اجرای فعلی:
    ورودِ لیمیتِ صبور (maker ~۰.۰۲٪، بدونِ لغزش) + خروجِ حد/هدف (بدترین حالت taker ~۰.۰۵٪).
    فرضِ قبلی (taker دوطرفه + لغزشِ کامل) EV همهٔ سیگنال‌ها را ~۰.۱٪ اضافه جریمه می‌کرد."""
    try:
        _, tickers = market.get_top_symbols(TOP_N)
        qv = (tickers.get(symbol) or {}).get("quote_vol", 0.0)
    except Exception:  # noqa: BLE001
        qv = 0.0
    if qv >= 5e8:
        return 0.08
    if qv >= 1e8:
        return 0.11
    if qv >= 2e7:
        return 0.18
    return 0.30


def _cost_basis():
    """مبنای هزینهٔ هر دفتر/عدد — برچسبِ صریح کنارِ هر API (LP-10).

    عمداً یکی نشده‌اند: کارنامه و دفترِ تصمیمِ قاعده با کارمزدِ ثابتِ ``decision.COST_PCT`` بازپخش/داوری
    می‌شوند؛ دفترِ کاندیدها (که اثباتِ سایه با آزمونِ منجمد مقایسه‌اش می‌کند) مدلِ هزینهٔ پیش‌ثبت‌شده
    (``costs.event_cost_pct``) را دارد و فقط جزءِ ردهٔ نقدشوندگی‌اش را. بک‌تستِ کوچکِ هر نماد همان رده را
    از ``extras["cost"]`` می‌گیرد (همان عدد به مدل‌ها هم می‌رود، پس اینجا فقط برچسب می‌خورد).
    """
    return {
        "decision_ledger": {"model": "flat", "cost_pct": decision.COST_PCT,
                            "maker_cost_pct": decision.MAKER_COST_PCT,
                            "desc_fa": "کارمزدِ ثابتِ رفت‌وبرگشت (تیکرِ فیوچرزِ کوکوین ×۲ + لغزش) — کارنامه و دفترِ تصمیم"},
        "candidates": dict(candidates.COST_BASIS),
        "symbol_backtest": {"model": "tier",
                            "desc_fa": "ردهٔ نقدشوندگیِ همان نماد (۰٫۰۸ تا ۰٫۳۰٪، ورودِ میکر) — «cost»ِ هر ردیف"},
    }


def _funding_info(symbol):
    """غیرمسدودکننده: اگر کش نبود، واکشی به پس‌زمینه می‌رود."""
    hit = _fz_cache.get(symbol)
    if hit and time.time() - hit[0] < 1800:
        return hit[1]
    if symbol not in _fz_pending and len(_fz_pending) < 8:
        _fz_pending.add(symbol)
        threading.Thread(target=_fetch_funding, args=(symbol,), daemon=True).start()
    if hit:
        return hit[1]
    return {"z": 0.0, "persist": 0, "crowded_long": False, "crowded_short": False}


def _funding_z(symbol):
    return float(_funding_info(symbol).get("z") or 0.0)


def _fetch_oi(symbol):
    blob = {"oi": None, "oi_chg_pct": 0.0, "oi_z": 0.0,
            "oi_rising": False, "oi_falling": False, "ok": False}
    try:
        blob = market.oi_crowd_stats(symbol, period="1h")
    except Exception:  # noqa: BLE001
        log.exc()
    _oi_cache[symbol] = (time.time(), blob)
    _oi_pending.discard(symbol)


def _oi_info(symbol):
    hit = _oi_cache.get(symbol)
    if hit and time.time() - hit[0] < 1800:
        return hit[1]
    if symbol not in _oi_pending and len(_oi_pending) < 6:
        _oi_pending.add(symbol)
        threading.Thread(target=_fetch_oi, args=(symbol,), daemon=True).start()
    if hit:
        return hit[1]
    return {"oi": None, "oi_chg_pct": 0.0, "oi_z": 0.0,
            "oi_rising": False, "oi_falling": False, "ok": False}
app = FastAPI(title="CTP Multi-Coin Bot", version=APP_VERSION)
TFS = list(tf_spec.TFS)   # 5m | 15m | 1h | 4h | 1d — ثبتِ واحد: bot/tf_spec.py
TOP_N = 200               # تعداد ارزهای برتر تحت پایش
CALIB_TOP = 100           # استخر آموزش مدل‌ها (پرحجم‌ترین‌ها) — با آموزشِ موازی، داده بیشتر = مدل قوی‌تر
ANALYSIS_TTL = tf_spec.ANALYSIS_TTL
# حداکثر فاصله مجاز قیمت زنده از کندلِ سیگنال (٪) — متناسب با نوسان طبیعی هر تایم‌فریم
DRIFT_CAP = tf_spec.DRIFT_CAP

_an_lock = threading.Lock()
_an_cache: dict = {}      # (symbol, tf) -> (ts, analysis | {"error": str})
_inflight: dict = {}      # (symbol, tf) -> (Event, نخِ محاسبه‌گر) — جلوگیری از محاسبه تکراری هم‌زمان
_pool = ThreadPoolExecutor(max_workers=8)
INFLIGHT_POLL = 5.0       # منتظر هر چند ثانیه زنده‌بودنِ محاسبه‌گر را می‌پاید
INFLIGHT_MAX_WAIT = 300.0  # سقفِ کلِ انتظارِ یک منتظر؛ بعدش خطا برمی‌گردد (نشانگرِ محاسبه‌گر دست نمی‌خورد)
ERROR_TTL = 60.0           # نتیجهٔ خطا فقط ۶۰ ثانیه کش می‌شود، نه تا مرزِ کندل (خطای گذرا روی 1d یک روز می‌ماند)


def _an_fresh(hit, tf, max_age):
    now = time.time()
    res = hit[1]
    if isinstance(res, dict) and "error" in res:
        return now - hit[0] < (min(max_age, ERROR_TTL) if max_age else ERROR_TTL)
    boundary = now - (now % (tf_spec.minutes(tf) * 60))
    # تحلیل تا پایان کندل جاری معتبر است (+۴۵ ثانیه مهلت بعد از بسته‌شدن)
    return hit[0] >= boundary or now - hit[0] < (max_age or 45)


def get_analysis(symbol: str, tf: str, max_age=None):
    """تحلیلِ کش‌شده؛ برای هر کلید فقط یک نخ محاسبه می‌کند و بقیه نتیجهٔ همان را می‌گیرند.

    باگِ قبلی (overview روزانه ۵۱۵ ثانیه): منتظر پس از wait(timeout=180) نشانگرِ inflight را
    برمی‌داشت در حالی که محاسبه‌گر هنوز کار می‌کرد ⇒ محاسبهٔ تکراری؛ و محاسبه‌گرِ اول در پایان
    نشانگرِ **نخِ دوم** را پاک می‌کرد ⇒ نخِ سوم هم از نو شروع می‌کرد. زنجیرهٔ HTF و BTC
    (15m→4h→1d) این را چندبرابر می‌کرد. حالا منتظر تا وقتی محاسبه‌گر زنده است می‌ماند،
    و هر نخ فقط نشانگرِ خودش را برمی‌دارد.
    """
    key = (symbol, tf)
    me = threading.current_thread()
    deadline = time.time() + INFLIGHT_MAX_WAIT      # سقفِ کلِ انتظار برای محاسبه‌گرِ دیگر
    while True:
        with _an_lock:
            hit = _an_cache.get(key)
            if hit and _an_fresh(hit, tf, max_age):
                return hit[1]
            cur = _inflight.get(key)
            if cur is None:
                mine = (threading.Event(), me)
                _inflight[key] = mine
                break                          # این نخ مسئول محاسبه است
        ev_o, owner = cur
        if owner is me:                        # بازگشتِ حلقوی — هرگز نباید رخ دهد؛ قفلِ ابدی نساز
            return {"symbol": symbol, "tf": tf, "error": "بازگشتِ حلقوی در زنجیرهٔ تحلیل"}
        t_wait = time.time()
        timed_out = False
        # نخ دیگری در حال محاسبه همین کلید است
        while not ev_o.wait(max(0.0, min(INFLIGHT_POLL, deadline - time.time()))):
            if not owner.is_alive():
                break
            if time.time() >= deadline:
                timed_out = True
                break
        with _an_lock:
            hit = _an_cache.get(key)
            if hit and (hit[0] >= t_wait or _an_fresh(hit, tf, max_age)):
                return hit[1]                  # نتیجهٔ همان محاسبه‌گر
            if timed_out:
                # محاسبه‌گر زنده ولی کُند است: نشانگرش را برنمی‌داریم و محاسبهٔ تکراری هم نمی‌سازیم؛
                # این خطا کش نمی‌شود و درخواستِ بعدی دوباره نتیجهٔ همان محاسبه‌گر را می‌گیرد.
                return {"symbol": symbol, "tf": tf,
                        "error": f"مهلتِ انتظار برای تحلیلِ {symbol} {tf} ({INFLIGHT_MAX_WAIT:.0f} ثانیه) "
                                 "تمام شد — محاسبه هنوز در جریان است؛ کمی بعد دوباره تلاش کنید"}
            if _inflight.get(key) is cur:      # محاسبه‌گر بی‌نتیجه مُرد — نشانگرِ او را بردار
                _inflight.pop(key, None)
    try:
        res = _compute_analysis(symbol, tf)
        with _an_lock:
            _an_cache[key] = (time.time(), res)
    finally:
        with _an_lock:
            if _inflight.get(key) is mine:     # فقط نشانگرِ خودم
                _inflight.pop(key, None)
        mine[0].set()
    return res


_htf_z_cache: dict = {}   # (symbol, htf) -> (آخرین کندل, {ts: z خام})


def _htf_sign_live(symbol, htf, tk):
    """جهتِ تایم‌بالاتر با **همان** تابعِ آموزش (``calib._htf_sign``) روی zِ خام.

    قبلاً zِ گردشدهٔ خروجیِ تحلیل (۲ رقم؛ z_prev ۳ رقم) با آستانهٔ ±۰٫۳ مقایسه می‌شد و zِ خامِ
    (0.3, 0.305) به 0.30 می‌افتاد: htf_sign یک‌طرفه ±۱→۰ روی ~۰٫۲۴-۰٫۳۳٪ کندل‌ها (calib-F10).
    z از همان کندل‌های تایم‌بالاتر که تحلیلش دید، یک‌بار برای هر کندلِ تازه حساب و کش می‌شود.
    """
    try:
        hk = market.get_klines(symbol, htf)
        last = int(hk["t"][-1])
        hit = _htf_z_cache.get((symbol, htf))
        if hit and hit[0] == last:
            zmap = hit[1]
        else:
            cs = engine.component_series(*[np.asarray(hk[k], float) for k in ("o", "h", "l", "c", "v")])
            zmap = {int(t): float(z) for t, z in zip(hk["t"], cs["z"])}
            _htf_z_cache[(symbol, htf)] = (last, zmap)
        return calib._htf_sign(zmap, tk, htf)
    except Exception:  # noqa: BLE001
        log.exc(f"htf_sign {symbol} {htf}")
        return 0


def _compute_analysis(symbol, tf):
    try:
        btc_z = None
        if symbol != "BTCUSDT":
            btc = get_analysis("BTCUSDT", tf)          # عمق بازگشت حداکثر ۱
            btc_z = btc.get("z") if "error" not in btc else None   # همان متغیر آموزش (z نه score)
        # کندلِ تحلیل‌شده پیش از ورودی‌های مدل: همهٔ زمینه‌ها (فاندینگ، وضعیتِ بازار، طلا) در
        # **همین** کندل (tk = زمانِ بازِ آخرین کندلِ بسته) ساخته می‌شوند، مثلِ آموزش
        kl = market.get_klines(symbol, tf)
        tk, bar_ms = int(kl["t"][-1]), tf_spec.bar_ms(tf)
        htf_sign = 0
        htf = HTF_OF.get(tf)
        if htf:
            ha = get_analysis(symbol, htf)             # زنجیره: 5m→1h→4h→1d→پایان، 15m→4h
            if "error" not in ha and ha.get("zt") is not None:
                # هم‌ترازی دقیق با آموزش: کندلِ بستهٔ *قبل از* سطلِ تایم‌بالاترِ سیگنال (بدون نشتی/کندلِ در حال شکل‌گیری)
                htf_sign = _htf_sign_live(symbol, htf, tk)
        finfo = _funding_info(symbol)
        oinfo = _oi_info(symbol)
        extras = {# ویژگیِ مدل با فرمولِ آموزش ساخته می‌شود، نه با z شلوغیِ متا-گیت
                  "funding_z": features.live_funding_z(symbol, tk, bar_ms),
                  "funding_crowd_z": float(finfo.get("z") or 0.0),
                  "funding_persist": int(finfo.get("persist") or 0),
                  "crowded_long": bool(finfo.get("crowded_long")),
                  "crowded_short": bool(finfo.get("crowded_short")),
                  "oi_z": float(oinfo.get("oi_z") or 0.0),
                  "oi_chg_pct": float(oinfo.get("oi_chg_pct") or 0.0),
                  "oi_rising": bool(oinfo.get("oi_rising")),
                  "oi_falling": bool(oinfo.get("oi_falling")),
                  "oi_ok": bool(oinfo.get("ok")),
                  "rs_rank": _rs_rank(tf).get(symbol, 0.5),
                  "htf_sign": htf_sign}
        extras.update(_macro(tf, tk))
        extras.update(_market_state(tf, tk))
        extras["cost"] = _symbol_cost(symbol)
        extras["symbol"] = symbol                    # گیت نماد را هم می‌سنجد
        extras["suspended"] = _suspended_setups()
        extras["tf_suspended"] = _suspended_tfs().get(tf)
        pockets, susp_combos = _live_edge_book()
        extras["edge_pockets"] = pockets
        extras["suspended_combos"] = susp_combos
        extras["toxic_tf"] = tf == "1d"
        extras["short_needs_pocket"] = True
        extras["btc_macro"] = _btc_macro()
        extras["btc_z"] = btc_z
        try:
            import meta_gate
            extras["session"] = meta_gate.session_quality()
        except Exception:  # noqa: BLE001
            extras["session"] = {"tier": "mid", "mult": 0.92}
        # ⚰️ گاردِ فیدِ مرده: جفتِ حذف‌شده/بی‌معامله چارتِ یخ‌زده دارد و سیگنالش بی‌معناست (درسِ MATIC/DNT)
        if time.time() * 1000 - kl["t"][-1] > tf_spec.bar_ms(tf) * 3:
            age_d = (time.time() * 1000 - kl["t"][-1]) / 86400000
            res = {"symbol": symbol, "tf": tf,
                   "error": f"فیدِ داده مرده است (آخرین کندل {age_d:.0f} روز پیش) — جفتِ حذف‌شده یا بدونِ معامله"}
        else:
            res = engine.analyze(kl, tf, btc_z=btc_z, predict_fn=calib.predict,
                                 extras=extras, dir_fn=calib.predict_dir,
                                 action_fn=calib.predict_action)
            # analyze این دو را فقط در مسیرِ مجوزدار به status می‌برد؛ ماتریسِ تصمیم باید همیشه ببیندشان
            res["tf_suspended"] = extras["tf_suspended"]
            res["btc_z"] = btc_z
            # منبعِ کندل‌ها: «okx» یعنی پشتیبان (حجمِ OKX، بی‌qv/n/tbv) — کارنامه روی بایننس است
            res["data_src"] = kl.get("src")
            if isinstance(res.get("backtest"), dict):   # مبنای هزینهٔ بک‌تستِ کوچک (نه ۰٫۱۴٪ِ کارنامه)
                res["backtest"].update(cost_pct=extras["cost"], cost_basis="tier")
        res["symbol"] = symbol
    except Exception as e:  # noqa: BLE001
        res = {"symbol": symbol, "tf": tf, "error": str(e)}
    return res


_rebuild_proc = {"p": None, "started": None}


def _rebuild_running():
    p = _rebuild_proc.get("p")
    return p is not None and p.poll() is None


def _calib_status():
    """``calib.status()`` + بازآموزیِ پروسهٔ جدا. ``_state``ِ calib در همین پروسه از rebuild_models.py خبر ندارد،
    پس بی این، در تمامِ مدتِ بازآموزی «building: false» گزارش می‌شد و UI مدلِ کهنه را جاری نشان می‌داد."""
    st = calib.status()
    if not st.get("building") and _rebuild_running():
        st.update(building=True, progress=st.get("progress") or "بازآموزی در پروسهٔ جدا (rebuild_models.py)…",
                  rebuild_started=_rebuild_proc.get("started"))
    return st


def _calib_builder():
    """بازآموزی در **پروسهٔ جدا** (rebuild_models.py)، نه داخلِ پروسهٔ معامله.

    قبلاً calib.build همین‌جا اجرا می‌شد: ProcessPool درونِ uvicorn روی ویندوز،
    رقابتِ CPU با چرخهٔ معامله، و پاک‌شدنِ کشِ تحلیل وسطِ کار. حالا پروسهٔ جدا
    جدول را روی دیسک می‌نویسد و این‌جا فقط بازخوانی می‌شود.
    """
    import subprocess
    p = _rebuild_proc.get("p")
    if p is not None and p.poll() is None:
        return                                     # یکی در جریان است
    script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "rebuild_models.py")
    log_path = paths.data("logs", "rebuild.log")
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    try:
        with open(log_path, "a", encoding="utf-8") as out:
            proc = subprocess.Popen([sys.executable, script, "--top", str(CALIB_TOP)],
                                    stdout=out, stderr=subprocess.STDOUT,
                                    cwd=os.path.dirname(script),
                                    env=dict(os.environ, PYTHONIOENCODING="utf-8"))
        _rebuild_proc.update(p=proc, started=time.time())
        proc.wait()
        calib._table = None                        # جدولِ تازه از دیسک
        with _an_lock:
            _an_cache.clear()                      # تحلیل‌ها با مدلِ تازه از نو
        log.info("rebuild finished", returncode=proc.returncode)
    except Exception:  # noqa: BLE001
        log.exc("rebuild subprocess")


@app.on_event("startup")
def _startup():
    if calib.is_stale():
        threading.Thread(target=_calib_builder, daemon=True).start()

    def _daily_retrain():
        # بازآموزیِ روزانه حتی بدونِ ری‌استارت: هر ساعت چک، اگر جدول کهنه شد از نو بساز
        while True:
            time.sleep(3600)
            try:
                if calib.is_stale() and not calib.status().get("building"):
                    _calib_builder()
            except Exception:  # noqa: BLE001
                log.exc()
    threading.Thread(target=_daily_retrain, daemon=True).start()

    def _health_and_report():
        # نمونهٔ سلامت هر ۱۰ دقیقه (برای گیتِ عملیات: ۹۹٪ سبز در ۳۰ روز) + گزارشِ هفتگی
        import report
        last_week = None
        while True:
            try:
                h = health()
                report.record_health_sample(h.get("status"), h.get("problems"))
                week = time.strftime("%Y-%W", time.gmtime())
                if week != last_week:
                    report.write(report.build(closed_positions=paper.list_positions().get("closed") or []))
                    last_week = week
            except Exception:  # noqa: BLE001
                log.exc("health/report loop")
            try:
                # دفترِ رو-به-جلوِ روند باید حتی وقتی کسی صفحه را باز نکرده ثبت شود
                snap = trend.snapshot()
                trend.sync_demo(snap.get("now"))      # حدضررِ پوزیشن‌های دموی روند هم‌پای قاعده
            except Exception:  # noqa: BLE001
                log.exc("trend tracker")
            try:
                msnap = momentum.snapshot()
                momentum.sync_demo(msnap)             # دفترِ مومنتوم + خروجِ دمو در تصمیمِ «نقد»
                notify.maybe_notify_monday(msnap)     # تلگرام — فقط اگر کاربر تنظیم کرده باشد
            except Exception:  # noqa: BLE001
                log.exc("momentum tracker")
            try:
                trend_exec.step()           # فقط تست‌نت؛ بی‌مجوز یا بی‌کلید کاری نمی‌کند
            except Exception:  # noqa: BLE001
                log.exc("trend testnet executor")
            time.sleep(600)
    threading.Thread(target=_health_and_report, daemon=True).start()
    # میزِ تصمیم: ثبت/داوریِ دفترِ رو-به-جلو ~۲۰ ثانیه پس از هر مرزِ ۱۵دقیقه‌ای (فقط تحلیل)
    threading.Thread(target=_decision_loop, name="decision-ledger", daemon=True).start()
    autotrader.init(overview, DRIFT_CAP, get_analysis, health_fn=health)   # 🤖 ربات معامله‌گر خودکار
    autotrader.start()


class BotCfgReq(BaseModel):
    enabled: bool | None = None
    level: int | None = None


@app.get("/api/bot")
def bot_state():
    return autotrader.get_state()


@app.post("/api/bot/config")
def bot_config(req: BotCfgReq):
    return autotrader.set_config(req.enabled, req.level)


@app.get("/api/calib/status")
def calib_status():
    return _calib_status()


@app.get("/api/gates")
def gates_status():
    """قفلِ ایمنیِ سرمایه: ترکیب‌های مجاز، وضعیتِ لایو و سقف‌های ریسک (فقط‌خواندنی)."""
    return gates.status()


class KillReq(BaseModel):
    reason: str = "دستی"


@app.post("/api/kill")
def kill(req: KillReq):
    """🛑 کلیدِ قطعِ اضطراری — پوزیشن‌های ربات بسته، ربات خاموش، فهرستِ مجاز و لایو خاموش."""
    return {"ok": True, **autotrader.kill_switch(req.reason)}


@app.get("/api/report")
def report_now(write: bool = False):
    """گزارشِ مرحله‌ای: سایه، اجرا، یکپارچگی، عملیات — و آماده‌بودن برای لایو."""
    import report
    db = paper.list_positions()
    rep = report.build(closed_positions=db.get("closed") or [])
    if write:
        rep["path"] = report.write(rep)
    return rep


@app.get("/api/trend")
def trend_status(force: bool = False):
    """روندِ روزانه — نامزدِ پژوهشی و اثبات‌نشده: وضعیتِ قاعده‌ها روی ۲۰ ارزِ بزرگ،
    سیگنال‌های امروز، دفترِ رو-به-جلو و شواهدِ اکتشاف. هیچ مجوزی نمی‌دهد."""
    return trend.view(force=force)        # عکسِ روزانه (کش) + قیمتِ زندهٔ سیگنال‌های قابل‌اقدام


@app.get("/api/momentum")
def momentum_status(force: bool = False):
    """مومنتومِ ۲۸روزهٔ BTC/ETH (در بازار یا نقد، تصمیمِ دوشنبه) — سازگار در پژوهش، اثبات‌نشده."""
    return momentum.view(force=force)


@app.get("/api/kucoin/contracts")
def kucoin_contracts():
    try:
        return kucoin_desk.contracts()
    except Exception as e:  # noqa: BLE001
        raise HTTPException(502, f"کوکوین پاسخ نداد: {e}")


class SizeReq(BaseModel):
    sym: str
    balance: float
    risk_pct: float = 1.0
    entry: float
    stop: float
    fee_side: str = "taker"
    leverage: float | None = None


@app.post("/api/kucoin/size")
def kucoin_size(req: SizeReq):
    spec = kucoin_contracts().get(req.sym)
    if not spec:
        raise HTTPException(404, "قرارداد پیدا نشد")
    try:
        return kucoin_desk.size_position(req.balance, req.risk_pct, req.entry, req.stop, spec,
                                         "maker" if req.fee_side == "maker" else "taker", req.leverage)
    except ValueError as e:
        raise HTTPException(400, str(e))


@app.get("/api/kucoin/hours")
def kucoin_hours():
    return kucoin_desk.hours_table()


class JournalReq(BaseModel):
    sym: str
    side: str
    market: str = "futures"
    entry: float
    exit: float
    notional: float
    fees: float = 0.0
    stop: float | None = None
    setup: str = ""
    opened_at: int | None = None
    closed_at: int | None = None


@app.get("/api/journal")
def journal_get(balance: float | None = None, daily_limit_pct: float | None = None):
    return kucoin_desk.journal_stats(daily_limit_pct, balance)


@app.post("/api/journal")
def journal_add(req: JournalReq):
    if req.sym not in kucoin_desk.CONTRACT or req.side not in ("long", "short") or req.entry <= 0 or req.exit <= 0 or req.notional <= 0:
        raise HTTPException(400, "جهت، قیمت‌ها و حجم را درست وارد کنید")
    return kucoin_desk.add_trade(req.model_dump())


@app.delete("/api/journal/{tid}")
def journal_delete(tid: str):
    kucoin_desk.delete_trade(tid)
    return {"ok": True}


class NotifyReq(BaseModel):
    enabled: bool = False
    token: str = ""
    chat_id: str = ""


@app.get("/api/notify")
def notify_get():
    return notify.public_cfg()                      # توکن هرگز برگردانده نمی‌شود


@app.post("/api/notify")
def notify_set(req: NotifyReq):
    cfg = notify.load_cfg()
    cfg["enabled"] = bool(req.enabled)
    if req.token.strip():
        cfg["token"] = req.token.strip()
    if req.chat_id.strip():
        cfg["chat_id"] = req.chat_id.strip()
    notify.save_cfg(cfg)
    return notify.public_cfg()


@app.post("/api/notify/test")
def notify_test():
    try:
        ok = notify.send("✅ اتصالِ اعلانِ میزِ فرمانِ CTP برقرار است.")
    except Exception as e:  # noqa: BLE001
        raise HTTPException(502, f"ارسال نشد: {e}")
    if not ok:
        raise HTTPException(409, "اعلان روشن نیست یا توکن/شناسهٔ چت وارد نشده")
    return {"ok": True}


class MomentumDemoReq(BaseModel):
    symbol: str
    alloc_pct: float = 20.0


@app.post("/api/momentum/demo")
def momentum_demo_open(req: MomentumDemoReq):
    try:
        return momentum.open_demo(req.symbol.upper(), req.alloc_pct)
    except ValueError as e:
        raise HTTPException(409, str(e))


class TrendDemoReq(BaseModel):
    symbol: str
    risk_pct: float = 0.5


@app.post("/api/trend/demo")
def trend_demo_open(req: TrendDemoReq):
    """«خرید در دمو» برای سیگنالِ قابل‌اقدامِ امروز — حسابِ شبیه‌سازِ محلی، نه صرافی."""
    try:
        return trend.open_demo(req.symbol.upper(), req.risk_pct)
    except ValueError as e:
        raise HTTPException(409, str(e))


@app.get("/api/trend/testnet")
def trend_testnet_status():
    """اجرای قاعدهٔ روند روی تست‌نت: مجوز، اتصال، پوزیشن‌ها و کیفیتِ اجرا در برابرِ کاغذ."""
    out = trend_exec.summary()
    cfg = broker.load_cfg()
    out["broker"] = cfg.get("broker", "local")
    out["has_keys"] = bool(cfg.get("api_key") and cfg.get("api_secret"))
    return out


class TestnetResearchReq(BaseModel):
    enable: bool


@app.post("/api/trend/testnet")
def trend_testnet_toggle(req: TestnetResearchReq):
    """روشن/خاموش‌کردنِ اجرای روند روی تست‌نت — فقط با کلیکِ خودِ کاربر. پول واقعی دست نمی‌خورد."""
    g = gates.load_gates(force=True)
    keys = [trend_exec.STRATEGY] if req.enable else []
    gates.set_testnet_research(keys, reason=("کاربر: اجرای روندِ روزانه روی تست‌نت روشن" if req.enable
                                             else "کاربر: اجرای روندِ روزانه روی تست‌نت خاموش"),
                               user_token=f"user:{g['version']}")
    return trend_testnet_status()


@app.get("/api/research")
def research_status():
    """پیش‌ثبت و حکمِ آزمونِ منجمد — تنها منبعِ فهرستِ مجاز.

    برای هر فرضیه می‌گوید «لبه اثبات شد»، «لبه نیست» یا «شواهد کافی نیست»، و
    چند مشاهدهٔ مستقل برای اثباتِ اثرِ دیده‌شده لازم بود.
    """
    import research
    doc = research.load_prereg()
    try:
        integrity = "ok" if doc and research.verify_prereg(doc) else "missing"
    except research.PreregistrationError as e:
        integrity = f"broken: {e}"
    return {
        "preregistration": ({k: doc[k] for k in ("hash", "registered_at", "family", "symbols",
                                                "gates", "final_windows", "n_trials")}
                            if doc else None),
        "integrity": integrity,
        "judgement": research.last_judgement(),
        "allowed_combos": gates.allowed_combos(),
    }


HEALTH_MAX_CANDIDATE_AGE_SEC = 3 * max(ANALYSIS_TTL.values())
HEALTH_MAX_MODEL_AGE_DAYS = 30
# پس از بسته‌شدنِ هر کندل این‌قدر مهلت تا کش تازه شود (حلقه‌ها هر ۱ تا ۳۰ دقیقه می‌گیرند)
KLINE_GRACE_SEC = tf_spec.KLINE_GRACE_SEC


@app.get("/api/health")
def health():
    """یک نگاه: آیا این سیستم الان قابلِ اتکاست؟

    قرمز یعنی عددهایی که می‌بینید ممکن است کهنه یا ناقص باشند. چون خطاها در
    سراسرِ کد بلعیده می‌شوند و ویژگیِ گم‌شده به ۰٫۰ تبدیل می‌شود، بدونِ این نقطه
    هیچ راهی برای فهمیدنِ «دادهٔ من مرده است» وجود نداشت.
    """
    problems, warnings = [], []
    out = {"checked_at": time.time()}

    g = gates.status()
    out["gates"] = g
    if g["live_effective"]:
        warnings.append("معاملهٔ واقعی فعال است")

    cs = _calib_status()
    age_h = cs.get("age_hours")
    out["model"] = {"version": cs.get("version"), "required": cs.get("required_version"),
                    "age_hours": age_h, "stale": cs.get("stale"), "building": cs.get("building"),
                    "error": cs.get("error")}
    if cs.get("error"):
        problems.append(f"ساخت مدل خطا داد: {cs['error']}")
    if cs.get("version") != cs.get("required_version"):
        problems.append("نسخهٔ مدل با نسخهٔ کد نمی‌خواند — تحلیل‌ها بدون مدل‌اند")
    elif age_h is not None and age_h > HEALTH_MAX_MODEL_AGE_DAYS * 24:
        problems.append(f"مدل {age_h / 24:.0f} روز کهنه است")

    try:
        cnt = candidates.counts()
    except Exception as e:  # noqa: BLE001
        cnt = {"error": str(e)}
        problems.append("دفترِ کاندیدها خوانده نشد")
    out["candidates"] = cnt
    last = cnt.get("last_logged_at")
    if last is None:
        warnings.append("هنوز هیچ کاندیدی ثبت نشده — یک بار /api/overview را صدا بزنید")
    elif time.time() - last > HEALTH_MAX_CANDIDATE_AGE_SEC:
        problems.append(f"از آخرین ثبتِ کاندید {(time.time() - last) / 3600:.1f} ساعت گذشته")

    try:
        out["data"] = market.cache_stats()
        now_s = time.time()
        for tf, row in (out["data"].get("klines_by_tf") or {}).items():
            # کهنه یعنی: این تایم‌فریم در حالِ استفاده است، مهلتِ پس از بسته‌شدنِ کندل گذشته،
            # و حتی تازه‌ترین داده آخرین کندلِ بسته‌شده را ندارد. قاعدهٔ قبلی (سن از زمانِ دریافت)
            # هر کندلِ ساعتی را پس از ۶ دقیقه «کهنه» می‌خواند، سلامت را همیشه زرد می‌کرد و گیتِ
            # عملیات (۹۹٪ سبز) را هرگز قبول‌شدنی نمی‌گذاشت.
            if not tf_spec.is_known(tf):
                continue                        # تایم‌فریمِ ناشناخته با فرضِ بی‌صدای ۶۰ دقیقه سنجیده نمی‌شود
            tf_sec = tf_spec.minutes(tf) * 60
            in_use = (row.get("newest_age_sec") or 1e12) < 2 * tf_sec
            past_grace = now_s % tf_sec > KLINE_GRACE_SEC[tf]
            behind = row.get("behind_candles_min") or 0
            if in_use and past_grace and behind >= 1:
                warnings.append(f"دادهٔ کندلِ {tf} عقب است: تازه‌ترین کش {behind} کندل پشتِ آخرین بسته است")
    except Exception as e:  # noqa: BLE001
        out["data"] = {"error": str(e)}
        problems.append("لایهٔ داده پاسخ نداد")

    try:
        db = paper.list_positions()
        bot_cfg = autotrader.load_cfg()
        out["trading"] = {
            "open_positions": len(db["open"]),
            "closed_recorded": len(db["closed"]),
            "bot_enabled": bot_cfg["enabled"],
            "bot_level": bot_cfg["level"],
            "wallet": paper.wallet_summary(),
        }
    except Exception as e:  # noqa: BLE001
        out["trading"] = {"error": str(e)}
        problems.append("وضعیت پوزیشن‌ها خوانده نشد")

    try:
        out["logs"] = log.counts()
        recent = log.tail(200, level="warning", since=log.STARTED_AT)
        out["logs"]["recent_warnings"] = len(recent)
        out["logs"]["sample"] = [r.get("msg") for r in recent[-5:]]
    except Exception:  # noqa: BLE001
        out["logs"] = {"error": "unavailable"}

    out["status"] = "red" if problems else ("amber" if warnings else "green")
    out["problems"] = problems
    out["warnings"] = warnings
    return out


@app.get("/api/version")
def version_info():
    """شناسهٔ دقیق کد و مدلِ در حال اجرا برای راستی‌آزمایی انتشار."""
    cs = _calib_status()
    return {
        "app_version": APP_VERSION,
        "ai_core_version": AI_CORE_VERSION,
        "release_date": RELEASE_DATE,
        "model_version": cs.get("version"),
        "required_model_version": cs.get("required_version", calib.CALIB_VERSION),
        "model_built_at": cs.get("built_at"),
        "model_stale": cs.get("stale", True),
        "model_building": cs.get("building", False),
    }


@app.post("/api/calib/rebuild")
def calib_rebuild():
    if calib.status().get("building") or _rebuild_running():
        return {"ok": False, "msg": "در حال ساخت است"}
    threading.Thread(target=_calib_builder, daemon=True).start()
    return {"ok": True}


@app.get("/api/live-stats")
def live_stats(tf: str | None = None):
    """داوری سیگنال‌های معلق + آمار عملکرد زنده ۳۰ روز اخیر + پرچمِ drift (تیر ۵.۱)."""
    try:
        shadow.resolve(market.get_klines)
    except Exception:  # noqa: BLE001
        log.exc()
    try:
        candidates.resolve(market.get_klines, limit=120)
    except Exception:  # noqa: BLE001
        log.exc()
    st = shadow.stats(tf)
    drift = None
    # drift با بازده خالص تشخیص داده می‌شود، نه win-rate خام که نسبت سود/ضرر را نادیده می‌گیرد.
    # فقط وقتی بازخوردِ زنده فعال است: متنش «ورودها متوقف می‌شوند» است و بی‌تعلیقِ واقعی دروغ می‌شد.
    if (LIVE_FEEDBACK_ACTIVE and tf and st.get("n", 0) >= 20 and st.get("avg_r") is not None
            and st["avg_r"] <= 0):
        drift = (f"⚠️ عملکرد زندهٔ {tf} ({st['avg_r']:+.2f}R از {st['n']} سیگنال) "
                 "پس از هزینه مثبت نیست — ورودهای تازهٔ این تایم‌فریم متوقف می‌شوند.")
    return {"tf": tf, "stats": st, "all": shadow.stats(None), "drift": drift,
            "suspended": _suspended_setups(),
            "edge_pockets": _live_edge_book()[0],
            "suspended_combos": _live_edge_book()[1],
            # stats/all/suspended/edge_pockets/drift از دفترِ سایهٔ بی‌نویسنده‌اند — «feedback» می‌گوید غیرفعال‌اند
            "feedback": _live_feedback(),
            "cost_basis": _cost_basis(),
            # دفترِ کاندیدها: همهٔ ستاپ‌های دیده‌شده، نه فقط مجازها
            "candidates": candidates.counts(),
            "candidate_stats": candidates.stats(tf),
            "candidate_stats_tradeable_only": candidates.stats(tf, only_tradeable=True)}


@app.get("/api/edge-health")
def edge_health():
    """گزارش سلامت جیب‌های زنده + کیفیت پر شدن سفارش صبور."""
    try:
        import edge_book
        import fill_quality
        fq = fill_quality.summary(14)
        if not LIVE_FEEDBACK_ACTIVE:                # منبعِ جیب‌ها تغذیه نمی‌شود: «منجمد» نه، «غیرفعال»
            h = {"updated": time.time(), "pocket_count": 0, "pockets": {}, "suspended_count": 0,
                 "verdict": "inactive", "advice": LIVE_FEEDBACK_NOTE_FA}
            return {"ok": True, "active": False, "health": h, "fill_quality": fq, "scan_hint": [],
                    "feedback": _live_feedback(),
                    "version": {"app": APP_VERSION, "ai_core": AI_CORE_VERSION}}
        h = edge_book.get_health()
        return {"ok": True, "active": True, "health": h, "fill_quality": fq,
                "scan_hint": edge_book.scan_tfs_for(["1h", "4h", "15m"]),
                "feedback": _live_feedback(),
                "version": {"app": APP_VERSION, "ai_core": AI_CORE_VERSION}}
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "error": str(e)}


@app.get("/api/overview")
def overview(tf: str = "1h"):
    if tf not in TFS:
        raise HTTPException(400, "تایم‌فریم نامعتبر")
    try:
        # فقط فهرستِ پنج‌ارزیِ کاربر (watchlist.py) — به‌جای ۲۰۰ ارزِ برتر، تا همه‌چیز سریع‌تر باشد
        symbols = list(watchlist.SYMBOLS)
        try:
            _, tickers = market.get_top_symbols(50)
        except Exception:  # noqa: BLE001, silent-ok — تغییرِ ۲۴ساعته اختیاری است
            tickers = {}
    except Exception:  # noqa: BLE001 — قطعیِ شبکه/DNS: پاسخِ مرتب به‌جای crash
        return {"tf": tf, "updated": time.time(), "coins": [],
                "net_error": "اتصال به صرافی برقرار نیست (شبکه/DNS/فیلترینگ). اینترنت یا VPN را بررسی کنید؛ خودکار دوباره تلاش می‌شود.",
                "btc": None, "calib": {}}
    results = list(_pool.map(lambda s: get_analysis(s, tf), symbols))
    coins = []
    for s, a in zip(symbols, results):
        t = tickers.get(s, {})
        row = {"symbol": s, "name": s[:-4], "chg24h": round(t.get("chg24h", 0.0), 2)}
        if "error" in a:
            row["error"] = a["error"]
        else:
            row.update(price=a["price"], score=a["score"], z=a["z"], zt=a.get("zt"),
                       votes_bull=a["votes_bull"], votes_bear=a["votes_bear"],
                       regime=a["regime"], p_up=a["p_up"], p_calibrated=a.get("p_calibrated", False),
                       target_pct=round(a["forecast"]["target_pct"], 2),
                       target=a["forecast"]["target"],
                       side=a["trade"].get("side"), grade=a["trade"].get("grade"),
                       status=a["trade"].get("status"), rr=a["trade"].get("rr"),
                       win_rate=a["backtest"]["win_rate"], bt_n=a["backtest"]["n"],
                       viable=a["trade"].get("viable", False),
                       tradeable=a["trade"].get("tradeable", a["trade"].get("viable", False)),
                       ev_pct=a["trade"].get("ev_pct"), p_win=a["trade"].get("p_win"),
                       p_win_low=a["trade"].get("p_win_low"),
                       p_win_high=a["trade"].get("p_win_high"),
                       p_uncertainty=a["trade"].get("p_uncertainty"),
                       ev_lcb_pct=a["trade"].get("ev_lcb_pct"),
                       edge_r=a["trade"].get("edge_r"),
                       edge_lcb_r=a["trade"].get("edge_lcb_r"),
                       edge_rank_ic=a["trade"].get("edge_rank_ic"),
                       edge_trusted=a["trade"].get("edge_trusted", False),
                       policy_trusted=a["trade"].get("policy_trusted", False),
                       policy_pass=a["trade"].get("policy_pass", False),
                       policy_score=a["trade"].get("policy_score"),
                       policy_margin=a["trade"].get("policy_margin"),
                       policy_test=a["trade"].get("policy_test"),
                       market_rank_required=a["trade"].get("market_rank_required", False),
                       select_quantile=a["trade"].get("select_quantile"),
                       regime_veto=a["trade"].get("regime_veto", False),
                       regime_ok=a["trade"].get("regime_ok", False),
                       model_confidence=a["trade"].get("model_confidence"),
                       n_hist=a["trade"].get("n_hist", 0), reliability=a["trade"].get("reliability"),
                       btc_align=a["trade"].get("btc_align", True),
                       setup=a["trade"].get("setup"),
                       setup_fa=a["trade"].get("setup_fa"),
                       readiness=a["trade"].get("readiness"),
                       ready_side=a["trade"].get("ready_side"),
                       wait_why="، ".join(a["trade"].get("reasons", [])[:2]),
                       entry=a["trade"].get("entry"), sl=a["trade"].get("sl"),
                       tp=a["trade"].get("tp"), risk_pct=a["trade"].get("risk_pct"),
                       time_stop_min=a["trade"].get("time_stop_min"),
                       signal_score=a["trade"].get("signal_score"),
                       signal_tier=a["trade"].get("signal_tier"),
                       market_headwind=a["trade"].get("market_headwind", False),
                       patient_viable=a["trade"].get("patient_viable", False),
                       observe_only=a["trade"].get("observe_only", False),
                       recommendation=a["trade"].get("recommendation"),
                       authority=a["trade"].get("authority"),
                       gate_allowed=a["trade"].get("gate_allowed", False),
                       pocket_grade=a["trade"].get("pocket_grade"),
                       pocket_size_hint=a["trade"].get("pocket_size_hint"),
                       pocket_lcb_r=a["trade"].get("pocket_lcb_r"),
                       setup_observed=a["trade"].get("setup_observed", False),
                       meta_score=a["trade"].get("meta_score"),
                       meta_size_mult=a["trade"].get("meta_size_mult"),
                       meta_regime=a["trade"].get("meta_regime"),
                       meta_blocks=a["trade"].get("meta_blocks"),
                       meta_confluence=a["trade"].get("meta_confluence"),
                       meta_session=a["trade"].get("meta_session"),
                       meta_reasons=a["trade"].get("meta_reasons"),
                       oi_z=a["trade"].get("oi_z"),
                       atr14=a["trade"].get("atr14"),
                       combo_suspended=bool(a["trade"].get("setup") and
                                            f"{tf}|{a['trade'].get('setup')}" in
                                            (_live_edge_book()[1] or {})),
                       cost=_symbol_cost(s))
            try:
                row["spark"] = [round(x, 8) for x in market.get_klines(s, tf)["c"][-28:]]
            except Exception:  # noqa: BLE001
                row["spark"] = None
            tr = a["trade"]
            if tr.get("side") and tr.get("tradeable"):
                live_px = t.get("price")               # دیررسیده: قیمت زنده از کندل سیگنال دور شده
                if live_px:
                    dr = abs(live_px - a["price"]) / max(a["price"], 1e-12) * 100
                    if dr > DRIFT_CAP[tf]:
                        row["tradeable"] = False
                        row["drift_reject"] = True
                        row["status"] = f"دیررسیده — قیمت {dr:.1f}٪ از کندل سیگنال حرکت کرده؛ با کندل بعدی تازه می‌شود"
        coins.append(row)
    coins = [r for r in coins if "داده کافی" not in r.get("error", "")][:TOP_N]   # حذف ارزهای تازه‌لیست‌شده بدون سابقه
    # سیاست انتخاب عمل در آموزش «در هر timestamp» داوری شده؛ قرارداد live نیز دقیقاً
    # همان است و آستانهٔ مطلقِ یک رژیم به رژیم بعد حمل نمی‌شود.
    action_active = [r for r in coins if r.get("side") and r.get("tradeable")
                     and r.get("market_rank_required")]
    action_active.sort(key=lambda r: (r.get("policy_score") or -99), reverse=True)
    for rank, row in enumerate(action_active):
        rank_pct = 100.0 if len(action_active) == 1 else \
            100.0 * (len(action_active) - 1 - rank) / (len(action_active) - 1)
        row["market_rank_pct"] = round(rank_pct, 1)
        row["market_rank_enforced"] = True
        required_pct = 100.0 * float(row.get("select_quantile") or 0.75)
        if len(action_active) < 8:
            row["tradeable"] = False
            row["status"] = "منتظر مقایسهٔ مقطعی — حداقل ۸ کاندید هم‌زمان برای رتبه‌بندی لازم است"
        elif rank_pct < required_pct:
            row["tradeable"] = False
            row["status"] = (f"ردِ رتبهٔ مقطعی — صدک {rank_pct:.0f} بازار؛ "
                             f"سیاست فقط صدک {required_pct:.0f} به بالا را مجاز می‌کند")

    # انتخاب مقطعی نهایی روی همهٔ فرصت‌های معتبر: فقط بهترین ۴۰٪ اجازهٔ ورود دارند.
    active = [r for r in coins if r.get("side") and r.get("tradeable")]
    # مرتب‌سازی فقط با اعدادی که مرجعِ معتبر دارند؛ سیاستِ مردود دیگر عددی نمی‌دهد.
    active.sort(key=lambda r: (
        (r.get("policy_margin") if r.get("policy_trusted") and r.get("policy_margin") is not None else -99),
        r.get("signal_score") or 0,
        (r.get("edge_r") if r.get("edge_trusted") and r.get("edge_r") is not None else -99),
    ), reverse=True)
    for rank, row in enumerate(active):
        rank_pct = 100.0 if len(active) == 1 else 100.0 * (len(active) - 1 - rank) / (len(active) - 1)
        row["market_rank_pct"] = round(rank_pct, 1)
        row["market_rank_enforced"] = len(active) >= 4
        row["entry_quality"] = round(
            0.65 * float(row.get("signal_score") or 0) + 0.35 * rank_pct, 1)
        if row["market_rank_enforced"] and rank_pct < 60.0:
            row["tradeable"] = False
            row["status"] = (f"ردِ مقطعی — این فرصت در صدک {rank_pct:.0f} بازار است؛ "
                             "فقط بهترین ۴۰٪ سیگنال‌های هم‌زمان معامله می‌شوند")
    # ── دفترِ کاندیدها: **هر** ستاپِ دیده‌شده ثبت می‌شود، مسدود یا نه ──
    # ثبتِ فقط ردیف‌های مجاز، حلقهٔ یادگیری را در بن‌بست گذاشته بود: بدونِ مدلِ معتبر
    # هیچ ردیفی ثبت نمی‌شد و بدونِ ردیف هیچ لبه‌ای قابلِ کشف نبود.
    logged = 0
    for row in coins:
        if tf not in tf_spec.MODEL_TFS:
            break                   # 5m فقط تحلیل/ماتریس است؛ دفترِ رو به جلویش decision_ledger است، نه دفترِ پژوهشی
        if not row.get("side") or row.get("error"):
            continue
        try:
            logged += bool(candidates.log_candidate(row, tf))
        except Exception:  # noqa: BLE001 — ثبت هرگز نباید پاسخِ API را بشکند
            log.exc()
    if logged:
        print(f"[candidates] {tf}: {logged} ردیف تازه ثبت شد", flush=True)
    coins.sort(key=lambda r: (
        r.get("tradeable", False),
        r.get("entry_quality") or r.get("signal_score") or 0,
        abs(r.get("score", 0) or 0),
    ), reverse=True)
    btc_a = get_analysis("BTCUSDT", tf)
    btc_state = None
    if "error" not in btc_a:
        sc = btc_a["score"]
        btc_state = {"score": sc, "z": btc_a.get("z"),
                     "dir": "صعودی" if sc >= 0.15 else "نزولی" if sc <= -0.15 else "خنثی"}
    cs = _calib_status()
    mtf = (cs.get("models") or {}).get(tf, {})
    dir_lift = mtf.get("dir_lift")
    setup_lift = mtf.get("oos_lift")
    action_ok = bool(mtf.get("action_trusted"))
    setup_policy_ok = bool(mtf.get("policy_trusted"))
    tf_trust = {
        "dir_ok": bool(mtf.get("dir_trusted")),
        "setup_ok": bool(mtf.get("trusted")),
        "edge_ok": bool(mtf.get("edge_trusted")),
        "policy_ok": setup_policy_ok or action_ok,
        "policy_type": "action" if action_ok else "setup" if setup_policy_ok else None,
        "policy_n": mtf.get("action_n") if action_ok else mtf.get("policy_n"),
        "policy_avg_net_r": mtf.get("action_avg_net_r") if action_ok else mtf.get("policy_avg_net_r"),
        "policy_lcb_net_r": mtf.get("action_lcb_net_r") if action_ok else mtf.get("policy_lcb_net_r"),
        "policy_profit_factor": mtf.get("action_profit_factor") if action_ok else mtf.get("policy_profit_factor"),
        "policy_uplift_r": mtf.get("action_uplift_r") if action_ok else mtf.get("policy_uplift_r"),
        "policy_quantile": mtf.get("action_quantile") if action_ok else mtf.get("policy_quantile"),
        "policy_halves_avg_r": mtf.get("action_halves_avg_r") if action_ok else mtf.get("policy_halves_avg_r"),
        "dir_lift": dir_lift, "setup_lift": setup_lift,
        "edge_rank_ic": mtf.get("edge_rank_ic"),
        "edge_lift_r": mtf.get("edge_lift_r"),
    }
    return {"tf": tf, "updated": time.time(), "coins": coins, "btc": btc_state, "tf_trust": tf_trust,
            "version": {"app": APP_VERSION, "ai_core": AI_CORE_VERSION,
                        "model": cs.get("version"), "required_model": calib.CALIB_VERSION},
            "live_suspended": _suspended_tfs().get(tf),
            "edge_pockets": {k: v for k, v in _live_edge_book()[0].items()
                             if k.startswith(tf + "|")},
            "live_feedback": _live_feedback(),       # live_suspended/edge_pockets/combo_suspended: غیرفعال؟
            "cost_basis": _cost_basis(),             # «cost»ِ هر ردیف = ردهٔ نقدشوندگی، نه ۰٫۱۴٪ِ کارنامه
            "btc_macro": _btc_macro(),
            "breadth_macro": round((_market_state("1d") or {}).get("breadth", 0.0), 2),
            "calib": {"building": cs.get("building"), "progress": cs.get("progress"),
                      "age_hours": cs.get("age_hours"), "events": (cs.get("events") or {}).get(tf)}}


@app.get("/api/coin/{symbol}")
def coin_detail(symbol: str):
    results = dict(zip(TFS, _pool.map(lambda tf: get_analysis(symbol, tf), TFS)))
    return {"symbol": symbol, "tfs": results}


# ───────────────────────── میزِ تصمیم (لانگ/شورت/صبر) — فقط تحلیل، هرگز مجوزِ معامله ─────────────────────────
DECISION_WARM_DELAY = 30.0   # ثانیه پس از راه‌اندازی تا گرم‌کردنِ کارنامهٔ دوساله (در پس‌زمینه)
DECISION_LAG = 20.0          # ثانیه پس از هر مرزِ ۵دقیقه‌ای UTC: کندل بسته شده و کشِ کندل تازه است
# ۵ دقیقه = کوچک‌ترین تایم‌فریم (با ۹۰۰ ثانیه، decision.record با STALE_BARS=3 فقط ۱ از ۳ کندلِ 5m را می‌دید).
# هر دور فقط تایم‌فریم‌هایی را تازه می‌کند که کندلشان از دورِ قبل بسته شده (5m همیشه، 1d فقط نیمه‌شبِ UTC).
DECISION_PERIOD = 300
CANDIDATE_RESOLVE_LIMIT = 120   # داوریِ کاندیدها در هر دور (مثلِ ‎/api/live-stats)
_decision_state = {"last_boundary": None,   # آخرین مرزِ ۵دقیقه‌ای که تصمیم‌هایش جمع شد
                   "retry": {}}              # (sym, tf) -> openِ کندلی که هنوز دیده نشده (خطا/تحلیلِ کهنه)


def _decision_lookup(symbols=None, tfs=None, keys=None):
    """همهٔ تحلیل‌های ارز × تایم‌فریمِ میزِ تصمیم (۵×۵=۲۵) را موازی روی ``_pool`` می‌سازد
    و یک تابعِ جست‌وجوی ``(sym, tf) -> analysis`` برای ``decision.collect/refresh`` برمی‌گرداند.
    ``keys`` (اختیاری): فقط همین خانه‌های ``(sym, tf)`` به‌جای ضربِ symbols × tfs.

    ترتیب: تایم‌فریمِ بالاتر و BTC اول — پیش‌نیازِ زنجیرهٔ HTF/BTCِ بقیه‌اند، پس منتظرها کمتر معطل می‌شوند.
    """
    symbols = tuple(symbols or decision.SYMBOLS)
    tfs = tuple(tfs or decision.TFS)
    keys = sorted(keys if keys is not None else ((s, tf) for s in symbols for tf in tfs),
                  key=lambda k: (-tf_spec.MINUTES.get(k[1], 0), k[0] != "BTCUSDT"))

    def _one(k):
        try:
            return get_analysis(k[0], k[1])
        except Exception as e:  # noqa: BLE001 — یک خانهٔ خراب نباید کلِ جدول را بیندازد
            log.exc("decision analysis", symbol=k[0], tf=k[1])
            return {"symbol": k[0], "tf": k[1], "error": str(e)}

    results = dict(zip(keys, _pool.map(_one, keys)))

    def lookup(sym, tf):
        hit = results.get((sym, tf))
        return hit if hit is not None else get_analysis(sym, tf)
    return lookup


def _decision_boundary(now):
    """مرزِ ۵دقیقه‌ایِ UTCِ این دور (ثانیه) — دورِ دیررسیده هم به همان مرزِ خودش نسبت داده می‌شود."""
    return int((now - DECISION_LAG) // DECISION_PERIOD) * DECISION_PERIOD


def _due_tfs(boundary, last_boundary=None):
    """تایم‌فریم‌هایی که کندلشان در ``(last_boundary, boundary]`` بسته شده.

    دورِ اول (یا ساعتِ عقب‌رفته) ⇒ همه؛ همان مرزِ دورِ قبل ⇒ هیچ. اگر دوری جا بیفتد (دورِ قبلی بیش
    از ۵ دقیقه طول کشید)، مرزِ ساعتی/۴ساعتی/روزانهٔ وسطِ آن گم نمی‌شود."""
    if last_boundary is None or boundary < last_boundary:
        return tuple(decision.TFS)
    return tuple(tf for tf in decision.TFS if tf_spec.closed_between(tf, last_boundary, boundary))


def _expected_bar(tf, boundary):
    """openِ (ms) آخرین کندلِ بسته‌شدهٔ ``tf`` در مرزِ ``boundary`` (ثانیهٔ UTC).

    همان ``_last_closed_open`` (کلیدِ کشِ حالتِ بازار/طلا)؛ یک تعریف تا چکِ کهنگیِ زمان‌بند و آن کلید از هم جدا نشوند.
    """
    return _last_closed_open(tf, int(boundary))


def _collect_cells(cells):
    """تصمیمِ خانه‌های ``cells`` (فهرستِ ``(sym, tf)``)، هر تایم‌فریم یک ``decision.collect``."""
    if not cells:
        return []
    lookup = _decision_lookup(keys=cells)
    by_tf = {}
    for s, tf in cells:
        by_tf.setdefault(tf, []).append(s)
    out = []
    for tf, syms in by_tf.items():
        out += decision.collect(lookup, symbols=tuple(syms), tfs=(tf,)) or []
    return out


def _drop_analysis(keys):
    """تحلیلِ کش‌شدهٔ این خانه‌ها دور انداخته می‌شود تا از نو ساخته شود (کشِ کندل تازگی را خودش می‌سنجد)."""
    with _an_lock:
        for k in keys:
            _an_cache.pop(k, None)


def _screen_decisions(decisions, boundary, retry):
    """فقط تصمیم‌هایی که کندلِ مورد انتظارِ همین مرز را دارند برای ثبت می‌مانند.

    * خطا ⇒ خانه در ``retry`` می‌ماند و دورِ ۵دقیقه‌ایِ بعد دوباره تحلیل می‌شود (LP-5)؛ قبلاً تا مرزِ
      بعدیِ همان تایم‌فریم تلاشی نبود و کندلِ 1h/4h/1d با یک خطای گذرا برای همیشه گم می‌شد.
    * کندلِ قبلی (تحلیلِ کهنه — کشِ سردِ درست پیش از مرز، یا صرافیِ دیررس) ⇒ تحلیلِ کش‌شده دور انداخته و
      همین حالا یک‌بار از نو ساخته می‌شود؛ اگر باز کهنه بود **ثبت نمی‌شود** و دورِ بعد دوباره (LP-4).
      اگر BTC کهنه بود، آلت‌های همان تایم‌فریم هم از نو ساخته می‌شوند (btc_z از همان تحلیل می‌آید).
    خروجی: ``(تصمیم‌های قابلِ ثبت، تعدادِ کهنه‌ها)``.
    """
    def split(ds):
        good, stale = [], []
        for d in ds:
            key = (d.get("sym"), d.get("tf"))
            exp = _expected_bar(key[1], boundary)
            try:
                ts = int(d.get("candle_ts"))
            except (TypeError, ValueError):
                ts = None
            if d.get("error") or ts is None:
                retry[key] = exp
            elif ts < exp:
                retry[key] = exp
                stale.append(key)
            else:
                retry.pop(key, None)
                good.append(d)
        return good, stale

    good, stale = split(decisions)
    if not stale:
        return good, 0
    redo = set(stale)
    redo |= {(s, tf) for b, tf in stale if b == "BTCUSDT" for s in decision.SYMBOLS}
    _drop_analysis(redo)
    good = [d for d in good if (d.get("sym"), d.get("tf")) not in redo]
    again, _still = split(_collect_cells(sorted(redo)))
    return good + again, len(stale)


def _decision_cycle(now=None, tfs=None):
    """یک دورِ پس‌زمینه: تصمیم‌ها → ثبت در دفترِ رو-به-جلو → داوریِ ردیف‌های باز → کارنامه (بی‌انتظار).

    فقط تایم‌فریم‌هایی که کندلشان تازه بسته شده دوباره تحلیل و ثبت می‌شوند (``tfs`` برای تست/اجبار)،
    به‌علاوهٔ خانه‌هایی که کندلِ آخرشان هنوز دیده نشده (خطا/تحلیلِ کهنه در دورهای قبل — ``_screen_decisions``).
    داوریِ ردیف‌های باز (دفترِ تصمیم و دفترِ کاندیدها) هر دور برای همه انجام می‌شود. هر گام جدا
    خطاگیری می‌شود تا خرابیِ یکی بقیه را نیندازد. خروجی: شمارنده‌ها برای لاگ/تست."""
    now = time.time() if now is None else now
    boundary = _decision_boundary(now)
    # کپی و جایگزینی (نه جهشِ درجا): یک دورِ نیمه‌کاره یا patchِ تست حالتِ مشترک را آلوده نمی‌کند
    retry = {k: exp for k, exp in (_decision_state.get("retry") or {}).items()
             if exp == _expected_bar(k[1], boundary)}      # کندلِ تازه‌تری بسته شده ⇒ از راهِ tfs می‌آید
    if tfs is None:
        tfs = _due_tfs(boundary, _decision_state["last_boundary"])
    tfs = tuple(tfs)
    cells = [(s, tf) for tf in tfs for s in decision.SYMBOLS]
    extra = sorted(k for k in retry if k not in set(cells))
    out = {"decisions": 0, "tfs": list(tfs), "retried": len(extra), "stale": 0,
           "recorded": None, "resolved": None, "candidates_resolved": None}
    collected = _collect_cells(cells + extra)
    decisions, out["stale"] = _screen_decisions(collected, boundary, retry)
    _decision_state["last_boundary"] = boundary
    _decision_state["retry"] = retry
    out["decisions"] = len(collected)
    out["pending_retry"] = len(retry)
    if decisions:
        try:
            out["recorded"] = decision.record(decisions)
        except Exception:  # noqa: BLE001
            log.exc("decision ledger record")
    try:
        out["resolved"] = decision.resolve(market.get_klines)
    except Exception:  # noqa: BLE001
        log.exc("decision ledger resolve")
    try:
        # دفترِ کاندیدها هم در پس‌زمینه داوری می‌شود، نه فقط وقتی UI ‎/api/live-stats را صدا می‌زند؛
        # وگرنه با UIِ بسته پنجرهٔ ۴۲۰ کندلی از کندلِ ورود می‌گذشت (LP-7: حالا «no_data» می‌شود).
        out["candidates_resolved"] = candidates.resolve(market.get_klines, limit=CANDIDATE_RESOLVE_LIMIT)
    except Exception:  # noqa: BLE001
        log.exc("candidates resolve")
    try:
        decision.scorecards(wait=False)             # حداکثر روزی یک‌بار، در نخِ جدا
    except Exception:  # noqa: BLE001
        log.exc("decision scorecard refresh")
    return out


def _next_decision_run(now):
    """زمانِ اجرای بعدی: DECISION_LAG ثانیه پس از نزدیک‌ترین مرزِ ۵دقیقه‌ای UTC که هنوز نگذشته."""
    base = ((now - DECISION_LAG) // DECISION_PERIOD) * DECISION_PERIOD
    return base + DECISION_PERIOD + DECISION_LAG


def _decision_wait(now):
    """ثانیه تا دورِ بعد. اگر مرزی که مهلتش گذشته هنوز پردازش نشده (دورِ قبل بیش از ۵ دقیقه طول کشید)
    صفر — وگرنه ``_next_decision_run`` آن مرز را جا می‌انداخت و کندلِ 5mِ آن هرگز دیده نمی‌شد (LP-5)."""
    last = _decision_state.get("last_boundary")
    if last is not None and _decision_boundary(now) > last:
        return 0.0
    return max(1.0, _next_decision_run(now) - now)


def _decision_loop():
    """دفترِ میزِ تصمیم باید حتی وقتی کسی صفحه را باز نکرده ثبت و داوری شود."""
    time.sleep(DECISION_WARM_DELAY)
    try:
        decision.scorecards(wait=False)             # بازسازیِ کارنامه در پس‌زمینه، اگر کهنه است
    except Exception:  # noqa: BLE001
        log.exc("decision scorecard warm-up")
    while True:
        try:
            time.sleep(_decision_wait(time.time()))
            info = _decision_cycle()
            log.info("decision cycle", **info)
        except Exception:  # noqa: BLE001
            log.exc("decision loop")
            time.sleep(60)


@app.get("/api/decisions")
def decisions_view():
    """میزِ تصمیم: لانگ/شورت/صبرِ هر ارز × تایم‌فریم کنارِ کارنامهٔ دوسالهٔ خالص و دفترِ زنده.

    ``def`` همگام است عمداً: FastAPI آن را در threadpool اجرا می‌کند و حلقهٔ رویداد بسته نمی‌شود.
    """
    try:
        return decision.refresh(_decision_lookup(), market.get_klines)
    except HTTPException:
        raise
    except Exception:  # noqa: BLE001
        log.exc("decisions endpoint")
        raise HTTPException(500, "ساختِ جدولِ تصمیم‌ها ناموفق بود — جزئیات در لاگِ سرور است؛ کمی بعد دوباره تلاش کنید")


class OpenReq(BaseModel):
    symbol: str
    tf: str = "1h"
    size_usdt: float = 1000.0
    side: str | None = None       # اگر خالی باشد از پیشنهاد تحلیل استفاده می‌شود
    force: bool = False           # ورود آگاهانه با وجود حرکت قیمت از کندل سیگنال
    force_risk: bool = False      # نادیده‌گرفتنِ سقفِ ریسکِ پرتفوی


MAX_PORTFOLIO_HEAT = 3.0          # حداکثر جمعِ ریسکِ حدضررها به‌صورت درصدِ موجودی
MAX_SIDE_HEAT = 1.8               # سقفِ ریسکِ هم‌جهت؛ بازار کریپتو معمولاً یک بتای مشترک دارد
MAX_SAME_SIDE = 3                 # حداکثر پوزیشنِ هم‌جهتِ باز (کنترلِ همبستگی)


def _portfolio_guard(new_side, new_risk_usd):
    """کنترلِ ریسکِ سطحِ پرتفوی: گرمایِ کل + تعدادِ هم‌جهت. رشتهٔ هشدار یا None."""
    db = paper.list_positions()
    w = paper.wallet_summary()
    open_risk = sum(p["size_usdt"] * abs(p["entry"] - p["sl"]) / max(p["entry"], 1e-12) for p in db["open"])
    heat_after = (open_risk + new_risk_usd) / max(w["equity"], 1e-9) * 100
    side_risk = sum(
        p["size_usdt"] * abs(p["entry"] - p["sl"]) / max(p["entry"], 1e-12)
        for p in db["open"] if p.get("side") == new_side)
    side_heat_after = (side_risk + new_risk_usd) / max(w["equity"], 1e-9) * 100
    same_side = sum(1 for p in db["open"] if p["side"] == new_side)
    if heat_after > MAX_PORTFOLIO_HEAT:
        return (f"HEAT|{heat_after:.1f}|جمعِ ریسکِ حدضررها با این معامله به {heat_after:.1f}٪ موجودی می‌رسد "
                f"(سقفِ امن {MAX_PORTFOLIO_HEAT:.0f}٪). چند پوزیشنِ همبسته = یک شرطِ بزرگ. حجم را کم کنید یا آگاهانه تأیید کنید.")
    if side_heat_after > MAX_SIDE_HEAT:
        return (f"HEAT|{side_heat_after:.1f}|ریسکِ پوزیشن‌های هم‌جهت با این معامله به {side_heat_after:.1f}٪ "
                f"موجودی می‌رسد (سقف جهت‌دار {MAX_SIDE_HEAT:.1f}٪). اندازه را کم کنید یا سمت دیگر را بررسی کنید.")
    if same_side >= MAX_SAME_SIDE:
        _sd = "خرید" if new_side == "long" else "فروش"
        return (f"HEAT|{same_side}|از قبل {same_side} پوزیشنِ {_sd} باز دارید — این‌ها همبسته‌اند و با هم می‌بازند. "
                f"تنوع بدهید یا آگاهانه تأیید کنید.")
    return None


@app.post("/api/positions")
def open_pos(req: OpenReq):
    if req.tf not in TFS:
        raise HTTPException(400, "تایم‌فریم نامعتبر")
    a = get_analysis(req.symbol, req.tf, max_age=90)
    if "error" in a:
        raise HTTPException(502, a["error"])
    tr = a["trade"]
    side = req.side or tr.get("side")
    if side not in ("long", "short"):
        raise HTTPException(400, "ستاپ معتبری برای این نماد/تایم‌فریم وجود ندارد: " + "، ".join(tr.get("reasons", [])))
    # 🛡 سپرِ آمار: معامله‌ای که مدلِ خودمان علیه آن است، فقط با تأییدِ دوبارهٔ آگاهانه باز می‌شود
    #    (درسِ ۵ ژوئیه: ۱۳ سیگنال با میانگین p_win=۴۲٪ باز شد و ۱۰تایش باخت — این مسیر بدون سد بود)
    if not req.force:
        weak = []
        pw = tr.get("p_win")
        if side == tr.get("side"):
            if not tr.get("tradeable", tr.get("viable", True)):
                weak.append("سیگنال مسدود است — " + (tr.get("status") or ""))
            elif pw is not None and pw < 45:
                weak.append(f"احتمال بردِ مدل فقط {pw:.0f}٪ است")
        else:
            # فقط p_upِ مدلِ جهت‌یابِ کالیبره «اطمینان» است و آستانهٔ ۵۸٪ دارد؛ امتیازِ اکتشافی
            # (p_up_kind="heuristic") احتمال نیست و نباید با «٪» و «آمار علیه» به کاربر نشان داده شود
            p_up = advisor.model_p_up(a)
            conf = None if p_up is None else (p_up if side == "long" else 100 - p_up)
            if conf is not None and conf < 58:
                weak.append(f"اطمینانِ جهت‌یاب برای این سمت فقط {conf:.0f}٪ است (آستانهٔ اطمینان ۵۸٪)")
            if tr.get("side") and side != tr["side"]:
                weak.append(f"جهتِ انتخابی خلافِ ستاپِ پیشنهادیِ فعلی ({'خرید' if tr['side'] == 'long' else 'فروش'}) است")
        if weak:
            raise HTTPException(409, "WEAK|" + (f"{pw:.0f}" if pw is not None else "؟") + "|" +
                                "؛ ".join(weak) + " — آمار علیه این معامله است.")
    if side == tr.get("side"):
        if not tr.get("tradeable", tr.get("viable", True)) and not req.side and not req.force:
            raise HTTPException(400, "این ستاپ مسدود است: " + tr.get("status", "") +
                                " — اگر با مسئولیت خودتان می‌خواهید، جهت را دستی انتخاب کنید")
        entry, sl, tp = tr["entry"], tr["sl"], tr["tp"]
        grade = tr.get("grade", "")
        tstop = tr["time_stop_min"]
    else:  # کاربر دستی جهت انتخاب کرده (خلاف پیشنهاد یا بدون ستاپ) — حدها بازسازی می‌شوند
        entry = a["price"]
        d = 1 if side == "long" else -1
        if tr.get("entry") is not None and tr.get("sl") is not None:
            r = abs(tr["entry"] - tr["sl"])
        else:                                   # ستاپی نبود: ریسک پیش‌فرض متناسب تایم‌فریم
            r = entry * tf_spec.MANUAL_RISK_FRAC[req.tf]
        sl, tp = entry - d * r, entry + d * 1.8 * r
        grade = "دستی"
        tstop = tr.get("time_stop_min", 40 * tf_spec.minutes(req.tf))

    # ⚠️ رفع باگ ورود: تحلیلْ کش است (کندل بسته قبلی) — ورود باید با قیمت زندهٔ لحظه کلیک باشد
    d = 1 if side == "long" else -1
    try:
        live = market.last_price(req.symbol)
    except Exception:  # noqa: BLE001
        live = None
    if live:
        drift_pct = abs(live - entry) / max(entry, 1e-12) * 100
        cap = DRIFT_CAP[req.tf]
        if drift_pct > cap and not req.force:
            raise HTTPException(409, f"DRIFT|{drift_pct:.1f}|قیمت از کندلِ سیگنال {drift_pct:.1f}٪ حرکت کرده (حد مجاز {req.tf}: {cap}٪) و شرایط ورود بدتر از بک‌تست است — برای ورود آگاهانه دوباره تأیید کنید")
        r_dist, tp_dist = abs(entry - sl), abs(tp - entry)
        entry = live
        sl = live - d * r_dist
        tp = live + d * tp_dist

    # کنترلِ ریسکِ سطحِ پرتفوی (تیر ۴) — جلوگیری از انباشتِ پوزیشن‌های همبسته
    new_risk_usd = req.size_usdt * abs(entry - sl) / max(entry, 1e-12)
    if not req.force_risk:
        warn = _portfolio_guard(side, new_risk_usd)
        if warn:
            raise HTTPException(409, warn)

    cfg = broker.load_cfg()
    bk = broker.make_broker(cfg)
    if bk is not None:
        # فیوچرزِ یک‌طرفه پوزیشن‌های هم‌نماد را یکی می‌کند: بستنِ معاملهٔ دستی پوزیشنِ
        # قاعدهٔ روند را هم می‌بست و سنجشِ اجرا را خراب می‌کرد
        held = {t["sym"] for t in trend_exec.replay()[0].values() if t["status"] == "open"}
        if req.symbol in held:
            raise HTTPException(409, f"{req.symbol} الان در دستِ اجرای روندِ تست‌نت است — "
                                     "معاملهٔ دستی روی همین نماد پوزیشنِ آن را هم تغییر می‌دهد")
        try:
            res = bk.open_bracket(req.symbol, side, req.size_usdt, sl, tp)
        except broker.TestnetError as e:
            raise HTTPException(502, str(e))
        pos = paper.open_position(req.symbol, req.tf, side, res["price"], res["sl"], res["tp"],
                                  req.size_usdt, tstop, grade, mode="testnet", qty=res["qty"],
                                  cost_pct=_symbol_cost(req.symbol))
        return {"ok": True, "position": pos, "broker": "binance_testnet"}
    pos = paper.open_position(req.symbol, req.tf, side, entry, sl, tp, req.size_usdt, tstop, grade,
                              cost_pct=_symbol_cost(req.symbol))
    return {"ok": True, "position": pos, "broker": "local"}


def _opened_ms(pos):
    return datetime.fromisoformat(pos["opened_at"]).timestamp() * 1000


def _sync_testnet(bk):
    """هم‌گام‌سازی متادیتای محلی با پوزیشن‌های واقعی تست‌نت (بسته‌شدن هدف/حدضرر سمت صرافی)."""
    db = paper.list_positions()
    try:
        ex = {p["symbol"]: p for p in bk.positions()}
    except broker.TestnetError:
        return
    for pos in list(db["open"]):
        if pos.get("mode") != "testnet":
            continue
        live = ex.get(pos["symbol"])
        if live is None:                    # صرافی بسته — هدف یا حدضرر خورده
            realized = bk.realized_pnl_since(pos["symbol"], _opened_ms(pos))
            reason = "هدف ✅" if realized > 0 else "حدضرر" if realized < 0 else "بسته شد"
            paper.close_with(pos["id"], pos.get("last_price") or pos["entry"], reason + " · تست‌نت", realized)
            continue
        age_min = (datetime.now(timezone.utc) - datetime.fromisoformat(pos["opened_at"])).total_seconds() / 60
        if age_min >= pos["time_stop_min"]:
            try:
                bk.close_symbol(pos["symbol"])
                realized = bk.realized_pnl_since(pos["symbol"], _opened_ms(pos))
                paper.close_with(pos["id"], live["mark"], "حد زمانی · تست‌نت", realized)
            except broker.TestnetError:
                log.exc()
        else:
            pct = live["pnl_usdt"] / max(pos["size_usdt"], 1e-9) * 100
            paper.set_live(pos["id"], live["mark"], live["pnl_usdt"], pct)


@app.get("/api/positions")
def positions():
    cfg = broker.load_cfg()
    bk = broker.make_broker(cfg)
    if bk is not None:
        _sync_testnet(bk)
    db = paper.list_positions()
    prices = {}
    for pos in db["open"]:
        if pos.get("mode") != "testnet" and pos["symbol"] not in prices:
            try:
                prices[pos["symbol"]] = market.last_price(pos["symbol"])
            except Exception:  # noqa: BLE001
                log.exc()
    # شبیه‌سازِ واقع‌گرا: ویکِ کندل‌ها + لغزشِ استاپ + فاندینگ (نه فقط قیمتِ نمونه‌برداری‌شده)
    db = paper.refresh(prices, klines_fn=market.get_klines_cached,
                       funding_fn=market.get_funding_history)
    # 🧭 مشاورِ پوزیشن: برای هر پوزیشنِ باز، توصیهٔ مدیریتِ زنده — تیلت فقط با مدلِ جهت‌یابِ کالیبره
    #    (مثلِ autotrader)؛ امتیازِ اکتشافیِ p_up به مشاور نمی‌رسد و خروجی «فقط هندسه» است
    for pos in db["open"]:
        try:
            a = get_analysis(pos["symbol"], pos["tf"])
            b = get_analysis("BTCUSDT", pos["tf"])
            bz = b.get("z") if "error" not in b else None
            model = ({"p_up": a.get("p_up"), "p_calibrated": True}
                     if "error" not in a and a.get("p_calibrated") else None)
            pos["advice"] = advisor.advise(pos, model, btc_z=bz)
        except Exception:  # noqa: BLE001 — توصیه هرگز نباید نمایشِ پوزیشن را بشکند
            pos["advice"] = None
    total_open = round(sum(p["pnl_usdt"] for p in db["open"]), 2)
    total_closed = round(sum(p["pnl_usdt"] for p in db["closed"]), 2)
    # ⚖️ دیدِ پرتفوی: چند پوزیشنِ هم‌جهت در بازارِ مخالف = یک شرطِ بزرگِ همبسته (درسِ ۹ لانگِ هم‌زمان)
    note = None
    try:
        longs = sum(1 for p in db["open"] if p["side"] == "long")
        shorts = len(db["open"]) - longs
        br = (_market_state("1d") or {}).get("breadth", 0.0)
        share = round((br / 2 + 0.5) * 100)
        if longs >= 4 and br < -0.3:
            note = (f"⚖️ {longs} پوزیشنِ خرید هم‌زمان داری در حالی که فقط {share}٪ بازار بالای EMA50 است — "
                    f"این‌ها عملاً یک شرطِ همبسته‌اند و با هم می‌بازند؛ کم‌کردنِ حجم یا بستنِ ضعیف‌ترها را جدی بگیر.")
        elif shorts >= 4 and br > 0.3:
            note = (f"⚖️ {shorts} پوزیشنِ فروش هم‌زمان داری در بازاری که {share}٪ آن بالای EMA50 است — "
                    f"تمرکزِ همبسته خلافِ جریان؛ کم‌کردنِ حجم را جدی بگیر.")
    except Exception:  # noqa: BLE001
        log.exc()
    return {"open": db["open"], "closed": db["closed"][:30],
            "total_open_pnl": total_open, "total_closed_pnl": total_closed,
            "portfolio_note": note,
            "broker": cfg.get("broker", "local"), "wallet": paper.wallet_summary()}


@app.get("/api/wallet")
def wallet():
    return paper.wallet_summary()


class WalletReq(BaseModel):
    start_balance: float | None = None
    keep_history: bool = False


@app.post("/api/wallet/reset")
def wallet_reset(req: WalletReq):
    return {"ok": True, "wallet": paper.reset_wallet(req.start_balance, req.keep_history)}


@app.post("/api/wallet/balance")
def wallet_balance(req: WalletReq):
    if not req.start_balance or req.start_balance < 100:
        raise HTTPException(400, "موجودی اولیه حداقل ۱۰۰ دلار")
    return {"ok": True, "wallet": paper.set_start_balance(req.start_balance)}


@app.post("/api/positions/{pos_id}/breakeven")
def move_breakeven(pos_id: str):
    """🛡️ انتقالِ حدضرر به نقطهٔ ورود — معامله دیگر نمی‌تواند به ضرر تبدیل شود."""
    db = paper.list_positions()
    target = next((p for p in db["open"] if p["id"] == pos_id), None)
    if not target:
        raise HTTPException(404, "پوزیشن باز پیدا نشد")
    if target.get("mode") == "testnet":
        raise HTTPException(400, "حدضررِ پوزیشنِ تست‌نت روی صرافی است — از پنل صرافی جابه‌جا کنید")
    pos = paper.move_sl(pos_id, target["entry"])
    if not pos:
        raise HTTPException(400, "حدضرر همین حالا در سربه‌سر یا بهتر است")
    return {"ok": True, "position": pos}


@app.post("/api/positions/{pos_id}/close")
def close_pos(pos_id: str):
    db = paper.list_positions()
    target = next((p for p in db["open"] if p["id"] == pos_id), None)
    if not target:
        raise HTTPException(404, "پوزیشن پیدا نشد")
    if target.get("mode") == "testnet":
        bk = broker.make_broker()
        if bk is None:
            raise HTTPException(400, "کلید تست‌نت پیکربندی نشده است")
        try:
            bk.close_symbol(target["symbol"])
            realized = bk.realized_pnl_since(target["symbol"], _opened_ms(target))
            price = bk.mark_price(target["symbol"])
        except broker.TestnetError as e:
            raise HTTPException(502, str(e))
        pos = paper.close_with(pos_id, price, "دستی · تست‌نت", realized)
        return {"ok": True, "position": pos}
    try:
        price = market.last_price(target["symbol"])
    except Exception:  # noqa: BLE001 — جفتِ حذف‌شده تیکِر ندارد؛ با آخرین قیمتِ ثبت‌شده ببند
        price = target.get("last_price") or target["entry"]
    pos = paper.close_manual(pos_id, price)
    return {"ok": True, "position": pos}


class ConfigReq(BaseModel):
    broker: str = "local"                 # "local" یا "binance_testnet"
    api_key: str = ""
    api_secret: str = ""
    leverage: int = 2


@app.get("/api/config")
def get_config():
    cfg = broker.load_cfg()
    out = {"broker": cfg.get("broker", "local"), "leverage": cfg.get("leverage", 2),
           "has_keys": bool(cfg.get("api_key") and cfg.get("api_secret"))}
    if out["broker"] == "binance_testnet" and out["has_keys"]:
        try:
            avail, total = broker.make_broker(cfg).balance_usdt()
            out.update(connected=True, balance=round(avail, 2), balance_total=round(total, 2))
        except Exception as e:  # noqa: BLE001
            out.update(connected=False, error=str(e))
    return out


@app.post("/api/config")
def set_config(req: ConfigReq):
    if req.broker not in ("local", "binance_testnet"):
        raise HTTPException(400, "بروکر نامعتبر")
    cfg = broker.load_cfg()
    cfg["broker"] = req.broker
    cfg["leverage"] = max(1, min(req.leverage, 10))
    if req.api_key.strip():
        cfg["api_key"] = req.api_key.strip()
    if req.api_secret.strip():
        cfg["api_secret"] = req.api_secret.strip()
    if req.broker == "binance_testnet":
        if not (cfg.get("api_key") and cfg.get("api_secret")):
            raise HTTPException(400, "کلید و سکرت تست‌نت لازم است — از testnet.binancefuture.com بگیرید")
        try:
            avail, _ = broker.BinanceTestnet(cfg["api_key"], cfg["api_secret"], cfg["leverage"]).balance_usdt()
        except Exception as e:  # noqa: BLE001
            raise HTTPException(502, "اتصال به تست‌نت ناموفق: " + str(e))
        broker.save_cfg(cfg)
        return {"ok": True, "connected": True, "balance": round(avail, 2)}
    broker.save_cfg(cfg)
    return {"ok": True, "connected": False}


STATIC = os.path.join(os.path.dirname(__file__), "static")


@app.get("/")
@app.get("/lab")
def index():
    """صفحهٔ اصلی: محیطِ معامله (جدولِ ارزها، تایم‌فریم‌ها، پوزیشن‌ها). /lab همان صفحه است تا نشانی‌های قبلی کار کنند."""
    return FileResponse(os.path.join(STATIC, "index.html"))


@app.get("/desk")
def desk():
    """صفحهٔ سادهٔ تصمیمِ هفتگی و ابزارهای کوکوین — دیگر در منو نیست، فقط با همین نشانی."""
    return FileResponse(os.path.join(STATIC, "app.html"))


app.mount("/static", StaticFiles(directory=STATIC), name="static")


@app.middleware("http")
async def _no_stale_pages(request, call_next):
    """بی‌هدرِ Cache-Control، مرورگر خودش حدس می‌زد صفحه تا کی تازه است؛ صفحهٔ قدیمی که مدت‌ها
    عوض نشده بود ساعت‌ها از کش نشان داده می‌شد و کاربر «بعضی وقت‌ها» ظاهرِ قدیم را می‌دید.
    حالا HTML همیشه با سرور تطبیق داده می‌شود (ETag ⇒ اگر عوض نشده، پاسخِ کوچکِ 304) و API
    اصلاً کش نمی‌شود."""
    resp = await call_next(request)
    path = request.url.path
    if path.startswith("/api/"):
        resp.headers["Cache-Control"] = "no-store"
    elif path in ("/", "/lab", "/desk") or path.endswith(".html"):
        resp.headers["Cache-Control"] = "no-cache, must-revalidate"
    return resp

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("TRADERBOT_PORT") or os.environ.get("PORT") or 8787)
    print(f"🤖 ربات پایش بازار روی http://127.0.0.1:{port} بالا آمد — Ctrl+C برای توقف")
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")
