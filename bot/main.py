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
import broker
import calib
import shadow
import autotrader
import advisor
import gates
from app_meta import APP_VERSION, AI_CORE_VERSION, RELEASE_DATE
from datetime import datetime, timezone

_mstate_cache: dict = {}   # tf -> (ts, {"breadth","dom","ethbtc"})
_susp_cache = {"ts": 0.0, "map": {}}


def _market_state(tf):
    """وضعیتِ کلِ بازار به‌صورت زنده (پهنای بازار، مومنتوم دامیننس BTC، مومنتوم ETH/BTC) — فقط از کشِ کندل‌ها."""
    hit = _mstate_cache.get(tf)
    if hit and time.time() - hit[0] < 300:
        return hit[1]
    out = {"breadth": 0.0, "dom": 0.0, "ethbtc": 0.0}
    try:
        symbols, _ = market.get_top_symbols(calib.CALIB_UNIVERSE_N)   # همان جمعیتِ آموزش
        above = []
        volq: dict = {}
        btcv: dict = {}
        for s in symbols:
            kl = market.get_klines_cached(s, tf)
            if not kl or len(kl["c"]) < 60:
                continue
            cc = np.array(kl["c"], float)
            e50 = engine.ema(cc, 50)
            above.append(1.0 if cc[-1] > e50[-1] else 0.0)
            for t, c_, v_ in zip(kl["t"], kl["c"], kl["v"]):
                volq[t] = volq.get(t, 0.0) + c_ * v_
                if s == "BTCUSDT":
                    btcv[t] = c_ * v_
        if len(above) >= universe.MIN_POPULATION:
            out["breadth"] = (sum(above) / len(above) - 0.5) * 2
        dom_ts = sorted(t for t in btcv if volq.get(t, 0) > 0)
        if len(dom_ts) > 60:
            dm = calib.mom_norm_map(dom_ts, [btcv[t] / volq[t] for t in dom_ts])
            if dm:
                out["dom"] = list(dm.values())[-1]
        kb, ke = market.get_klines_cached("BTCUSDT", tf), market.get_klines_cached("ETHUSDT", tf)
        if kb and ke:
            bb_ = dict(zip(kb["t"], kb["c"]))
            ee_ = dict(zip(ke["t"], ke["c"]))
            common = [t for t in ke["t"] if t in bb_]
            if len(common) > 60:
                em = calib.mom_norm_map(common, [ee_[t] / bb_[t] for t in common])
                if em:
                    out["ethbtc"] = list(em.values())[-1]
    except Exception:  # noqa: BLE001
        log.exc()
    _mstate_cache[tf] = (time.time(), out)
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


def _suspended_setups():
    """تعلیق سراسری فقط برای ستاپ‌های واقعاً سمی.
    تعلیقِ خفیفِ سراسری (مثل zx با −۰٫۰۱R) جیب‌های سوددهٔ تایم‌فریم‌محور را می‌کشت."""
    if time.time() - _susp_cache["ts"] < 120:
        return _susp_cache["map"]
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
    try:
        import edge_book
        pockets, suspended, _ = edge_book.refresh()
        return pockets, suspended
    except Exception:  # noqa: BLE001
        return {}, {}


def _suspended_tfs():
    """تایم‌فریم‌هایی که بازده خالص زندهٔ ۳۰روزه‌شان با n کافی مثبت نیست — معلق تا بهبود.
    اگر همان TF جیب سوددهٔ ترکیبی داشته باشد، تعلیقِ کل TF اعمال نمی‌شود."""
    if time.time() - _tf_susp_cache["ts"] < 120:
        return _tf_susp_cache["map"]
    m = {}
    pockets, _ = _live_edge_book()
    pocket_tfs = {p["tf"] for p in pockets.values()}
    try:
        for tf in ("15m", "1h", "4h", "1d"):
            if tf in pocket_tfs:
                continue
            st = shadow.stats(tf, days=30, since_ts=_model_era(tf))
            if st.get("n", 0) >= 30 and st.get("avg_r") is not None and st["avg_r"] <= 0.0:
                m[tf] = st["avg_r"]
    except Exception:  # noqa: BLE001
        log.exc()
    _tf_susp_cache.update(ts=time.time(), map=m)
    return m

HTF_OF = {"15m": "4h", "1h": "4h", "4h": "1d", "1d": None}
_rs_cache: dict = {}      # tf -> (ts, {sym: rank01})
_fz_cache: dict = {}      # sym -> (ts, {"z", "persist", "crowded_long", "crowded_short"})
_fz_pending: set = set()
_oi_cache: dict = {}      # sym -> (ts, oi_stats)
_oi_pending: set = set()
_btc_macro_cache = {"ts": 0.0, "data": None}
_macro_cache: dict = {}   # tf -> (ts, {"gold": x})


def _rs_rank(tf):
    """رتبه قدرت نسبی ۰..۱ هر ارز بین تاپ نمادها (بازده ۲۰ کندلی)."""
    now = time.time()
    hit = _rs_cache.get(tf)
    if hit and now - hit[0] < 180:
        return hit[1]
    ranks = {}
    try:
        # همان جمعیتِ آموزش: ۱۰۰ ارزِ برتر، با حداقلِ جمعیت (وگرنه همه خنثی).
        # قبلاً روی کشِ نیمه‌خالیِ پس از ری‌استارت، چند ارز رتبهٔ ۰ یا ۱ می‌گرفتند.
        symbols, _ = market.get_top_symbols(calib.CALIB_UNIVERSE_N)
        rets = {}
        for s in symbols:
            kl = market.get_klines_cached(s, tf)
            if kl and len(kl["c"]) > 21:
                rets[s] = kl["c"][-1] / kl["c"][-21] - 1
        ranks = universe.rank_within(rets)
    except Exception:  # noqa: BLE001
        log.exc()
    _rs_cache[tf] = (now, ranks)
    return ranks


def _macro(tf):
    """مومنتوم زندهٔ طلا (PAXG) — از همان منبع و نرمال‌سازیِ آموزش (بدون skew).

    شاخصِ دلار حذف شد: فایلِ dxy_daily.json هرگز پر نشد و ویژگی در تمامِ آموزش
    ثابتِ ۰ بود؛ زنده‌کردنش بعداً یعنی دادنِ ورودیِ ندیده به مدل.
    """
    now = time.time()
    hit = _macro_cache.get(tf)
    if hit and now - hit[0] < 900:
        return hit[1]
    out = {"gold": 0.0}
    try:
        gk = market.get_history("PAXGUSDT", tf, calib.BARS.get(tf, 3000))
        gmap = calib.mom_norm_map(gk["t"], gk["c"])
        if gmap:
            out["gold"] = list(gmap.values())[-1]
    except Exception:  # noqa: BLE001
        log.exc()
    _macro_cache[tf] = (now, out)
    return out


def _btc_macro():
    """رژیم کلان BTC روی روزانه + پهنای 1d — کش ۱۵ دقیقه‌ای."""
    import meta_gate
    now = time.time()
    if _btc_macro_cache["data"] and now - _btc_macro_cache["ts"] < 900:
        return _btc_macro_cache["data"]
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
TFS = ["15m", "1h", "4h", "1d"]
TOP_N = 200               # تعداد ارزهای برتر تحت پایش
CALIB_TOP = 100           # استخر آموزش مدل‌ها (پرحجم‌ترین‌ها) — با آموزشِ موازی، داده بیشتر = مدل قوی‌تر
ANALYSIS_TTL = {"15m": 60, "1h": 120, "4h": 300, "1d": 900}
# حداکثر فاصله مجاز قیمت زنده از کندلِ سیگنال (٪) — متناسب با نوسان طبیعی هر تایم‌فریم
DRIFT_CAP = {"15m": 1.2, "1h": 2.5, "4h": 5.0, "1d": 9.0}

_an_lock = threading.Lock()
_an_cache: dict = {}      # (symbol, tf) -> (ts, analysis | {"error": str})
_inflight: dict = {}      # (symbol, tf) -> threading.Event — جلوگیری از محاسبه تکراری هم‌زمان
_pool = ThreadPoolExecutor(max_workers=8)


def get_analysis(symbol: str, tf: str, max_age=None):
    key = (symbol, tf)
    while True:
        with _an_lock:
            hit = _an_cache.get(key)
            if hit:
                now = time.time()
                boundary = now - (now % (market.TF_MINUTES[tf] * 60))
                # تحلیل تا پایان کندل جاری معتبر است (+۴۵ ثانیه مهلت بعد از بسته‌شدن)
                if hit[0] >= boundary or now - hit[0] < (max_age or 45):
                    return hit[1]
            other = _inflight.get(key)
            if other is None:
                ev = threading.Event()
                _inflight[key] = ev
                break                          # این نخ مسئول محاسبه است
        other.wait(timeout=180)                # نخ دیگری در حال محاسبه همین کلید است
        with _an_lock:
            hit = _an_cache.get(key)
            if hit:
                return hit[1]
            if _inflight.get(key) is other:    # محاسبه‌گر قبلی بی‌نتیجه ماند
                _inflight.pop(key, None)
    try:
        btc_z = None
        if symbol != "BTCUSDT":
            btc = get_analysis("BTCUSDT", tf)          # عمق بازگشت حداکثر ۱
            btc_z = btc.get("z") if "error" not in btc else None   # همان متغیر آموزش (z نه score)
        htf_sign = 0
        htf = HTF_OF.get(tf)
        if htf:
            ha = get_analysis(symbol, htf)             # زنجیره: 15m/1h→4h→1d→پایان
            if "error" not in ha and ha.get("zt") is not None:
                # هم‌ترازی دقیق با آموزش: کندلِ بستهٔ *قبل از* سطلِ تایم‌بالاترِ سیگنال (بدون نشتی/کندلِ در حال شکل‌گیری)
                htf_p = market.TF_MINUTES[htf] * 60000
                try:
                    ts_now = market.get_klines(symbol, tf)["t"][-1]
                    want = (ts_now // htf_p) * htf_p - htf_p
                    hz = ha.get("z_prev", 0.0) if ha["zt"] > want else ha.get("z", 0.0)
                except Exception:  # noqa: BLE001
                    hz = ha.get("z", 0.0)
                htf_sign = 1 if (hz or 0) > 0.3 else -1 if (hz or 0) < -0.3 else 0
        finfo = _funding_info(symbol)
        oinfo = _oi_info(symbol)
        extras = {# ویژگیِ مدل با فرمولِ آموزش ساخته می‌شود، نه با z شلوغیِ متا-گیت
                  "funding_z": features.live_funding_z(symbol),
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
        extras.update(_macro(tf))
        extras.update(_market_state(tf))
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
        kl = market.get_klines(symbol, tf)
        # ⚰️ گاردِ فیدِ مرده: جفتِ حذف‌شده/بی‌معامله چارتِ یخ‌زده دارد و سیگنالش بی‌معناست (درسِ MATIC/DNT)
        if time.time() * 1000 - kl["t"][-1] > market.TF_MINUTES[tf] * 60000 * 3:
            age_d = (time.time() * 1000 - kl["t"][-1]) / 86400000
            res = {"symbol": symbol, "tf": tf,
                   "error": f"فیدِ داده مرده است (آخرین کندل {age_d:.0f} روز پیش) — جفتِ حذف‌شده یا بدونِ معامله"}
        else:
            res = engine.analyze(kl, tf, btc_z=btc_z, predict_fn=calib.predict,
                                 extras=extras, dir_fn=calib.predict_dir,
                                 action_fn=calib.predict_action)
        res["symbol"] = symbol
    except Exception as e:  # noqa: BLE001
        res = {"symbol": symbol, "tf": tf, "error": str(e)}
    with _an_lock:
        _an_cache[key] = (time.time(), res)
        _inflight.pop(key, None)
    ev.set()
    return res


def _calib_builder():
    """ساخت/تازه‌سازی جدول کالیبراسیون در پس‌زمینه (روزی یک‌بار)."""
    try:
        symbols, _ = market.get_top_symbols(CALIB_TOP)
        calib.build(symbols)
        with _an_lock:
            _an_cache.clear()          # تحلیل‌ها با احتمال کالیبره از نو
    except Exception:  # noqa: BLE001
        log.exc()


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
    autotrader.init(overview, DRIFT_CAP, get_analysis)     # 🤖 ربات معامله‌گر خودکار
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
    return calib.status()


@app.get("/api/gates")
def gates_status():
    """قفلِ ایمنیِ سرمایه: ترکیب‌های مجاز، وضعیتِ لایو و سقف‌های ریسک (فقط‌خواندنی)."""
    return gates.status()


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

    cs = calib.status()
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
        for tf, row in (out["data"].get("klines_by_tf") or {}).items():
            limit = 3 * market.KLINE_TTL.get(tf, 300)
            if row.get("oldest_age_sec", 0) > limit:
                warnings.append(f"کشِ کندلِ {tf} کهنه است ({row['oldest_age_sec'] / 60:.0f} دقیقه)")
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
        recent = log.tail(200, level="warning")
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
    cs = calib.status()
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
    if calib.status().get("building"):
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
    if tf and st.get("n", 0) >= 20 and st.get("avg_r") is not None and st["avg_r"] <= 0:
        drift = (f"⚠️ عملکرد زندهٔ {tf} ({st['avg_r']:+.2f}R از {st['n']} سیگنال) "
                 "پس از هزینه مثبت نیست — ورودهای تازهٔ این تایم‌فریم متوقف می‌شوند.")
    return {"tf": tf, "stats": st, "all": shadow.stats(None), "drift": drift,
            "suspended": _suspended_setups(),
            "edge_pockets": _live_edge_book()[0],
            "suspended_combos": _live_edge_book()[1],
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
        h = edge_book.get_health()
        fq = fill_quality.summary(14)
        return {"ok": True, "health": h, "fill_quality": fq,
                "scan_hint": edge_book.scan_tfs_for(["1h", "4h", "15m"]),
                "version": {"app": APP_VERSION, "ai_core": AI_CORE_VERSION}}
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "error": str(e)}


@app.get("/api/overview")
def overview(tf: str = "1h"):
    if tf not in TFS:
        raise HTTPException(400, "تایم‌فریم نامعتبر")
    try:
        symbols, tickers = market.get_top_symbols(TOP_N + 6)   # چند ارز اضافه، تا کم‌سابقه‌ها جایگزین داشته باشند
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
                    if dr > DRIFT_CAP.get(tf, 2.5):
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
    cs = calib.status()
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
            "btc_macro": _btc_macro(),
            "breadth_macro": round((_market_state("1d") or {}).get("breadth", 0.0), 2),
            "calib": {"building": cs.get("building"), "progress": cs.get("progress"),
                      "age_hours": cs.get("age_hours"), "events": (cs.get("events") or {}).get(tf)}}


@app.get("/api/coin/{symbol}")
def coin_detail(symbol: str):
    results = dict(zip(TFS, _pool.map(lambda tf: get_analysis(symbol, tf), TFS)))
    return {"symbol": symbol, "tfs": results}


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
            p_up = a.get("p_up")
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
            r = entry * {"15m": 0.006, "1h": 0.008, "4h": 0.012, "1d": 0.02}.get(req.tf, 0.01)
        sl, tp = entry - d * r, entry + d * 1.8 * r
        grade = "دستی"
        tstop = tr.get("time_stop_min", 40 * {"15m": 15, "1h": 60, "4h": 240, "1d": 1440}.get(req.tf, 60))

    # ⚠️ رفع باگ ورود: تحلیلْ کش است (کندل بسته قبلی) — ورود باید با قیمت زندهٔ لحظه کلیک باشد
    d = 1 if side == "long" else -1
    try:
        live = market.last_price(req.symbol)
    except Exception:  # noqa: BLE001
        live = None
    if live:
        drift_pct = abs(live - entry) / max(entry, 1e-12) * 100
        cap = DRIFT_CAP.get(req.tf, 2.5)
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
    # 🧭 مشاورِ پوزیشن: برای هر پوزیشنِ باز، توصیهٔ مدیریتِ زنده با مدلِ کالیبره
    for pos in db["open"]:
        try:
            a = get_analysis(pos["symbol"], pos["tf"])
            b = get_analysis("BTCUSDT", pos["tf"])
            bz = b.get("z") if "error" not in b else None
            pos["advice"] = advisor.advise(
                pos, {"p_up": a.get("p_up")} if "error" not in a else None, btc_z=bz)
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
def index():
    return FileResponse(os.path.join(STATIC, "index.html"))


app.mount("/static", StaticFiles(directory=STATIC), name="static")

if __name__ == "__main__":
    import uvicorn
    print("🤖 ربات پایش بازار روی http://127.0.0.1:8787 بالا آمد — Ctrl+C برای توقف")
    uvicorn.run(app, host="127.0.0.1", port=8787, log_level="warning")
