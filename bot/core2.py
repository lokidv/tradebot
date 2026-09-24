# -*- coding: utf-8 -*-
"""هستهٔ v2 — walk-forwardِ پیش‌ثبت‌شدهٔ LightGBM (``research/prereg_core2.json``؛ الزام‌آور).

برای هر تایم‌فریم دو رگرسور (M_long و M_short) امیدِ R خالص را پیش‌بینی می‌کنند:
ویژگی‌ها از ``core_feats`` روی کندل‌های **اسپات** (+ زمینهٔ BTC/ETH/پهنا/فاندینگِ کوکوین)، برچسب از
``core_feats.labels`` روی کندل‌های **فیوچرز** با همان زمان (سطرِ بی‌کندلِ فیوچرز حذف)، پنج ارز با وزنِ
کلِ برابر، در 5m هر سطرِ دوم، float32.

walk-forward: برای هر ماهِ M یک برازشِ تازه با سطرهای ``t_i ≤ start(M) − 81 کندل`` که خروجِ برچسبشان
پیش از ``start(M)`` بسته شده؛ توقفِ زودهنگام روی ۱۵٪ آخرِ زمانیِ آموزش (فاصلهٔ ۴۱ کندل، صبر ۱۰۰، حداکثر
۱۵۰۰ دور) و سپس برازشِ دوباره روی کلِ آموزش با همان تعدادِ دور. پیش‌بینیِ هر ماه فقط از مدلِ همان ماه.

تصمیم: لانگ اگر E_L ≥ 0.05 و E_L − E_S ≥ 0.05؛ شورت قرینه؛ وگرنه صبر. شبیه‌سازیِ معامله دقیقاً مثلِ
``decision.replay`` (یک معامله در لحظه برای هر ارز × تایم‌فریم، ورود در بازِ کندلِ بعد، ۰٫۱۴٪).

اختیار: v2 هرگز ``tradeable`` را روشن نمی‌کند، به gates.json نمی‌نویسد و autotrader آن را نمی‌خواند.
پنجرهٔ holdout فقط با پرچمِ صریح و فقط برای تایم‌فریمی که گزارشِ اعتبارسنجی «پذیرفته» کرده اجرا می‌شود.
"""
import glob
import json
import math
import os
import re
import time
from datetime import datetime, timezone

import numpy as np

import bracket
import core_feats as cf
import decision
import engine
import paths
from watchlist import SYMBOLS

# ───────────────────────── ثابت‌های پیش‌ثبت (تغییر پس از دیدنِ نتیجه ممنوع) ─────────────────────────
PREREG_PATH = paths.data("research", "prereg_core2.json")
RESEARCH_DIR = paths.data("research")
MICRO_DIR = paths.data("hist_research", "micro")
OUT_DIR = paths.data("hist_research", "core2")
REPORT_PREFIX = "core2_validation_"

TFS = cf.TFS
BAR_MS = cf.BAR_MS
DAY_MS = cf.DAY_MS
WEEK_MS = 7 * DAY_MS
MONDAY0_MS = 4 * DAY_MS          # 1970-01-05 دوشنبه — مرزِ هفتهٔ تقویمی (UTC)

WINDOWS = {"discovery": ("2022-01-01", "2024-06-30"),
           "validation": ("2024-07-01", "2025-06-30"),
           "holdout": ("2025-07-01", "2026-08-31")}

PURGE_BARS = 41                  # ورود در کندلِ بعد + سقفِ ۴۰ کندل
EMBARGO_BARS = 40
GAP_BARS = PURGE_BARS + EMBARGO_BARS          # = 81
ES_FRAC = 0.15
ES_GAP_BARS = 41
ES_PATIENCE = 100
MAX_ROUNDS = 1500
SUBSAMPLE = {"5m": 2}            # «every second row at 5m» — فقط برای آموزش؛ پیش‌بینی روی همهٔ کندل‌ها
TRAIN_WARMUP = {tf: cf.ZW[tf] + 400 for tf in TFS}   # سطرهای پیش از گرم‌شدن (ZW+400) نه آموزش نه پیش‌بینی

COST = decision.COST_PCT         # 0.14
MAKER_COST = decision.MAKER_COST_PCT   # 0.10 — فقط توصیفی
MARGIN = 0.05
SENS_MARGINS = (0.0, 0.10, 0.20)      # فقط حساسیت
# ممیزی DATA-5: پیش‌تر ۴۵ (= MAX_BARS + 5) کندلِ فیوچرزِ پس از پایانِ پنجره برای بستنِ براکتِ سیگنال‌های آخر
# خوانده می‌شد، پس برچسب و معاملهٔ پایانِ اعتبارسنجی به کندل‌های holdout وابسته بود. اکنون صفر: هیچ کندلی در/پس از
# پایانِ پنجره خوانده نمی‌شود و براکتی که تا پایانِ پنجره بسته نشده نامعلوم است (برچسب NaN، معامله شمرده نمی‌شود).
LABEL_TAIL_BARS = 0

BOOT_B = 2000
BOOT_SEED = 42
HOLM_ALPHA = 0.10
ADOPT_EDGE_R = 0.05
N_MIN = {"1d": 40}               # بقیه ۱۰۰
N_MIN_DEFAULT = 100
HOLD_N_MIN = {"1d": 20}          # بقیه ۳۰
HOLD_N_MIN_DEFAULT = 30
MIN_POS_COINS = 3
MAX_COIN_SHARE = 0.5
NUM_THREADS = 24
_NO_EXIT = np.iinfo(np.int64).max


def _ms(day):
    return int(datetime.strptime(day, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp() * 1000)


def window_ms(name):
    """``(شروع, پایانِ انحصاری)`` به ms — پایان = روزِ بعد از آخرین روزِ پنجره."""
    a, b = WINDOWS[name]
    return _ms(a), _ms(b) + DAY_MS


def _date(ms):
    return datetime.fromtimestamp(int(ms) / 1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M")


def month_starts(t0, t1):
    """ماه‌های تقویمیِ ``[t0, t1)`` ⇒ فهرستِ ``(start_ms, end_ms, yyyymm)``."""
    d = datetime.fromtimestamp(t0 / 1000, tz=timezone.utc).replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    out = []
    while int(d.timestamp() * 1000) < t1:
        nxt = d.replace(year=d.year + (d.month == 12), month=d.month % 12 + 1)
        out.append((max(int(d.timestamp() * 1000), int(t0)), min(int(nxt.timestamp() * 1000), int(t1)),
                    d.year * 100 + d.month))
        d = nxt
    return out


def check_prereg(path=None):
    """پنجره‌ها و پارامترهای کلیدیِ این فایل باید عیناً همان پیش‌ثبت باشند (وگرنه خطا)."""
    with open(path or PREREG_PATH, "r", encoding="utf-8") as f:
        pr = json.load(f)
    for k, v in WINDOWS.items():
        key = "holdout_one_shot" if k == "holdout" else k
        if list(pr["windows"][key]) != list(v):
            raise ValueError(f"پنجرهٔ {k} با پیش‌ثبت یکی نیست")
    m = pr["model"]
    for s in ("huber (alpha 1.0)", "learning_rate 0.03", "num_leaves 15", "max_depth 5", "max(500, n/400)",
              "200 at 1h", "100 at 4h", "1d num_leaves 7, min_data_in_leaf 60", "feature_fraction 0.7",
              "bagging_fraction 0.7", "bagging_freq 1", "lambda_l2 10", "max_bin 63", "seed 42",
              "41-bar gap", "patience 100", "max 1500 rounds", "every second row at 5m"):
        if s not in m:
            raise ValueError(f"پارامترِ «{s}» در پیش‌ثبت پیدا نشد")
    if "month start - 81 bars" not in pr["walk_forward"] or "0.05" not in pr["decision_rule_fixed"]:
        raise ValueError("walk-forward یا قاعدهٔ تصمیم با پیش‌ثبت یکی نیست")
    return pr


def file_sha256(path):
    import hashlib
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


# ───────────────────────── مدل ─────────────────────────
def lgb_params(tf, n_train, threads=NUM_THREADS):
    """پارامترهای عیناً پیش‌ثبت. ``n_train`` = تعدادِ سطرهای آموزشِ همان ماه (برای min_data_in_leaf)."""
    p = {"objective": "huber", "alpha": 1.0, "learning_rate": 0.03, "num_leaves": 15, "max_depth": 5,
         "feature_fraction": 0.7, "bagging_fraction": 0.7, "bagging_freq": 1, "lambda_l2": 10.0,
         "max_bin": 63, "deterministic": True, "force_col_wise": True, "seed": 42,
         "num_threads": int(threads), "verbosity": -1, "metric": "l2"}
    if tf in ("5m", "15m"):
        p["min_data_in_leaf"] = max(500, int(math.ceil(n_train / 400.0)))
    elif tf == "1h":
        p["min_data_in_leaf"] = 200
    elif tf == "4h":
        p["min_data_in_leaf"] = 100
    elif tf == "1d":
        p["num_leaves"], p["min_data_in_leaf"] = 7, 60
    else:
        raise KeyError(tf)
    return p


def coin_weights(sym):
    """وزنِ کلِ برابر برای هر ارزِ حاضر؛ میانگینِ وزن = ۱."""
    sym = np.asarray(sym)
    u, inv, cnt = np.unique(sym, return_inverse=True, return_counts=True)
    return (len(sym) / (len(u) * cnt[inv])).astype(np.float64)


def es_split(t, tf):
    """اعتبارسنجیِ درونی = ۱۵٪ آخرِ سطرهای آموزش به ترتیبِ زمان؛ آموزشِ درونی = ``t ≤ t_split − 41 کندل``."""
    t = np.asarray(t, dtype=np.int64)
    ts = np.sort(t)
    t_split = int(ts[min(len(ts) - 1, int(math.floor((1.0 - ES_FRAC) * len(ts))))])
    va = t >= t_split
    tr = t <= t_split - ES_GAP_BARS * BAR_MS[tf]
    return tr, va, t_split


def _fit_side(lgb, params, X, y, sym, tr, va, names):
    """توقفِ زودهنگام روی ۱۵٪ آخر، سپس برازشِ دوباره روی همهٔ سطرها با بهترین تعدادِ دور."""
    t0 = time.time()
    dtr = lgb.Dataset(X[tr], y[tr], weight=coin_weights(sym[tr]), feature_name=names, free_raw_data=True)
    dva = lgb.Dataset(X[va], y[va], weight=coin_weights(sym[va]), reference=dtr, free_raw_data=True)
    es = lgb.train(params, dtr, num_boost_round=MAX_ROUNDS, valid_sets=[dva],
                   callbacks=[lgb.early_stopping(ES_PATIENCE, verbose=False)])
    best = int(es.best_iteration or 0) or int(es.current_iteration())
    best_score = float(es.best_score["valid_0"]["l2"]) if es.best_score else None
    t1 = time.time()
    dall = lgb.Dataset(X, y, weight=coin_weights(sym), feature_name=names, free_raw_data=True)
    full = lgb.train(params, dall, num_boost_round=best)
    t2 = time.time()
    return full, {"best_iter": best, "es_val_l2": best_score, "es_s": round(t1 - t0, 2), "refit_s": round(t2 - t1, 2)}


def fit_pair(X, yL, yS, t, sym, tf, names, threads=NUM_THREADS):
    """M_long و M_short برای یک ماه ⇒ ``(predict(X) -> (E_L, E_S), meta)``."""
    import lightgbm as lgb
    params = lgb_params(tf, len(t), threads)
    tr, va, t_split = es_split(t, tf)
    mL, metaL = _fit_side(lgb, params, X, yL, sym, tr, va, names)
    mS, metaS = _fit_side(lgb, params, X, yS, sym, tr, va, names)

    def predict(Xte):
        return (mL.predict(Xte, num_threads=int(threads)).astype(np.float32),
                mS.predict(Xte, num_threads=int(threads)).astype(np.float32))

    meta = {"min_data_in_leaf": params["min_data_in_leaf"], "num_leaves": params["num_leaves"],
            "es_split": _date(t_split), "n_es_train": int(tr.sum()), "n_es_val": int(va.sum()),
            "long": metaL, "short": metaS,
            "gain_long": mL.feature_importance("gain").astype(float).tolist(),
            "gain_short": mS.feature_importance("gain").astype(float).tolist()}
    return predict, meta


# ───────────────────────── داده ─────────────────────────
def load_bars(prefix, sym, tf, t_end=None, micro_dir=None):
    """``<prefix>_<SYM>_<tf>.npz`` ⇒ کندل‌های مرتب و بی‌تکرار؛ ``t_end`` (انحصاری) داده را همان‌جا می‌بُرد."""
    path = os.path.join(micro_dir or MICRO_DIR, f"{prefix}_{sym}_{tf}.npz")
    with np.load(path) as z:
        t = np.asarray(z["t"], dtype=np.int64)
        arr = {k: np.asarray(z[k], dtype=np.float64) for k in ("o", "h", "l", "c", "v", "qv", "n", "tbv") if k in z}
    t, uniq = np.unique(t, return_index=True)
    out = {k: v[uniq] for k, v in arr.items()}
    out["t"] = t
    if t_end is not None:
        k = int(np.searchsorted(t, int(t_end)))
        out = {kk: v[:k] for kk, v in out.items()}
    return out


def load_kfund(sym, t_end=None, micro_dir=None):
    """فاندینگِ تسویه‌شدهٔ کوکوین؛ ``t_end`` انحصاری است (تسویه‌های ``t < t_end``). برای «تا بسته‌شدنِ آخرین کندل،
    خودش هم» (همان ``≤ close`` در ``core_feats._kfund``) ``t_end = پایان + 1`` بدهید."""
    with np.load(os.path.join(micro_dir or MICRO_DIR, f"kfund_{sym}.npz")) as z:
        out = {"t": np.asarray(z["t"], dtype=np.int64), "rate": np.asarray(z["rate"], dtype=np.float64),
               "interval_h": np.asarray(z["interval_h"], dtype=np.float64)}
    if t_end is not None:
        k = int(np.searchsorted(out["t"], int(t_end)))
        out = {kk: v[:k] for kk, v in out.items()}
    return out


def build_dataset(tf, spot_end, fut_end, symbols=SYMBOLS, spot=None, fut=None, kfund=None, log=None):
    """سطرهای هر پنج ارز: ویژگیِ اسپات (کندل‌های ``< spot_end``) + برچسبِ فیوچرز (کندل‌های ``< fut_end``).

    فقط سطرهایی که کندلِ فیوچرزِ هم‌زمان دارند و از گرم‌شدن (``TRAIN_WARMUP``) گذشته‌اند. برچسبِ نامعلوم
    (NaN) می‌ماند تا سطرِ پیش‌بینی حذف نشود؛ ماسکِ آموزش آن را کنار می‌گذارد.
    ``spot``/``fut``/``kfund`` (اختیاری، برای تست) = ``{نماد: کندل‌ها}``؛ نبود ⇒ از micro/.
    فاندینگِ کوکوین تا بسته‌شدنِ آخرین کندل، خودش هم (``t ≤ spot_end``، ممیزی model-5): ``_kfund`` تسویهٔ هم‌لحظه
    با بسته‌شدن را معلوم می‌داند، پس برشِ ``t < spot_end`` سطرِ آخر را با محاسبهٔ کل‌تاریخچه/زنده متفاوت می‌کرد.
    """
    t0 = time.time()
    spot = spot or {s: load_bars("spot", s, tf, spot_end) for s in symbols}
    spot = {s: {k: v[:int(np.searchsorted(b["t"], int(spot_end)))] for k, v in b.items()} for s, b in spot.items()}
    fut = fut or {s: load_bars("um", s, tf, fut_end) for s in symbols}
    fut = {s: {k: v[:int(np.searchsorted(b["t"], int(fut_end)))] for k, v in b.items()} for s, b in fut.items()}
    if kfund is None:
        kfund = {s: load_kfund(s, int(spot_end) + 1) for s in symbols}
    cache = {}
    parts = []
    bar = BAR_MS[tf]
    for k, s in enumerate(symbols):
        B = spot[s]
        ctx = {"btc": spot.get("BTCUSDT"), "eth": spot.get("ETHUSDT"), "closes5": spot,
               "kfund": kfund.get(s), "_cache": cache}
        F = cf.compute(tf, B, ctx)
        X = cf.stack(tf, F)
        Fb = fut[s]
        yL, yS, ex, rp = cf.labels(tf, Fb, cost=COST)
        tf_ = Fb["t"]
        t = B["t"]
        j = np.searchsorted(tf_, t)
        jc = np.minimum(j, len(tf_) - 1)
        has = (j < len(tf_)) & (tf_[jc] == t)
        keep = has & (np.arange(len(t)) >= TRAIN_WARMUP[tf])
        jj = jc[keep]
        exj = ex[jj]
        exit_close = np.where(exj >= 0, tf_[np.maximum(exj, 0)] + bar, _NO_EXIT)
        parts.append({"sym": np.full(int(keep.sum()), k, dtype=np.int8), "t": t[keep], "X": X[keep],
                      "yL": yL[jj], "yS": yS[jj], "exit_close": exit_close.astype(np.int64),
                      "risk_pct": rp[jj].astype(np.float32)})
        if log:
            log(f"  {tf} {s}: {int(keep.sum())} سطر ({time.time() - t0:.1f}s)")
    ds = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    ds.update(tf=tf, symbols=list(symbols), features=list(cf.FEATURES[tf]), build_s=round(time.time() - t0, 1))
    return ds


# ───────────────────────── walk-forward ─────────────────────────
def train_mask(t, exit_close, yL, yS, tf, m0):
    """سطرهای آموزشِ مدلِ ماهِ شروع‌شده در ``m0``: ``t ≤ m0 − 81 کندل``، برچسبِ معلوم که خروجش تا ``m0``
    بسته شده (در شکافِ داده هم نشت نمی‌کند)، و در 5m فقط کندل‌های زوج."""
    bar = BAR_MS[tf]
    t = np.asarray(t, dtype=np.int64)
    m = (t <= int(m0) - GAP_BARS * bar) & (np.asarray(exit_close) <= int(m0))
    m &= np.isfinite(yL) & np.isfinite(yS)
    k = SUBSAMPLE.get(tf, 1)
    if k > 1:
        m &= (t // bar) % k == 0
    return m


def walk_forward(ds, months, fit_fn=None, threads=NUM_THREADS, log=None, window_end=None):
    """برای هر ماه: برازشِ تازه روی ``train_mask`` و پیش‌بینیِ فقط سطرهای همان ماه.

    ``window_end`` (انحصاری) سقفِ سخت است: هیچ سطرِ پیش‌بینی‌ای در/پس از آن ساخته نمی‌شود.
    خروجی: ``(oos, meta)``؛ ``oos`` = sym, t, E_L, E_S, month, yL, yS.
    """
    tf = ds["tf"]
    fit_fn = fit_fn or fit_pair
    t = ds["t"]
    out = {k: [] for k in ("sym", "t", "E_L", "E_S", "month", "yL", "yS")}
    meta = []
    if window_end is not None and any(m1 > window_end for _, m1, _ in months):
        raise ValueError("ماهِ خارج از پنجره — پیش‌بینیِ holdout در حالتِ اعتبارسنجی ممنوع است")
    for m0, m1, ym in months:
        tr = train_mask(t, ds["exit_close"], ds["yL"], ds["yS"], tf, m0)
        te = (t >= m0) & (t < m1)
        if not tr.any() or not te.any():
            meta.append({"month": ym, "n_train": int(tr.sum()), "n_test": int(te.sum()), "skipped": True})
            continue
        assert int(t[tr].max()) <= m0 - GAP_BARS * BAR_MS[tf] and int(ds["exit_close"][tr].max()) <= m0
        t0 = time.time()
        predict, fm = fit_fn(ds["X"][tr], ds["yL"][tr], ds["yS"][tr], t[tr], ds["sym"][tr], tf,
                             ds["features"], threads)
        EL, ES = predict(ds["X"][te])
        for k, v in (("sym", ds["sym"][te]), ("t", t[te]), ("E_L", EL), ("E_S", ES),
                     ("month", np.full(int(te.sum()), ym, dtype=np.int32)), ("yL", ds["yL"][te]),
                     ("yS", ds["yS"][te])):
            out[k].append(np.asarray(v))
        per_coin = np.bincount(ds["sym"][tr].astype(int), minlength=len(ds["symbols"])).tolist()
        row = {"month": ym, "train_cutoff": _date(m0 - GAP_BARS * BAR_MS[tf]), "n_train": int(tr.sum()),
               "n_train_per_coin": dict(zip(ds["symbols"], per_coin)), "n_test": int(te.sum()),
               "train_t_max": _date(int(t[tr].max())), "train_exit_close_max": _date(int(ds["exit_close"][tr].max())),
               "fit_s": round(time.time() - t0, 1)}
        row.update(fm or {})
        meta.append(row)
        if log:
            bl = (fm or {}).get("long", {}).get("best_iter")
            bs = (fm or {}).get("short", {}).get("best_iter")
            log(f"  {tf} {ym}: train {row['n_train']} test {row['n_test']} rounds L/S {bl}/{bs} ({row['fit_s']}s)")
    oos = {k: (np.concatenate(v) if v else np.zeros(0)) for k, v in out.items()}
    if window_end is not None and len(oos["t"]):
        assert int(oos["t"].max()) < window_end
    return oos, meta


# ───────────────────────── تصمیم و بازپخش ─────────────────────────
def decide(E_L, E_S, margin=MARGIN):
    """قاعدهٔ ثابتِ پیش‌ثبت: ۱ لانگ، −۱ شورت، ۰ صبر (NaN ⇒ صبر). ``margin`` هم کفِ E و هم فاصلهٔ دو طرف."""
    EL = np.asarray(E_L, dtype=np.float64)
    ES = np.asarray(E_S, dtype=np.float64)
    L = (EL >= margin) & (EL - ES >= margin)
    S = (ES >= margin) & (ES - EL >= margin)
    side = np.zeros(len(EL), dtype=np.int8)
    side[L & ~S] = 1
    side[S & ~L] = -1
    return side


def replay_sides(kl, side, first, cost_pct=COST, a14=None, max_bars=bracket.MAX_BARS):
    """همان حلقهٔ ``decision.replay`` برای هر سریِ جهتِ دلخواه: یک معامله در لحظه، سیگنالِ کندلِ i با
    ``bracket.signal_trade`` (ورود در بازِ i+1)، تا بسته‌شدنِ براکت سیگنالِ تازه نادیده، معاملهٔ نامعلوم
    در انتهای داده شمرده نمی‌شود."""
    o, h, l, c = (np.asarray(kl[k], dtype=float) for k in ("o", "h", "l", "c"))
    t = np.asarray(kl["t"], dtype=np.int64)
    n = len(c)
    a14 = engine.atr(h, l, c, 14) if a14 is None else a14
    side = np.asarray(side)
    first = max(int(first), decision.LIVE_BARS - 1)
    trades = []
    free_from = first
    for i in np.flatnonzero(side[first:]) + first:
        i = int(i)
        if i < free_from:
            continue
        if i + 1 >= n:
            break
        sg = int(side[i])
        res = bracket.signal_trade(o, h, l, c, float(a14[i]), i, sg, max_bars)
        if res is None or (res["timed_out"] and i + max_bars > n - 1):
            break
        net = bracket.net_r(res["gross_r"], res["risk_pct"], cost_pct)
        trades.append({"i": i, "t": int(t[i]), "side": "long" if sg == 1 else "short", "setup": None,
                       "entry": float(res["entry"]), "sl": float(res["sl"]), "tp": float(res["tp"]),
                       "risk_pct": float(res["risk_pct"]), "outcome": res["outcome"],
                       "gross_r": float(res["gross_r"]), "net_r": float(net),
                       "exit_idx": int(res["exit_idx"]), "exit_t": int(t[res["exit_idx"]]),
                       "bars_held": int(res["bars_held"])})
        free_from = int(res["exit_idx"])
    return trades


def trade_array(trades_by_sym, symbols, cost_pct=COST):
    """فهرستِ معامله‌ها ⇒ آرایه‌های هم‌طول (sym, t, net, side) با هزینهٔ ``cost_pct`` (همان معامله‌ها)."""
    sym, t, net, sd = [], [], [], []
    for k, s in enumerate(symbols):
        for x in trades_by_sym.get(s, []):
            sym.append(k)
            t.append(x["t"])
            net.append(bracket.net_r(x["gross_r"], x["risk_pct"], cost_pct))
            sd.append(1 if x["side"] == "long" else -1)
    return {"sym": np.asarray(sym, dtype=np.int16), "t": np.asarray(t, dtype=np.int64),
            "net": np.asarray(net, dtype=np.float64), "side": np.asarray(sd, dtype=np.int8)}


def trade_stats(ta, symbols):
    net = ta["net"]
    n = len(net)
    tot = float(net.sum()) if n else 0.0
    out = {"n": n, "mean": float(net.mean()) if n else None, "total": round(tot, 3),
           "win_rate": round(float((net > 0).mean() * 100), 1) if n else None,
           "se": float(net.std(ddof=1) / math.sqrt(n)) if n >= 2 else None,
           "long_n": int((ta["side"] == 1).sum()), "short_n": int((ta["side"] == -1).sum()),
           "long_mean": float(net[ta["side"] == 1].mean()) if (ta["side"] == 1).any() else None,
           "short_mean": float(net[ta["side"] == -1].mean()) if (ta["side"] == -1).any() else None,
           "per_coin": {}}
    for k, s in enumerate(symbols):
        m = ta["sym"] == k
        ns = int(m.sum())
        ts = float(net[m].sum()) if ns else 0.0
        out["per_coin"][s] = {"n": ns, "mean": float(net[m].mean()) if ns else None, "total": round(ts, 3),
                              "share_of_total": (ts / tot) if tot > 0 else None}
    out["coins_positive"] = sum(1 for v in out["per_coin"].values() if v["mean"] is not None and v["mean"] > 0)
    shares = [v["total"] for v in out["per_coin"].values()]
    out["max_coin_share"] = (max(shares) / tot) if (tot > 0 and shares) else None
    return out


# ───────────────────────── آمار: بوت‌استرپِ بلوکِ هفتگی و Holm ─────────────────────────
def week_id(t):
    return (np.asarray(t, dtype=np.int64) - MONDAY0_MS) // WEEK_MS


def block_bootstrap(t_a, r_a, t_b, r_b, w0, w1, B=BOOT_B, seed=BOOT_SEED):
    """بوت‌استرپِ بلوکیِ هفته‌های تقویمیِ ``[w0, w1)`` — هر پنج ارز با هم (هفته = بلوک).

    ``a`` = v2، ``b`` = قاعده. خروجی: میانگینِ a، بازهٔ ۹۵٪ a، تفاوت a−b و بازهٔ ۹۵٪اش، و p یک‌طرفهٔ
    «میانگینِ a ≤ 0» = ``(1 + #{a* ≤ 0}) / (B + 1)``.
    """
    weeks = np.arange(int(week_id([w0])[0]), int(week_id([w1 - 1])[0]) + 1)
    W = len(weeks)

    def per_week(t, r):
        t = np.asarray(t, dtype=np.int64)
        r = np.asarray(r, dtype=np.float64)
        k = week_id(t) - weeks[0]
        if len(k) and (k.min() < 0 or k.max() >= W):
            raise ValueError("معامله بیرون از هفته‌های پنجره")
        return np.bincount(k, weights=r, minlength=W), np.bincount(k, minlength=W).astype(np.float64)

    Sa, Na = per_week(t_a, r_a)
    Sb, Nb = per_week(t_b, r_b)
    rng = np.random.RandomState(seed)
    idx = rng.randint(0, W, size=(B, W))
    with np.errstate(invalid="ignore", divide="ignore"):
        ma = Sa[idx].sum(1) / Na[idx].sum(1)
        mb = Sb[idx].sum(1) / Nb[idx].sum(1)
    mb = np.where(np.isfinite(mb), mb, 0.0) if Nb.sum() == 0 else mb   # قاعده بی‌معامله ⇒ میانگینِ صفر
    d = ma - mb
    mean_a = float(Sa.sum() / Na.sum()) if Na.sum() else None
    mean_b = float(Sb.sum() / Nb.sum()) if Nb.sum() else 0.0
    ok_a = np.isfinite(ma)
    ok_d = np.isfinite(d)
    return {"weeks": int(W), "B": int(B), "seed": int(seed),
            "mean": mean_a, "ci95": [float(np.percentile(ma[ok_a], 2.5)), float(np.percentile(ma[ok_a], 97.5))]
            if ok_a.any() else None,
            "diff": (mean_a - mean_b) if mean_a is not None else None,
            "diff_ci95": [float(np.percentile(d[ok_d], 2.5)), float(np.percentile(d[ok_d], 97.5))]
            if ok_d.any() else None,
            # نمونه‌ای که هیچ معامله‌ای ندارد میانگینِ تعریف‌نشده دارد ⇒ به نفعِ فرضِ صفر (≤ 0) شمرده می‌شود
            "p_one_sided": float((1 + int(np.sum(~ok_a | (ma <= 0)))) / (B + 1)) if Na.sum() else 1.0}


def holm(pvals, alpha=HOLM_ALPHA):
    """Holm–Bonferroni: ``{نام: p}`` ⇒ ``{نام: (p_تعدیل‌شده, رد؟)}``؛ p نبود (None) = ۱."""
    items = sorted(((k, 1.0 if v is None else float(v)) for k, v in pvals.items()), key=lambda kv: (kv[1], kv[0]))
    m = len(items)
    out = {}
    run = 0.0
    for r, (k, p) in enumerate(items):
        run = max(run, min(1.0, (m - r) * p))
        out[k] = (run, run <= alpha)
    return out


def n_min(tf):
    return N_MIN.get(tf, N_MIN_DEFAULT)


def adopt_checks(tf, res, holm_reject):
    """چهار شرطِ پذیرشِ پیش‌ثبت روی پنجرهٔ اعتبارسنجی."""
    v2, rule, bt = res["v2"], res["rule"], res["bootstrap"]
    m2 = v2["mean"]
    mr = rule["mean"] if rule["mean"] is not None else 0.0
    c1 = (m2 is not None and m2 >= mr + ADOPT_EDGE_R and bt["diff_ci95"] is not None and bt["diff_ci95"][0] > 0)
    c2 = (m2 is not None and m2 > 0 and v2["n"] >= n_min(tf) and bool(holm_reject))
    c3 = v2["coins_positive"] >= MIN_POS_COINS
    c4 = v2["max_coin_share"] is not None and v2["max_coin_share"] <= MAX_COIN_SHARE
    return {"edge_vs_rule_and_diff_ci_above_0": bool(c1), "mean_pos_n_and_holm": bool(c2),
            "coins_positive_ge_3": bool(c3), "no_coin_over_50pct": bool(c4), "adopted": bool(c1 and c2 and c3 and c4)}


# ───────────────────────── ارزیابی ─────────────────────────
def _spearman(a, b):
    from scipy.stats import spearmanr
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 10:
        return None
    r = spearmanr(a[m], b[m]).statistic
    return float(r) if np.isfinite(r) else None


def ic_stats(oos, symbols):
    EL, ES = oos["E_L"].astype(float), oos["E_S"].astype(float)
    yL, yS = oos["yL"].astype(float), oos["yS"].astype(float)
    out = {"pooled": {"E_L~yL": _spearman(EL, yL), "E_S~yS": _spearman(ES, yS),
                      "(E_L-E_S)~(yL-yS)": _spearman(EL - ES, yL - yS),
                      "n": int((np.isfinite(yL) & np.isfinite(yS)).sum())},
           "per_coin": {}, "deciles": {}}
    for k, s in enumerate(symbols):
        m = oos["sym"] == k
        out["per_coin"][s] = {"E_L~yL": _spearman(EL[m], yL[m]), "E_S~yS": _spearman(ES[m], yS[m])}
    for name, E, y in (("long", EL, yL), ("short", ES, yS)):
        m = np.isfinite(E) & np.isfinite(y)
        if m.sum() < 100:
            continue
        q = np.quantile(E[m], np.linspace(0, 1, 11))
        b = np.clip(np.searchsorted(q, E[m], side="right") - 1, 0, 9)
        out["deciles"][name] = [{"E_mean": round(float(E[m][b == d].mean()), 4),
                                 "y_mean": round(float(y[m][b == d].mean()), 4), "n": int((b == d).sum())}
                                for d in range(10) if (b == d).any()]
    return out


def window_bars(F, w0, w1):
    """کندل‌های فیوچرزِ یک پنجره برای بازپخش: ``WARMUP_BARS`` کندلِ پیش از ``w0`` (همان گرم‌شدنِ scorecard_for) تا
    پیش از ``w1`` — **هیچ** کندلی در/پس از پایانِ پنجره (ممیزی DATA-5)؛ براکتی که تا آن‌جا بسته نشده در ``replay``
    و ``replay_sides`` شمرده نمی‌شود."""
    t = np.asarray(F["t"], dtype=np.int64)
    s0 = int(np.searchsorted(t, w0))
    e1 = int(np.searchsorted(t, w1))
    cut = max(0, s0 - decision.WARMUP_BARS)
    return {kk: np.asarray(v)[cut:e1] for kk, v in F.items() if kk in ("t", "o", "h", "l", "c", "v")}


def evaluate(tf, oos, w0, w1, fut, symbols=SYMBOLS, log=None):
    """معامله‌های v2 (همهٔ حاشیه‌ها) و قاعدهٔ فعلی روی همان کندل‌های فیوچرز و همان پنجره (``window_bars``)."""
    bar = BAR_MS[tf]
    v2 = {m: {} for m in (MARGIN,) + SENS_MARGINS}
    rule = {}
    replay_equal = True
    for k, s in enumerate(symbols):
        sub = window_bars(fut[s], w0, w1)
        st = sub["t"]
        rep = decision.replay(sub, start_ts=w0, cost_pct=COST)
        rule[s] = [x for x in rep["trades"] if x["t"] < w1]
        a14 = engine.atr(sub["h"], sub["l"], sub["c"], 14)
        # خودآزمایی روی دادهٔ واقعی: همان جهت‌های قاعده از حلقهٔ replay_sides ⇒ همان معامله‌ها
        rs = np.asarray(rep["side"]).copy()
        rs[st >= w1] = 0
        chk = replay_sides(sub, rs, rep["first"], COST, a14)
        keys = ("i", "t", "side", "entry", "sl", "tp", "net_r", "exit_idx")
        if [tuple(x[q] for q in keys) for x in chk] != [tuple(x[q] for q in keys) for x in rule[s]]:
            replay_equal = False
        m = oos["sym"] == k
        ot = oos["t"][m]
        j = np.searchsorted(st, ot)
        jc = np.minimum(j, len(st) - 1)
        hit = (j < len(st)) & (st[jc] == ot) & (ot >= w0) & (ot < w1)
        for mg in v2:
            side = np.zeros(len(st), dtype=np.int8)
            side[jc[hit]] = decide(oos["E_L"][m][hit], oos["E_S"][m][hit], mg)
            side[st >= w1] = 0
            v2[mg][s] = replay_sides(sub, side, rep["first"], COST, a14)
        if log:
            log(f"  {tf} {s}: v2 {len(v2[MARGIN][s])} معامله، قاعده {len(rule[s])}")
    return v2, rule, replay_equal


def run_window(tf, window="validation", threads=NUM_THREADS, log=None, fit_fn=None, allow_holdout=False):
    """یک تایم‌فریم روی یک پنجره: ساختِ داده، walk-forward، ارزیابی. هیچ کندلی (اسپات یا فیوچرز) در/پس از پایانِ
    پنجره بارگذاری نمی‌شود (ممیزی DATA-5): برچسبی که براکتش تا پایان بسته نشده NaN است و در IC نمی‌آید، و معاملهٔ
    بازمانده شمرده نمی‌شود — در ``validation`` و ``holdout`` یکسان.
    ``holdout`` فقط با ``allow_holdout=True`` و فقط اگر ``holdout_guard`` اجازه دهد."""
    if window not in ("validation", "holdout"):
        raise ValueError(window)
    if window == "holdout":
        report, _ = latest_validation_report()
        ok, why = holdout_guard(tf, report)
        if not allow_holdout or not ok:
            raise PermissionError(why or "holdout بدونِ پرچمِ صریح اجرا نمی‌شود")
    wall = time.time()
    w0, w1 = window_ms(window)
    fut_end = w1                                      # بی‌دُم: هیچ کندلِ فیوچرزی در/پس از پایانِ پنجره
    ds = build_dataset(tf, w1, fut_end, log=log)
    months = month_starts(w0, w1)
    t_fit = time.time()
    oos, meta = walk_forward(ds, months, fit_fn=fit_fn, threads=threads, log=log, window_end=w1)
    fit_s = time.time() - t_fit
    fut = {s: load_bars("um", s, tf, fut_end) for s in SYMBOLS}
    t_ev = time.time()
    v2, rule, replay_equal = evaluate(tf, oos, w0, w1, fut, log=log)
    syms = list(SYMBOLS)
    ta = trade_array(v2[MARGIN], syms)
    tr = trade_array(rule, syms)
    bt = block_bootstrap(ta["t"], ta["net"], tr["t"], tr["net"], w0, w1)
    sens = {}
    for mg, tb in v2.items():
        sens[f"margin_{mg:.2f}"] = {f"cost_{c:.2f}": _compact(trade_stats(trade_array(tb, syms, c), syms))
                                    for c in (COST, MAKER_COST)}
    names = ds["features"]
    imp = {}
    for side in ("long", "short"):
        g = np.zeros(len(names))
        for row in meta:
            if f"gain_{side}" in row:
                g += np.asarray(row[f"gain_{side}"])
        tot = g.sum()
        order = np.argsort(-g)[:20]
        imp[side] = [{"feature": names[i], "gain_share": round(float(g[i] / tot), 4) if tot > 0 else 0.0}
                     for i in order]
    for row in meta:                                  # اهمیتِ خام در فراداده نمی‌ماند (فقط جمعِ بالا)
        row.pop("gain_long", None)
        row.pop("gain_short", None)
    res = {"tf": tf, "window": window, "window_ms": [w0, w1], "window_dates": list(WINDOWS[window]),
           "run_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
           "core_feats_sha256": cf.source_hash(), "core2_sha256": file_sha256(os.path.abspath(__file__)),
           "features": names, "n_features": len(names), "dropped_features": cf.DROPPED.get(tf, {}),
           "params": {k: v for k, v in lgb_params(tf, meta[-1].get("n_train", 0) if meta else 0, threads).items()
                      if k != "min_data_in_leaf"},
           "train_warmup_bars": TRAIN_WARMUP[tf], "subsample": SUBSAMPLE.get(tf, 1),
           "dataset_rows": int(len(ds["t"])), "oos_rows": int(len(oos["t"])),
           "futures_end": "window end (exclusive): no futures bar at/after it; brackets still open there are not counted",
           "label_tail_bars": LABEL_TAIL_BARS,
           "oos_rows_label_unresolved": int((~(np.isfinite(oos["yL"]) & np.isfinite(oos["yS"]))).sum()),
           "v2": trade_stats(ta, syms), "rule": trade_stats(tr, syms),
           "rule_maker_0.10": _compact(trade_stats(trade_array(rule, syms, MAKER_COST), syms)),
           "bootstrap": bt, "sensitivity": sens, "importance_gain_top20": imp, "ic": ic_stats(oos, syms),
           "replay_equal_to_decision_replay": bool(replay_equal),
           "months": meta,
           "timings_s": {"dataset": ds["build_s"], "fit_total": round(fit_s, 1),
                         "evaluate": round(time.time() - t_ev, 1), "wall": round(time.time() - wall, 1)}}
    return res, oos


def _compact(st):
    return {k: st[k] for k in ("n", "mean", "total", "win_rate", "coins_positive", "max_coin_share")}


def save_oos(tf, oos, window="validation", out_dir=None):
    d = out_dir or OUT_DIR
    os.makedirs(d, exist_ok=True)
    path = os.path.join(d, f"{tf}_{'val' if window == 'validation' else 'hold'}_oos.npz")
    np.savez_compressed(path, sym=np.asarray(SYMBOLS)[oos["sym"].astype(int)] if len(oos["sym"]) else np.zeros(0, "U7"),
                        t=oos["t"].astype(np.int64), E_L=oos["E_L"].astype(np.float32),
                        E_S=oos["E_S"].astype(np.float32), month=oos["month"].astype(np.int32),
                        yL=oos["yL"].astype(np.float32), yS=oos["yS"].astype(np.float32))
    return path


def result_path(tf, window="validation", out_dir=None):
    return os.path.join(out_dir or OUT_DIR, f"{tf}_{'val' if window == 'validation' else 'hold'}_result.json")


def to_json(x):
    if isinstance(x, dict):
        return {str(k): to_json(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [to_json(v) for v in x]
    if isinstance(x, np.ndarray):
        return [to_json(v) for v in x.tolist()]
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating, float)):
        return None if not math.isfinite(float(x)) else round(float(x), 6)
    if isinstance(x, np.bool_):
        return bool(x)
    return x


def assemble(results, tfs=TFS):
    """گزارشِ کلِ اعتبارسنجی: Holm روی **پنج** تایم‌فریم (نبود ⇒ p=1 و گزارش ناقص) و قاعدهٔ پذیرش."""
    pv = {tf: (results[tf]["bootstrap"]["p_one_sided"] if tf in results else None) for tf in tfs}
    hm = holm(pv, HOLM_ALPHA)
    cells = {}
    for tf in tfs:
        if tf not in results:
            cells[tf] = {"missing": True, "adopted": False}
            continue
        r = dict(results[tf])
        r["holm_p"], r["holm_reject"] = hm[tf]
        r["n_min"] = n_min(tf)
        r["adopt"] = adopt_checks(tf, r, hm[tf][1])
        r["adopted"] = r["adopt"]["adopted"]
        cells[tf] = r
    cur = cf.source_hash()
    complete = all(tf in results for tf in tfs) and all(results[tf].get("core_feats_sha256") == cur for tf in tfs)
    return {"title": "core v2 — validation (walk-forward, pre-registered)", "window": "validation",
            "window_dates": list(WINDOWS["validation"]), "prereg": "bot/data/research/prereg_core2.json",
            "prereg_sha256": file_sha256(PREREG_PATH) if os.path.exists(PREREG_PATH) else None,
            "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
            "core_feats_sha256": cur, "complete": bool(complete),
            "holm_alpha": HOLM_ALPHA, "bootstrap": {"B": BOOT_B, "seed": BOOT_SEED, "block": "calendar week (UTC, Mon), all 5 coins jointly"},
            "decision_rule": {"margin": MARGIN, "sensitivity_margins": list(SENS_MARGINS)},
            "cost_pct": COST, "maker_cost_pct_descriptive": MAKER_COST,
            "holdout_touched": False,
            "futures_end": "window end (exclusive): labels and trades use no bar at/after it (audit DATA-5)",
            "note_fa": ("فقط پنجرهٔ اعتبارسنجی (۲۰۲۴-۰۷ تا ۲۰۲۵-۰۶). هیچ آماری روی holdout حساب نشده است: برچسب و "
                        "معامله هیچ کندلی در/پس از پایانِ پنجره نمی‌خوانند و براکتِ بازمانده شمرده نمی‌شود. "
                        "v2 هرگز tradeable را روشن نمی‌کند و به gates.json دست نمی‌زند."),
            "adopted": [tf for tf in tfs if cells[tf].get("adopted")],
            "timeframes": cells}


def summary_lines(report):
    out = []
    for tf, r in report["timeframes"].items():
        if r.get("missing"):
            out.append(f"{tf:>4}: (اجرا نشده)")
            continue
        v, ru, bt = r["v2"], r["rule"], r["bootstrap"]
        f = (lambda x: "—" if x is None else f"{x:+.3f}")
        ci = bt["ci95"] or [None, None]
        dci = bt["diff_ci95"] or [None, None]
        out.append(f"{tf:>4}: v2 n={v['n']} mean={f(v['mean'])} CI[{f(ci[0])},{f(ci[1])}] | "
                   f"rule n={ru['n']} mean={f(ru['mean'])} | diff CI[{f(dci[0])},{f(dci[1])}] | "
                   f"p={bt['p_one_sided']:.3f} holm={r['holm_p']:.3f} | coins+={v['coins_positive']} "
                   f"maxshare={f(v['max_coin_share'])} | ADOPT={'YES' if r['adopted'] else 'no'} | "
                   f"wall={r['timings_s']['wall']}s")
    return out


# ───────────────────────── نگهبانِ holdout ─────────────────────────
def report_files(prefix, research_dir):
    """فقط گزارش‌های زمان‌دارِ ``<prefix>YYYYMMDD_HHMMSS.json`` به ترتیبِ زمان — نه سنجاق یا اِراتا که همان پیشوند را دارند."""
    pat = re.compile(re.escape(prefix) + r"\d{8}_\d{6}\.json$")
    return sorted(p for p in glob.glob(os.path.join(research_dir, prefix + "*.json")) if pat.match(os.path.basename(p)))


def latest_validation_report(research_dir=None):
    files = report_files(REPORT_PREFIX, research_dir or RESEARCH_DIR)
    if not files:
        return None, None
    with open(files[-1], "r", encoding="utf-8") as f:
        return json.load(f), files[-1]


def holdout_guard(tf, report, out_dir=None):
    """``(مجاز؟, دلیل)`` — holdout فقط یک‌بار و فقط برای تایم‌فریمی که گزارشِ کاملِ اعتبارسنجی پذیرفته."""
    if tf not in TFS:
        return False, f"تایم‌فریمِ ناشناخته: {tf}"
    if not report:
        return False, "گزارشِ اعتبارسنجی پیدا نشد"
    if report.get("window") != "validation" or not report.get("complete"):
        return False, "گزارشِ اعتبارسنجی کامل نیست (هر پنج تایم‌فریم با همان core_feats لازم است)"
    cell = (report.get("timeframes") or {}).get(tf) or {}
    if cell.get("adopted") is not True:
        return False, f"{tf} در اعتبارسنجی پذیرفته نشده — holdout اجرا نمی‌شود"
    if os.path.exists(result_path(tf, "holdout", out_dir)):
        return False, f"holdoutِ {tf} قبلاً یک‌بار اجرا شده (یک‌بارمصرف)"
    return True, ""


def holdout_verdict(tf, res):
    v = res["v2"]
    nm = HOLD_N_MIN.get(tf, HOLD_N_MIN_DEFAULT)
    return {"n_min": nm, "passed": bool(v["mean"] is not None and v["mean"] > 0 and v["n"] >= nm)}
