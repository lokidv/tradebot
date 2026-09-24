# -*- coding: utf-8 -*-
"""هستهٔ v2.1 — همان ویژگی‌ها و برچسبِ v2 با مشخصهٔ مدلِ تازه (``research/prereg_core2_1.json``؛ الزام‌آور).

تفاوت با v2 فقط در مدل و قاعدهٔ تصمیم است:
  * LightGBM با هدفِ ``regression`` (l2، میانگین را پیش‌بینی می‌کند)، ۳۰۰ دورِ **ثابت** (1d: ۱۵۰)، بی‌توقفِ زودهنگام.
  * لانگ اگر ``E_L > 0`` و ``E_L > E_S``؛ شورت قرینه؛ وگرنه صبر (حاشیه‌های ۰٫۰۵ و ۰٫۱۰ فقط حساسیت).
  * آموزش فقط از سطرهای ``t ≥ 2022-01-01``؛ اعتبارسنجی ۲۰۲۵-۰۷..۲۰۲۵-۱۲؛ holdoutِ یک‌بارمصرف ۲۰۲۶-۰۱..۲۰۲۶-۰۸.

ساختِ داده، برشِ بی‌نشت (``core2.train_mask``)، وزنِ برابرِ ارزها، بازپخشِ معامله، بوت‌استرپِ هفتگی و Holm از
``core2`` می‌آیند. ``core_feats.py`` باید عیناً همان نسخهٔ commit b982f84 باشد (``CORE_FEATS_SHA256_LF``)؛ وگرنه
اجرا رد می‌شود.

پس از سنجاق (ممیزیِ ۲۰۲۶-۰۹) این فایل و ``core2.py`` برای مشخصه‌های بعدی اصلاح شدند: بی‌دُمِ فیوچرز پس از پایانِ
پنجره، IC روی R ناخالص/جهت‌دار/جزئی، فاندینگِ کوکوین تا بسته‌شدنِ سطرِ آخر، و خط‌های پایه/فاندینگِ توصیفی. پس هشِ
کد با سنجاق یکی نیست و نگهبانِ holdout رد می‌کند — عمداً: در اعتبارسنجیِ v2.1 هیچ تایم‌فریمی پذیرفته نشد و holdout
هرگز اجرا نمی‌شود.

holdout (سخت‌گیرانه‌تر از v2): فقط برای تایم‌فریمِ پذیرفته، فقط وقتی
  ۱. سنجاقِ ``core2_1_validation_pin.json`` در گیت commit شده و هشِ گزارشِ اعتبارسنجی با آن برابر است،
  ۲. هشِ کد و پیش‌ثبت با آنچه سنجاق شده برابر است،
  ۳. رکوردِ ``holdout_start`` همان تایم‌فریم در نسخهٔ commit‌شدهٔ ``core2_holdout_log.jsonl`` هست و فایلِ کاری فقط
     به آن افزوده شده (فقط-افزودنی)،
  ۴. هنوز ``holdout_done``ی برای آن ثبت نشده.
هر تلاش پیش از محاسبه (``holdout_attempt``) و هر نتیجه (``holdout_done``) به همان دفتر افزوده می‌شود.
هش‌ها پس از یکسان‌کردنِ پایانِ خط (CRLF → LF) حساب می‌شوند تا نسخهٔ ویندوز و گیت یک هش داشته باشند.

اختیار: v2.1 هرگز ``tradeable`` را روشن نمی‌کند، به gates.json نمی‌نویسد و autotrader آن را نمی‌خواند.
"""
import hashlib
import json
import math
import os
import subprocess
import time
from datetime import datetime, timezone

import numpy as np

import core2
import core_feats as cf
import decision
import engine
import paths
from watchlist import SYMBOLS

# ───────────────────────── ثابت‌های پیش‌ثبت (تغییر پس از دیدنِ نتیجه ممنوع) ─────────────────────────
PREREG_PATH = paths.data("research", "prereg_core2_1.json")
RESEARCH_DIR = paths.data("research")
OUT_DIR = paths.data("hist_research", "core2_1")
REPORT_PREFIX = "core2_1_validation_"
HOLDOUT_PREFIX = "core2_1_holdout_"
PIN_PATH = paths.data("research", "core2_1_validation_pin.json")
HOLDOUT_LOG = paths.data("research", "core2_holdout_log.jsonl")

# sha256 (پایانِ خطِ LF) از bot/core_feats.py در commit b982f84 — «ویژگی‌ها عیناً همان v2»
CORE_FEATS_SHA256_LF = "794f25df3d0db3d7161d0831465bf9c6d1263d5c9c1673847c1fc65a1bfd17b9"

TFS = cf.TFS
BAR_MS = cf.BAR_MS
DAY_MS = cf.DAY_MS

WINDOWS = {"design_only": ("2022-01-01", "2025-06-30"),
           "validation": ("2025-07-01", "2025-12-31"),
           "holdout": ("2026-01-01", "2026-08-31")}
TRAIN_START = "2022-01-01"        # «all bars from 2022-01»

GAP_BARS = core2.GAP_BARS         # 81 = ۴۱ پاک‌سازی + ۴۰ embargo
ROUNDS = {"1d": 150}              # بقیه ۳۰۰ — تعدادِ ثابت، بی‌توقفِ زودهنگام
ROUNDS_DEFAULT = 300

COST = decision.COST_PCT          # 0.14
MAKER_COST = decision.MAKER_COST_PCT   # 0.10 — فقط توصیفی
MARGIN = 0.0                      # قاعدهٔ اصلی: E > 0 و E > طرفِ دیگر
SENS_MARGINS = (0.05, 0.10)       # فقط حساسیت
LABEL_TAIL_BARS = core2.LABEL_TAIL_BARS   # 0 — ممیزی DATA-5: هیچ کندلی در/پس از پایانِ پنجره

HOLM_ALPHA = core2.HOLM_ALPHA     # 0.10
ADOPT_EDGE_R = 0.05
N_MIN = {"1d": 20}                # بقیه ۶۰
N_MIN_DEFAULT = 60
HOLD_N_MIN = {"1d": 10}           # بقیه ۳۰
HOLD_N_MIN_DEFAULT = 30
MIN_POS_COINS = core2.MIN_POS_COINS     # 3
MAX_COIN_SHARE = core2.MAX_COIN_SHARE   # 0.5
NUM_THREADS = core2.NUM_THREADS

_BOT = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_BOT)
# هر فایلی که پیش‌بینی، برچسب یا معاملهٔ v2.1 را تعیین می‌کند؛ هشِ همه در سنجاق ثبت و پیش از holdout بررسی می‌شود
CODE_FILES = ("bot/core2_1.py", "bot/core2.py", "bot/core_feats.py", "bot/bracket.py", "bot/decision.py",
              "bot/engine.py", "bot/tf_spec.py", "bot/watchlist.py", "tools/train_core2_1.py")


def _ms(day):
    return int(datetime.strptime(day, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp() * 1000)


TRAIN_START_MS = _ms(TRAIN_START)


def window_ms(name):
    """``(شروع, پایانِ انحصاری)`` به ms — پایان = روزِ بعد از آخرین روزِ پنجره."""
    a, b = WINDOWS[name]
    return _ms(a), _ms(b) + DAY_MS


def _utc():
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


# ───────────────────────── هش (مستقل از پایانِ خط) ─────────────────────────
def sha256_lf_bytes(b):
    return hashlib.sha256(b.replace(b"\r\n", b"\n")).hexdigest()


def sha256_lf(path):
    with open(path, "rb") as fh:
        return sha256_lf_bytes(fh.read())


def code_hashes(root=None):
    root = root or _ROOT
    return {rel: sha256_lf(os.path.join(root, *rel.split("/"))) for rel in CODE_FILES}


def check_features():
    """``core_feats.py`` باید عیناً نسخهٔ پیش‌ثبت‌شده (b982f84) باشد."""
    h = sha256_lf(os.path.join(_BOT, "core_feats.py"))
    if h != CORE_FEATS_SHA256_LF:
        raise ValueError(f"core_feats.py با نسخهٔ پیش‌ثبت (b982f84) یکی نیست: {h}")
    return h


def check_prereg(path=None):
    """پنجره‌ها و پارامترهای کلیدیِ این فایل باید عیناً همان پیش‌ثبت باشند (وگرنه خطا)."""
    with open(path or PREREG_PATH, "r", encoding="utf-8") as f:
        pr = json.load(f)
    for k, v in WINDOWS.items():
        key = "holdout_one_shot" if k == "holdout" else k
        if list(pr["windows"][key]) != list(v):
            raise ValueError(f"پنجرهٔ {k} با پیش‌ثبت یکی نیست")
    m = pr["model"]
    for s in ("objective regression (l2", "learning_rate 0.02", "num_leaves 15", "max_depth 5", "max(500, n/400)",
              "200 at 1h", "100 at 4h", "feature_fraction 0.5", "bagging_fraction 0.7", "bagging_freq 1",
              "lambda_l2 50", "max_bin 63", "seed 42", "FIXED 300 boosting rounds with no early stopping",
              "1d: num_leaves 7, min_data_in_leaf 60, 150 rounds", "every second row at 5m"):
        if s not in m:
            raise ValueError(f"پارامترِ «{s}» در پیش‌ثبت پیدا نشد")
    if "from 2022-01" not in pr["walk_forward"] or "month start - 81 bars" not in pr["walk_forward"]:
        raise ValueError("walk-forward با پیش‌ثبت یکی نیست")
    if "E_L > 0 and E_L > E_S" not in pr["decision_rule_fixed"] or "0.05 and 0.10" not in pr["decision_rule_fixed"]:
        raise ValueError("قاعدهٔ تصمیم با پیش‌ثبت یکی نیست")
    rules = " ".join(pr["adopt_rule_per_timeframe_on_validation"])
    if "n >= 60 trades (1d: n >= 20)" not in rules or "+ 0.05R" not in rules:
        raise ValueError("قاعدهٔ پذیرش با پیش‌ثبت یکی نیست")
    if "n >= 30 (1d: n >= 10)" not in pr["holdout"] or "core2_holdout_log.jsonl" not in pr["holdout"]:
        raise ValueError("قاعدهٔ holdout با پیش‌ثبت یکی نیست")
    return pr


# ───────────────────────── مدل ─────────────────────────
def n_rounds(tf):
    return ROUNDS.get(tf, ROUNDS_DEFAULT)


def lgb_params(tf, n_train, threads=NUM_THREADS):
    """پارامترهای عیناً پیش‌ثبت. ``n_train`` = تعدادِ سطرهای آموزشِ همان ماه (برای min_data_in_leaf)."""
    p = {"objective": "regression", "learning_rate": 0.02, "num_leaves": 15, "max_depth": 5,
         "feature_fraction": 0.5, "bagging_fraction": 0.7, "bagging_freq": 1, "lambda_l2": 50.0,
         "max_bin": 63, "deterministic": True, "force_col_wise": True, "seed": 42,
         "num_threads": int(threads), "verbosity": -1}
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


def fit_pair(X, yL, yS, t, sym, tf, names, threads=NUM_THREADS):
    """M_long و M_short برای یک ماه با تعدادِ دورِ ثابت ⇒ ``(predict(X) -> (E_L, E_S), meta)``."""
    import lightgbm as lgb
    params = lgb_params(tf, len(t), threads)
    rounds = n_rounds(tf)
    w = core2.coin_weights(sym)
    models, meta = {}, {"min_data_in_leaf": params["min_data_in_leaf"], "num_leaves": params["num_leaves"],
                        "rounds": rounds}
    for side, y in (("long", yL), ("short", yS)):
        t0 = time.time()
        d = lgb.Dataset(X, y, weight=w, feature_name=names, free_raw_data=True)
        models[side] = lgb.train(params, d, num_boost_round=rounds)
        pred_in = models[side].predict(X, num_threads=int(threads))
        meta[side] = {"trees": int(models[side].num_trees()), "fit_s": round(time.time() - t0, 2),
                      "train_label_mean_w": float(np.average(y, weights=w)),
                      "train_pred_mean_w": float(np.average(pred_in, weights=w)),
                      "train_pred_share_pos": float((pred_in > 0).mean())}
        meta[f"gain_{side}"] = models[side].feature_importance("gain").astype(float).tolist()

    def predict(Xte):
        return (models["long"].predict(Xte, num_threads=int(threads)).astype(np.float32),
                models["short"].predict(Xte, num_threads=int(threads)).astype(np.float32))

    return predict, meta


# ───────────────────────── walk-forward ─────────────────────────
def train_mask(t, exit_close, yL, yS, tf, m0):
    """برشِ v2 (``t ≤ m0 − 81 کندل``، خروجِ برچسب تا ``m0`` بسته، برچسبِ معلوم، 5m هر سطرِ دوم) + ``t ≥ 2022-01-01``."""
    m = core2.train_mask(t, exit_close, yL, yS, tf, m0)
    return m & (np.asarray(t, dtype=np.int64) >= TRAIN_START_MS)


def walk_forward(ds, months, fit_fn=None, threads=NUM_THREADS, log=None, window_end=None):
    """برای هر ماه: برازشِ تازه روی ``train_mask`` و پیش‌بینیِ فقط سطرهای همان ماه (مثلِ ``core2.walk_forward``
    با برشِ پایینِ ۲۰۲۲-۰۱). ``window_end`` (انحصاری) سقفِ سخت است. خروجی ``(oos, meta)``."""
    tf = ds["tf"]
    fit_fn = fit_fn or fit_pair
    t = ds["t"]
    out = {k: [] for k in core2.oos_keys(ds)}
    meta = []
    if window_end is not None and any(m1 > window_end for _, m1, _ in months):
        raise ValueError("ماهِ خارج از پنجره — پیش‌بینیِ پس از پایانِ پنجره ممنوع است")
    for m0, m1, ym in months:
        tr = train_mask(t, ds["exit_close"], ds["yL"], ds["yS"], tf, m0)
        te = (t >= m0) & (t < m1)
        if not tr.any() or not te.any():
            meta.append({"month": ym, "n_train": int(tr.sum()), "n_test": int(te.sum()), "skipped": True})
            continue
        assert int(t[tr].max()) <= m0 - GAP_BARS * BAR_MS[tf] and int(ds["exit_close"][tr].max()) <= m0
        assert int(t[tr].min()) >= TRAIN_START_MS
        t0 = time.time()
        predict, fm = fit_fn(ds["X"][tr], ds["yL"][tr], ds["yS"][tr], t[tr], ds["sym"][tr], tf,
                             ds["features"], threads)
        EL, ES = predict(ds["X"][te])
        for k, v in (("sym", ds["sym"][te]), ("t", t[te]), ("E_L", EL), ("E_S", ES),
                     ("month", np.full(int(te.sum()), ym, dtype=np.int32)), ("yL", ds["yL"][te]),
                     ("yS", ds["yS"][te])) + tuple((k, ds[k][te]) for k in core2.OOS_EXTRA if k in ds):
            out[k].append(np.asarray(v))
        per_coin = np.bincount(ds["sym"][tr].astype(int), minlength=len(ds["symbols"])).tolist()
        row = {"month": ym, "train_cutoff": core2._date(m0 - GAP_BARS * BAR_MS[tf]), "n_train": int(tr.sum()),
               "n_train_per_coin": dict(zip(ds["symbols"], per_coin)), "n_test": int(te.sum()),
               "train_t_min": core2._date(int(t[tr].min())), "train_t_max": core2._date(int(t[tr].max())),
               "train_exit_close_max": core2._date(int(ds["exit_close"][tr].max())),
               "fit_s": round(time.time() - t0, 1)}
        row.update(fm or {})
        meta.append(row)
        if log:
            log(f"  {tf} {ym}: train {row['n_train']} test {row['n_test']} rounds {(fm or {}).get('rounds')} "
                f"({row['fit_s']}s)")
    oos = {k: (np.concatenate(v) if v else np.zeros(0)) for k, v in out.items()}
    if window_end is not None and len(oos["t"]):
        assert int(oos["t"].max()) < window_end
    return oos, meta


# ───────────────────────── تصمیم و ارزیابی ─────────────────────────
def decide(E_L, E_S, margin=MARGIN):
    """قاعدهٔ ثابتِ پیش‌ثبت: ۱ لانگ اگر ``E_L > m`` و ``E_L > E_S + m``؛ ۱− شورت قرینه؛ وگرنه ۰ (برابری و NaN ⇒ صبر).
    قاعدهٔ اصلی ``m = 0``؛ ۰٫۰۵ و ۰٫۱۰ فقط حساسیت."""
    EL = np.asarray(E_L, dtype=np.float64)
    ES = np.asarray(E_S, dtype=np.float64)
    L = (EL > margin) & (EL > ES + margin)
    S = (ES > margin) & (ES > EL + margin)
    side = np.zeros(len(EL), dtype=np.int8)
    side[L & ~S] = 1
    side[S & ~L] = -1
    return side


def evaluate(tf, oos, w0, w1, fut, symbols=SYMBOLS, log=None):
    """معامله‌های v2.1 (قاعدهٔ اصلی + حاشیه‌های حساسیت) و قاعدهٔ فعلی روی همان کندل‌های فیوچرز و همان پنجره.
    کندل‌ها از ``core2.window_bars``: گرم‌شدنِ scorecard_for تا **پیش از** ``w1``؛ براکتِ بازمانده شمرده نمی‌شود."""
    v21 = {m: {} for m in (MARGIN,) + SENS_MARGINS}
    rule = {}
    replay_equal = True
    for k, s in enumerate(symbols):
        sub = core2.window_bars(fut[s], w0, w1)
        st = sub["t"]
        rep = decision.replay(sub, start_ts=w0, cost_pct=COST)
        rule[s] = [x for x in rep["trades"] if x["t"] < w1]
        a14 = engine.atr(sub["h"], sub["l"], sub["c"], 14)
        rs = np.asarray(rep["side"]).copy()           # خودآزمایی: جهت‌های قاعده از همان حلقه ⇒ همان معامله‌ها
        rs[st >= w1] = 0
        chk = core2.replay_sides(sub, rs, rep["first"], COST, a14)
        keys = ("i", "t", "side", "entry", "sl", "tp", "net_r", "exit_idx")
        if [tuple(x[q] for q in keys) for x in chk] != [tuple(x[q] for q in keys) for x in rule[s]]:
            replay_equal = False
        m = oos["sym"] == k
        ot = oos["t"][m]
        j = np.searchsorted(st, ot)
        jc = np.minimum(j, len(st) - 1)
        hit = (j < len(st)) & (st[jc] == ot) & (ot >= w0) & (ot < w1)
        for mg in v21:
            side = np.zeros(len(st), dtype=np.int8)
            side[jc[hit]] = decide(oos["E_L"][m][hit], oos["E_S"][m][hit], mg)
            side[st >= w1] = 0
            v21[mg][s] = core2.replay_sides(sub, side, rep["first"], COST, a14)
        if log:
            log(f"  {tf} {s}: v2.1 {len(v21[MARGIN][s])} معامله، قاعده {len(rule[s])}")
    return v21, rule, replay_equal


def pred_stats(oos, symbols):
    """توزیعِ پیش‌بینی‌های بیرون از نمونه — سهمِ E > 0 همان چیزی است که v2 را بی‌معامله کرد."""
    out = {}
    for name in ("E_L", "E_S"):
        E = np.asarray(oos[name], dtype=np.float64)
        E = E[np.isfinite(E)]
        out[name] = ({"n": int(len(E)), "mean": float(E.mean()), "p05": float(np.percentile(E, 5)),
                      "p50": float(np.percentile(E, 50)), "p95": float(np.percentile(E, 95)),
                      "share_pos": float((E > 0).mean())} if len(E) else {"n": 0})
    side = decide(oos["E_L"], oos["E_S"]) if len(oos["t"]) else np.zeros(0, np.int8)
    out["bars_long"] = int((side == 1).sum())
    out["bars_short"] = int((side == -1).sum())
    out["bars_wait"] = int((side == 0).sum())
    return out


def _coverage(tf, w1, spot_end, symbols=SYMBOLS):
    """دادهٔ اسپات باید تا پایانِ پنجره برسد (حداکثر یک روز کسری)؛ وگرنه اجرا بی‌صدا معامله‌های کمتری می‌دید."""
    short = []
    for s in symbols:
        b = core2.load_bars("spot", s, tf, spot_end)
        if not len(b["t"]) or int(b["t"][-1]) < w1 - DAY_MS:
            short.append(f"{s} تا {core2._date(int(b['t'][-1])) if len(b['t']) else '—'}")
    if short:
        raise RuntimeError(f"{tf}: دادهٔ اسپات تا پایانِ پنجره نمی‌رسد ({'، '.join(short)}) — "
                           "python tools/fetch_core_history.py --market spot")


def run_window(tf, window="validation", threads=NUM_THREADS, log=None, fit_fn=None, allow_holdout=False):
    """یک تایم‌فریم روی یک پنجره: ساختِ داده، walk-forward، ارزیابی. هیچ کندلی (اسپات یا فیوچرز) در/پس از پایانِ
    پنجره بارگذاری نمی‌شود (ممیزی DATA-5: پیش‌تر ۴۵ کندلِ فیوچرزِ دُم، برچسب و معاملهٔ پایانِ اعتبارسنجی را به
    کندل‌های holdout وابسته می‌کرد): برچسبِ بازمانده NaN و بیرون از IC، معاملهٔ بازمانده شمرده نمی‌شود — در
    ``validation`` و ``holdout`` یکسان. خط‌های پایه و فاندینگِ کوکوین کنارِ نتیجه فقط توصیفی‌اند.
    ``holdout`` فقط با ``allow_holdout=True`` و فقط اگر ``holdout_guard`` اجازه دهد — **پیش از** خواندنِ هر داده."""
    if window not in ("validation", "holdout"):
        raise ValueError(window)
    if window == "holdout":
        ok, why = holdout_guard(tf)
        if not allow_holdout or not ok:
            raise PermissionError(why or "holdout بدونِ پرچمِ صریح اجرا نمی‌شود")
    feats_lf = check_features()
    wall = time.time()
    w0, w1 = window_ms(window)
    fut_end = w1                                      # بی‌دُم: هیچ کندلِ فیوچرزی در/پس از پایانِ پنجره
    if fit_fn is None:
        _coverage(tf, w1, w1)
    ds = core2.build_dataset(tf, w1, fut_end, log=log)
    months = core2.month_starts(w0, w1)
    t_fit = time.time()
    oos, meta = walk_forward(ds, months, fit_fn=fit_fn, threads=threads, log=log, window_end=w1)
    fit_s = time.time() - t_fit
    fut = {s: core2.load_bars("um", s, tf, fut_end) for s in SYMBOLS}
    t_ev = time.time()
    v21, rule, replay_equal = evaluate(tf, oos, w0, w1, fut, log=log if window == "validation" else None)
    syms = list(SYMBOLS)
    ta = core2.trade_array(v21[MARGIN], syms)
    tr = core2.trade_array(rule, syms)
    bt = core2.block_bootstrap(ta["t"], ta["net"], tr["t"], tr["net"], w0, w1)
    base, fund = core2.descriptive_lines(tf, w0, w1, fut, {"v21": v21[MARGIN], "rule": rule},
                                         core2.load_kfund_all(w1 + 1), syms)
    sens = {}
    for mg, tb in v21.items():
        sens[f"margin_{mg:.2f}"] = {f"cost_{c:.2f}": core2._compact(core2.trade_stats(core2.trade_array(tb, syms, c), syms))
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
           "run_utc": _utc(), "core_feats_sha256": cf.source_hash(), "core_feats_sha256_lf": feats_lf,
           "code_sha256_lf": code_hashes(),
           "features": names, "n_features": len(names), "dropped_features": cf.DROPPED.get(tf, {}),
           "params": {k: v for k, v in lgb_params(tf, meta[-1].get("n_train", 0) if meta else 0, threads).items()
                      if k not in ("min_data_in_leaf", "num_threads")},
           "rounds": n_rounds(tf), "train_start": TRAIN_START,
           "train_warmup_bars": core2.TRAIN_WARMUP[tf], "subsample": core2.SUBSAMPLE.get(tf, 1),
           "dataset_rows": int(len(ds["t"])), "oos_rows": int(len(oos["t"])),
           "futures_end": "window end (exclusive): no futures bar at/after it; brackets still open there are not counted",
           "label_tail_bars": LABEL_TAIL_BARS,
           "oos_rows_label_unresolved": int((~(np.isfinite(oos["yL"]) & np.isfinite(oos["yS"]))).sum()),
           "v21": core2.trade_stats(ta, syms), "rule": core2.trade_stats(tr, syms),
           "rule_maker_0.10": core2._compact(core2.trade_stats(core2.trade_array(rule, syms, MAKER_COST), syms)),
           "baselines": base, "funding_kucoin": fund,
           "bootstrap": bt, "sensitivity": sens, "predictions": pred_stats(oos, syms),
           "importance_gain_top20": imp, "ic": core2.ic_stats(oos, syms),
           "replay_equal_to_decision_replay": bool(replay_equal),
           "months": meta,
           "timings_s": {"dataset": ds["build_s"], "fit_total": round(fit_s, 1),
                         "evaluate": round(time.time() - t_ev, 1), "wall": round(time.time() - wall, 1)}}
    return res, oos


def save_oos(tf, oos, window="validation", out_dir=None):
    return core2.save_oos(tf, oos, window, out_dir or OUT_DIR)


def result_path(tf, window="validation", out_dir=None):
    return core2.result_path(tf, window, out_dir or OUT_DIR)


# ───────────────────────── پذیرش و گزارش ─────────────────────────
def n_min(tf):
    return N_MIN.get(tf, N_MIN_DEFAULT)


def adopt_checks(tf, res, holm_reject):
    """چهار شرطِ پذیرشِ پیش‌ثبت روی پنجرهٔ اعتبارسنجی."""
    v, rule, bt = res["v21"], res["rule"], res["bootstrap"]
    m = v["mean"]
    mr = rule["mean"] if rule["mean"] is not None else 0.0
    c1 = (m is not None and m >= mr + ADOPT_EDGE_R and bt["diff_ci95"] is not None and bt["diff_ci95"][0] > 0)
    c2 = (m is not None and m > 0 and v["n"] >= n_min(tf) and bool(holm_reject))
    c3 = v["coins_positive"] >= MIN_POS_COINS
    c4 = v["max_coin_share"] is not None and v["max_coin_share"] <= MAX_COIN_SHARE
    return {"edge_vs_rule_and_diff_ci_above_0": bool(c1), "mean_pos_n_and_holm": bool(c2),
            "coins_positive_ge_3": bool(c3), "no_coin_over_50pct": bool(c4), "adopted": bool(c1 and c2 and c3 and c4)}


def assemble(results, tfs=TFS):
    """گزارشِ کلِ اعتبارسنجی: Holm روی **پنج** تایم‌فریم (نبود ⇒ p=1 و گزارش ناقص) و قاعدهٔ پذیرش."""
    pv = {tf: (results[tf]["bootstrap"]["p_one_sided"] if tf in results else None) for tf in tfs}
    hm = core2.holm(pv, HOLM_ALPHA)
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
    cur = code_hashes()
    complete = all(tf in results for tf in tfs) and all(results[tf].get("code_sha256_lf") == cur for tf in tfs)
    return {"title": "core v2.1 — validation (walk-forward, pre-registered)", "window": "validation",
            "window_dates": list(WINDOWS["validation"]), "prereg": "bot/data/research/prereg_core2_1.json",
            "prereg_sha256_lf": sha256_lf(PREREG_PATH) if os.path.exists(PREREG_PATH) else None,
            "created_utc": _utc(), "code_sha256_lf": cur, "core_feats_sha256_lf": cur["bot/core_feats.py"],
            "complete": bool(complete),
            "holm_alpha": HOLM_ALPHA,
            "bootstrap": {"B": core2.BOOT_B, "seed": core2.BOOT_SEED,
                          "block": "calendar week (UTC, Mon), all 5 coins jointly"},
            "decision_rule": {"main": "long if E_L > 0 and E_L > E_S; short if E_S > 0 and E_S > E_L; else wait",
                              "sensitivity_margins": list(SENS_MARGINS)},
            "cost_pct": COST, "maker_cost_pct_descriptive": MAKER_COST,
            "holdout_touched": False,
            "futures_end": "window end (exclusive): labels and trades use no bar at/after it (audit DATA-5)",
            "descriptive_only": ["ic", "baselines", "funding_kucoin", "sensitivity", "rule_maker_0.10", "predictions"],
            "note_fa": ("فقط پنجرهٔ اعتبارسنجی (۲۰۲۵-۰۷ تا ۲۰۲۵-۱۲). هیچ آماری روی holdout (۲۰۲۶-۰۱ تا ۲۰۲۶-۰۸) حساب "
                        "نشده است: برچسب و معامله هیچ کندلی در/پس از پایانِ پنجره نمی‌خوانند و براکتِ بازمانده شمرده "
                        "نمی‌شود. IC، خط‌های پایه و فاندینگ فقط توصیفی‌اند. "
                        "v2.1 هرگز tradeable را روشن نمی‌کند و به gates.json دست نمی‌زند."),
            "adopted": [tf for tf in tfs if cells[tf].get("adopted")],
            "timeframes": cells}


def summary_lines(report):
    out = []
    f = (lambda x: "—" if x is None else f"{x:+.3f}")
    for tf, r in report["timeframes"].items():
        if r.get("missing"):
            out.append(f"{tf:>4}: (اجرا نشده)")
            continue
        v, ru, bt = r["v21"], r["rule"], r["bootstrap"]
        ci = bt["ci95"] or [None, None]
        dci = bt["diff_ci95"] or [None, None]
        out.append(f"{tf:>4}: v2.1 n={v['n']} mean={f(v['mean'])} CI[{f(ci[0])},{f(ci[1])}] | "
                   f"rule n={ru['n']} mean={f(ru['mean'])} | diff CI[{f(dci[0])},{f(dci[1])}] | "
                   f"p={bt['p_one_sided']:.3f} holm={r['holm_p']:.3f} | coins+={v['coins_positive']} "
                   f"maxshare={f(v['max_coin_share'])} | ADOPT={'YES' if r['adopted'] else 'no'} | "
                   f"wall={r['timings_s']['wall']}s")
        d = core2.descriptive_summary(r, "v21")
        if d:
            out.append(d)
    return out


def holdout_verdict(tf, res):
    v = res["v21"]
    nm = HOLD_N_MIN.get(tf, HOLD_N_MIN_DEFAULT)
    return {"n_min": nm, "passed": bool(v["mean"] is not None and v["mean"] > 0 and v["n"] >= nm),
            "label_if_passed": "passed validation and holdout; live test pending"}


def latest_validation_report(research_dir=None):
    """آخرین گزارشِ زمان‌دار — سنجاق (``..._pin.json``) و اِراتا (``..._erratum.json``) هم همین پیشوند را دارند و
    پیش‌تر در مرتب‌سازی پس از گزارش می‌آمدند."""
    files = core2.report_files(REPORT_PREFIX, research_dir or RESEARCH_DIR)
    if not files:
        return None, None
    with open(files[-1], "r", encoding="utf-8") as f:
        return json.load(f), files[-1]


# ───────────────────────── گیت: «commit شده» یعنی در HEAD ─────────────────────────
def git_head_bytes(path):
    """محتوای فایل در HEADِ مخزنی که فایل در آن است؛ ``None`` اگر گیت نیست یا فایل در HEAD نیست."""
    d, name = os.path.split(os.path.abspath(path))
    try:
        r = subprocess.run(["git", "-C", d, "show", f"HEAD:./{name}"], capture_output=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return None
    return r.stdout if r.returncode == 0 else None


def committed_unchanged(path):
    """``(درست؟, دلیل)`` — فایل در HEAD هست و فایلِ کاری (با پایانِ خطِ یکسان) همان است."""
    if not os.path.isfile(path):
        return False, f"{os.path.basename(path)} وجود ندارد"
    head = git_head_bytes(path)
    if head is None:
        return False, f"{os.path.basename(path)} در گیت commit نشده (یا گیت در دسترس نیست)"
    with open(path, "rb") as fh:
        work = fh.read()
    if sha256_lf_bytes(head) != sha256_lf_bytes(work):
        return False, f"{os.path.basename(path)} پس از commit تغییر کرده"
    return True, ""


# ───────────────────────── سنجاق و دفترِ holdout ─────────────────────────
def make_pin(report_path, root=None):
    """سنجاقِ گزارشِ اعتبارسنجی: هشِ گزارش، کد و پیش‌ثبت. فقط برای گزارشِ کامل؛ فقط یک‌بار."""
    with open(report_path, "r", encoding="utf-8") as f:
        rep = json.load(f)
    if rep.get("window") != "validation" or not rep.get("complete"):
        raise ValueError("گزارشِ اعتبارسنجی کامل نیست (هر پنج تایم‌فریم با همان کد لازم است)")
    cur = code_hashes(root)
    if rep.get("code_sha256_lf") != cur:
        bad = sorted(k for k in cur if (rep.get("code_sha256_lf") or {}).get(k) != cur[k])
        raise ValueError(f"کد پس از ساختِ گزارش تغییر کرده: {bad}")
    return {"what": "core v2.1 validation pin — holdout refuses unless every hash below matches",
            "created_utc": _utc(), "report": os.path.basename(report_path),
            "report_sha256_lf": sha256_lf(report_path), "prereg_sha256_lf": sha256_lf(PREREG_PATH),
            "code_sha256_lf": cur, "adopted": list(rep.get("adopted") or [])}


def write_pin(report_path, pin_path=None, root=None):
    pin_path = pin_path or PIN_PATH
    if os.path.exists(pin_path):
        raise FileExistsError("سنجاق قبلاً ساخته شده — «the last spec tried on these windows»؛ دوباره ساخته نمی‌شود")
    pin = make_pin(report_path, root)
    with open(pin_path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(pin, f, ensure_ascii=False, indent=1)
        f.write("\n")
    return pin


def read_log(path=None):
    path = path or HOLDOUT_LOG
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(x) for x in f.read().splitlines() if x.strip()]


def append_log(rec, path=None):
    """فقط-افزودنی: یک خطِ JSON به انتها (هرگز بازنویسی)."""
    path = path or HOLDOUT_LOG
    rec = dict(rec, utc=_utc())
    with open(path, "a", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(rec, ensure_ascii=False, sort_keys=True) + "\n")
    return rec


def _log_committed_prefix(path):
    """``(رکوردهای HEAD, دلیلِ رد)`` — دفتر در HEAD هست و فایلِ کاری فقط به آن افزوده شده."""
    if not os.path.exists(path):
        return None, "دفترِ holdout (core2_holdout_log.jsonl) وجود ندارد"
    head = git_head_bytes(path)
    if head is None:
        return None, "دفترِ holdout در گیت commit نشده (یا گیت در دسترس نیست)"
    with open(path, "rb") as fh:
        work = fh.read().replace(b"\r\n", b"\n")
    head = head.replace(b"\r\n", b"\n")
    if not work.startswith(head):
        return None, "دفترِ holdout فقط-افزودنی نیست: فایلِ کاری با نسخهٔ commit‌شده شروع نمی‌شود"
    return [json.loads(x) for x in head.decode("utf-8").splitlines() if x.strip()], ""


def holdout_guard(tf, pin_path=None, log_path=None, research_dir=None, out_dir=None, root=None):
    """``(مجاز؟, دلیل)`` — همهٔ شرط‌های پیش‌ثبت پیش از هر محاسبهٔ holdout."""
    pin_path = pin_path or PIN_PATH
    log_path = log_path or HOLDOUT_LOG
    research_dir = research_dir or RESEARCH_DIR
    if tf not in TFS:
        return False, f"تایم‌فریمِ ناشناخته: {tf}"
    ok, why = committed_unchanged(pin_path)
    if not ok:
        return False, "سنجاق: " + why
    with open(pin_path, "r", encoding="utf-8") as f:
        pin = json.load(f)
    if not pin.get("report"):
        return False, "سنجاق نامِ گزارش را ندارد"
    rpath = os.path.join(research_dir, os.path.basename(pin["report"]))
    ok, why = committed_unchanged(rpath)
    if not ok:
        return False, "گزارشِ اعتبارسنجی: " + why
    if sha256_lf(rpath) != pin.get("report_sha256_lf"):
        return False, "هشِ گزارشِ اعتبارسنجی با سنجاق یکی نیست"
    if sha256_lf(PREREG_PATH) != pin.get("prereg_sha256_lf"):
        return False, "پیش‌ثبت پس از سنجاق تغییر کرده"
    cur = code_hashes(root)
    bad = sorted(k for k in cur if (pin.get("code_sha256_lf") or {}).get(k) != cur[k])
    if bad:
        return False, f"کد پس از سنجاق تغییر کرده: {bad}"
    with open(rpath, "r", encoding="utf-8") as f:
        rep = json.load(f)
    if rep.get("window") != "validation" or not rep.get("complete"):
        return False, "گزارشِ سنجاق‌شده کامل نیست"
    if ((rep.get("timeframes") or {}).get(tf) or {}).get("adopted") is not True:
        return False, f"{tf} در اعتبارسنجی پذیرفته نشده — holdout اجرا نمی‌شود"
    head_recs, why = _log_committed_prefix(log_path)
    if head_recs is None:
        return False, why
    pin_sha = sha256_lf(pin_path)
    if not any(r.get("event") == "holdout_start" and r.get("tf") == tf and r.get("pin_sha256_lf") == pin_sha
               for r in head_recs):
        return False, f"رکوردِ holdout_start برای {tf} با همین سنجاق در دفترِ commit‌شده نیست"
    if any(r.get("event") == "holdout_done" and r.get("tf") == tf for r in read_log(log_path)):
        return False, f"holdoutِ {tf} قبلاً یک‌بار اجرا شده (یک‌بارمصرف)"
    if os.path.exists(result_path(tf, "holdout", out_dir)):
        return False, f"holdoutِ {tf} قبلاً یک‌بار اجرا شده (فایلِ نتیجه هست)"
    return True, ""


def start_record(tf, pin_path=None, log_path=None, research_dir=None, root=None):
    """رکوردِ ``holdout_start`` (پس از آن باید commit شود). همان شرط‌های نگهبان جز خودِ رکورد."""
    pin_path = pin_path or PIN_PATH
    log_path = log_path or HOLDOUT_LOG
    if tf not in TFS:
        raise ValueError(f"تایم‌فریمِ ناشناخته: {tf}")
    ok, why = committed_unchanged(pin_path)
    if not ok:
        raise PermissionError("سنجاق: " + why)
    with open(pin_path, "r", encoding="utf-8") as f:
        pin = json.load(f)
    if not pin.get("report"):
        raise PermissionError("سنجاق نامِ گزارش را ندارد")
    if tf not in (pin.get("adopted") or []):
        raise PermissionError(f"{tf} در اعتبارسنجی پذیرفته نشده — holdout اجرا نمی‌شود")
    if any(r.get("tf") == tf for r in read_log(log_path)):
        raise PermissionError(f"برای {tf} قبلاً رکوردی در دفترِ holdout هست")
    rpath = os.path.join(research_dir or RESEARCH_DIR, os.path.basename(pin["report"]))
    if sha256_lf(rpath) != pin.get("report_sha256_lf"):
        raise PermissionError("هشِ گزارشِ اعتبارسنجی با سنجاق یکی نیست")
    cur = code_hashes(root)
    if cur != pin.get("code_sha256_lf"):
        raise PermissionError("کد پس از سنجاق تغییر کرده")
    return append_log({"event": "holdout_start", "tf": tf, "window": list(WINDOWS["holdout"]),
                       "pin_sha256_lf": sha256_lf(pin_path), "report": pin["report"],
                       "report_sha256_lf": pin["report_sha256_lf"], "prereg_sha256_lf": pin["prereg_sha256_lf"]},
                      log_path)
