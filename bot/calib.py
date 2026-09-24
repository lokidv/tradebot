# -*- coding: utf-8 -*-
"""کالیبراسیون نسل ۳:
طبقه‌بند ensemble احتمال برد و مدل مستقل Ridge/LightGBM برای بازده خالص روی ۲۳ ویژگی.
انتخاب، کالیبراسیون و آزمون نهایی پنجره‌های زمانی جدا و purged دارند. خروجی زنده
همراه بازهٔ عدم‌قطعیت، کران پایین EV، اعتبار رژیم و تشخیص خارج‌ازدامنه است."""
import json
import math
import os
import threading
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

import numpy as np

import bracket
import costs
import log
import engine
import features
import market
import paths
import research
import stats
import universe
import watchlist

try:
    import lightgbm as lgb
    HAS_LGBM = True
except Exception:  # noqa: BLE001
    HAS_LGBM = False

CALIB_PATH = paths.data("calib.json")
COST_PCT = 0.15
# هفتگی، نه روزانه: بازسازیِ روزانه همان پنجرهٔ آزمون را ۳۶۵ بار در سال دوباره قضاوت می‌کرد
REBUILD_SEC = 7 * 86400
TRUST_STREAK = 2         # اعتماد فقط پس از دو ساختِ متوالیِ موفق روشن می‌شود
CALIB_VERSION = 23       # با هر تغییرِ ویژگی‌ها/براکت/فرمتِ مدل یک واحد اضافه شود تا مدل قدیمی خودکار بازساخته شود
# ۲۳: یک تعریف برای آموزش و اجرا — وضعیتِ بازار از سبدِ ثابتِ پنج‌ارزی (پهنا و رتبهٔ نسبی خنثی)،
#     فاندینگِ آخرین تسویهٔ پیش از بسته‌شدنِ کندل با تاریخچهٔ کامل، شبکهٔ نمونهٔ متراکمِ سراسری،
#     براکتِ کاملِ ۴۰ کندلی، هزینهٔ زندهٔ هم‌واحد با آموزش (R)، embargoِ ۴۱ کندلیِ سیاستِ عمل
# ۲۲: پنجرهٔ آزمونِ منجمد از آموزش کنار گذاشته می‌شود؛ اعتماد با هیسترزیسِ دو ساخت
# ۲۱: جهانِ نقطه‌-در-زمان — رویداد فقط اگر ارز در همان ماه جزوِ ۱۰۰ برتر بوده
# ۲۰: هزینهٔ هر رویداد = ردهٔ نقدشوندگیِ لحظه‌ای + فاندینگِ واقعیِ مدتِ نگه‌داری
# ۱۸: براکتِ واحد (bracket.py) — برچسب‌ها حالا گپِ پشتِ حدضرر را روی قیمتِ مشاهده‌شده می‌بندند
# ۱۹: حذفِ ویژگیِ مردهٔ dxy_dir (۲۴ → ۲۳ ویژگی) + یکسان‌سازیِ فرمولِ فاندینگِ آموزش/اجرا
WF_FOLDS = 5             # تعداد فولدهای Walk-Forward
CALIB_UNIVERSE_N = 100   # اندازهٔ جهانِ نقطه‌-در-زمان در هر ماه
# سبدِ مرجعِ «وضعیتِ کلِ بازار» (دامیننس BTC، ETH/BTC) — **همان** پنج ارزی که برنامه زنده
# می‌خواند (watchlist، به خواستِ کاربر برای سرعت). قبلاً آموزش ۱۰۰ ارز را می‌دید و اجرا فقط
# کشِ نیمه‌تازهٔ همین پنج تا را؛ دامیننس زنده در ~۹۷٪ کندل‌ها روی +۱ می‌ماند (calib-F1).
MARKET_BASKET = tuple(watchlist.SYMBOLS)
DENSE_STRIDE = 4         # نمونهٔ متراکم: هر ۴ کندل — روی شبکهٔ زمانیِ سراسری، نه اندیسِ هر ارز
WF_EMBARGO = 24          # fallback فقط برای ورودی‌های بدون timestamp
MIN_BRIER_SKILL = 0.01   # حداقل ۱٪ بهبود نسبت به پیش‌بینیِ ثابتِ نرخ پایه
MIN_SETUP_LIFT = 4.0
MIN_DIR_LIFT = 5.0
MIN_EDGE_RANK_IC = 0.03
MIN_EDGE_LIFT_R = 0.12
# دادگاهِ سیاست نهایی: این آستانه‌ها پیش از دیدن test تعریف شده‌اند. بر خلاف
# نسخهٔ قبلی، معتبرنبودنِ یک یادگیرندهٔ کمکی به‌تنهایی کل سیاست را خاموش نمی‌کند.
POLICY_MIN_TEST = 35
POLICY_MIN_AVG_R = 0.02
POLICY_MIN_PF = 1.05
POLICY_MIN_UPLIFT_R = 0.08
POLICY_MIN_LCB_R = -0.10
ACTION_MAX_SAMPLES = 80_000
ACTION_ROLLING_WINDOWS = {
    "1d": (180, 365, 730),
    "4h": (270, 540, 1080),
    "1h": (720, 1440, 2880),
    "15m": (1440, 2880, 5760),
}
LGBM_ROUNDS = 260
LGBM_PARAMS = dict(objective="binary", num_leaves=16, max_depth=4, learning_rate=0.03,
                   min_child_samples=60, subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                   reg_lambda=1.5, reg_alpha=0.5, verbosity=-1, n_jobs=2, seed=42)
# جست‌وجوی کوچکِ تنظیماتِ LightGBM: سه پیکربندی با پیچیدگیِ کم/متوسط/زیاد — انتخاب فقط روی بخشِ آموزشیِ OOF
LGBM_GRID = [dict(),
             dict(num_leaves=31, max_depth=5, learning_rate=0.05, min_child_samples=40),
             dict(num_leaves=8, max_depth=3, learning_rate=0.02, min_child_samples=120)]
EDGE_LGB_PARAMS = dict(
    objective="huber", metric="l1", num_leaves=12, max_depth=4,
    learning_rate=0.025, min_child_samples=100,
    subsample=0.8, subsample_freq=1, colsample_bytree=0.75,
    reg_lambda=3.0, reg_alpha=1.0, verbosity=-1, n_jobs=2, seed=73,
    num_iterations=220,
)
_lgb_cache = {}          # hash(model_str) -> Booster (کش لود در زمان اجرا)


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


def _logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


# ── مدل‌های پایه: LightGBM (اگر باشد) یا لجستیکِ numpy ──
def _logit_fit(X, y):
    mu, sd = X.mean(axis=0), X.std(axis=0)
    sd = np.where(sd < 1e-9, 1.0, sd)
    Xn = np.hstack([np.ones((len(y), 1)), (X - mu) / sd])
    w = np.zeros(Xn.shape[1])
    for _ in range(600):
        p = _sigmoid(Xn @ w)
        w -= 0.3 * (Xn.T @ (p - y) / len(y) + 0.01 * w)
    return {"w": w, "mu": mu, "sd": sd}


def _logit_pred(m, X):
    Xn = (X - m["mu"]) / m["sd"]
    return _sigmoid(m["w"][0] + Xn @ m["w"][1:])


def _lgb_fit(X, y, extra=None):
    ds = lgb.Dataset(X, label=y, free_raw_data=False)
    return lgb.train({**LGBM_PARAMS, **(extra or {}), "num_iterations": LGBM_ROUNDS}, ds)


def _lgb_pred(m, X):
    return m.predict(X)


# ── شبکهٔ عصبیِ سبک (MLP یک‌لایه، numpy خالص — بدون وابستگیِ سنگین) ──
def _mlp_fit_one(X, y, hidden=18, iters=400, lr=0.08, l2=3e-3, seed=0):
    mu, sd = X.mean(axis=0), X.std(axis=0)
    sd = np.where(sd < 1e-9, 1.0, sd)
    Xn = (X - mu) / sd
    rng = np.random.RandomState(seed)
    n, d = Xn.shape
    W1 = rng.randn(d, hidden) * math.sqrt(2.0 / d)
    b1 = np.zeros(hidden)
    W2 = rng.randn(hidden) * math.sqrt(1.0 / hidden)
    b2 = 0.0
    for _ in range(iters):
        z1 = Xn @ W1 + b1
        h = np.tanh(z1)
        p = _sigmoid(h @ W2 + b2)
        g = (p - y) / n
        gW2 = h.T @ g + l2 * W2
        gb2 = float(np.sum(g))
        gh = np.outer(g, W2) * (1 - h * h)
        gW1 = Xn.T @ gh + l2 * W1
        gb1 = gh.sum(axis=0)
        W1 -= lr * gW1; b1 -= lr * gb1; W2 -= lr * gW2; b2 -= lr * gb2
    return {"W1": W1, "b1": b1, "W2": W2, "b2": b2, "mu": mu, "sd": sd}


def _mlp_pred_one(m, X):
    Xn = (X - m["mu"]) / m["sd"]
    h = np.tanh(Xn @ m["W1"] + m["b1"])
    return _sigmoid(h @ m["W2"] + m["b2"])


MLP_SEEDS = (0, 7, 42, 101, 2024)


def _mlp_fit(X, y, seeds=MLP_SEEDS, hidden=18, iters=400, lr=0.08, l2=3e-3):
    """آموزشِ تنسوریِ هم‌زمانِ همهٔ seedها در یک پاس (batched) — همان ریاضیِ تک‌به‌تک،
    ولی همهٔ شبکه‌ها با هم در ضرب‌های ماتریسیِ بزرگ‌تر آموزش می‌بینند (چند برابر سریع‌تر روی چندهسته).
    ۵ seed به‌جای ۳ → واریانسِ آموزش باز هم پایین‌تر، تقریباً بدونِ هزینهٔ اضافه."""
    mu, sd = X.mean(axis=0), X.std(axis=0)
    sd = np.where(sd < 1e-9, 1.0, sd)
    Xn = (X - mu) / sd
    n, d = Xn.shape
    S = len(seeds)
    W1l, W2l = [], []
    for s in seeds:                       # همان مقداردهیِ اولیهٔ نسخهٔ تک‌به‌تک (توالیِ RNG یکسان)
        rng = np.random.RandomState(s)
        W1l.append(rng.randn(d, hidden) * math.sqrt(2.0 / d))
        W2l.append(rng.randn(hidden) * math.sqrt(1.0 / hidden))
    W1 = np.stack(W1l, axis=2)            # (d, h, S)
    b1 = np.zeros((hidden, S))
    W2 = np.stack(W2l, axis=1)            # (h, S)
    b2 = np.zeros(S)
    yc = y[:, None]
    for _ in range(iters):
        z1 = np.tensordot(Xn, W1, axes=(1, 0)) + b1          # (n, h, S) — BLAS
        h_ = np.tanh(z1)
        p = _sigmoid((h_ * W2[None]).sum(axis=1) + b2)       # (n, S)
        g = (p - yc) / n                                     # (n, S)
        gW2 = (h_ * g[:, None, :]).sum(axis=0) + l2 * W2     # (h, S)
        gb2 = g.sum(axis=0)
        gh = g[:, None, :] * W2[None] * (1 - h_ * h_)        # (n, h, S)
        gW1 = np.tensordot(Xn.T, gh, axes=(1, 0)) + l2 * W1  # (d, h, S) — BLAS
        gb1 = gh.sum(axis=0)
        W1 -= lr * gW1; b1 -= lr * gb1; W2 -= lr * gW2; b2 -= lr * gb2
    return {"nets": [{"W1": W1[:, :, k], "b1": b1[:, k], "W2": W2[:, k], "b2": float(b2[k]),
                      "mu": mu, "sd": sd} for k in range(S)]}


def _mlp_pred(m, X):
    if "nets" in m:
        return np.mean([_mlp_pred_one(n, X) for n in m["nets"]], axis=0)
    return _mlp_pred_one(m, X)                       # سازگاری با فرمتِ قدیمی


# ── سریال‌سازی/بارگذاریِ هر یادگیرنده برای ذخیره در JSON ──
def _ser(kind, m):
    if kind == "lgbm":
        return {"lgbm": m.model_to_string()}
    if kind == "mlp":
        return {"nets": [{k: np.asarray(net[k]).tolist() for k in ("W1", "b1", "W2", "b2", "mu", "sd")}
                         for net in m["nets"]]}
    return {"w": m["w"].tolist(), "mu": m["mu"].tolist(), "sd": m["sd"].tolist()}


def _load_net(blob):
    m = {k: np.array(blob[k], float) for k in ("W1", "b1", "W2", "mu", "sd")}
    m["b2"] = float(np.asarray(blob["b2"]))
    return m


def _load_member(kind, blob):
    if kind == "lgbm":
        return blob                                  # رشتهٔ مدل؛ در امتیازدهی از _get_booster
    if kind == "mlp":
        if "nets" in blob:
            return {"nets": [_load_net(b) for b in blob["nets"]]}
        return _load_net(blob)                        # فرمتِ تک‌شبکه‌ایِ قدیمی
    return {"w": np.array(blob["w"], float), "mu": np.array(blob["mu"], float), "sd": np.array(blob["sd"], float)}


def _member_pred(kind, model, X):
    if kind == "lgbm":
        bk = _get_booster(model["lgbm"] if isinstance(model, dict) else model)
        return bk.predict(X) if bk is not None else np.full(len(X), 0.5)
    if kind == "mlp":
        return _mlp_pred(model, X)
    return _logit_pred(model, X)


_BASE = {"logit": (_logit_fit, _logit_pred),
         "lgbm": (_lgb_fit, _lgb_pred),
         "mlp": (_mlp_fit, _mlp_pred)}


def _platt(p, y):
    """کالیبراسیونِ پلَت: sigmoid(a·logit(p)+b) تا خروجیِ مدل واقعاً «احتمال» باشد."""
    z = _logit(p)
    a, b = 1.0, 0.0
    for _ in range(400):
        q = _sigmoid(a * z + b)
        a -= 0.2 * float(np.mean((q - y) * z))
        b -= 0.2 * float(np.mean(q - y))
    return float(a), float(b)


def _get_booster(model_str):
    h = hash(model_str)
    bk = _lgb_cache.get(h)
    if bk is None and HAS_LGBM:
        bk = lgb.Booster(model_str=model_str)
        _lgb_cache[h] = bk
    return bk


# ── «نودهای یادگیری»: استخرِ پردازه‌ها — چندهسته‌ایِ واقعی بدونِ GIL و بدونِ ازدحامِ BLAS ──
_pool = None
_pool_guard = threading.Lock()


def _can_spawn():
    try:
        import __main__
        main_file = str(getattr(__main__, "__file__", "") or "")
        return bool(main_file) and not main_file.startswith("<")
    except Exception:  # noqa: BLE001
        return False


def _get_pool():
    global _pool
    with _pool_guard:
        if _pool is None:
            for v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
                os.environ[v] = "2"                  # هر نود حداکثر ۲ رشتهٔ BLAS — نودها با هم رقابت نمی‌کنند
            _pool = ProcessPoolExecutor(max_workers=max(2, min(16, (os.cpu_count() or 8) - 4)))
    return _pool


def _close_pool():
    global _pool
    with _pool_guard:
        if _pool is not None:
            _pool.shutdown(wait=False)
            _pool = None


def _wf_task(kind, extra, X_tr, y_tr, X_te):
    """بدنهٔ یک فولدِ Walk-Forward — روی یک نودِ جداگانه اجرا می‌شود."""
    if kind == "lgbm":
        return _lgb_pred(_lgb_fit(X_tr, y_tr, extra), X_te)
    if kind == "mlp_seed":                           # یک seed از شبکهٔ عصبی (ریزترین واحدِ کار)
        return _mlp_pred_one(_mlp_fit_one(X_tr, y_tr, seed=extra), X_te)
    fit, pred = _BASE[kind]
    return pred(fit(X_tr, y_tr), X_te)


def _full_fit_task(kind, extra, X, y):
    """فیتِ نهایی روی همهٔ داده — خروجی blob سریال‌شدهٔ JSON (قابل انتقال بین پردازه‌ها)."""
    if kind == "mlp_seed":
        net = _mlp_fit_one(X, y, seed=extra)
        return {k: np.asarray(net[k]).tolist() for k in ("W1", "b1", "W2", "b2", "mu", "sd")}
    m = _lgb_fit(X, y, extra) if kind == "lgbm" else _BASE[kind][0](X, y)
    return _ser(kind, m)


def _submit(kind, extra, X_tr, y_tr, X_te):
    try:
        if not _can_spawn():
            return None                              # notebook/stdin/تعامل زنده: spawn ویندوز امن نیست
        return _get_pool().submit(_wf_task, kind, extra, X_tr, y_tr, X_te)
    except Exception:  # noqa: BLE001 — استخر در دسترس نیست
        return None


def _walk_forward_splits(ts, embargo_ms=0):
    """فولدهای زمانیِ group-aware؛ هیچ timestamp مشترکی دو طرف مرز نیست.

    purge به اندازهٔ افق برچسب اعمال می‌شود تا outcomeهای train وارد بازهٔ test
    نشوند. برای دادهٔ فاقد timestamp، رفتار قدیمیِ embargo ردیفی حفظ می‌شود.
    """
    ts = np.asarray(ts)
    if len(ts) == 0:
        return []
    uniq = np.unique(ts)
    start = max(int(len(uniq) * 0.45), 1)
    bounds = np.linspace(start, len(uniq), WF_FOLDS + 1).astype(int)
    out = []
    for i in range(WF_FOLDS):
        lo, hi = int(bounds[i]), int(bounds[i + 1])
        if lo >= hi:
            continue
        test_start = uniq[lo]
        test_end = uniq[hi] if hi < len(uniq) else None
        train_cutoff = test_start - embargo_ms
        tr_idx = np.where(ts < train_cutoff)[0]
        te_mask = ts >= test_start
        if test_end is not None:
            te_mask &= ts < test_end
        te_idx = np.where(te_mask)[0]
        if len(tr_idx) and len(te_idx):
            out.append((tr_idx, te_idx))
    return out


def _purge_boundary(rows_a, rows_b, ts, purge_ms):
    """ردیف‌هایی از پنجرهٔ قبلی که پنجرهٔ برچسبشان وارد پنجرهٔ بعدی می‌شود را حذف می‌کند.

    embargo فقط روی مرزِ فولدهای Walk-Forward اعمال می‌شد؛ مرزهای dev/cal/test
    با اندیسِ ردیف بریده می‌شدند، پس برچسبِ آخرین رویدادهای dev تا داخلِ cal و
    test ادامه داشت — نشتی‌ای که «آزمونِ دست‌نخورده» را آلوده می‌کرد.
    """
    if ts is None or not len(rows_a) or not len(rows_b) or purge_ms <= 0:
        return rows_a
    ts = np.asarray(ts, dtype=np.float64)
    start_b = float(ts[rows_b].min())
    keep = ts[rows_a] < start_b - float(purge_ms)
    return rows_a[keep]


def _walk_forward(X, y, kind, extra=None, ts=None, embargo_ms=0):
    """پیش‌بینی‌های برون‌نمونه‌ایِ Walk-Forward با embargo (بدون نشتیِ زمانی).
    هر فولد (و برای شبکهٔ عصبی، هر seed از هر فولد) یک تسکِ مستقل روی نودهای یادگیری است."""
    n = len(y)
    if ts is None:
        ts = np.arange(n)
        embargo_ms = WF_EMBARGO
    splits = _walk_forward_splits(ts, embargo_ms)
    oos_p = np.full(n, np.nan)
    futs = []
    for tr_idx, te_idx in splits:
        if len(np.unique(y[tr_idx])) < 2:
            continue
        if kind == "mlp":                             # هر seed یک نود — میانگین در والد (هم‌ارزِ دقیقِ قبلی)
            fs = [(s, _submit("mlp_seed", s, X[tr_idx], y[tr_idx], X[te_idx])) for s in MLP_SEEDS]
            futs.append((tr_idx, te_idx, fs))
        else:
            futs.append((tr_idx, te_idx, _submit(kind, extra, X[tr_idx], y[tr_idx], X[te_idx])))
    for tr_idx, te_idx, fut in futs:
        try:
            if isinstance(fut, list):                 # mlp: میانگینِ پیش‌بینیِ seedها
                preds = [(f.result() if f is not None else
                          _wf_task("mlp_seed", s, X[tr_idx], y[tr_idx], X[te_idx])) for s, f in fut]
                oos_p[te_idx] = np.mean(preds, axis=0)
            else:
                oos_p[te_idx] = fut.result() if fut is not None else \
                    _wf_task(kind, extra, X[tr_idx], y[tr_idx], X[te_idx])
        except Exception:  # noqa: BLE001 — process pool ناموجود/خراب: محاسبهٔ محلی
            try:
                if isinstance(fut, list):
                    oos_p[te_idx] = np.mean([
                        _wf_task("mlp_seed", s, X[tr_idx], y[tr_idx], X[te_idx])
                        for s, _f in fut], axis=0)
                else:
                    oos_p[te_idx] = _wf_task(kind, extra, X[tr_idx], y[tr_idx], X[te_idx])
            except Exception:  # noqa: BLE001 — فقط همان فولد واقعاً خراب است
                continue
    return oos_p


def _thirds(pred):
    """اندیس‌های یک‌سومِ پایین و بالای ``pred``؛ تساوی با ترتیبِ شبه‌تصادفیِ ثابت شکسته می‌شود.

    ``argsort`` روی مقدارهای برابر ترتیبِ اندیس — یعنی **زمان** — را نگه می‌داشت؛ با پیش‌بینیِ
    ثابت (شیبِ کالیبراسیونِ صفر) «لیفت» همان روندِ زمانیِ پنجرهٔ آزمون می‌شد (4h: ۰٫۵۵R با
    rank_ic صفر — calib-F11). پیش‌بینیِ (تقریباً) ثابت هیچ رتبه‌ای ندارد ⇒ None.
    """
    pred = np.asarray(pred, float)
    if len(pred) < 2 or float(np.std(pred)) < 1e-12:
        return None
    tie_break = np.random.RandomState(len(pred)).random_sample(len(pred))
    order = np.lexsort((tie_break, pred))          # کلیدِ اصلی pred؛ بی‌تساوی همان argsort است
    third = max(len(order) // 3, 1)
    return order[:third], order[-third:]


def _lift(oos_p, y):
    mask = ~np.isnan(oos_p)
    if mask.sum() < 80:
        return -99.0, mask
    po, yo = oos_p[mask], y[mask]
    th = _thirds(po)
    if th is None:
        return 0.0, mask
    return float(yo[th[1]].mean() - yo[th[0]].mean()) * 100, mask


def _lift_on(p, yv):
    th = _thirds(p)
    if th is None:
        return 0.0
    yv = np.asarray(yv, float)
    return float(yv[th[1]].mean() - yv[th[0]].mean()) * 100


def _fit_model(X, y, champion=None, ts=None, champion_ts=0.0, embargo_ms=0):
    """مغزِ ترکیب‌کننده (Stacking Ensemble): چند یادگیرندهٔ متنوع (لجستیک/LightGBM/شبکهٔ عصبی) با
    Walk-Forward آموزش می‌بینند؛ یک متا-مدل ترکیبشان را یاد می‌گیرد؛ برندهٔ برون‌نمونه‌ای انتخاب و پلَت-کالیبره می‌شود.
    خروجی: (متادیتای مدل، احتمال‌های کالیبره روی همه، پیش‌بینی OOSِ کالیبره)."""
    n = len(y)
    if n < 150 or len(np.unique(y)) < 2:
        return None, None, None
    kinds = ["logit"]
    if HAS_LGBM and n >= 500:
        kinds.append("lgbm")
    if n >= 500:
        kinds.append("mlp")

    # پیش‌بینیِ OOF هر یادگیرنده — همهٔ فولدهای همهٔ خانواده‌ها (و پیکربندی‌های LightGBM) روی نودها
    jobs = []                                            # (نام, نوع, تنظیماتِ اضافه)
    for k in kinds:
        if k == "lgbm":
            jobs += [(f"lgbm#{gi}", "lgbm", extra) for gi, extra in enumerate(LGBM_GRID)]
        else:
            jobs.append((k, k, None))
    with ThreadPoolExecutor(max_workers=len(jobs)) as tpool:   # هر رشته فقط منتظرِ نتیجهٔ نودهاست
        oof_all = dict(zip([j[0] for j in jobs],
                           tpool.map(lambda j: _walk_forward(X, y, j[1], j[2], ts, embargo_ms), jobs)))
    oof = {k: oof_all[k] for k in kinds if k != "lgbm"}
    lgb_extra = None
    if "lgbm" in kinds:
        # انتخابِ پیکربندی فقط روی بخشِ اولِ OOF (بخشِ ارزیابیِ ترکیب دست‌نخورده می‌ماند)
        cand = {gi: oof_all[f"lgbm#{gi}"] for gi in range(len(LGBM_GRID))}
        valid = ~np.isnan(cand[0])
        sel = np.where(valid)[0][:max(int(valid.sum() * 0.6), 1)]
        best_gi = max(cand, key=lambda gi: _lift_on(cand[gi][sel], y[sel]) if len(sel) >= 60 else 0.0)
        oof["lgbm"] = cand[best_gi]
        lgb_extra = LGBM_GRID[best_gi]
    common = np.ones(n, bool)
    for k in kinds:
        common &= ~np.isnan(oof[k])
    idx = np.where(common)[0]
    if len(idx) < 300:
        return None, None, None
    # سه پنجرهٔ زمانی مجزا: توسعه/انتخاب، کالیبراسیون و آزمون نهایی.
    # آزمون نهایی نه مدل را انتخاب می‌کند و نه Platt را برازش می‌دهد.
    dev_end = int(len(idx) * 0.60)
    cal_end = int(len(idx) * 0.80)
    st, cal_rows, test_rows = idx[:dev_end], idx[dev_end:cal_end], idx[cal_end:]
    ts_arr = np.asarray(ts, dtype=np.float64) if ts is not None else None
    st = _purge_boundary(st, cal_rows, ts_arr, embargo_ms)
    cal_rows = _purge_boundary(cal_rows, test_rows, ts_arr, embargo_ms)
    if min(len(st), len(cal_rows), len(test_rows)) < 60:
        return None, None, None

    single_lift = {k: _lift_on(oof[k][st], y[st]) for k in kinds}
    best_single = max(single_lift, key=single_lift.get)

    # ترکیب فقط روی پنجرهٔ calibration انتخاب می‌شود؛ test کاملاً دست‌نخورده است.
    best_kind = best_single
    stacker_select = None
    if len(kinds) >= 2:
        stk_tmp = _logit_fit(np.column_stack([_logit(oof[k][st]) for k in kinds]), y[st])
        ens_cal = _logit_pred(stk_tmp, np.column_stack([_logit(oof[k][cal_rows]) for k in kinds]))
        if _lift_on(ens_cal, y[cal_rows]) - _lift_on(
                oof[best_single][cal_rows], y[cal_rows]) >= 1.0:
            best_kind = "ensemble"
            stacker_select = stk_tmp

    if best_kind == "ensemble":
        p_cal = _logit_pred(
            stacker_select, np.column_stack([_logit(oof[k][cal_rows]) for k in kinds]))
        p_oos = _logit_pred(
            stacker_select, np.column_stack([_logit(oof[k][test_rows]) for k in kinds]))
    else:
        p_cal = oof[best_kind][cal_rows]
        p_oos = oof[best_kind][test_rows]
    y_cal, yo, oos_rows = y[cal_rows], y[test_rows], test_rows
    lift = _lift_on(p_oos, yo)
    ev, yv = test_rows, yo  # سازگاری شاخهٔ champion با پنجرهٔ آزمون نهایی

    # ── Champion/Challenger (P1): داوریِ منصفانه فقط روی داده‌ای که قهرمان هرگز در آموزش ندیده ──
    # (مقایسه روی کلِ پنجرهٔ ارزیابی جانب‌دارانه است: قهرمان بیشترِ آن ردیف‌ها را in-sample دیده و همیشه می‌بَرد)
    if champion is not None and champion.get("n_feat") == X.shape[1]:
        try:
            champ_ts = float(champion.get("trained_ts") or champion_ts or 0)
            champ_p = np.clip(_score_all(champion, X[ev]), 1e-6, 1 - 1e-6)
            chall_p = p_oos if best_kind == "ensemble" else oof[best_kind][ev]
            fresh = None
            if ts is not None and champ_ts > 0:
                fresh = np.where(np.asarray(ts, float)[ev] > champ_ts * 1000.0)[0]
            if fresh is not None and len(fresh) >= 60:
                # دادگاهِ منصفانه: هر دو مدل روی رویدادهای پس از آموزشِ قهرمان (برای هر دو برون‌نمونه‌ای)؛
                # قهرمان می‌ماند مگر چالشگر «معناداری» بهتر باشد (ثبات > تازگیِ بی‌دلیل)
                hold = _lift_on(chall_p[fresh], yv[fresh]) <= _lift_on(champ_p[fresh], yv[fresh]) + 1.0
            else:
                # هنوز دادهٔ تازهٔ کافی برای داوری نیست — قهرمان می‌ماند مگر بیش از ۱۴ روز کهنه باشد
                hold = champ_ts > 0 and (time.time() - champ_ts) / 86400 <= 14
            if hold:
                meta = dict(champion)
                meta["held"] = True
                meta.setdefault("trained_ts", champ_ts if champ_ts > 0 else time.time())
                if fresh is not None and len(fresh) >= 60:
                    # کالیبراسیون و آمار فقط از پنجرهٔ منصفانه (داده‌ای که قهرمان هرگز ندیده).
                    # ⚠️ برازشِ Platt روی دادهٔ دیده‌شده شیب را باد می‌کند (a>1) و مدل «بیش‌مطمئن» می‌شود —
                    # همان باگی که احتمال‌های ۷۰٪+ کاذب ساخت و پرتفوی را پر از ورودِ به‌ظاهر عالی کرد.
                    a, b = _platt(champ_p[fresh], yv[fresh])
                    meta["platt"] = [a, b]
                    po_c = _sigmoid(a * _logit(champ_p) + b)
                    meta.update(n_oos=int(len(fresh)),
                                oos_lift=round(_lift_on(champ_p[fresh], yv[fresh]), 1),
                                oos_brier=round(float(((po_c[fresh] - yv[fresh]) ** 2).mean()), 4),
                                oos_base=round(float(yv[fresh].mean()) * 100, 1))
                else:
                    # پنجرهٔ منصفانه نداریم → Platt و آمارِ صادقانهٔ قبلیِ قهرمان دست‌نخورده می‌مانند
                    a, b = (champion.get("platt") or [1.0, 0.0])
                    po_c = _sigmoid(a * _logit(champ_p) + b)
                p_all = _sigmoid(a * _logit(np.clip(_score_all(meta, X), 1e-6, 1 - 1e-6)) + b)
                oos_pc = np.full(n, np.nan)
                oos_pc[ev] = po_c
                return meta, p_all, oos_pc
        except Exception:  # noqa: BLE001 — قهرمانِ ناسازگار/خراب: چالشگر جایگزین می‌شود
            log.exc()

    a, b = _platt(p_cal, y_cal)
    po_c = _sigmoid(a * _logit(p_oos) + b)
    base_rate = float(yo.mean())
    base_brier = base_rate * (1.0 - base_rate)
    model_brier = float(((po_c - yo) ** 2).mean())
    meta = {"n_feat": X.shape[1], "platt": [a, b], "trained_ts": time.time(),
            "n_train": len(idx), "n_oos": len(yo), "oos_lift": round(lift, 1),
            "oos_brier": round(model_brier, 4),
            "oos_brier_base": round(base_brier, 4),
            "oos_brier_skill": round(1.0 - model_brier / max(base_brier, 1e-9), 4),
            "oos_base": round(base_rate * 100, 1)}

    def _final_blob(k):                                 # فیتِ نهایی روی همهٔ داده، روی نودها — lgbm با پیکربندیِ برنده
        try:
            if not _can_spawn():
                raise RuntimeError("process spawn unavailable")
            if k == "mlp":                              # هر seed یک نود، مونتاژ در والد
                fs = [_get_pool().submit(_full_fit_task, "mlp_seed", s, X, y) for s in MLP_SEEDS]
                return {"nets": [f.result() for f in fs]}
            return _get_pool().submit(_full_fit_task, k, lgb_extra if k == "lgbm" else None, X, y).result()
        except Exception:  # noqa: BLE001 — استخر در دسترس نیست: همین‌جا
            return _full_fit_task(k, lgb_extra if k == "lgbm" else None, X, y) if k != "mlp" \
                else _ser("mlp", _mlp_fit(X, y))

    if best_kind == "ensemble":
        meta["kind"] = "ensemble"
        with ThreadPoolExecutor(max_workers=len(kinds)) as tpool:
            blobs = list(tpool.map(_final_blob, kinds))
        meta["members"] = [{"kind": k, "blob": b} for k, b in zip(kinds, blobs)]
        Xall = np.column_stack([_logit(oof[k][idx]) for k in kinds])
        stk = _logit_fit(Xall, y[idx])                  # متا-مدلِ نهایی روی همهٔ ردیف‌های مشترک
        meta["stacker"] = {"w": stk["w"].tolist(), "mu": stk["mu"].tolist(), "sd": stk["sd"].tolist()}
        meta["member_kinds"] = kinds
    else:
        meta["kind"] = best_kind
        meta.update(_final_blob(best_kind))

    p_all = _sigmoid(a * _logit(_score_all(meta, X)) + b)
    oos_pc = np.full(n, np.nan)
    oos_pc[oos_rows] = po_c
    return meta, p_all, oos_pc


def _score_all(meta, X):
    """احتمالِ خامِ مدل (قبل از پلَت) روی ماتریسِ X — برای تک‌مدل یا ترکیب."""
    kind = meta["kind"]
    if kind == "ensemble":
        cols = []
        for mem in meta["members"]:
            model = _load_member(mem["kind"], mem["blob"])
            cols.append(_logit(np.clip(_member_pred(mem["kind"], model, X), 1e-6, 1 - 1e-6)))
        Xe = np.column_stack(cols)
        stk = {"w": np.array(meta["stacker"]["w"]), "mu": np.array(meta["stacker"]["mu"]), "sd": np.array(meta["stacker"]["sd"])}
        return _logit_pred(stk, Xe)
    return _member_pred(kind, _load_member(kind, meta), X)


# ── مدل دومِ مستقل: رگرسیونِ مستقیمِ بازده خالص (R) ──
def _ridge_fit(X, y, l2=8.0):
    """رگرسیون Ridge پایدار؛ هدف، R خالص پس از هزینه است نه صرفاً برد/باخت."""
    mu, sd = X.mean(axis=0), X.std(axis=0)
    sd = np.where(sd < 1e-9, 1.0, sd)
    Xn = (X - mu) / sd
    X1 = np.hstack([np.ones((len(Xn), 1)), Xn])
    reg = np.eye(X1.shape[1]) * float(l2)
    reg[0, 0] = 0.0
    try:
        w = np.linalg.solve(X1.T @ X1 + reg, X1.T @ y)
    except np.linalg.LinAlgError:
        w = np.linalg.pinv(X1.T @ X1 + reg) @ X1.T @ y
    return {"w": w, "mu": mu, "sd": sd}


def _ridge_pred(m, X):
    Xn = (X - m["mu"]) / m["sd"]
    return m["w"][0] + Xn @ m["w"][1:]


def _edge_fit_kind(kind, X, y):
    if kind == "lgb_edge":
        ds = lgb.Dataset(X, label=y, free_raw_data=False)
        return lgb.train(EDGE_LGB_PARAMS, ds)
    return _ridge_fit(X, y)


def _edge_pred_kind(kind, model, X):
    return model.predict(X) if kind == "lgb_edge" else _ridge_pred(model, X)


def _walk_forward_edge(X, y, ts, embargo_ms):
    kinds = ["ridge_edge"] + (["lgb_edge"] if HAS_LGBM else [])
    oos = {kind: np.full(len(y), np.nan) for kind in kinds}
    for tr_idx, te_idx in _walk_forward_splits(ts, embargo_ms):
        if len(tr_idx) < 120:
            continue
        for kind in kinds:
            model = _edge_fit_kind(kind, X[tr_idx], y[tr_idx])
            oos[kind][te_idx] = _edge_pred_kind(kind, model, X[te_idx])
    return oos


def _walk_forward_ridge_rolling(X, y, ts, embargo_ms, lookback_groups):
    """Ridge walk-forward فقط با N timestamp اخیرِ قبل از embargo؛ سازگار با تغییر رژیم."""
    ts = np.asarray(ts, dtype=np.int64)
    oos = np.full(len(y), np.nan)
    for tr_idx, te_idx in _walk_forward_splits(ts, embargo_ms):
        train_ts = np.unique(ts[tr_idx])
        if len(train_ts) > lookback_groups:
            cutoff = train_ts[-lookback_groups]
            tr_idx = tr_idx[ts[tr_idx] >= cutoff]
        if len(tr_idx) < 240:
            continue
        model = _ridge_fit(X[tr_idx], y[tr_idx])
        oos[te_idx] = _ridge_pred(model, X[te_idx])
    return oos


def _avg_rank(a):
    """رتبهٔ میانگین (مقدارهای برابر رتبهٔ یکسان می‌گیرند، نه رتبه به ترتیبِ زمان)."""
    a = np.asarray(a, float)
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty(len(a), float)
    ranks[order] = np.arange(len(a), dtype=float)
    _, inv, counts = np.unique(a, return_inverse=True, return_counts=True)
    sums = np.bincount(inv, weights=ranks)
    return (sums / counts)[inv]


def _rank_corr(a, b):
    if len(a) < 3 or np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return 0.0
    return float(np.corrcoef(_avg_rank(a), _avg_rank(b))[0, 1])


def _edge_lift(pred, actual):
    th = _thirds(pred)
    if th is None:
        return 0.0                                 # پیش‌بینیِ ثابت ⇒ لیفت صفر، نه روندِ زمانیِ آزمون
    actual = np.asarray(actual, float)
    return float(actual[th[1]].mean() - actual[th[0]].mean())


def _trained_cost_meta(tf, cost_pct, risk_pct, rows):
    """هزینه‌ای که مدل **واقعاً** یاد گرفته، برای تعدیلِ زنده (calib-F2).

    برچسب‌ها ``r − هزینهٔ هر ردیف/ریسکِ همان ردیف`` هستند، ولی مدل‌ها بینِ نمادها تجمیعی‌اند و
    هیچ ویژگی‌ای نقدشوندگی یا نوسانِ مطلق را نمی‌بیند؛ پس خروجیِ کالیبره فقط **میانگینِ**
    هزینه در واحدِ R را کم می‌کند — روی ردیف‌های calibration، جایی که intercept تنظیم شد.
    قبلاً ``trained_cost_pct=0.15`` ذخیره می‌شد و زنده ``(ردهٔ نماد − ۰٫۱۵)/ریسک`` کم می‌شد؛
    خطایی تا ±۰٫۱۵R، هم‌اندازهٔ آستانه‌های اعتماد (TRX 1h باد می‌کرد، SOL فرو می‌رفت).
    """
    cost_pct = np.asarray(cost_pct, float)[rows]
    risk_pct = np.asarray(risk_pct, float)[rows]
    return {
        "trained_cost_pct": round(float(np.mean(cost_pct)), 6) if len(cost_pct) else float(COST_PCT),
        "trained_cost_r": round(float(np.mean(cost_pct / risk_pct)), 6) if len(cost_pct) else 0.0,
        # فاندینگِ برآوردیِ زنده — همان تعریفِ هزینهٔ نمونه‌های متراکم (نیمِ براکت)
        "live_funding_pct": costs.expected_funding_pct(tf, bracket.MAX_BARS / 2),
    }


def _live_cost_delta_r(m, cost_pct, risk_pct):
    """چند R باید از خروجیِ کالیبرهٔ مدل کم شود تا هزینهٔ **همین** معامله را ببیند.

    ``(ردهٔ زنده + فاندینگِ برآوردی)/ریسکِ زنده − میانگینِ هزینهٔ R ِ آموزش``. جدولِ
    پیش از v23 (بی ``trained_cost_r``) همان فرمولِ قدیمی را می‌گیرد.
    """
    risk = max(float(risk_pct), 0.05)
    if m.get("trained_cost_r") is None:
        return (float(cost_pct) - float(m.get("trained_cost_pct", COST_PCT))) / risk
    live = float(cost_pct) + float(m.get("live_funding_pct") or 0.0)
    return live / risk - float(m["trained_cost_r"])


def _fit_edge_model(events, tf, cost_pct=COST_PCT):
    """مدل بازدهِ خالص با دادگاه زمانی مستقل و کالیبراسیون affine روی پنجرهٔ جدا."""
    if len(events) < 500:
        return None
    evs = sorted(events, key=lambda e: e["ts"])
    X = np.asarray([e["feats"] for e in evs], float)
    risk = np.asarray([max(float(e.get("risk_pct") or 0), 0.05) for e in evs])
    # هزینهٔ **هر رویداد** (ردهٔ نقدشوندگیِ لحظه‌ای + فاندینگِ واقعیِ مدتِ نگه‌داری)
    # از outcome کم می‌شود؛ هزینهٔ ثابتِ ۰٫۱۵٪ آلتِ کم‌عمق و نگه‌داریِ ۴۰روزه را نمی‌دید.
    ev_cost = np.asarray([float(e.get("cost_pct", cost_pct)) for e in evs])
    y = np.clip(
        np.asarray([float(e["r"]) for e in evs]) - ev_cost / risk,
        -1.5, 2.2,
    )
    ts = np.asarray([e["ts"] for e in evs], dtype=np.int64)
    oof_all = _walk_forward_edge(X, y, ts, 40 * TF_MS[tf])
    common = np.ones(len(y), dtype=bool)
    for pred in oof_all.values():
        common &= ~np.isnan(pred)
    idx = np.where(common)[0]
    if len(idx) < 300:
        return None
    dev_end = int(len(idx) * 0.60)
    cal_end = int(len(idx) * 0.80)
    dev, cal_rows, test_rows = idx[:dev_end], idx[dev_end:cal_end], idx[cal_end:]
    purge = bracket.MAX_BARS * TF_MS[tf]
    dev = _purge_boundary(dev, cal_rows, ts, purge)
    cal_rows = _purge_boundary(cal_rows, test_rows, ts, purge)
    if min(len(dev), len(cal_rows), len(test_rows)) < 60:
        return None

    # خانوادهٔ مدل فقط روی dev انتخاب می‌شود. calibration و test هیچ نقشی در انتخاب ندارند.
    selection = {}
    for kind, pred in oof_all.items():
        selection[kind] = {
            "rank_ic": _rank_corr(pred[dev], y[dev]),
            "lift_r": _edge_lift(pred[dev], y[dev]),
            "mae": float(np.mean(np.abs(pred[dev] - y[dev]))),
        }
    best_kind = max(
        selection,
        key=lambda k: selection[k]["rank_ic"] + 0.10 * np.clip(selection[k]["lift_r"], -1.0, 1.0),
    )
    oof = oof_all[best_kind]

    # فقط calibration شیب/بایاس را می‌بیند؛ test برای گزارش نهایی دست‌نخورده است.
    pc, yc = oof[cal_rows], y[cal_rows]
    var_pc = float(np.var(pc))
    slope = float(np.cov(pc, yc, ddof=0)[0, 1] / var_pc) if var_pc > 1e-10 else 0.0
    slope = max(0.0, min(slope, 2.0))
    intercept = float(yc.mean() - slope * pc.mean())
    pred_test = np.clip(slope * oof[test_rows] + intercept, -1.5, 2.2)
    actual_test = y[test_rows]
    rank_ic = _rank_corr(pred_test, actual_test)
    lift_r = _edge_lift(pred_test, actual_test)
    mae = float(np.mean(np.abs(pred_test - actual_test)))
    baseline = float(np.median(y[np.concatenate([dev, cal_rows])]))
    mae_base = float(np.mean(np.abs(actual_test - baseline)))
    mae_skill = 1.0 - mae / max(mae_base, 1e-9)
    residual_sd = float(np.std(actual_test - pred_test))

    by_regime = {}
    for regime in ("trend", "range", "volatile", "transition"):
        rows = [k for k, i in enumerate(test_rows) if evs[i].get("regime") == regime]
        if not rows:
            continue
        vals = actual_test[rows]
        preds = pred_test[rows]
        cutoff = max(float(np.quantile(pred_test, 0.67)), 0.0)
        selected = vals[preds >= cutoff]
        by_regime[regime] = {
            "n": len(rows),
            "avg_net_r": round(float(vals.mean()), 3),
            "win": round(float((vals > 0).mean()) * 100, 1),
            "selected_n": len(selected),
            "selected_avg_net_r": round(float(selected.mean()), 3) if len(selected) else None,
        }

    fitted = _edge_fit_kind(best_kind, X, y)
    mu, sd = X.mean(axis=0), X.std(axis=0)
    sd = np.where(sd < 1e-9, 1.0, sd)
    blob = {
        "kind": best_kind, "n_feat": X.shape[1],
        "mu": mu.tolist(), "sd": sd.tolist(),
        "calibration": [slope, intercept], "trained_ts": time.time(),
        **_trained_cost_meta(tf, ev_cost, risk, cal_rows),
        "n_train": len(idx), "n_oos": len(test_rows),
        "oos_rank_ic": round(rank_ic, 4),
        "oos_lift_r": round(lift_r, 4),
        "oos_mae": round(mae, 4), "oos_mae_base": round(mae_base, 4),
        "oos_mae_skill": round(mae_skill, 4),
        "residual_sd": round(residual_sd, 4),
        "by_regime": by_regime,
        "selection": {k: {mk: round(float(mv), 4) for mk, mv in v.items()}
                      for k, v in selection.items()},
    }
    if best_kind == "lgb_edge":
        blob["lgbm"] = fitted.model_to_string()
    else:
        blob["w"] = fitted["w"].tolist()
        # برای ridge همان scaler برازش نهایی باید ذخیره شود.
        blob["mu"], blob["sd"] = fitted["mu"].tolist(), fitted["sd"].tolist()
    return blob


def _edge_model_trusted(m):
    return bool(
        m and int(m.get("n_oos") or 0) >= 60
        and float(m.get("oos_rank_ic") or -99) >= MIN_EDGE_RANK_IC
        and float(m.get("oos_lift_r") or -99) >= MIN_EDGE_LIFT_R
        and float(m.get("oos_mae_skill") or -99) >= 0.0
    )


def _score_edge(m, feats, risk_pct, cost_pct, regime=None):
    x = np.asarray(feats, float).reshape(1, -1)
    mu, sd = np.asarray(m["mu"], float), np.asarray(m["sd"], float)
    if m.get("kind") == "lgb_edge":
        booster = _get_booster(m["lgbm"])
        raw = float(booster.predict(x)[0])
    else:
        model = {"w": np.asarray(m["w"], float), "mu": mu, "sd": sd}
        raw = float(_ridge_pred(model, x)[0])
    slope, intercept = m.get("calibration", [1.0, 0.0])
    edge_r = float(np.clip(float(slope) * raw + float(intercept), -1.5, 2.2))
    # هزینهٔ همین نماد/ریسک به‌جای میانگینِ هزینه‌ای که مدل یاد گرفته (calib-F2)
    edge_r -= _live_cost_delta_r(m, cost_pct, risk_pct)
    n_oos = max(int(m.get("n_oos") or 1), 1)
    # خطای یک پیش‌بینی تازه نباید مثل خطای میانگین با sqrt(n) تقریباً صفر شود.
    # n/20 یک اندازه‌نمونهٔ مؤثر محافظه‌کارانه برای پنجره‌های زمانی هم‌بسته است.
    uncertainty = 1.28 * float(m.get("residual_sd") or 1.0) / math.sqrt(max(n_oos / 20.0, 1.0))
    zmax = float(np.max(np.abs((x[0] - mu) / sd)))
    if zmax > 4.0:
        uncertainty += min((zmax - 4.0) * 0.05, 0.30)
    regime_stats = (m.get("by_regime") or {}).get(regime) if regime else None
    regime_ok = bool(
        regime_stats and regime_stats.get("n", 0) >= 30
        and regime_stats.get("selected_n", 0) >= 10
        and float(regime_stats.get("selected_avg_net_r") or 0) > 0.03
    )
    return {
        "edge_r": round(edge_r, 3),
        "edge_lcb_r": round(edge_r - uncertainty, 3),
        "edge_uncertainty_r": round(uncertainty, 3),
        "edge_rank_ic": m.get("oos_rank_ic"),
        "edge_lift_r": m.get("oos_lift_r"),
        "edge_mae_skill": m.get("oos_mae_skill"),
        "feature_zmax": round(zmax, 2),
        "regime_stats": regime_stats,
        "regime_ok": regime_ok,
    }


def _profit_factor(values):
    values = np.asarray(values, float)
    gains = float(values[values > 0].sum())
    losses = float(-values[values <= 0].sum())
    return gains / losses if losses > 1e-12 else (99.0 if gains > 0 else 0.0)


def _policy_sample_stats(values, baseline=0.0, ts=None, block_ms=None):
    """آمارِ یک سیاست با کرانِ پایینِ بوت‌استرپِ بلوکی.

    نسخهٔ قبلی ``n`` مؤثر را حدسی «یک‌چهارمِ n» می‌گرفت و کران را با ``sd/√n``
    می‌ساخت. با مهرِ زمانی، حالا بلوک‌های زمانی دست‌نخورده نمونه‌برداری می‌شوند
    و ``n`` مؤثر از هم‌پوشانیِ واقعی و همبستگیِ مقطعی درمی‌آید. اگر مهرِ زمانی
    در دست نباشد همان تقریبِ محافظه‌کارِ قبلی می‌ماند.
    """
    values = np.asarray(values, float)
    if not len(values):
        return {
            "n": 0, "n_eff": 0.0, "avg_net_r": None, "profit_factor": None,
            "win_rate": None, "lcb_net_r": None, "uplift_r": None, "sd_net_r": None,
        }
    avg = float(values.mean())
    sd = float(values.std())
    if ts is not None and len(ts) == len(values):
        block = block_ms or (bracket.MAX_BARS * TF_MS["1h"])
        n_eff = stats.effective_n(values, ts, block)
        lcb = stats.block_bootstrap_lcb(values, ts, block, alpha=0.10, B=400)
        if lcb is None:                       # خوشهٔ واحد: عدم‌قطعیت برآوردپذیر نیست
            lcb = avg - 1.28 * sd / math.sqrt(max(n_eff, 1.0))
    else:
        n_eff = max(len(values) / 4.0, 1.0)
        lcb = avg - 1.28 * sd / math.sqrt(n_eff)
    return {
        "n": int(len(values)),
        "n_eff": round(float(n_eff), 2),
        "avg_net_r": round(avg, 4),
        "profit_factor": round(_profit_factor(values), 4),
        "win_rate": round(float((values > 0).mean()) * 100, 2),
        "lcb_net_r": round(float(lcb), 4),
        "uplift_r": round(avg - float(baseline), 4),
        "sd_net_r": round(sd, 4),
    }


def _fit_policy_model(events, tf, cost_pct=COST_PCT):
    """سیاست نهایی انتخاب معامله با design/calibration/test زمانیِ جدا.

    دو یادگیرنده (احتمالِ برد خالص و بازده خالص) فقط ورودی سیاست‌اند. خانواده،
    وزن و نرخ گزینش روی dev/calibration انتخاب می‌شوند و مجوز نهایی فقط از
    پنجرهٔ test دست‌نخورده می‌آید. بنابراین شکست یکی از مدل‌های کمکی الزاماً
    به veto کاذب منجر نمی‌شود؛ خود تصمیم معامله داوری می‌شود.
    """
    if len(events) < 700:
        return None
    evs = sorted(events, key=lambda e: e["ts"])
    X = np.asarray([e["feats"] for e in evs], float)
    risk = np.asarray([max(float(e.get("risk_pct") or 0), 0.05) for e in evs])
    ev_cost = np.asarray([float(e.get("cost_pct", cost_pct)) for e in evs])
    net_r = np.clip(
        np.asarray([float(e["r"]) for e in evs]) - ev_cost / risk,
        -1.5, 2.2,
    )
    y = (net_r > 0).astype(float)
    ts = np.asarray([e["ts"] for e in evs], dtype=np.int64)
    embargo = 40 * TF_MS[tf]

    prob_kinds = ["logit"] + (["lgbm"] if HAS_LGBM else [])
    prob_oof = {k: _walk_forward(X, y, k, ts=ts, embargo_ms=embargo) for k in prob_kinds}
    edge_oof = _walk_forward_edge(X, net_r, ts, embargo)
    common = np.ones(len(net_r), dtype=bool)
    for pred in list(prob_oof.values()) + list(edge_oof.values()):
        common &= ~np.isnan(pred)
    idx = np.where(common)[0]
    if len(idx) < 420:
        return None
    dev_end = int(len(idx) * 0.50)
    cal_end = int(len(idx) * 0.75)
    dev, cal_rows, test_rows = idx[:dev_end], idx[dev_end:cal_end], idx[cal_end:]
    dev = _purge_boundary(dev, cal_rows, ts, embargo)
    cal_rows = _purge_boundary(cal_rows, test_rows, ts, embargo)
    if min(len(dev), len(cal_rows), len(test_rows)) < 90:
        return None

    # انتخاب خانواده فقط روی dev.
    prob_selection = {}
    base_brier = float(y[dev].mean() * (1.0 - y[dev].mean()))
    for kind, pred in prob_oof.items():
        brier = float(np.mean((pred[dev] - y[dev]) ** 2))
        skill = 1.0 - brier / max(base_brier, 1e-9)
        lift = _lift_on(pred[dev], y[dev])
        prob_selection[kind] = {"brier_skill": skill, "lift": lift}
    prob_kind = max(
        prob_selection,
        key=lambda k: prob_selection[k]["brier_skill"] + 0.003 * prob_selection[k]["lift"],
    )
    edge_selection = {}
    for kind, pred in edge_oof.items():
        edge_selection[kind] = {
            "rank_ic": _rank_corr(pred[dev], net_r[dev]),
            "lift_r": _edge_lift(pred[dev], net_r[dev]),
        }
    edge_kind = max(
        edge_selection,
        key=lambda k: edge_selection[k]["rank_ic"] + 0.10 * np.clip(edge_selection[k]["lift_r"], -1, 1),
    )

    # calibration فقط نگاشت خروجی و طراحی سیاست را می‌بیند.
    prob_raw_cal = np.clip(prob_oof[prob_kind][cal_rows], 1e-6, 1 - 1e-6)
    platt = _platt(prob_raw_cal, y[cal_rows])
    prob_cal = _sigmoid(platt[0] * _logit(prob_raw_cal) + platt[1])
    edge_raw_cal = edge_oof[edge_kind][cal_rows]
    var_edge = float(np.var(edge_raw_cal))
    edge_slope = (float(np.cov(edge_raw_cal, net_r[cal_rows], ddof=0)[0, 1]) / var_edge
                  if var_edge > 1e-10 else 0.0)
    edge_slope = max(0.0, min(edge_slope, 2.0))
    edge_intercept = float(net_r[cal_rows].mean() - edge_slope * edge_raw_cal.mean())
    edge_cal = np.clip(edge_slope * edge_raw_cal + edge_intercept, -1.5, 2.2)

    p_mu, p_sd = float(prob_cal.mean()), max(float(prob_cal.std()), 1e-6)
    e_mu, e_sd = float(edge_cal.mean()), max(float(edge_cal.std()), 1e-6)
    pz_cal, ez_cal = (prob_cal - p_mu) / p_sd, (edge_cal - e_mu) / e_sd
    min_selected = max(40, int(len(cal_rows) * 0.07))
    candidates = []
    # اگر شیبِ کالیبراسیونِ edge صفر شده، جزءِ edge اطلاعاتی ندارد؛ وزن‌دادن به آن
    # یعنی تصمیم بر اساسِ اختلافِ هزینه تقسیم بر ۱e-۶. فقط وزنِ صفر مجاز است.
    edge_weights = (0.0,) if (edge_slope <= 0.0 or e_sd < MIN_LIVE_SD) else (0.0, 0.25, 0.50, 0.75, 1.0)
    for edge_weight in edge_weights:
        score = edge_weight * ez_cal + (1.0 - edge_weight) * pz_cal
        for quantile in (0.65, 0.75, 0.85, 0.90):
            threshold = float(np.quantile(score, quantile))
            chosen = net_r[cal_rows][score >= threshold]
            if len(chosen) < min_selected:
                continue
            st = _policy_sample_stats(chosen, float(net_r[cal_rows].mean()),
                                      ts=ts[cal_rows][score >= threshold], block_ms=embargo)
            # معیار انتخاب، میانگینِ جریمه‌شده با خطای نمونه و تمرکز است؛ test هنوز دیده نشده.
            objective = float(st["avg_net_r"]) - 0.50 * float(st["sd_net_r"]) / math.sqrt(max(len(chosen) / 4, 1))
            objective += 0.04 * min(float(st["profit_factor"]) - 1.0, 1.0)
            candidates.append({
                "edge_weight": edge_weight, "quantile": quantile,
                "threshold": threshold, "objective": objective, **st,
            })
    if not candidates:
        return None
    design = max(candidates, key=lambda x: x["objective"])

    # دادگاه نهایی: وزن/quantile/threshold ثابت‌اند و هیچ انتخابی روی test انجام نمی‌شود.
    prob_raw_test = np.clip(prob_oof[prob_kind][test_rows], 1e-6, 1 - 1e-6)
    prob_test = _sigmoid(platt[0] * _logit(prob_raw_test) + platt[1])
    edge_test = np.clip(
        edge_slope * edge_oof[edge_kind][test_rows] + edge_intercept, -1.5, 2.2,
    )
    test_score = (
        design["edge_weight"] * ((edge_test - e_mu) / e_sd)
        + (1.0 - design["edge_weight"]) * ((prob_test - p_mu) / p_sd)
    )
    selected_mask = test_score >= design["threshold"]
    selected_rows = test_rows[selected_mask]
    selected_values = net_r[selected_rows]
    test_stats = _policy_sample_stats(selected_values, float(net_r[test_rows].mean()),
                                      ts=ts[selected_rows], block_ms=embargo)
    half = len(test_rows) // 2
    halves = []
    for rows in (test_rows[:half], test_rows[half:]):
        if not len(rows):
            halves.append(None)
            continue
        rawp = np.clip(prob_oof[prob_kind][rows], 1e-6, 1 - 1e-6)
        pp = _sigmoid(platt[0] * _logit(rawp) + platt[1])
        ee = np.clip(edge_slope * edge_oof[edge_kind][rows] + edge_intercept, -1.5, 2.2)
        ss = design["edge_weight"] * ((ee - e_mu) / e_sd) + \
            (1.0 - design["edge_weight"]) * ((pp - p_mu) / p_sd)
        vals = net_r[rows][ss >= design["threshold"]]
        halves.append(round(float(vals.mean()), 4) if len(vals) else None)

    by_regime = {}
    for regime in ("trend", "range", "volatile", "transition"):
        rows = [i for i in selected_rows if evs[i].get("regime") == regime]
        if not rows:
            continue
        rst = _policy_sample_stats(net_r[rows], ts=ts[rows], block_ms=embargo)
        by_regime[regime] = rst
        by_regime[regime]["veto"] = bool(
            rst["n"] >= 20 and float(rst["avg_net_r"]) < -0.05
            and float(rst["profit_factor"]) < 0.95
        )

    # مدل‌های نهایی روی همهٔ تاریخ؛ normalizer خروجیِ live فقط از توزیع ویژگی/پیش‌بینی می‌آید.
    prob_fit = _BASE[prob_kind][0](X, y)
    edge_fit = _edge_fit_kind(edge_kind, X, net_r)
    prob_final_raw = np.clip(_BASE[prob_kind][1](prob_fit, X), 1e-6, 1 - 1e-6)
    prob_final = _sigmoid(platt[0] * _logit(prob_final_raw) + platt[1])
    edge_final_raw = _edge_pred_kind(edge_kind, edge_fit, X)
    edge_final = np.clip(edge_slope * edge_final_raw + edge_intercept, -1.5, 2.2)
    live_norm = {
        "p_mu": float(prob_final.mean()), "p_sd": max(float(prob_final.std()), 1e-6),
        "edge_mu": float(edge_final.mean()), "edge_sd": max(float(edge_final.std()), 1e-6),
    }
    feat_mu, feat_sd = X.mean(axis=0), X.std(axis=0)
    feat_sd = np.where(feat_sd < 1e-9, 1.0, feat_sd)
    meta = {
        "n_feat": X.shape[1], "trained_ts": time.time(),
        **_trained_cost_meta(tf, ev_cost, risk, cal_rows),
        "prob_kind": prob_kind, "prob_model": _ser(prob_kind, prob_fit),
        "edge_kind": edge_kind,
        "edge_model": ({"lgbm": edge_fit.model_to_string()} if edge_kind == "lgb_edge"
                       else {"w": edge_fit["w"].tolist(), "mu": edge_fit["mu"].tolist(),
                             "sd": edge_fit["sd"].tolist()}),
        "platt": [float(platt[0]), float(platt[1])],
        "edge_calibration": [edge_slope, edge_intercept],
        "edge_weight": float(design["edge_weight"]),
        "select_quantile": float(design["quantile"]),
        "threshold": float(design["threshold"]),
        "calibration_policy": {k: (round(float(v), 4) if isinstance(v, (int, float)) else v)
                               for k, v in design.items()},
        "test": test_stats,
        "test_baseline_avg_r": round(float(net_r[test_rows].mean()), 4),
        "test_halves_avg_r": halves,
        "by_regime": by_regime,
        "prob_selection": {k: {mk: round(float(mv), 4) for mk, mv in v.items()}
                           for k, v in prob_selection.items()},
        "edge_selection": {k: {mk: round(float(mv), 4) for mk, mv in v.items()}
                           for k, v in edge_selection.items()},
        "live_norm": live_norm,
        "feat_mu": feat_mu.tolist(), "feat_sd": feat_sd.tolist(),
        "base_win": round(float(y[test_rows].mean()) * 100, 2),
    }
    return meta


MIN_LIVE_SD = 1e-3       # زیرِ این، نرمال‌سازیِ زنده امتیازِ ±۱۰⁵ می‌سازد


def _policy_is_degenerate(m):
    """سیاستی که جزءِ edgeاش ثابت است ولی وزن دارد، روی ردهٔ هزینهٔ نماد تصمیم می‌گیرد، نه ویژگی‌ها."""
    norm = (m or {}).get("live_norm") or {}
    w = float((m or {}).get("edge_weight") or 0.0)
    slope = float(((m or {}).get("edge_calibration") or [1.0, 0.0])[0])
    if w > 0 and (slope <= 0.0 or float(norm.get("edge_sd") or 0.0) < MIN_LIVE_SD):
        return True
    return w < 1.0 and float(norm.get("p_sd") or 0.0) < MIN_LIVE_SD


def _policy_model_trusted(m):
    if not m:
        return False
    if _policy_is_degenerate(m):
        return False
    st = m.get("test") or {}
    halves = [v for v in (m.get("test_halves_avg_r") or []) if v is not None]
    return bool(
        int(st.get("n") or 0) >= POLICY_MIN_TEST
        and float(st.get("avg_net_r") or -99) >= POLICY_MIN_AVG_R
        and float(st.get("profit_factor") or 0) >= POLICY_MIN_PF
        and float(st.get("uplift_r") or -99) >= POLICY_MIN_UPLIFT_R
        and float(st.get("lcb_net_r") or -99) >= POLICY_MIN_LCB_R
        and len(halves) == 2 and min(float(v) for v in halves) >= -0.20
    )


def _score_policy(m, feats, risk_pct, cost_pct, regime=None):
    x = np.asarray(feats, float).reshape(1, -1)
    prob_model = _load_member(m["prob_kind"], m["prob_model"])
    raw_p = float(np.clip(_member_pred(m["prob_kind"], prob_model, x)[0], 1e-6, 1 - 1e-6))
    pa, pb = m.get("platt", [1.0, 0.0])
    p = float(_sigmoid(pa * _logit(np.asarray([raw_p]))[0] + pb))
    edge_blob = m["edge_model"]
    if m["edge_kind"] == "lgb_edge":
        raw_edge = float(_get_booster(edge_blob["lgbm"]).predict(x)[0])
    else:
        edge_model = {
            "w": np.asarray(edge_blob["w"], float),
            "mu": np.asarray(edge_blob["mu"], float),
            "sd": np.asarray(edge_blob["sd"], float),
        }
        raw_edge = float(_ridge_pred(edge_model, x)[0])
    es, ei = m.get("edge_calibration", [1.0, 0.0])
    edge_r = float(np.clip(float(es) * raw_edge + float(ei), -1.5, 2.2))
    edge_r -= _live_cost_delta_r(m, cost_pct, risk_pct)
    norm = m["live_norm"]
    pz = (p - float(norm["p_mu"])) / max(float(norm["p_sd"]), 1e-6)
    ez = (edge_r - float(norm["edge_mu"])) / max(float(norm["edge_sd"]), 1e-6)
    weight = float(m["edge_weight"])
    score = weight * ez + (1.0 - weight) * pz
    margin = score - float(m["threshold"])
    feat_mu, feat_sd = np.asarray(m["feat_mu"], float), np.asarray(m["feat_sd"], float)
    zmax = float(np.max(np.abs((x[0] - feat_mu) / feat_sd)))
    regime_stats = (m.get("by_regime") or {}).get(regime) if regime else None
    regime_veto = bool((regime_stats or {}).get("veto", False))
    tst = m.get("test") or {}
    mean_uncertainty = max(
        float(tst.get("avg_net_r") or 0) - float(tst.get("lcb_net_r") or 0), 0.0,
    )
    return {
        "p_win": round(p * 100, 1),
        "p_win_low": round(max(0.01, p - 0.08) * 100, 1),
        "p_win_high": round(min(0.99, p + 0.08) * 100, 1),
        "p_uncertainty": 8.0,
        "edge_r": round(edge_r, 3),
        "edge_lcb_r": round(edge_r - mean_uncertainty, 3),
        "edge_uncertainty_r": round(mean_uncertainty, 3),
        "policy_score": round(score, 4),
        "policy_margin": round(margin, 4),
        "policy_pass": bool(margin >= 0 and zmax <= 6.0 and not regime_veto),
        "policy_test": tst,
        "feature_zmax": round(zmax, 2),
        "regime_stats": regime_stats,
        "regime_veto": regime_veto,
        "regime_ok": not regime_veto,
    }


def _time_partitions(rows, timestamps, purge_ms=0):
    """سه بخش زمانی بدون شکستن گروه‌های دارای timestamp یکسان."""
    rows = np.asarray(rows, dtype=int)
    if not len(rows):
        return np.array([], int), np.array([], int), np.array([], int)
    ts = np.asarray(timestamps, dtype=np.int64)
    uniq = np.unique(ts[rows])
    if len(uniq) < 12:
        return np.array([], int), np.array([], int), np.array([], int)
    dcut = uniq[min(int(len(uniq) * 0.50), len(uniq) - 2)]
    ccut = uniq[min(int(len(uniq) * 0.75), len(uniq) - 1)]
    purge = int(purge_ms or 0)
    dev = rows[ts[rows] < dcut - purge]                 # purge روی مرز، نه فقط برش
    cal = rows[(ts[rows] >= dcut) & (ts[rows] < ccut - purge)]
    test = rows[ts[rows] >= ccut]
    return dev, cal, test


def _cross_section_mask(score, rows, timestamps, quantile, min_group=8):
    """انتخاب بهترین درصد در هر لحظه؛ قرارداد آموزش دقیقاً با رتبه‌بندی live یکسان می‌شود."""
    score = np.asarray(score, float)
    rows = np.asarray(rows, dtype=int)
    timestamps = np.asarray(timestamps, dtype=np.int64)
    chosen = np.zeros(len(rows), dtype=bool)
    for stamp in np.unique(timestamps[rows]):
        pos = np.where(timestamps[rows] == stamp)[0]
        if len(pos) < min_group:
            continue
        cutoff = float(np.quantile(score[pos], quantile))
        chosen[pos] = score[pos] >= cutoff
    return chosen


def _fit_action_policy(dir_X, dir_y, dir_R, tf, cost_pct=COST_PCT):
    """مدل مستقیمِ انتخاب بین long / short / no-trade روی نمونه‌های متراکم.

    برای هر timestamp هر دو نتیجهٔ barrier در دسترس است، اما تقسیم walk-forward
    بر اساس خود timestamp انجام می‌شود؛ دو سمتِ یک لحظه هرگز دو سوی مرز train/test
    قرار نمی‌گیرند. آستانهٔ no-trade روی calibration طراحی و فقط یک‌بار روی test
    دست‌نخورده داوری می‌شود.
    """
    if len(dir_y) < 1_500:
        return None
    pairs = sorted(zip(dir_y, dir_X, dir_R), key=lambda p: p[0][0])
    if len(pairs) > ACTION_MAX_SAMPLES:
        # پوشش یکنواختِ کل تاریخ، نه بریدنِ صرفاً ابتدای/انتهای رژیم؛ timestamp در split حفظ می‌شود.
        keep = np.linspace(0, len(pairs) - 1, ACTION_MAX_SAMPLES, dtype=int)
        pairs = [pairs[i] for i in keep]
    if not pairs or not (
        isinstance(pairs[0][1], (list, tuple)) and len(pairs[0][1]) == 2
        and isinstance(pairs[0][1][0], (list, tuple, np.ndarray))
    ):
        return None
    X_long = np.asarray([p[1][0] for p in pairs], float)
    X_short = np.asarray([p[1][1] for p in pairs], float)
    timestamps = np.asarray([p[0][0] for p in pairs], dtype=np.int64)
    risks = np.asarray([max(float(p[2][2]), 0.05) for p in pairs], float)
    sample_cost = np.asarray([float(p[2][3]) if len(p[2]) > 3 else float(cost_pct)
                              for p in pairs], float)
    net_long = np.clip(np.asarray([p[2][0] for p in pairs], float) - sample_cost / risks, -1.5, 2.2)
    net_short = np.clip(np.asarray([p[2][1] for p in pairs], float) - sample_cost / risks, -1.5, 2.2)
    n, d = X_long.shape
    X = np.empty((2 * n, d), float)
    target = np.empty(2 * n, float)
    X[0::2], X[1::2] = X_long, X_short
    target[0::2], target[1::2] = net_long, net_short
    ts_stack = np.repeat(timestamps, 2)
    # برچسب‌ها نتیجهٔ براکتِ ۴۰ کندلی‌اند (کندل‌های i+1..i+40)، نه افقِ جهت‌یاب (۳۰-۳۶ کندل):
    # purge/embargo و بلوکِ بوت‌استرپ باید کلِ عمرِ برچسب را بپوشانند (calib-F8)
    embargo = (bracket.MAX_BARS + 1) * TF_MS[tf]
    oof_all = _walk_forward_edge(X, target, ts_stack, embargo)
    for window in ACTION_ROLLING_WINDOWS[tf]:
        oof_all[f"ridge_roll_{window}"] = _walk_forward_ridge_rolling(
            X, target, ts_stack, embargo, window)
    valid = np.ones(n, dtype=bool)
    for pred in oof_all.values():
        valid &= ~np.isnan(pred[0::2]) & ~np.isnan(pred[1::2])
    rows = np.where(valid)[0]
    dev, cal_rows, test_rows = _time_partitions(rows, timestamps, purge_ms=embargo)
    if min(len(dev), len(cal_rows), len(test_rows)) < 180:
        return None

    selection = {}
    for kind, pred in oof_all.items():
        pl, ps = pred[0::2], pred[1::2]
        chosen = np.where(pl[dev] >= ps[dev], net_long[dev], net_short[dev])
        strength = np.maximum(pl[dev], ps[dev])
        selection[kind] = {
            "rank_ic": _rank_corr(strength, chosen),
            "lift_r": _edge_lift(strength, chosen),
            "avg_chosen_r": float(chosen.mean()),
        }
    best_kind = max(
        selection,
        key=lambda k: selection[k]["rank_ic"] + 0.10 * np.clip(selection[k]["lift_r"], -1, 1),
    )
    raw = oof_all[best_kind]

    # یک calibration مشترک برای دو سمت؛ مدل باید بازده را در یک مقیاس قابل‌مقایسه بسازد.
    cal_stack = np.ravel(np.column_stack([2 * cal_rows, 2 * cal_rows + 1]))
    rc, yc = raw[cal_stack], target[cal_stack]
    var_rc = float(np.var(rc))
    slope = float(np.cov(rc, yc, ddof=0)[0, 1] / var_rc) if var_rc > 1e-10 else 0.0
    slope = max(0.0, min(slope, 2.0))
    intercept = float(yc.mean() - slope * rc.mean())
    pred_l_cal = np.clip(slope * raw[0::2][cal_rows] + intercept, -1.5, 2.2)
    pred_s_cal = np.clip(slope * raw[1::2][cal_rows] + intercept, -1.5, 2.2)
    strength_cal = np.maximum(pred_l_cal, pred_s_cal)
    margin_cal = np.abs(pred_l_cal - pred_s_cal)
    chosen_cal = np.where(pred_l_cal >= pred_s_cal, net_long[cal_rows], net_short[cal_rows])
    s_mu, s_sd = float(strength_cal.mean()), max(float(strength_cal.std()), 1e-6)
    m_mu, m_sd = float(margin_cal.mean()), max(float(margin_cal.std()), 1e-6)
    sz, mz = (strength_cal - s_mu) / s_sd, (margin_cal - m_mu) / m_sd
    min_selected = max(80, int(len(cal_rows) * 0.05))
    candidates = []
    for margin_weight in (0.0, 0.20, 0.40):
        score = sz + margin_weight * mz
        for quantile in (0.50, 0.65, 0.75, 0.85, 0.90):
            select_mask = _cross_section_mask(score, cal_rows, timestamps, quantile)
            vals = chosen_cal[select_mask]
            if len(vals) < min_selected:
                continue
            static_cal = max(float(net_long[cal_rows].mean()), float(net_short[cal_rows].mean()))
            st = _policy_sample_stats(vals, static_cal,
                                      ts=timestamps[cal_rows][select_mask], block_ms=embargo)
            objective = float(st["avg_net_r"]) - 0.50 * float(st["sd_net_r"]) / math.sqrt(max(len(vals) / 4, 1))
            objective += 0.04 * min(float(st["profit_factor"]) - 1.0, 1.0)
            candidates.append({
                "margin_weight": margin_weight, "quantile": quantile,
                "threshold": 0.0, "objective": objective, **st,
            })
    if not candidates:
        return None
    design = max(candidates, key=lambda x: x["objective"])

    def _evaluate_ts(base_rows):
        """مهرِ زمانیِ سطرهای انتخاب‌شده — برای بوت‌استرپِ بلوکی لازم است."""
        _vals, _all, _score = None, None, None
        pl = np.clip(slope * raw[0::2][base_rows] + intercept, -1.5, 2.2)
        ps = np.clip(slope * raw[1::2][base_rows] + intercept, -1.5, 2.2)
        strength = np.maximum(pl, ps)
        margin = np.abs(pl - ps)
        score = (strength - s_mu) / s_sd + design["margin_weight"] * ((margin - m_mu) / m_sd)
        mask = _cross_section_mask(score, base_rows, timestamps, design["quantile"])
        return timestamps[base_rows][mask]

    def _evaluate(base_rows):
        pl = np.clip(slope * raw[0::2][base_rows] + intercept, -1.5, 2.2)
        ps = np.clip(slope * raw[1::2][base_rows] + intercept, -1.5, 2.2)
        strength = np.maximum(pl, ps)
        margin = np.abs(pl - ps)
        score = (strength - s_mu) / s_sd + design["margin_weight"] * ((margin - m_mu) / m_sd)
        actual = np.where(pl >= ps, net_long[base_rows], net_short[base_rows])
        select_mask = _cross_section_mask(
            score, base_rows, timestamps, design["quantile"])
        return actual[select_mask], actual, score

    selected_test, all_test, _ = _evaluate(test_rows)
    # benchmark واقعی، سیاستِ پیش‌بینی‌گر نیست: بهترین سمت ثابتِ long یا short در
    # همان test (حتی با مزیت نگاه پس از واقعه). سیاست باید از این حریفِ سخت بهتر باشد.
    static_test = max(float(net_long[test_rows].mean()), float(net_short[test_rows].mean()))
    _sel_ts = _evaluate_ts(test_rows)
    test_stats = _policy_sample_stats(selected_test, static_test,
                                      ts=_sel_ts, block_ms=embargo)
    halves = []
    test_ts = np.unique(timestamps[test_rows])
    hcut = test_ts[len(test_ts) // 2]
    for hr in (test_rows[timestamps[test_rows] < hcut], test_rows[timestamps[test_rows] >= hcut]):
        vals, _all, _score = _evaluate(hr)
        halves.append(round(float(vals.mean()), 4) if len(vals) else None)

    rolling_window = int(best_kind.rsplit("_", 1)[1]) if best_kind.startswith("ridge_roll_") else None
    base_fit = np.arange(n)
    if rolling_window is not None:
        uniq_fit = np.unique(timestamps)
        if len(uniq_fit) > rolling_window:
            base_fit = np.where(timestamps >= uniq_fit[-rolling_window])[0]
    fit_stack = np.ravel(np.column_stack([2 * base_fit, 2 * base_fit + 1]))
    fit_kind = "ridge_edge" if best_kind.startswith("ridge_roll_") else best_kind
    fitted = _edge_fit_kind(fit_kind, X[fit_stack], target[fit_stack])
    final_l_raw = _edge_pred_kind(fit_kind, fitted, X_long[base_fit])
    final_s_raw = _edge_pred_kind(fit_kind, fitted, X_short[base_fit])
    final_l = np.clip(slope * final_l_raw + intercept, -1.5, 2.2)
    final_s = np.clip(slope * final_s_raw + intercept, -1.5, 2.2)
    final_strength = np.maximum(final_l, final_s)
    final_margin = np.abs(final_l - final_s)
    feat_mu, feat_sd = X.mean(axis=0), X.std(axis=0)
    feat_sd = np.where(feat_sd < 1e-9, 1.0, feat_sd)
    return {
        "kind": best_kind, "n_feat": d, "trained_ts": time.time(),
        **_trained_cost_meta(tf, sample_cost, risks, cal_rows),
        "model": ({"lgbm": fitted.model_to_string()} if fit_kind == "lgb_edge"
                  else {"w": fitted["w"].tolist(), "mu": fitted["mu"].tolist(), "sd": fitted["sd"].tolist()}),
        "rolling_groups": rolling_window,
        "calibration": [slope, intercept],
        "margin_weight": float(design["margin_weight"]),
        "select_quantile": float(design["quantile"]),
        "threshold": 0.0,
        "selection_mode": "cross_section",
        "calibration_policy": {k: (round(float(v), 4) if isinstance(v, (int, float)) else v)
                               for k, v in design.items()},
        "test": test_stats,
        "test_baseline_avg_r": round(static_test, 4),
        "test_model_chosen_all_avg_r": round(float(all_test.mean()), 4),
        "test_halves_avg_r": halves,
        "selection": {k: {mk: round(float(mv), 4) for mk, mv in v.items()}
                      for k, v in selection.items()},
        "live_norm": {
            "strength_mu": float(final_strength.mean()),
            "strength_sd": max(float(final_strength.std()), 1e-6),
            "margin_mu": float(final_margin.mean()),
            "margin_sd": max(float(final_margin.std()), 1e-6),
        },
        "feat_mu": feat_mu.tolist(), "feat_sd": feat_sd.tolist(),
    }


def _action_policy_trusted(m):
    if not m:
        return False
    st = m.get("test") or {}
    halves = [v for v in (m.get("test_halves_avg_r") or []) if v is not None]
    return bool(
        int(st.get("n") or 0) >= 80
        and float(st.get("avg_net_r") or -99) >= 0.03
        and float(st.get("profit_factor") or 0) >= 1.05
        and float(st.get("uplift_r") or -99) >= 0.06
        and float(st.get("lcb_net_r") or -99) >= -0.08
        and len(halves) == 2 and min(float(v) for v in halves) >= -0.15
    )


def _action_edge(m, x):
    blob = m["model"]
    if m["kind"] == "lgb_edge":
        return float(_get_booster(blob["lgbm"]).predict(x)[0])
    model = {
        "w": np.asarray(blob["w"], float),
        "mu": np.asarray(blob["mu"], float),
        "sd": np.asarray(blob["sd"], float),
    }
    return float(_ridge_pred(model, x)[0])


def _score_action_policy(m, long_feats, short_feats, risk_pct, cost_pct):
    xl = np.asarray(long_feats, float).reshape(1, -1)
    xs = np.asarray(short_feats, float).reshape(1, -1)
    slope, intercept = m.get("calibration", [1.0, 0.0])
    long_r = float(np.clip(float(slope) * _action_edge(m, xl) + float(intercept), -1.5, 2.2))
    short_r = float(np.clip(float(slope) * _action_edge(m, xs) + float(intercept), -1.5, 2.2))
    cost_delta = _live_cost_delta_r(m, cost_pct, risk_pct)      # هر دو سمت یکسان ⇒ سمت عوض نمی‌شود
    long_r -= cost_delta
    short_r -= cost_delta
    side = "long" if long_r >= short_r else "short"
    strength, margin = max(long_r, short_r), abs(long_r - short_r)
    norm = m["live_norm"]
    score = (
        (strength - float(norm["strength_mu"])) / max(float(norm["strength_sd"]), 1e-6)
        + float(m["margin_weight"])
        * (margin - float(norm["margin_mu"])) / max(float(norm["margin_sd"]), 1e-6)
    )
    # در سیاست مقطعی score فقط برای رتبه‌دادن به ارزهای همان snapshot است؛
    # آستانهٔ مطلق بین رژیم‌ها حمل نمی‌شود.
    policy_margin = score
    feat_mu, feat_sd = np.asarray(m["feat_mu"], float), np.asarray(m["feat_sd"], float)
    chosen_x = xl[0] if side == "long" else xs[0]
    zmax = float(np.max(np.abs((chosen_x - feat_mu) / feat_sd)))
    tst = m.get("test") or {}
    mean_uncertainty = max(
        float(tst.get("avg_net_r") or 0) - float(tst.get("lcb_net_r") or 0), 0.0,
    )
    edge_r = long_r if side == "long" else short_r
    return {
        "side": side,
        "source": "action_policy",
        "policy_trusted": True,
        "policy_pass": bool(zmax <= 6.0),
        "policy_score": round(score, 4),
        "policy_margin": round(policy_margin, 4),
        "market_rank_required": True,
        "select_quantile": float(m.get("select_quantile", 0.75)),
        "policy_test": tst,
        "p_win": None, "p_win_low": None, "p_win_high": None, "p_uncertainty": None,
        "n": int(tst.get("n") or 0),
        "avg_r": round(edge_r, 3),
        "edge_r": round(edge_r, 3),
        "edge_lcb_r": round(edge_r - mean_uncertainty, 3),
        "edge_uncertainty_r": round(mean_uncertainty, 3),
        "ev_pct": round(edge_r * float(risk_pct), 3),
        "ev_lcb_pct": round((edge_r - mean_uncertainty) * float(risk_pct), 3),
        "feature_zmax": round(zmax, 2),
        "regime_veto": False, "regime_ok": True, "regime_stats": None,
        "reliability": ("خوب" if int(tst.get("n") or 0) >= 200
                        and float(tst.get("lcb_net_r") or -99) >= 0 else "متوسط"),
        "base": None, "oos_lift": round(float(tst.get("uplift_r") or 0) * 100, 1),
        "edge_trusted": True, "edge_rank_ic": None, "edge_lift_r": None,
        "authority": "action_policy",
    }


# عمق تاریخچه آموزش به‌ازای هر تایم‌فریم — نمونه‌های مدل را چند برابر می‌کند
# 4h از ۴۰۰۰ به ۹۰۰۰ کندل (~۱٫۸ → ~۴ سال): با برچسبِ ۴۰ کندلی هر بلوکِ مستقل ۶٫۷ روز است؛
# پنجرهٔ ۲۵٪ِ ۱٫۸ سال فقط ~۲۵ بلوک می‌داد و هیچ فرضیه‌ای حتی در اصل به کفِ n مؤثر
# نمی‌رسید. داده‌ی بیشتر توانِ آزمون را یکنواخت بالا می‌برد؛ پیش از هر پیش‌ثبت تعیین شد.
BARS = {"15m": 8000, "1h": 6000, "4h": 9000, "1d": 3000}
HTF_OF = {"15m": "4h", "1h": "4h", "4h": "1d", "1d": None}
TF_MS = {"15m": 900000, "1h": 3600000, "4h": 14400000, "1d": 86400000}

_lock = threading.Lock()
_state = {"building": False, "progress": "", "done": 0, "total": 0, "error": None}
_table = None


def zbin(z):
    az = abs(z)
    return "z1" if az < 2.0 else "z2" if az < 2.5 else "z3"


def keys_for(direction, z, votes, trending, btc_align):
    d = "L" if direction == 1 else "S"
    v = "v4" if votes >= 4 else "v3"
    t = "T" if trending else "R"
    b = "B1" if btc_align else "B0"
    zb = zbin(z)
    return [f"{d}|{zb}|{v}|{t}|{b}", f"{d}|{zb}|{t}", f"{zb}|{t}", "all"]


def mom_norm_map(ts_list, closes, look=20, win=250):
    """نقشه ts→مومنتوم نرمال‌شده [-1,1] با انحراف‌معیارِ *علّیِ متحرک* (فقط داده گذشته — بدون نشتی آینده).
    مشترک آموزش/اجرا: هر ردیف با انحراف‌معیار پنجرهٔ گذشتهٔ خودش مقیاس می‌شود."""
    out, rets = {}, []
    for i in range(look, len(closes)):
        r = closes[i] / closes[i - look] - 1
        rets.append(r)
        seg = rets[-win:]
        sd = (float(np.std(seg)) if len(seg) >= 10 else 0.02) or 0.02
        out[ts_list[i]] = max(-1.0, min(1.0, r / (2 * sd)))
    return out


def market_state_maps(hists, basket=None):
    """نقشه‌های وضعیتِ کلِ بازار ``{"dom": {ts: m}, "ethbtc": {ts: m}}`` — **یک** کد برای آموزش و اجرا.

    تعریف (calib-F1، نسخهٔ ۲۳):
    * ``dom`` — مومنتومِ نرمال‌شدهٔ سهمِ BTC از حجمِ دلاریِ سبدِ ثابتِ ``MARKET_BASKET``، فقط در
      لحظه‌هایی که **همهٔ** ارزهای سبد کندل دارند (جمعیتِ ناقص ⇒ بی‌مقدار ⇒ ۰). قبلاً آموزش
      ۱۰۰ ارز را جمع می‌زد و اجرا هر چه در کش بود؛ کشِ نیمه‌تازه سهمِ BTC در آخرین کندل را ۱٫۰ و
      دامیننس را در ~۹۷٪ کندل‌ها +۱ می‌کرد.
    * ``ethbtc`` — مومنتومِ نسبتِ ETH/BTC روی کندل‌های مشترک (بی‌تغییر).
    * پهنای بازار و رتبهٔ نسبی خنثی‌اند (``features.DISABLED``).

    مقدارِ هر لحظه فقط به ۲۷۰ کندلِ مشترکِ قبلش بسته است، پس ۴۲۰ کندلِ زنده همان عددِ آموزش را می‌دهد.
    """
    basket = tuple(basket or MARKET_BASKET)
    out = {"dom": {}, "ethbtc": {}}
    if basket and all((hists or {}).get(s) for s in basket):
        tot, btc, cnt = {}, {}, {}
        for s in basket:
            kl = hists[s]
            for t, c_, v_ in zip(kl["t"], kl["c"], kl["v"]):
                dv = float(c_) * float(v_)
                if not math.isfinite(dv):
                    continue
                tot[t] = tot.get(t, 0.0) + dv
                cnt[t] = cnt.get(t, 0) + 1
                if s == "BTCUSDT":
                    btc[t] = dv
        dom_ts = sorted(t for t, k in cnt.items() if k == len(basket) and tot[t] > 0 and t in btc)
        if len(dom_ts) > 40:
            out["dom"] = mom_norm_map(dom_ts, [btc[t] / tot[t] for t in dom_ts])
    kb, ke = (hists or {}).get("BTCUSDT"), (hists or {}).get("ETHUSDT")
    if kb and ke:
        bb_ = dict(zip(kb["t"], kb["c"]))
        ee_ = dict(zip(ke["t"], ke["c"]))
        common = sorted(t for t in ee_ if t in bb_)
        if len(common) > 40:
            out["ethbtc"] = mom_norm_map(common, [ee_[t] / bb_[t] for t in common])
    return out


def market_state_at(hists, tk, basket=None):
    """وضعیتِ بازار در **یک** کندلِ مشخص (زمانِ بازِ ``tk``) — همان کلیدهای ``extras`` زنده.
    کندلی که در سبد کامل نیست مقدارِ خنثی (۰) می‌گیرد، مثلِ ``dict.get(ts, 0.0)`` ِ آموزش."""
    maps = market_state_maps(hists, basket)
    return {"breadth": 0.0,
            "dom": float(maps["dom"].get(tk, 0.0)),
            "ethbtc": float(maps["ethbtc"].get(tk, 0.0))}


def _htf_sign(zmap, ts, htf):
    """جهت آخرین کندل بسته‌شده تایم بالاتر پیش از ts — بدون نگاه به آینده."""
    if not zmap or htf is None:
        return 0
    p = TF_MS[htf]
    t0 = (ts // p) * p - p
    z = zmap.get(t0)
    if z is None:
        z = zmap.get(t0 - p)
    if z is None:
        return 0
    return 1 if z > 0.3 else -1 if z < -0.3 else 0


# ───────────────────── استخراج رویدادها + نمونه‌های جهت‌یاب ─────────────────────
def extract_events(sym, kl, tf, fz_list, rs_map, htf_zmap, btc_zmap, gold_map=None,
                   breadth_map=None, dom_map=None, ethbtc_map=None, funding_rows=None):
    o = np.array(kl["o"], float); h = np.array(kl["h"], float)
    l = np.array(kl["l"], float); c = np.array(kl["c"], float)
    v = np.array(kl["v"], float); ts_arr = kl["t"]
    cs = engine.component_series(o, h, l, c, v)
    events = []
    cooldown = 0
    n = len(c)

    # نمونه‌های مدل جهت‌یاب همیشه‌روشن: هر ۴ کندل یک نمونه — برچسب = جهت بازده تا افق H
    H = engine.HORIZON[tf]
    dir_X, dir_y, dir_R = [], [], []
    # فاندینگ: آخرین تسویهٔ پیش از **بسته‌شدنِ** کندل — همان قاعدهٔ اجرا (features.funding_z_at)
    fz_keys = [row[0] for row in fz_list] if fz_list else []

    def _fz(ts_):
        return features.funding_z_at(fz_list, ts_, TF_MS[tf], keys=fz_keys)

    def _barrier(i, sig):
        res = bracket.signal_trade(o, h, l, c, cs["a14"][i], i, sig)
        return None if res is None else res["gross_r"]

    # ── شبکهٔ نمونه‌ها ──
    # * سراسری، نه اندیسِ هر ارز (calib-F14): ارزی که وسطِ پنجره لیست شده بود از اندیسِ خودش
    #   هر ۴ کندل نمونه می‌گرفت و با بقیه هیچ مهرِ زمانیِ مشترکی نداشت؛ گروه‌های مقطعیِ
    #   سیاستِ عمل (رتبه در هر لحظه) به ۴ زیرگروه می‌شکست. حالا فقط کندل‌هایی با
    #   (ts / طولِ کندل) بخش‌پذیر بر ۴ — برای همهٔ ارزها یکسان.
    # * پایان: براکتِ کاملِ ۴۰ کندلی باید جا شود (calib-F13)؛ قبلاً نمونه‌های آخر پس از
    #   H+1..۳۹ کندل روی c[n-1] «تایم‌اوت» می‌خوردند (همان چیزی که core2.replay_sides دور می‌ریزد).
    bar_ms = TF_MS[tf]
    dense_end = min(n - H - 1, n - bracket.MAX_BARS)
    for i in range(260, dense_end):
        ts = ts_arr[i]
        if (int(ts) // bar_ms) % DENSE_STRIDE:
            continue
        entry = o[i + 1]
        ret = c[i + H] / entry - 1
        bz = btc_zmap.get(ts) if btc_zmap else None
        common_kwargs = dict(
            funding_z=_fz(ts),
            rs_rank=(rs_map.get(ts, {}) or {}).get(sym, 0.5),
            htf_sign=_htf_sign(htf_zmap, ts, HTF_OF[tf]),
            gold_m=(gold_map or {}).get(ts, 0.0),
            breadth_m=(breadth_map or {}).get(ts, 0.0), dom_m=(dom_map or {}).get(ts, 0.0),
            ethbtc_m=(ethbtc_map or {}).get(ts, 0.0),
        )
        long_feats = engine.event_features(
            cs, c, i, 1, None, ts,
            btc_align=(sym == "BTCUSDT") or (bz is None) or (bz > -0.3),
            **common_kwargs)
        short_feats = engine.event_features(
            cs, c, i, -1, None, ts,
            btc_align=(sym == "BTCUSDT") or (bz is None) or (bz < 0.3),
            **common_kwargs)
        dir_X.append((long_feats, short_feats))
        dir_y.append((ts, 1.0 if ret > 0 else 0.0))
        lv = bracket.levels(entry, cs["a14"][i], 1)
        # هزینهٔ لحظه‌ای (از حجمِ ۲۴ ساعتهٔ همان کندل‌ها) + فاندینگِ برآوردیِ نیمهٔ افق
        dense_cost = (costs.point_in_time_tier(c, v, i, tf)
                      + costs.expected_funding_pct(tf, bracket.MAX_BARS / 2))
        dir_R.append((_barrier(i, 1), _barrier(i, -1),
                      lv["risk_pct"], round(dense_cost, 6)))  # (لانگ، شورت، ریسک٪، هزینه٪)
    for i in range(260, n - 42):
        if cooldown > 0:
            cooldown -= 1
            continue
        sig, setup = engine.setup_signal(cs, o, h, l, c, i)
        if sig == 0:
            continue
        ts = ts_arr[i]
        res = bracket.signal_trade(o, h, l, c, cs["a14"][i], i, sig)
        if res is None:
            continue
        out_r = res["gross_r"]
        entry_ts = ts_arr[i + 1]
        exit_ts = ts_arr[min(res["exit_idx"], n - 1)]
        ev_cost = costs.event_cost_pct(sig, c, v, i, tf, entry_ts, exit_ts, funding_rows)
        b, s = engine.votes_at(cs, i)
        votes = b if sig == 1 else s
        trending = cs["adx"][i] >= 22 and cs["chop"][i] < 55
        if sym == "BTCUSDT":
            btc_align = True
        else:
            bz = btc_zmap.get(ts) if btc_zmap else None
            btc_align = (bz is None) or (bz * sig > -0.3)
        feats = engine.event_features(
            cs, c, i, sig, setup, ts,
            funding_z=_fz(ts),
            rs_rank=(rs_map.get(ts, {}) or {}).get(sym, 0.5),
            htf_sign=_htf_sign(htf_zmap, ts, HTF_OF[tf]),
            btc_align=btc_align,
            gold_m=(gold_map or {}).get(ts, 0.0),
            breadth_m=(breadth_map or {}).get(ts, 0.0), dom_m=(dom_map or {}).get(ts, 0.0),
            ethbtc_m=(ethbtc_map or {}).get(ts, 0.0))
        events.append({"ts": ts, "sym": sym, "dir": sig, "setup": setup, "feats": feats,
                       "r": round(float(out_r), 3),
                       "risk_pct": round(float(res["risk_pct"]), 4),
                       "bars_held": int(res["bars_held"]),
                       "outcome": res["outcome"],
                       "cost_pct": ev_cost,            # رده‌ای + فاندینگ — نه ۰٫۱۵٪ ثابت
                       "z": float(cs["z"][i]), "votes": int(votes),
                       "trend": bool(trending), "btc": bool(btc_align),
                       "regime": engine.regime_at(cs, c, i)[0]})
        cooldown = 6
    zmap = {t: float(zz) for t, zz in zip(ts_arr, cs["z"])}
    return events, zmap, dir_X, dir_y, dir_R


def _frozen_exclusion(tf, prereg):
    """ts → آیا این ردیف بیرون از آموزش می‌ماند: داخلِ پنجرهٔ منجمد، یا در ``MAX_BARS+1`` کندلِ
    پیش از شروعش — برچسبِ ۴۰ کندلیِ آن ردیف تا داخلِ پنجره می‌رسد (purge، calib-F9)."""
    w = ((prereg or {}).get("final_windows") or {}).get(tf)
    if not w:
        return lambda _ts: False
    lo = int(w["start_ms"]) - (bracket.MAX_BARS + 1) * TF_MS[tf]
    hi = int(w["end_ms"])
    return lambda ts: lo <= int(ts) < hi


def _event_cells(events):
    """سطل‌های پس‌گردِ ``lookup`` — [n، برد، جمعِ R، جمعِ ریسک٪] برای هر کلید."""
    cells = {}
    for e in events:
        for key in keys_for(e["dir"], e["z"], e["votes"], e["trend"], e["btc"]):
            cell = cells.setdefault(key, [0, 0, 0.0, 0.0])
            cell[0] += 1
            cell[1] += 1 if e["r"] > 0 else 0
            cell[2] += e["r"]
            cell[3] += e["risk_pct"]
    return cells


# ───────────────────── آموزش لجستیک (هسته مشترک، numpy خالص) ─────────────────────
def _fit_logistic(X, y):
    """برازش با تقسیم زمانی ۷۰/۳۰ (داده باید از قبل بر اساس زمان مرتب باشد)."""
    n = len(y)
    cut = int(n * 0.7)
    mu, sd = X[:cut].mean(axis=0), X[:cut].std(axis=0)
    sd = np.where(sd < 1e-9, 1.0, sd)
    Xn = (X - mu) / sd
    X1 = np.hstack([np.ones((n, 1)), Xn])
    w = np.zeros(X1.shape[1])
    lam, lr = 0.01, 0.3
    for _ in range(600):
        p = 1 / (1 + np.exp(-np.clip(X1[:cut] @ w, -30, 30)))
        grad = X1[:cut].T @ (p - y[:cut]) / cut + lam * w
        w -= lr * grad
    p_oos = 1 / (1 + np.exp(-np.clip(X1[cut:] @ w, -30, 30)))
    y_oos = y[cut:]
    return {"w": w.tolist(), "mu": mu.tolist(), "sd": sd.tolist(),
            "n_train": cut, "n_oos": n - cut,
            "oos_brier": round(float(((p_oos - y_oos) ** 2).mean()), 4),
            "oos_base": round(float(y_oos.mean()) * 100, 1),
            "oos_lift": round(_lift_on(p_oos, y_oos), 1)}


def _train_dir_model(dir_X, dir_y, dir_R, tf, champion=None, champion_ts=0.0):
    """مدل جهت‌یاب همیشه‌روشن + آمار معامله «dm» فقط از پیش‌بینی‌های برون‌نمونه‌ایِ Walk-Forward."""
    if len(dir_y) < 800:
        return None
    pairs = sorted(zip(dir_y, dir_X, dir_R), key=lambda t: t[0][0])
    # از v14 هر نمونه هر دو نمایش جهت‌دار را دارد؛ جهت‌یاب p(up) نمایش لانگ را مصرف می‌کند.
    X = np.array([p[1][0] if isinstance(p[1], (list, tuple)) and len(p[1]) == 2
                  and isinstance(p[1][0], (list, tuple, np.ndarray)) else p[1] for p in pairs], float)
    y = np.array([p[0][1] for p in pairs])
    R = [p[2] for p in pairs]
    ts_arr = [p[0][0] for p in pairs]
    meta, _p_all, oos_pc = _fit_model(
        X, y, champion=champion, ts=ts_arr, champion_ts=champion_ts,
        embargo_ms=engine.HORIZON[tf] * TF_MS[tf])
    if meta is None:
        return None
    oos_idx = [i for i in range(len(y)) if not np.isnan(oos_pc[i])]

    # آستانه‌های «تطبیقی» جهت‌یاب: به‌جای ۶۰/۴۰ِ ثابت، دهک‌های واقعیِ خروجیِ کالیبرهٔ مدل —
    # کالیبراسیونِ صادقانه خروجی را فشرده می‌کند (مثلاً ۳۷-۴۶٪) و آستانهٔ ثابت هرگز فعال نمی‌شد.
    po = np.array([oos_pc[i] for i in oos_idx], float)
    if len(po) >= 200:
        thr_lo, thr_hi = float(np.quantile(po, 0.15)), float(np.quantile(po, 0.85))
        meta["dm_thr"] = [round(thr_lo * 100, 1), round(thr_hi * 100, 1)]
        meta["p_q"] = {f"q{q}": round(float(np.quantile(po, q / 100)) * 100, 1)
                       for q in (10, 15, 40, 60, 85, 90)}
    else:
        thr_lo, thr_hi = 0.40, 0.60

    def _dm_stats(side):
        if side == "long":
            rows = [(R[i][0], R[i][2]) for i in oos_idx if oos_pc[i] >= thr_hi]
        else:
            rows = [(R[i][1], R[i][2]) for i in oos_idx if oos_pc[i] <= thr_lo]
        if len(rows) < 40:
            return None
        rs = [r for r, _ in rows]
        wins = sum(1 for r in rs if r > 0)
        return {"n": len(rows), "win": round(wins / len(rows) * 100, 1),
                "avg_r": round(float(np.mean(rs)), 3),
                "avg_risk": round(float(np.mean([rk for _, rk in rows])), 4)}

    meta["dm_long"] = _dm_stats("long")
    meta["dm_short"] = _dm_stats("short")
    return meta


def _train_logistic(events, tf, champion=None, champion_ts=0.0):
    if len(events) < 120:
        return None
    evs = sorted(events, key=lambda e: e["ts"])
    X = np.array([e["feats"] for e in evs], float)
    y = np.array([1.0 if e["r"] > 0 else 0.0 for e in evs])
    meta, _p_all, oos_pc = _fit_model(X, y, champion=champion,
                                      ts=[e["ts"] for e in evs], champion_ts=champion_ts,
                                      embargo_ms=40 * TF_MS[tf])
    if meta is None:
        return None
    honest_rows = [evs[i] for i in range(len(evs)) if not np.isnan(oos_pc[i])]
    wins = [e["r"] for e in honest_rows if e["r"] > 0]
    loss = [abs(e["r"]) for e in honest_rows if e["r"] <= 0]
    by_setup = {}
    for e in evs:
        d = by_setup.setdefault(e["setup"], [0, 0])
        d[0] += 1
        d[1] += 1 if e["r"] > 0 else 0
    meta.update({
        "feats": engine.FEATS,
        "avg_win_r": round(float(np.mean(wins)) if wins else 1.8, 3),
        "avg_loss_r": round(float(np.mean(loss)) if loss else 1.0, 3),
        "by_setup": {k: {"n": v[0], "win": round(v[1] / v[0] * 100, 1)} for k, v in by_setup.items()}})
    return meta


# ───────────────────── ساخت کامل ─────────────────────
def build(symbols, tfs=("1d", "4h", "1h", "15m"), bars=3000):
    global _table
    with _lock:
        if _state["building"]:
            return
        _state.update(building=True, progress="دریافت فاندینگ", done=0,
                      total=len(symbols) * len(tfs), error=None)
    try:
        # تاریخچهٔ فاندینگ تا ابتدای عمیق‌ترین پنجرهٔ آموزش صفحه‌بندی می‌شود (calib-F4)؛ قبلاً
        # فقط ۱۰۰۰ ردیفِ آخر بود و funding_dir در بیشترِ نمونه‌های 4h/1d صفر می‌ماند.
        fund_since = int(time.time() * 1000) - max(BARS.get(tf, bars) * TF_MS[tf] for tf in tfs)

        def _fz_one(s):
            try:
                return s, market.funding_z_map(s, since_ms=fund_since)
            except Exception:  # noqa: BLE001
                return s, []

        def _fr_one(s):
            try:
                return s, market.get_funding_history(s, since_ms=fund_since)
            except Exception:  # noqa: BLE001
                return s, []
        with ThreadPoolExecutor(max_workers=8) as pool:      # دریافتِ موازیِ فاندینگ (I/O شبکه)
            fz = dict(pool.map(_fz_one, symbols))
            fraw = dict(pool.map(_fr_one, symbols))
        table = {"built_at": time.time(), "cost_pct": COST_PCT, "version": CALIB_VERSION, "tfs": {}}
        universe_report, excluded_events = {}, {}
        final_events = {}                                 # tf -> رویدادهای پنجرهٔ منجمد
        zmaps = {}                                        # tf -> sym -> {ts: z}
        pending = {}                                      # tf -> ورودی‌های آموزش (آموزشِ همه در پایان، موازی)
        for tf in tfs:                                    # از بالا به پایین تا HTF آماده باشد
            tf_bars = BARS.get(tf, bars)
            hists = {}

            def _hist_one(sym, tf=tf, tf_bars=tf_bars):
                try:
                    return sym, market.get_history(sym, tf, tf_bars)
                except Exception:  # noqa: BLE001
                    return sym, None
            got = 0
            with ThreadPoolExecutor(max_workers=6) as pool:   # دریافتِ موازیِ تاریخچه (I/O شبکه/دیسک)
                for sym, kl in pool.map(_hist_one, symbols):
                    got += 1
                    with _lock:
                        _state["progress"] = f"داده {tf} ({got}/{len(symbols)})"
                    if kl:
                        hists[sym] = kl
            # ── جهانِ نقطه‌-در-زمان: برترین‌های هر ماه بر اساسِ حجمِ ۳۰ روزِ پیش از آن ──
            # (نه ۱۰۰ ارزِ پرحجمِ امروز — آن سوگیریِ بقا و نگاه به آینده در انتخاب بود)
            snaps = universe.snapshots_from_histories(hists, top_n=CALIB_UNIVERSE_N)
            universe_report[tf] = universe.summary(snaps)

            def _member(sym, ts_, _snaps=snaps):
                return universe.in_universe(_snaps, sym, ts_)

            # رتبهٔ قدرتِ نسبی و پهنای بازار عمداً خنثی‌اند (features.DISABLED، calib-F1): اجرا فقط
            # پنج ارز را می‌خواند و تعریفِ ۱۰۰ ارزی/کفِ جمعیتِ ۴۰ در اجرا همیشه ۰٫۵/۰ بود.
            rs_map = {}
            try:
                gk = market.get_history("PAXGUSDT", tf, tf_bars)  # طلای توکنی = پروکسی طلا
                gold_map = mom_norm_map(gk["t"], gk["c"])
            except Exception:  # noqa: BLE001
                gold_map = {}

            # ── P2: وضعیتِ کلِ بازار (دامیننس BTC، ETH/BTC) از سبدِ ثابتِ پنج‌ارزی — همان کدِ اجرا ──
            with _lock:
                _state["progress"] = f"وضعیت بازار {tf}"
            basket_h = {s: hists[s] for s in MARKET_BASKET if s in hists}
            for s in MARKET_BASKET:                       # سبد حتی اگر بیرونِ فهرستِ آموزش باشد
                if s not in basket_h:
                    _s, kl_ = _hist_one(s)
                    if kl_:
                        basket_h[s] = kl_
            mstate = market_state_maps(basket_h)
            breadth_map, dom_map, ethbtc_map = {}, mstate["dom"], mstate["ethbtc"]

            btc_events_zmap_src = None
            all_events = []
            dir_X_all, dir_y_all, dir_R_all = [], [], []
            zmaps[tf] = {}
            order = ["BTCUSDT"] + [s for s in symbols if s != "BTCUSDT"]
            for sym in order:
                if sym not in hists:
                    with _lock:
                        _state["done"] += 1
                    continue
                with _lock:
                    _state["progress"] = f"تحلیل {sym} {tf}"
                htf = HTF_OF[tf]
                htf_zmap = (zmaps.get(htf) or {}).get(sym)
                try:
                    evs, zmap, dx, dy, dr = extract_events(sym, hists[sym], tf, fz.get(sym, []),
                                                           rs_map, htf_zmap, btc_events_zmap_src,
                                                           gold_map,
                                                           breadth_map, dom_map, ethbtc_map,
                                                           funding_rows=fraw.get(sym))
                except Exception:  # noqa: BLE001
                    log.exc(f"extract_events {sym} {tf}")
                    evs, zmap, dx, dy, dr = [], {}, [], [], []
                zmaps[tf][sym] = zmap
                # فقط رویدادهایی که ارزشان **در همان لحظه** عضوِ جهان بوده
                kept = [e for e in evs if _member(sym, e["ts"])]
                excluded_events[tf] = excluded_events.get(tf, 0) + (len(evs) - len(kept))
                evs = kept
                keep_dense = [k for k, yy in enumerate(dy) if _member(sym, yy[0])]
                dx = [dx[k] for k in keep_dense]
                dy = [dy[k] for k in keep_dense]
                dr = [dr[k] for k in keep_dense]
                if sym == "BTCUSDT":
                    btc_events_zmap_src = zmap
                all_events.extend(evs)
                dir_X_all.extend(dx)
                dir_y_all.extend(dy)
                dir_R_all.extend(dr)
                with _lock:
                    _state["done"] += 1
            pending[tf] = (all_events, dir_X_all, dir_y_all, dir_R_all)

        # ── آموزشِ همهٔ تایم‌فریم‌ها موازی روی هسته‌ها (استخراج ترتیبی بود تا HTF آماده باشد؛ آموزش مستقل است) ──
        with _lock:
            _state["progress"] = f"آموزش موازی {len(pending)} تایم‌فریم روی هسته‌ها"

        # پیش‌ثبت اگر نیست، همین‌جا — پس از استخراج و **پیش از** هر برازش و داوری
        try:
            prereg, fresh = research.ensure_registered({
                tf: (int(min(e["ts"] for e in pending[tf][0])),
                     int(max(e["ts"] for e in pending[tf][0])))
                for tf in pending if pending[tf][0]})
            if fresh:
                with _lock:
                    _state["progress"] = "پیش‌ثبتِ فرضیه‌ها انجام شد — پنجرهٔ آزمون منجمد شد"
        except Exception:  # noqa: BLE001 — بدونِ پیش‌ثبت، هیچ ترکیبی مجاز نمی‌شود (fail-closed)
            log.exc("preregistration")
            prereg = None
        # پنجرهٔ منجمد فقط تا داوریِ یک‌بارمصرف معنا دارد. پس از آن (judged_<hash>.json) کنارگذاشتنش
        # فقط ۲۵-۲۸٪ از تاریخِ 4h/1d را برای همیشه از مدل‌ها می‌گرفت (calib-F15)؛ پیش‌ثبتِ تازه
        # پنجرهٔ تازه‌ای می‌سازد که تا داوری‌اش دوباره کنار گذاشته می‌شود.
        judged_done = False
        if prereg:
            try:
                judged_done = bool((research.last_judgement() or {}).get("judged"))
            except Exception:  # noqa: BLE001 — نامعلوم ⇒ محافظه‌کار: کنار بگذار
                log.exc("last_judgement")
        freeze = prereg if (prereg and not judged_done) else None

        def _train_tf(tf):
            all_events, dx, dy, dr = pending[tf]
            # ── پنجرهٔ آزمونِ منجمد (تا داوری): از **همهٔ** آموزش‌ها کنار گذاشته می‌شود ──
            # فقط داورِ یک‌بارمصرف (research.judge) این رویدادها را می‌بیند.
            if freeze:
                held = [e for e in all_events if research.in_final_window(tf, e["ts"], freeze)]
                out_of_train = _frozen_exclusion(tf, freeze)
                all_events = [e for e in all_events if not out_of_train(e["ts"])]
                keep = [k for k, yy in enumerate(dy) if not out_of_train(yy[0])]
                dx, dy, dr = [dx[k] for k in keep], [dy[k] for k in keep], [dr[k] for k in keep]
                final_events[tf] = held
            # سطل‌های پس‌گرد (lookup) هم فقط از رویدادهای آموزش — قبلاً پیش از کنارگذاشتن ساخته
            # می‌شدند و آمارِ پنجرهٔ منجمد را نشان می‌دادند (calib-F9)
            cells = _event_cells(all_events)
            # ── سلامتِ ویژگی‌ها پیش از آموزش ──
            # ویژگیِ با واریانسِ صفر چیزی برای یادگرفتن ندارد و فقط بُعد اضافه می‌کند؛
            # dxy_dir دقیقاً همین بود و کسی متوجه نشد. حالا در جدول ثبت می‌شود.
            feat_health = features.health([e["feats"] for e in all_events],
                                          expected=features.expected_constant(tf))
            dense_health = features.health([row[0] for row in dx],
                                           expected=features.expected_constant(tf, dense=True))
            dead_any = sorted(set(feat_health["dead"]) | set(dense_health["dead"]))
            if dead_any:
                with _lock:
                    _state["progress"] = f"⚠️ {tf}: ویژگیِ بی‌واریانس {', '.join(dead_any)}"
            # ویژگیِ «۰ = نبود» با پوششِ ناچیز در آموزش **و** اجرا ۰ می‌شود (calib-F4): مدل از چند
            # ردیفِ پراکنده وزنِ پرنویز یاد نمی‌گیرد و در اجرا ورودیِ خارج از توزیع نمی‌بیند.
            zero_ev, zero_dn = features.live_zeroed(feat_health), features.live_zeroed(dense_health)
            if zero_ev:
                all_events = [dict(e, feats=features.zero_features(e["feats"], zero_ev))
                              for e in all_events]
            if zero_dn:
                dx = [(features.zero_features(lf, zero_dn), features.zero_features(sf, zero_dn))
                      for lf, sf in dx]
            model = _train_logistic(all_events, tf)
            edge_model = _fit_edge_model(all_events, tf, COST_PCT)
            policy_model = _fit_policy_model(all_events, tf, COST_PCT)
            dir_model = _train_dir_model(dx, dy, dr, tf)
            action_model = _fit_action_policy(dx, dy, dr, tf, COST_PCT)
            return tf, {"cells": cells, "events": len(all_events),
                        "held_out_final_events": len(final_events.get(tf) or []),
                        "feature_health": feat_health,
                        "feature_health_dense": dense_health,
                        "live_zeroed": {"events": zero_ev, "dense": zero_dn},
                        "model": model, "edge_model": edge_model,
                        "policy_model": policy_model, "dir_model": dir_model,
                        "action_model": action_model}
        with ThreadPoolExecutor(max_workers=len(pending) or 1) as pool:
            for tf, blob in pool.map(_train_tf, list(pending)):
                table["tfs"][tf] = blob
        table["trust_history"] = _update_trust_history(load() or {}, table)
        table["spans"] = {tf: [int(min(e["ts"] for e in pending[tf][0])),
                               int(max(e["ts"] for e in pending[tf][0]))]
                          for tf in pending if pending[tf][0]}
        table["frozen_window"] = {"excluded": bool(freeze), "judged": judged_done}
        if freeze and final_events:
            try:
                judged = research.last_judgement() or {}
                if not judged.get("judged"):
                    table["judgement"] = research.judge(final_events)
            except research.AlreadyJudged as e:
                table["judgement_skipped"] = str(e)       # یک‌بارمصرف است — درست است که رد شود
            except research.PreregistrationError as e:
                table["judgement_error"] = str(e)
        table["universe"] = {"per_tf": universe_report,
                             "excluded_events_outside_universe": excluded_events,
                             "top_n": CALIB_UNIVERSE_N}
        os.makedirs(os.path.dirname(CALIB_PATH), exist_ok=True)
        tmp = CALIB_PATH + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(table, f)
        os.replace(tmp, CALIB_PATH)
        with _lock:
            _table = table
            _state.update(building=False, progress="کامل شد")
    except Exception as e:  # noqa: BLE001
        with _lock:
            _state.update(building=False, error=str(e))
    finally:
        _close_pool()                                 # نودهای یادگیری تا بازآموزیِ بعدی آزاد می‌شوند (حافظه)


def load():
    global _table
    with _lock:
        if _table is not None:
            return _table
        if os.path.exists(CALIB_PATH):
            with open(CALIB_PATH, "r", encoding="utf-8") as f:
                _table = json.load(f)
            return _table
    return None


def status():
    with _lock:
        st = dict(_state)
    t = load()
    if t:
        valid_version = t.get("version") == CALIB_VERSION
        st["version"] = t.get("version")
        st["required_version"] = CALIB_VERSION
        st["stale"] = not valid_version or time.time() - t.get("built_at", 0) > REBUILD_SEC
        st["built_at"] = t.get("built_at")
        st["age_hours"] = round((time.time() - t["built_at"]) / 3600, 1)
        st["events"] = {tf: d.get("events", 0) for tf, d in t.get("tfs", {}).items()}
        st["breakeven_win_rate"] = round(bracket.breakeven_win_rate() * 100, 1)
        st["feature_health"] = {
            tf: {**{k: (d.get("feature_health") or {}).get(k)
                    for k in ("ok", "dead", "low_coverage", "expected_constant", "n", "n_feat")},
                 "dense_dead": (d.get("feature_health_dense") or {}).get("dead"),
                 "live_zeroed": d.get("live_zeroed")}
            for tf, d in t.get("tfs", {}).items() if d.get("feature_health")
        }
        st["models"] = {}

        def _eff(tf_, key, ok):
            # پرچمِ «معتبر» همان چیزی است که predict به کار می‌برد: این ساخت + هیسترزیسِ دو ساخت؛
            # قبلاً این‌جا فقط تک‌ساخت بود و tf_trust «معتبر» نشان می‌داد در حالی که predict رد می‌کرد
            return bool(ok and trusted_with_hysteresis(tf_, key, t))
        for tf, d in t.get("tfs", {}).items():
            m = d.get("model")
            if m:
                st["models"][tf] = {k: m.get(k) for k in
                                    ("n_train", "n_oos", "oos_brier", "oos_brier_base",
                                     "oos_brier_skill", "oos_base", "oos_lift", "by_setup")}
                st["models"][tf]["kind"] = m.get("kind", "logit")
                ok = valid_version and _model_trusted(m, MIN_SETUP_LIFT)
                st["models"][tf]["trusted_this_build"] = bool(ok)
                st["models"][tf]["trusted"] = _eff(tf, "model", ok)
                if m.get("kind") == "ensemble":
                    st["models"][tf]["members"] = m.get("member_kinds", [])
            dm = d.get("dir_model")
            if dm:
                st["models"].setdefault(tf, {})
                st["models"][tf]["dir_lift"] = dm["oos_lift"]
                st["models"][tf]["dir_n"] = dm["n_train"] + dm["n_oos"]
                ok = valid_version and _model_trusted(dm, MIN_DIR_LIFT)
                st["models"][tf]["dir_trusted_this_build"] = bool(ok)
                st["models"][tf]["dir_trusted"] = _eff(tf, "dir_model", ok)
            em = d.get("edge_model")
            if em:
                st["models"].setdefault(tf, {})
                ok = valid_version and _edge_model_trusted(em)
                st["models"][tf].update({
                    "edge_trusted": _eff(tf, "edge_model", ok),
                    "edge_trusted_this_build": bool(ok),
                    "edge_n": em.get("n_oos"),
                    "edge_rank_ic": em.get("oos_rank_ic"),
                    "edge_lift_r": em.get("oos_lift_r"),
                    "edge_mae_skill": em.get("oos_mae_skill"),
                })
            pm = d.get("policy_model")
            if pm:
                st["models"].setdefault(tf, {})
                pst = pm.get("test") or {}
                ok = valid_version and _policy_model_trusted(pm)
                st["models"][tf].update({
                    "policy_trusted": _eff(tf, "policy_model", ok),
                    "policy_trusted_this_build": bool(ok),
                    "policy_n": pst.get("n"),
                    "policy_avg_net_r": pst.get("avg_net_r"),
                    "policy_lcb_net_r": pst.get("lcb_net_r"),
                    "policy_profit_factor": pst.get("profit_factor"),
                    "policy_uplift_r": pst.get("uplift_r"),
                    "policy_quantile": pm.get("select_quantile"),
                    "policy_prob_kind": pm.get("prob_kind"),
                    "policy_edge_kind": pm.get("edge_kind"),
                    "policy_halves_avg_r": pm.get("test_halves_avg_r"),
                })
            am = d.get("action_model")
            if am:
                st["models"].setdefault(tf, {})
                ast = am.get("test") or {}
                ok = valid_version and _action_policy_trusted(am)
                st["models"][tf].update({
                    "action_trusted": _eff(tf, "action_model", ok),
                    "action_trusted_this_build": bool(ok),
                    "action_n": ast.get("n"),
                    "action_avg_net_r": ast.get("avg_net_r"),
                    "action_lcb_net_r": ast.get("lcb_net_r"),
                    "action_profit_factor": ast.get("profit_factor"),
                    "action_uplift_r": ast.get("uplift_r"),
                    "action_quantile": am.get("select_quantile"),
                    "action_kind": am.get("kind"),
                    "action_halves_avg_r": am.get("test_halves_avg_r"),
                })
    return st


TRUST_KEYS = {
    "model": lambda m: _model_trusted(m, MIN_SETUP_LIFT),
    "dir_model": lambda m: _model_trusted(m, MIN_DIR_LIFT),
    "edge_model": lambda m: _edge_model_trusted(m),
    "policy_model": lambda m: _policy_model_trusted(m),
    "action_model": lambda m: _action_policy_trusted(m),
}


def _update_trust_history(old_table, new_table):
    """تاریخچهٔ اعتمادِ هر مدل در ساخت‌های پیاپی (حداکثر ۱۰ تا).

    پرچمِ اعتماد قبلاً از **یک** پنجرهٔ ۲۰٪ آخر در همان ساخت محاسبه می‌شد؛ پس هر
    بازسازی یک پرتابِ سکهٔ تازه بود (lift روی 4h بینِ دو ساخت از +۱۴٫۷ به −۰٫۶ رفت).
    """
    hist = dict((old_table or {}).get("trust_history") or {})
    if (old_table or {}).get("version") != new_table.get("version"):
        hist = {}                                  # نسخهٔ تازه = تاریخچهٔ تازه
    for tf, blob in (new_table.get("tfs") or {}).items():
        row = dict(hist.get(tf) or {})
        for key, fn in TRUST_KEYS.items():
            try:
                ok = bool(blob.get(key) and fn(blob.get(key)))
            except Exception:  # noqa: BLE001
                ok = False
            row[key] = (list(row.get(key) or []) + [ok])[-10:]
        hist[tf] = row
    return hist


def trusted_with_hysteresis(tf, key, table=None):
    """اعتماد فقط اگر **دو ساختِ پیاپیِ آخر** هر دو موفق بوده باشند؛ یک شکست کافی است تا خاموش شود."""
    t = table if table is not None else load()
    runs = (((t or {}).get("trust_history") or {}).get(tf) or {}).get(key) or []
    return len(runs) >= TRUST_STREAK and all(runs[-TRUST_STREAK:])


def is_stale():
    t = load()
    if t is None or time.time() - t.get("built_at", 0) > REBUILD_SEC:
        return True
    if t.get("version") != CALIB_VERSION:                              # ویژگی‌ها عوض شده‌اند
        return True
    return not any(d.get("policy_model") for d in t.get("tfs", {}).values())  # نسخهٔ ناقص بدون سیاست نهایی


# ───────────────────── پیش‌بینی زنده ─────────────────────
def _model_nfeat(m):
    return m.get("n_feat", len(m.get("mu", [])))


def _live_feats(tf_blob, feats, which):
    """همان ویژگی‌هایی که آموزش صفر کرد (پوششِ ناچیز) در اجرا هم صفر — ``which``: events یا dense."""
    names = ((tf_blob or {}).get("live_zeroed") or {}).get(which) or []
    return features.zero_features(feats, names) if names else feats


def _score_model(m, feats):
    """احتمالِ کالیبره از مدل (تک‌مدل یا ترکیبی) + کالیبراسیونِ پلَت."""
    x = np.array(feats, float).reshape(1, -1)
    raw = float(np.clip(_score_all(m, x)[0], 1e-6, 1 - 1e-6))
    a, b = m.get("platt", [1.0, 0.0])
    return float(_sigmoid(a * _logit(np.array([raw]))[0] + b))


def _score_model_details(m, feats):
    """پیش‌بینی احتمالی همراه با بازهٔ محافظه‌کارانه؛ اختلاف اعضای ensemble هم جزو عدم‌قطعیت است."""
    p = _score_model(m, feats)
    member_ps = []
    if m.get("kind") == "ensemble":
        x = np.asarray(feats, float).reshape(1, -1)
        a, b = m.get("platt", [1.0, 0.0])
        for mem in m.get("members", []):
            model = _load_member(mem["kind"], mem["blob"])
            raw = float(np.clip(_member_pred(mem["kind"], model, x)[0], 1e-6, 1 - 1e-6))
            member_ps.append(float(_sigmoid(a * _logit(np.array([raw]))[0] + b)))
    member_sd = float(np.std(member_ps)) if len(member_ps) >= 2 else 0.0
    n_oos = max(int(m.get("n_oos") or 1), 1)
    statistical_se = math.sqrt(max(float(m.get("oos_brier") or 0.25), 1e-6) / n_oos)
    # ۹۰٪ lower/upper confidence bound؛ اختلاف اعضا و خطای آزمون هر دو لحاظ می‌شوند.
    uncertainty = min(1.64 * math.sqrt(member_sd ** 2 + statistical_se ** 2), 0.25)
    return {
        "p": p,
        "p_low": max(0.01, p - uncertainty),
        "p_high": min(0.99, p + uncertainty),
        "p_uncertainty": uncertainty,
        "member_sd": member_sd,
    }


def _model_trusted(m, min_lift):
    """مدل فقط وقتی مجاز است که هم ranking و هم probability از baseline بهتر باشند."""
    if not m or int(m.get("n_oos") or 0) < 60:
        return False
    skill = m.get("oos_brier_skill")
    if skill is None:
        base = float(m.get("oos_base") or 0) / 100.0
        base_brier = base * (1.0 - base)
        brier = float(m.get("oos_brier") or 1.0)
        skill = 1.0 - brier / max(base_brier, 1e-9)
    return float(m.get("oos_lift") or -99) >= min_lift and float(skill) >= MIN_BRIER_SKILL


def predict_dir(tf, feats):
    """احتمال بالارفتن تا افق H از مدل جهت‌یاب همیشه‌روشن (فقط اگر در آزمون OOS لیفت داشته باشد)."""
    t = load()
    if not t or t.get("version") != CALIB_VERSION or tf not in t.get("tfs", {}):
        return None
    dm = t["tfs"][tf].get("dir_model")
    if not _model_trusted(dm, MIN_DIR_LIFT) or _model_nfeat(dm) != len(feats):
        return None
    if not trusted_with_hysteresis(tf, "dir_model", t):
        return None
    feats = _live_feats(t["tfs"][tf], feats, "dense")
    pd = _score_model_details(dm, feats)
    p = pd["p"]
    return {"p_up": round(p * 100, 1),
            "p_up_low": round(pd["p_low"] * 100, 1),
            "p_up_high": round(pd["p_high"] * 100, 1),
            "p_uncertainty": round(pd["p_uncertainty"] * 100, 1),
            "oos_lift": dm["oos_lift"],
            "n": dm["n_train"] + dm["n_oos"], "oos_base": dm["oos_base"],
            "dm_long": dm.get("dm_long"), "dm_short": dm.get("dm_short"),
            "p_q": dm.get("p_q"), "dm_thr": dm.get("dm_thr"),
            "cost": t.get("cost_pct", COST_PCT)}


def predict_action(tf, long_feats, short_feats, risk_pct, cost=None, regime=None):
    """انتخاب مستقیم long/short/no-trade فقط وقتی سیاست متراکم test مستقل را پاس کرده باشد."""
    t = load()
    if not t or t.get("version") != CALIB_VERSION or tf not in t.get("tfs", {}):
        return None
    m = t["tfs"][tf].get("action_model")
    if not _action_policy_trusted(m) or _model_nfeat(m) != len(long_feats) or len(short_feats) != len(long_feats):
        return None
    if not trusted_with_hysteresis(tf, "action_model", t):
        return None
    actual_cost = cost if cost is not None else t.get("cost_pct", COST_PCT)
    long_feats = _live_feats(t["tfs"][tf], long_feats, "dense")
    short_feats = _live_feats(t["tfs"][tf], short_feats, "dense")
    return _score_action_policy(m, long_feats, short_feats, risk_pct, actual_cost)


def _policy_untrusted_payload(pm, ps, risk_pct):
    """سیاستی که دادگاهِ OOS را نباخته — فقط تشخیص، بدونِ هیچ عددِ قابلِ‌اتکا.

    عمداً هیچ امتیاز/EV/edge از سیاستِ مردود پخش نمی‌شود: وقتی شیبِ کالیبراسیون صفر
    می‌شود، edge_sd به ۱e-۶ می‌رسد و policy_score به ±۱۰⁵ می‌پرد که فقط تابعِ ردهٔ
    هزینهٔ نماد است — این عدد قبلاً به مرتب‌سازیِ UI و انتخابِ مقطعی نشت می‌کرد.
    """
    tst = pm.get("test") or {}
    return {
        "p_win": None, "p_win_low": None, "p_win_high": None, "p_uncertainty": None,
        "n": int(tst.get("n") or 0),
        "avg_r": None, "ev_pct": None, "ev_lcb_pct": None,
        "edge_r": None, "edge_lcb_r": None, "edge_uncertainty_r": None,
        "policy_score": None, "policy_margin": None,
        "policy_test": tst,
        "feature_zmax": ps.get("feature_zmax"),
        "regime_stats": ps.get("regime_stats"),
        "regime_veto": bool(ps.get("regime_veto")),
        "regime_ok": bool(ps.get("regime_ok")),
        "reliability": "ردِ دادگاه",
        "base": pm.get("base_win"),
        "oos_lift": round(float(tst.get("uplift_r") or 0) * 100, 1),
        "oos_brier": None,
        "source": "policy",
        "policy_trusted": False,
        "policy_pass": False,
        "policy_court_failed": True,
        "edge_trusted": False,
        "edge_rank_ic": None, "edge_lift_r": None,
        "authority": None,
    }


def predict(tf, feats, risk_pct, legacy=None, cost=None, regime=None):
    """یک مرجع تصمیم: سیاست معتبر → مدل ستاپ معتبر → در غیر این صورت فقط تشخیص.

    باگ قدیمی: وقتی سیاست در دادگاه مردود بود، به سطل/lookup سقوط می‌کرد و UI
    همان آمار را مثل احتمال معامله نشان می‌داد («خرید A» + وین‌ریت خوب + مسدود).
    """
    t = load()
    if not t or t.get("version") != CALIB_VERSION or tf not in t.get("tfs", {}):
        return None
    d = t["tfs"][tf]
    actual_cost = cost if cost is not None else t.get("cost_pct", COST_PCT)
    feats = _live_feats(d, feats, "events")
    pm = d.get("policy_model")
    policy_diag = None
    if pm and _model_nfeat(pm) == len(feats):
        ps = _score_policy(pm, feats, risk_pct, actual_cost, regime)
        if _policy_model_trusted(pm) and trusted_with_hysteresis(tf, "policy_model", t):
            tst = pm.get("test") or {}
            edge_r = float(ps["edge_r"])
            ev_pct = edge_r * float(risk_pct)
            ev_lcb_pct = float(ps["edge_lcb_r"]) * float(risk_pct)
            reliability = (
                "خوب" if int(tst.get("n") or 0) >= 80 and float(tst.get("lcb_net_r") or -99) >= 0
                else "متوسط"
            )
            return {
                **ps,
                "n": int(tst.get("n") or 0),
                "avg_r": round(edge_r, 3),
                "ev_pct": round(ev_pct, 3),
                "ev_lcb_pct": round(ev_lcb_pct, 3),
                "reliability": reliability,
                "base": pm.get("base_win"),
                "oos_lift": round(float(tst.get("uplift_r") or 0) * 100, 1),
                "oos_brier": None,
                "source": "policy",
                "policy_trusted": True,
                "edge_trusted": True,
                "edge_rank_ic": (pm.get("edge_selection") or {}).get(pm.get("edge_kind"), {}).get("rank_ic"),
                "edge_lift_r": (pm.get("edge_selection") or {}).get(pm.get("edge_kind"), {}).get("lift_r"),
                "authority": "policy",
            }
        policy_diag = _policy_untrusted_payload(pm, ps, risk_pct)

    m = d.get("model")
    # دروازه علمی: مدلی که در آزمون برون‌نمونه‌ای لیفت معنادار نشان نداده، حق صدور احتمال ندارد
    # — و مثلِ بقیهٔ مدل‌ها فقط پس از دو ساختِ پیاپیِ موفق (calib-F12؛ یک ساختِ خوش‌شانس کافی نیست)
    if m and not (_model_trusted(m, MIN_SETUP_LIFT) and trusted_with_hysteresis(tf, "model", t)):
        m = None
    if m and _model_nfeat(m) != len(feats):
        m = None                                   # مدل قدیمی با ویژگی‌های جدید ناسازگار است
    if m:
        pd = _score_model_details(m, feats)
        p, p_low = pd["p"], pd["p_low"]
        shrink = m["n_train"] / (m["n_train"] + 150)
        ev_r = p * m["avg_win_r"] - (1 - p) * m["avg_loss_r"]
        ev_lcb_r = p_low * m["avg_win_r"] - (1 - p_low) * m["avg_loss_r"]
        ev_pct = ev_r * shrink * risk_pct - actual_cost
        ev_lcb_pct = ev_lcb_r * shrink * risk_pct - actual_cost
        lift, n = m["oos_lift"], m["n_oos"]
        reliability = "خوب" if (lift >= 8 and n >= 200) else "متوسط"
        out = {
            "p_win": round(p * 100, 1),
            "p_win_low": round(p_low * 100, 1),
            "p_win_high": round(pd["p_high"] * 100, 1),
            "p_uncertainty": round(pd["p_uncertainty"] * 100, 1),
            "n": n, "avg_r": round(ev_r, 3),
            "ev_pct": round(ev_pct, 3), "ev_lcb_pct": round(ev_lcb_pct, 3),
            "reliability": reliability,
            "base": m.get("oos_base"),                 # نرخِ پایهٔ برد — مرجعِ «کفِ نسبی»
            "oos_lift": lift, "oos_brier": m["oos_brier"], "source": "model",
            "edge_trusted": False,
            # مدل ستاپ هرگز مرجعِ ورود نیست (فاز ۰): فقط احتمالِ تشخیصی می‌دهد.
            "policy_trusted": False,
            "policy_pass": False,
            "authority": None,
        }
        if policy_diag:
            out["policy_test"] = policy_diag.get("policy_test")
            out["policy_court_failed"] = True
            out["policy_avg_net_r"] = (policy_diag.get("policy_test") or {}).get("avg_net_r")
        em = d.get("edge_model")
        if (_edge_model_trusted(em) and trusted_with_hysteresis(tf, "edge_model", t)
                and _model_nfeat(em) == len(feats)):
            out["edge_trusted"] = True
            out.update(_score_edge(em, feats, risk_pct, actual_cost, regime))
        return out

    # سیاست مردود و مدل ستاپ هم نیست → همان تشخیص سیاست، بدون آمار سطلِ گمراه‌کننده
    if policy_diag:
        return policy_diag

    if legacy:
        out = lookup(tf, legacy["direction"], legacy["z"], legacy["votes"],
                     legacy["trending"], legacy["btc_align"])
        if out:
            out["diagnostic_only"] = True
            out["policy_trusted"] = False
            out["authority"] = None
            if d.get("model"):
                out["model_rejected"] = True      # مدل وجود داشت ولی در آزمون OOS رد شد
            if policy_diag:
                out["policy_test"] = policy_diag.get("policy_test")
                out["policy_court_failed"] = True
        return out
    return None


def lookup(tf, direction, z, votes, trending, btc_align):
    """پس‌گرد سطلی (وقتی مدل موجود نیست)."""
    t = load()
    if not t or tf not in t.get("tfs", {}):
        return None
    cells = t["tfs"][tf]["cells"]
    d = 1 if direction == "long" else -1
    for level, key in enumerate(keys_for(d, z, votes, trending, btc_align)):
        cell = cells.get(key)
        if cell and cell[0] >= 25:
            n, wins, sum_r, sum_risk = cell
            p = (wins + 2) / (n + 4)
            avg_r = sum_r / n
            avg_risk = sum_risk / max(n, 1)
            shrink = n / (n + 15)
            ev_pct = avg_r * shrink * avg_risk - t.get("cost_pct", COST_PCT)
            reliability = "خوب" if (n >= 80 and level == 0) else "متوسط" if n >= 40 else "کم"
            if level >= 2:
                reliability = "کم"
            return {"p_win": round(p * 100, 1), "avg_r": round(avg_r, 3), "n": n,
                    "level": level, "reliability": reliability, "ev_pct": round(ev_pct, 3),
                    "source": "bins"}
    return None
