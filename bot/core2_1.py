# -*- coding: utf-8 -*-
"""هستهٔ v2.1 — همان ویژگی‌ها و برچسبِ v2، با مشخصاتِ مدلِ تازه، روی دادهٔ دست‌نخورده
(``research/prereg_core2_1.json``؛ الزام‌آور).

تفاوت با v2 (bot/core2.py، که دست نخورده می‌ماند): رگرسیونِ l2 (میانگین را پیش‌بینی می‌کند)، تعدادِ دورِ
ثابت و بدونِ توقفِ زودهنگام، و قاعدهٔ تصمیمِ «لانگ اگر E_L > 0 و E_L > E_S». پنجره‌ها: اعتبارسنجی
۲۰۲۵-۰۷ تا ۲۰۲۵-۱۲ و holdoutِ یک‌بارمصرف ۲۰۲۶-۰۱ تا ۲۰۲۶-۰۸.

هر چیزِ دیگر (برش‌های بی‌نشتِ walk-forward، ساختِ داده، بازپخش، بوت‌استرپ، Holm) از core2 می‌آید.
اختیار: هرگز tradeable را روشن نمی‌کند، به gates.json نمی‌نویسد و autotrader آن را نمی‌خواند.

نگهبانِ holdout: (۱) گزارشِ کاملِ اعتبارسنجی با sha256ی که در ``core2_1_validation_pin.json`` سنجاق شده؛
(۲) هشِ کد و پیش‌ثبت و core_feats همان‌هایی که سنجاق شد؛ (۳) رکوردِ شروعِ append-only در
``core2_holdout_log.jsonl`` که پیش از هر محاسبه در گیت commit شده؛ (۴) هنوز اجرا نشده.
"""
import glob
import json
import math
import os
import subprocess
import time
from datetime import datetime, timezone

import numpy as np

import core2 as c2
import core_feats as cf
import decision
import engine
import paths
from watchlist import SYMBOLS

PREREG_PATH = paths.data("research", "prereg_core2_1.json")
RESEARCH_DIR = paths.data("research")
OUT_DIR = paths.data("hist_research", "core2_1")
REPORT_PREFIX = "core2_1_validation_"
PIN_PATH = paths.data("research", "core2_1_validation_pin.json")
HOLDOUT_LOG = paths.data("research", "core2_holdout_log.jsonl")

TFS = c2.TFS
BAR_MS = c2.BAR_MS
COST = c2.COST
MAKER_COST = c2.MAKER_COST

WINDOWS = {"validation": ("2025-07-01", "2025-12-31"), "holdout": ("2026-01-01", "2026-08-31")}
ROUNDS = {"5m": 300, "15m": 300, "1h": 300, "4h": 300, "1d": 150}
MARGINS = (0.0, 0.05, 0.10)          # 0.0 = قاعدهٔ اصلی (E_L > 0 و E_L > E_S)؛ بقیه فقط حساسیت
PRIMARY = 0.0
N_MIN = {"1d": 20}
N_MIN_DEFAULT = 60
HOLD_N_MIN = {"1d": 10}
HOLD_N_MIN_DEFAULT = 30


def window_ms(name):
    a, b = WINDOWS[name]
    return c2._ms(a), c2._ms(b) + c2.DAY_MS


def n_min(tf):
    return N_MIN.get(tf, N_MIN_DEFAULT)


def check_prereg(path=None):
    """پنجره‌ها و پارامترهای کلیدیِ این فایل باید عیناً همان پیش‌ثبت باشند (وگرنه خطا)."""
    with open(path or PREREG_PATH, "r", encoding="utf-8") as f:
        pr = json.load(f)
    if list(pr["windows"]["validation"]) != list(WINDOWS["validation"]) or \
            list(pr["windows"]["holdout_one_shot"]) != list(WINDOWS["holdout"]):
        raise ValueError("پنجره‌ها با پیش‌ثبت یکی نیست")
    m = pr["model"]
    for s in ("objective regression", "learning_rate 0.02", "num_leaves 15", "max_depth 5", "max(500, n/400)",
              "200 at 1h", "100 at 4h", "feature_fraction 0.5", "bagging_fraction 0.7", "bagging_freq 1",
              "lambda_l2 50", "max_bin 63", "seed 42", "FIXED 300 boosting rounds", "no early stopping",
              "num_leaves 7, min_data_in_leaf 60, 150 rounds", "every second row at 5m"):
        if s not in m:
            raise ValueError(f"پارامترِ «{s}» در پیش‌ثبت پیدا نشد")
    if "month start - 81 bars" not in pr["walk_forward"] or "E_L > 0" not in pr["decision_rule_fixed"]:
        raise ValueError("walk-forward یا قاعدهٔ تصمیم با پیش‌ثبت یکی نیست")
    return pr


# ───────────────────────── مدل ─────────────────────────
def lgb_params(tf, n_train, threads=c2.NUM_THREADS):
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


def fit_pair(X, yL, yS, t, sym, tf, names, threads=c2.NUM_THREADS):
    """M_long و M_short با دورِ ثابت روی همهٔ سطرهای آموزشِ آن ماه ⇒ ``(predict(X) -> (E_L, E_S), meta)``."""
    import lightgbm as lgb
    params = lgb_params(tf, len(t), threads)
    w = c2.coin_weights(sym)
    models, fit_s = {}, {}
    for side, y in (("long", yL), ("short", yS)):
        t0 = time.time()
        ds = lgb.Dataset(X, y, weight=w, feature_name=names, free_raw_data=True)
        models[side] = lgb.train(params, ds, num_boost_round=ROUNDS[tf])
        fit_s[side] = round(time.time() - t0, 2)
    mL, mS = models["long"], models["short"]

    def predict(Xte):
        return (mL.predict(Xte, num_threads=int(threads)).astype(np.float32),
                mS.predict(Xte, num_threads=int(threads)).astype(np.float32))

    meta = {"min_data_in_leaf": params["min_data_in_leaf"], "num_leaves": params["num_leaves"],
            "rounds": ROUNDS[tf], "fit_s_long": fit_s["long"], "fit_s_short": fit_s["short"],
            "gain_long": mL.feature_importance("gain").astype(float).tolist(),
            "gain_short": mS.feature_importance("gain").astype(float).tolist()}
    return predict, meta


# ───────────────────────── تصمیم و ارزیابی ─────────────────────────
def decide(E_L, E_S, margin=PRIMARY):
    """۱ لانگ، −۱ شورت، ۰ صبر (NaN ⇒ صبر). ``margin=0``: لانگ اگر E_L > 0 و E_L > E_S (شورت قرینه).
    ``margin>0`` (فقط حساسیت): E ≥ margin و فاصلهٔ دو طرف ≥ margin."""
    EL = np.asarray(E_L, dtype=np.float64)
    ES = np.asarray(E_S, dtype=np.float64)
    if margin <= 0:
        L, S = (EL > 0) & (EL > ES), (ES > 0) & (ES > EL)
    else:
        L = (EL >= margin) & (EL - ES >= margin)
        S = (ES >= margin) & (ES - EL >= margin)
    side = np.zeros(len(EL), dtype=np.int8)
    side[L & ~S] = 1
    side[S & ~L] = -1
    return side


def evaluate(tf, oos, w0, w1, fut, symbols=SYMBOLS, log=None):
    """معامله‌های v2.1 (همهٔ حاشیه‌ها) و قاعدهٔ فعلی روی همان کندل‌های فیوچرز و همان پنجره."""
    v21 = {m: {} for m in MARGINS}
    rule = {}
    replay_equal = True
    for k, s in enumerate(symbols):
        F = fut[s]
        t = F["t"]
        s0 = int(np.searchsorted(t, w0))
        cut = max(0, s0 - decision.WARMUP_BARS)
        sub = {kk: np.asarray(v)[cut:] for kk, v in F.items() if kk in ("t", "o", "h", "l", "c", "v")}
        st = sub["t"]
        rep = decision.replay(sub, start_ts=w0, cost_pct=COST)
        rule[s] = [x for x in rep["trades"] if x["t"] < w1]
        a14 = engine.atr(sub["h"], sub["l"], sub["c"], 14)
        rs = np.asarray(rep["side"]).copy()
        rs[st >= w1] = 0
        chk = c2.replay_sides(sub, rs, rep["first"], COST, a14)
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
            v21[mg][s] = c2.replay_sides(sub, side, rep["first"], COST, a14)
        if log:
            log(f"  {tf} {s}: v2.1 {len(v21[PRIMARY][s])} معامله، قاعده {len(rule[s])}")
    return v21, rule, replay_equal


def adopt_checks(tf, res, holm_reject):
    """چهار شرطِ پذیرشِ پیش‌ثبت روی پنجرهٔ اعتبارسنجی."""
    v, rule, bt = res["v21"], res["rule"], res["bootstrap"]
    m = v["mean"]
    mr = rule["mean"] if rule["mean"] is not None else 0.0
    c1 = (m is not None and m >= mr + c2.ADOPT_EDGE_R and bt["diff_ci95"] is not None and bt["diff_ci95"][0] > 0)
    c2_ = (m is not None and m > 0 and v["n"] >= n_min(tf) and bool(holm_reject))
    c3 = v["coins_positive"] >= c2.MIN_POS_COINS
    c4 = v["max_coin_share"] is not None and v["max_coin_share"] <= c2.MAX_COIN_SHARE
    return {"edge_vs_rule_and_diff_ci_above_0": bool(c1), "mean_pos_n_and_holm": bool(c2_),
            "coins_positive_ge_3": bool(c3), "no_coin_over_50pct": bool(c4), "adopted": bool(c1 and c2_ and c3 and c4)}


def result_path(tf, window="validation", out_dir=None):
    return os.path.join(out_dir or OUT_DIR, f"{tf}_{'val' if window == 'validation' else 'hold'}_result.json")


def save_oos(tf, oos, window="validation", out_dir=None):
    d = out_dir or OUT_DIR
    os.makedirs(d, exist_ok=True)
    path = os.path.join(d, f"{tf}_{'val' if window == 'validation' else 'hold'}_oos.npz")
    np.savez_compressed(path, sym=np.asarray(SYMBOLS)[oos["sym"].astype(int)] if len(oos["sym"]) else np.zeros(0, "U7"),
                        t=oos["t"].astype(np.int64), E_L=oos["E_L"].astype(np.float32),
                        E_S=oos["E_S"].astype(np.float32), month=oos["month"].astype(np.int32),
                        yL=oos["yL"].astype(np.float32), yS=oos["yS"].astype(np.float32))
    return path


def run_window(tf, window="validation", threads=c2.NUM_THREADS, log=None, fit_fn=None, guard=None):
    """یک تایم‌فریم روی یک پنجره. ``holdout`` فقط وقتی ``guard`` (نتیجهٔ ``holdout_guard``) اجازه داده باشد."""
    if window not in ("validation", "holdout"):
        raise ValueError(window)
    if window == "holdout" and not (guard and guard[0]):
        raise PermissionError((guard or (False, "holdout بدونِ نگهبان اجرا نمی‌شود"))[1])
    wall = time.time()
    w0, w1 = window_ms(window)
    bar = BAR_MS[tf]
    fut_end = w1 + c2.LABEL_TAIL_BARS * bar
    ds = c2.build_dataset(tf, w1, fut_end, log=log)
    months = c2.month_starts(w0, w1)
    t_fit = time.time()
    oos, meta = c2.walk_forward(ds, months, fit_fn=fit_fn or fit_pair, threads=threads, log=log, window_end=w1)
    fit_s = time.time() - t_fit
    fut = {s: c2.load_bars("um", s, tf, fut_end) for s in SYMBOLS}
    t_ev = time.time()
    v21, rule, replay_equal = evaluate(tf, oos, w0, w1, fut, log=log)
    syms = list(SYMBOLS)
    ta = c2.trade_array(v21[PRIMARY], syms)
    tr = c2.trade_array(rule, syms)
    bt = c2.block_bootstrap(ta["t"], ta["net"], tr["t"], tr["net"], w0, w1)
    sens = {}
    for mg, tb in v21.items():
        sens[f"margin_{mg:.2f}"] = {f"cost_{c:.2f}": c2._compact(c2.trade_stats(c2.trade_array(tb, syms, c), syms))
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
    for row in meta:
        row.pop("gain_long", None)
        row.pop("gain_short", None)
    res = {"tf": tf, "window": window, "window_ms": [w0, w1], "window_dates": list(WINDOWS[window]),
           "run_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
           "core_feats_sha256": cf.source_hash(), "core2_1_sha256": c2.file_sha256(os.path.abspath(__file__)),
           "features": names, "n_features": len(names), "dropped_features": cf.DROPPED.get(tf, {}),
           "params": {k: v for k, v in lgb_params(tf, meta[-1].get("n_train", 0) if meta else 0, threads).items()
                      if k != "min_data_in_leaf"}, "rounds": ROUNDS[tf],
           "train_warmup_bars": c2.TRAIN_WARMUP[tf], "subsample": c2.SUBSAMPLE.get(tf, 1),
           "dataset_rows": int(len(ds["t"])), "oos_rows": int(len(oos["t"])),
           "v21": c2.trade_stats(ta, syms), "rule": c2.trade_stats(tr, syms),
           "rule_maker_0.10": c2._compact(c2.trade_stats(c2.trade_array(rule, syms, MAKER_COST), syms)),
           "bootstrap": bt, "sensitivity": sens, "importance_gain_top20": imp, "ic": c2.ic_stats(oos, syms),
           "replay_equal_to_decision_replay": bool(replay_equal),
           "months": meta,
           "timings_s": {"dataset": ds["build_s"], "fit_total": round(fit_s, 1),
                         "evaluate": round(time.time() - t_ev, 1), "wall": round(time.time() - wall, 1)}}
    return res, oos


def assemble(results, tfs=TFS):
    """گزارشِ کلِ اعتبارسنجی: Holm روی **پنج** تایم‌فریم (نبود ⇒ p=1 و گزارش ناقص) و قاعدهٔ پذیرش."""
    pv = {tf: (results[tf]["bootstrap"]["p_one_sided"] if tf in results else None) for tf in tfs}
    hm = c2.holm(pv, c2.HOLM_ALPHA)
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
    return {"title": "core v2.1 — validation (walk-forward, pre-registered)", "window": "validation",
            "window_dates": list(WINDOWS["validation"]), "prereg": "bot/data/research/prereg_core2_1.json",
            "prereg_sha256": c2.file_sha256(PREREG_PATH) if os.path.exists(PREREG_PATH) else None,
            "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
            "core_feats_sha256": cur, "complete": bool(complete),
            "holm_alpha": c2.HOLM_ALPHA,
            "bootstrap": {"B": c2.BOOT_B, "seed": c2.BOOT_SEED, "block": "calendar week (UTC, Mon), all 5 coins jointly"},
            "decision_rule": {"primary": "E_L > 0 and E_L > E_S (short mirrored)", "sensitivity_margins": [0.05, 0.10]},
            "cost_pct": COST, "maker_cost_pct_descriptive": MAKER_COST, "holdout_touched": False,
            "adopted": [tf for tf in tfs if cells[tf].get("adopted")], "timeframes": cells}


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
    return out


# ───────────────────────── نگهبانِ holdout ─────────────────────────
def latest_validation_report(research_dir=None):
    files = sorted(glob.glob(os.path.join(research_dir or RESEARCH_DIR, REPORT_PREFIX + "*.json")))
    if not files:
        return None, None
    with open(files[-1], "r", encoding="utf-8") as f:
        return json.load(f), files[-1]


def make_pin(report_path, pin_path=None):
    """سنجاقِ گزارشِ اعتبارسنجی (بلافاصله پس از اعتبارسنجی commit می‌شود)."""
    pin = {"validation_report": os.path.basename(report_path), "validation_report_sha256": c2.file_sha256(report_path),
           "core2_1_sha256": c2.file_sha256(os.path.abspath(__file__)),
           "prereg_sha256": c2.file_sha256(PREREG_PATH), "core_feats_sha256": cf.source_hash(),
           "pinned_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")}
    p = pin_path or PIN_PATH
    with open(p, "w", encoding="utf-8") as f:
        json.dump(pin, f, indent=1)
    return pin


def _read_log(log_path=None):
    p = log_path or HOLDOUT_LOG
    if not os.path.exists(p):
        return []
    out = []
    with open(p, encoding="utf-8") as f:
        for line in f:
            try:
                out.append(json.loads(line))
            except ValueError:
                continue
    return out


def committed_in_head(path):
    """متنِ فایل در HEAD (آنچه واقعاً commit شده)، یا ``None``."""
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    rel = os.path.relpath(path, root).replace(os.sep, "/")
    try:
        r = subprocess.run(["git", "show", f"HEAD:{rel}"], cwd=root, capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return None
    return r.stdout if r.returncode == 0 else None


def write_start_record(tf, report_path, log_path=None):
    """رکوردِ شروعِ append-only؛ باید پیش از اجرای holdout commit شود."""
    rec = {"event": "holdout_start", "tf": tf, "validation_report": os.path.basename(report_path),
           "validation_report_sha256": c2.file_sha256(report_path),
           "core2_1_sha256": c2.file_sha256(os.path.abspath(__file__)),
           "utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")}
    p = log_path or HOLDOUT_LOG
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return rec


def holdout_guard(tf, report, report_path, out_dir=None, pin_path=None, log_path=None, head_reader=None):
    """``(مجاز؟, دلیل)``."""
    if tf not in TFS:
        return False, f"تایم‌فریمِ ناشناخته: {tf}"
    if not report or not report_path:
        return False, "گزارشِ اعتبارسنجی پیدا نشد"
    if report.get("window") != "validation" or not report.get("complete"):
        return False, "گزارشِ اعتبارسنجی کامل نیست (هر پنج تایم‌فریم با همان core_feats لازم است)"
    if ((report.get("timeframes") or {}).get(tf) or {}).get("adopted") is not True:
        return False, f"{tf} در اعتبارسنجی پذیرفته نشده — holdout اجرا نمی‌شود"
    pp = pin_path or PIN_PATH
    if not os.path.exists(pp):
        return False, "سنجاقِ گزارشِ اعتبارسنجی (core2_1_validation_pin.json) وجود ندارد"
    with open(pp, encoding="utf-8") as f:
        pin = json.load(f)
    if pin.get("validation_report_sha256") != c2.file_sha256(report_path):
        return False, "هشِ گزارشِ اعتبارسنجی با سنجاق یکی نیست"
    if pin.get("core2_1_sha256") != c2.file_sha256(os.path.abspath(__file__)):
        return False, "کدِ core2_1 پس از سنجاق تغییر کرده"
    if pin.get("prereg_sha256") != c2.file_sha256(PREREG_PATH) or pin.get("core_feats_sha256") != cf.source_hash():
        return False, "پیش‌ثبت یا core_feats پس از سنجاق تغییر کرده"
    if os.path.exists(result_path(tf, "holdout", out_dir)):
        return False, f"holdoutِ {tf} قبلاً یک‌بار اجرا شده (یک‌بارمصرف)"
    lp = log_path or HOLDOUT_LOG
    recs = [r for r in _read_log(lp) if r.get("tf") == tf]
    if any(r.get("event") == "holdout_done" for r in recs):
        return False, f"holdoutِ {tf} قبلاً یک‌بار اجرا شده (یک‌بارمصرف)"
    start = [r for r in recs if r.get("event") == "holdout_start"
             and r.get("validation_report_sha256") == pin["validation_report_sha256"]]
    if not start:
        return False, "رکوردِ شروعِ holdout در دفترِ append-only نیست"
    head = (head_reader or committed_in_head)(lp)
    if head is None or json.dumps(start[-1], ensure_ascii=False) not in head:
        return False, "رکوردِ شروعِ holdout هنوز در گیت commit نشده"
    return True, ""


def holdout_verdict(tf, res):
    v = res["v21"]
    nm = HOLD_N_MIN.get(tf, HOLD_N_MIN_DEFAULT)
    return {"n_min": nm, "passed": bool(v["mean"] is not None and v["mean"] > 0 and v["n"] >= nm)}
