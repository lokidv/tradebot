# -*- coding: utf-8 -*-
"""پیش‌ثبت و داوریِ یک‌بارمصرف — تنها راهِ رسیدن به ``gates.allowed_combos``.

چرخهٔ قبلی: هر روز مدل‌ها از نو ساخته می‌شدند، یک «آزمونِ دست‌نخورده» روی ۲۰٪
آخرِ داده قضاوت می‌کرد، و فردا دوباره. یعنی همان پنجره ۳۶۵ بار در سال دیده
می‌شد و هر بار شانسی تازه برای عبور از گیت بود. به‌علاوه ده‌ها ترکیب بدونِ هیچ
تصحیحی هم‌زمان آزموده می‌شد.

اینجا:

۱. **پیش‌ثبت** — پیش از دیدنِ نتیجه، خانوادهٔ فرضیه‌ها (کدام تایم‌فریم × ستاپ ×
   جهت × کدام ارزها)، براکت، مدلِ هزینه، پنجرهٔ آزمون و **اعدادِ گیت** در
   ``preregistration.json`` نوشته و هش می‌شوند؛ هش در ``gates.json`` می‌نشیند.
۲. **پنجرهٔ منجمد** — بازهٔ زمانیِ آزمون با مهرِ زمانیِ مطلق ثبت می‌شود و
   ``calib.build`` رویدادهای آن بازه را **تا داوری** از **همهٔ** آموزش‌ها کنار می‌گذارد.
   پس از داوریِ همین پیش‌ثبت (``judged_<hash>.json``) آن بازه دوباره در آموزش می‌آید (calib-F15)؛
   پیش‌ثبتِ تازه پنجرهٔ تازه‌ای منجمد می‌کند تا داوریِ خودش.
۳. **داوریِ یک‌بار** — هر پیش‌ثبت فقط یک‌بار قضاوت می‌شود؛ بارِ دوم استثنا
   می‌دهد. نتیجه با تصحیحِ Romano-Wolf روی کلِ خانواده و کرانِ پایینِ بوت‌استرپ
   سنجیده می‌شود.
۴. **تنها نویسنده** — فقط همین داور می‌تواند ``allowed_combos`` را بنویسد.

تغییرِ هر عددی پس از دیدنِ نتیجه یعنی پیش‌ثبتِ تازه، هشِ تازه و آزمونِ تازه —
و شمارِ تلاش‌ها (``n_trials``) بالا می‌رود تا آستانهٔ معناداری سخت‌تر شود.
"""
import hashlib
import json
import os
import time

import numpy as np

import bracket
import gates
import paths
import stats

DATA_DIR = paths.DATA_DIR
PREREG_PATH = os.path.join(DATA_DIR, "preregistration.json")
FINAL_DIR = os.path.join(DATA_DIR, "final_test")

TF_MS = {"15m": 900_000, "1h": 3_600_000, "4h": 14_400_000, "1d": 86_400_000}

# ارزهای بزرگ و پایدار — جایی که سوگیریِ بقا تقریباً صفر است
MAJORS = (
    "BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT", "DOGEUSDT", "ADAUSDT",
    "AVAXUSDT", "LINKUSDT", "DOTUSDT", "TRXUSDT", "LTCUSDT", "BCHUSDT", "ATOMUSDT",
    "ETCUSDT", "NEARUSDT", "APTUSDT", "ARBUSDT", "FILUSDT", "UNIUSDT",
)

# خانوادهٔ پیش‌فرض: {4h, 1d} × {zx, sq} × {long, short} — ۸ فرضیه، فقط ارزهای بزرگ.
# 15m و 1h عمداً بیرون‌اند: بازدهٔ خالصِ پایه‌شان پس از هزینه منفی است
# (−۰٫۲۱R و −۰٫۰۷R)، پس آزمودنشان فقط شمارِ تلاش‌ها را بالا می‌برد.
DEFAULT_FAMILY = [
    {"tf": tf, "setup": setup, "side": side}
    for tf in ("4h", "1d") for setup in ("zx", "sq") for side in ("long", "short")
]

# اعدادِ گیت — پیش از دیدنِ هر نتیجهٔ واقعی ثابت شده‌اند.
#
# چرا min_n_eff = ۱۰۰ و نه ۱۵۰ِ پیش‌نویسِ پلن: با برچسبِ ۴۰ کندلی و همبستگیِ
# مقطعیِ ۰٫۶، بیست ارزِ بزرگ روی ۴ ساعته در ۱۵ ماه فقط ~۱۱۰ مشاهدهٔ مستقل می‌سازند.
# با ۱۵۰، حتی لبهٔ +۰٫۴۷R با t=۳٫۵ و p_adj≈۰ رد می‌شد — یعنی آزمون **از ساختار**
# غیرقابل‌عبور بود، که ردِ صادقانه نیست بلکه ابزارِ خراب است. کنترلِ کشفِ کاذب
# را کرانِ پایینِ بوت‌استرپ و Romano-Wolf انجام می‌دهند؛ n_eff فقط کفِ مقدارِ شاهد است.
# این تغییر پیش از هر قضاوت روی دادهٔ واقعی انجام شده است.
DEFAULT_GATES = {
    "min_n": 300,
    "min_n_eff": 100,
    "lcb_alpha": 0.05,            # کرانِ پایینِ یک‌طرفهٔ ۹۵٪ باید > ۰ باشد
    "min_mean_net_r": 0.10,
    "min_profit_factor": 1.15,
    "max_drawdown_r": 15.0,
    "family_alpha": 0.05,         # p تصحیح‌شدهٔ Romano-Wolf
    "min_positive_quarters": 3,   # از ۴ بلوکِ زمانیِ مساوی
}

FINAL_TEST_FRACTION = 0.25        # ۲۵٪ آخرِ بازهٔ هر تایم‌فریم، با مهرِ زمانیِ مطلق


class PreregistrationError(RuntimeError):
    pass


class AlreadyJudged(RuntimeError):
    pass


# ───────────────────────── پیش‌ثبت ─────────────────────────
def _canonical(obj):
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def prereg_hash(doc):
    body = {k: v for k, v in doc.items() if k not in ("hash", "judged")}
    return hashlib.sha256(_canonical(body).encode("utf-8")).hexdigest()


def load_prereg():
    if not os.path.exists(PREREG_PATH):
        return None
    with open(PREREG_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def final_windows_from_spans(spans, fraction=FINAL_TEST_FRACTION, end_ms=None):
    """پنجرهٔ آزمونِ منجمد برای هر تایم‌فریم: ``fraction`` آخرِ بازه، تا لحظهٔ ثبت."""
    out = {}
    for tf, (t0, t1) in spans.items():
        end = min(int(t1), int(end_ms)) if end_ms else int(t1)
        start = int(end - (end - int(t0)) * float(fraction))
        out[tf] = {"start_ms": start, "end_ms": end}
    return out


def register(spans, family=None, gate_numbers=None, symbols=MAJORS, note="", now=None):
    """ثبتِ خانوادهٔ فرضیه‌ها و اعدادِ گیت، **پیش از** هر قضاوتی.

    ``spans``: ``{tf: (first_ts_ms, last_ts_ms)}`` از داده‌ای که موجود است.
    شمارِ تلاش‌ها از پیش‌ثبت‌های قبلی به ارث می‌رسد و یکی بالا می‌رود.
    """
    now = now if now is not None else time.time()
    prev = load_prereg()
    n_trials = int((prev or {}).get("n_trials", 0)) + 1
    family = list(family or DEFAULT_FAMILY)
    windows = final_windows_from_spans(spans, end_ms=int(now * 1000))
    # آزمونِ تازه فقط روی داده‌ای که **هیچ داوریِ قبلی ندیده**. بدونِ این، پیش‌ثبتِ دوم
    # «۲۵٪ آخر تا حالا» را می‌گرفت که با پنجرهٔ داوری‌شدهٔ قبلی هم‌پوشان است — یعنی
    # همان استفادهٔ دوباره از دادهٔ آزمون که کلِ این سازوکار برای جلوگیری از آن است.
    for tf, w in windows.items():
        prev_w = ((prev or {}).get("final_windows") or {}).get(tf)
        if prev_w:
            w["start_ms"] = max(int(w["start_ms"]), int(prev_w["end_ms"]))
            w["after_previous_trial"] = prev.get("hash")
    doc = {
        "schema": 1,
        "registered_at": now,
        "note": note,
        "family": family,
        "symbols": sorted(symbols),
        "gates": dict(gate_numbers or DEFAULT_GATES),
        "bracket": {"r_atr_mult": bracket.R_ATR_MULT, "r_min_pct": bracket.R_MIN_PCT,
                    "tp_r": bracket.TP_R, "max_bars": bracket.MAX_BARS,
                    "entry": "next_bar_open"},
        "cost_model": "costs.event_cost_pct (point-in-time tier + funding)",
        "final_windows": windows,
        "n_trials": n_trials,
        "previous_hash": (prev or {}).get("hash"),
    }
    doc["hash"] = prereg_hash(doc)
    os.makedirs(DATA_DIR, exist_ok=True)
    tmp = PREREG_PATH + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(doc, f, ensure_ascii=False, indent=1)
    os.replace(tmp, PREREG_PATH)
    gates.bind_preregistration(doc["hash"], reason=f"preregistered trial {n_trials}")
    return doc


def ensure_registered(spans, now=None):
    """اگر هنوز پیش‌ثبتی نیست، همین حالا ثبت کن — **پیش از** آموزش و داوری.

    ``calib.build`` این را پس از استخراجِ رویدادها و پیش از هر برازشی صدا می‌زند؛
    پس تا داوری هیچ مدلی رویدادهای پنجرهٔ منجمد را ندیده و داور اولین کسی است که نگاهشان
    می‌کند (پس از داوری، ``calib.build`` آن بازه را دوباره در آموزش می‌آورد). اعدادِ گیت و خانوادهٔ فرضیه ثابت‌های کد هستند که پیش از هر قضاوتی
    تعیین شده‌اند. اگر پیش‌ثبت وجود داشته باشد دست نمی‌خورد (همان یک‌بار).
    """
    doc = load_prereg()
    if doc:
        return doc, False
    spans = {tf: v for tf, v in (spans or {}).items()
             if tf in {h["tf"] for h in DEFAULT_FAMILY}}
    if not spans:
        return None, False
    return register(spans, note="auto: first build with frozen-test support", now=now), True


def verify_prereg(doc=None):
    """پیش‌ثبت دست‌نخورده است و هشش با gates.json می‌خواند؟"""
    doc = doc or load_prereg()
    if not doc:
        raise PreregistrationError("هیچ پیش‌ثبتی وجود ندارد — اول register() را اجرا کنید")
    if prereg_hash(doc) != doc.get("hash"):
        raise PreregistrationError("preregistration.json پس از ثبت ویرایش شده است")
    bound = gates.load_gates(force=True).get("preregistration_hash")
    if bound != doc["hash"]:
        raise PreregistrationError("هشِ پیش‌ثبت با gates.json نمی‌خواند")
    return doc


def in_final_window(tf, ts_ms, doc=None):
    """آیا این رویداد جزوِ پنجرهٔ منجمد است (یعنی از آموزش کنار گذاشته شود)؟"""
    doc = doc if doc is not None else load_prereg()
    if not doc:
        return False
    w = (doc.get("final_windows") or {}).get(tf)
    return bool(w and w["start_ms"] <= int(ts_ms) < w["end_ms"])


# ───────────────────────── داوری ─────────────────────────
def _lock_path(doc):
    return os.path.join(FINAL_DIR, f"judged_{doc['hash'][:16]}.json")


def _max_drawdown_r(values):
    equity = np.cumsum(np.asarray(values, float))
    peak = np.maximum.accumulate(np.concatenate([[0.0], equity]))[1:]
    return float(np.max(peak - equity)) if len(equity) else 0.0


def _positive_quarters(values, ts):
    if len(values) < 4:
        return 0
    order = np.argsort(ts)
    v = np.asarray(values, float)[order]
    return int(sum(1 for chunk in np.array_split(v, 4) if len(chunk) and chunk.mean() > 0))


def _classify(s, g):
    """«لبه نیست» با «شواهد کافی نیست» فرق دارد و کاربر باید بداند کدام است.

    * ``edge_proven`` — همهٔ گیت‌ها عبور کرد.
    * ``no_edge`` — نمونهٔ کافی هست و میانگین یا کرانِ پایین ≤ ۰ است.
    * ``insufficient_evidence`` — اثر مثبت است ولی نمونهٔ مستقل برای اثباتش کم است.
    * ``weak_edge`` — نمونه کافی و اثر مثبت، ولی زیرِ آستانهٔ اقتصادی یا پرنوسان.
    """
    if s.get("verdict") == "pass":
        return "edge_proven"
    enough = (s.get("n") or 0) >= g["min_n"] and (s.get("n_eff") or 0) >= g["min_n_eff"]
    mean, lcb = s.get("mean"), s.get("lcb")
    if mean is None:
        return "insufficient_evidence"
    if mean <= 0:
        return "no_edge"
    if not enough or lcb is None:
        return "insufficient_evidence"
    if lcb <= 0:
        return "no_edge"
    return "weak_edge"


def select_events(events, hyp, symbols, window):
    """رویدادهای یک فرضیه در پنجرهٔ منجمد، با بازدهٔ خالص."""
    d = 1 if hyp["side"] == "long" else -1
    out_r, out_ts = [], []
    for e in events:
        if e.get("setup") != hyp["setup"] or int(e.get("dir") or 0) != d:
            continue
        if e.get("sym") not in symbols:
            continue
        ts = int(e["ts"])
        if not (window["start_ms"] <= ts < window["end_ms"]):
            continue
        risk = max(float(e.get("risk_pct") or 0), 0.05)
        cost = float(e.get("cost_pct", 0.15))
        out_r.append(float(e["r"]) - cost / risk)
        out_ts.append(ts)
    return np.asarray(out_r, float), np.asarray(out_ts, np.int64)


def evaluate(events_by_tf, doc):
    """آمار و حکمِ هر فرضیه — بدونِ نوشتنِ چیزی. برای تست و پیش‌نمایش."""
    g = doc["gates"]
    symbols = set(doc["symbols"])
    per, family = {}, {}
    for hyp in doc["family"]:
        key = gates.combo_key(hyp["tf"], hyp["setup"], hyp["side"])
        window = doc["final_windows"].get(hyp["tf"])
        if not window:
            per[key] = {"verdict": "no_window"}
            continue
        vals, ts = select_events(events_by_tf.get(hyp["tf"]) or [], hyp, symbols, window)
        block = bracket.MAX_BARS * TF_MS[hyp["tf"]]
        s = stats.summarize(vals, ts, block, alpha=g["lcb_alpha"], B=1000)
        s["max_drawdown_r"] = round(_max_drawdown_r(vals), 3)
        s["positive_quarters"] = _positive_quarters(vals, ts)
        per[key] = s
        if len(vals) >= 2:
            family[key] = (vals, ts)
    rw = stats.romano_wolf(family, bracket.MAX_BARS * TF_MS["4h"],
                           alpha=g["family_alpha"], B=1000) if family else {}
    passing = []
    for key, s in per.items():
        if "n" not in s:
            continue
        reasons = []
        if s["n"] < g["min_n"]:
            reasons.append(f"n={s['n']} < {g['min_n']}")
        if s["n_eff"] < g["min_n_eff"]:
            reasons.append(f"n_eff={s['n_eff']} < {g['min_n_eff']}")
        if s["lcb"] is None:
            reasons.append("کرانِ پایین نامعلوم (خوشه‌های زمانیِ ناکافی)")
        elif s["lcb"] <= 0:
            reasons.append(f"LCB={s['lcb']} ≤ 0")
        if s["mean"] is None or s["mean"] < g["min_mean_net_r"]:
            reasons.append(f"mean={s['mean']} < {g['min_mean_net_r']}")
        if (s["profit_factor"] or 0) < g["min_profit_factor"]:
            reasons.append(f"PF={s['profit_factor']} < {g['min_profit_factor']}")
        if s["max_drawdown_r"] > g["max_drawdown_r"]:
            reasons.append(f"DD={s['max_drawdown_r']}R > {g['max_drawdown_r']}R")
        if s["positive_quarters"] < g["min_positive_quarters"]:
            reasons.append(f"فصل‌های مثبت {s['positive_quarters']}/4")
        p_adj = (rw.get(key) or {}).get("p_adj")
        s["p_adj"] = p_adj
        if p_adj is None or p_adj >= g["family_alpha"]:
            reasons.append(f"p_adj={p_adj} ≥ {g['family_alpha']}")
        s["fail_reasons"] = reasons
        s["verdict"] = "pass" if not reasons else "fail"
        s["evidence"] = _classify(s, g)
        # چند مشاهدهٔ مستقل لازم بود تا اثرِ دیده‌شده با توانِ ۸۰٪ تشخیص داده شود؟
        s["n_eff_needed_for_observed_effect"] = (
            stats.power_sample_size(s["mean"], sd=s["sd"] or 1.25)
            if s.get("mean") and s["mean"] > 0 else None)
        if not reasons:
            passing.append(key)
    return {"hypotheses": per, "passing": sorted(passing),
            "n_trials": doc.get("n_trials", 1)}


def judge(events_by_tf, doc=None, now=None):
    """قضاوتِ **یک‌باره**. نتیجه در final_test ثبت و ترکیب‌های قبول‌شده در gates نوشته می‌شوند.

    بارِ دوم برای همان پیش‌ثبت ``AlreadyJudged`` می‌دهد — حتی اگر نتیجهٔ بار
    اول بد بوده باشد. راهِ دوباره آزمودن، پیش‌ثبتِ تازه است که شمارِ تلاش را بالا می‌برد.
    """
    doc = verify_prereg(doc)
    os.makedirs(FINAL_DIR, exist_ok=True)
    lock = _lock_path(doc)
    if os.path.exists(lock):
        raise AlreadyJudged(f"این پیش‌ثبت قبلاً قضاوت شده است: {lock}")
    result = evaluate(events_by_tf, doc)
    result.update(prereg_hash=doc["hash"], judged_at=now if now is not None else time.time(),
                  symbols=doc["symbols"])
    # قفل با O_EXCL: دو فراخوانِ هم‌زمان هم نمی‌توانند هر دو قضاوت کنند
    fd = os.open(lock, os.O_WRONLY | os.O_CREAT | os.O_EXCL)
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=1, default=float)
    gver = gates.load_gates(force=True)["version"]
    gates.set_allowed_combos(result["passing"],
                             reason=f"judged prereg {doc['hash'][:12]}",
                             judge_token=f"judge:{gver}",
                             symbols=doc["symbols"])
    return result


def last_judgement():
    doc = load_prereg()
    if not doc:
        return None
    lock = _lock_path(doc)
    if not os.path.exists(lock):
        return {"prereg_hash": doc["hash"], "judged": False}
    with open(lock, "r", encoding="utf-8") as f:
        out = json.load(f)
    out["judged"] = True
    return out
