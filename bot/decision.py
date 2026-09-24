# -*- coding: utf-8 -*-
"""تصمیمِ روشنِ هر ارز × تایم‌فریم — لانگ / شورت / صبر — همیشه کنارِ کارنامهٔ اندازه‌گیری‌شده‌اش.

کاربر دستی روی کوکوین معامله می‌کند و مثلِ نسخهٔ ژوئیه یک جوابِ روشن می‌خواهد. این ماژول همان
قاعدهٔ موتور (``engine.analyze``) را به یک «تصمیم» تبدیل می‌کند و **هیچ قاعدهٔ تازه‌ای** نمی‌سازد:

* اگر یکی از ستاپ‌های رویدادی در کندلِ بسته‌شدهٔ فعلی/قبلی رخ داده باشد، جهتِ همان ستاپ؛
* وگرنه قاعدهٔ z/رأی: ``z ≥ 1.2`` و دست‌کم ۳ رأی از ۵ (چهار دسته + رأیِ kNN «الگوی مشابه») و
  حداکثر ۱ رأیِ مخالف — و قرینه‌اش برای شورت؛
* براکتِ واحدِ ``bracket.py``: حدضرر 1R و هدف 1.8R.

⚠️ این یک **تحلیل** است، نه لبهٔ تأییدشده. بازپخشِ دوسالهٔ همین قاعده (پس از هزینهٔ ۰٫۱۴٪) در 15m
و 1h روی هر پنج ارز زیان‌ده است، در 4h حدوداً سربه‌سر، و در 1d معامله‌ها برای قضاوت کم‌اند.
پس هر تصمیم با دو کارنامه نمایش داده می‌شود:

1. ``scorecards()`` — بازپخشِ تاریخیِ دقیقاً همین قاعده روی ۷۳۰ روزِ اخیر (علّی، یک معامله در
   هر لحظه، ورود در بازِ کندلِ بعد، هزینهٔ رفت‌وبرگشتِ ۰٫۱۴٪ = تیکرِ فیوچرزِ کوکوین ۰٫۰۶٪×۲ + ۰٫۰۲٪ لغزش)
   با بازهٔ اطمینانِ ۹۵٪ روی میانگینِ R؛ ستاپ‌های معلق‌شده و مسیرِ «سیاست» بازپخش نمی‌شوند؛
2. ``live_stats()`` — دفترِ رو به جلوی ``decision_ledger.jsonl`` (فقط افزودنی) که هر پیشنهادِ زنده را
   ثبت و بعداً با همان براکت داوری می‌کند.

این ماژول هیچ مجوزِ معامله‌ای نمی‌دهد و به ``gates``/``autotrader`` دست نمی‌زند؛ مرجعِ معامله فقط gates.json است.
"""
import json
import math
import os
import threading
import time
from datetime import datetime, timezone

import numpy as np

import bracket
import engine
import paths
import tf_spec
from watchlist import SYMBOLS

import log

TFS = tf_spec.TFS          # 5m | 15m | 1h | 4h | 1d — ثبتِ واحد: bot/tf_spec.py
TF_MINUTES = tf_spec.MINUTES

COST_PCT = 0.14            # رفت‌وبرگشت: تیکرِ فیوچرزِ کوکوین ۰٫۰۶٪ × ۲ + ۰٫۰۲٪ لغزش
# گونهٔ دوم (فقط نمایش، ثانوی): ورودِ «میکر» با سفارشِ محدود — ۰٫۰۲٪ + خروجِ تیکر ۰٫۰۶٪ + ۰٫۰۲٪ لغزش.
# همان معامله‌ها (انتخابِ معامله به هزینه بستگی ندارد)، فقط هزینهٔ کمتر؛ پرشدنِ سفارشِ محدود تضمینی نیست.
MAKER_COST_PCT = 0.10
WINDOW_DAYS = 730          # پنجرهٔ کارنامه: دو سالِ اخیرِ دادهٔ موجود
LIVE_BARS = 420            # موتورِ زنده روی market.get_klines(limit=420) اجرا می‌شود — kNN هم همین پنجره را می‌بیند
WARMUP_BARS = 2000         # کندل‌های پیش از پنجره فقط برای گرم‌شدنِ اندیکاتورها
Z_ENTRY = 1.2              # همان آستانهٔ engine.trade_suggestion
AI_VOTE = 0.2              # رأیِ kNN: s_ml ≥ 0.2 صعودی، ≤ −0.2 نزولی
LEAN_MIN = 50              # «تمایل» فقط وقتی دست‌کم نیمی از شرط‌ها پر شده
MIN_TRADES = 30
Z95 = 1.96                 # بازهٔ ۹۵٪ (تقریبِ نرمال) روی میانگینِ R خالص
SCORECARD_TTL = 86400      # حداکثر روزی یک‌بار بازسازی
SCORECARD_VERSION = 3      # ۲: بازهٔ اطمینان + verdict_code + پنجرهٔ واقعی؛ ۳: 5m، آرشیوِ بومیِ هر تایم‌فریم، گونهٔ میکر
STALE_BARS = 3             # تصمیمِ کهنه‌تر از این (فیدِ مرده) در دفتر ثبت نمی‌شود
MIN_BARS = LIVE_BARS + 5   # کمتر از این، بازپخش حتی یک کندلِ قابلِ داوری ندارد
SETUP_LOOKBACK = max(int(getattr(engine, "SETUP_LOOKBACK", 2)), 1)   # کندلِ فعلی + قبلی، مثلِ analyze
MAX_WARNINGS = 4
SOURCE_ARCHIVE_FMT = "binance-futures-{tf}-archive"          # آرشیوِ بومیِ همان تایم‌فریم
SOURCE_RESAMPLED_FMT = "binance-futures-{base}-archive→{tf}"   # فقط از آرشیوِ ریزتر، هرگز درشت‌تر
SOURCE_FALLBACK = "binance-spot-history"

VERDICT_FA = {"small": "نمونهٔ کم", "loss": "زیان‌ده", "positive": "مثبت (تأییدنشده)",
              "unclear": "نامشخص — بازه شامل صفر"}

SCORECARD_PATH = paths.data("research", "decision_scorecard.json")
LEDGER_PATH = paths.data("decision_ledger.jsonl")
MICRO_DIR = paths.data("hist_research", "micro")     # um_<SYM>_<tf>.npz: فیوچرزِ بایننس ۲۰۲۲-۰۱..۲۰۲۶-۰۸

# (کلید، برچسب، سریِ component_series، آستانهٔ رأی) — همان آستانه‌های engine.votes_at
COMPONENTS = (
    ("trend", "روند", "s_trend", 0.30),
    ("mom", "مومنتوم", "s_mom", 0.25),
    ("vol", "حجم", "s_vol", 0.25),
    ("struct", "ساختار", "s_struct", 0.50),
    ("ai", engine.KNN_LABEL, None, AI_VOTE),   # رأیِ kNN — برچسبِ صادقانه (قبلاً «هوش مصنوعی»)؛ رأی و شمارش همان
)
# برچسب‌های قدیمیِ کلیدِ components (خروجیِ کش‌شده/آزمون‌های پیش از تغییرِ برچسب) — فقط خواندن
_LEGACY_FA = {"ai": engine.KNN_LABEL_LEGACY}
SIDE_FA = {"long": "لانگ (خرید)", "short": "شورت (فروش)"}

NOTE_FA = ("این‌ها تحلیلِ موتورِ چندشاخصه + رأیِ الگوی مشابه (kNN) هستند، نه لبهٔ تأییدشده. بازپخشِ دوسالهٔ همین قاعده "
           "پس از هزینهٔ رفت‌وبرگشتِ ۰٫۱۴٪ در 15m و 1h روی هر پنج ارز زیان‌ده است، در 4h حدوداً سربه‌سر است، "
           "و در 1d تعدادِ معامله‌ها برای قضاوت کافی نیست. در 5m حدضرر از همه کوچک‌تر است، پس همین هزینه سهمِ "
           "بزرگ‌تری از هر R را می‌خورد. پیشنهادهای زنده‌ای که از ستاپِ معلق‌شده یا از مسیرِ "
           "«سیاستِ انتخابِ عمل» می‌آیند در این بازپخش پوشش داده نشده‌اند. گونهٔ «ورودِ میکر» (۰٫۱۰٪) فقط برای "
           "مقایسه است و پرشدنِ سفارشِ محدود را تضمین نمی‌کند. هر پیشنهاد در دفترِ رو به جلو ثبت "
           "و داوری می‌شود؛ مجوزِ معامله فقط از gates.json می‌آید.")
METHOD_FA = ("بازپخشِ علّیِ قاعدهٔ زنده روی آرشیوِ بومیِ فیوچرزِ بایننسِ همان تایم‌فریم (5m، 15m، 1h، 4h، 1d): ستاپِ رویدادیِ "
             f"{engine.SETUP_LOOKBACK_FA} کندلِ بستهٔ اخیر، وگرنه z≥1.2 با ≥۳ رأی از ۵ (kNN روی پنجرهٔ ۴۲۰ کندلی "
             "مثلِ موتور) و ≤۱ رأیِ مخالف؛ ورود در بازِ کندلِ بعد، حدضرر 1R، هدف 1.8R، سقفِ ۴۰ کندل، یک معامله "
             "در هر لحظه، هزینهٔ رفت‌وبرگشتِ ۰٫۱۴٪. تعلیقِ زندهٔ ستاپ‌ها و مسیرِ «سیاست» بازپخش نمی‌شوند. "
             "برچسب با بازهٔ ۹۵٪ میانگینِ R: «زیان‌ده» فقط وقتی کلِ بازه زیرِ صفر است و «مثبت» فقط وقتی کلش "
             "بالای صفر است؛ زیرِ ۳۰ معامله «نمونهٔ کم». گونهٔ دوم (ثانوی): همان معامله‌ها با ورودِ میکر و "
             "هزینهٔ رفت‌وبرگشتِ ۰٫۱۰٪.")

_ledger_lock = threading.Lock()
_sc_lock = threading.Lock()
_bg_lock = threading.Lock()
_sc_thread = None


# ───────────────────────── کمکی‌ها ─────────────────────────
def _num(x):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def _date(ms):
    if ms is None:
        return None
    return datetime.fromtimestamp(int(ms) / 1000, timezone.utc).strftime("%Y-%m-%d")


def _read_json(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def _write_json(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)
    os.replace(tmp, path)


def _append(path, row):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _read_jsonl(path):
    if not os.path.exists(path):
        return []
    out = []
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except ValueError:
                continue          # خطِ نیمه‌نوشته (قطعِ برق) — بقیه معتبرند
            if isinstance(row, dict):
                out.append(row)
    return out


def _int(x):
    try:
        v = int(x)
    except (TypeError, ValueError, OverflowError):
        return None
    return v if v >= 0 else None


# ───────────────────────── ۱) تصمیم از خروجیِ engine.analyze ─────────────────────────
def _component_votes(comp, vb=None, vs=None):
    """رأیِ هر دسته از مقدارهای گردشدهٔ analyze، با همان آستانه‌های votes_at.

    analyze مقدارها را تا دو رقم گرد می‌کند؛ مقدارِ پیوسته‌ای که گردشده‌اش دقیقاً روی آستانه است
    (مثلاً ۰٫۳۰ از ۰٫۲۹۶) مبهم است — رأیِ واقعی را از جمعِ رأی‌های خودِ موتور (votes_bull/bear)
    درمی‌آوریم تا فهرستِ دسته‌ها هیچ‌وقت با شمارشِ موتور ناسازگار نباشد. رأیِ kNN مبهم نیست:
    s_ml میانگینِ ۱۰ برچسبِ ±۱ است و روی شبکهٔ ۰٫۱ می‌نشیند، پس گردکردن چیزی از آن پنهان نمی‌کند
    و رأی‌اش دقیقاً ``≥ 0.2`` / ``≤ −0.2`` است؛ آشتی فقط میانِ چهار دستهٔ پیوسته انجام می‌شود.
    """
    rows, amb = [], {1: [], -1: []}
    certain = {1: 0, -1: 0}
    for key, fa, series, thr in COMPONENTS:
        val = _num(comp.get(fa, comp.get(key, comp.get(_LEGACY_FA.get(key)))))
        vote = 0
        if val is not None:
            rv, t2 = round(val, 2), round(thr, 2)
            if series is None:                      # kNN: دقیق
                vote = 1 if rv >= t2 else -1 if rv <= -t2 else 0
            elif rv > t2:
                vote = 1
            elif rv < -t2:
                vote = -1
            elif rv == t2:
                amb[1].append(len(rows))
            elif rv == -t2:
                amb[-1].append(len(rows))
        if vote:
            certain[vote] += 1
        rows.append({"key": key, "fa": fa, "value": val, "vote": vote})
    totals = {1: vb, -1: vs}
    for side in (1, -1):
        idxs = amb[side]
        tot = totals[side]
        need = len(idxs) if tot is None else max(0, min(len(idxs), int(tot) - certain[side]))
        for j in idxs[:need]:
            rows[j]["vote"] = side
    bull = sum(1 for r in rows if r["vote"] == 1)
    bear = sum(1 for r in rows if r["vote"] == -1)
    return rows, (int(vb) if vb is not None else bull), (int(vs) if vs is not None else bear)


def _names(comps, vote):
    return "، ".join(c["fa"] for c in comps if c["vote"] == vote)


def _reasons(d):
    """چند جملهٔ کوتاهِ فارسی: چرا این تصمیم — یا برای «صبر»، چه چیزی کم است."""
    z = d.get("z") or 0.0
    vb, vs = d["votes_bull"], d["votes_bear"]
    comps = d["components"]
    out = []
    if d["action"] in ("long", "short"):
        is_long = d["action"] == "long"
        sd = 1 if is_long else -1
        same, opp = (vb, vs) if is_long else (vs, vb)
        if d.get("basis") == "setup":
            out.append(f"ستاپِ «{d.get('setup_fa')}» روی کندلِ بسته‌شدهٔ اخیر رخ داد")
        elif d.get("basis") == "policy":
            out.append("سیاستِ انتخابِ عمل (مدل) این جهت را داد؛ ستاپِ رویدادی دیده نشد")
        else:
            out.append(f"z امتیازِ ترکیبی {z:+.2f} از آستانهٔ {'+' if is_long else '-'}{Z_ENTRY} گذشته")
        nm = _names(comps, sd)
        out.append(f"{same} رأی از ۵ هم‌جهت" + (f" ({nm})" if nm else "") + f"، {opp} رأیِ مخالف")
        if z * sd < 0:
            out.append(f"هشدار: z ({z:+.2f}) خلافِ این جهت است")
        ai = next((c for c in comps if c["key"] == "ai"), None)
        if ai and ai["vote"] == -sd:
            out.append(f"{engine.KNN_LABEL} خلافِ این جهت رأی داده")
    else:
        lean = d.get("lean")
        if abs(z) >= Z_ENTRY:
            sd = 1 if z > 0 else -1
        else:
            sd = 1 if (lean == "long" or (lean is None and z >= 0)) else -1
        same, opp = (vb, vs) if sd == 1 else (vs, vb)
        word = "صعودی" if sd == 1 else "نزولی"
        if abs(z) < Z_ENTRY:
            out.append(f"z فعلی {z:+.2f} است؛ برای پیشنهاد به ±{Z_ENTRY} نیاز است")
        if same < 3:
            out.append(f"فقط {same} رأیِ {word} از ۵ — دست‌کم ۳ لازم است")
        if opp > 1:
            out.append(f"{opp} رأیِ مخالف — حداکثر ۱ مجاز است")
        if vb >= 2 and vs >= 2:
            out.append("دسته‌ها دوپاره‌اند")
        if not out:
            out.append(f"z ({z:+.2f}) درست روی مرزِ آستانه است")
        out.append(f"هیچ ستاپ رویدادی در {engine.SETUP_LOOKBACK_FA} کندلِ بسته‌شدهٔ اخیر رخ نداده")
        if lean:
            out.append(f"نزدیک‌ترین حالت: {SIDE_FA[lean]} — {d['readiness']}٪ از شرط‌ها پر است")
    if d.get("regime"):
        out.append(f"رژیمِ بازار: {d['regime']}")
    return out


def _empty(sym, tf):
    return {"sym": sym, "tf": tf, "candle_ts": None, "price": None,
            "action": "wait", "lean": None, "readiness": 0, "strength": 0, "grade": None,
            "basis": None, "setup": None, "setup_fa": None,
            "entry": None, "sl": None, "tp": None, "rr": None, "risk_pct": None, "atr": None,
            "components": [], "votes_bull": 0, "votes_bear": 0, "z": None, "regime": None,
            "reasons": [], "warnings": [], "scorecard_applies": True}


# وضعیتِ موتور که هشدار است (مسدود/معلق/رد/قفل، یا احتیاطِ مشخص) — نه متن‌های عمومیِ «منتظر…» /
# «مرجعِ معتبر نیست» که روی همهٔ خانه‌ها یکسان است و یادداشتِ بالای جدول آن را می‌گوید
_BLOCK_PREFIXES = ("مسدود", "معلق", "ردِ", "رد ", "قفل ایمنی")
_CAUTION_MARKERS = ("زیان‌ده", "پاس نکرده", "چاقوی سقوط", "کافی نیست", "مخالف")
COST_WARN_R = 0.2          # هزینه‌ای که دست‌کم ۰٫۲R از هر معامله می‌خورد گفته می‌شود


def _status_is_caution(s):
    s = s.strip()
    return s.startswith(_BLOCK_PREFIXES) or any(m in s for m in _CAUTION_MARKERS)


def _warnings(a, tr, side, basis):
    """هشدارها و مسدودیت‌های موتور که جدول نباید پنهان کند — فارسی، بی‌تکرار، حداکثر ۴ تا.

    ``analyze`` برای هر پیشنهاد یک ``trade.status`` می‌گذارد؛ فقط وضعیت‌های مسدود/معلق/رد و
    احتیاط‌های مشخص (روزانهٔ زیان‌ده، سیاستِ مردود، چاقوی سقوط، جهت‌یابِ مخالف) گزارش می‌شوند.
    """
    out = []

    def add(msg):
        if isinstance(msg, str) and msg.strip() and msg.strip() not in out:
            out.append(msg.strip())

    status = tr.get("status")
    if isinstance(status, str) and _status_is_caution(status):
        add(status)
    if basis == "policy":
        add("این جهت از مسیرِ «سیاستِ انتخابِ عمل» آمده و بازپخشِ دوساله آن را پوشش نمی‌دهد")
    sd = {"long": 1, "short": -1}.get(side)
    if sd:
        tf_s = _num(a.get("tf_suspended", tr.get("tf_suspended")))
        if tf_s is not None:
            add(f"معلق — بازده خالصِ زندهٔ این تایم‌فریم منفی است ({_fa(tf_s, signed=True)}R)")
        combo_s = a.get("combo_suspended", tr.get("combo_suspended"))
        if combo_s:
            add("معلق — ترکیبِ زندهٔ این تایم‌فریم و ستاپ ضعیف است")
        if tr.get("regime_veto"):
            add("مسدود — آزمونِ مستقل در رژیمِ فعلی زیانِ ساختاری نشان داده")
        btc_z = _num(a.get("btc_z", tr.get("btc_z")))
        btc_said = any("بیت‌کوین" in w for w in out)     # وضعیتِ خودِ موتور همین را گفته
        if btc_said:
            pass
        elif btc_z is not None and btc_z * sd < -0.8:
            add(f"مسدود — مخالفتِ شدید با روند بیت‌کوین (z={_fa(btc_z, 1, signed=True)})")
        elif tr.get("btc_align") is False:
            add("خلافِ جهتِ بیت‌کوین — z بیت‌کوین در همین تایم‌فریم مخالفِ این جهت است")
        add(tr.get("direction_warning"))
        for b in tr.get("meta_blocks") or []:
            add(f"متا-گیت: {b}")
        risk = _num(tr.get("risk_pct"))
        if risk and risk > 0 and COST_PCT / risk >= COST_WARN_R:
            add(f"هزینهٔ رفت‌وبرگشتِ ۰٫۱۴٪ حدودِ {_fa(COST_PCT / risk)}R از هر معامله را می‌خورد "
                f"(حدضرر {_fa(risk)}٪)")
    return out[:MAX_WARNINGS]


_FA_DIGITS = str.maketrans("0123456789.-+", "۰۱۲۳۴۵۶۷۸۹٫−+")


def _fa(x, d=2, signed=False):
    """عدد با ارقامِ فارسی برای متن‌های فارسی (هشدارها)."""
    return (f"{x:+.{d}f}" if signed else f"{x:.{d}f}").translate(_FA_DIGITS)


def from_analysis(a):
    """یک خروجیِ ``engine.analyze``/``main.get_analysis`` → یک تصمیم.

    ``action`` دقیقاً ``a["trade"]["side"]`` است (ستاپ، وگرنه قاعدهٔ z/رأی)؛ هیچ فیلترِ تازه‌ای
    اضافه نمی‌شود. ``setup`` فقط وقتی پر است که موتور واقعاً ستاپی دیده باشد (``setup_observed``)؛
    «zx» ای که analyze برای قاعدهٔ z پر می‌کند ستاپ حساب نمی‌شود.
    """
    a = a or {}
    sym, tf = a.get("symbol") or a.get("sym"), a.get("tf")
    d = _empty(sym, tf)
    err = a.get("error") if a else "تحلیل در دسترس نیست"
    if err:
        d["error"] = str(err)
        d["reasons"] = [str(err)]
        return d
    tr = a.get("trade") if isinstance(a.get("trade"), dict) else {}
    z = _num(a.get("z"))
    comp_in = a.get("components") if isinstance(a.get("components"), dict) else {}
    comps, vb, vs = _component_votes(comp_in, _int(a.get("votes_bull")), _int(a.get("votes_bear")))
    d.update(candle_ts=a.get("zt"), price=_num(a.get("price")), components=comps,
             votes_bull=vb, votes_bear=vs, z=z, regime=a.get("regime"))
    side = tr.get("side") if tr.get("side") in ("long", "short") else None
    if side:
        sd = 1 if side == "long" else -1
        same = vb if side == "long" else vs
        setup = tr.get("setup") if tr.get("setup_observed") and tr.get("setup") else None
        basis = "setup" if setup else ("policy" if tr.get("setup") == "ap" else "rule")
        # z خلافِ جهت (ستاپِ فیدِ روند می‌تواند با z منفی لانگ بدهد) قدرت نمی‌افزاید
        z_dir = max((z or 0.0) * sd, 0.0)
        strength = round(50 + 25 * min(z_dir / 2.5, 1.0) + 25 * min(same, 5) / 5)
        d.update(action=side, lean=side, readiness=100, strength=int(strength),
                 grade=tr.get("grade"), basis=basis, setup=setup,
                 setup_fa=engine.SETUP_FA.get(setup, setup) if setup else None,
                 entry=_num(tr.get("entry")), sl=_num(tr.get("sl")), tp=_num(tr.get("tp")),
                 rr=_num(tr.get("rr")), risk_pct=_num(tr.get("risk_pct")), atr=_num(tr.get("atr14")))
    else:
        readiness = int(min(max(_num(tr.get("readiness")) or 0, 0), 100))
        ready_side = tr.get("ready_side")
        lean = ready_side if readiness >= LEAN_MIN and ready_side in ("long", "short") else None
        d.update(readiness=readiness, strength=readiness, lean=lean)
    d["reasons"] = _reasons(d)
    d["warnings"] = _warnings(a, tr, side, d["basis"])
    d["scorecard_applies"] = d["basis"] != "policy"    # مسیرِ سیاست در بازپخش نیست
    return d


def collect(get_analysis, symbols=SYMBOLS, tfs=TFS):
    """تصمیمِ همهٔ ارز × تایم‌فریم‌ها با یک تابعِ تحلیل (مثلاً main.get_analysis)."""
    out = []
    for sym in symbols:
        for tf in tfs:
            try:
                a = get_analysis(sym, tf)
            except Exception as e:  # noqa: BLE001 — یک خانهٔ خراب نباید کلِ جدول را بیندازد
                a = {"error": str(e) or type(e).__name__}
            if not a or not isinstance(a, dict):
                a = {"error": "تحلیل در دسترس نیست"}
            a = dict(a)
            a.setdefault("symbol", sym)
            a.setdefault("tf", tf)
            out.append(from_analysis(a))
    return out


# ───────────────────────── ۲) kNN علّی — برابرِ engine.knn_ml روی پیشوند ─────────────────────────
def knn_features(h, l, c, cs):
    """همان چهار ویژگیِ engine.knn_ml، یک‌بار برای کلِ سری (همه علّی‌اند؛ فقط ۱۹ سطرِ اول
    با میانگینِ کل پر می‌شوند که kNN هرگز از آن‌ها استفاده نمی‌کند — نمونه‌ها از اندیسِ ۶۰ شروع می‌شوند)."""
    hlc3 = (h + l + c) / 3
    cci_raw = hlc3 - np.nan_to_num(engine.roll_mean(hlc3, 20), nan=hlc3.mean())
    cci_den = np.maximum(np.nan_to_num(engine.roll_std(hlc3, 20), nan=1.0), engine.EPS)
    return np.stack([cs["rsi"] / 100.0,
                     np.clip(cs["adx"] / 50.0, 0, 1),
                     (np.clip(cci_raw / (2 * cci_den), -1, 1) + 1) / 2,
                     engine.rsi(c, 9) / 100.0], axis=1)


def knn_series(h, l, c, cs, k=10, stride=2, max_samples=1500, window=None, idx=None):
    """رأیِ kNN در هر کندلِ ``i`` فقط با کندل‌های ``≤ i``.

    ``window=None``: دقیقاً ``engine.knn_ml(h[:i+1], l[:i+1], c[:i+1], cs_پیشوند)``.
    ``window=W``: همان الگوریتم روی پنجرهٔ W کندلیِ منتهی به i (شبکهٔ نمونه از ابتدای پنجره) —
    موتورِ زنده روی ۴۲۰ کندل اجرا می‌شود، پس کارنامه از ``window=LIVE_BARS`` استفاده می‌کند.
    ``idx``: فقط این اندیس‌ها محاسبه شوند (خروجی هم‌ترتیبِ idx)؛ پیش‌فرض همهٔ کندل‌ها.
    """
    h = np.asarray(h, float)
    l = np.asarray(l, float)
    c = np.asarray(c, float)
    n = len(c)
    f = knn_features(h, l, c, cs)
    lab_all = np.zeros(n)
    if n > 4:
        lab_all[:n - 4] = np.sign(c[4:] - c[:-4])      # برچسبِ نمونهٔ j: c[j+4]، و j ≤ i−5 → فقط گذشته
    idx = np.arange(n) if idx is None else np.asarray(idx, dtype=int).reshape(-1)
    out = np.zeros(len(idx))
    for p, i in enumerate(idx):
        s = 0 if window is None else max(0, int(i) - int(window) + 1)
        J = np.arange(s + 60, int(i) - 4, stride)      # = range(60, m−5, stride) درونِ پنجره
        if len(J) < 30:
            continue
        lab = lab_all[J]
        keep = lab != 0
        J, lab = J[keep], lab[keep]
        if len(J) < 30:
            continue
        if len(J) > max_samples:
            J, lab = J[-max_samples:], lab[-max_samples:]
        t = np.log1p(np.abs(f[J] - f[i]))
        dist = ((t[:, 0] + t[:, 1]) + t[:, 2]) + t[:, 3]  # همان ترتیبِ جمعِ ۴ عنصری
        order = np.argsort(dist)[:k]
        out[p] = float(np.mean(lab[order]))
    return out


# ───────────────────────── ۲) بازپخشِ تاریخی — دقیقاً قاعدهٔ زنده ─────────────────────────
def _arrays(kl):
    out = {k: np.asarray(kl[k], dtype=float) for k in ("o", "h", "l", "c", "v")}
    out["t"] = np.asarray(kl["t"], dtype=np.int64)
    return out


def side_series(kl, first, cs=None, knn_window=LIVE_BARS):
    """جهتِ تصمیم در هر کندلِ ``i ≥ first`` (۱ لانگ، −۱ شورت، ۰ صبر) و نامِ ستاپ.

    همان ترتیبِ ``engine.analyze``: ستاپِ کندلِ i، وگرنه i−1 (اولین ستاپِ یافته‌شده تصمیم را می‌بندد؛
    اگر قیمت از کندلِ ستاپ بیش از ۱٫۲ ATR خلافش رفته باشد، سیگنال باطل است و به قاعدهٔ z نمی‌رسد)،
    وگرنه ``trade_suggestion``: z≥1.2 و bull5≥3 و bear5≤1 (و قرینه) با رأیِ kNN علّی.
    ستاپ‌های «معلق‌شده» و «سیاستِ انتخابِ عمل» که به وضعیتِ زندهٔ سرور وابسته‌اند بازپخش نمی‌شوند.
    """
    a = _arrays(kl)
    o, h, l, c, v = a["o"], a["h"], a["l"], a["c"], a["v"]
    n = len(c)
    cs = cs if cs is not None else engine.component_series(o, h, l, c, v)
    lb = SETUP_LOOKBACK                              # = engine.SETUP_LOOKBACK: کندلِ i و قبلی‌ها
    first = max(int(first), lb - 1)
    ssig = np.zeros(n, dtype=np.int8)
    sname = [None] * n
    for j in range(first - (lb - 1), n):
        sg, st = engine.setup_signal(cs, o, h, l, c, j)
        if sg:
            ssig[j], sname[j] = sg, st
    side = np.zeros(n, dtype=np.int8)
    setup = [None] * n
    a14 = cs["a14"]
    for i in range(first, n):
        for back in range(lb):
            j = i - back
            if ssig[j]:
                if ssig[j] * (c[i] - c[j]) >= -1.2 * float(a14[i]):
                    side[i], setup[i] = ssig[j], sname[j]
                break
    b4 = ((cs["s_trend"] >= 0.30).astype(int) + (cs["s_mom"] >= 0.25) + (cs["s_vol"] >= 0.25)
          + (cs["s_struct"] >= 0.50))
    s4 = ((cs["s_trend"] <= -0.30).astype(int) + (cs["s_mom"] <= -0.25) + (cs["s_vol"] <= -0.25)
          + (cs["s_struct"] <= -0.50))
    z = cs["z"]
    live = np.arange(n) >= first
    free = live & (side == 0)
    need = np.flatnonzero(free & (np.abs(z) >= Z_ENTRY))   # جای دیگر رأیِ kNN نمی‌تواند تصمیم را عوض کند
    ml = np.zeros(n)
    if len(need):
        ml[need] = knn_series(h, l, c, cs, idx=need, window=knn_window)
    bull5 = b4 + (ml >= AI_VOTE)
    bear5 = s4 + (ml <= -AI_VOTE)
    side[free & (z >= Z_ENTRY) & (bull5 >= 3) & (bear5 <= 1)] = 1
    side[free & (z <= -Z_ENTRY) & (bear5 >= 3) & (bull5 <= 1)] = -1
    return side, setup, cs


def replay(kl, start_ts=None, first=None, cost_pct=COST_PCT, knn_window=LIVE_BARS,
           max_bars=bracket.MAX_BARS):
    """معامله‌های قاعدهٔ زنده روی تاریخچه — یک معامله در هر لحظه برای این ارز × تایم‌فریم.

    سیگنالِ کندلِ i با ``bracket.signal_trade`` (ورود در بازِ کندلِ i+1) داوری می‌شود؛ تا براکت بسته
    نشده سیگنالِ تازه نادیده گرفته می‌شود (کندلِ خروج خودش می‌تواند سیگنالِ بعدی بدهد). معامله‌ای که
    تا انتهای داده به نتیجه نرسیده شمرده نمی‌شود.
    """
    a = _arrays(kl)
    o, h, l, c, t = a["o"], a["h"], a["l"], a["c"], a["t"]
    n = len(c)
    if first is None:
        first = int(np.searchsorted(t, int(start_ts))) if start_ts is not None else 0
    first = max(int(first), LIVE_BARS - 1)
    trades = []
    if first >= n:
        return {"first": first, "side": np.zeros(n, dtype=np.int8), "setup": [None] * n,
                "trades": trades, "t": t}
    side, setup, cs = side_series(kl, first, knn_window=knn_window)
    free_from = first
    for i in np.flatnonzero(side[first:]) + first:
        i = int(i)
        if i < free_from:
            continue
        if i + 1 >= n:
            break
        sg = int(side[i])
        res = bracket.signal_trade(o, h, l, c, float(cs["a14"][i]), i, sg, max_bars)
        if res is None or (res["timed_out"] and i + max_bars > n - 1):
            break                                   # هنوز باز است — نتیجه‌اش معلوم نیست
        net = bracket.net_r(res["gross_r"], res["risk_pct"], cost_pct)
        trades.append({
            "i": i, "t": int(t[i]), "side": "long" if sg == 1 else "short", "setup": setup[i],
            "entry": float(res["entry"]), "sl": float(res["sl"]), "tp": float(res["tp"]),
            "risk_pct": float(res["risk_pct"]), "outcome": res["outcome"],
            "gross_r": float(res["gross_r"]), "net_r": float(net),
            "exit_idx": int(res["exit_idx"]), "exit_t": int(t[res["exit_idx"]]),
            "bars_held": int(res["bars_held"]),
        })
        free_from = int(res["exit_idx"])
    return {"first": first, "side": side, "setup": setup, "trades": trades, "t": t}


def verdict_code(n, avg_r, se):
    """``small`` (n<30) / ``loss`` (کلِ بازهٔ ۹۵٪ زیرِ صفر) / ``positive`` (کلش بالای صفر) / ``unclear``."""
    if not n or n < MIN_TRADES or avg_r is None or se is None:
        return "small"
    if avg_r + Z95 * se < 0:
        return "loss"
    if avg_r - Z95 * se > 0:
        return "positive"
    return "unclear"


def verdict(n, avg_r, se=None):
    """برچسبِ فارسیِ کارنامه (همان نگاشتِ ``VERDICT_FA``)."""
    return VERDICT_FA[verdict_code(n, avg_r, se)]


def _stats(rs):
    """آمارِ R خالص + خطای معیار (انحرافِ معیارِ نمونه/√n) و بازهٔ ۹۵٪ و برچسب."""
    rs = np.asarray(rs, dtype=float).reshape(-1)
    rs = rs[np.isfinite(rs)]
    n = len(rs)
    if not n:
        return {"n": 0, "win_rate": None, "avg_r": None, "profit_factor": None, "total_r": 0.0,
                "se": None, "ci_lo": None, "ci_hi": None,
                "verdict_code": "small", "verdict": VERDICT_FA["small"]}
    avg = float(rs.mean())
    se = float(rs.std(ddof=1) / math.sqrt(n)) if n >= 2 else None
    code = verdict_code(n, avg, se)
    gw = float(rs[rs > 0].sum())
    gl = float(-rs[rs <= 0].sum())
    return {"n": int(n), "win_rate": round(float((rs > 0).mean() * 100), 1),
            "avg_r": round(avg, 4),
            "profit_factor": round(gw / gl, 2) if gl > 0 else None,
            "total_r": round(float(rs.sum()), 2),
            "se": round(se, 4) if se is not None else None,
            "ci_lo": round(avg - Z95 * se, 4) if se is not None else None,
            "ci_hi": round(avg + Z95 * se, 4) if se is not None else None,
            "verdict_code": code, "verdict": VERDICT_FA[code]}


def summarize(trades, cost_pct=COST_PCT, t_from=None, t_to=None):
    rs = [x["net_r"] for x in trades]
    st = _stats(rs)
    gross = [x["gross_r"] for x in trades if x.get("gross_r") is not None]
    longs = [x["net_r"] for x in trades if x["side"] == "long"]
    shorts = [x["net_r"] for x in trades if x["side"] == "short"]
    st.update(
        long_n=len(longs), short_n=len(shorts),
        long_avg_r=round(float(np.mean(longs)), 4) if longs else None,
        short_avg_r=round(float(np.mean(shorts)), 4) if shorts else None,
        setup_n=sum(1 for x in trades if x.get("setup")),
        rule_n=sum(1 for x in trades if not x.get("setup")),
        # پیش از هزینه — تا روشن باشد چه‌قدر از زیان کارمزد است (در 15m حدضرر کوچک است و ۰٫۱۴٪ سنگین)
        avg_gross_r=round(float(np.mean(gross)), 4) if gross else None,
        avg_cost_r=round(float(np.mean([cost_pct / max(x["risk_pct"], 0.05) for x in trades])), 4)
        if trades else None,
        **{"from": _date(t_from), "to": _date(t_to)},
        from_ts=int(t_from) if t_from is not None else None,
        to_ts=int(t_to) if t_to is not None else None,
        window_days_actual=(round((int(t_to) - int(t_from)) / 86400000, 1)
                            if t_from is not None and t_to is not None else None),
        cost_pct=cost_pct,
    )
    return st


def scorecard_for(kl, window_days=WINDOW_DAYS, cost_pct=COST_PCT, knn_window=LIVE_BARS):
    """کارنامهٔ یک ارز × تایم‌فریم روی ``window_days`` روزِ آخرِ دادهٔ داده‌شده."""
    a = _arrays(kl)
    t = a["t"]
    if len(t) < MIN_BARS:
        raise ValueError(f"دادهٔ کافی نیست ({len(t)} کندل؛ دست‌کم {MIN_BARS} لازم است)")
    start_ts = int(t[-1]) - int(window_days) * 86400000
    s0 = int(np.searchsorted(t, start_ts))
    cut = max(0, s0 - WARMUP_BARS)                 # فقط برای سرعت: گرم‌شدن کافی است
    sub = {k: v[cut:] for k, v in a.items()}
    rep = replay(sub, start_ts=start_ts, cost_pct=cost_pct, knn_window=knn_window)
    ts = rep["t"]
    first = min(rep["first"], len(ts) - 1)
    out = summarize(rep["trades"], cost_pct, int(ts[first]), int(ts[-1]))
    out["bars"] = int(len(ts) - first)
    out["signal_bars"] = int(np.count_nonzero(rep["side"][first:]))
    out["maker"] = maker_variant(rep["trades"])
    return out


MAKER_KEYS = ("n", "win_rate", "avg_r", "profit_factor", "total_r", "se", "ci_lo", "ci_hi",
              "verdict_code", "verdict", "long_avg_r", "short_avg_r", "avg_cost_r", "cost_pct")


def maker_variant(trades, cost_pct=MAKER_COST_PCT):
    """گونهٔ ثانویِ «ورودِ میکر»: **همان** معامله‌ها (انتخابِ یک‌معامله‌در‌لحظه به هزینه بستگی ندارد)
    با هزینهٔ رفت‌وبرگشتِ کمتر؛ قراردادِ اصلیِ کارنامه (تیکر ۰٫۱۴٪) دست نمی‌خورد."""
    mk = [dict(x, net_r=bracket.net_r(x["gross_r"], x["risk_pct"], cost_pct)) for x in trades or []]
    st = summarize(mk, cost_pct)
    return {k: st.get(k) for k in MAKER_KEYS}


def _empty_maker():
    return maker_variant([])


# ───────────────────────── داده ─────────────────────────
def resample(kl, minutes):
    """تجمیعِ کندل‌ها به سطل‌های UTC هم‌ترازِ epoch (همان مرزِ کندل‌های 1h/4h/1d بایننس).

    سطلِ ناقصِ اول/آخر (هنوز بسته‌نشده) حذف می‌شود؛ سطلِ ناقصِ میانی (شکافِ داده) می‌ماند.
    """
    a = _arrays(kl)
    t = a["t"]
    if len(t) < 2:
        return a
    step = int(minutes) * 60000
    base = int(np.median(np.diff(t[:1000])))
    per = max(1, step // max(base, 1))
    b = (t // step) * step
    starts = np.flatnonzero(np.r_[True, b[1:] != b[:-1]])
    ends = np.r_[starts[1:], len(t)] - 1
    cnt = ends - starts + 1
    out = {"t": b[starts], "o": a["o"][starts], "c": a["c"][ends],
           "h": np.maximum.reduceat(a["h"], starts), "l": np.minimum.reduceat(a["l"], starts),
           "v": np.add.reduceat(a["v"], starts)}
    keep = np.ones(len(starts), dtype=bool)
    keep[0] = cnt[0] >= per
    keep[-1] = keep[-1] and cnt[-1] >= per
    return {k: v[keep] for k, v in out.items()}


def _load_local(sym, tf):
    """آرشیوِ بومیِ ``um_<SYM>_<tf>.npz`` (فیوچرزِ بایننس)؛ نبود ⇒ ``None``."""
    path = os.path.join(MICRO_DIR, f"um_{sym}_{tf}.npz")
    if not os.path.exists(path):
        return None
    with np.load(path) as z:
        t = np.asarray(z["t"], dtype=np.int64)
        arr = {k: np.asarray(z[k], dtype=float) for k in ("o", "h", "l", "c", "v")}
    t, uniq = np.unique(t, return_index=True)       # مرتب و بی‌تکرار
    out = {k: v[uniq] for k, v in arr.items()}
    out["t"] = t
    return out


def load_history(sym, tf):
    """طولانی‌ترین تاریخچهٔ موجود: آرشیوِ بومیِ فیوچرزِ همان تایم‌فریم (``um_<SYM>_<tf>.npz``)؛ اگر نبود،
    تجمیعِ آرشیوِ **ریزتری** که مضربش است (با برچسبِ «→tf»)؛ وگرنه تاریخچهٔ اسپاتِ شبکه.

    هرگز دادهٔ درشت‌تر با برچسبِ ریزتر برنمی‌گردد (قبلاً resample(15m, 5) کندلِ ۱۵ دقیقه‌ای را «5m» می‌نامید).
    جایگزینِ شبکه باید دست‌کم ``MIN_BARS`` کندل بدهد؛ وگرنه خطای روشن (برای همان خانه) —
    نه افتادنِ بی‌صدا به ۴۲۰ کندلِ زنده که هیچ معامله‌ای را داوری نمی‌کند.
    """
    m = tf_spec.minutes(tf)
    native = _load_local(sym, tf)
    if native is not None and len(native["t"]) > MIN_BARS:
        return native, SOURCE_ARCHIVE_FMT.format(tf=tf)
    finer = sorted((b for b in TFS if tf_spec.minutes(b) < m and m % tf_spec.minutes(b) == 0),
                   key=tf_spec.minutes, reverse=True)       # درشت‌ترینِ ریزترها اول (15m پیش از 5m)
    for b in finer:
        base = _load_local(sym, b)
        if base is not None and len(base["t"]) > 1000:
            return resample(base, m), SOURCE_RESAMPLED_FMT.format(base=b, tf=tf)
    import market                                   # فقط وقتی دادهٔ محلی نیست (تست‌ها loader می‌دهند)
    try:
        kl = market.get_history(sym, tf, bars=3000)
    except Exception as e:  # noqa: BLE001
        raise ValueError(f"آرشیوِ محلی نیست و تاریخچهٔ جایگزین ({SOURCE_FALLBACK}) در دسترس نیست: {e}") from e
    got = len(kl["t"]) if isinstance(kl, dict) and kl.get("t") is not None else 0
    if got < MIN_BARS:
        raise ValueError(f"آرشیوِ محلی نیست و تاریخچهٔ جایگزین ({SOURCE_FALLBACK}) فقط {got} کندل داد؛ "
                         f"دست‌کم {MIN_BARS} لازم است")
    return kl, SOURCE_FALLBACK


def build_scorecards(loader=None, symbols=SYMBOLS, tfs=TFS):
    """کارنامهٔ همهٔ ترکیب‌ها (بدونِ نوشتن روی دیسک)."""
    loader = loader or load_history
    t0 = time.time()
    cells = {}
    for sym in symbols:
        cells[sym] = {}
        for tf in tfs:
            t1 = time.time()
            src = None
            try:
                kl, src = loader(sym, tf)
                cell = scorecard_for(kl)
                cell["source"] = src
            except Exception as e:  # noqa: BLE001
                cell = _stats([])
                cell.update(error=str(e) or type(e).__name__, cost_pct=COST_PCT, source=src,
                            maker=_empty_maker())
            cell["elapsed_s"] = round(time.time() - t1, 2)
            cells[sym][tf] = cell
    out = {"version": SCORECARD_VERSION, "built_at": time.time(),
           "elapsed_s": round(time.time() - t0, 1), "cost_pct": COST_PCT,
           "maker_cost_pct": MAKER_COST_PCT,
           "window_days": WINDOW_DAYS, "knn_window": LIVE_BARS, "method": METHOD_FA,
           "cells": cells}
    out.update(_window_meta(cells))
    return out


def _window_meta(cells):
    """پنجره‌ای که واقعاً استفاده شد (نه ۷۳۰ِ اسمی): کمینهٔ شروع و بیشینهٔ پایانِ خانه‌های سالم."""
    ok = [c for row in cells.values() for c in row.values()
          if not c.get("error") and c.get("from_ts") is not None and c.get("to_ts") is not None]
    srcs = sorted({str(c.get("source")) for row in cells.values() for c in row.values()
                   if c.get("source")})
    if not ok:
        return {"window_from": None, "window_to": None, "window_days_actual": None,
                "window_days_min": None, "source": "، ".join(srcs) or None}
    lo = min(int(c["from_ts"]) for c in ok)
    hi = max(int(c["to_ts"]) for c in ok)
    return {"window_from": _date(lo), "window_to": _date(hi),
            "window_days_actual": round((hi - lo) / 86400000, 1),
            "window_days_min": min(float(c.get("window_days_actual") or 0) for c in ok),
            "source": "، ".join(srcs) or None}


def _fresh(sc, now):
    try:
        return bool(sc and sc.get("version") == SCORECARD_VERSION and sc.get("built_at")
                    and now - float(sc["built_at"]) < SCORECARD_TTL)
    except (TypeError, ValueError, AttributeError):
        return False


def scorecards(force=False, wait=True, loader=None, now=None):
    """کارنامهٔ کش‌شده؛ حداکثر روزی یک‌بار بازسازی می‌شود.

    ``wait=False`` (برای endpoint): اگر کش کهنه است، بازسازی در پس‌زمینه شروع می‌شود و همان کشِ
    موجود (یا خالی) با ``building=True`` برمی‌گردد تا درخواستِ کاربر یک دقیقه معطل نماند.
    """
    now = time.time() if now is None else now
    cached = _read_json(SCORECARD_PATH)
    if _fresh(cached, now) and not force:
        return cached
    if not wait:
        same = isinstance(cached, dict) and cached.get("version") == SCORECARD_VERSION
        # نسخهٔ قدیمی (بی‌بازهٔ اطمینان) نمایش داده نمی‌شود — برچسب‌هایش با قراردادِ فعلی نمی‌خواند
        out = dict(cached) if same else {"version": SCORECARD_VERSION, "built_at": None, "cells": {}}
        fail = _sc_failed
        if fail and now - fail[0] < RETRY_AFTER_FAIL_SEC and not force:
            out["building"] = False                 # ساختِ اخیر شکست خورد؛ تا مهلت دوباره تلاش نمی‌شود
            out["error"] = fail[1]
            return out
        _start_background(force, loader)
        out["building"] = True
        return out
    with _sc_lock:
        cached = _read_json(SCORECARD_PATH)
        if _fresh(cached, now) and not force:
            return cached                           # نخِ دیگری همین حالا ساخت
        out = build_scorecards(loader)
        _write_json(SCORECARD_PATH, out)
        return out


def _start_background(force, loader):
    global _sc_thread
    with _bg_lock:
        if _sc_thread is not None and _sc_thread.is_alive():
            return
        _sc_thread = threading.Thread(target=_bg_build, args=(force, loader),
                                      name="decision-scorecard", daemon=True)
        _sc_thread.start()


RETRY_AFTER_FAIL_SEC = 1800
_sc_failed = None                                   # (زمان، پیام) آخرین ساختِ شکست‌خورده


def _bg_build(force, loader):
    global _sc_failed
    try:
        scorecards(force=force, wait=True, loader=loader)
        _sc_failed = None
    except Exception as e:  # noqa: BLE001 — نخِ پس‌زمینه؛ پس از RETRY_AFTER_FAIL_SEC دوباره تلاش می‌شود
        _sc_failed = (time.time(), f"ساختِ کارنامه شکست خورد: {e}")
        log.exc("decision scorecard build")


# ───────────────────────── ۳) دفترِ رو به جلو (فقط افزودنی) ─────────────────────────
def ledger_id(sym, tf, candle_ts, side):
    return f"{sym}|{tf}|{int(candle_ts)}|{side}"


def _open_ok(r):
    """ردیفِ «open» که داوری/شمارشش ممکن است؛ ردیفِ ناقص/خراب نادیده گرفته می‌شود (هرگز استثنا نمی‌دهد)."""
    return (isinstance(r, dict) and r.get("kind") == "open" and isinstance(r.get("id"), str)
            and bool(r.get("id")) and isinstance(r.get("sym"), str) and bool(r.get("sym"))
            and r.get("tf") in TF_MINUTES and r.get("side") in ("long", "short")
            and _int(r.get("candle_ts")) is not None)


def record(decisions, now=None):
    """برای هر تصمیمِ لانگ/شورت روی یک کندلِ بسته یک ردیفِ «open» می‌افزاید (بی‌تکرار).

    خروجی: تعدادِ ردیف‌های تازه. تصمیمِ کهنه‌تر از ``STALE_BARS`` کندل (فیدِ مرده) ثبت نمی‌شود.
    """
    now = time.time() if now is None else now
    added = 0
    with _ledger_lock:
        seen = {r.get("id") for r in _read_jsonl(LEDGER_PATH) if r.get("kind") == "open"}
        for d in decisions or []:
            if not isinstance(d, dict):
                continue
            side = d.get("action")
            if side not in ("long", "short"):
                continue
            sym, tf, ts = d.get("sym"), d.get("tf"), _int(d.get("candle_ts"))
            entry, sl, atr = _num(d.get("entry")), _num(d.get("sl")), _num(d.get("atr"))
            if not (sym and tf in TF_MINUTES and ts is not None and entry and sl is not None and atr):
                continue
            bar_ms = TF_MINUTES[tf] * 60000
            if now * 1000 - int(ts) > STALE_BARS * bar_ms:
                continue
            rid = ledger_id(sym, tf, ts, side)
            if rid in seen:
                continue
            row = {"kind": "open", "id": rid, "sym": sym, "tf": tf, "side": side,
                   "candle_ts": int(ts), "entry": entry, "sl": sl, "tp": _num(d.get("tp")),
                   "atr": atr, "risk_pct": _num(d.get("risk_pct")),
                   "basis": d.get("basis"), "setup": d.get("setup"), "grade": d.get("grade"),
                   "strength": d.get("strength"), "z": d.get("z"),
                   "votes_bull": d.get("votes_bull"), "votes_bear": d.get("votes_bear"),
                   "cost_pct": COST_PCT, "recorded_at": round(now, 3)}
            _append(LEDGER_PATH, row)
            seen.add(rid)
            added += 1
    return added


def _resolve_one(op, kl, now):
    """داوریِ یک ردیفِ باز با همان قراردادِ bracket. ``None`` یعنی هنوز زود است.

    ورود فقط روی بازِ **دقیقاً** کندلِ بعد از کندلِ سیگنال (``candle_ts + یک کندل``) — مثلِ کارنامه.
    اگر آن کندل در داده نیست (شکاف، یا پنجرهٔ داده از آن گذشته) ردیف با ``skipped="no_data"`` بسته می‌شود؛
    ردیفِ خراب (ATR نامعتبر) با ``skipped="bad_row"``.
    """
    sig = 1 if op["side"] == "long" else -1
    ts = int(op["candle_ts"])
    bar_ms = TF_MINUTES[op["tf"]] * 60000
    base = {"kind": "result", "id": op["id"], "sym": op["sym"], "tf": op["tf"], "side": op["side"],
            "candle_ts": ts, "resolved_at": round(now, 3)}
    atr = _num(op.get("atr"))
    if not atr or atr <= 0:
        base["skipped"] = "bad_row"
        return base
    t = [int(x) for x in kl["t"]]
    start = next((i for i, x in enumerate(t) if x > ts), None)
    if start is None:
        return None                                 # کندلِ بعد هنوز بسته نشده
    if t[start] != ts + bar_ms:
        base["skipped"] = "no_data"                 # کندلِ ورود (سیگنال + یک کندل) در داده نیست
        return base
    entry = float(kl["o"][start])                   # ورود = بازِ کندلِ بعد، مثلِ کارنامه
    lv = bracket.levels(entry, atr, sig)
    res = bracket.resolve_path(kl["o"], kl["h"], kl["l"], kl["c"], start, sig,
                               lv["entry"], lv["sl"], lv["tp"])
    if res["timed_out"] and len(t) - start < bracket.MAX_BARS:
        return None                                 # هنوز به سقفِ زمانی نرسیده و چیزی نخورده
    cost = _num(op.get("cost_pct"))
    cost = COST_PCT if cost is None else cost
    base.update(entry=lv["entry"], sl=lv["sl"], tp=lv["tp"], risk_pct=round(lv["risk_pct"], 4),
                outcome=res["outcome"], bars_held=res["bars_held"],
                gross_r=round(res["gross_r"], 4),
                net_r=round(bracket.net_r(res["gross_r"], lv["risk_pct"], cost), 4),
                exit_ts=t[res["exit_idx"]])
    return base


def _done_ids(rows):
    return {r.get("id") for r in rows if r.get("kind") == "result" and r.get("id")}


def resolve(get_klines, now=None):
    """ردیف‌های بازِ دفتر را با کندل‌های بسته‌شدهٔ بعدی داوری می‌کند و ردیفِ «result» می‌افزاید.

    هیچ خطی بازنویسی نمی‌شود. ردیفِ ناقص/خراب نادیده گرفته می‌شود و هیچ‌وقت استثنا بیرون نمی‌آید
    (جز خطای نوشتنِ دیسک). چک و افزودن زیرِ یک قفل‌اند: «داوری‌شده؟» زیرِ قفل دوباره خوانده می‌شود،
    پس دو فراخوانیِ هم‌زمان نتیجهٔ تکراری نمی‌افزایند. خروجی: تعدادِ ردیف‌های تازه داوری‌شده.
    """
    now = time.time() if now is None else now
    with _ledger_lock:
        rows = _read_jsonl(LEDGER_PATH)
    done = _done_ids(rows)
    pending, seen = {}, set()
    for r in rows:
        if not _open_ok(r) or r["id"] in done or r["id"] in seen:
            continue
        seen.add(r["id"])
        pending.setdefault((r["sym"], r["tf"]), []).append(r)
    added = 0
    for (sym, tf), ops in pending.items():
        try:
            kl = get_klines(sym, tf)
        except Exception:  # noqa: BLE001 — دفعهٔ بعد
            log.exc("decision.resolve get_klines", sym=sym, tf=tf)
            continue
        found = []
        for op in ops:
            try:
                res = _resolve_one(op, kl, now)
            except Exception:  # noqa: BLE001 — یک ردیف/دادهٔ خراب نباید بقیه را بیندازد
                log.exc("decision.resolve row", id=op.get("id"))
                continue
            if res is not None:
                found.append(res)
        if not found:
            continue
        with _ledger_lock:                          # چک + افزودن اتمی
            done_now = _done_ids(_read_jsonl(LEDGER_PATH))
            for res in found:
                if res["id"] in done_now:
                    continue                        # فراخوانیِ دیگری همین حالا داوری کرد
                _append(LEDGER_PATH, res)
                done_now.add(res["id"])
                added += 1
    return added


def _cell_stats(opens, results):
    """یک معامله در هر لحظه، مثلِ کارنامه: سیگنالی که پیش از خروجِ معاملهٔ قبلی آمده شمرده نمی‌شود."""
    taken, open_n, overlap = [], 0, 0
    blocked_until = -math.inf
    for op in sorted(opens, key=lambda r: _int(r["candle_ts"])):
        ts = _int(op["candle_ts"])
        if ts < blocked_until:
            overlap += 1
            continue
        res = results.get(op["id"])
        if res is None:
            open_n += 1
            blocked_until = math.inf               # خروجِ این یکی هنوز معلوم نیست
            continue
        net = _num(res.get("net_r"))
        if res.get("skipped") or net is None:
            continue
        taken.append(net)
        ex = _int(res.get("exit_ts"))
        blocked_until = ex if ex is not None else ts
    st = _stats(taken)
    st.update(open=open_n, signals=len(opens), overlapping=overlap)
    return st, taken


def _empty_live():
    st = _stats([])
    st.update(open=0, signals=0, overlapping=0)
    return st


def live_stats():
    """کارنامهٔ زندهٔ دفتر (بی‌تکرار بر اساسِ id) برای هر ارز × تایم‌فریم و کل.

    ردیفِ ناقص/خراب نادیده گرفته می‌شود؛ هیچ‌وقت استثنا بیرون نمی‌آید.
    """
    try:
        rows = _read_jsonl(LEDGER_PATH)
    except Exception:  # noqa: BLE001
        log.exc("decision.live_stats read")
        rows = []
    opens, results = {}, {}
    for r in rows:
        if _open_ok(r):
            opens.setdefault(r["id"], r)
        elif r.get("kind") == "result" and isinstance(r.get("id"), str):
            results.setdefault(r["id"], r)
    by = {}
    for op in opens.values():
        by.setdefault((op["sym"], op["tf"]), []).append(op)
    cells, all_r, all_open, all_sig = {}, [], 0, 0
    for (sym, tf), ops in by.items():
        try:
            st, taken = _cell_stats(ops, results)
        except Exception:  # noqa: BLE001 — یک خانهٔ خراب نباید کلِ کارنامه را بیندازد
            log.exc("decision.live_stats cell", sym=sym, tf=tf)
            continue
        cells.setdefault(sym, {})[tf] = st
        all_r += taken
        all_open += st["open"]
        all_sig += st["signals"]
    overall = _stats(all_r)
    overall.update(open=all_open, signals=all_sig)
    return {"cells": cells, "overall": overall}


# ───────────────────────── ۴) خروجیِ endpoint ─────────────────────────
def payload(decisions, scorecard=None, live=None):
    """بدنهٔ ``GET /api/decisions``: هر خانه = تصمیم + کارنامهٔ دوساله + کارنامهٔ زنده."""
    sc = scorecard if scorecard is not None else scorecards(wait=False)
    lv = live if live is not None else live_stats()
    sc = sc if isinstance(sc, dict) else {}
    lv = lv if isinstance(lv, dict) else {}
    sc_cells = sc.get("cells") or {}
    lv_cells = lv.get("cells") or {}
    cells = {sym: {} for sym in SYMBOLS}
    for d in decisions or []:
        sym, tf = d.get("sym"), d.get("tf")
        if not sym or not tf:
            continue
        cell = dict(d)
        cell["scorecard"] = (sc_cells.get(sym) or {}).get(tf)
        cell["live"] = (lv_cells.get(sym) or {}).get(tf) or _empty_live()
        cells.setdefault(sym, {})[tf] = cell
    meta = {k: sc.get(k) for k in ("built_at", "window_from", "window_to", "window_days_actual",
                                    "window_days_min", "source", "window_days", "knn_window", "method")}
    meta["building"] = bool(sc.get("building"))
    meta["cost_pct"] = sc.get("cost_pct") if sc.get("cost_pct") is not None else COST_PCT
    meta["maker_cost_pct"] = (sc.get("maker_cost_pct") if sc.get("maker_cost_pct") is not None
                              else MAKER_COST_PCT)
    return {
        "generated_at": int(time.time()),
        "coins": list(SYMBOLS), "tfs": list(TFS),
        "cells": cells,
        "scorecard_meta": meta,
        "live_overall": lv.get("overall"),
        "note": NOTE_FA,
    }


def refresh(get_analysis, get_klines=None, scorecard=None):
    """مسیرِ کاملِ endpoint: تصمیم‌ها → ثبت در دفتر → داوریِ ردیف‌های باز → payload.

    دفترِ خراب یا دیسکِ پر هرگز نمایشِ تصمیم را نمی‌اندازد. ``scorecard`` فقط برای تست/فراخواننده‌ای
    که کارنامه را از قبل دارد؛ پیش‌فرض کشِ روزانه (بازسازی در پس‌زمینه).
    """
    decisions = collect(get_analysis)
    try:
        record(decisions)
    except Exception:  # noqa: BLE001
        log.exc("decision.refresh record")
    if get_klines is not None:
        try:
            resolve(get_klines)
        except Exception:  # noqa: BLE001
            log.exc("decision.refresh resolve")
    return payload(decisions, scorecard=scorecard, live=live_stats())
