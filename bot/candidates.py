# -*- coding: utf-8 -*-
"""دفترِ همهٔ کاندیدها — اندازه‌گیریِ صادقانه، مستقل از اینکه معامله شد یا نه.

لاگِ سایهٔ قبلی (``shadow.py``) فقط ردیف‌هایی را ثبت می‌کرد که **همین حالا**
قابلِ‌معامله بودند، و جیبِ زنده هم فقط از همان ردیف‌ها ساخته می‌شد. نتیجه یک
بن‌بست بود: تا وقتی مدلی معتبر نبود هیچ ردیفی ثبت نمی‌شد، و تا وقتی ردیفی ثبت
نمی‌شد هیچ لبه‌ای قابلِ کشف نبود (از ۱۶ جولای تا ۱۱ سپتامبر صفر رکورد).

اینجا **هر** ستاپِ دیده‌شده ثبت می‌شود، همراه با وضعیتِ تک‌تکِ گیت‌هایی که جلویش
را گرفتند. پس می‌توان بعداً پرسید «اگر این گیت نبود چه می‌شد؟» بدونِ اینکه یک
دلار ریسک شده باشد.

قرارداد اندازه‌گیری دقیقاً همان قراردادِ برچسبِ آموزش است (``bracket.py``):
ورود روی **بازِ کندلِ بعد**، ۱R از ATRِ کندلِ سیگنال، هدف 1.8R، سقف ۴۰ کندل.
برای همین به‌جای قیمتِ لحظهٔ نمایش، ``atr14`` ذخیره می‌شود تا داوری بعداً دقیقاً
مثل برچسب بازسازی شود.

دو فایلِ فقط-افزودنی:
* ``candidates.jsonl`` — یک خط به ازای هر کاندید
* ``candidate_results.jsonl`` — یک خط به ازای هر داوری
"""
import hashlib
import json
import math
import os
import threading
import time

import bracket
import stats as statsmod   # «stats» نامِ تابعِ همین ماژول است

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
CAND_PATH = os.path.join(DATA_DIR, "candidates.jsonl")
RESULT_PATH = os.path.join(DATA_DIR, "candidate_results.jsonl")

TF_MINUTES = {"15m": 15, "1h": 60, "4h": 240, "1d": 1440}
MAX_CANDLE_AGE_BARS = 3        # کندلِ کهنه‌تر = فیدِ مرده
MIN_RISK_PCT = 0.05            # زیر این مقدار، R بی‌معنا می‌شود

GATE_FIELDS = (
    "viable", "tradeable", "gate_allowed", "authority",
    "policy_trusted", "policy_pass", "regime_veto", "tf_suspended",
    "combo_suspended", "market_rank_pct", "market_rank_enforced",
    "drift_reject", "meta_approved", "meta_blocks", "signal_score",
)

_lock = threading.Lock()
_seen = None                   # مجموعهٔ شناسه‌های ثبت‌شده (کشِ حافظه)


def candidate_id(symbol, tf, side, candle_ts):
    raw = f"{symbol}|{tf}|{side}|{int(candle_ts)}"
    return hashlib.sha1(raw.encode()).hexdigest()[:16]


def _append(path, row):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _read(path):
    if not os.path.exists(path):
        return []
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except ValueError:
                continue          # خطِ نیمه‌نوشته (قطعِ برق) — بقیه معتبرند
    return out


def _load_seen():
    global _seen
    if _seen is None:
        _seen = {r["id"] for r in _read(CAND_PATH) if "id" in r}
    return _seen


def log_candidate(row, tf, extra=None):
    """ثبتِ یک کاندید. ``row`` همان دیکشنریِ ردیفِ overview است.

    خروجی: ``True`` اگر تازه ثبت شد. ثبت هیچ ربطی به مجوزِ معامله ندارد.
    """
    symbol, side, candle_ts = row.get("symbol"), row.get("side"), row.get("zt")
    if not (symbol and side and candle_ts):
        return False
    entry, sl = row.get("entry"), row.get("sl")
    atr14 = row.get("atr14")
    if entry is None or sl is None or not atr14:
        return False
    risk_pct = abs(float(entry) - float(sl)) / max(abs(float(entry)), 1e-12) * 100
    if risk_pct < MIN_RISK_PCT or not math.isfinite(risk_pct):
        return False
    bar_ms = TF_MINUTES.get(tf, 60) * 60000
    if time.time() * 1000 - float(candle_ts) > MAX_CANDLE_AGE_BARS * bar_ms:
        return False                      # فیدِ مرده
    cid = candidate_id(symbol, tf, side, candle_ts)
    with _lock:
        seen = _load_seen()
        if cid in seen:
            return False
        rec = {
            "id": cid, "symbol": symbol, "tf": tf, "side": side,
            "setup": row.get("setup") or "zx",
            "candle_ts": int(candle_ts), "logged_at": time.time(),
            "atr14": float(atr14),
            "proposed_entry": float(entry),
            "risk_pct": round(risk_pct, 4),
            # ⚠️ ``or`` اینجا اشتباه است: هزینهٔ صفر (تستِ بدونِ کارمزد) falsy است
            "cost_pct": float(row["cost"] if row.get("cost") is not None else 0.15),
            "p_win": row.get("p_win"), "ev_pct": row.get("ev_pct"),
            "regime": row.get("regime"),
            "gate_state": {k: row.get(k) for k in GATE_FIELDS if k in row},
        }
        if extra:
            rec["gate_state"].update(extra)
        _append(CAND_PATH, rec)
        seen.add(cid)
        return True


def _resolved_ids():
    return {r["id"] for r in _read(RESULT_PATH) if "id" in r}


def resolve(get_klines, limit=None):
    """داوریِ کاندیدهای بازِ با کندل‌های واقعی. خروجی: تعداد ردیفِ تازه داوری‌شده."""
    with _lock:
        pending = [c for c in _read(CAND_PATH) if c.get("id") not in _resolved_ids()]
    if limit:
        pending = pending[:limit]
    done = 0
    for c in pending:
        try:
            kl = get_klines(c["symbol"], c["tf"])
        except Exception:  # noqa: BLE001 — نماد در دسترس نیست؛ دفعهٔ بعد
            continue
        res = resolve_one(c, kl)
        if res is None:
            continue
        with _lock:
            _append(RESULT_PATH, res)
        done += 1
    return done


def resolve_one(cand, kl):
    """داوریِ یک کاندید با همان قراردادِ برچسبِ آموزش. ``None`` یعنی هنوز زود است."""
    sig = 1 if cand["side"] == "long" else -1
    ts = int(cand["candle_ts"])
    start = next((i for i, t in enumerate(kl["t"]) if t > ts), None)
    if start is None:
        return None                       # هنوز کندلِ بعدی بسته نشده
    entry = float(kl["o"][start])          # ورود = بازِ کندلِ بعد، دقیقاً مثل برچسب
    lv = bracket.levels(entry, cand["atr14"], sig)
    if lv["risk_pct"] < MIN_RISK_PCT:
        return {"id": cand["id"], "resolved_at": time.time(), "skipped": "zero_risk"}
    available = len(kl["c"]) - start
    res = bracket.resolve_path(kl["o"], kl["h"], kl["l"], kl["c"], start, sig,
                               lv["entry"], lv["sl"], lv["tp"])
    if res["timed_out"] and available < bracket.MAX_BARS:
        return None                       # هنوز به حدِ زمانی نرسیده و باری نخورده
    cost_pct = cand["cost_pct"] if cand.get("cost_pct") is not None else 0.15
    net = bracket.net_r(res["gross_r"], lv["risk_pct"], cost_pct)
    return {
        "id": cand["id"], "symbol": cand["symbol"], "tf": cand["tf"],
        "side": cand["side"], "setup": cand.get("setup") or "zx",
        "candle_ts": ts, "resolved_at": time.time(),
        "entry": lv["entry"], "sl": lv["sl"], "tp": lv["tp"],
        "risk_pct": round(lv["risk_pct"], 4),
        "outcome": res["outcome"], "bars_held": res["bars_held"],
        "gross_r": round(res["gross_r"], 4),
        "net_r": round(net, 4),
        "was_tradeable": bool((cand.get("gate_state") or {}).get("tradeable")),
        "blocking_gates": _blocking_gates(cand.get("gate_state") or {}),
    }


def _blocking_gates(gs):
    """کدام گیت‌ها جلوی این کاندید را گرفتند — برای پاسخ به «اگر این گیت نبود چه می‌شد؟»."""
    out = []
    if not gs.get("gate_allowed", True):
        out.append("gates_allowlist")
    if gs.get("policy_trusted") is False:
        out.append("policy_untrusted")
    elif gs.get("policy_pass") is False:
        out.append("policy_reject")
    if gs.get("regime_veto"):
        out.append("regime_veto")
    if gs.get("tf_suspended") is not None:
        out.append("tf_suspended")
    if gs.get("combo_suspended"):
        out.append("combo_suspended")
    if gs.get("drift_reject"):
        out.append("price_drift")
    if gs.get("meta_approved") is False:
        out.append("meta_gate")
    if gs.get("market_rank_enforced") and (gs.get("market_rank_pct") or 0) < 60:
        out.append("cross_section_rank")
    return out


def stats(tf=None, days=30, only_tradeable=False, setup=None, side=None):
    """آمارِ خالصِ کاندیدها. پیش‌فرض: **همهٔ** کاندیدها، نه فقط آن‌هایی که مجوز گرفتند."""
    cutoff = (time.time() - days * 86400) * 1000
    rows = [r for r in _read(RESULT_PATH)
            if r.get("net_r") is not None and r.get("candle_ts", 0) >= cutoff]
    if tf:
        rows = [r for r in rows if r.get("tf") == tf]
    if setup:
        rows = [r for r in rows if r.get("setup") == setup]
    if side:
        rows = [r for r in rows if r.get("side") == side]
    if only_tradeable:
        rows = [r for r in rows if r.get("was_tradeable")]
    out = {"n": len(rows), "breakeven_win_rate": round(bracket.breakeven_win_rate() * 100, 1),
           "win_rate": None, "avg_net_r": None, "sum_net_r": None,
           "by_combo": {}, "by_blocking_gate": {}}
    if not rows:
        return out
    vals = [float(r["net_r"]) for r in rows]
    out["win_rate"] = round(sum(1 for v in vals if v > 0) / len(vals) * 100, 1)
    out["avg_net_r"] = round(sum(vals) / len(vals), 4)
    out["sum_net_r"] = round(sum(vals), 3)
    # کرانِ پایین با بوت‌استرپِ بلوکی و n مؤثر — همان معیاری که داورِ آزمونِ منجمد دارد
    stamps = [float(r.get("candle_ts") or 0) for r in rows]
    block = bracket.MAX_BARS * TF_MINUTES.get(tf or "1h", 60) * 60000
    out["lcb_net_r_90"] = statsmod.block_bootstrap_lcb(vals, stamps, block, alpha=0.10, B=500)
    out["n_eff"] = statsmod.effective_n(vals, stamps, block)
    for r in rows:
        key = f"{r['tf']}|{r['setup']}|{r['side']}"
        b = out["by_combo"].setdefault(key, {"n": 0, "wins": 0, "sum_r": 0.0})
        b["n"] += 1
        b["wins"] += 1 if float(r["net_r"]) > 0 else 0
        b["sum_r"] += float(r["net_r"])
        for g in (r.get("blocking_gates") or ["none"]):
            gb = out["by_blocking_gate"].setdefault(g, {"n": 0, "sum_r": 0.0})
            gb["n"] += 1
            gb["sum_r"] += float(r["net_r"])
    for b in out["by_combo"].values():
        b["win_rate"] = round(b["wins"] / b["n"] * 100, 1)
        b["avg_net_r"] = round(b["sum_r"] / b["n"], 4)
        b["sum_r"] = round(b["sum_r"], 3)
    for gb in out["by_blocking_gate"].values():
        gb["avg_net_r"] = round(gb["sum_r"] / gb["n"], 4)
        gb["sum_r"] = round(gb["sum_r"], 3)
    return out


def counts():
    """شمارشِ سریع برای health — بدونِ محاسبهٔ آمار."""
    cands = _read(CAND_PATH)
    results = _read(RESULT_PATH)
    return {
        "candidates": len(cands),
        "resolved": len(results),
        "open": len(cands) - len(results),
        "last_logged_at": max((c.get("logged_at") or 0 for c in cands), default=None),
        "last_resolved_at": max((r.get("resolved_at") or 0 for r in results), default=None),
    }


def _reset_cache():
    """فقط برای تست‌ها."""
    global _seen
    _seen = None
