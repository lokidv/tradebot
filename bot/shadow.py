# -*- coding: utf-8 -*-
"""ردیابی سایه — هر سیگنال قابل‌معامله‌ای که به کاربر نمایش داده می‌شود ثبت
و بعداً با قیمت واقعی داوری می‌شود (برد/باخت/حدزمانی). عملکرد زنده = آینه بدون تعارف."""
import json
import os
import threading
import time
import uuid

import bracket

SHADOW_PATH = os.path.join(os.path.dirname(__file__), "data", "signals_log.json")
MIN_RISK_PCT = 0.05        # زیر این فاصله، گردکردنِ قیمت حدضرر را روی ورود می‌آورد و R بی‌معنا می‌شود
MAX_ABS_R = 5.0            # هر |R| بزرگ‌تر، خطای داده است نه نتیجهٔ معامله
_lock = threading.Lock()


def _load():
    if not os.path.exists(SHADOW_PATH):
        return {"pending": [], "resolved": []}
    with open(SHADOW_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def _save(db):
    os.makedirs(os.path.dirname(SHADOW_PATH), exist_ok=True)
    tmp = SHADOW_PATH + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(db, f, ensure_ascii=False)
    os.replace(tmp, SHADOW_PATH)


TF_MINUTES = {"15m": 15, "1h": 60, "4h": 240, "1d": 1440}
# سیگنالِ کندلِ کهنه = فیدِ مرده؛ هرگز حل نمی‌شود و ثبت را قفل می‌کند. سه کندل
# سخاوتمندانه است (زمانِ ثبت نسبت به زمانِ *باز شدنِ* کندل سنجیده می‌شود) ولی
# ردیف‌های واقعیِ مسموم ۴۰ تا ۱۳۶۸ روز فاصله داشتند.
MAX_CANDLE_AGE_BARS = 3


def is_clean_row(r):
    """ردیفی که آماری معنادار دارد. ردیف‌های مسموم آمارِ زنده را وارونه می‌کردند:
    یک SHIBUSDT با entry==sl مقدار r_mult=997 ساخته بود و گیت‌های تعلیق را خاموش کرد."""
    try:
        entry, sl = float(r["entry"]), float(r["sl"])
    except (KeyError, TypeError, ValueError):
        return False
    risk_pct = abs(entry - sl) / max(abs(entry), 1e-12) * 100
    if risk_pct < MIN_RISK_PCT:
        return False
    rm = r.get("r_mult")
    return rm is not None and abs(float(rm)) <= MAX_ABS_R


def log_signal(symbol, tf, side, entry, sl, tp, setup, p_win, ev_pct, candle_ts, horizon_min,
               cost_pct=0.15):
    """ثبت سیگنال نمایش‌داده‌شده (بدون تکرار برای همان کندل/جهت)."""
    risk_pct = abs(float(entry) - float(sl)) / max(abs(float(entry)), 1e-12) * 100
    if risk_pct < MIN_RISK_PCT:
        return False                               # حدضرر روی ورود گرد شده — R بی‌معنا می‌شود
    bar_ms = TF_MINUTES.get(tf, 60) * 60000
    if candle_ts and time.time() * 1000 - float(candle_ts) > MAX_CANDLE_AGE_BARS * bar_ms:
        return False                               # کندلِ کهنه/فیدِ مرده
    with _lock:
        db = _load()
        for p in db["pending"]:
            if p["symbol"] == symbol and p["tf"] == tf and p["side"] == side:
                return False                       # هنوز یکی در جریان است
        for r in db["resolved"][-200:]:
            if r["symbol"] == symbol and r["tf"] == tf and r["ts"] == candle_ts and r["side"] == side:
                return False
        db["pending"].append({
            "id": uuid.uuid4().hex[:8], "ts": candle_ts, "logged_at": time.time(),
            "symbol": symbol, "tf": tf, "side": side, "setup": setup,
            "entry": entry, "sl": sl, "tp": tp,
            "p_win": p_win, "ev_pct": ev_pct,
            "cost_pct": float(cost_pct),
            "cost_r": float(cost_pct) / max(risk_pct, 0.05),
            "deadline": candle_ts + horizon_min * 60000,
        })
        _save(db)
        return True


def resolve(get_klines):
    """داوری سیگنال‌های معلق با کندل‌های واقعی بعدی."""
    with _lock:
        db = _load()
        if not db["pending"]:
            return 0
        done = 0
        still = []
        for p in db["pending"]:
            try:
                kl = get_klines(p["symbol"], p["tf"])
            except Exception:  # noqa: BLE001
                still.append(p)
                continue
            d = 1 if p["side"] == "long" else -1
            risk = abs(p["entry"] - p["sl"])
            if risk / max(abs(float(p["entry"])), 1e-12) * 100 < MIN_RISK_PCT:
                continue                            # ردیفِ مسموم — دور انداخته می‌شود
            # همان قواعدِ محافظه‌کارانهٔ براکتِ واحد (گپ، تساویِ SL/TP، حدِ زمانی)
            start = next((i for i, t in enumerate(kl["t"]) if t > p["ts"]), None)
            outcome = None
            if start is not None:
                deadline_idx = next(
                    (i for i, t in enumerate(kl["t"]) if i >= start and t >= p["deadline"]), None)
                last_idx = deadline_idx if deadline_idx is not None else len(kl["t"]) - 1
                res = bracket.resolve_path(
                    kl["o"], kl["h"], kl["l"], kl["c"], start, d,
                    p["entry"], p["sl"], p["tp"], max_bars=last_idx - start + 1)
                reached_end = res["exit_idx"] >= last_idx and res["timed_out"]
                if not res["timed_out"]:
                    label = "برد" if res["gross_r"] > 0 else "باخت"
                    outcome = (label, res["gross_r"])
                elif deadline_idx is not None or time.time() * 1000 > p["deadline"] + 86400000:
                    outcome = ("حدزمانی", res["gross_r"]) if reached_end else None
            if outcome:
                gross_r = float(outcome[1])
                if "cost_r" in p:
                    cost_r = float(p["cost_r"])
                else:
                    risk_pct = risk / max(abs(float(p["entry"])), 1e-12) * 100
                    cost_r = float(p.get("cost_pct", 0.15)) / max(risk_pct, 0.05)
                p["result"], p["gross_r"] = outcome[0], round(gross_r, 3)
                p["r_mult"] = round(gross_r - cost_r, 3)
                p["resolved_at"] = time.time()
                db["resolved"].insert(0, p)
                done += 1
            else:
                still.append(p)
        db["pending"] = still
        db["resolved"] = db["resolved"][:2000]
        if done:
            _save(db)
        return done


def stats(tf=None, days=30, since_ts=None):
    """since_ts (ثانیه): فقط سیگنال‌های ثبت‌شده بعد از این لحظه — برای قضاوتِ «دورهٔ مدلِ فعلی»
    (سیگنال‌های مدلِ قدیمیِ آلوده نباید مدلِ تازه را محکوم کنند)."""
    with _lock:
        db = _load()
    cutoff = (time.time() - days * 86400) * 1000
    if since_ts:
        cutoff = max(cutoff, float(since_ts) * 1000)
    rows = [r for r in db["resolved"]
            if r["ts"] >= cutoff and (tf is None or r["tf"] == tf) and is_clean_row(r)]
    pending = [p for p in db["pending"] if tf is None or p["tf"] == tf]
    out = {"n": len(rows), "pending": len(pending), "win_rate": None, "avg_r": None,
           "by_setup": {}, "by_combo": {}, "by_side": {}}
    if rows:
        wins = [r for r in rows if r["r_mult"] > 0]
        out["win_rate"] = round(len(wins) / len(rows) * 100, 1)
        out["avg_r"] = round(sum(r["r_mult"] for r in rows) / len(rows), 3)
        for r in rows:
            setup = r.get("setup") or "zx"
            side = r.get("side") or "?"
            tf_r = r.get("tf") or "?"
            d = out["by_setup"].setdefault(setup, {"n": 0, "wins": 0, "sum_r": 0.0})
            d["n"] += 1
            d["wins"] += 1 if r["r_mult"] > 0 else 0
            d["sum_r"] += float(r["r_mult"])
            for key in (f"{tf_r}|{setup}", f"{tf_r}|{setup}|{side}"):
                c = out["by_combo"].setdefault(key, {"n": 0, "wins": 0, "sum_r": 0.0})
                c["n"] += 1
                c["wins"] += 1 if r["r_mult"] > 0 else 0
                c["sum_r"] += float(r["r_mult"])
            sd = out["by_side"].setdefault(side, {"n": 0, "wins": 0, "sum_r": 0.0})
            sd["n"] += 1
            sd["wins"] += 1 if r["r_mult"] > 0 else 0
            sd["sum_r"] += float(r["r_mult"])
        for bucket in ("by_setup", "by_combo", "by_side"):
            for k, v in out[bucket].items():
                v["win_rate"] = round(v["wins"] / v["n"] * 100, 1)
                v["avg_r"] = round(v["sum_r"] / v["n"], 3)
                v["sum_r"] = round(v["sum_r"], 3)
    return out
