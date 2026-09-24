# -*- coding: utf-8 -*-
"""میزِ معاملهٔ کوکوین — ابزارهای کنترلِ ریسک و سنجشِ عملکردِ خودِ کاربر.

پژوهشِ ۱۵دقیقه‌ای (micro.py، ۲۵ آزمونِ ازپیش‌ثبت‌شده) هیچ سیگنالِ جهت‌داری با لبه پیدا نکرد؛ پس این‌جا
سیگنال نیست. چیزی هست که روی نتیجهٔ واقعی اثرِ مستقیم دارد: اندازهٔ درستِ پوزیشن و اهرم، کارمزد،
ساعت‌هایی که حرکت از هزینه بزرگ‌تر است، و دفتری که با عدد نشان می‌دهد روشِ خودِ کاربر لبه دارد یا نه.
هیچ اتصالی به حسابِ کاربر نیست و هیچ سفارشی داده نمی‌شود — فقط دادهٔ عمومی.
"""
import json
import os
import threading
import time
import uuid

import httpx
import numpy as np

import paths
import watchlist

FUT = "https://api-futures.kucoin.com"
CONTRACT = {"BTCUSDT": "XBTUSDTM", "ETHUSDT": "ETHUSDTM", "BNBUSDT": "BNBUSDTM",
            "SOLUSDT": "SOLUSDTM", "TRXUSDT": "TRXUSDTM"}
SPOT_FEE = 0.001                  # سطحِ پایهٔ اسپاتِ کوکوین (بدونِ ورود به حساب قابلِ خواندن نیست)
JOURNAL = paths.data("trade_journal.jsonl")
HOURS_CACHE = paths.data("research", "kucoin_hours.json")
TEHRAN_OFFSET_MIN = 210
_cache = {}
_lock = threading.Lock()


def contracts(get=None, ttl=3600):
    """مشخصاتِ قراردادهای فیوچرزِ پنج ارز از API عمومیِ کوکوین (یک ساعت کش)."""
    with _lock:
        hit = _cache.get("contracts")
        if hit and time.time() - hit[0] < ttl:
            return hit[1]
    get = get or (lambda u: httpx.get(u, timeout=20).json())
    rows = get(FUT + "/api/v1/contracts/active")["data"]
    by = {r["symbol"]: r for r in rows}
    out = {}
    for sym, csym in CONTRACT.items():
        r = by.get(csym)
        if not r:
            continue
        out[sym] = {"contract": csym, "multiplier": float(r["multiplier"]), "tick": float(r["tickSize"]),
                    "lot": float(r.get("lotSize") or 1), "mark": float(r.get("markPrice") or 0),
                    "maker": float(r["makerFeeRate"]), "taker": float(r["takerFeeRate"]),
                    "mmr": float(r.get("maintainMargin") or 0.004), "max_leverage": float(r.get("maxLeverage") or 0),
                    "funding_rate": float(r.get("fundingFeeRate") or 0)}
    with _lock:
        _cache["contracts"] = (time.time(), out)
    return out


def size_position(balance, risk_pct, entry, stop, spec, fee_side="taker", leverage=None):
    """حجم بر اساسِ ریسک: ضررِ تا حدضرر (به‌علاوهٔ کارمزدِ رفت‌وبرگشت) = risk_pct٪ موجودی.

    اهرم فقط مارجین را تعیین می‌کند، نه ریسک را. اهرمِ پیشنهادی کمترین اهرمی است که مارجین از
    موجودی بیشتر نشود؛ و اگر قیمتِ لیکوئید (تقریبِ ایزوله) پیش از حدضرر برسد، هشدار داده می‌شود.
    """
    side = 1 if stop < entry else -1
    dist = abs(entry - stop) / entry
    if dist <= 0:
        raise ValueError("حدضرر نباید با قیمتِ ورود برابر باشد")
    fee = spec[fee_side]
    risk_usdt = balance * risk_pct / 100.0
    notional = risk_usdt / (dist + 2 * fee)
    contracts_n = int(notional / (entry * spec["multiplier"]) // spec["lot"] * spec["lot"])
    notional = contracts_n * entry * spec["multiplier"]
    lev_needed = max(1.0, notional / balance) if balance > 0 else None
    lev = float(leverage) if leverage else float(np.ceil(lev_needed or 1))
    mmr = spec["mmr"]
    liq = entry * (1 - 1 / lev + mmr) if side == 1 else entry * (1 + 1 / lev - mmr)
    liq_before_stop = (liq >= stop) if side == 1 else (liq <= stop)
    fees = notional * 2 * fee
    loss_at_stop = notional * dist + fees
    return {"side": "long" if side == 1 else "short", "contracts": contracts_n, "notional": round(notional, 2),
            "margin": round(notional / lev, 2), "leverage": lev, "min_leverage": round(lev_needed or 0, 2),
            "liquidation": liq, "liq_before_stop": bool(liq_before_stop),
            "fees_round_trip": round(fees, 2), "loss_at_stop": round(loss_at_stop, 2),
            "loss_at_stop_pct": round(loss_at_stop / balance * 100, 3) if balance else None,
            "breakeven_move_pct": round(2 * fee * 100, 3), "stop_distance_pct": round(dist * 100, 3)}


def hours_table(force=False):
    """میانهٔ حرکتِ ۲ساعته به تفکیکِ ساعتِ روز (تهران) در برابرِ هزینهٔ taker — از دادهٔ ۱۵دقیقه‌ای."""
    if not force and os.path.exists(HOURS_CACHE):
        with open(HOURS_CACHE, encoding="utf-8") as f:
            return json.load(f)
    out = {"cost_taker_pct": 0.14, "cost_maker_pct": 0.04, "assets": {}}
    for sym in watchlist.SYMBOLS:
        p = paths.data("hist_research", "micro", f"um_{sym}_15m.npz")
        if not os.path.exists(p):
            continue
        d = np.load(p)
        t, c = d["t"], d["c"]
        last2y = t >= t[-1] - 730 * 86_400_000
        mv = np.full(len(c), np.nan)
        mv[:-8] = np.abs(c[8:] / c[:-8] - 1) * 100
        hour = (((t // 60000) + TEHRAN_OFFSET_MIN) // 60) % 24
        rows = []
        for hh in range(24):
            m = last2y & (hour == hh) & np.isfinite(mv)
            med = float(np.median(mv[m])) if m.any() else None
            rows.append({"hour_tehran": hh, "median_move_2h_pct": round(med, 3) if med else None,
                         "cost_share_taker_pct": round(0.14 / med * 100, 1) if med else None})
        out["assets"][sym] = rows
    os.makedirs(os.path.dirname(HOURS_CACHE), exist_ok=True)
    with open(HOURS_CACHE, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    return out


# ───────────────────────── دفترِ معاملاتِ واقعی ─────────────────────────
def _read():
    if not os.path.exists(JOURNAL):
        return []
    rows = []
    with open(JOURNAL, encoding="utf-8") as f:
        for line in f:
            try:
                rows.append(json.loads(line))
            except ValueError:
                continue
    deleted = {r["id"] for r in rows if r.get("deleted")}
    return [r for r in rows if not r.get("deleted") and r.get("id") not in deleted]


def add_trade(t):
    """یک معاملهٔ بسته‌شده. pnl خالص از قیمت‌ها، حجم و کارمزد حساب می‌شود؛ R اگر حدضرر داده شده باشد."""
    side = 1 if t["side"] == "long" else -1
    entry, exit_, qty = float(t["entry"]), float(t["exit"]), float(t["notional"]) / float(t["entry"])
    fees = float(t.get("fees") or 0)
    pnl = side * (exit_ - entry) * qty - fees
    stop = t.get("stop")
    risk = abs(entry - float(stop)) * qty if stop not in (None, "", 0) else None
    row = {"id": uuid.uuid4().hex[:10], "logged_at": time.time(), "sym": t["sym"], "side": t["side"],
           "market": t.get("market", "futures"), "entry": entry, "exit": exit_, "notional": float(t["notional"]),
           "fees": fees, "stop": float(stop) if risk else None, "pnl": round(pnl, 4),
           "r": round(pnl / risk, 3) if risk else None, "setup": (t.get("setup") or "").strip()[:40],
           "opened_at": int(t.get("opened_at") or time.time() * 1000),
           "closed_at": int(t.get("closed_at") or time.time() * 1000)}
    os.makedirs(os.path.dirname(JOURNAL), exist_ok=True)
    with open(JOURNAL, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return row


def delete_trade(tid):
    """حذف با ثبتِ یک ردیفِ «حذف‌شده» — فایل فقط-افزودنی می‌ماند."""
    with open(JOURNAL, "a", encoding="utf-8") as f:
        f.write(json.dumps({"id": tid, "deleted": True, "logged_at": time.time()}) + "\n")


def _group(rows, key):
    g = {}
    for r in rows:
        g.setdefault(key(r), []).append(r)
    return {k: {"n": len(v), "win_rate": round(sum(x["pnl"] > 0 for x in v) / len(v) * 100, 1),
                "pnl": round(sum(x["pnl"] for x in v), 2)} for k, v in g.items()}


def journal_stats(daily_limit_pct=None, balance=None, rows=None):
    rows = sorted(_read() if rows is None else rows, key=lambda r: r["closed_at"])
    out = {"trades": rows[-200:][::-1], "n": len(rows)}
    if not rows:
        return out
    pnl = np.asarray([r["pnl"] for r in rows])
    wins, losses = pnl[pnl > 0], pnl[pnl <= 0]
    eq = np.concatenate([[0.0], np.cumsum(pnl)])
    rs = [r["r"] for r in rows if r.get("r") is not None]
    out.update({
        "total_pnl": round(float(pnl.sum()), 2), "fees": round(sum(r["fees"] for r in rows), 2),
        "win_rate": round(float((pnl > 0).mean()) * 100, 1),
        "avg_win": round(float(wins.mean()), 2) if len(wins) else None,
        "avg_loss": round(float(losses.mean()), 2) if len(losses) else None,
        "expectancy": round(float(pnl.mean()), 2),
        "avg_r": round(float(np.mean(rs)), 3) if rs else None,
        "profit_factor": round(float(wins.sum() / -losses.sum()), 2) if len(losses) and losses.sum() < 0 else None,
        "max_drawdown": round(float(np.max(np.maximum.accumulate(eq) - eq)), 2),
        "equity": [round(float(x), 2) for x in eq[-300:]],
        "by_coin": _group(rows, lambda r: r["sym"]),
        "by_hour_tehran": _group(rows, lambda r: int(((r["opened_at"] // 60000 + TEHRAN_OFFSET_MIN) // 60) % 24)),
        "by_setup": _group(rows, lambda r: r["setup"] or "—"),
        "by_side": _group(rows, lambda r: r["side"]),
    })
    today = time.strftime("%Y-%m-%d", time.gmtime(time.time() + TEHRAN_OFFSET_MIN * 60))
    tpnl = sum(r["pnl"] for r in rows
               if time.strftime("%Y-%m-%d", time.gmtime(r["closed_at"] / 1000 + TEHRAN_OFFSET_MIN * 60)) == today)
    out["today_pnl"] = round(tpnl, 2)
    if daily_limit_pct and balance:
        out["daily_limit_hit"] = tpnl <= -balance * daily_limit_pct / 100
    # آیا هنوز داده برای قضاوت کافی است؟ با کمتر از ۳۰ معامله هر عددی عمدتاً شانس است
    out["enough_data"] = len(rows) >= 30
    if len(rows) >= 30:
        se = pnl.std(ddof=1) / np.sqrt(len(pnl))
        out["expectancy_lcb90"] = round(float(pnl.mean() - 1.2816 * se), 2)
    return out
