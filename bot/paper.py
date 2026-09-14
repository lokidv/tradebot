# -*- coding: utf-8 -*-
"""پوزیشن‌های دمو (شبیه‌سازی محلی) — ذخیره در فایل JSON، بستن خودکار در هدف/حدضرر/حد زمانی."""
import json
import os
import threading
import uuid
from datetime import datetime, timezone

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
PATH = os.path.join(DATA_DIR, "positions.json")
_lock = threading.RLock()


def _now():
    return datetime.now(timezone.utc).isoformat()


DEFAULT_BALANCE = 10000.0
DEFAULT_COST_PCT = 0.15


def _load():
    if not os.path.exists(PATH):
        return {"open": [], "closed": [], "wallet": {"start": DEFAULT_BALANCE, "reset_at": _now()}}
    with open(PATH, "r", encoding="utf-8") as f:
        db = json.load(f)
    db.setdefault("wallet", {"start": DEFAULT_BALANCE, "reset_at": _now()})
    return db


def _save(db):
    os.makedirs(os.path.dirname(PATH), exist_ok=True)
    tmp = PATH + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(db, f, ensure_ascii=False, indent=1)
    os.replace(tmp, PATH)


def list_positions():
    with _lock:
        return _load()


def open_position(symbol, tf, side, entry, sl, tp, size_usdt, time_stop_min, grade="", mode="local", qty=None,
                  opened_by="user", cost_pct=DEFAULT_COST_PCT):
    entry, sl, tp, size_usdt = map(float, (entry, sl, tp, size_usdt))
    if side not in ("long", "short") or min(entry, sl, tp, size_usdt) <= 0:
        raise ValueError("پارامترهای پوزیشن نامعتبر است")
    if (side == "long" and not sl < entry < tp) or (side == "short" and not tp < entry < sl):
        raise ValueError("حدضرر/هدف با جهت معامله سازگار نیست")
    pos = {
        "id": uuid.uuid4().hex[:10],
        "symbol": symbol, "tf": tf, "side": side, "grade": grade,
        "entry": entry, "sl": sl, "tp": tp, "size_usdt": size_usdt,
        "opened_at": _now(), "time_stop_min": time_stop_min,
        "last_price": entry, "pnl_pct": 0.0, "pnl_usdt": 0.0,
        "mode": mode, "qty": qty, "opened_by": opened_by,
        "cost_pct": max(float(cost_pct), 0.0),
        "r0": abs(entry - sl),                 # ریسکِ اولیه (برای محاسبهٔ R حتی بعد از جابه‌جاییِ حدضرر)
    }
    with _lock:
        db = _load()
        db["open"].append(pos)
        _save(db)
    return pos


def _pnl(pos, price):
    d = 1 if pos["side"] == "long" else -1
    gross_pct = d * (price / pos["entry"] - 1) * 100
    pct = gross_pct - float(pos.get("cost_pct", DEFAULT_COST_PCT))
    return pct, pos["size_usdt"] * pct / 100


def _close(pos, price, reason):
    pct, usd = _pnl(pos, price)
    d = 1 if pos["side"] == "long" else -1
    gross_pct = d * (price / pos["entry"] - 1) * 100
    pos.update(closed_at=_now(), exit_price=price, pnl_pct=round(pct, 3),
               gross_pnl_pct=round(gross_pct, 3),
               cost_usdt=round(pos["size_usdt"] * float(pos.get("cost_pct", DEFAULT_COST_PCT)) / 100, 3),
               pnl_usdt=round(usd, 3), close_reason=reason, last_price=price)
    return pos


def refresh(prices):
    """به‌روزرسانی با قیمت‌های جدید؛ بستن خودکار در هدف/حدضرر/حد زمانی. prices: dict symbol->price"""
    with _lock:
        db = _load()
        still = []
        for pos in db["open"]:
            if pos.get("mode") == "testnet":     # چرخه تست‌نت را صرافی مدیریت می‌کند، نه شبیه‌ساز محلی
                still.append(pos)
                continue
            price = prices.get(pos["symbol"])
            if price is None:
                still.append(pos)
                continue
            d = 1 if pos["side"] == "long" else -1
            opened = datetime.fromisoformat(pos["opened_at"])
            age_min = (datetime.now(timezone.utc) - opened).total_seconds() / 60
            if (d == 1 and price <= pos["sl"]) or (d == -1 and price >= pos["sl"]):
                # اگر بین دو نمونهٔ قیمت گپ رخ داده باشد، خروج را خوش‌بینانه روی SL فرض نکن.
                exit_price = min(price, pos["sl"]) if d == 1 else max(price, pos["sl"])
                db["closed"].insert(0, _close(pos, exit_price, "حدضرر"))
            elif (d == 1 and price >= pos["tp"]) or (d == -1 and price <= pos["tp"]):
                db["closed"].insert(0, _close(pos, pos["tp"], "هدف ✅"))
            elif age_min >= pos["time_stop_min"]:
                db["closed"].insert(0, _close(pos, price, "حد زمانی"))
            else:
                pct, usd = _pnl(pos, price)
                pos.update(last_price=price, pnl_pct=round(pct, 3), pnl_usdt=round(usd, 3))
                still.append(pos)
        db["open"] = still
        db["closed"] = db["closed"][:200]
        _save(db)
        return db


def close_manual(pos_id, price):
    with _lock:
        db = _load()
        for i, pos in enumerate(db["open"]):
            if pos["id"] == pos_id:
                db["open"].pop(i)
                db["closed"].insert(0, _close(pos, price, "دستی"))
                _save(db)
                return pos
    return None


def close_with(pos_id, exit_price, reason, pnl_usdt=None):
    """بستن با سود/زیانِ محقق‌شده بیرونی (مثلاً از تست‌نت صرافی)."""
    with _lock:
        db = _load()
        for i, pos in enumerate(db["open"]):
            if pos["id"] == pos_id:
                db["open"].pop(i)
                p = _close(pos, exit_price, reason)
                if pnl_usdt is not None:
                    p["pnl_usdt"] = round(pnl_usdt, 3)
                    p["pnl_pct"] = round(pnl_usdt / max(p["size_usdt"], 1e-9) * 100, 3)
                db["closed"].insert(0, p)
                _save(db)
                return p
    return None


def partial_close(pos_id, price, fraction=0.5, reason="💰 برداشت پله‌ای سود"):
    """بستنِ بخشی از پوزیشن در قیمتِ فعلی — باقی‌مانده با همان حدضرر/هدف باز می‌ماند."""
    with _lock:
        db = _load()
        for pos in db["open"]:
            if pos["id"] == pos_id:
                if not (0 < fraction < 1) or pos.get("mode") == "testnet":
                    return None
                part = dict(pos)
                part["id"] = pos["id"] + "-p"
                part["size_usdt"] = round(pos["size_usdt"] * fraction, 2)
                closed = _close(part, price, reason)
                pos["size_usdt"] = round(pos["size_usdt"] - part["size_usdt"], 2)
                pos["partial_done"] = True
                pct, usd = _pnl(pos, price)
                pos.update(last_price=price, pnl_pct=round(pct, 3), pnl_usdt=round(usd, 3))
                db["closed"].insert(0, closed)
                _save(db)
                return closed
    return None


def move_sl(pos_id, new_sl):
    """جابه‌جاییِ حدضرر — فقط در جهتِ کاهشِ ریسک (مثلاً انتقال به سربه‌سر برای قفل‌کردنِ معامله)."""
    with _lock:
        db = _load()
        for pos in db["open"]:
            if pos["id"] == pos_id:
                d = 1 if pos["side"] == "long" else -1
                if d * (float(new_sl) - pos["sl"]) <= 0:      # شل‌کردنِ حدضرر ممنوع
                    return None
                pos["sl"] = float(new_sl)
                _save(db)
                return pos
    return None


def set_live(pos_id, price=None, pnl_usdt=None, pnl_pct=None):
    with _lock:
        db = _load()
        for pos in db["open"]:
            if pos["id"] == pos_id:
                if price is not None:
                    pos["last_price"] = price
                if pnl_usdt is not None:
                    pos["pnl_usdt"] = round(pnl_usdt, 3)
                if pnl_pct is not None:
                    pos["pnl_pct"] = round(pnl_pct, 3)
                _save(db)
                return pos
    return None


# ─────────────────────── کیف‌پول دمو ───────────────────────
def wallet_summary():
    with _lock:
        db = _load()
        w = db["wallet"]
        start = float(w.get("start", DEFAULT_BALANCE))
        realized = sum(p.get("pnl_usdt", 0.0) for p in db["closed"])
        open_pnl = sum(p.get("pnl_usdt", 0.0) for p in db["open"])
        wins = [p for p in db["closed"] if p.get("pnl_usdt", 0) > 0]
        losses = [p for p in db["closed"] if p.get("pnl_usdt", 0) < 0]
        breakeven = [p for p in db["closed"] if p.get("pnl_usdt", 0) == 0]
        nclosed = len(db["closed"])
        gross_win = sum(p["pnl_usdt"] for p in wins)
        gross_loss = -sum(p["pnl_usdt"] for p in losses)
        # پیک اکوییتی برای افت سرمایه: از قدیمی به جدید
        eq, peak, mdd = start, start, 0.0
        for p in reversed(db["closed"]):
            eq += p.get("pnl_usdt", 0.0)
            peak = max(peak, eq)
            mdd = max(mdd, peak - eq)
        return {
            "start": round(start, 2),
            "realized": round(realized, 2),
            "open_pnl": round(open_pnl, 2),
            "balance": round(start + realized, 2),
            "equity": round(start + realized + open_pnl, 2),
            "return_pct": round(realized / max(start, 1e-9) * 100, 2),
            "trades": nclosed,
            "wins": len(wins),
            "losses": len(losses),
            "breakeven": len(breakeven),
            "win_rate": round(len(wins) / max(len(wins) + len(losses), 1) * 100, 1)
                        if wins or losses else None,
            "profit_factor": round(gross_win / gross_loss, 2) if gross_loss > 0 else None,
            "max_drawdown": round(mdd, 2),
            "max_drawdown_pct": round(mdd / max(peak, 1e-9) * 100, 2),
            "open_count": len(db["open"]),
            "reset_at": w.get("reset_at"),
        }


def reset_wallet(start_balance=None, keep_history=False):
    """ریست کیف‌پول دمو از صفر: پاک‌سازی پوزیشن‌ها/تاریخچه و تنظیم موجودی اولیه."""
    with _lock:
        db = _load()
        start = float(start_balance) if start_balance else float(db["wallet"].get("start", DEFAULT_BALANCE))
        new_db = {
            "open": [],
            "closed": db["closed"] if keep_history else [],
            "wallet": {"start": start, "reset_at": _now()},
        }
        _save(new_db)
        return wallet_summary()


def set_start_balance(x):
    with _lock:
        db = _load()
        db["wallet"]["start"] = float(x)
        _save(db)
    return wallet_summary()
