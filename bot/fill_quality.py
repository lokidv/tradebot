# -*- coding: utf-8 -*-
"""کیفیت پر شدن سفارش‌های صبور — لغزش، نرخ انقضا، هزینهٔ واقعی."""
from __future__ import annotations

import json
import os
import threading
import time
from typing import Any

import paths

PATH = paths.data("fill_quality.json")
_lock = threading.Lock()


def _load():
    if not os.path.exists(PATH):
        return {"events": []}
    try:
        with open(PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:  # noqa: BLE001
        return {"events": []}


def _save(db):
    os.makedirs(os.path.dirname(PATH), exist_ok=True)
    tmp = PATH + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(db, f, ensure_ascii=False)
    os.replace(tmp, PATH)


import journal

# هر رویدادِ سفارش هم‌زمان به ژورنالِ ماندگار می‌رود تا ری‌استارت وضعیتِ
# سفارش‌های صبور را گم نکند (قبلاً ``_pending`` فقط در حافظه بود).
_JOURNAL_KIND = {
    "filled": journal.ORDER_FILLED,
    "expired": journal.ORDER_EXPIRED,
    "cancelled": journal.ORDER_CANCELLED,
}


def log_event(kind: str, **kw):
    """kind: filled | expired | cancelled"""
    with _lock:
        db = _load()
        ev = {"ts": time.time(), "kind": kind, **kw}
        db["events"] = (db.get("events") or [])[-500:]
        db["events"].append(ev)
        _save(db)
    jk = _JOURNAL_KIND.get(kind)
    if jk:
        try:
            journal.append(jk, **kw)
        except Exception:  # noqa: BLE001 — ژورنال نباید مسیرِ معامله را بشکند
            pass


def summary(days=14) -> dict[str, Any]:
    cutoff = time.time() - days * 86400
    with _lock:
        evs = [e for e in (_load().get("events") or []) if e.get("ts", 0) >= cutoff]
    filled = [e for e in evs if e.get("kind") == "filled"]
    expired = [e for e in evs if e.get("kind") == "expired"]
    cancelled = [e for e in evs if e.get("kind") == "cancelled"]
    slips = [float(e["slip_r"]) for e in filled if e.get("slip_r") is not None]
    avg_slip = sum(slips) / len(slips) if slips else None
    # slip_r منفی = ورود ارزان‌تر از زندهٔ لحظهٔ سفارش (خوب)
    n = len(filled) + len(expired) + len(cancelled)
    fill_rate = round(len(filled) / n * 100, 1) if n else None
    return {
        "days": days,
        "n": n,
        "filled": len(filled),
        "expired": len(expired),
        "cancelled": len(cancelled),
        "fill_rate_pct": fill_rate,
        "avg_slip_r": round(avg_slip, 3) if avg_slip is not None else None,
        "verdict": (
            "good" if fill_rate is not None and fill_rate >= 35 and (avg_slip is None or avg_slip <= 0.15)
            else "ok" if n >= 5
            else "thin" if n > 0
            else "empty"
        ),
    }
