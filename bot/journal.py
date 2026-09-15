# -*- coding: utf-8 -*-
"""ژورنالِ رویدادیِ فقط-افزودنی — حافظهٔ ربات که با ری‌استارت پاک نمی‌شود.

ترمزهای ریسک در متغیرهای ماژول زندگی می‌کردند: ``_risk_off_day`` (توقفِ ورود پس
از حدِ ضررِ روزانه)، ``_storm_until`` (ترمزِ طوفان)، ``_cooldown`` (ضدِ معاملهٔ
انتقامی) و ``_pending`` (سفارش‌های صبور). یک ری‌استارت — یا حتی یک کرشِ ساده —
همهٔ این‌ها را صفر می‌کرد: رباتی که تازه حدِ ضررِ روزانه‌اش را زده بود، پس از
بالا آمدن دوباره معامله باز می‌کرد. دقیقاً همان کاری که ترمز قرار بود جلویش را بگیرد.

هر تغییرِ وضعیت اینجا **اول** ثبت می‌شود و بعد اعمال؛ ``replay()`` وضعیت را از
روی تاریخچه بازمی‌سازد. فایل فقط-افزودنی است تا نوشتنِ هم‌زمان امن بماند و خطِ
نیمه‌نوشته (قطعِ برق) فقط همان خط را از دست بدهد.
"""
import json
import os
import threading
import time

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
PATH = os.path.join(DATA_DIR, "journal.jsonl")
MAX_REPLAY_AGE_SEC = 7 * 86400     # رویدادِ کهنه‌تر از یک هفته دیگر وضعیتِ فعلی نیست

# انواعِ رویداد
ORDER_PLACED = "order_placed"
ORDER_FILLED = "order_filled"
ORDER_EXPIRED = "order_expired"
ORDER_CANCELLED = "order_cancelled"
COOLDOWN_SET = "cooldown_set"
RISK_OFF = "risk_off"
STORM = "storm"
POSITION_CLOSED = "position_closed"
TRUST_CHANGE = "trust_change"
GATE_CHANGE = "gate_change"

_lock = threading.Lock()
_seq = None


def _next_seq(rows=None):
    global _seq
    if _seq is None:
        rows = rows if rows is not None else _read_raw()
        _seq = max((int(r.get("seq") or 0) for r in rows), default=0)
    _seq += 1
    return _seq


def _read_raw():
    if not os.path.exists(PATH):
        return []
    out = []
    with open(PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except ValueError:
                continue                     # خطِ نیمه‌نوشته — بقیه معتبرند
    return out


def append(kind, **payload):
    """ثبتِ یک رویداد. خروجی: خودِ رویداد (با ``seq`` و ``ts``)."""
    with _lock:
        ev = {"seq": _next_seq(), "ts": time.time(), "kind": kind, **payload}
        os.makedirs(os.path.dirname(PATH), exist_ok=True)
        # اگر نوشتنِ قبلی نیمه‌کاره مانده (قطعِ برق)، اول خط را ببند تا رویدادِ
        # سالمِ بعدی به دنبالهٔ خرابِ قبلی نچسبد و هر دو از دست نروند.
        if _needs_newline():
            with open(PATH, "a", encoding="utf-8") as f:
                f.write("\n")
        with open(PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(ev, ensure_ascii=False) + "\n")
        return ev


def _needs_newline():
    try:
        if os.path.getsize(PATH) == 0:
            return False
        with open(PATH, "rb") as f:
            f.seek(-1, os.SEEK_END)
            return f.read(1) != b"\n"
    except OSError:
        return False


def events(kinds=None, since=None, limit=None):
    rows = _read_raw()
    if kinds:
        kinds = set(kinds if isinstance(kinds, (list, tuple, set)) else [kinds])
        rows = [r for r in rows if r.get("kind") in kinds]
    if since is not None:
        rows = [r for r in rows if (r.get("ts") or 0) >= since]
    return rows[-limit:] if limit else rows


def replay(now=None):
    """بازسازیِ وضعیتِ ریسک از روی تاریخچه.

    خروجی: ``{pending, cooldown, risk_off_day, storm_until, seq}``
    """
    now = now if now is not None else time.time()
    rows = sorted(_read_raw(), key=lambda r: int(r.get("seq") or 0))
    pending, cooldown = {}, {}
    risk_off_day, storm_until = None, 0.0
    for r in rows:
        if (r.get("ts") or 0) < now - MAX_REPLAY_AGE_SEC:
            continue
        kind, sym = r.get("kind"), r.get("symbol")
        if kind == ORDER_PLACED and sym:
            pending[sym] = r.get("order") or {}
        elif kind in (ORDER_FILLED, ORDER_EXPIRED, ORDER_CANCELLED) and sym:
            pending.pop(sym, None)
        elif kind == COOLDOWN_SET and sym:
            cooldown[sym] = max(float(cooldown.get(sym, 0)), float(r.get("until") or 0))
        elif kind == RISK_OFF:
            risk_off_day = r.get("day")
        elif kind == STORM:
            storm_until = max(storm_until, float(r.get("until") or 0))
    # سفارشِ منقضی‌شده در زمانِ خاموشی نباید زنده برگردد
    pending = {s: o for s, o in pending.items() if float(o.get("expires") or 0) > now}
    cooldown = {s: u for s, u in cooldown.items() if u > now}
    with _lock:
        seq = max((int(r.get("seq") or 0) for r in rows), default=0)
    return {"pending": pending, "cooldown": cooldown,
            "risk_off_day": risk_off_day, "storm_until": storm_until, "seq": seq}


def summary():
    rows = _read_raw()
    by_kind = {}
    for r in rows:
        by_kind[r.get("kind")] = by_kind.get(r.get("kind"), 0) + 1
    return {"events": len(rows), "by_kind": by_kind,
            "last_seq": max((int(r.get("seq") or 0) for r in rows), default=0),
            "last_ts": max((r.get("ts") or 0 for r in rows), default=None)}


def _reset_cache():
    """فقط برای تست‌ها."""
    global _seq
    _seq = None
