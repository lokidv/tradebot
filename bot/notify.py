# -*- coding: utf-8 -*-
"""اعلانِ تلگرام (اختیاری) — فقط وقتی کاربر خودش توکنِ ربات و شناسهٔ چت را وارد کرده باشد.

پیام فقط برای تصمیمِ دوشنبهٔ مومنتومِ BTC/ETH فرستاده می‌شود (هفته‌ای یک‌بار). تنظیمات در
data/notify.json (gitignored) — توکن هرگز در git یا لاگ نمی‌رود.
"""
import json
import os
import time

import httpx

import log
import paths

CFG_PATH = paths.data("notify.json")
DEFAULT = {"enabled": False, "token": "", "chat_id": "", "last_monday_ms": 0}


def load_cfg():
    try:
        with open(CFG_PATH, encoding="utf-8") as f:
            return {**DEFAULT, **json.load(f)}
    except (OSError, ValueError):
        return dict(DEFAULT)


def save_cfg(cfg):
    os.makedirs(os.path.dirname(CFG_PATH), exist_ok=True)
    with open(CFG_PATH + ".tmp", "w", encoding="utf-8") as f:
        json.dump({k: cfg.get(k, v) for k, v in DEFAULT.items()}, f, ensure_ascii=False)
    os.replace(CFG_PATH + ".tmp", CFG_PATH)


def public_cfg():
    c = load_cfg()
    return {"enabled": bool(c["enabled"]), "has_token": bool(c["token"]), "chat_id": c["chat_id"]}


def send(text, cfg=None, post=None):
    cfg = cfg or load_cfg()
    if not (cfg.get("enabled") and cfg.get("token") and cfg.get("chat_id")):
        return False
    post = post or (lambda url, data: httpx.post(url, data=data, timeout=20))
    r = post(f"https://api.telegram.org/bot{cfg['token']}/sendMessage",
             {"chat_id": cfg["chat_id"], "text": text})
    ok = getattr(r, "status_code", 200) == 200
    if not ok:
        log.warn("telegram send failed", status=getattr(r, "status_code", None))
    return ok


def monday_message(snap):
    lines = []
    for sym, a in (snap.get("assets") or {}).items():
        cur = a.get("current") or {}
        if cur:
            state = "🟢 در بازار (خرید/نگه‌داری)" if cur.get("in_market") else "⚪ نقد (بیرون)"
            lines.append(f"{sym.replace('USDT', '')}: {state} — بستهٔ یکشنبه {cur.get('sunday_close'):,.2f} "
                         f"({cur.get('change_pct'):+.1f}٪ نسبت به ۲۸ روز قبل)")
    if not lines:
        return None
    return ("🧭 تصمیمِ دوشنبهٔ مومنتومِ ۲۸روزه\n" + "\n".join(lines) +
            "\n\nسازگار در پژوهش، اثبات‌نشده — بی‌اهرم و فقط با مبلغی که از دست دادنش مشکلی نمی‌سازد.")


def maybe_notify_monday(snap, post=None):
    """یک‌بار برای هر دوشنبه، پس از اینکه تصمیمِ آن دوشنبه در دسترس شد."""
    cfg = load_cfg()
    mondays = [((a.get("current") or {}).get("monday_ms") or 0) for a in (snap.get("assets") or {}).values()]
    monday = max(mondays) if mondays else 0
    if not monday or monday <= int(cfg.get("last_monday_ms") or 0):
        return False
    text = monday_message(snap)
    if text and send(text, cfg, post):
        cfg["last_monday_ms"] = monday
        save_cfg(cfg)
        return True
    return False
