# -*- coding: utf-8 -*-
"""قفل ایمنی سرمایه — تنها مرجعِ «مجوز ورود» در کل سیستم.

هیچ مدل، سیاست، جیب زنده یا امتیازی نمی‌تواند به‌تنهایی معامله‌ای را «قابل‌اجرا» کند؛
فقط ترکیب‌های ``tf|setup|side`` که با پیش‌ثبت و عبور از آزمون منجمد به ``allowed_combos``
نوشته شده‌اند مجازند. پول واقعی هم فقط با ``live_allowed`` **و** تأیید صریح محیطی باز می‌شود.

فایل: ``bot/data/gates.json`` (نسخه‌دار، در git). این ماژول وابستگی سنگین ندارد تا از هر جا
(engine/autotrader/broker/main) بدون import چرخه‌ای قابل استفاده باشد.
"""
import json
import os
import threading
import time

GATES_PATH = os.path.join(os.path.dirname(__file__), "data", "gates.json")
LIVE_ACK_ENV = "TRADERBOT_LIVE_ACK"

DEFAULTS = {
    "version": 1,
    "preregistered_at": None,
    "preregistration_hash": None,
    "live_allowed": False,
    "max_risk_pct_per_trade": 0.25,
    "max_open": 3,
    "max_side_open": 2,
    "heat_cap_pct": 0.75,
    "daily_loss_halt_pct": 1.0,
    "weekly_loss_halt_pct": 2.5,
    "allowed_combos": [],
    "gate_results": {},
    "history": [],
}

_lock = threading.Lock()
_cache = {"mtime": None, "data": None}


class GateError(RuntimeError):
    pass


def combo_key(tf, setup, side):
    return f"{tf}|{setup}|{side}"


def _validate(d):
    out = dict(DEFAULTS)
    out.update({k: v for k, v in (d or {}).items() if k in DEFAULTS})
    if not isinstance(out["allowed_combos"], list):
        raise GateError("allowed_combos باید فهرست باشد")
    for c in out["allowed_combos"]:
        if not isinstance(c, str) or c.count("|") != 2:
            raise GateError(f"ترکیب نامعتبر در allowed_combos: {c!r}")
    for k in ("max_risk_pct_per_trade", "heat_cap_pct", "daily_loss_halt_pct", "weekly_loss_halt_pct"):
        v = float(out[k])
        if not (0 < v <= 5):
            raise GateError(f"{k} خارج از محدودهٔ امن (0, 5]: {v}")
        out[k] = v
    out["max_open"] = int(out["max_open"])
    out["max_side_open"] = int(out["max_side_open"])
    out["live_allowed"] = bool(out["live_allowed"])
    out["version"] = int(out["version"])
    return out


def load_gates(force=False):
    """خواندن gates.json با کش بر اساس mtime. نبودِ فایل = پیش‌فرض‌های بسته (fail-closed)."""
    with _lock:
        try:
            mtime = os.path.getmtime(GATES_PATH)
        except OSError:
            mtime = None
        if not force and _cache["data"] is not None and _cache["mtime"] == mtime:
            return _cache["data"]
        data = dict(DEFAULTS)
        if mtime is not None:
            with open(GATES_PATH, "r", encoding="utf-8") as f:
                data = _validate(json.load(f))
        _cache.update(mtime=mtime, data=data)
        return data


def _save(data, reason):
    """نوشتن اتمیک + ثبت تاریخچه. فقط توسط توابع همین ماژول صدا زده می‌شود."""
    data = _validate(data)
    hist = list(data.get("history") or [])
    hist.append({"ts": time.time(), "reason": reason,
                 "allowed_combos": list(data["allowed_combos"]),
                 "live_allowed": data["live_allowed"],
                 "max_risk_pct_per_trade": data["max_risk_pct_per_trade"]})
    data["history"] = hist[-200:]
    os.makedirs(os.path.dirname(GATES_PATH), exist_ok=True)
    tmp = GATES_PATH + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=1)
    os.replace(tmp, GATES_PATH)
    with _lock:
        _cache.update(mtime=os.path.getmtime(GATES_PATH), data=data)
    return data


def is_combo_allowed(tf, setup, side):
    if not (tf and setup and side):
        return False
    return combo_key(tf, setup, side) in set(load_gates().get("allowed_combos") or [])


def allowed_combos():
    return list(load_gates().get("allowed_combos") or [])


def live_allowed():
    """پول واقعی فقط با پرچم فایل **و** تأیید محیطی که با نسخهٔ گیت برابر است."""
    g = load_gates()
    return bool(g.get("live_allowed")) and os.environ.get(LIVE_ACK_ENV, "") == str(g.get("version"))


def risk_caps():
    g = load_gates()
    return {k: g[k] for k in ("max_risk_pct_per_trade", "max_open", "max_side_open",
                              "heat_cap_pct", "daily_loss_halt_pct", "weekly_loss_halt_pct")}


def set_allowed_combos(combos, reason, judge_token=None):
    """فقط تابع داورِ آزمون منجمد (فاز ۲) حق نوشتن دارد؛ ``judge_token`` باید با نسخهٔ گیت بخواند.
    در فاز ۰ عمداً هیچ مسیر خودکاری این را صدا نمی‌زند."""
    g = dict(load_gates())
    if judge_token != f"judge:{g['version']}":
        raise GateError("نوشتن allowed_combos فقط از مسیر داور آزمون منجمد مجاز است")
    g["allowed_combos"] = sorted(set(combos))
    return _save(g, reason)


def status():
    g = load_gates()
    return {
        "version": g["version"],
        "live_allowed": g["live_allowed"],
        "live_effective": live_allowed(),
        "allowed_combos": list(g["allowed_combos"]),
        "preregistration_hash": g["preregistration_hash"],
        "caps": risk_caps(),
        "path": GATES_PATH,
    }
