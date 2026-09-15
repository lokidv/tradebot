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
    # ترکیبی که روی ارزهای بزرگ پذیرفته شده، به‌طور خودکار روی آلت‌های کم‌عمق مجاز نیست.
    # خالی = هیچ نمادی (نه همه). داور همراهِ ترکیب‌ها می‌نویسدش.
    "allowed_symbols": [],
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
    if not isinstance(out["allowed_symbols"], list):
        raise GateError("allowed_symbols باید فهرست باشد")
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
    # هر تغییرِ گیت در ژورنال — در طولِ دورهٔ اثبات، ساعت را صفر می‌کند (report.py)
    try:
        import journal
        journal.append(journal.GATE_CHANGE, reason=reason,
                       allowed_combos=list(data["allowed_combos"]),
                       live_allowed=data["live_allowed"],
                       max_risk_pct_per_trade=data["max_risk_pct_per_trade"])
    except Exception:  # noqa: BLE001, silent-ok — gates پیش از لاگ بارگذاری می‌شود؛ ژورنال اختیاری است
        pass
    return data


def set_live_risk(risk_pct, reason, report_token=None):
    """فقط تصمیمِ پیش‌ثبت‌شدهٔ مقیاس (report.live_scale_decision) ریسکِ لایو را عوض می‌کند."""
    g = dict(load_gates())
    if report_token != f"scale:{g['version']}":
        raise GateError("تغییرِ ریسکِ لایو فقط از مسیرِ تصمیمِ مقیاسِ پیش‌ثبت‌شده مجاز است")
    g["max_risk_pct_per_trade"] = float(risk_pct)
    return _save(g, reason)


def kill(reason):
    """کلیدِ قطعِ اضطراری: لایو خاموش و فهرستِ مجاز خالی. برگرداندنش پیش‌ثبت و قضاوتِ تازه می‌خواهد."""
    g = dict(load_gates())
    g["live_allowed"] = False
    g["allowed_combos"] = []
    return _save(g, f"KILL: {reason}")


def is_combo_allowed(tf, setup, side, symbol=None):
    """ترکیبِ پیش‌ثبت‌شده **و** نمادِ داخلِ دامنه‌ای که روی آن قضاوت شده است.

    اگر ``symbol`` داده نشود فقط ترکیب سنجیده می‌شود (سازگاری با فراخوان‌های قدیمی)،
    ولی موتور همیشه نماد را می‌فرستد.
    """
    if not (tf and setup and side):
        return False
    g = load_gates()
    if combo_key(tf, setup, side) not in set(g.get("allowed_combos") or []):
        return False
    if symbol is None:
        return True
    return symbol in set(g.get("allowed_symbols") or [])


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


def set_allowed_combos(combos, reason, judge_token=None, symbols=None):
    """فقط ``research.judge`` حق نوشتن دارد؛ ``judge_token`` باید با نسخهٔ گیت بخواند."""
    g = dict(load_gates())
    if judge_token != f"judge:{g['version']}":
        raise GateError("نوشتن allowed_combos فقط از مسیر داور آزمون منجمد مجاز است")
    g["allowed_combos"] = sorted(set(combos))
    if symbols is not None:
        g["allowed_symbols"] = sorted(set(symbols))
    return _save(g, reason)


def bind_preregistration(prereg_hash, reason):
    """هشِ پیش‌ثبت را قفل می‌کند. پیش‌ثبتِ تازه یعنی فهرستِ مجاز از صفر.

    بدونِ پاک‌کردن، ترکیب‌هایی که زیرِ پیش‌ثبتِ قبلی قبول شده بودند زیرِ
    فرضیه‌های تازه هم مجاز می‌ماندند.
    """
    g = dict(load_gates())
    g["preregistration_hash"] = prereg_hash
    g["preregistered_at"] = time.time()
    g["allowed_combos"] = []
    g["allowed_symbols"] = []
    return _save(g, reason)


def status():
    g = load_gates()
    return {
        "version": g["version"],
        "live_allowed": g["live_allowed"],
        "live_effective": live_allowed(),
        "allowed_combos": list(g["allowed_combos"]),
        "allowed_symbols": list(g["allowed_symbols"]),
        "preregistration_hash": g["preregistration_hash"],
        "caps": risk_caps(),
        "path": GATES_PATH,
    }
