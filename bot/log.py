# -*- coding: utf-8 -*-
"""لاگِ ساخت‌یافته — پایانِ خطاهای بلعیده‌شده.

در کدِ قبلی بیش از ۷۰ بلوکِ ``except Exception: pass`` وجود داشت و هیچ ماژولِ
لاگی در کار نبود. نتیجه این بود که «فیدِ داده مرده است»، «صرافی جواب نمی‌دهد» و
«همه‌چیز عالی است» از بیرون یکسان به‌نظر می‌رسیدند، و ویژگیِ گم‌شده بی‌صدا به
۰٫۰ تبدیل می‌شد — یعنی مدل روی ورودیِ جعلی تصمیم می‌گرفت.

اینجا هر خطا با محلِ دقیقش ثبت می‌شود، ولی با **سرکوبِ تکرار** تا یک قطعیِ
شبکه لاگ را پر نکند: خطای تکراری از یک محل، در پنجرهٔ ۶۰ ثانیه فقط یک‌بار با
شمارشِ دفعات نوشته می‌شود.
"""
import json
import logging
import logging.handlers
import os
import sys
import threading
import time
import traceback

LOG_DIR = os.path.join(os.path.dirname(__file__), "data", "logs")
LOG_PATH = os.path.join(LOG_DIR, "bot.log")
MAX_BYTES = 5 * 1024 * 1024
BACKUPS = 3
DEDUP_WINDOW_SEC = 60

_lock = threading.Lock()
_logger = None
_seen = {}                 # (file, lineno) -> [آخرین لاگ, تعدادِ سرکوب‌شده]
_counts = {"error": 0, "warning": 0, "info": 0}


class _JsonFormatter(logging.Formatter):
    def format(self, record):
        row = {
            "ts": round(record.created, 3),
            "level": record.levelname.lower(),
            "logger": record.name,
            "msg": record.getMessage(),
        }
        for k, v in (getattr(record, "extra_fields", None) or {}).items():
            row[k] = v
        if record.exc_info:
            row["exc"] = "".join(traceback.format_exception(*record.exc_info))[-2000:]
        return json.dumps(row, ensure_ascii=False)


def get_logger():
    global _logger
    with _lock:
        if _logger is not None:
            return _logger
        lg = logging.getLogger("ctp")
        lg.setLevel(logging.INFO)
        lg.propagate = False
        if not lg.handlers:
            try:
                os.makedirs(LOG_DIR, exist_ok=True)
                fh = logging.handlers.RotatingFileHandler(
                    LOG_PATH, maxBytes=MAX_BYTES, backupCount=BACKUPS, encoding="utf-8")
                fh.setFormatter(_JsonFormatter())
                lg.addHandler(fh)
            except Exception:  # noqa: BLE001 — بدونِ فایل هم باید کار کند
                sh = logging.StreamHandler(sys.stderr)
                sh.setFormatter(_JsonFormatter())
                lg.addHandler(sh)
        _logger = lg
        return lg


def _emit(level, msg, exc_info=False, **fields):
    lg = get_logger()
    _counts[level if level in _counts else "info"] = _counts.get(level, 0) + 1
    rec_level = {"error": logging.ERROR, "warning": logging.WARNING}.get(level, logging.INFO)
    lg.log(rec_level, msg, exc_info=exc_info, extra={"extra_fields": fields})


def info(msg, **fields):
    _emit("info", msg, **fields)


def warn(msg, **fields):
    _emit("warning", msg, **fields)


def error(msg, **fields):
    _emit("error", msg, **fields)


def exc(context="", **fields):
    """ثبتِ استثنایی که عمداً بلعیده شده — با محلِ دقیق و سرکوبِ تکرار.

    به‌جای ``except Exception: pass`` صدا زده می‌شود؛ رفتارِ برنامه عوض نمی‌شود
    (استثنا همچنان بلعیده می‌شود) ولی دیگر نامرئی نیست.
    """
    frame = sys._getframe(1)
    where = f"{os.path.basename(frame.f_code.co_filename)}:{frame.f_lineno}"
    now = time.time()
    with _lock:
        last, suppressed = _seen.get(where, (0.0, 0))
        if now - last < DEDUP_WINDOW_SEC:
            _seen[where] = (last, suppressed + 1)
            return
        _seen[where] = (now, 0)
    if suppressed:
        fields["suppressed_repeats"] = suppressed
    _emit("warning", f"swallowed: {context or where}", exc_info=True,
          where=where, func=frame.f_code.co_name, **fields)


def counts():
    with _lock:
        return dict(_counts)


def tail(n=100, level=None):
    """آخرین خطوطِ لاگ — برای نمایش در /api/health بدونِ بازکردنِ فایل روی دیسک."""
    if not os.path.exists(LOG_PATH):
        return []
    try:
        with open(LOG_PATH, "r", encoding="utf-8") as f:
            lines = f.readlines()[-(n * 4):]
    except OSError:
        return []
    out = []
    for line in lines:
        try:
            row = json.loads(line)
        except ValueError:
            continue
        if level and row.get("level") != level:
            continue
        out.append(row)
    return out[-n:]
