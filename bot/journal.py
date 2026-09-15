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
import contextlib
import json
import os
import threading
import time

import paths

DATA_DIR = paths.DATA_DIR
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
# اصلاحیه: ژورنال هرگز بازنویسی نمی‌شود. رویدادِ اشتباه (مثلاً نوشته‌شده توسطِ
# تست‌ها) با یک رویدادِ QUARANTINE که ``(seq, ts)`` آن را نام می‌برد باطل می‌شود —
# مثلِ دفترِ حساب که خطا را با سندِ اصلاحی جبران می‌کند، نه با پاک‌کن. خطِ اصلی
# در فایل می‌ماند؛ events() و replay() دیگر آن را نمی‌بینند.
QUARANTINE = "quarantine"

LOCK_TIMEOUT_SEC = 10.0
TAIL_BYTES = 64 * 1024

_lock = threading.Lock()


# دو پروسه در این فایل می‌نویسند: برنامه (سفارش، ترمز، قطعِ اضطراری) و پروسهٔ
# بازسازیِ هفتگی (پیش‌ثبت و حکم). هر کدام شمارندهٔ seq را در حافظهٔ خودش نگه
# می‌داشت و هر دو یک seq نوشتند (۱۱۱ دو بار). حالا seq زیرِ قفلِ بین‌پروسه‌ای از
# دمِ خودِ فایل خوانده می‌شود؛ قفلِ سیستم‌عامل با مرگِ پروسه آزاد می‌شود.
if os.name == "nt":
    import msvcrt

    def _acquire(fd):
        deadline = time.time() + LOCK_TIMEOUT_SEC
        while True:
            try:
                os.lseek(fd, 0, os.SEEK_SET)
                msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
                return
            except OSError:
                if time.time() > deadline:
                    raise
                time.sleep(0.01)

    def _release(fd):
        os.lseek(fd, 0, os.SEEK_SET)
        msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
else:
    import fcntl

    def _acquire(fd):
        fcntl.flock(fd, fcntl.LOCK_EX)

    def _release(fd):
        fcntl.flock(fd, fcntl.LOCK_UN)


@contextlib.contextmanager
def _interprocess_lock():
    os.makedirs(os.path.dirname(PATH), exist_ok=True)
    fd = os.open(PATH + ".lock", os.O_RDWR | os.O_CREAT, 0o644)
    try:
        _acquire(fd)
        try:
            yield
        finally:
            _release(fd)
    finally:
        os.close(fd)


def _last_seq():
    """بزرگ‌ترین seq — از دمِ فایل، تا افزودن با بزرگ شدنِ ژورنال کند نشود."""
    try:
        size = os.path.getsize(PATH)
    except OSError:
        return 0
    with open(PATH, "rb") as f:
        f.seek(max(0, size - TAIL_BYTES))
        chunk = f.read()
    seqs = []
    for line in chunk.splitlines():
        try:
            seqs.append(int(json.loads(line).get("seq") or 0))
        except (ValueError, AttributeError):
            continue                          # خطِ بریده در ابتدای تکه یا نیمه‌نوشته
    if seqs or size <= TAIL_BYTES:
        return max(seqs, default=0)
    return max((int(r.get("seq") or 0) for r in _read_raw()), default=0)


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
    with _lock, _interprocess_lock():
        ev = {"seq": _last_seq() + 1, "ts": time.time(), "kind": kind, **payload}
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


def _ref(r):
    return int(r.get("seq") or 0), float(r.get("ts") or 0)


def _effective(rows):
    """سطرها منهای رویدادهایی که یک QUARANTINE باطلشان کرده (با seq **و** ts، چون
    پیش از قفلِ بین‌پروسه‌ای دو رویداد می‌توانستند یک seq داشته باشند)."""
    voided = {(int(x.get("seq") or 0), float(x.get("ts") or 0))
              for r in rows if r.get("kind") == QUARANTINE for x in (r.get("events") or [])}
    if not voided:
        return rows
    return [r for r in rows if r.get("kind") == QUARANTINE or _ref(r) not in voided]


def quarantine(refs, reason):
    """باطل‌کردنِ رویدادها بدونِ پاک‌کردنشان. ``refs``: رویدادها یا ``{"seq", "ts"}``."""
    refs = [{"seq": _ref(r)[0], "ts": _ref(r)[1]} for r in refs]
    return append(QUARANTINE, reason=reason, events=refs, count=len(refs))


def events(kinds=None, since=None, limit=None):
    rows = _effective(_read_raw())
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
    rows = sorted(_effective(_read_raw()), key=lambda r: int(r.get("seq") or 0))
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
    raw = _read_raw()
    rows = _effective(raw)
    by_kind = {}
    for r in rows:
        by_kind[r.get("kind")] = by_kind.get(r.get("kind"), 0) + 1
    return {"events": len(rows), "by_kind": by_kind, "quarantined": len(raw) - len(rows),
            "last_seq": max((int(r.get("seq") or 0) for r in raw), default=0),
            "last_ts": max((r.get("ts") or 0 for r in raw), default=None)}


def _reset_cache():
    """فقط برای تست‌ها — seq دیگر کش نمی‌شود؛ برای سازگاریِ تست‌های قدیمی مانده."""
