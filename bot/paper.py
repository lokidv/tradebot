# -*- coding: utf-8 -*-
"""پوزیشن‌های دمو (شبیه‌سازی محلی) — ذخیره در فایل JSON، بستن خودکار در هدف/حدضرر/حد زمانی."""
import json
import os
import threading
import uuid
from datetime import datetime, timezone

import log
import paths

DATA_DIR = paths.DATA_DIR
PATH = os.path.join(DATA_DIR, "positions.json")
# دفترِ کامل و فقط-افزودنیِ معامله‌های بسته. positions.json فقط ۲۰۰تای آخر را برای
# رابط نگه می‌دارد؛ افتِ سرمایه و ضریبِ سود باید از کلِ تاریخچه حساب شوند.
# مسیرش همیشه کنارِ PATH است تا جابه‌جاییِ PATH (مثلاً در تست) دفتر را هم جابه‌جا کند.
LEDGER_NAME = "closed_trades.jsonl"


def _ledger_path():
    return os.path.join(os.path.dirname(PATH), LEDGER_NAME)
_lock = threading.RLock()


def _now():
    return datetime.now(timezone.utc).isoformat()


DEFAULT_BALANCE = 10000.0
DEFAULT_COST_PCT = 0.15

# لغزشِ خروجِ استاپ‌مارکت بر حسب ردهٔ نقدشوندگی (نقطهٔ پایه). شبیه‌سازِ قبلی صفر
# فرض می‌کرد و حدضرر را دقیقاً روی سطحِ برنامه پر می‌کرد — روی آلت‌های کم‌عمق غیرواقعی است.
SLIPPAGE_BPS = {0.08: 3.0, 0.11: 5.0, 0.18: 10.0, 0.30: 20.0}
DEFAULT_SLIPPAGE_BPS = 10.0
FUNDING_INTERVAL_MS = 8 * 3600 * 1000     # پرپچوالِ بایننس هر ۸ ساعت تسویه می‌شود


def slippage_bps(cost_pct):
    """لغزشِ متناسب با ردهٔ هزینه — هرچه جفت کم‌عمق‌تر، خروجِ استاپ بدتر."""
    try:
        return SLIPPAGE_BPS[round(float(cost_pct), 2)]
    except (KeyError, TypeError, ValueError):
        return DEFAULT_SLIPPAGE_BPS


def _apply_slippage(price, side, cost_pct):
    """خروجِ اضطراری همیشه بدتر از سطحِ برنامه پر می‌شود."""
    d = 1 if side == "long" else -1
    return float(price) * (1 - d * slippage_bps(cost_pct) / 10_000.0)


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
                  opened_by="user", cost_pct=DEFAULT_COST_PCT, strategy=None, market="perp"):
    """``tp=None``: بی‌هدف (قاعدهٔ روند — فقط حدضررِ دنباله‌دار). ``market="spot"``: بی‌فاندینگ."""
    entry, sl, size_usdt = map(float, (entry, sl, size_usdt))
    tp = None if tp is None else float(tp)
    if side not in ("long", "short") or min(entry, sl, size_usdt) <= 0 or (tp is not None and tp <= 0):
        raise ValueError("پارامترهای پوزیشن نامعتبر است")
    if tp is None:
        if (side == "long" and not sl < entry) or (side == "short" and not entry < sl):
            raise ValueError("حدضرر با جهت معامله سازگار نیست")
    elif (side == "long" and not sl < entry < tp) or (side == "short" and not tp < entry < sl):
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
        "strategy": strategy, "market": market,
    }
    with _lock:
        db = _load()
        db["open"].append(pos)
        _save(db)
    return pos


def _total_cost_pct(pos):
    """هزینهٔ رفت‌وبرگشت + فاندینگِ انباشته (هر دو برحسبِ درصدِ نُشنال)."""
    return float(pos.get("cost_pct", DEFAULT_COST_PCT)) + float(pos.get("funding_pct", 0.0))


def _pnl(pos, price):
    d = 1 if pos["side"] == "long" else -1
    gross_pct = d * (price / pos["entry"] - 1) * 100
    pct = gross_pct - _total_cost_pct(pos)
    return pct, pos["size_usdt"] * pct / 100


def _ledger_append(pos):
    """افزودن به دفتر. بارِ اول، تاریخچهٔ فعلیِ positions.json را پیش از آن می‌نویسد؛
    وگرنه با اولین بسته‌شدن، ۷۰ معاملهٔ قبلی از خلاصه ناپدید می‌شدند."""
    path = _ledger_path()
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        seed = []
        if not os.path.exists(path):
            seed = list(reversed(_load().get("closed") or []))   # قدیمی → جدید
        with open(path, "a", encoding="utf-8") as f:
            for old in seed:
                f.write(json.dumps(dict(old, ledger_seeded=True), ensure_ascii=False) + "\n")
            f.write(json.dumps(pos, ensure_ascii=False) + "\n")
    except OSError:
        log.exc("closed-trade ledger append")


def ledger(since_reset=True):
    """همهٔ معامله‌های بسته (قدیمی → جدید). اگر دفتر هنوز ساخته نشده از positions.json می‌خواند."""
    rows = []
    path = _ledger_path()
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except ValueError:
                    continue
    if not rows:
        rows = list(reversed(_load().get("closed") or []))
    if since_reset:
        reset_at = (_load().get("wallet") or {}).get("reset_at")
        if reset_at:
            rows = [r for r in rows if str(r.get("closed_at") or "") >= str(reset_at)]
    return rows


def _close(pos, price, reason):
    pct, usd = _pnl(pos, price)
    d = 1 if pos["side"] == "long" else -1
    gross_pct = d * (price / pos["entry"] - 1) * 100
    pos.update(closed_at=_now(), exit_price=price, pnl_pct=round(pct, 3),
               gross_pnl_pct=round(gross_pct, 3),
               cost_usdt=round(pos["size_usdt"] * _total_cost_pct(pos) / 100, 3),
               funding_usdt=round(pos["size_usdt"] * float(pos.get("funding_pct", 0.0)) / 100, 3),
               pnl_usdt=round(usd, 3), close_reason=reason, last_price=price)
    _ledger_append(pos)
    return pos


def _opened_ms(pos):
    return datetime.fromisoformat(pos["opened_at"]).timestamp() * 1000


def _accrue_funding(pos, funding_rows, now_ms):
    """فاندینگِ پرپچوال را برای هر تسویهٔ ۸ساعته‌ای که پوزیشن باز بوده اضافه می‌کند.

    شبیه‌سازِ قبلی این را کاملاً نادیده می‌گرفت، در حالی که با حدِ زمانیِ ۴۰ کندل،
    یک پوزیشنِ ۱d تا ۴۰ روز باز می‌ماند: با ~۰٫۰۳٪ در روز این ~۱٫۲٪ است — هم‌اندازهٔ
    کلِ پاداشِ 1.8R روی ریسکِ ~۰٫۷٪.
    """
    if not funding_rows:
        return 0.0
    d = 1 if pos["side"] == "long" else -1
    since = float(pos.get("funding_ts") or _opened_ms(pos))
    added = 0.0
    latest = since
    for ts, rate in funding_rows:
        ts = float(ts)
        if since < ts <= now_ms:
            added += d * float(rate) * 100.0      # لانگ نرخِ مثبت می‌پردازد
            latest = max(latest, ts)
    if added or latest > since:
        pos["funding_pct"] = round(float(pos.get("funding_pct", 0.0)) + added, 6)
        pos["funding_ts"] = latest
    return added


def _barrier_exit(pos, kl):
    """برخوردِ حدضرر/هدف را روی **ویکِ کندل‌ها** می‌سنجد، نه فقط قیمتِ نمونه‌برداری‌شده.

    پایشِ هر ۱۲ ثانیه، ویک‌هایی را که بینِ دو نمونه می‌آیند و حدضرر را می‌زنند
    نمی‌دید. قواعد همان قواعدِ محافظه‌کارانهٔ ``bracket`` است: گپ روی قیمتِ باز،
    برخوردِ هم‌زمان = حدضرر.
    """
    if not kl or not kl.get("t"):
        return None
    d = 1 if pos["side"] == "long" else -1
    sl = float(pos["sl"])
    tp = None if pos.get("tp") is None else float(pos["tp"])      # بی‌هدف: فقط حدضرر
    since = float(pos.get("bar_ts") or _opened_ms(pos))
    for i, t in enumerate(kl["t"]):
        if t <= since:
            continue
        o, h, l = float(kl["o"][i]), float(kl["h"][i]), float(kl["l"][i])
        if (d == 1 and o <= sl) or (d == -1 and o >= sl):
            return _apply_slippage(o, pos["side"], pos.get("cost_pct")), "حدضرر (گپ)", t
        if (l <= sl if d == 1 else h >= sl):
            return _apply_slippage(sl, pos["side"], pos.get("cost_pct")), "حدضرر", t
        if tp is not None and (h >= tp if d == 1 else l <= tp):
            return tp, "هدف ✅", t
        pos["bar_ts"] = t
    return None


def refresh(prices, klines_fn=None, funding_fn=None):
    """به‌روزرسانی با قیمت‌های جدید؛ بستن خودکار در هدف/حدضرر/حد زمانی.

    ``klines_fn(symbol, tf)`` اگر داده شود، برخوردِ حدها روی ویکِ کندل‌ها سنجیده
    می‌شود (واقع‌گرا)؛ وگرنه فقط آخرین قیمتِ نمونه‌برداری‌شده ملاک است (خوش‌بینانه).
    ``funding_fn(symbol)`` تاریخچهٔ ``[[ts, rate], ...]`` می‌دهد تا فاندینگ کسر شود.
    """
    now_ms = datetime.now(timezone.utc).timestamp() * 1000
    with _lock:
        db = _load()
        still = []
        for pos in db["open"]:
            if pos.get("mode") == "testnet":     # چرخه تست‌نت را صرافی مدیریت می‌کند، نه شبیه‌ساز محلی
                still.append(pos)
                continue
            if funding_fn is not None and pos.get("market") != "spot":     # اسپات فاندینگ ندارد
                try:
                    _accrue_funding(pos, funding_fn(pos["symbol"]), now_ms)
                except Exception:  # noqa: BLE001 — نبودِ فاندینگ نباید پوزیشن را بشکند
                    log.exc()
            if klines_fn is not None:
                try:
                    hit = _barrier_exit(pos, klines_fn(pos["symbol"], pos["tf"]))
                except Exception:  # noqa: BLE001
                    hit = None
                if hit is not None:
                    exit_price, reason, _t = hit
                    db["closed"].insert(0, _close(pos, exit_price, reason))
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
                worst = min(price, pos["sl"]) if d == 1 else max(price, pos["sl"])
                exit_price = _apply_slippage(worst, pos["side"], pos.get("cost_pct"))
                db["closed"].insert(0, _close(pos, exit_price, "حدضرر"))
            elif pos.get("tp") is not None and ((d == 1 and price >= pos["tp"]) or (d == -1 and price <= pos["tp"])):
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
                    _ledger_append(dict(p, ledger_correction=True))   # PnLِ صرافی جایگزین شد
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
def _dedupe_ledger(rows):
    """اصلاحیهٔ PnLِ صرافی جایگزینِ ردیفِ اولیهٔ همان شناسه می‌شود."""
    by_id = {}
    for r in rows:
        by_id[r.get("id")] = r
    return sorted(by_id.values(), key=lambda r: str(r.get("closed_at") or ""))


def wallet_summary():
    with _lock:
        db = _load()
        w = db["wallet"]
        start = float(w.get("start", DEFAULT_BALANCE))
        history = _dedupe_ledger(ledger(since_reset=True))
        realized = sum(p.get("pnl_usdt", 0.0) for p in history)
        open_pnl = sum(p.get("pnl_usdt", 0.0) for p in db["open"])
        wins = [p for p in history if p.get("pnl_usdt", 0) > 0]
        losses = [p for p in history if p.get("pnl_usdt", 0) < 0]
        breakeven = [p for p in history if p.get("pnl_usdt", 0) == 0]
        nclosed = len(history)
        gross_win = sum(p["pnl_usdt"] for p in wins)
        gross_loss = -sum(p["pnl_usdt"] for p in losses)
        # پیک اکوییتی برای افت سرمایه: از قدیمی به جدید
        eq, peak, mdd = start, start, 0.0
        for p in history:                          # قدیمی → جدید، کلِ تاریخچه
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
        # با keep_history، مرزِ قبلی می‌ماند؛ وگرنه خلاصه (که از دفتر و بعد از reset_at
        # می‌خواند) همان تاریخچه‌ای را حذف می‌کرد که کاربر خواسته بود نگه دارد.
        reset_at = (db["wallet"].get("reset_at") or _now()) if keep_history else _now()
        new_db = {
            "open": [],
            "closed": db["closed"] if keep_history else [],
            "wallet": {"start": start, "reset_at": reset_at},
        }
        _save(new_db)
        return wallet_summary()


def set_start_balance(x):
    with _lock:
        db = _load()
        db["wallet"]["start"] = float(x)
        _save(db)
    return wallet_summary()
