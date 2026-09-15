# -*- coding: utf-8 -*-
"""پچ — دفترِ کاملِ معامله‌های بسته + بازآموزی در پروسهٔ جدا.

۱. ``positions.json`` فقط ۲۰۰ معاملهٔ بستهٔ آخر را نگه می‌داشت و ``wallet_summary``
   افتِ سرمایه و ضریبِ سود را روی همین تاریخچهٔ بریده حساب می‌کرد. پس از ۲۰۰
   معامله، بدترین افتِ واقعی می‌توانست از گزارش ناپدید شود. حالا هر بسته‌شدن در
   ``closed_trades.jsonl`` (فقط-افزودنی) ثبت می‌شود و خلاصه از همان محاسبه می‌شود.

۲. بازآموزی داخلِ همان پروسه‌ای اجرا می‌شد که معامله می‌کند (ProcessPool درونِ
   uvicorn روی ویندوز، و پاک‌شدنِ کشِ تحلیل وسطِ چرخه). حالا ``rebuild_models.py``
   در پروسهٔ جدا اجرا می‌شود و پس از پایان، جدول از دیسک بازخوانی می‌شود.
"""
import io
import os

BOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "bot")


def patch(name, pairs):
    path = os.path.join(BOT, name)
    s = io.open(path, encoding="utf-8").read()
    for old, new, label in pairs:
        n = s.count(old)
        assert n == 1, f"{name} / {label}: found {n}"
        s = s.replace(old, new)
    io.open(path, "w", encoding="utf-8", newline="\n").write(s)


patch("paper.py", [
    ('''DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
PATH = os.path.join(DATA_DIR, "positions.json")''',
     '''DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
PATH = os.path.join(DATA_DIR, "positions.json")
# دفترِ کامل و فقط-افزودنیِ معامله‌های بسته. positions.json فقط ۲۰۰تای آخر را برای
# رابط نگه می‌دارد؛ افتِ سرمایه و ضریبِ سود باید از کلِ تاریخچه حساب شوند.
LEDGER_PATH = os.path.join(DATA_DIR, "closed_trades.jsonl")''', "ledger path"),

    ('''def _close(pos, price, reason):
    pct, usd = _pnl(pos, price)''',
     '''def _ledger_append(pos):
    try:
        os.makedirs(os.path.dirname(LEDGER_PATH), exist_ok=True)
        with open(LEDGER_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(pos, ensure_ascii=False) + "\\n")
    except OSError:
        log.exc("closed-trade ledger append")


def ledger(since_reset=True):
    """همهٔ معامله‌های بسته (قدیمی → جدید). اگر دفتر خالی است از positions.json می‌خواند."""
    rows = []
    if os.path.exists(LEDGER_PATH):
        with open(LEDGER_PATH, "r", encoding="utf-8") as f:
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
    pct, usd = _pnl(pos, price)''', "ledger helpers"),

    ('''               pnl_usdt=round(usd, 3), close_reason=reason, last_price=price)
    return pos''',
     '''               pnl_usdt=round(usd, 3), close_reason=reason, last_price=price)
    _ledger_append(pos)
    return pos''', "append on close"),

    ('''                if pnl_usdt is not None:
                    p["pnl_usdt"] = round(pnl_usdt, 3)
                    p["pnl_pct"] = round(pnl_usdt / max(p["size_usdt"], 1e-9) * 100, 3)''',
     '''                if pnl_usdt is not None:
                    p["pnl_usdt"] = round(pnl_usdt, 3)
                    p["pnl_pct"] = round(pnl_usdt / max(p["size_usdt"], 1e-9) * 100, 3)
                    _ledger_append(dict(p, ledger_correction=True))   # PnLِ صرافی جایگزین شد''',
     "testnet pnl correction"),

    ('''def wallet_summary():
    with _lock:
        db = _load()
        w = db["wallet"]
        start = float(w.get("start", DEFAULT_BALANCE))
        realized = sum(p.get("pnl_usdt", 0.0) for p in db["closed"])
        open_pnl = sum(p.get("pnl_usdt", 0.0) for p in db["open"])
        wins = [p for p in db["closed"] if p.get("pnl_usdt", 0) > 0]
        losses = [p for p in db["closed"] if p.get("pnl_usdt", 0) < 0]
        breakeven = [p for p in db["closed"] if p.get("pnl_usdt", 0) == 0]
        nclosed = len(db["closed"])''',
     '''def _dedupe_ledger(rows):
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
        nclosed = len(history)''', "summary from ledger"),

    ('''        eq, peak, mdd = start, start, 0.0
        for p in reversed(db["closed"]):''',
     '''        eq, peak, mdd = start, start, 0.0
        for p in history:                          # قدیمی → جدید، کلِ تاریخچه''', "drawdown from ledger"),
])

patch("main.py", [
    ('''def _calib_builder():
    """ساخت/تازه‌سازی جدول کالیبراسیون در پس‌زمینه (روزی یک‌بار)."""
    try:
        symbols, _ = market.get_top_symbols(CALIB_TOP)
        calib.build(symbols)
        with _an_lock:
            _an_cache.clear()          # تحلیل‌ها با احتمال کالیبره از نو
    except Exception:  # noqa: BLE001
        log.exc()''',
     '''_rebuild_proc = {"p": None, "started": None}


def _calib_builder():
    """بازآموزی در **پروسهٔ جدا** (rebuild_models.py)، نه داخلِ پروسهٔ معامله.

    قبلاً calib.build همین‌جا اجرا می‌شد: ProcessPool درونِ uvicorn روی ویندوز،
    رقابتِ CPU با چرخهٔ معامله، و پاک‌شدنِ کشِ تحلیل وسطِ کار. حالا پروسهٔ جدا
    جدول را روی دیسک می‌نویسد و این‌جا فقط بازخوانی می‌شود.
    """
    import subprocess
    p = _rebuild_proc.get("p")
    if p is not None and p.poll() is None:
        return                                     # یکی در جریان است
    script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "rebuild_models.py")
    log_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "logs", "rebuild.log")
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    try:
        with open(log_path, "a", encoding="utf-8") as out:
            proc = subprocess.Popen([sys.executable, script, "--top", str(CALIB_TOP)],
                                    stdout=out, stderr=subprocess.STDOUT,
                                    cwd=os.path.dirname(script),
                                    env=dict(os.environ, PYTHONIOENCODING="utf-8"))
        _rebuild_proc.update(p=proc, started=time.time())
        proc.wait()
        calib._table = None                        # جدولِ تازه از دیسک
        with _an_lock:
            _an_cache.clear()                      # تحلیل‌ها با مدلِ تازه از نو
        log.info("rebuild finished", returncode=proc.returncode)
    except Exception:  # noqa: BLE001
        log.exc("rebuild subprocess")''', "out-of-process rebuild"),

    ('''@app.post("/api/calib/rebuild")
def calib_rebuild():
    if calib.status().get("building"):
        return {"ok": False, "msg": "در حال ساخت است"}''',
     '''@app.post("/api/calib/rebuild")
def calib_rebuild():
    p = _rebuild_proc.get("p")
    if calib.status().get("building") or (p is not None and p.poll() is None):
        return {"ok": False, "msg": "در حال ساخت است"}''', "rebuild endpoint guard"),
])
print("ledger + retrain patch applied")
