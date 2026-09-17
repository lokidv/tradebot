# -*- coding: utf-8 -*-
"""اجرای خودکارِ قاعدهٔ روندِ روزانه روی تست‌نتِ فیوچرزِ بایننس — پولِ مجازی، هرگز شبکهٔ اصلی.

کاربر در ۱۴۰۵/۰۶/۲۴ صریحاً خواست همین نامزدِ پژوهشی (``trend.PRIMARY``) روی تست‌نت اجرا
شود. مجوزش در ``gates.testnet_research`` است و به پولِ واقعی هیچ راهی ندارد: این ماژول فقط
با نمونهٔ ``broker.BinanceTestnet`` کار می‌کند که آدرسش در کد به تست‌نت قفل است.

هدف سنجشِ **اجرا** است، نه اثباتِ لبه. هر معامله همان شناسهٔ دفترِ کاغذی (``trend.py``) را
دارد، پس برای هر کدام معلوم است ورود چقدر با openِ روز فاصله داشت، حدضرر کجا پر شد، و R
واقعیِ صرافی (با کمیسیون و فاندینگ) در برابرِ R کاغذی چقدر بود.

قواعدِ اجرا:

* فقط سیگنال‌های رو-به-جلو؛ ورودِ مارکت تا ۶ ساعت پس از openِ روز. دیرتر یعنی «از دست
  رفته» — دنبالِ قیمت نمی‌دویم، ولی در کارنامه می‌ماند.
* حدضررِ اولیه دقیقاً همان سطحِ دفترِ کاغذی؛ هر روز فقط رو به بالا، «اول تازه، بعد لغوِ قدیمی».
* قیمتِ تست‌نت بیش از ۱٫۵٪ دور از بازارِ واقعی ⇒ ورود نه (دفترِ سفارشِ تست‌نت گاهی از بازار
  جدا می‌شود و سنجش را بی‌معنا می‌کند).
* پوزیشنی که خودش باز نکرده را دست نمی‌زند.
* بی‌مجوز: ورودِ تازه نه، ولی پوزیشن‌های باز تا خروج مدیریت می‌شوند؛ قطعِ اضطراری همه را می‌بندد.
"""
import json
import os
import threading
import time

import numpy as np

import broker as brokermod
import gates
import log
import market
import paths
import trend

STRATEGY = "1d|trend20|long"
RULE = trend.PRIMARY
EVENTS_PATH = paths.data("trend_exec.jsonl")
ENTRY_WINDOW_SEC = 6 * 3600
MAX_PRICE_GAP_PCT = 1.5
DAY_MS = 86_400_000

_lock = threading.Lock()
_last = {"at": None, "result": None}


# ───────────────────────── رویدادها (فقط-افزودنی) ─────────────────────────
def _append(kind, **payload):
    os.makedirs(os.path.dirname(EVENTS_PATH), exist_ok=True)
    ev = {"kind": kind, "at": time.time(), **payload}
    with open(EVENTS_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(ev, ensure_ascii=False, default=float) + "\n")
    return ev


def events():
    if not os.path.exists(EVENTS_PATH):
        return []
    out = []
    with open(EVENTS_PATH, "r", encoding="utf-8") as f:
        for line in f:
            try:
                out.append(json.loads(line))
            except ValueError:
                continue
    return out


def replay(evs=None):
    """وضعیت از روی رویدادها — پس از ری‌استارت هم همان. خروجی: ``(معامله‌ها، شناسه‌های تصمیم‌گرفته)``."""
    trades, decided = {}, set()
    for e in (events() if evs is None else evs):
        tid, kind = e.get("id"), e.get("kind")
        if kind == "open":
            trades[tid] = {**e, "status": "open"}
            decided.add(tid)
        elif kind == "missed":
            decided.add(tid)
        elif kind == "stop_moved" and tid in trades:
            trades[tid]["stop"] = e["new"]
            trades[tid]["stop_order_id"] = e.get("order_id")
        elif kind == "closed" and tid in trades:
            trades[tid].update(status="closed", reason=e.get("reason"), exit_px=e.get("exit_px"),
                               realized_usdt=e.get("realized_usdt"), realized_r=e.get("realized_r"),
                               stop_slippage_bps=e.get("stop_slippage_bps"), closed_at=e.get("at"))
    return trades, decided


def testnet_client(bk):
    """فقط کلاینتِ تست‌نت؛ هر چیزِ دیگر — حتی شبیه‌سازِ محلی — یعنی «اجرا نکن»."""
    if isinstance(bk, brokermod.BinanceTestnet) and "testnet" in brokermod.TESTNET_BASE:
        return bk
    return None


# ───────────────────────── مدیریت و ورود ─────────────────────────
def _exit_px(bk, tr):
    """میانگینِ قیمتِ فروش‌هایی که این لانگ را بستند."""
    try:
        fills = [f for f in bk.fills_since(tr["sym"], tr["opened_ms"]) if f.get("side") == "SELL"]
    except Exception:  # noqa: BLE001, silent-ok — قیمتِ خروج اختیاری است؛ R واقعی از income می‌آید
        return None
    qty = sum(float(f["qty"]) for f in fills)
    return sum(float(f["price"]) * float(f["qty"]) for f in fills) / qty if qty else None


def _close_record(bk, tr, reason):
    realized = float(bk.realized_pnl_since(tr["sym"], tr["opened_ms"]))
    exit_px = _exit_px(bk, tr)
    slip = (tr["stop"] - exit_px) / tr["stop"] * 1e4 if (exit_px and reason == "exchange_closed") else None
    return _append("closed", id=tr["id"], sym=tr["sym"], reason=reason, exit_px=exit_px,
                   realized_usdt=realized,
                   realized_r=realized / tr["risk_usdt"] if tr.get("risk_usdt") else None,
                   stop_slippage_bps=slip)


def _manage(bk, tr, pos, paper, paper_closed):
    if pos is None or pos.get("side") != "long":
        # یک پاسخِ ناقص نباید معاملهٔ زنده را «بسته» ثبت کند و مدیریتِ حدضررش را رها کند
        if any(p["symbol"] == tr["sym"] and p.get("side") == "long" for p in bk.positions()):
            return []
        return [_close_record(bk, tr, "exchange_closed")]          # حدضرر خورد (یا دستی بسته شد)
    if paper is None and paper_closed:
        bk.close_symbol(tr["sym"])                                  # قاعده بیرون آمد؛ صرافی هنوز نه
        return [_close_record(bk, tr, "rule_exit_sync")]
    if paper is not None and float(paper["stop"]) > float(tr["stop"]) * (1 + 1e-9):
        new_id = bk.move_stop(tr["sym"], "long", tr["qty"], float(paper["stop"]))
        return [_append("stop_moved", id=tr["id"], sym=tr["sym"], old=tr["stop"],
                        new=float(paper["stop"]), order_id=new_id)]
    return []


def _enter(bk, tid, sig, live, now):
    sym = sig["sym"]
    bar_open_ms = int(sig["signal_ts"]) + DAY_MS
    if now * 1000 - bar_open_ms > ENTRY_WINDOW_SEC * 1000:
        return _append("missed", id=tid, sym=sym, reason="late")
    if sym in live:
        return _append("missed", id=tid, sym=sym, reason="foreign_position")
    try:
        step, _tick, min_qty, min_notional = bk.symbol_rules(sym)
    except brokermod.TestnetError:
        return _append("missed", id=tid, sym=sym, reason="not_on_testnet")
    bar = market.current_bar(sym, "1d")
    ref_ts, ref_open = int(bar["t"]), float(bar["o"])
    if ref_ts != bar_open_ms:
        raise RuntimeError(f"کندلِ جاریِ {sym} ({ref_ts}) کندلِ ورود ({bar_open_ms}) نیست")   # دورِ بعد
    spot, mark = market.spot_price(sym), bk.mark_price(sym)
    gap = abs(mark / spot - 1) * 100
    if gap > MAX_PRICE_GAP_PCT:
        return _append("missed", id=tid, sym=sym, reason="testnet_price_gap", gap_pct=round(gap, 3))
    stop = ref_open - float(sig["stop_distance"])                   # همان حدضررِ دفترِ کاغذی
    if mark <= stop:
        return _append("missed", id=tid, sym=sym, reason="below_stop")
    if float(bar["l"]) <= stop:
        # کفِ همین امروز حدضرر را زده: قاعدهٔ کاغذی همین حالا بیرون است — ورود یعنی معامله‌ای که آزموده نشده
        return _append("missed", id=tid, sym=sym, reason="stop_breached_today")
    _avail, equity = bk.balance_usdt()
    risk_usdt = float(equity) * gates.risk_caps()["max_risk_pct_per_trade"] / 100.0
    qty = brokermod.BinanceTestnet._round_step(risk_usdt / (mark - stop), step)
    if qty < min_qty or qty * mark < min_notional:
        return _append("missed", id=tid, sym=sym, reason="too_small", qty=qty)
    res = bk.open_with_stop(sym, "long", qty, stop)
    return _append("open", id=tid, sym=sym, qty=res["qty"], fill_px=res["fill"], ref_open=ref_open,
                   stop=stop, stop_order_id=res["stop_order_id"], order_id=res["order_id"],
                   risk_usdt=risk_usdt, stop_distance=float(sig["stop_distance"]),
                   signal_ts=int(sig["signal_ts"]), opened_ms=int(now * 1000),
                   entry_slippage_bps=(res["fill"] / ref_open - 1) * 1e4, mark_gap_pct=round(gap, 3))


def _done(result, now):
    _last.update(at=now, result=result)
    return result


def step(bk=None, snap=None, now=None):
    """یک دور: همگام‌سازی با صرافی، جابه‌جاییِ حدضرر، ورودهای تازه. خطا ثبت می‌شود و دورِ بعد تکرار."""
    now = time.time() if now is None else now
    permitted = gates.testnet_research_allowed(STRATEGY)
    trades, decided = replay()
    open_trades = [tr for tr in trades.values() if tr["status"] == "open"]
    if not permitted and not open_trades:
        return _done({"status": "not_permitted"}, now)
    try:
        bk = brokermod.make_broker() if bk is None else bk
    except Exception as e:  # noqa: BLE001
        return _done({"status": "broker_error", "error": str(e)}, now)
    bk = testnet_client(bk)
    if bk is None:
        return _done({"status": "no_testnet"}, now)
    with _lock:
        if snap is None:
            snap = trend.snapshot()
            if (snap.get("last_closed_bar_ms") or 0) + 2 * DAY_MS <= now * 1000:
                snap = trend.snapshot(force=True)                   # روزِ تازه بسته شده؛ کش کهنه است
        rule = snap["rules"][RULE]
        paper_open = {p["id"]: p for p in rule["open"]}
        paper_closed = {r["id"] for r in trend._read() if r.get("kind") == "exit"}
        paper_closed |= {r["id"] for r in rule.get("closed_recent") or []}
        live = {p["symbol"]: p for p in bk.positions()}
        acted = 0
        for tr in open_trades:
            try:
                acted += len(_manage(bk, tr, live.get(tr["sym"]), paper_open.get(tr["id"]),
                                     tr["id"] in paper_closed))
            except Exception as e:  # noqa: BLE001
                _append("error", id=tr["id"], sym=tr["sym"], stage="manage", msg=str(e))
        if permitted:
            for p in rule["open"]:
                if p["id"] not in decided:          # سیگنالی که به‌موقع دیده نشد — دنبالش نمی‌رویم
                    _append("missed", id=p["id"], sym=p["sym"], reason="late")
                    decided.add(p["id"])
            for sig in rule["signals"]:
                tid = f"{RULE}:{sig['sym']}:{int(sig['signal_ts'])}"
                if tid in decided:
                    continue
                try:
                    _enter(bk, tid, sig, live, now)
                    decided.add(tid)
                    acted += 1
                except Exception as e:  # noqa: BLE001
                    _append("error", id=tid, sym=sig["sym"], stage="enter", msg=str(e))
        return _done({"status": "running" if permitted else "winding_down", "actions": acted}, now)


def close_all(reason, bk=None):
    """قطعِ اضطراری: همهٔ پوزیشن‌هایی که این ماژول باز کرده بسته می‌شوند."""
    trades, _ = replay()
    open_trades = [tr for tr in trades.values() if tr["status"] == "open"]
    if not open_trades:
        return 0
    try:
        bk = testnet_client(brokermod.make_broker() if bk is None else bk)
    except Exception:  # noqa: BLE001
        log.exc("trend kill: broker")
        return 0
    if bk is None:
        return 0
    closed = 0
    for tr in open_trades:
        try:
            bk.close_symbol(tr["sym"])
            _close_record(bk, tr, f"kill: {reason}")
            closed += 1
        except Exception:  # noqa: BLE001
            log.exc("trend kill: close", symbol=tr["sym"])
    return closed


# ───────────────────────── کارنامه ─────────────────────────
def _mean(xs):
    xs = [float(x) for x in xs if x is not None]
    return round(float(np.mean(xs)), 3) if xs else None


def summary():
    evs = events()
    trades, _ = replay(evs)
    paper = {r["id"]: r for r in trend._read() if r.get("kind") == "exit"}
    rows = sorted(trades.values(), key=lambda t: -float(t.get("opened_ms") or 0))
    closed = [t for t in rows if t["status"] == "closed"]
    missed = [e for e in evs if e.get("kind") == "missed"]
    by_reason = {}
    for e in missed:
        by_reason[e.get("reason")] = by_reason.get(e.get("reason"), 0) + 1
    gaps = [t["realized_r"] - paper[t["id"]]["net_r"] for t in closed
            if t.get("realized_r") is not None and t["id"] in paper]
    return {
        "strategy": STRATEGY,
        "rule": RULE,
        "permitted": gates.testnet_research_allowed(STRATEGY),
        "last_step": dict(_last),
        "open": [t for t in rows if t["status"] == "open"],
        "closed": [{**t, "paper_r": (paper.get(t["id"]) or {}).get("net_r")} for t in closed][:20],
        "execution": {
            "entries": len(rows),
            "missed": len(missed),
            "missed_by_reason": by_reason,
            "entry_rate": round(len(rows) / (len(rows) + len(missed)), 3) if (rows or missed) else None,
            "entry_slippage_bps_mean": _mean(t.get("entry_slippage_bps") for t in rows),
            "stop_slippage_bps_mean": _mean(t.get("stop_slippage_bps") for t in closed),
            "realized_minus_paper_r_mean": _mean(gaps),
            "closed": len(closed),
            "realized_r_sum": round(sum(float(t["realized_r"]) for t in closed
                                        if t.get("realized_r") is not None), 3),
        },
        "errors": [e for e in evs if e.get("kind") == "error"][-5:],
    }
