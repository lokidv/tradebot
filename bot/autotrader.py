# -*- coding: utf-8 -*-
"""🤖 ربات معامله‌گرِ خودکار — با مغزِ هوش مصنوعیِ موجود (مدل‌های کالیبره) تصمیم می‌گیرد،
شفاف «فکر» می‌کند (جریان افکار زنده)، فقط دمو معامله می‌کند، و ضریبِ فعالیت ۱ تا ۱۰ دارد.
سطح بالاتر آستانه را کمی پایین می‌آورد و ظرفیت را زیاد می‌کند، اما هیچ سطحی
ربات را مجبور به معامله یا استفاده از پیش‌بینیِ جهتِ بدون ستاپ نمی‌کند."""
import json
import os
import threading
import time
from collections import deque

import advisor
import fill_quality
import gates
import market
import paper

CFG_PATH = os.path.join(os.path.dirname(__file__), "data", "autobot.json")
CYCLE_SEC = 45                     # فاصلهٔ چرخه‌های فکر کردن (تحلیل و شکار)
MONITOR_SEC = 12                   # پایشِ لحظه‌ایِ پوزیشن‌های باز (تریلینگ/برداشت پله‌ای/حدها)

_lock = threading.Lock()
_thoughts = deque(maxlen=150)      # جریان افکار (جدیدترین اول)
_status = {"current": "خاموش", "last_cycle": None, "next_cycle": None, "cycle_n": 0}
_fns = {}                          # تزریق وابستگی از main (بدون import چرخه‌ای)
_started = False
_cooldown = {}                     # symbol -> تا این زمان دوباره واردش نشو (بعد از خروجِ چرخشی/حدضرر)
COOLDOWN_SEC = 1800
_peak = {}                         # pos_id -> بهترین R دیده‌شده (برای حدضررِ متحرک)
_pending = {}                      # symbol -> سفارشِ صبور (لیمیتِ بازگشتی — تعقیبِ قیمت ممنوع)
PENDING_TTL = {"15m": 300, "1h": 1200, "4h": 4800, "1d": 14400}   # مهلتِ پر شدن (ثانیه)
_announced = set()                 # بسته‌شده‌هایی که یک‌بار اعلام شده‌اند (جلوگیری از اعلامِ دوباره)
_storm_until = 0.0                 # ⛈ ترمزِ طوفان: تا این زمان پوزیشنِ جدید باز نکن
_risk_off_day = None               # 🛑 روزی که حدِ ضررِ روزانه فعال شده (توقفِ ورود تا فردا)
_toxic_1d_pass = 0.0               # آخرین گذرِ پاکسازی پوزیشن‌های سمی ۱روزه


def init(overview_fn, drift_cap, analysis_fn=None):
    _fns["overview"] = overview_fn
    _fns["drift_cap"] = drift_cap
    _fns["analysis"] = analysis_fn


# ───────────────────── پیکربندی ─────────────────────
def load_cfg():
    try:
        with open(CFG_PATH, "r", encoding="utf-8") as f:
            d = json.load(f)
    except Exception:  # noqa: BLE001
        d = {}
    return {"enabled": bool(d.get("enabled", False)),
            "level": max(1, min(int(d.get("level", 5)), 10))}


def save_cfg(cfg):
    os.makedirs(os.path.dirname(CFG_PATH), exist_ok=True)
    tmp = CFG_PATH + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False)
    os.replace(tmp, CFG_PATH)


def level_params(lv):
    """فعالیت، آستانه و ظرفیت را تغییر می‌دهد؛ هرگز ربات را مجبور به معامله نمی‌کند.
    اسکن پیش‌فرض روی تایم‌فریم‌های با لبهٔ زنده (۱h اول) است — نه ۴h زیان‌ده."""
    lv = max(1, min(int(lv), 10))
    # 15m هرگز اسکن نمی‌شود: هزینهٔ رفت‌وبرگشت ~۰٫۲R هر معامله و بازده خالصِ تاریخی −۰٫۲۱R
    scan = ["1h"] if lv <= 3 else ["1h", "4h"]
    return {
        "level": lv,
        "min_score": round(82 - lv * 2.5),                 # ۱→۸۰، ۱۰→۵۷
        "min_open": 0,
        "max_open": 2 + (lv - 1) // 3,                    # ۲ تا حداکثر ۵
        "risk_pct_equity": round(0.25 + lv * 0.025, 2),   # ۰٫۲۸٪ تا حداکثر ۰٫۵٪
        "heat_cap": round(1.25 + lv * 0.15, 1),           # ۱٫۴٪ تا حداکثر ۲٫۸٪
        "scan_tfs": scan,
        "allow_lean": False,
        "opens_per_cycle": 1 if lv <= 6 else 2,
    }


# ───────────────────── افکار ─────────────────────
def think(msg, kind="info"):
    with _lock:
        _thoughts.appendleft({"t": time.time(), "msg": msg, "kind": kind})
        _status["current"] = msg


def bot_stats():
    db = paper.list_positions()
    closed = [p for p in db["closed"] if p.get("opened_by") == "bot"]
    op = [p for p in db["open"] if p.get("opened_by") == "bot"]
    wins = [p for p in closed if p.get("pnl_usdt", 0) > 0]
    pnl = sum(p.get("pnl_usdt", 0) for p in closed)
    return {"open": len(op), "trades": len(closed), "wins": len(wins),
            "win_rate": round(len(wins) / len(closed) * 100, 1) if closed else None,
            "pnl": round(pnl, 2),
            "open_pnl": round(sum(p.get("pnl_usdt", 0) for p in op), 2),
            "open_list": [{"symbol": p["symbol"], "tf": p["tf"], "side": p["side"],
                           "pnl_pct": p.get("pnl_pct")} for p in op]}


def get_state():
    cfg = load_cfg()
    with _lock:
        th = list(_thoughts)[:40]
        st = dict(_status)
    now = time.time()
    return {"enabled": cfg["enabled"], "level": cfg["level"],
            "params": level_params(cfg["level"]),
            "current": st["current"], "cycle_n": st["cycle_n"],
            "last_cycle": st["last_cycle"], "next_cycle": st["next_cycle"],
            "thoughts": th, "stats": bot_stats(), "cycle_sec": CYCLE_SEC,
            "guards": {"storm": now < _storm_until,
                       "risk_off": _risk_off_day == time.strftime("%Y-%m-%d", time.gmtime()),
                       "monitor_sec": MONITOR_SEC}}


def set_config(enabled=None, level=None):
    cfg = load_cfg()
    if enabled is not None:
        cfg["enabled"] = bool(enabled)
    if level is not None:
        cfg["level"] = max(1, min(int(level), 10))
    save_cfg(cfg)
    if enabled is True:
        think(f"✳️ روشن شدم — سطحِ فعالیت {cfg['level']} از ۱۰. اولین چرخهٔ بررسی تا چند ثانیهٔ دیگر.", "open")
    elif enabled is False:
        think("⏸ خاموش شدم — پوزیشن‌های بازِ قبلی طبق حدضرر/هدف خودکار مدیریت می‌شوند.", "warn")
    if level is not None and enabled is None:
        p = level_params(cfg["level"])
        think(f"🎚 سطحِ فعالیت شد {cfg['level']}: بدون اجبار به معامله، "
              f"سقف {p['max_open']} پوزیشن، آستانهٔ امتیاز {p['min_score']}+")
    return get_state()


# ───────────────────── منطقِ معامله ─────────────────────
def _live(symbol):
    try:
        return market.last_price(symbol)
    except Exception:  # noqa: BLE001
        return None


def _portfolio_risk_usd(db, side=None):
    """ریسک تا حدضرر؛ با side می‌توان تمرکز جهت‌دار (همبستگی بازار کریپتو) را جدا سنجید."""
    return sum(
        p["size_usdt"] * abs(p["entry"] - p["sl"]) / max(p["entry"], 1e-12)
        for p in db["open"] if side is None or p.get("side") == side
    )


def _pending_risk_usd(side=None, exclude_symbol=None):
    return sum(
        float(o.get("size") or 0) * float(o.get("r_abs") or 0) / max(float(o.get("target") or 0), 1e-12)
        for sym, o in _pending.items()
        if sym != exclude_symbol and (side is None or o.get("side") == side)
    )


def _r0(p):
    return float(p.get("r0") or abs(p["entry"] - p["sl"])) or 0.0


def _announce_closes(db, before_ids):
    """اعلامِ بسته‌شده‌های ربات (فقط یک‌بار) + کول‌داونِ ضدِ معاملهٔ انتقامی بعد از حدضرر."""
    for p in db["closed"][:15]:
        if p.get("opened_by") != "bot" or p["id"] not in before_ids or p["id"] in _announced:
            continue
        _announced.add(p["id"])
        win = p.get("pnl_usdt", 0) > 0
        think(f"{'✅' if win else '🔴'} بسته شد: {p['symbol'].replace('USDT','')} "
              f"({p.get('close_reason')}) → {p.get('pnl_usdt'):+.1f}$", "open" if win else "close")
        # کول‌داونِ همگانی بعد از هر بسته‌شدن: ضدِ انتقام بعد از ضرر (۳۰ دق) و ضدِ حلقهٔ بازشدنِ فوری بعد از هر خروج (۱۵ دق)
        cd = COOLDOWN_SEC if p.get("pnl_usdt", 0) <= 0 else 900
        _cooldown[p["symbol"]] = max(_cooldown.get(p["symbol"], 0), time.time() + cd)
    if len(_announced) > 400:
        _announced.clear()


def _daily_pnl_bot(db):
    """زیان روزانهٔ ربات؛ زیان باز هم برای ترمز ریسک حساب می‌شود."""
    today = time.strftime("%Y-%m-%d", time.gmtime())
    realized = sum(p.get("pnl_usdt", 0) for p in db["closed"]
                   if p.get("opened_by") == "bot" and str(p.get("closed_at", "")).startswith(today))
    open_loss = sum(min(p.get("pnl_usdt", 0), 0) for p in db["open"]
                    if p.get("opened_by") == "bot")
    return realized + open_loss


def _loss_streak(db):
    """تعدادِ باخت‌های پیاپیِ اخیرِ ربات (جدیدترین اول) — معاملاتِ تقریباً-صفر (زیر ۱$) نویزند، نه باخت."""
    n = 0
    for p in db["closed"]:
        if p.get("opened_by") != "bot" or abs(p.get("pnl_usdt", 0)) < 1.0:
            continue
        if p.get("pnl_usdt", 0) <= 0:
            n += 1
        else:
            break
    return n


def _retire_toxic_1d():
    """کارنامهٔ زندهٔ ۱روزه ~−۰٫۳۸R است — پوزیشن‌های ربات روی ۱d را خارج کن (حفظ سرمایه)."""
    global _toxic_1d_pass
    now = time.time()
    if now - _toxic_1d_pass < 90:
        return
    _toxic_1d_pass = now
    db = paper.list_positions()
    toxic = [p for p in db["open"]
             if p.get("opened_by") == "bot" and p.get("tf") == "1d"
             and p.get("mode") != "testnet"]
    if not toxic:
        return
    think(f"🧹 {len(toxic)} پوزیشن سمی روی تایم‌فریم روزانه — کارنامه زنده زیان‌ده است؛ در حال خروج…", "warn")
    for p in toxic:
        lp = _live(p["symbol"]) or p.get("last_price") or p["entry"]
        pnl = p.get("pnl_usdt") or 0
        reason = ("ربات: قفل سود — خروج از ۱d سمی" if pnl >= 0
                  else "ربات: قطع زیان — تایم‌فریم ۱d در کارنامه زنده سمی است")
        paper.close_with(p["id"], lp, reason)
        _cooldown[p["symbol"]] = now + COOLDOWN_SEC
        think(f"{'💰' if pnl >= 0 else '✂️'} {p['symbol'].replace('USDT','')}: {reason} "
              f"({pnl:+.1f}$)", "close")


def _process_pending():
    """🎯 سفارش‌های صبور: قیمت به لیمیت رسید → باز کن؛ مهلت گذشت → با دیسیپلین رد شو."""
    now = time.time()
    if not _pending:
        return
    if _risk_off_day == time.strftime("%Y-%m-%d", time.gmtime()) or now < _storm_until:
        for sym, o in list(_pending.items()):
            fill_quality.log_event("cancelled", symbol=sym, tf=o.get("tf"), side=o.get("side"),
                                   why="storm_or_risk_off", authority=o.get("authority"))
        _pending.clear()
        return
    db = paper.list_positions()
    held = {p["symbol"] for p in db["open"]}
    equity = paper.wallet_summary()["equity"]
    for sym in list(_pending):
        o = _pending.get(sym)
        if not o:
            continue
        if now > o["expires"]:
            _pending.pop(sym, None)
            fill_quality.log_event("expired", symbol=sym, tf=o.get("tf"), side=o.get("side"),
                                   target=o.get("target"), score=o.get("score"),
                                   authority=o.get("authority"))
            think(f"⌛ {sym.replace('USDT','')}: قیمت به سفارشِ صبور برنگشت — تعقیب نکردم و رد شدم (دیسیپلین > فومو).", "info")
            continue
        if sym in held or _cooldown.get(sym, 0) > now:
            _pending.pop(sym, None)
            fill_quality.log_event("cancelled", symbol=sym, tf=o.get("tf"), side=o.get("side"),
                                   why="held_or_cooldown", authority=o.get("authority"))
            continue
        lp = _live(sym)
        if lp is None:
            continue
        d = 1 if o["side"] == "long" else -1
        if d * (lp - o["target"]) <= 0:                      # قیمت به هدفِ صبور رسید (یا بهتر)
            analysis_fn = _fns.get("analysis")
            if analysis_fn is not None:
                fresh = analysis_fn(sym, o["tf"], max_age=1)
                tr = (fresh or {}).get("trade") or {}
                P = level_params(load_cfg()["level"])
                need = P["min_score"]
                auth = tr.get("authority") or o.get("authority")
                bad = ((fresh or {}).get("error") or not tr.get("tradeable")
                       or tr.get("side") != o["side"]
                       or (tr.get("signal_score") or 0) < need
                       or bool(tr.get("regime_veto"))
                       or not tr.get("policy_trusted") or not tr.get("policy_pass")
                       or not tr.get("gate_allowed"))
                if bad:
                    _pending.pop(sym, None)
                    fill_quality.log_event("cancelled", symbol=sym, tf=o.get("tf"), side=o.get("side"),
                                           why="revalidate_fail", authority=auth)
                    think(f"🚫 {sym.replace('USDT','')}: هنگام رسیدن قیمت، مدل/ستاپ دوباره تأیید نشد؛ سفارش صبور لغو شد.", "warn")
                    continue
            # لغزش نسبت به هدف: منفی = ارزان‌تر (خوب برای لانگ)
            r_abs = max(float(o.get("r_abs") or 0), 1e-12)
            slip_r = d * (lp - o["target"]) / r_abs
            # هزینهٔ واقع‌گرایانه: اگر پر شدن بدتر از هدف بود، کمی هزینه اضافه
            cost = float(o.get("cost") or 0.15)
            if slip_r > 0.05:
                cost = min(cost + 0.03, 0.28)
            sl = lp - d * o["r_abs"]
            tp = lp + d * o["tp_abs"]
            new_risk = o["size"] * abs(lp - sl) / max(lp, 1e-12)
            P = level_params(load_cfg()["level"])
            bot_open = [p for p in db["open"] if p.get("opened_by") == "bot"]
            same_side = sum(1 for p in bot_open if p.get("side") == o["side"])
            max_same = max(2, round(P["max_open"] * 0.6))
            if len(bot_open) >= P["max_open"] or same_side >= max_same:
                _pending.pop(sym, None)
                fill_quality.log_event("cancelled", symbol=sym, why="capacity", authority=o.get("authority"))
                think(f"🛑 {sym.replace('USDT','')}: هنگام پرشدن، سقف تعداد/تمرکز پورتفو پر بود؛ لغو شد.", "warn")
                continue
            reserved = _pending_risk_usd(exclude_symbol=sym)
            if (_portfolio_risk_usd(db) + reserved + new_risk) / max(equity, 1e-9) * 100 > P["heat_cap"]:
                _pending.pop(sym, None)
                fill_quality.log_event("cancelled", symbol=sym, why="heat", authority=o.get("authority"))
                think(f"🛑 {sym.replace('USDT','')}: سفارشِ صبور رسید ولی سقفِ ریسکِ پرتفوی پر است — لغو شد.", "warn")
                continue
            side_reserved = _pending_risk_usd(o["side"], exclude_symbol=sym)
            side_heat = (_portfolio_risk_usd(db, o["side"]) + side_reserved + new_risk) / max(equity, 1e-9) * 100
            if side_heat > P["heat_cap"] * 0.65:
                _pending.pop(sym, None)
                fill_quality.log_event("cancelled", symbol=sym, why="side_heat", authority=o.get("authority"))
                think(f"🛑 {sym.replace('USDT','')}: ریسکِ هم‌جهت به {side_heat:.1f}٪ می‌رسید — سفارش لغو شد.", "warn")
                continue
            paper.open_position(sym, o["tf"], o["side"], lp, sl, tp, o["size"], o["tstop"],
                                grade=o["grade"], opened_by="bot", cost_pct=cost)
            _pending.pop(sym, None)
            fill_quality.log_event("filled", symbol=sym, tf=o.get("tf"), side=o.get("side"),
                                   target=o.get("target"), fill=lp, slip_r=round(slip_r, 3),
                                   cost_pct=cost, score=o.get("score"), authority=o.get("authority"))
            db = paper.list_positions()
            equity = paper.wallet_summary()["equity"]
            held.add(sym)
            think(f"🟢 سفارشِ صبور پر شد: {('لانگ ▲' if o['side'] == 'long' else 'شورت ▼')} {sym.replace('USDT','')} "
                  f"در {lp:.6g} (لغزش {slip_r:+.2f}R) — ورودِ ارزان‌تر بدونِ تعقیب "
                  f"(امتیاز {o['score']}, حجم {o['size']:.0f}$).", "open")


def _monitor():
    """👁 پایشِ لحظه‌ای (هر ۱۲ ثانیه، جدا از چرخهٔ تحلیل): سفارش‌های صبور، حدها، برداشتِ پله‌ای، حدضررِ متحرک."""
    cfg = load_cfg()
    if not cfg["enabled"]:
        _pending.clear()
        # خاموش‌کردنِ ورودی‌های تازه نباید حفاظتِ پوزیشن‌های قبلی را خاموش کند.
        db = paper.list_positions()
        bot_local = [p for p in db["open"] if p.get("opened_by") == "bot"
                     and p.get("mode") != "testnet"]
        before_ids = {p["id"] for p in bot_local}
        prices = {}
        for p in bot_local:
            if p["symbol"] not in prices:
                lp = _live(p["symbol"])
                if lp:
                    prices[p["symbol"]] = lp
        if prices:
            _announce_closes(paper.refresh(prices), before_ids)
        try:
            _retire_toxic_1d()
        except Exception:  # noqa: BLE001
            pass
        return
    try:
        _retire_toxic_1d()
    except Exception:  # noqa: BLE001
        pass
    try:
        _process_pending()
    except Exception:  # noqa: BLE001
        pass
    db = paper.list_positions()
    bot_local = [p for p in db["open"] if p.get("opened_by") == "bot" and p.get("mode") != "testnet"]
    if not bot_local:
        return
    before_ids = {p["id"] for p in bot_local}
    prices = {}
    for p in bot_local:
        if p["symbol"] not in prices:
            lp = _live(p["symbol"])
            if lp:
                prices[p["symbol"]] = lp
    db = paper.refresh(prices)                      # برخوردِ حدضرر/هدف/حدزمانی همین‌جا بسته می‌شود
    _announce_closes(db, before_ids)

    # خوشهٔ همبسته؟ وقتی ≥۶ پوزیشنِ هم‌جهت باز است، سودها با هم می‌آیند و با هم می‌روند →
    # زودتر نقد کن (درسِ پایش: +۳۵$ شناور در ۲ ساعت دود شد چون هیچ‌کدام به +۱R نرسیده بود)
    side_n = {}
    for p in db["open"]:
        if p.get("opened_by") == "bot" and p.get("mode") != "testnet":
            side_n[p["side"]] = side_n.get(p["side"], 0) + 1

    open_ids = set()
    for p in db["open"]:
        if p.get("opened_by") != "bot" or p.get("mode") == "testnet":
            continue
        open_ids.add(p["id"])
        price = prices.get(p["symbol"])
        r0 = _r0(p)
        if price is None or r0 <= 0:
            continue
        d = 1 if p["side"] == "long" else -1
        pnl_r = d * (price - p["entry"]) / r0
        peak = _peak[p["id"]] = max(_peak.get(p["id"], pnl_r), pnl_r)
        nm = p["symbol"].replace("USDT", "")
        cluster = side_n.get(p["side"], 0) >= 6
        need_r, trail_from, trail_gap = (0.6, 1.0, 0.6) if cluster else (1.0, 1.3, 0.8)

        # 💰 برداشتِ پله‌ای: نصف را نقد کن و حدضرر را به سربه‌سر ببر — بقیه بدونِ ریسک می‌دود
        if pnl_r >= need_r and not p.get("partial_done"):
            part = paper.partial_close(p["id"], price, 0.5)
            if part:
                paper.move_sl(p["id"], p["entry"])
                think(f"💰 {nm} به +{pnl_r:.1f}R رسید: نصف را نقد کردم ({part['pnl_usdt']:+.1f}$) و حدضررِ باقی‌مانده سربه‌سر شد"
                      + (" — چون خوشهٔ بزرگِ هم‌جهت داریم، زودتر از معمول قفل کردم." if cluster else "."), "open")
            continue

        # 🛡 حدضررِ متحرک: از اوج به بعد، عقب‌ترِ اوج را قفل کن (فقط سفت‌تر، با گامِ معنادار)
        if peak >= trail_from:
            want_sl = p["entry"] + d * (peak - trail_gap) * r0
            if d * (want_sl - p["sl"]) >= 0.15 * r0:
                if paper.move_sl(p["id"], want_sl):
                    think(f"🛡 {nm}: اوجِ سود +{peak:.1f}R — حدضررِ متحرک را جلو کشیدم؛ "
                          f"حداقل +{peak - trail_gap:.1f}R قفل شد.", "open")

    for pid in list(_peak):                          # پاک‌سازیِ اوج‌های پوزیشن‌های بسته‌شده
        if pid not in open_ids:
            _peak.pop(pid, None)


def _open_from_candidate(c, tf, kind, P, equity, db, risk_mult=1.0):
    """بازکردنِ پوزیشن از یک کاندید (سیگنالِ تأییدشده یا پیش‌بینیِ جهت). خروجی: پیام یا None."""
    if kind != "signal":
        think(f"🚫 {c.get('symbol', '')}: ورود بدون ستاپِ تأییدشده مجاز نیست.", "warn")
        return None
    sym = c["symbol"]
    live = _live(sym)
    if live is None:
        think(f"🚫 {sym.replace('USDT','')}: قیمتِ زنده در دسترس نیست — رد شد", "warn")
        return None
    if kind == "signal":
        entry0, sl0, tp0 = c.get("entry"), c.get("sl"), c.get("tp")
        side = c.get("side")
        risk_pct = c.get("risk_pct") or 2.0
        tstop = c.get("time_stop_min") or 40 * {"15m": 15, "1h": 60, "4h": 240, "1d": 1440}[tf]
        drift = abs(live - entry0) / max(entry0, 1e-12) * 100
        cap = _fns.get("drift_cap", {}).get(tf, 2.5)
        if drift > cap:
            think(f"🚫 {sym.replace('USDT','')}: قیمت {drift:.1f}٪ از کندلِ سیگنال دور شده (حد {cap}٪) — تعقیب نمی‌کنم", "warn")
            return None
        d = 1 if side == "long" else -1
        sl = live - d * abs(entry0 - sl0)
        tp = live + d * abs(tp0 - entry0)
    else:                                   # پیش‌بینیِ جهت (lean)
        p_up = c.get("p_up", 50)
        side = "long" if p_up >= 52 else "short"
        d = 1 if side == "long" else -1
        risk_pct = {"15m": 0.6, "1h": 0.8, "4h": 1.2, "1d": 2.0}[tf]
        r = live * risk_pct / 100
        sl, tp = live - d * r, live + d * 1.8 * r
        tstop = 40 * {"15m": 15, "1h": 60, "4h": 240, "1d": 1440}[tf]
    # حجم بر اساس ریسک (٪ از موجودی) — دقت و بقا مقدم بر هیجان
    # سایزدهی محافظه‌کارانه با اطمینان مدل و رتبهٔ مقطعی؛ هیچ سیگنالی بیش از بودجهٔ پایه ریسک نمی‌گیرد.
    conv = 1.0
    if kind == "signal":
        confidence = max(0.0, min(float(c.get("model_confidence") or 0.0), 1.0))
        rank_factor = 0.8 + 0.2 * max(0.0, min(float(c.get("market_rank_pct") or 0.0) / 100.0, 1.0))
        meta_mult = c.get("meta_size_mult")
        if meta_mult is None:
            meta_mult = 1.0
        meta_mult = max(0.0, min(float(meta_mult), 1.2))
        conv = (0.55 + 0.45 * confidence) * rank_factor * meta_mult
        if meta_mult <= 0:
            think(f"🚫 {sym.replace('USDT','')}: متا-گیت حجم را صفر کرده — ورود رد شد", "warn")
            return None
    risk_usd = equity * P["risk_pct_equity"] / 100 * conv * risk_mult
    size = min(risk_usd / (risk_pct / 100), equity * 0.15)
    if size < 25.0:
        think(f"🚫 {sym.replace('USDT','')}: حجم لازم زیر حداقل عملی ۲۵$ است؛ برای رعایت ریسک معامله رد شد.", "warn")
        return None
    new_risk = size * risk_pct / 100
    heat_after = (_portfolio_risk_usd(db) + _pending_risk_usd() + new_risk) / max(equity, 1e-9) * 100
    if heat_after > P["heat_cap"]:
        think(f"🛑 {sym.replace('USDT','')} رد شد: جمعِ ریسک به {heat_after:.1f}٪ می‌رسید (سقفِ سطحِ {P['level']}: {P['heat_cap']}٪)", "warn")
        return None
    side_heat_after = (_portfolio_risk_usd(db, side) + _pending_risk_usd(side) + new_risk) / max(equity, 1e-9) * 100
    if side_heat_after > P["heat_cap"] * 0.65:
        think(f"🛑 {sym.replace('USDT','')} رد شد: ریسکِ هم‌جهت به {side_heat_after:.1f}٪ می‌رسید "
              f"(سقف جهت‌دار {P['heat_cap'] * 0.65:.1f}٪)", "warn")
        return None
    nm = sym.replace("USDT", "")
    if kind == "signal":
        # 🎯 ورودِ صبور: تعقیبِ قیمت ممنوع — لیمیت در ~۰.۳۳×ATR بهتر و صبر؛ نیامد؟ رد می‌شویم.
        # (کالبدشکافی ORDI/TIA/ALLO: ورودِ فوری به‌طور سیستماتیک ~۲۷٪ از R گران‌تر می‌خرید)
        target = live - d * 0.25 * (risk_pct / 100) * live
        meta_note = ""
        if c.get("meta_confluence") is not None:
            meta_note = f" · متا {float(c.get('meta_score') or 0)*100:.0f}%/تلاقی {c.get('meta_confluence')}"
        if c.get("authority"):
            meta_note += f" · مرجع {c.get('authority')}"
        _pending[sym] = {"side": side, "target": target, "tf": tf,
                         "r_abs": abs(entry0 - sl0), "tp_abs": abs(tp0 - entry0),
                         "size": round(size, 1), "tstop": tstop, "grade": c.get("grade") or "",
                         "score": c.get("signal_score"), "cost": c.get("cost") or 0.15,
                         "signal_ts": c.get("zt"), "expires": time.time() + PENDING_TTL.get(tf, 1200),
                         "meta_size_mult": c.get("meta_size_mult"),
                         "authority": c.get("authority")}
        think(f"🎯 {nm}: سفارشِ صبور در {target:.6g} (زنده {live:.6g}) — "
              f"حجم {size:.0f}$ (ریسک {new_risk:.0f}$){meta_note} — "
              f"حداکثر {PENDING_TTL.get(tf, 1200) // 60} دقیقه", "info")
        return sym
    paper.open_position(sym, tf, side, live, sl, tp, round(size, 1), tstop,
                        grade=c.get("grade") or "جهت", opened_by="bot",
                        cost_pct=c.get("cost") or 0.15)
    think(f"🟡 {('لانگ' if side=='long' else 'شورت')} {nm} روی {tf} از روی پیش‌بینیِ جهت "
          f"(p={c.get('p_up')}٪ — کم‌اعتبارتر؛ برای حفظِ کفِ {P['min_open']} پوزیشن در سطح {P['level']}), "
          f"حجم {size:.0f}$، ریسک {new_risk:.0f}$", "open")
    return sym


def _cycle():
    cfg = load_cfg()
    with _lock:
        _status["cycle_n"] += 1
        n = _status["cycle_n"]
    if not cfg["enabled"]:
        with _lock:
            _status["current"] = "خاموش — منتظرِ روشن‌شدن"
        return
    P = dict(level_params(cfg["level"]))
    ov_fn = _fns.get("overview")
    if ov_fn is None:
        return
    allowed = gates.allowed_combos()
    think(f"🔄 چرخهٔ {n}: پایشِ بازار روی {'/'.join(P['scan_tfs'])} "
          f"(سطح {P['level']}, آستانه {P['min_score']}+ · "
          f"ترکیب‌های مجاز در gates: {len(allowed)})…")
    if not allowed:
        think("🔒 قفلِ ایمنی: هیچ ترکیبی در gates.json مجاز نیست — فقط پایش و ثبت؛ "
              "تا پیش‌ثبت و عبور از آزمونِ منجمد هیچ پوزیشنی باز نمی‌شود.", "warn")

    # ۱) مدیریتِ پوزیشن‌های باز (حدضرر/هدف/حدزمانی) + خروج از ۱d سمی
    try:
        _retire_toxic_1d()
    except Exception:  # noqa: BLE001
        pass
    db = paper.list_positions()
    before_ids = {p["id"] for p in db["open"] if p.get("opened_by") == "bot"}
    prices = {}
    for p in db["open"]:
        if p["symbol"] not in prices:
            lp = _live(p["symbol"])
            if lp:
                prices[p["symbol"]] = lp
    db = paper.refresh(prices)
    _announce_closes(db, before_ids)

    # ۲) عکسِ فوری از بازار
    ovs = {}
    for tf in P["scan_tfs"]:
        try:
            ov = ov_fn(tf)
            if ov.get("live_suspended") is not None:
                think(f"⛔ تایم‌فریم {tf} معلق است (بازده خالصِ زنده: {ov['live_suspended']:+.2f}R) — "
                      f"تا بهبودِ آمار واردش نمی‌شوم؛ سرمایه به تایم‌فریم‌های سودده می‌رود.", "warn")
                continue
            ovs[tf] = ov
        except Exception as e:  # noqa: BLE001
            think(f"⚠️ داده‌های {tf} در دسترس نیست: {e}", "warn")
    if not ovs:
        return

    # ۳) خروجِ هوشمند: مشاورِ پوزیشن روی تک‌تکِ پوزیشن‌های ربات
    #    (چرخشِ مدل، EV منفیِ نگه‌داشتن، قفلِ سودِ کم‌رمق — همان موتوری که به کاربر توصیه می‌دهد)
    bot_open = [p for p in db["open"] if p.get("opened_by") == "bot"]
    for p in bot_open:
        ov = ovs.get(p["tf"])
        if not ov:
            continue
        row = next((c for c in ov.get("coins", []) if c.get("symbol") == p["symbol"]), None)
        if row is not None and "مرده" in str(row.get("error") or ""):
            # ⚰️ جفتِ حذف‌شده/فیدِ یخ‌زده: پول را در چارتِ مرده حبس نکن
            paper.close_with(p["id"], p.get("last_price") or p["entry"], "ربات: جفتِ حذف‌شده — فیدِ مرده")
            think(f"⚰️ {p['symbol'].replace('USDT','')}: فیدِ داده مرده است (جفتِ حذف‌شده) — بستم و در لیستِ سیاهِ موقت گذاشتم.", "close")
            _cooldown[p["symbol"]] = time.time() + 86400
            continue
        if not row or row.get("p_up") is None or not row.get("p_calibrated"):
            continue
        adv = advisor.advise(p, {"p_up": row["p_up"]}, btc_z=(ov.get("btc") or {}).get("z"))
        if adv["urgency"] >= 2 and adv["action"] in ("close_now", "lock_profit"):
            lp = prices.get(p["symbol"]) or _live(p["symbol"])
            if lp:
                reason = "ربات: قفل سود 🧭" if adv["action"] == "lock_profit" else "ربات: مشاور — " + adv["title"]
                paper.close_with(p["id"], lp, reason)
                _cooldown[p["symbol"]] = time.time() + COOLDOWN_SEC
                think(f"🧭 {p['symbol'].replace('USDT','')}: {adv['title']} — {adv['reasons'][0]}. "
                      f"تا ۳۰ دقیقه سراغش نمی‌روم.", "close")
        elif adv["action"] == "move_be" and p.get("mode") != "testnet":
            if paper.move_sl(p["id"], p["entry"]):
                think(f"🛡️ {p['symbol'].replace('USDT','')}: سود {adv['pnl_r']}R شد — حدضرر را به سربه‌سر بردم؛ "
                      f"این معامله دیگر نمی‌تواند ضرر شود.", "open")
        elif adv["action"] == "time_exit":
            if (adv.get("pnl_r") or 0) < 0.2:                # راکد و بی‌ثمر → سرمایه را آزاد کن
                lp = prices.get(p["symbol"]) or _live(p["symbol"])
                if lp:
                    paper.close_with(p["id"], lp, "ربات: خروج زمانی — سرمایه آزاد شد")
                    _cooldown[p["symbol"]] = time.time() + COOLDOWN_SEC
                    think(f"⏳ {p['symbol'].replace('USDT','')}: {adv['reasons'][0]} — بستم و سرمایه را آزاد کردم.", "close")
            else:
                think(f"🧭 {p['symbol'].replace('USDT','')}: {adv['title']} — {adv['reasons'][0]}", "warn")

    # ۴) گاردهای حرفه‌ای پیش از شکار
    global _storm_until, _risk_off_day
    db = paper.list_positions()
    bot_open = [p for p in db["open"] if p.get("opened_by") == "bot"]
    held = {p["symbol"] for p in db["open"]}
    equity = paper.wallet_summary()["equity"]
    now = time.time()

    # ⛈ ترمزِ طوفان: حرکتِ شدیدِ بیت‌کوین = شرایطِ آشوب — ۱۰ دقیقه ورودِ جدید ممنوع (خروج‌ها فعال می‌مانند)
    for ov in ovs.values():
        bz = (ov.get("btc") or {}).get("z")
        if bz is not None and abs(bz) >= 2.8:
            if now >= _storm_until:
                think(f"⛈ طوفانِ بازار: بیت‌کوین z={bz:+.1f} — تا ۱۰ دقیقه پوزیشنِ تازه باز نمی‌کنم؛ "
                      f"فقط پوزیشن‌های باز را مدیریت می‌کنم. حرفه‌ای‌ها در آشوب معامله نمی‌سازند.", "warn")
            _storm_until = now + 600
            break

    # 🛑 حدِ ضررِ روزانه: اگر امروز بیش از حدِ مجاز باختم، تا فردا فقط نظاره‌گرم (دیسیپلین > هیجان)
    today = time.strftime("%Y-%m-%d", time.gmtime())
    day_pnl = _daily_pnl_bot(db)
    day_limit = equity * 1.5 / 100
    if day_pnl <= -day_limit and _risk_off_day != today:
        _risk_off_day = today
        think(f"🛑 حدِ ضررِ روزانه فعال شد: امروز {day_pnl:+.1f}$ (حدِ مجاز −{day_limit:.0f}$). "
              f"تا پایانِ روز ورودِ جدید ممنوع — بدترین کار بعد از باخت، تلاش برای جبرانِ فوری است.", "close")
    risk_off = _risk_off_day == today

    # 🎯 ضدِ تیلت: بعد از ۳ باختِ پیاپی، ریسکِ معامله‌های بعدی نصف می‌شود تا یک برد بیاید
    streak = _loss_streak(db)
    risk_mult = 0.5 if streak >= 3 else 1.0
    if streak >= 3:
        think(f"🎯 {streak} باختِ پیاپی — ریسکِ ورودی‌های بعدی را نصف می‌کنم تا دوباره برد ببینم.", "warn")

    if risk_off or now < _storm_until:
        _pending.clear()                                 # سفارش‌های صبورِ معلق هم در شرایطِ خطر لغو می‌شوند
        with _lock:
            _status["current"] = "🛑 توقفِ ورود (حدِ روزانه)" if risk_off else "⛈ ترمزِ طوفان — فقط مدیریتِ پوزیشن‌ها"
        return

    # ۵) شکارِ فرصت
    # بادِ کلانِ بازار (پهنای 1d): برای پوزیشنی که ساعت‌ها باز می‌ماند، رژیمِ کلان مهم‌تر از پهنای همان tf است
    macro_br = next((ov.get("breadth_macro") for ov in ovs.values()
                     if ov.get("breadth_macro") is not None), 0.0) or 0.0

    def _against_wind(c_):
        sd_ = c_.get("side")
        return bool(c_.get("market_headwind")) or \
            (sd_ == "long" and macro_br < -0.3) or (sd_ == "short" and macro_br > 0.3)

    cands = []
    for tf, ov in ovs.items():
        for c in ov.get("coins", []):
            if c.get("error") or c.get("symbol") in held or c.get("symbol") in _pending:
                continue
            if _cooldown.get(c.get("symbol"), 0) > now:
                continue                                     # تازه با چرخشِ سیگنال از این ارز خارج شده‌ام
            if _against_wind(c) and (c.get("signal_score") or 0) < P["min_score"] + 10:
                continue                                     # خلافِ باد (tf یا کلان) فقط با سیگنالِ به‌مراتب قوی‌تر
            if (c.get("cost") or 0.15) > 0.25:
                continue                                     # کفِ نقدشوندگی: جفت‌های کم‌عمق پرهزینه/پرلغزش‌اند (درسِ SPELL/MUBARAK)
            if not c.get("policy_trusted") or not c.get("policy_pass") or c.get("regime_veto"):
                continue
            if not c.get("gate_allowed"):
                continue                                     # قفلِ ایمنی: ترکیبِ پیش‌ثبت‌نشده
            if c.get("market_rank_enforced") and c.get("market_rank_pct", 0) < 60:
                continue
            need_score = P["min_score"]
            if c.get("tradeable") and (c.get("signal_score") or 0) >= need_score:
                cands.append((c.get("entry_quality") or c.get("signal_score") or 0, tf, c, "signal"))
    cands.sort(key=lambda x: -x[0])
    side_ct = {"long": sum(1 for p in bot_open if p["side"] == "long"),
               "short": sum(1 for p in bot_open if p["side"] == "short")}
    for o in _pending.values():                              # سفارش‌های صبورِ در انتظار هم تعهدِ جهت‌دارند
        side_ct[o["side"]] = side_ct.get(o["side"], 0) + 1

    need = max(P["min_open"] - len(bot_open), 0)
    capacity = max(P["max_open"] - len(bot_open) - len(_pending), 0)
    budget = min(P["opens_per_cycle"], capacity)
    think(f"🧠 {len(cands)} کاندیدِ تأییدشده بالای آستانه | بازِ ربات: {len(bot_open)}"
          f"{' | برای رسیدن به کف ' + str(P['min_open']) + ' هنوز ' + str(need) + ' لازم است' if need else ''}")

    opened = 0
    # سقفِ تمرکزِ هم‌جهت: خلافِ باد حداکثر ۳؛ هم‌جهت با باد هم حداکثر ~۷۰٪ ظرفیت
    # (درسِ پایشِ ۸ساعته: ۱۲/۱۲ شورت = یک شرطِ واحد که ±۴۰$ در ساعت نفس می‌کشید)
    max_same = max(2, round(P["max_open"] * 0.6))
    _warned_side = set()
    for score, tf, c, kind in cands:
        if opened >= budget:
            break
        if c["symbol"] in held:
            continue                                 # هر ارز فقط یک پوزیشن (حتی اگر چند تایم‌فریم سیگنال بدهند)
        sd = c.get("side")
        cap_side = 3 if _against_wind(c) else max_same       # بادِ کلان (درسِ ALLO: چهارمین لانگ در بازارِ ریزشی)
        if sd and side_ct.get(sd, 0) >= cap_side:
            if sd not in _warned_side:
                _warned_side.add(sd)
                think(f"⚖️ از قبل {side_ct[sd]} پوزیشنِ {'خرید' if sd == 'long' else 'فروش'} دارم (سقفِ تمرکز: {cap_side}) — "
                      f"پوزیشن‌های هم‌جهت با هم می‌بازند؛ بیشتر نمی‌سازم.", "warn")
            continue
        if _open_from_candidate(c, tf, kind, P, equity, paper.list_positions(), risk_mult):
            opened += 1
            need = max(need - 1, 0)
            held.add(c["symbol"])
            if sd:
                side_ct[sd] = side_ct.get(sd, 0) + 1

    # ۵) پرکردنِ کف با پیش‌بینیِ جهت (فقط سطوحِ بالا — شفاف و برچسب‌خورده)
    still_need = max(P["min_open"] - (len(bot_open) + opened), 0)
    if still_need > 0 and P["allow_lean"] and opened < budget:
        tf0 = P["scan_tfs"][0]
        leans = [c for c in ovs.get(tf0, {}).get("coins", [])
                 if not c.get("error") and c.get("symbol") not in held and c.get("p_up") is not None
                 and _cooldown.get(c.get("symbol"), 0) <= now
                 and (c["p_up"] >= 58 or c["p_up"] <= 42) and not c.get("tradeable")]
        leans.sort(key=lambda c: -abs(c["p_up"] - 50))
        if leans:
            think(f"📉 سیگنالِ تأییدشدهٔ کافی نبود؛ برای کفِ سطح {P['level']} از قوی‌ترین پیش‌بینی‌های جهت استفاده می‌کنم (با ریسکِ کم)")
        for c in leans:
            if opened >= budget or still_need <= 0:
                break
            if _open_from_candidate(c, tf0, "lean", P, equity, paper.list_positions(), risk_mult):
                opened += 1
                still_need -= 1
                held.add(c["symbol"])

    if opened == 0:
        if not cands and need == 0:
            think("😴 هیچ فرصتی با کیفیتِ کافی نیست — دست نگه می‌دارم. دقت مهم‌تر از تعداد است.")
    else:
        think(f"✔️ این چرخه {opened} پوزیشن باز کردم. چرخهٔ بعد تا {CYCLE_SEC} ثانیهٔ دیگر.")


def _loop():
    while True:
        try:
            _cycle()
        except Exception as e:  # noqa: BLE001
            think(f"⚠️ خطا در چرخه: {e}", "warn")
        with _lock:
            _status["last_cycle"] = time.time()
            _status["next_cycle"] = time.time() + CYCLE_SEC
        time.sleep(CYCLE_SEC)


def _monitor_loop():
    while True:
        time.sleep(MONITOR_SEC)
        try:
            _monitor()
        except Exception:  # noqa: BLE001 — پایشگر هرگز نباید بمیرد
            pass


def start():
    global _started
    if _started:
        return
    _started = True
    threading.Thread(target=_loop, daemon=True).start()
    threading.Thread(target=_monitor_loop, daemon=True).start()
