# -*- coding: utf-8 -*-
"""پچِ فاز ۴ — سقف‌های ریسک از gates.json، توقفِ هفتگی، سلامتِ قرمز = بدونِ ورود، کلیدِ قطع."""
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


patch("autotrader.py", [
    # ── سقف‌ها: «سطحِ فعالیت» فقط می‌تواند سخت‌گیرتر از gates.json باشد، هرگز شل‌تر ──
    ('''    return {
        "level": lv,
        "min_score": round(82 - lv * 2.5),                 # ۱→۸۰، ۱۰→۵۷
        "min_open": 0,
        "max_open": 2 + (lv - 1) // 3,                    # ۲ تا حداکثر ۵
        "risk_pct_equity": round(0.25 + lv * 0.025, 2),   # ۰٫۲۸٪ تا حداکثر ۰٫۵٪
        "heat_cap": round(1.25 + lv * 0.15, 1),           # ۱٫۴٪ تا حداکثر ۲٫۸٪
        "scan_tfs": scan,
        "allow_lean": False,
        "opens_per_cycle": 1 if lv <= 6 else 2,
    }''',
     '''    caps = gates.risk_caps()
    # سطحِ فعالیت فقط **درونِ** سقف‌های gates.json حرکت می‌کند؛ هرگز شل‌ترشان نمی‌کند.
    # قبلاً سطحِ ۱۰ ریسکِ ۰٫۵٪ و گرمای ۲٫۸٪ می‌داد، مستقل از اینکه چه چیزی اثبات شده بود.
    return {
        "level": lv,
        "min_score": round(82 - lv * 2.5),                 # ۱→۸۰، ۱۰→۵۷
        "min_open": 0,
        "max_open": min(2 + (lv - 1) // 3, caps["max_open"]),
        "max_side_open": caps["max_side_open"],
        "risk_pct_equity": min(round(0.25 + lv * 0.025, 2), caps["max_risk_pct_per_trade"]),
        "heat_cap": min(round(1.25 + lv * 0.15, 1), caps["heat_cap_pct"]),
        "daily_loss_halt_pct": caps["daily_loss_halt_pct"],
        "weekly_loss_halt_pct": caps["weekly_loss_halt_pct"],
        "scan_tfs": scan,
        "allow_lean": False,
        "opens_per_cycle": 1 if lv <= 6 else 2,
    }''', "level params from gates"),

    # ── زیانِ هفتگی ──
    ('''def _loss_streak(db):''',
     '''def _weekly_pnl_bot(db, now=None):
    """زیانِ ۷ روزِ اخیرِ ربات (بسته + بازِ زیان‌ده)."""
    from datetime import datetime, timedelta, timezone as _tz
    cutoff = (datetime.now(_tz.utc) - timedelta(days=7)).isoformat()
    realized = sum(p.get("pnl_usdt", 0) for p in db["closed"]
                   if p.get("opened_by") == "bot" and str(p.get("closed_at", "")) >= cutoff)
    open_loss = sum(min(p.get("pnl_usdt", 0), 0) for p in db["open"]
                    if p.get("opened_by") == "bot")
    return realized + open_loss


def _health_blocks_entries():
    """سلامتِ قرمز = دادهٔ کهنه یا مدلِ ناسازگار ⇒ ورودِ تازه ممنوع (خروج‌ها ادامه دارند)."""
    fn = _fns.get("health")
    if fn is None:
        return None
    try:
        h = fn()
    except Exception:  # noqa: BLE001
        log.exc("health check")
        return "سلامت قابلِ‌سنجش نیست"
    if h.get("status") == "red":
        return "؛ ".join(h.get("problems") or ["سلامت قرمز"])
    return None


def _loss_streak(db):''', "weekly pnl + health guard"),

    ('''    day_pnl = _daily_pnl_bot(db)
    day_limit = equity * 1.5 / 100''',
     '''    day_pnl = _daily_pnl_bot(db)
    day_limit = equity * P["daily_loss_halt_pct"] / 100     # از gates.json (قبلاً ۱٫۵٪ ثابت)''',
     "daily limit from gates"),

    ('''    risk_off = _risk_off_day == today
''',
     '''    risk_off = _risk_off_day == today

    # 🛑 حدِ ضررِ هفتگی: فراتر از آن، ورودِ تازه تا بازبینیِ دستی بسته می‌ماند
    week_pnl = _weekly_pnl_bot(db)
    week_limit = equity * P["weekly_loss_halt_pct"] / 100
    if week_pnl <= -week_limit:
        if _risk_off_day != today:
            _set_risk_off(today)
        think(f"🛑 حدِ ضررِ هفتگی: ۷ روزِ اخیر {week_pnl:+.1f}$ (حد −{week_limit:.0f}$) — "
              "ورودِ تازه بسته است تا کسی دستی بازبینی کند.", "close")
        risk_off = True

    # 🩺 سلامتِ قرمز ⇒ ورودِ تازه ممنوع
    health_block = _health_blocks_entries()
    if health_block:
        think(f"🩺 سلامتِ سیستم قرمز است — ورودِ تازه ممنوع: {health_block}", "warn")
        risk_off = True
''', "weekly halt + health"),

    ('''def init(overview_fn, drift_cap, analysis_fn=None):
    _fns["overview"] = overview_fn
    _fns["drift_cap"] = drift_cap
    _fns["analysis"] = analysis_fn''',
     '''def init(overview_fn, drift_cap, analysis_fn=None, health_fn=None):
    _fns["overview"] = overview_fn
    _fns["drift_cap"] = drift_cap
    _fns["analysis"] = analysis_fn
    _fns["health"] = health_fn


def kill_switch(reason):
    """قطعِ اضطراری: همهٔ پوزیشن‌های ربات بسته، سفارش‌ها لغو، ربات خاموش، گیت‌ها بسته."""
    closed = 0
    db = paper.list_positions()
    for p in db["open"]:
        if p.get("opened_by") != "bot" or p.get("mode") == "testnet":
            continue
        lp = _live(p["symbol"]) or p.get("last_price") or p["entry"]
        if paper.close_with(p["id"], lp, f"🛑 قطعِ اضطراری: {reason}"):
            closed += 1
    _clear_pending("kill_switch")
    cfg = load_cfg()
    cfg["enabled"] = False
    save_cfg(cfg)
    gates.kill(reason)
    today = time.strftime("%Y-%m-%d", time.gmtime())
    _set_risk_off(today)
    think(f"🛑 قطعِ اضطراری: {reason} — {closed} پوزیشن بسته، ربات خاموش، فهرستِ مجاز خالی.", "close")
    return {"closed": closed, "enabled": False}''', "init + kill switch"),
])

patch("main.py", [
    ('''    autotrader.init(overview, DRIFT_CAP, get_analysis)     # 🤖 ربات معامله‌گر خودکار''',
     '''    autotrader.init(overview, DRIFT_CAP, get_analysis, health_fn=health)   # 🤖 ربات معامله‌گر خودکار''',
     "init with health"),
    ('''@app.get("/api/research")''',
     '''class KillReq(BaseModel):
    reason: str = "دستی"


@app.post("/api/kill")
def kill(req: KillReq):
    """🛑 کلیدِ قطعِ اضطراری — پوزیشن‌های ربات بسته، ربات خاموش، فهرستِ مجاز و لایو خاموش."""
    return {"ok": True, **autotrader.kill_switch(req.reason)}


@app.get("/api/report")
def report_now(write: bool = False):
    """گزارشِ مرحله‌ای: سایه، اجرا، یکپارچگی، عملیات — و آماده‌بودن برای لایو."""
    import report
    db = paper.list_positions()
    rep = report.build(closed_positions=db.get("closed") or [])
    if write:
        rep["path"] = report.write(rep)
    return rep


@app.get("/api/research")''', "kill + report endpoints"),
    ('''    threading.Thread(target=_daily_retrain, daemon=True).start()''',
     '''    threading.Thread(target=_daily_retrain, daemon=True).start()

    def _health_and_report():
        # نمونهٔ سلامت هر ۱۰ دقیقه (برای گیتِ عملیات: ۹۹٪ سبز در ۳۰ روز) + گزارشِ هفتگی
        import report
        last_week = None
        while True:
            try:
                h = health()
                report.record_health_sample(h.get("status"), h.get("problems"))
                week = time.strftime("%Y-%W", time.gmtime())
                if week != last_week:
                    report.write(report.build(closed_positions=paper.list_positions().get("closed") or []))
                    last_week = week
            except Exception:  # noqa: BLE001
                log.exc("health/report loop")
            time.sleep(600)
    threading.Thread(target=_health_and_report, daemon=True).start()''', "health sampler"),
])
print("phase-4 patch applied")
