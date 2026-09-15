# -*- coding: utf-8 -*-
"""پچِ فاز ۱f — سیم‌کشیِ ژورنال به نقاطِ تغییرِ وضعیتِ ریسک در autotrader."""
import io
import os

BOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "bot")
P = os.path.join(BOT, "autotrader.py")
s = io.open(P, encoding="utf-8").read()


def sub(old, new, times, label):
    n = s.count(old)
    assert n == times, f"{label}: expected {times}, found {n}"
    return s.replace(old, new)


# ── کول‌داون‌ها ──
s = sub('_cooldown[p["symbol"]] = max(_cooldown.get(p["symbol"], 0), time.time() + cd)',
        '_set_cooldown(p["symbol"], time.time() + cd)', 1, "cooldown after close")
s = sub('        _cooldown[p["symbol"]] = now + COOLDOWN_SEC',
        '        _set_cooldown(p["symbol"], now + COOLDOWN_SEC)', 1, "cooldown toxic 1d")
s = sub('            _cooldown[p["symbol"]] = time.time() + 86400',
        '            _set_cooldown(p["symbol"], time.time() + 86400)', 1, "cooldown dead feed")
s = sub('_cooldown[p["symbol"]] = time.time() + COOLDOWN_SEC',
        '_set_cooldown(p["symbol"], time.time() + COOLDOWN_SEC)', 2, "cooldown advisor exits")
s = sub('        _cooldown[c["symbol"]] = ', '        _set_cooldown_kv(c["symbol"], ', 0, "noop")

# ── سفارشِ صبور: ثبتِ گذاشتنِ سفارش ──
s = sub('''                         "meta_size_mult": c.get("meta_size_mult"),
                         "authority": c.get("authority")}
''', '''                         "meta_size_mult": c.get("meta_size_mult"),
                         "authority": c.get("authority")}
        journal.append(journal.ORDER_PLACED, symbol=sym, order=_pending[sym])
''', 1, "order placed")

# ── پاک‌سازیِ کلیِ سفارش‌ها: با ثبتِ تک‌تک ──
s = sub('''    if _risk_off_day == time.strftime("%Y-%m-%d", time.gmtime()) or now < _storm_until:
        for sym, o in list(_pending.items()):
            fill_quality.log_event("cancelled", symbol=sym, tf=o.get("tf"), side=o.get("side"),
                                   why="storm_or_risk_off", authority=o.get("authority"))
        _pending.clear()
        return''', '''    if _risk_off_day == time.strftime("%Y-%m-%d", time.gmtime()) or now < _storm_until:
        _clear_pending("storm_or_risk_off")
        return''', 1, "clear pending storm")
s = sub('''    cfg = load_cfg()
    if not cfg["enabled"]:
        _pending.clear()''', '''    cfg = load_cfg()
    if not cfg["enabled"]:
        _clear_pending("bot_disabled")''', 1, "clear pending disabled")
s = sub('''        _pending.clear()                                 # سفارش‌های صبورِ معلق هم در شرایطِ خطر لغو می‌شوند''',
        '''        _clear_pending("risk_off_or_storm")              # سفارش‌های صبورِ معلق در شرایطِ خطر لغو می‌شوند''',
        1, "clear pending risk off")

# ── ترمزها ──
s = sub('            _storm_until = now + 600', '            _set_storm(now + 600)', 1, "storm")
s = sub('''        _risk_off_day = today''', '''        _set_risk_off(today)''', 1, "risk off")

# ── بازیابی هنگام استارت ──
s = sub('''    _started = True
    threading.Thread(target=_loop, daemon=True).start()''',
        '''    _started = True
    restore_state()                      # ترمزهای ریسک نباید با ری‌استارت پاک شوند
    threading.Thread(target=_loop, daemon=True).start()''', 1, "restore on start")

io.open(P, "w", encoding="utf-8", newline="\n").write(s)
print("phase-1f patch applied")
