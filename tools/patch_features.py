# -*- coding: utf-8 -*-
"""پچِ فاز ۱e — حذفِ ویژگیِ مردهٔ dxy و یکسان‌سازیِ مسیرِ ساختِ ویژگی.

این اسکریپت یک‌بارمصرف است و برای بازتولیدِ تغییر در تاریخچه نگه داشته می‌شود.
"""
import io
import os

BOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "bot")


def rd(p):
    return io.open(os.path.join(BOT, p), encoding="utf-8").read()


def wr(p, s):
    io.open(os.path.join(BOT, p), "w", encoding="utf-8", newline="\n").write(s)


def sub1(s, old, new, label):
    n = s.count(old)
    assert n == 1, f"{label}: expected 1 occurrence, found {n}"
    return s.replace(old, new)


def subn(s, old, new, label, times):
    n = s.count(old)
    assert n == times, f"{label}: expected {times} occurrences, found {n}"
    return s.replace(old, new)


# ── engine.py: زمینه از features ساخته می‌شود ──
e = rd("engine.py")
e = subn(e, 'gold_m=ex.get("gold", 0.0), dxy_m=ex.get("dxy", 0.0),',
         'gold_m=ex.get("gold", 0.0),', "engine analyze feats", 2)
assert "dxy_m" not in e, "dxy_m still in engine.py"
wr("engine.py", e)

# ── calib.py: نقشهٔ dxy و آرگومانش حذف می‌شود ──
c = rd("calib.py")
c = sub1(c, '''def _dxy_at(dxy_map, ts):
    """مقدار DXY برای روزِ حاوی ts (تا ۶ روز عقب‌گرد برای تعطیلات)."""
    if not dxy_map:
        return 0.0
    day = (ts // 86400000) * 86400000
    for k in range(7):
        v = dxy_map.get(day - k * 86400000)
        if v is not None:
            return v
    return 0.0


''', "", "calib _dxy_at")
c = sub1(c, "def extract_events(sym, kl, tf, fz_list, rs_map, htf_zmap, btc_zmap, gold_map=None, dxy_map=None,\n"
            "                   breadth_map=None, dom_map=None, ethbtc_map=None):",
         "def extract_events(sym, kl, tf, fz_list, rs_map, htf_zmap, btc_zmap, gold_map=None,\n"
         "                   breadth_map=None, dom_map=None, ethbtc_map=None):", "calib extract_events sig")
c = subn(c, 'gold_m=(gold_map or {}).get(ts, 0.0), dxy_m=_dxy_at(dxy_map, ts),',
         'gold_m=(gold_map or {}).get(ts, 0.0),', "calib gold/dxy kwargs", 2)
c = sub1(c, '''        dxy_rows = market.get_dxy_daily()
        dxy_norm = mom_norm_map([r[0] for r in dxy_rows], [r[1] for r in dxy_rows]) if dxy_rows else {}
        dxy_map = {(t // 86400000) * 86400000: v for t, v in dxy_norm.items()}
''', "", "calib build dxy map")
c = sub1(c, "                                                           gold_map, dxy_map,\n",
         "                                                           gold_map,\n", "calib extract call")
assert "dxy" not in c, "dxy still in calib.py"
wr("calib.py", c)

# ── main.py: منبعِ فاندینگِ ویژگی = همان فرمولِ آموزش؛ dxy از ماکرو حذف ──
m = rd("main.py")
m = sub1(m, '_macro_cache: dict = {}   # tf -> (ts, {"gold": x, "dxy": y})',
         '_macro_cache: dict = {}   # tf -> (ts, {"gold": x})', "main macro cache comment")
m = sub1(m, '''    """مومنتوم زنده طلا (PAXG) و شاخص دلار — از همان منبع و نرمال‌سازی آموزش (بدون skew)."""
    now = time.time()
    hit = _macro_cache.get(tf)
    if hit and now - hit[0] < 900:
        return hit[1]
    out = {"gold": 0.0, "dxy": 0.0}
    try:
        gk = market.get_history("PAXGUSDT", tf, calib.BARS.get(tf, 3000))
        gmap = calib.mom_norm_map(gk["t"], gk["c"])
        if gmap:
            out["gold"] = list(gmap.values())[-1]
    except Exception:  # noqa: BLE001
        pass
    try:
        rows = market.get_dxy_daily()
        if rows:
            dmap = calib.mom_norm_map([r[0] for r in rows], [r[1] for r in rows])
            if dmap:
                out["dxy"] = list(dmap.values())[-1]
    except Exception:  # noqa: BLE001
        pass
    _macro_cache[tf] = (now, out)
    return out''', '''    """مومنتوم زندهٔ طلا (PAXG) — از همان منبع و نرمال‌سازیِ آموزش (بدون skew).

    شاخصِ دلار حذف شد: فایلِ dxy_daily.json هرگز پر نشد و ویژگی در تمامِ آموزش
    ثابتِ ۰ بود؛ زنده‌کردنش بعداً یعنی دادنِ ورودیِ ندیده به مدل.
    """
    now = time.time()
    hit = _macro_cache.get(tf)
    if hit and now - hit[0] < 900:
        return hit[1]
    out = {"gold": 0.0}
    try:
        gk = market.get_history("PAXGUSDT", tf, calib.BARS.get(tf, 3000))
        gmap = calib.mom_norm_map(gk["t"], gk["c"])
        if gmap:
            out["gold"] = list(gmap.values())[-1]
    except Exception:  # noqa: BLE001
        pass
    _macro_cache[tf] = (now, out)
    return out''', "main _macro")
m = sub1(m, '''        extras = {"funding_z": float(finfo.get("z") or 0.0),''',
         '''        extras = {# ویژگیِ مدل با فرمولِ آموزش ساخته می‌شود، نه با z شلوغیِ متا-گیت
                  "funding_z": features.live_funding_z(symbol),
                  "funding_crowd_z": float(finfo.get("z") or 0.0),''', "main extras funding")
m = sub1(m, "import candidates\nimport engine\n", "import candidates\nimport engine\nimport features\n",
         "main import features")
wr("main.py", m)
print("phase-1e patch applied")
