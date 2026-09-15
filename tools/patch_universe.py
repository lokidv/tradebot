# -*- coding: utf-8 -*-
"""پچِ فاز ۲c — جهانِ نقطه‌-در-زمان در آموزش و رتبه‌بندیِ زنده."""
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


patch("calib.py", [
    ("import stats\n", "import stats\nimport universe\n", "import universe"),
    ('''            # رتبه قدرت نسبی مقطعی (بازده ۲۰ کندلی) در هر مهر زمانی
            rets = {}
            for sym, kl in hists.items():
                cc = kl["c"]
                for idx in range(20, len(cc)):
                    rets.setdefault(kl["t"][idx], []).append((sym, cc[idx] / cc[idx - 20] - 1))
            rs_map = {}
            for ts, lst in rets.items():
                if len(lst) >= 8:
                    lst.sort(key=lambda x: x[1])
                    m = len(lst) - 1
                    rs_map[ts] = {sym: (k / m if m else 0.5) for k, (sym, _) in enumerate(lst)}''',
     '''            # ── جهانِ نقطه‌-در-زمان: برترین‌های هر ماه بر اساسِ حجمِ ۳۰ روزِ پیش از آن ──
            # (نه ۱۰۰ ارزِ پرحجمِ امروز — آن سوگیریِ بقا و نگاه به آینده در انتخاب بود)
            snaps = universe.snapshots_from_histories(hists, top_n=CALIB_UNIVERSE_N)
            universe_report[tf] = universe.summary(snaps)

            def _member(sym, ts_, _snaps=snaps):
                return universe.in_universe(_snaps, sym, ts_)

            # رتبه قدرت نسبی مقطعی (بازده ۲۰ کندلی) — فقط بینِ اعضای جهانِ همان لحظه
            rets = {}
            for sym, kl in hists.items():
                cc = kl["c"]
                for idx in range(20, len(cc)):
                    t_ = kl["t"][idx]
                    if _member(sym, t_):
                        rets.setdefault(t_, []).append((sym, cc[idx] / cc[idx - 20] - 1))
            rs_map = {}
            for ts, lst in rets.items():
                if len(lst) >= universe.MIN_POPULATION:
                    lst.sort(key=lambda x: x[1])
                    m = len(lst) - 1
                    rs_map[ts] = {sym: (k / m if m else 0.5) for k, (sym, _) in enumerate(lst)}''',
     "rs_map PIT"),
    ('''                for j in range(50, len(cc)):
                    ts_ = kl["t"][j]
                    above.setdefault(ts_, []).append(1.0 if cc[j] > e50[j] else 0.0)''',
     '''                for j in range(50, len(cc)):
                    ts_ = kl["t"][j]
                    if _member(sym, ts_):
                        above.setdefault(ts_, []).append(1.0 if cc[j] > e50[j] else 0.0)''',
     "breadth PIT"),
    ('''            breadth_map = {t: (sum(v) / len(v) - 0.5) * 2 for t, v in above.items() if len(v) >= 8}''',
     '''            breadth_map = {t: (sum(v) / len(v) - 0.5) * 2 for t, v in above.items()
                           if len(v) >= universe.MIN_POPULATION}''',
     "breadth min population"),
    ('''                except Exception:  # noqa: BLE001
                    evs, zmap, dx, dy, dr = [], {}, [], [], []
                zmaps[tf][sym] = zmap''',
     '''                except Exception:  # noqa: BLE001
                    log.exc(f"extract_events {sym} {tf}")
                    evs, zmap, dx, dy, dr = [], {}, [], [], []
                zmaps[tf][sym] = zmap
                # فقط رویدادهایی که ارزشان **در همان لحظه** عضوِ جهان بوده
                kept = [e for e in evs if _member(sym, e["ts"])]
                excluded_events[tf] = excluded_events.get(tf, 0) + (len(evs) - len(kept))
                evs = kept
                keep_dense = [k for k, yy in enumerate(dy) if _member(sym, yy[0])]
                dx = [dx[k] for k in keep_dense]
                dy = [dy[k] for k in keep_dense]
                dr = [dr[k] for k in keep_dense]''',
     "extract PIT filter"),
    ('''        table = {"built_at": time.time(), "cost_pct": COST_PCT, "version": CALIB_VERSION, "tfs": {}}''',
     '''        table = {"built_at": time.time(), "cost_pct": COST_PCT, "version": CALIB_VERSION, "tfs": {}}
        universe_report, excluded_events = {}, {}''',
     "universe report init"),
    ('''        os.makedirs(os.path.dirname(CALIB_PATH), exist_ok=True)
        tmp = CALIB_PATH + ".tmp"''',
     '''        table["universe"] = {"per_tf": universe_report,
                             "excluded_events_outside_universe": excluded_events,
                             "top_n": CALIB_UNIVERSE_N}
        os.makedirs(os.path.dirname(CALIB_PATH), exist_ok=True)
        tmp = CALIB_PATH + ".tmp"''',
     "universe report save"),
    ('''WF_FOLDS = 5             # تعداد فولدهای Walk-Forward''',
     '''WF_FOLDS = 5             # تعداد فولدهای Walk-Forward
CALIB_UNIVERSE_N = 100   # اندازهٔ جهانِ نقطه‌-در-زمان در هر ماه''',
     "universe size const"),
    ('''CALIB_VERSION = 20       # با هر تغییرِ ویژگی‌ها/براکت/فرمتِ مدل یک واحد اضافه شود تا مدل قدیمی خودکار بازساخته شود''',
     '''CALIB_VERSION = 21       # با هر تغییرِ ویژگی‌ها/براکت/فرمتِ مدل یک واحد اضافه شود تا مدل قدیمی خودکار بازساخته شود
# ۲۱: جهانِ نقطه‌-در-زمان — رویداد فقط اگر ارز در همان ماه جزوِ ۱۰۰ برتر بوده''',
     "version bump"),
])

patch("main.py", [
    ("import features\n", "import features\nimport universe\n", "import universe"),
    ('''        symbols, _ = market.get_top_symbols(TOP_N)
        rets = []
        for s in symbols:
            kl = market.get_klines_cached(s, tf)
            if kl and len(kl["c"]) > 21:
                rets.append((s, kl["c"][-1] / kl["c"][-21] - 1))
        rets.sort(key=lambda x: x[1])
        m = max(len(rets) - 1, 1)
        ranks = {s: i / m for i, (s, _) in enumerate(rets)}''',
     '''        # همان جمعیتِ آموزش: ۱۰۰ ارزِ برتر، با حداقلِ جمعیت (وگرنه همه خنثی).
        # قبلاً روی کشِ نیمه‌خالیِ پس از ری‌استارت، چند ارز رتبهٔ ۰ یا ۱ می‌گرفتند.
        symbols, _ = market.get_top_symbols(calib.CALIB_UNIVERSE_N)
        rets = {}
        for s in symbols:
            kl = market.get_klines_cached(s, tf)
            if kl and len(kl["c"]) > 21:
                rets[s] = kl["c"][-1] / kl["c"][-21] - 1
        ranks = universe.rank_within(rets)''',
     "live rs_rank PIT"),
    ('''    out = {"breadth": 0.0, "dom": 0.0, "ethbtc": 0.0}
    try:
        symbols, _ = market.get_top_symbols(200)''',
     '''    out = {"breadth": 0.0, "dom": 0.0, "ethbtc": 0.0}
    try:
        symbols, _ = market.get_top_symbols(calib.CALIB_UNIVERSE_N)   # همان جمعیتِ آموزش''',
     "live market_state universe"),
    ('''        if len(above) >= 20:
            out["breadth"] = (sum(above) / len(above) - 0.5) * 2''',
     '''        if len(above) >= universe.MIN_POPULATION:
            out["breadth"] = (sum(above) / len(above) - 0.5) * 2''',
     "live breadth min population"),
])
print("phase-2c patch applied")
