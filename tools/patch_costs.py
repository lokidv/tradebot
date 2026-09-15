# -*- coding: utf-8 -*-
"""پچِ فاز ۲b — هزینهٔ واقعی (ردهٔ نقدشوندگیِ لحظه‌ای + فاندینگ) داخلِ برچسب‌ها."""
import io
import os

BOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "bot")
P = os.path.join(BOT, "calib.py")
s = io.open(P, encoding="utf-8").read()


def sub1(old, new, label):
    global s
    n = s.count(old)
    assert n == 1, f"{label}: found {n}"
    s = s.replace(old, new)


sub1("import bracket\nimport log\nimport engine\n", "import bracket\nimport costs\nimport log\nimport engine\n",
     "import costs")

# ── امضای extract_events: تاریخچهٔ خامِ فاندینگ ──
sub1('''def extract_events(sym, kl, tf, fz_list, rs_map, htf_zmap, btc_zmap, gold_map=None,
                   breadth_map=None, dom_map=None, ethbtc_map=None):''',
     '''def extract_events(sym, kl, tf, fz_list, rs_map, htf_zmap, btc_zmap, gold_map=None,
                   breadth_map=None, dom_map=None, ethbtc_map=None, funding_rows=None):''',
     "extract_events signature")

# ── نمونه‌های جهت‌یاب: هزینهٔ رده‌ای + فاندینگِ برآوردی به‌عنوانِ عنصرِ چهارم ──
sub1('''        lv = bracket.levels(entry, cs["a14"][i], 1)
        dir_R.append((_barrier(i, 1), _barrier(i, -1),
                      lv["risk_pct"]))                    # (نتیجه لانگ، نتیجه شورت، ریسک٪)''',
     '''        lv = bracket.levels(entry, cs["a14"][i], 1)
        # هزینهٔ لحظه‌ای (از حجمِ ۲۴ ساعتهٔ همان کندل‌ها) + فاندینگِ برآوردیِ نیمهٔ افق
        dense_cost = (costs.point_in_time_tier(c, v, i, tf)
                      + costs.expected_funding_pct(tf, bracket.MAX_BARS / 2))
        dir_R.append((_barrier(i, 1), _barrier(i, -1),
                      lv["risk_pct"], round(dense_cost, 6)))  # (لانگ، شورت، ریسک٪، هزینه٪)''',
     "dense dir_R cost")

# ── رویدادهای ستاپ: هزینهٔ دقیق ──
sub1('''        res = bracket.signal_trade(o, h, l, c, cs["a14"][i], i, sig)
        if res is None:
            continue
        out_r = res["gross_r"]
        b, s = engine.votes_at(cs, i)''',
     '''        res = bracket.signal_trade(o, h, l, c, cs["a14"][i], i, sig)
        if res is None:
            continue
        out_r = res["gross_r"]
        entry_ts = ts_arr[i + 1]
        exit_ts = ts_arr[min(res["exit_idx"], n - 1)]
        ev_cost = costs.event_cost_pct(sig, c, v, i, tf, entry_ts, exit_ts, funding_rows)
        b, s = engine.votes_at(cs, i)''',
     "setup event cost")
sub1('''                       "outcome": res["outcome"],''',
     '''                       "outcome": res["outcome"],
                       "cost_pct": ev_cost,            # رده‌ای + فاندینگ — نه ۰٫۱۵٪ ثابت''',
     "event cost field")

# ── build: تاریخچهٔ خامِ فاندینگ هم کنارِ z گرفته می‌شود ──
sub1('''        def _fz_one(s):
            try:
                return s, market.funding_z_map(s)
            except Exception:  # noqa: BLE001
                return s, []''',
     '''        def _fz_one(s):
            try:
                return s, market.funding_z_map(s)
            except Exception:  # noqa: BLE001
                return s, []

        def _fr_one(s):
            try:
                return s, market.get_funding_history(s)
            except Exception:  # noqa: BLE001
                return s, []''',
     "build funding raw fetch fn")
sub1('''            fz = dict(pool.map(_fz_one, symbols))''',
     '''            fz = dict(pool.map(_fz_one, symbols))
            fraw = dict(pool.map(_fr_one, symbols))''',
     "build funding raw fetch")
sub1('''                                                           gold_map,
                                                           breadth_map, dom_map, ethbtc_map)''',
     '''                                                           gold_map,
                                                           breadth_map, dom_map, ethbtc_map,
                                                           funding_rows=fraw.get(sym))''',
     "extract call funding rows")


# ── مدل‌های بازده: هزینهٔ هر رویداد به‌جای ثابت ──
def per_event_cost_block(var="evs"):
    return (f"    ev_cost = np.asarray([float(e.get(\"cost_pct\", cost_pct)) for e in {var}])\n")


sub1('''    evs = sorted(events, key=lambda e: e["ts"])
    X = np.asarray([e["feats"] for e in evs], float)
    risk = np.asarray([max(float(e.get("risk_pct") or 0), 0.05) for e in evs])
    # هزینهٔ رفت‌وبرگشت از خود outcome کم می‌شود؛ wins کوچک دیگر «برد» مصنوعی نیستند.
    y = np.clip(
        np.asarray([float(e["r"]) for e in evs]) - float(cost_pct) / risk,
        -1.5, 2.2,
    )''',
     '''    evs = sorted(events, key=lambda e: e["ts"])
    X = np.asarray([e["feats"] for e in evs], float)
    risk = np.asarray([max(float(e.get("risk_pct") or 0), 0.05) for e in evs])
    # هزینهٔ **هر رویداد** (ردهٔ نقدشوندگیِ لحظه‌ای + فاندینگِ واقعیِ مدتِ نگه‌داری)
    # از outcome کم می‌شود؛ هزینهٔ ثابتِ ۰٫۱۵٪ آلتِ کم‌عمق و نگه‌داریِ ۴۰روزه را نمی‌دید.
    ev_cost = np.asarray([float(e.get("cost_pct", cost_pct)) for e in evs])
    y = np.clip(
        np.asarray([float(e["r"]) for e in evs]) - ev_cost / risk,
        -1.5, 2.2,
    )''',
     "edge model per-event cost")
sub1('''    evs = sorted(events, key=lambda e: e["ts"])
    X = np.asarray([e["feats"] for e in evs], float)
    risk = np.asarray([max(float(e.get("risk_pct") or 0), 0.05) for e in evs])
    net_r = np.clip(
        np.asarray([float(e["r"]) for e in evs]) - float(cost_pct) / risk,
        -1.5, 2.2,
    )''',
     '''    evs = sorted(events, key=lambda e: e["ts"])
    X = np.asarray([e["feats"] for e in evs], float)
    risk = np.asarray([max(float(e.get("risk_pct") or 0), 0.05) for e in evs])
    ev_cost = np.asarray([float(e.get("cost_pct", cost_pct)) for e in evs])
    net_r = np.clip(
        np.asarray([float(e["r"]) for e in evs]) - ev_cost / risk,
        -1.5, 2.2,
    )''',
     "policy model per-event cost")

# ── سیاستِ متراکم: هزینهٔ هر نمونه اگر موجود باشد ──
sub1('''    risks = np.asarray([max(float(p[2][2]), 0.05) for p in pairs], float)
    net_long = np.clip(np.asarray([p[2][0] for p in pairs], float) - float(cost_pct) / risks, -1.5, 2.2)
    net_short = np.clip(np.asarray([p[2][1] for p in pairs], float) - float(cost_pct) / risks, -1.5, 2.2)''',
     '''    risks = np.asarray([max(float(p[2][2]), 0.05) for p in pairs], float)
    sample_cost = np.asarray([float(p[2][3]) if len(p[2]) > 3 else float(cost_pct)
                              for p in pairs], float)
    net_long = np.clip(np.asarray([p[2][0] for p in pairs], float) - sample_cost / risks, -1.5, 2.2)
    net_short = np.clip(np.asarray([p[2][1] for p in pairs], float) - sample_cost / risks, -1.5, 2.2)''',
     "action policy per-sample cost")

sub1('''CALIB_VERSION = 19       # با هر تغییرِ ویژگی‌ها/براکت/فرمتِ مدل یک واحد اضافه شود تا مدل قدیمی خودکار بازساخته شود''',
     '''CALIB_VERSION = 20       # با هر تغییرِ ویژگی‌ها/براکت/فرمتِ مدل یک واحد اضافه شود تا مدل قدیمی خودکار بازساخته شود
# ۲۰: هزینهٔ هر رویداد = ردهٔ نقدشوندگیِ لحظه‌ای + فاندینگِ واقعیِ مدتِ نگه‌داری''',
     "version bump")

io.open(P, "w", encoding="utf-8", newline="\n").write(s)
print("phase-2b patch applied")
