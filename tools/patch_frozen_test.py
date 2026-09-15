# -*- coding: utf-8 -*-
"""پچِ فاز ۲d — پنجرهٔ آزمونِ منجمد در calib.build + ثباتِ اعتماد + ردِ سیاستِ مخرب."""
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


sub1("import stats\nimport universe\n", "import research\nimport stats\nimport universe\n", "import research")

# ── بازسازیِ هفتگی (نه روزانه) ──
sub1("REBUILD_SEC = 86400\n",
     "# هفتگی، نه روزانه: بازسازیِ روزانه همان پنجرهٔ آزمون را ۳۶۵ بار در سال دوباره قضاوت می‌کرد\n"
     "REBUILD_SEC = 7 * 86400\n"
     "TRUST_STREAK = 2         # اعتماد فقط پس از دو ساختِ متوالیِ موفق روشن می‌شود\n",
     "weekly rebuild")

# ── جداکردنِ رویدادهای پنجرهٔ منجمد پیش از هر آموزش ──
sub1('''        def _train_tf(tf):
            cells, all_events, dx, dy, dr = pending[tf]''',
     '''        prereg = research.load_prereg()

        def _train_tf(tf):
            cells, all_events, dx, dy, dr = pending[tf]
            # ── پنجرهٔ آزمونِ منجمد: از **همهٔ** آموزش‌ها کنار گذاشته می‌شود ──
            # فقط داورِ یک‌بارمصرف (research.judge) این رویدادها را می‌بیند.
            if prereg:
                held = [e for e in all_events if research.in_final_window(tf, e["ts"], prereg)]
                all_events = [e for e in all_events
                              if not research.in_final_window(tf, e["ts"], prereg)]
                keep = [k for k, yy in enumerate(dy)
                        if not research.in_final_window(tf, yy[0], prereg)]
                dx, dy, dr = [dx[k] for k in keep], [dy[k] for k in keep], [dr[k] for k in keep]
                final_events[tf] = held''',
     "hold out final window")

sub1('''            return tf, {"cells": cells, "events": len(all_events),
                        "feature_health": feat_health,''',
     '''            return tf, {"cells": cells, "events": len(all_events),
                        "held_out_final_events": len(final_events.get(tf) or []),
                        "feature_health": feat_health,''',
     "report held out")

sub1('''        universe_report, excluded_events = {}, {}''',
     '''        universe_report, excluded_events = {}, {}
        final_events = {}                                 # tf -> رویدادهای پنجرهٔ منجمد''',
     "final events init")

# ── پس از ساخت: هیسترزیسِ اعتماد + داوریِ یک‌باره (اگر پیش‌ثبت هست و هنوز قضاوت نشده) ──
sub1('''        table["universe"] = {"per_tf": universe_report,''',
     '''        table["trust_history"] = _update_trust_history(load() or {}, table)
        table["spans"] = {tf: [int(min(e["ts"] for e in pending[tf][1])),
                               int(max(e["ts"] for e in pending[tf][1]))]
                          for tf in pending if pending[tf][1]}
        if prereg and final_events:
            try:
                judged = research.last_judgement() or {}
                if not judged.get("judged"):
                    table["judgement"] = research.judge(final_events)
            except research.AlreadyJudged:
                pass
            except research.PreregistrationError as e:
                table["judgement_error"] = str(e)
        table["universe"] = {"per_tf": universe_report,''',
     "judge after build")

# ── هیسترزیسِ اعتماد ──
sub1('''def is_stale():''',
     '''TRUST_KEYS = {
    "model": lambda m: _model_trusted(m, MIN_SETUP_LIFT),
    "dir_model": lambda m: _model_trusted(m, MIN_DIR_LIFT),
    "edge_model": lambda m: _edge_model_trusted(m),
    "policy_model": lambda m: _policy_model_trusted(m),
    "action_model": lambda m: _action_policy_trusted(m),
}


def _update_trust_history(old_table, new_table):
    """تاریخچهٔ اعتمادِ هر مدل در ساخت‌های پیاپی (حداکثر ۱۰ تا).

    پرچمِ اعتماد قبلاً از **یک** پنجرهٔ ۲۰٪ آخر در همان ساخت محاسبه می‌شد؛ پس هر
    بازسازی یک پرتابِ سکهٔ تازه بود (lift روی 4h بینِ دو ساخت از +۱۴٫۷ به −۰٫۶ رفت).
    """
    hist = dict((old_table or {}).get("trust_history") or {})
    if (old_table or {}).get("version") != new_table.get("version"):
        hist = {}                                  # نسخهٔ تازه = تاریخچهٔ تازه
    for tf, blob in (new_table.get("tfs") or {}).items():
        row = dict(hist.get(tf) or {})
        for key, fn in TRUST_KEYS.items():
            try:
                ok = bool(blob.get(key) and fn(blob.get(key)))
            except Exception:  # noqa: BLE001
                ok = False
            row[key] = (list(row.get(key) or []) + [ok])[-10:]
        hist[tf] = row
    return hist


def trusted_with_hysteresis(tf, key, table=None):
    """اعتماد فقط اگر **دو ساختِ پیاپیِ آخر** هر دو موفق بوده باشند؛ یک شکست کافی است تا خاموش شود."""
    t = table if table is not None else load()
    runs = (((t or {}).get("trust_history") or {}).get(tf) or {}).get(key) or []
    return len(runs) >= TRUST_STREAK and all(runs[-TRUST_STREAK:])


def is_stale():''',
     "trust hysteresis")

# ── predict / predict_dir / predict_action: هیسترزیس لازم است ──
sub1('''    pm = d.get("policy_model")
    policy_diag = None
    if pm and _model_nfeat(pm) == len(feats):
        ps = _score_policy(pm, feats, risk_pct, actual_cost, regime)
        if _policy_model_trusted(pm):''',
     '''    pm = d.get("policy_model")
    policy_diag = None
    if pm and _model_nfeat(pm) == len(feats):
        ps = _score_policy(pm, feats, risk_pct, actual_cost, regime)
        if _policy_model_trusted(pm) and trusted_with_hysteresis(tf, "policy_model", t):''',
     "policy hysteresis")
sub1('''    dm = t["tfs"][tf].get("dir_model")
    if not _model_trusted(dm, MIN_DIR_LIFT) or _model_nfeat(dm) != len(feats):
        return None''',
     '''    dm = t["tfs"][tf].get("dir_model")
    if not _model_trusted(dm, MIN_DIR_LIFT) or _model_nfeat(dm) != len(feats):
        return None
    if not trusted_with_hysteresis(tf, "dir_model", t):
        return None''',
     "dir hysteresis")
sub1('''    m = t["tfs"][tf].get("action_model")
    if not _action_policy_trusted(m) or _model_nfeat(m) != len(long_feats) or len(short_feats) != len(long_feats):
        return None''',
     '''    m = t["tfs"][tf].get("action_model")
    if not _action_policy_trusted(m) or _model_nfeat(m) != len(long_feats) or len(short_feats) != len(long_feats):
        return None
    if not trusted_with_hysteresis(tf, "action_model", t):
        return None''',
     "action hysteresis")

# ── ردِ سیاستِ مخرب (شیبِ صفر / edge_sd ≈ ۰) ──
sub1('''def _policy_model_trusted(m):
    if not m:
        return False''',
     '''MIN_LIVE_SD = 1e-3       # زیرِ این، نرمال‌سازیِ زنده امتیازِ ±۱۰⁵ می‌سازد


def _policy_is_degenerate(m):
    """سیاستی که جزءِ edgeاش ثابت است ولی وزن دارد، روی ردهٔ هزینهٔ نماد تصمیم می‌گیرد، نه ویژگی‌ها."""
    norm = (m or {}).get("live_norm") or {}
    w = float((m or {}).get("edge_weight") or 0.0)
    slope = float(((m or {}).get("edge_calibration") or [1.0, 0.0])[0])
    if w > 0 and (slope <= 0.0 or float(norm.get("edge_sd") or 0.0) < MIN_LIVE_SD):
        return True
    return w < 1.0 and float(norm.get("p_sd") or 0.0) < MIN_LIVE_SD


def _policy_model_trusted(m):
    if not m:
        return False
    if _policy_is_degenerate(m):
        return False''',
     "degenerate policy reject")

sub1('''    candidates = []
    for edge_weight in (0.0, 0.25, 0.50, 0.75, 1.0):
        score = edge_weight * ez_cal + (1.0 - edge_weight) * pz_cal''',
     '''    candidates = []
    # اگر شیبِ کالیبراسیونِ edge صفر شده، جزءِ edge اطلاعاتی ندارد؛ وزن‌دادن به آن
    # یعنی تصمیم بر اساسِ اختلافِ هزینه تقسیم بر ۱e-۶. فقط وزنِ صفر مجاز است.
    edge_weights = (0.0,) if (edge_slope <= 0.0 or e_sd < MIN_LIVE_SD) else (0.0, 0.25, 0.50, 0.75, 1.0)
    for edge_weight in edge_weights:
        score = edge_weight * ez_cal + (1.0 - edge_weight) * pz_cal''',
     "degenerate grid guard")

sub1('''CALIB_VERSION = 21       # با هر تغییرِ ویژگی‌ها/براکت/فرمتِ مدل یک واحد اضافه شود تا مدل قدیمی خودکار بازساخته شود''',
     '''CALIB_VERSION = 22       # با هر تغییرِ ویژگی‌ها/براکت/فرمتِ مدل یک واحد اضافه شود تا مدل قدیمی خودکار بازساخته شود
# ۲۲: پنجرهٔ آزمونِ منجمد از آموزش کنار گذاشته می‌شود؛ اعتماد با هیسترزیسِ دو ساخت''',
     "version bump")

io.open(P, "w", encoding="utf-8", newline="\n").write(s)
print("phase-2d patch applied")
