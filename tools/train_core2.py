# -*- coding: utf-8 -*-
"""walk-forwardِ پیش‌ثبت‌شدهٔ هستهٔ v2 (bot/core2.py؛ پیش‌ثبت: bot/data/research/prereg_core2.json).

فقط پنجرهٔ اعتبارسنجی (۲۰۲۴-۰۷-۰۱ تا ۲۰۲۵-۰۶-۳۰): برای هر ماه یک برازشِ تازه، پیش‌بینیِ بیرون از نمونه،
معامله‌های v2 و قاعدهٔ فعلی روی همان کندل‌ها، بوت‌استرپِ بلوکِ هفتگی، Holm روی پنج تایم‌فریم و قاعدهٔ پذیرش.

خروجی‌ها:
  hist_research/core2/<tf>_val_oos.npz     sym, t, E_L, E_S, month (+ yL, yS)
  hist_research/core2/<tf>_val_result.json نتیجهٔ همان تایم‌فریم + فراداده‌ی مدلِ هر ماه
  research/core2_validation_<UTC>.json     گزارشِ کل (هر پنج تایم‌فریم؛ Holm و پذیرش)

اجرا:
  python tools/train_core2.py --tf all                      # هر پنج تایم‌فریم + گزارش
  python tools/train_core2.py --tf 1d 4h --no-report        # فقط این‌ها (نتیجهٔ هر کدام جدا ذخیره می‌شود)
  python tools/train_core2.py --assemble                    # فقط ساختنِ گزارش از نتیجه‌های ذخیره‌شده

holdout یک‌بارمصرف است: ``--window holdout`` بدونِ ``--holdout`` رد می‌شود، و با آن هم فقط برای تایم‌فریمی
که آخرین گزارشِ کاملِ اعتبارسنجی «پذیرفته» باشد و هنوز اجرا نشده باشد.
"""
import argparse
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "bot"))

import core2  # noqa: E402

for _s in (sys.stdout, sys.stderr):          # کنسولِ ویندوز (cp1252) متنِ فارسی را نمی‌پذیرد
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass


def log(msg):
    print(time.strftime("%H:%M:%S ", time.gmtime()) + msg, flush=True)


def _write(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(core2.to_json(obj), f, ensure_ascii=False, indent=1)
    os.replace(tmp, path)


def load_results(tfs=core2.TFS):
    out = {}
    for tf in tfs:
        p = core2.result_path(tf, "validation")
        if os.path.exists(p):
            with open(p, "r", encoding="utf-8") as f:
                out[tf] = json.load(f)
    return out


def write_report():
    rep = core2.assemble(load_results())
    path = os.path.join(core2.RESEARCH_DIR, time.strftime(core2.REPORT_PREFIX + "%Y%m%d_%H%M%S.json", time.gmtime()))
    _write(path, rep)
    log(f"گزارش: {path} (کامل: {rep['complete']}، پذیرفته: {rep['adopted'] or 'هیچ'})")
    for line in core2.summary_lines(rep):
        print(line, flush=True)
    return rep, path


def holdout_refusal(tfs, flag):
    """``None`` اگر مجاز؛ وگرنه متنِ رد. بدونِ ``--holdout`` همیشه رد."""
    if not flag:
        return "holdout فقط با پرچمِ صریحِ --holdout (یک‌بارمصرف، فقط برای تایم‌فریمِ پذیرفته‌شده)"
    report, path = core2.latest_validation_report()
    for tf in tfs:
        ok, why = core2.holdout_guard(tf, report)
        if not ok:
            return f"{tf}: {why}" + (f" ({os.path.basename(path)})" if path else "")
    return None


def main(argv=None):
    ap = argparse.ArgumentParser(description="walk-forwardِ پیش‌ثبت‌شدهٔ هستهٔ v2")
    ap.add_argument("--tf", nargs="+", default=["all"], help="5m 15m 1h 4h 1d یا all")
    ap.add_argument("--window", choices=("validation", "holdout"), default="validation")
    ap.add_argument("--holdout", action="store_true", help="اجازهٔ صریحِ holdoutِ یک‌بارمصرف")
    ap.add_argument("--threads", type=int, default=core2.NUM_THREADS)
    ap.add_argument("--no-report", action="store_true", help="گزارشِ کل ساخته نشود (اجرای موازیِ تایم‌فریم‌ها)")
    ap.add_argument("--assemble", action="store_true", help="فقط گزارش از نتیجه‌های ذخیره‌شده")
    a = ap.parse_args(argv)
    tfs = list(core2.TFS) if "all" in a.tf else a.tf
    bad = [tf for tf in tfs if tf not in core2.TFS]
    if bad:
        ap.error(f"تایم‌فریمِ ناشناخته: {bad}")
    core2.check_prereg()
    if a.assemble:
        write_report()
        return 0
    if a.window == "holdout" or a.holdout:
        why = holdout_refusal(tfs, a.holdout and a.window == "holdout")
        if why:
            print("رد شد: " + why, file=sys.stderr, flush=True)
            return 2
        for tf in tfs:                       # فقط وقتی نگهبان اجازه داده (هرگز در این اجرا)
            res, oos = core2.run_window(tf, "holdout", a.threads, log, allow_holdout=True)
            res["holdout_verdict"] = core2.holdout_verdict(tf, res)
            core2.save_oos(tf, oos, "holdout")
            _write(core2.result_path(tf, "holdout"), res)
        return 0
    for tf in tfs:
        log(f"── {tf}: walk-forward اعتبارسنجی")
        res, oos = core2.run_window(tf, "validation", a.threads, log)
        p = core2.save_oos(tf, oos, "validation")
        _write(core2.result_path(tf, "validation"), res)
        v, r, b = res["v2"], res["rule"], res["bootstrap"]
        log(f"── {tf}: v2 n={v['n']} mean={v['mean']} | rule n={r['n']} mean={r['mean']} | "
            f"p={b['p_one_sided']:.4f} | replay_equal={res['replay_equal_to_decision_replay']} | "
            f"wall {res['timings_s']['wall']}s ⇒ {p}")
    if not a.no_report:
        write_report()
    return 0


if __name__ == "__main__":
    sys.exit(main())
