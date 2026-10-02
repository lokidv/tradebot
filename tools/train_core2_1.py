# -*- coding: utf-8 -*-
"""walk-forwardِ پیش‌ثبت‌شدهٔ هستهٔ v2.1 (bot/core2_1.py؛ پیش‌ثبت: bot/data/research/prereg_core2_1.json).

  python tools/train_core2_1.py --tf all                 # اعتبارسنجی ۲۰۲۵-۰۷..۱۲ برای هر پنج تایم‌فریم + گزارش
  python tools/train_core2_1.py --tf 1d 4h --no-report   # فقط این‌ها
  python tools/train_core2_1.py --assemble               # فقط گزارش از نتیجه‌های ذخیره‌شده
  python tools/train_core2_1.py --pin                    # سنجاقِ آخرین گزارشِ کامل (پس از آن commit شود)
  python tools/train_core2_1.py --holdout-record 1h      # رکوردِ شروعِ holdout (پس از آن commit شود)
  python tools/train_core2_1.py --window holdout --holdout --tf 1h   # holdoutِ یک‌بارمصرف؛ فقط اگر نگهبان اجازه دهد
"""
import argparse
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "bot"))

import core2  # noqa: E402
import core2_1 as c21  # noqa: E402

for _s in (sys.stdout, sys.stderr):
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


def load_results(tfs=c21.TFS):
    out = {}
    for tf in tfs:
        p = c21.result_path(tf, "validation")
        if os.path.exists(p):
            with open(p, "r", encoding="utf-8") as f:
                out[tf] = json.load(f)
    return out


def write_report():
    rep = c21.assemble(load_results())
    path = os.path.join(c21.RESEARCH_DIR, time.strftime(c21.REPORT_PREFIX + "%Y%m%d_%H%M%S.json", time.gmtime()))
    _write(path, rep)
    log(f"گزارش: {path} (کامل: {rep['complete']}، پذیرفته: {rep['adopted'] or 'هیچ'})")
    for line in c21.summary_lines(rep):
        print(line, flush=True)
    return rep, path


def main(argv=None):
    ap = argparse.ArgumentParser(description="walk-forwardِ پیش‌ثبت‌شدهٔ هستهٔ v2.1")
    ap.add_argument("--tf", nargs="+", default=["all"], help="5m 15m 1h 4h 1d یا all")
    ap.add_argument("--window", choices=("validation", "holdout"), default="validation")
    ap.add_argument("--holdout", action="store_true", help="اجازهٔ صریحِ holdoutِ یک‌بارمصرف")
    ap.add_argument("--threads", type=int, default=os.cpu_count() or 4)
    ap.add_argument("--no-report", action="store_true")
    ap.add_argument("--assemble", action="store_true")
    ap.add_argument("--pin", action="store_true")
    ap.add_argument("--holdout-record", metavar="TF")
    a = ap.parse_args(argv)
    tfs = list(c21.TFS) if "all" in a.tf else a.tf
    bad = [tf for tf in tfs if tf not in c21.TFS]
    if bad:
        ap.error(f"تایم‌فریمِ ناشناخته: {bad}")
    c21.check_prereg()
    if a.assemble:
        write_report()
        return 0
    if a.pin:
        rep, path = c21.latest_validation_report()
        if not rep or not rep.get("complete"):
            print("رد شد: گزارشِ کاملِ اعتبارسنجی نیست", file=sys.stderr)
            return 2
        print(json.dumps(c21.make_pin(path), indent=1))
        return 0
    if a.holdout_record:
        rep, path = c21.latest_validation_report()
        ok = bool(rep and rep.get("complete") and ((rep.get("timeframes") or {}).get(a.holdout_record) or {}).get("adopted"))
        if not ok:
            print(f"رد شد: {a.holdout_record} پذیرفته نشده", file=sys.stderr)
            return 2
        print(json.dumps(c21.write_start_record(a.holdout_record, path), ensure_ascii=False))
        return 0
    if a.window == "holdout" or a.holdout:
        if not (a.holdout and a.window == "holdout"):
            print("رد شد: holdout فقط با --window holdout --holdout", file=sys.stderr)
            return 2
        rep, path = c21.latest_validation_report()
        for tf in tfs:
            guard = c21.holdout_guard(tf, rep, path)
            if not guard[0]:
                print(f"رد شد: {tf}: {guard[1]}", file=sys.stderr)
                return 2
        for tf in tfs:
            res, oos = c21.run_window(tf, "holdout", a.threads, log, guard=c21.holdout_guard(tf, rep, path))
            res["holdout_verdict"] = c21.holdout_verdict(tf, res)
            c21.save_oos(tf, oos, "holdout")
            _write(c21.result_path(tf, "holdout"), res)
            with open(c21.HOLDOUT_LOG, "a", encoding="utf-8") as f:
                f.write(json.dumps({"event": "holdout_done", "tf": tf, "passed": res["holdout_verdict"]["passed"],
                                    "mean": res["v21"]["mean"], "n": res["v21"]["n"]}, ensure_ascii=False) + "\n")
        return 0
    for tf in tfs:
        log(f"── {tf}: walk-forward اعتبارسنجی v2.1")
        res, oos = c21.run_window(tf, "validation", a.threads, log)
        p = c21.save_oos(tf, oos, "validation")
        _write(c21.result_path(tf, "validation"), res)
        v, r, b = res["v21"], res["rule"], res["bootstrap"]
        log(f"── {tf}: v2.1 n={v['n']} mean={v['mean']} | rule n={r['n']} mean={r['mean']} | "
            f"p={b['p_one_sided']:.4f} | replay_equal={res['replay_equal_to_decision_replay']} | "
            f"wall {res['timings_s']['wall']}s ⇒ {p}")
    if not a.no_report:
        write_report()
    return 0


if __name__ == "__main__":
    sys.exit(main())
