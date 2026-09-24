# -*- coding: utf-8 -*-
"""walk-forwardِ پیش‌ثبت‌شدهٔ هستهٔ v2.1 (bot/core2_1.py؛ پیش‌ثبت: bot/data/research/prereg_core2_1.json).

ترتیبِ کار (هر گام فقط یک‌بار):
  1) python tools/train_core2_1.py --tf all
       اعتبارسنجی ۲۰۲۵-۰۷..۲۰۲۵-۱۲ برای هر پنج تایم‌فریم + گزارشِ research/core2_1_validation_<UTC>.json
  2) python tools/train_core2_1.py --pin
       سنجاقِ research/core2_1_validation_pin.json (هشِ گزارش، کد و پیش‌ثبت) ⇒ گزارش و سنجاق را همان‌جا commit کنید
  3) فقط اگر تایم‌فریمی پذیرفته شد:
     python tools/train_core2_1.py --holdout-start --tf 4h
       رکوردِ holdout_start در research/core2_holdout_log.jsonl ⇒ دفتر را commit کنید
  4) python tools/train_core2_1.py --window holdout --holdout --tf 4h
       holdoutِ یک‌بارمصرف ۲۰۲۶-۰۱..۲۰۲۶-۰۸؛ نتیجه در research/core2_1_holdout_<tf>_<UTC>.json و رکوردِ
       holdout_done در دفتر ⇒ هر دو را commit کنید

دیگر:
  python tools/train_core2_1.py --tf 1d 4h --no-report     # فقط این‌ها (نتیجهٔ هر کدام جدا ذخیره می‌شود)
  python tools/train_core2_1.py --assemble                 # فقط ساختنِ گزارش از نتیجه‌های ذخیره‌شده

خروجی‌های میانی (در گیت نیستند): hist_research/core2_1/<tf>_val_oos.npz و <tf>_val_result.json.
"""
import argparse
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "bot"))

import core2  # noqa: E402
import core2_1  # noqa: E402

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
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        json.dump(core2.to_json(obj), f, ensure_ascii=False, indent=1)
    os.replace(tmp, path)


def load_results(tfs=core2_1.TFS):
    out = {}
    for tf in tfs:
        p = core2_1.result_path(tf, "validation")
        if os.path.exists(p):
            with open(p, "r", encoding="utf-8") as f:
                out[tf] = json.load(f)
    return out


def write_report():
    rep = core2_1.assemble(load_results())
    path = os.path.join(core2_1.RESEARCH_DIR, time.strftime(core2_1.REPORT_PREFIX + "%Y%m%d_%H%M%S.json", time.gmtime()))
    _write(path, rep)
    log(f"گزارش: {path} (کامل: {rep['complete']}، پذیرفته: {rep['adopted'] or 'هیچ'})")
    for line in core2_1.summary_lines(rep):
        print(line, flush=True)
    return rep, path


def _refuse(why):
    print("رد شد: " + why, file=sys.stderr, flush=True)
    return 2


def run_holdout(tfs, threads):
    """نگهبان برای **همهٔ** تایم‌فریم‌ها پیش از هر محاسبه؛ سپس برای هر کدام: تلاش ⇒ محاسبه ⇒ نتیجه ⇒ done."""
    for tf in tfs:
        ok, why = core2_1.holdout_guard(tf)
        if not ok:
            return _refuse(f"{tf}: {why}")
    for tf in tfs:
        core2_1.append_log({"event": "holdout_attempt", "tf": tf})
        log(f"── {tf}: holdoutِ یک‌بارمصرف {core2_1.WINDOWS['holdout']}")
        res, oos = core2_1.run_window(tf, "holdout", threads, log, allow_holdout=True)
        res["holdout_verdict"] = core2_1.holdout_verdict(tf, res)
        core2_1.save_oos(tf, oos, "holdout")
        _write(core2_1.result_path(tf, "holdout"), res)
        path = os.path.join(core2_1.RESEARCH_DIR,
                            time.strftime(core2_1.HOLDOUT_PREFIX + tf + "_%Y%m%d_%H%M%S.json", time.gmtime()))
        _write(path, res)
        v, hv = res["v21"], res["holdout_verdict"]
        core2_1.append_log({"event": "holdout_done", "tf": tf, "result": os.path.basename(path),
                            "result_sha256_lf": core2_1.sha256_lf(path), "n": v["n"], "mean": v["mean"],
                            "passed": hv["passed"]})
        log(f"── {tf}: holdout n={v['n']} mean={v['mean']} ⇒ {'PASS' if hv['passed'] else 'FAIL'} ⇒ {path}")
    log("نتیجه و دفترِ holdout را commit کنید.")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description="walk-forwardِ پیش‌ثبت‌شدهٔ هستهٔ v2.1")
    ap.add_argument("--tf", nargs="+", default=["all"], help="5m 15m 1h 4h 1d یا all")
    ap.add_argument("--window", choices=("validation", "holdout"), default="validation")
    ap.add_argument("--holdout", action="store_true", help="اجازهٔ صریحِ holdoutِ یک‌بارمصرف")
    ap.add_argument("--threads", type=int, default=core2_1.NUM_THREADS)
    ap.add_argument("--no-report", action="store_true", help="گزارشِ کل ساخته نشود (اجرای موازیِ تایم‌فریم‌ها)")
    ap.add_argument("--assemble", action="store_true", help="فقط گزارش از نتیجه‌های ذخیره‌شده")
    ap.add_argument("--pin", action="store_true", help="سنجاقِ آخرین گزارشِ کاملِ اعتبارسنجی (یک‌بار)")
    ap.add_argument("--holdout-start", action="store_true", help="رکوردِ holdout_start برای --tf (سپس commit)")
    a = ap.parse_args(argv)
    tfs = list(core2_1.TFS) if "all" in a.tf else a.tf
    bad = [tf for tf in tfs if tf not in core2_1.TFS]
    if bad:
        ap.error(f"تایم‌فریمِ ناشناخته: {bad}")
    core2_1.check_prereg()
    core2_1.check_features()
    if a.assemble:
        write_report()
        return 0
    if a.pin:
        rep, path = core2_1.latest_validation_report()
        if not path:
            return _refuse("گزارشِ اعتبارسنجی پیدا نشد")
        try:
            pin = core2_1.write_pin(path)
        except (ValueError, FileExistsError) as e:
            return _refuse(str(e))
        log(f"سنجاق: {core2_1.PIN_PATH} ⇐ {pin['report']} (پذیرفته: {pin['adopted'] or 'هیچ'})")
        log("اکنون گزارش و سنجاق را commit کنید.")
        return 0
    if a.holdout_start:
        if "all" in a.tf:
            return _refuse("--holdout-start فقط با تایم‌فریمِ صریح (نه all)")
        for tf in tfs:
            try:
                rec = core2_1.start_record(tf)
            except (PermissionError, ValueError, OSError) as e:
                return _refuse(f"{tf}: {e}")
            log(f"holdout_start {tf} ⇒ {core2_1.HOLDOUT_LOG} ({rec['utc']})")
        log("اکنون دفترِ holdout را commit کنید؛ سپس --window holdout --holdout.")
        return 0
    if a.window == "holdout" or a.holdout:
        if not (a.holdout and a.window == "holdout"):
            return _refuse("holdout فقط با --window holdout --holdout (یک‌بارمصرف، فقط برای تایم‌فریمِ پذیرفته‌شده)")
        if "all" in a.tf:
            return _refuse("holdout فقط با تایم‌فریمِ صریح (نه all)")
        return run_holdout(tfs, a.threads)
    for tf in tfs:
        log(f"── {tf}: walk-forward اعتبارسنجی {core2_1.WINDOWS['validation']}")
        res, oos = core2_1.run_window(tf, "validation", a.threads, log)
        p = core2_1.save_oos(tf, oos, "validation")
        _write(core2_1.result_path(tf, "validation"), res)
        v, r, b, pr = res["v21"], res["rule"], res["bootstrap"], res["predictions"]
        log(f"── {tf}: v2.1 n={v['n']} mean={v['mean']} | rule n={r['n']} mean={r['mean']} | "
            f"p={b['p_one_sided']:.4f} | E_L>0 {pr['E_L'].get('share_pos')} E_S>0 {pr['E_S'].get('share_pos')} | "
            f"replay_equal={res['replay_equal_to_decision_replay']} | wall {res['timings_s']['wall']}s ⇒ {p}")
    if not a.no_report:
        write_report()
    return 0


if __name__ == "__main__":
    sys.exit(main())
