# -*- coding: utf-8 -*-
"""یک‌بار (۲۰۲۶-۰۹-۱۵): رویدادهای ساختگیِ تست‌ها را در ژورنالِ واقعی باطل کن.

تست‌ها gates.json را به پوشهٔ موقت می‌بردند ولی ژورنال را نه؛ هر اجرای کاملِ
تست‌ها ۲۲ رویدادِ GATE_CHANGE ساختگی در bot/data/journal.jsonl می‌نوشت. رویدادِ
واقعی فقط آن‌هایی است که با سندهای متعهدشده در git جور درمی‌آیند:

* پیش‌ثبت: ``registered_at`` در preregistration.json
* حکم:     ``judged_at``   در final_test/judged_<hash>.json

هیچ خطی پاک یا بازنویسی نمی‌شود: یک رویدادِ QUARANTINE بقیهٔ GATE_CHANGEها را با
``(seq, ts)`` باطل می‌کند و خطوطِ اصلی برای ممیزی در فایل می‌مانند. رویدادهای
غیرِ GATE_CHANGE دست نمی‌خورند.

اجرا:  python tools/clean_polluted_journal.py          (فقط گزارش)
       python tools/clean_polluted_journal.py --apply  (ثبتِ اصلاحیه)
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "bot"))

import journal  # noqa: E402
import research  # noqa: E402

TOLERANCE_SEC = 5.0


def _real_anchors():
    doc = research.load_prereg()
    if not doc:
        sys.exit("پیش‌ثبتی نیست")
    judged = research.last_judgement() or {}
    anchors = [("preregister", float(doc["registered_at"]), f"preregistered trial {doc['n_trials']}")]
    if judged.get("judged"):
        anchors.append(("judge", float(judged["judged_at"]), f"judged prereg {doc['hash'][:12]}"))
    return anchors


def main(apply):
    rows = journal.events()                     # از قبل باطل‌شده‌ها دوباره شمرده نمی‌شوند
    anchors = _real_anchors()
    keep, void, matched = [], [], set()
    for r in rows:
        if r.get("kind") != journal.GATE_CHANGE:
            keep.append(r)
            continue
        hit = None
        for name, ts, reason in anchors:
            if name not in matched and r.get("reason") == reason \
                    and 0 <= float(r.get("ts") or 0) - ts <= TOLERANCE_SEC:
                hit = name
        if hit:
            matched.add(hit)
            keep.append(r)
        else:
            void.append(r)
    print(f"events: {len(rows)}  keep: {len(keep)}  void: {len(void)}")
    for r in keep:
        print("  keep", r.get("seq"), r.get("kind"), r.get("reason"), r.get("ts"))
    missing = {a[0] for a in anchors} - matched
    if missing:
        sys.exit(f"رویدادِ واقعیِ {missing} در ژورنال پیدا نشد — دست نمی‌زنم")
    if not void:
        print("چیزی برای باطل‌کردن نیست")
        return
    if not apply:
        print("(فقط گزارش؛ برای ثبتِ اصلاحیه --apply)")
        return
    ev = journal.quarantine(void, "GATE_CHANGE events written by the test suite "
                                  "(tests redirected gates.json but not the journal)")
    print("quarantine event seq", ev["seq"], "voids", ev["count"])


if __name__ == "__main__":
    main(apply="--apply" in sys.argv)
