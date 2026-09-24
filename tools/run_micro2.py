# -*- coding: utf-8 -*-
"""اجرای دورِ دومِ پژوهشِ ۱۵دقیقه‌ای طبقِ prereg_15m_positioning.json و نوشتنِ گزارش.

holdout فقط برای آزمون‌هایی باز می‌شود که discovery و validation را گذرانده باشند.
اجرا:  python tools/run_micro2.py
"""
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "bot"))
import micro2  # noqa: E402
import paths  # noqa: E402


def main():
    t0 = time.time()
    res = micro2.study()
    ho = micro2.holdout(res)
    tests = {k: {kk: vv for kk, vv in v.items() if kk != "_trades"} for k, v in res.items()}
    out = {"preregistration": "prereg_15m_positioning.json", "run_at_utc": time.strftime("%Y-%m-%d %H:%M", time.gmtime()),
           "n_tests": len(tests), "passed_discovery": [k for k, v in tests.items() if v["passed_discovery"]],
           "passed_validation": [k for k, v in tests.items() if v["passed_validation"]],
           "holdout": ho, "adopted": [k for k, v in ho.items() if v["pass"]], "tests": tests,
           "implementation_notes": ["a metrics row older than 60 minutes at signal time counts as missing data",
                                    "z-scores need at least half of the 2880-bar window to be present"]}
    path = paths.data("research", f"micro_15m_round2_{time.strftime('%Y%m%d_%H%M', time.gmtime())}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1, default=float)
    print(path, f"{time.time() - t0:.0f}s")
    for k, v in tests.items():
        d, val = v["discovery"], v["validation"]
        print(f"{k:34s} disc n={d.get('n'):>5} mean={d.get('mean')} gross={d.get('mean_gross')} p_adj={v['p_adj']}"
              f" | val n={val.get('n')} mean={val.get('mean')} lcb={val.get('lcb')} {'PASS' if v['passed_validation'] else ''}")
    print("holdout:", ho)


if __name__ == "__main__":
    main()
