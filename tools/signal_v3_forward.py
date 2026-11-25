# -*- coding: utf-8 -*-
"""کارنامهٔ رو-به-جلوی سیگنالِ v3 (bot/signal_v3.py؛ پیش‌ثبت: bot/data/research/prereg_signal_v3.json).

همان مسیری که سرور می‌سازد: بازپخشِ قطعی روی کندل‌های بسته‌شدهٔ اسپاتِ بایننس از ۲۰۲۶-۰۹-۰۱، فقط معامله‌های
matured، ترمزِ هر تایم‌فریم، و آزمونِ بلوکِ هفتگی (تا موعدِ داوری فقط توصیفی). این ابزار همان را با همهٔ
معامله‌ها در یک گزارش می‌نویسد تا قابلِ بازبینی باشد:

  research/signal_v3_forward_<UTC>.json

اجرا:
  python tools/signal_v3_forward.py              # گزارشِ کامل
  python tools/signal_v3_forward.py --no-write   # فقط چاپ
"""
import argparse
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "bot"))

import decision  # noqa: E402
import paths  # noqa: E402
import signal_v3 as s3  # noqa: E402

for _s in (sys.stdout, sys.stderr):          # کنسولِ ویندوز (cp1252) متنِ فارسی را نمی‌پذیرد
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass


def _utc(ms):
    return time.strftime("%Y-%m-%d %H:%M", time.gmtime(int(ms) / 1000))


def _f(x, d=3):
    return "—" if x is None else f"{x:+.{d}f}"


def _trade_rows(cells, key):
    rows = []
    for c in cells:
        for x in c.get(key) or []:
            row = {"sym": c["sym"], "tf": c["tf"], "t": x["t"], "utc": _utc(x["t"]), "side": x["side"],
                   "setup": x.get("setup"), "entry": x["entry"], "sl": x["sl"], "tp": x["tp"],
                   "risk_pct": round(x["risk_pct"], 4), "outcome": x["outcome"], "bars_held": x["bars_held"],
                   "gross_r": round(x["gross_r"], 4), "net_r": round(x["net_r"], 4), "exit_utc": _utc(x["exit_t"]),
                   "matured": bool(x.get("matured"))}
            if key == "trades":
                row.update(mom_pct=round(x["mom"] * 100, 2) if x.get("mom") is not None else None,
                           votes=x.get("votes"), z=round(x["z"], 2))
            rows.append(row)
    return sorted(rows, key=lambda r: (r["t"], r["sym"], r["tf"]))


def summary_lines(rep):
    out = [f"از {_utc(rep['t0_ms'])} تا {_utc(rep['pooled']['matured_until'])} (matured) — {rep['venue']}، "
           f"هزینه {rep['cost_pct']}٪، کامل: {rep['complete']}"]
    for tf in s3.SIGNAL_TFS:
        t = rep["tfs"][tf]
        v, r, b = t["stats"], t["rule"], t["brake"]
        out.append(f"{tf:>3}: v3 n={v['n']} mean={_f(v['mean'])} CI[{_f(v['ci_lo'])},{_f(v['ci_hi'])}] win={v['win_pct']}٪ "
                   f"long {v['long_n']}/{_f(v['long_mean'])} short {v['short_n']}/{_f(v['short_mean'])} | "
                   f"rule n={r['n']} mean={_f(r['mean'])} | maker mean={_f(t['maker']['mean'])} | "
                   f"brake: {'MUTED ' + str(b['code']) if b['muted'] else 'open'} (armed={b['armed']})")
    p = rep["pooled"]
    out.append(f"pooled: v3 n={p['stats']['n']} mean={_f(p['stats']['mean'])} "
               f"CI[{_f(p['stats']['ci_lo'])},{_f(p['stats']['ci_hi'])}] | rule n={p['rule']['n']} mean={_f(p['rule']['mean'])} | "
               f"bootstrap p={p['test']['p']} (توصیفی) diff_vs_rule={_f((p['test'].get('rule') or {}).get('diff'))} "
               f"p_diff={(p['test'].get('rule') or {}).get('p_diff')}")
    for sym, row in rep["cells"].items():
        out.append("  " + sym + ": " + " | ".join(
            f"{tf} n={c['stats']['n']} {_f(c['stats']['mean'])} [{c['state']}{' muted' if c['muted'] else ''}]"
            for tf, c in row.items()))
    for s in rep["signals"]:
        out.append(f"  ▶ {s['sym']} {s['tf']} {s['state']} {s['side']} entry={s['entry']} sl={s['sl']} tp={s['tp']} "
                   f"signal={_utc(s['signal_t'])}{' (muted)' if s['muted'] else ''}")
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description="کارنامهٔ رو-به-جلوی سیگنالِ v3")
    ap.add_argument("--no-write", action="store_true", help="گزارش روی دیسک نوشته نشود")
    a = ap.parse_args(argv)
    s3.check_prereg()
    now_ms = int(time.time() * 1000)
    rep = s3.build(lambda sym, tf: s3.market_loader(sym, tf, now_ms), now_ms,
                   judgements=s3.read_judgements(), keep_cells=True)
    cells = rep.pop("_cells")
    rep["run_utc"] = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(now_ms / 1000))
    rep["note"] = ("Descriptive until the pre-registered judgement dates (look 1: 2027-10-15). Only matured trades "
                   "are counted. Bars are closed Binance spot klines; each cell lists its bar count, range and sha256.")
    rep["trades"] = _trade_rows(cells, "trades")
    rep["rule_trades"] = _trade_rows(cells, "rule_trades")
    for line in summary_lines(rep):
        print(line, flush=True)
    if rep["errors"]:
        print("خطا: " + json.dumps(rep["errors"], ensure_ascii=False), file=sys.stderr, flush=True)
    if not a.no_write:
        path = paths.data("research", time.strftime("signal_v3_forward_%Y%m%d_%H%M%S.json", time.gmtime(now_ms / 1000)))
        with open(path, "w", encoding="utf-8") as f:
            json.dump(decision_safe(rep), f, ensure_ascii=False, indent=1)
        print("saved:", path, flush=True)
    return 0 if rep["complete"] else 1


def decision_safe(x):
    """numpy → نوع‌های سادهٔ JSON."""
    import numpy as np
    if isinstance(x, dict):
        return {str(k): decision_safe(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [decision_safe(v) for v in x]
    if isinstance(x, np.generic):
        return x.item()
    return x


if __name__ == "__main__":
    sys.exit(main())
