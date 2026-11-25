# -*- coding: utf-8 -*-
"""سیاستِ نمایشِ صفحه (prereg_signal_v3.json): «بخر/بفروش» فقط برای سیگنالِ v3 در ۴ ساعته و روزانه.

تابعِ ``cellCall`` خودِ index.html در node اجرا می‌شود: تایم‌فریمِ کوتاه، کارنامهٔ نیامده یا کهنه، و تایم‌فریمِ
خاموش‌شده با ترمز هرگز جهتِ معامله نمی‌دهند.
"""
import json
import os
import shutil
import subprocess
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "bot"))
import _hermetic  # noqa: E402,F401  — پیش از هر ماژولِ ربات: داده به پوشهٔ موقت

HTML = os.path.join(ROOT, "bot", "static", "index.html")

HARNESS = r"""
const window = {};
const dmAct = x => (x && (x.action === "long" || x.action === "short")) ? x.action : "wait";
const leanOf = x => (x && x.lean) || null;
%s
const cases = %s;
const out = cases.map(c => { window.SIG = c.sig; const k = cellCall(c.sym, c.tf, c.x); return {kind: k.kind, side: k.side, label: k.label}; });
console.log(JSON.stringify(out));
"""


def _sig(state="wait", muted=False, actionable=True, side="long", code=None, base=None):
    plan = {"side": side, "entry": 100.0, "sl": 98.0, "tp": 103.6}
    cell = {"state": state, "muted": muted, "fresh": plan if state == "signal" else None,
            "open": plan if state == "in_trade" else None, "last": {"base": base}}
    return {"actionable": actionable, "signal_tfs": ["4h", "1d"],
            "tfs": {"4h": {"brake": {"muted": muted, "code": code}}},
            "cells": {"BTCUSDT": {"4h": cell}}}


@unittest.skipUnless(shutil.which("node"), "node نصب نیست")
class CellCallTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with open(HTML, encoding="utf-8") as f:
            html = f.read()
        cls.js = html[html.index("const SIG_TFS_DEFAULT"):html.index("function sgRecLine(tf){")]
        cls.html = html

    def _run(self, cases):
        src = HARNESS % (self.js, json.dumps(cases, ensure_ascii=False))
        r = subprocess.run(["node", "-e", src], capture_output=True, text=True, timeout=60)
        self.assertEqual(r.returncode, 0, r.stderr)
        return json.loads(r.stdout)

    def test_short_timeframes_are_status_only_whatever_the_rule_says(self):
        long_, short_ = {"action": "long"}, {"action": "short"}
        cases = [{"sig": sig, "sym": "BTCUSDT", "tf": tf, "x": x}
                 for sig in (None, _sig("signal")) for tf in ("5m", "15m", "1h") for x in (long_, short_, {"action": "wait"})]
        for k in self._run(cases):
            self.assertEqual((k["kind"], k["side"]), ("status", None))
            for word in ("خرید", "فروش", "لانگ", "شورت"):
                self.assertNotIn(word, k["label"])

    def test_no_call_without_a_current_record(self):
        x = {"action": "long"}
        out = self._run([{"sig": None, "sym": "BTCUSDT", "tf": "4h", "x": x},
                         {"sig": _sig("signal", actionable=False), "sym": "BTCUSDT", "tf": "4h", "x": x},
                         {"sig": _sig("signal"), "sym": "ETHUSDT", "tf": "4h", "x": x}])       # خانه‌ای که نیامده
        self.assertEqual([(k["kind"], k["side"]) for k in out], [("unknown", None)] * 3)

    def test_a_muted_timeframe_never_shows_a_call(self):
        x = {"action": "short"}
        out = self._run([{"sig": _sig(st, muted=True, side="short"), "sym": "BTCUSDT", "tf": "4h", "x": x}
                         for st in ("signal", "in_trade", "wait")]
                        + [{"sig": _sig("signal", muted=True, code="failed_test"), "sym": "BTCUSDT", "tf": "4h", "x": x}])
        self.assertEqual([(k["kind"], k["side"]) for k in out], [("muted", None)] * 4)

    def test_states_of_an_open_timeframe(self):
        x = {"action": "long"}
        out = self._run([{"sig": _sig("signal", side="long"), "sym": "BTCUSDT", "tf": "4h", "x": x},
                         {"sig": _sig("signal", side="short"), "sym": "BTCUSDT", "tf": "4h", "x": x},
                         {"sig": _sig("in_trade", side="short"), "sym": "BTCUSDT", "tf": "4h", "x": x},
                         {"sig": _sig("filtered", base="long"), "sym": "BTCUSDT", "tf": "4h", "x": x},
                         {"sig": _sig("wait"), "sym": "BTCUSDT", "tf": "4h", "x": x}])
        self.assertEqual([(k["kind"], k["side"]) for k in out],
                         [("signal", "long"), ("signal", "short"), ("in_trade", "short"), ("filtered", None), ("wait", None)])
        self.assertIn("خرید", out[0]["label"])
        self.assertIn("فروش", out[1]["label"])

    def test_page_wiring(self):
        html = self.html
        self.assertIn('id="sgWrap"', html)
        self.assertLess(html.index('id="sgWrap"'), html.index('id="dmWrap"'))          # سیگنال‌ها بالای ماتریس
        self.assertIn("loadSignals(); loadDecisions();", html)                         # بارِ اول
        self.assertIn("setInterval(() => loadSignals(), 60000);", html)
        self.assertIn('jfetch("/api/signals/size"', html)
        # ماتریس و ستونِ «اقدام» فقط از cellCall برچسب می‌گیرند، نه مستقیم از تصمیمِ قاعده
        for head in ("function dmCell(sym, tf, x){", "function dmActionHtml(c, x){"):
            body = html[html.index(head):html.index("\n}\n", html.index(head))]
            self.assertIn("cellCall(", body, head)
            self.assertNotIn("callLabel(", body, head)


if __name__ == "__main__":
    unittest.main()
