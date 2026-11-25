# -*- coding: utf-8 -*-
"""سیاستِ نمایشِ صفحه (prereg_signal_v3.json): «بخر/بفروش» فقط برای سیگنالِ v3 در ۴ ساعته و روزانه.

تابعِ ``cellCall`` خودِ index.html در node اجرا می‌شود: تایم‌فریمِ کوتاه، کارنامهٔ نیامده یا کهنه، و تایم‌فریمِ
خاموش‌شده با ترمز هرگز جهتِ معامله نمی‌دهند.
"""
import json
import os
import re
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


@unittest.skipUnless(shutil.which("node"), "node نصب نیست")
class TextDirectionTests(unittest.TestCase):
    """متنِ فارسی با «R» و عدد: اگر تکهٔ لاتین در قابِ خودش نباشد، مرورگر عددِ بعدی را به آن می‌چسباند و
    «BTC · ۱۵ دقیقه» به شکلِ «۱۵ · BTC دقیقه» دیده می‌شود. این تست همان الگوها را در منبعِ صفحه می‌بندد."""

    @classmethod
    def setUpClass(cls):
        with open(HTML, encoding="utf-8") as f:
            cls.html = f.read()

    def _helpers(self):
        lines = [ln for ln in self.html.splitlines()
                 if re.match(r"const (esc|faN|faS|bR|escNum) = ", ln)]
        self.assertEqual(len(lines), 5, lines)
        return "\n".join(lines)

    def _node(self, expr):
        r = subprocess.run(["node", "-e", self._helpers() + "\nconsole.log(JSON.stringify(%s));" % expr],
                           capture_output=True, text=True, timeout=60)
        self.assertEqual(r.returncode, 0, r.stderr)
        return json.loads(r.stdout)

    def test_signed_numbers_in_engine_text_get_their_own_ltr_box(self):
        box = lambda t: '<bdi dir="ltr">%s</bdi>' % t                                  # noqa: E731
        texts = ["z امتیازِ ترکیبی +1.43 از آستانهٔ +1.2 گذشته", "z فعلی -0.20 است؛ برای پیشنهاد به ±1.2 نیاز است",
                 "پیش‌ثبت 2026-09-24 و رو-به-جلو", "<b>x</b> −5%"]
        out = self._node("%s.map(escNum)" % json.dumps(texts, ensure_ascii=False))
        self.assertEqual(out[0], "z امتیازِ ترکیبی %s از آستانهٔ %s گذشته" % (box("+1.43"), box("+1.2")))
        self.assertEqual(out[1], "z فعلی %s است؛ برای پیشنهاد به %s نیاز است" % (box("-0.20"), box("±1.2")))
        self.assertEqual(out[2], texts[2])                                             # تاریخ و خط‌تیرهٔ واژه دست نمی‌خورد
        self.assertEqual(out[3], "&lt;b&gt;x&lt;/b&gt; " + box("−5%"))                 # هنوز HTML را بی‌اثر می‌کند

    def test_r_values_are_boxed(self):
        self.assertEqual(self._node("bR(-0.41)"), '<bdi dir="ltr">−۰٫۴۱R</bdi>')

    def test_no_bare_latin_token_runs_into_the_next_persian_number(self):
        html = re.sub(r"(?m)\s//\s.*$", "", self.html)                                # توضیحِ کد متنِ صفحه نیست
        # «…R، ۱۵ دقیقه»: مقدارِ R باید از bR بیاید، نه متنِ ساده
        self.assertEqual(re.findall(r"[۰-۹]R[،؛] ?[۰-۹]", html), [])
        self.assertEqual(re.findall(r"[−-][۰-۹]+٫[۰-۹]+R(?!</bdi>)", html), [])
        # نمادِ ارز پیش از «۴ ساعت/۱۵ دقیقه»: در HTML داخلِ <bdi>، در متنِ ساده با نشانهٔ راست‌به‌چپ
        heads = re.findall(r"(.{6}) · \$\{TF_FA\[", html)
        self.assertEqual(len(heads), 4, heads)
        for h in heads:
            self.assertTrue(h.endswith("</bdi>") or h.endswith("\\u200F"), h)
        # آستانهٔ قاعده یک فرمولِ چپ‌به‌راست است
        self.assertEqual(html.count("z ≥ ۱٫۲"), html.count('<bdi dir="ltr">z ≥ ۱٫۲</bdi>'))
        self.assertNotIn("۱R = ۱٫۳ برابرِ", html)


if __name__ == "__main__":
    unittest.main()
