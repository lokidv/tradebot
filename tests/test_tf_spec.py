# -*- coding: utf-8 -*-
"""ثبتِ واحدِ تایم‌فریم‌ها (bot/tf_spec.py) و افزودنِ 5m.

* هر ماژولِ مصرف‌کننده همان جدول‌های ثبت را می‌خواند (نه کپیِ خودش) و هیچ ``.get(tf, 60)`` بی‌صدایی نمانده؛
* 5m فقط تحلیل و ماتریسِ تصمیم است: calib/autotrader/edge_book/research/gates آن را نمی‌شناسند؛
* مسیرِ کاملِ تحلیلِ 5m در main با بازارِ ساختگی (بی‌شبکه) اجرا می‌شود و هرگز قابلِ معامله نیست.
"""
import json
import os
import re
import sys
import tempfile
import time
import unittest
from unittest import mock

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "bot"))
import _hermetic  # noqa: E402,F401  — پیش از هر ماژولِ ربات: داده به پوشهٔ موقت

import tf_spec  # noqa: E402
import autotrader  # noqa: E402
import calib  # noqa: E402
import candidates  # noqa: E402
import costs  # noqa: E402
import decision  # noqa: E402
import edge_book  # noqa: E402
import engine  # noqa: E402
import features  # noqa: E402
import gates  # noqa: E402
import main  # noqa: E402
import market  # noqa: E402
import research  # noqa: E402
import shadow  # noqa: E402

BOT = os.path.join(ROOT, "bot")
# فایل‌هایی که جدولِ تایم‌فریمِ محلی داشتند و حالا باید فقط از ثبت بخوانند
REGISTRY_USERS = ("engine.py", "main.py", "costs.py", "candidates.py", "shadow.py", "report.py", "decision.py")


class RegistryTests(unittest.TestCase):
    def test_every_table_covers_exactly_the_timeframes(self):
        self.assertEqual(tf_spec.TFS, ("5m", "15m", "1h", "4h", "1d"))
        for name in ("MINUTES", "BAR_MS", "BARS_PER_DAY", "HORIZON", "HTF_OF", "ANALYSIS_TTL", "KLINE_TTL",
                     "KLINE_GRACE_SEC", "DRIFT_CAP", "MANUAL_RISK_FRAC", "FA"):
            self.assertEqual(set(getattr(tf_spec, name)), set(tf_spec.TFS), name)
        mins = [tf_spec.MINUTES[tf] for tf in tf_spec.TFS]
        self.assertEqual(mins, sorted(mins))                               # کوتاه→بلند
        for tf in tf_spec.TFS:
            self.assertEqual(tf_spec.BAR_MS[tf], tf_spec.MINUTES[tf] * 60_000)
            self.assertEqual(tf_spec.BARS_PER_DAY[tf] * tf_spec.MINUTES[tf], 1440)
            htf = tf_spec.HTF_OF[tf]
            if htf is not None:
                self.assertIn(htf, tf_spec.TFS)
                self.assertGreater(tf_spec.MINUTES[htf], tf_spec.MINUTES[tf])
            self.assertLess(tf_spec.KLINE_GRACE_SEC[tf], tf_spec.MINUTES[tf] * 60)

    def test_five_minute_values(self):
        self.assertEqual((tf_spec.MINUTES["5m"], tf_spec.HORIZON["5m"], tf_spec.HTF_OF["5m"],
                          tf_spec.KLINE_GRACE_SEC["5m"], tf_spec.BARS_PER_DAY["5m"]), (5, 48, "1h", 60, 288))
        self.assertEqual(tf_spec.HTF_OF["1d"], None)

    def test_unknown_timeframe_is_an_error_not_sixty_minutes(self):
        for fn in (tf_spec.minutes, tf_spec.bar_ms, tf_spec.bars_per_day):
            with self.assertRaises(KeyError):
                fn("7m")
        self.assertFalse(tf_spec.is_known("7m"))
        with self.assertRaises(KeyError):
            costs.quote_volume_24h([1.0], [1.0], 0, "7m")
        self.assertFalse(candidates.log_candidate(
            {"symbol": "BTCUSDT", "side": "long", "zt": time.time() * 1000, "entry": 100.0, "sl": 99.0,
             "atr14": 1.0}, "7m"))

    def test_closed_between(self):
        day = 1_700_000_000 - (1_700_000_000 % 86400)
        self.assertTrue(tf_spec.closed_between("5m", day, day + 300))
        self.assertFalse(tf_spec.closed_between("15m", day, day + 300))
        self.assertTrue(tf_spec.closed_between("1d", day - 1, day))


class ConsumersUseTheRegistryTests(unittest.TestCase):
    def test_tables_are_the_registry(self):
        self.assertIs(engine.TF_MINUTES, tf_spec.MINUTES)
        self.assertIs(engine.HORIZON, tf_spec.HORIZON)
        self.assertEqual(main.TFS, list(tf_spec.TFS))
        for name in ("HTF_OF", "ANALYSIS_TTL", "DRIFT_CAP", "KLINE_GRACE_SEC"):
            self.assertIs(getattr(main, name), getattr(tf_spec, name), name)
        self.assertIs(costs.BARS_PER_DAY, tf_spec.BARS_PER_DAY)
        self.assertIs(costs.TF_MS, tf_spec.BAR_MS)
        self.assertIs(candidates.TF_MINUTES, tf_spec.MINUTES)
        self.assertIs(shadow.TF_MINUTES, tf_spec.MINUTES)
        self.assertEqual(decision.TFS, tf_spec.TFS)
        self.assertIs(decision.TF_MINUTES, tf_spec.MINUTES)
        self.assertEqual(costs.quote_volume_24h(np.ones(400), np.ones(400), 399, "5m"), 288.0)

    def test_no_silent_default_or_private_timeframe_table_left(self):
        silent = re.compile(r"\.get\(\s*(req\.)?tf[^,()]*(\([^)]*\))?\s*,\s*(60|24|3_600_000|2\.5|300)\s*\)")
        table = re.compile(r"\{\s*\"15m\"\s*:\s*[0-9]")
        for fn in REGISTRY_USERS:
            with open(os.path.join(BOT, fn), encoding="utf-8") as f:
                src = f.read()
            self.assertIsNone(silent.search(src), f"{fn}: {silent.search(src)}")
            self.assertIsNone(table.search(src), f"{fn}: جدولِ تایم‌فریمِ محلی")
            self.assertNotIn("TF_MINUTES.get(", src, fn)


class FiveMinuteStaysOutTests(unittest.TestCase):
    def test_model_timeframes_exclude_5m(self):
        self.assertNotIn("5m", tf_spec.MODEL_TFS)
        self.assertTrue(set(tf_spec.MODEL_TFS) < set(tf_spec.TFS))
        self.assertNotIn("5m", edge_book.ALLOWED_TFS)
        self.assertNotIn("5m", {h["tf"] for h in research.DEFAULT_FAMILY})
        for table in (calib.ACTION_ROLLING_WINDOWS, calib.BARS, calib.TF_MS):
            self.assertNotIn("5m", table)

    def test_calib_returns_none_for_5m_without_crashing(self):
        table = {"version": calib.CALIB_VERSION, "tfs": {"1h": {}}}
        with mock.patch.object(calib, "load", return_value=table):
            self.assertIsNone(calib.predict_dir("5m", [0.0] * 23))
            self.assertIsNone(calib.predict_action("5m", [0.0] * 23, [0.0] * 23, 0.5))
            self.assertIsNone(calib.predict("5m", [0.0] * 23, 0.5))

    def test_gates_never_allow_a_5m_combo(self):
        for setup in ("zx", "sq", "pb", "fd", "rg", "ap", None):
            for side in ("long", "short"):
                self.assertFalse(gates.is_combo_allowed("5m", setup, side))
                self.assertFalse(gates.is_combo_allowed("5m", setup, side, symbol="BTCUSDT"))

    def test_autotrader_never_scans_5m(self):
        for lv in range(1, 11):
            scan = autotrader.level_params(lv)["scan_tfs"]
            self.assertNotIn("5m", scan, lv)
            self.assertTrue(set(scan) <= set(tf_spec.MODEL_TFS), lv)
        pockets = {"5m|zx|long": {"tf": "5m"}, "1h|zx|long": {"tf": "1h"}}
        with mock.patch.object(edge_book, "refresh", return_value=(pockets, {}, {})):
            for lv in range(1, 11):
                self.assertNotIn("5m", edge_book.scan_tfs_for(autotrader.level_params(lv)["scan_tfs"]))
        with open(os.path.join(BOT, "autotrader.py"), encoding="utf-8") as f:
            self.assertNotIn('"5m"', f.read())


def _klines_5m(n=420, tf="5m", seed=7):
    """کندل‌های بستهٔ ساختگی که آخرینشان کندلِ تازه‌بسته‌شده است (گاردِ فیدِ مرده نگیرد)."""
    step = tf_spec.bar_ms(tf)
    last = (int(time.time() * 1000) // step) * step - step
    rng = np.random.RandomState(seed)
    c = 100 * np.exp(np.cumsum(rng.normal(0, 0.002, n)))
    o = np.r_[c[0], c[:-1]]
    wig = np.abs(rng.normal(0, 0.001, n)) * c
    return {"t": (last - (n - 1 - np.arange(n)) * step).tolist(), "o": o.tolist(),
            "h": (np.maximum(o, c) + wig).tolist(), "l": (np.minimum(o, c) - wig).tolist(),
            "c": c.tolist(), "v": rng.lognormal(8, 0.4, n).tolist()}


class FiveMinuteAnalysisPathTests(unittest.TestCase):
    """``main._compute_analysis(sym, "5m")`` با بازارِ ساختگی: زنجیرهٔ 5m→1h و BTC، بدونِ مدل، بدونِ مجوز."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self._old = gates.GATES_PATH
        gates.GATES_PATH = os.path.join(self.tmp.name, "gates.json")
        with open(gates.GATES_PATH, "w", encoding="utf-8") as f:
            json.dump(dict(gates.DEFAULTS), f)
        gates.load_gates(force=True)
        self.calls = []

        def fake_analysis(symbol, tf, max_age=None):
            self.calls.append((symbol, tf))
            return {"symbol": symbol, "tf": tf, "z": 0.9, "z_prev": 0.7, "zt": 0}

        self.patches = [
            mock.patch.object(main, "get_analysis", side_effect=fake_analysis),
            mock.patch.object(market, "get_klines", side_effect=lambda s, tf, *a, **k: _klines_5m(tf=tf)),
            mock.patch.object(market, "get_klines_cached", return_value=None),
            mock.patch.object(market, "get_top_symbols", return_value=([], {})),
            mock.patch.object(market, "get_history", side_effect=AssertionError("بی‌شبکه")),
            mock.patch.object(features, "live_funding_z", return_value=0.0),
            mock.patch.object(main, "_funding_info", return_value={}),
            mock.patch.object(main, "_oi_info", return_value={"ok": False}),
            mock.patch.object(main, "_btc_macro", return_value={"regime": "chop", "breadth": 0.0}),
            mock.patch.object(main, "_live_edge_book", return_value=({}, {})),
            mock.patch.object(main, "_suspended_setups", return_value={}),
            mock.patch.object(main, "_suspended_tfs", return_value={}),
            mock.patch.object(calib, "load", return_value={"version": calib.CALIB_VERSION, "built_at": time.time(),
                                                           "tfs": {"1h": {}}}),
        ]
        for p in self.patches:
            p.start()
        main._macro_cache.pop("5m", None)

    def tearDown(self):
        for p in reversed(self.patches):
            p.stop()
        gates.GATES_PATH = self._old
        gates._cache.update(mtime=None, data=None)
        self.tmp.cleanup()

    def test_5m_analysis_runs_end_to_end_and_is_never_tradeable(self):
        res = main._compute_analysis("ETHUSDT", "5m")
        self.assertNotIn("error", res, res.get("error"))
        self.assertEqual((res["tf"], res["symbol"]), ("5m", "ETHUSDT"))
        self.assertIn(("ETHUSDT", "1h"), self.calls)                     # HTF_OF["5m"] == "1h"
        self.assertIn(("BTCUSDT", "5m"), self.calls)
        self.assertNotIn(("ETHUSDT", "4h"), self.calls)
        self.assertEqual(res["forecast"]["horizon_min"], 48 * 5)
        tr = res["trade"]
        self.assertFalse(tr.get("tradeable", False))                     # gates.json مرجع است؛ 5m هرگز مجاز نیست
        if tr.get("time_stop_min") is not None:
            self.assertEqual(tr["time_stop_min"], 40 * 5)
        market.get_history.assert_not_called()                           # طلای PAXG برای 5m صفحه‌بندی نمی‌شود
        self.assertEqual(main._macro("5m"), {"gold": 0.0})
        d = decision.from_analysis(res)
        self.assertEqual(d["tf"], "5m")
        self.assertIn(d["action"], ("long", "short", "wait"))

    def test_overview_accepts_5m_but_keeps_it_out_of_the_candidate_ledger(self):
        res = main._compute_analysis("ETHUSDT", "5m")
        self.assertNotIn("error", res)
        main.get_analysis.side_effect = None
        with mock.patch.object(candidates, "log_candidate", return_value=True) as lc, \
                mock.patch.object(main, "_symbol_cost", return_value=0.11):
            for tf in ("5m", "1h"):
                a = json.loads(json.dumps(res))
                a.update(tf=tf, zt=int(time.time() * 1000) - 60_000)
                a["trade"].update(side="long", entry=100.0, sl=99.0, tp=101.8, atr14=0.8, risk_pct=1.0)
                main.get_analysis.return_value = a
                out = main.overview(tf)
                self.assertEqual(out["tf"], tf)
                self.assertTrue(out["coins"])
        self.assertEqual({c.args[1] for c in lc.call_args_list}, {"1h"})   # 5m ⇒ هیچ ردیفِ پژوهشی


if __name__ == "__main__":
    unittest.main()
