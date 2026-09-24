# -*- coding: utf-8 -*-
"""ابزارِ تاریخچهٔ هسته (tools/fetch_core_history.py) — ممیزی DATA-3: روزِ «کامل ولی خراب» آرشیوِ ماهانه
(کندل‌های تخت و بی‌حجم که فایلِ روزانه ندارد) از آرشیوِ روزانه دوباره گرفته و گزارش می‌شود؛ بی‌شبکه."""
import datetime as dt
import importlib.util
import os
import sys
import tempfile
import unittest

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "bot"))
import _hermetic  # noqa: E402,F401  — پیش از هر ماژولِ ربات: داده به پوشهٔ موقت

STEP = 300_000
DAY = 86_400_000
SPAN = ((2023, 11), (2023, 11))
M0 = int(dt.datetime(2023, 11, 1, tzinfo=dt.timezone.utc).timestamp() * 1000)


def load_tool():
    spec = importlib.util.spec_from_file_location("fetch_core_history_t", os.path.join(ROOT, "tools", "fetch_core_history.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def month_rows(seed=0):
    """یک ماهِ کاملِ ۵دقیقهٔ سالم با ستون‌های KLINE_KEYS."""
    n = 30 * DAY // STEP
    rng = np.random.RandomState(seed)
    c = 37000 * np.exp(np.cumsum(rng.normal(0, 0.001, n)))
    o = np.r_[c[0], c[:-1]]
    h, l = np.maximum(o, c) * 1.0005, np.minimum(o, c) * 0.9995
    v = rng.lognormal(5, 0.3, n)
    t = M0 + np.arange(n) * STEP
    return np.column_stack([t, o, h, l, c, v, v * c, np.floor(v) + 1, v * 0.5]).astype(float)


def freeze(a, day, k0, k1):
    """کندل‌های ``k0..k1`` روزِ ``day`` (۱-مبنا) تخت و بی‌حجم (همان شکلِ خرابیِ BTCUSDT ۲۰۲۳-۱۱-۱۰)."""
    b = a.copy()
    i = (day - 1) * (DAY // STEP)
    rows = slice(i + k0, i + k1)
    px = b[i + k0 - 1, 4]
    b[rows, 1:5] = px
    b[rows, 5:] = 0.0
    return b


class _Pool:
    @staticmethod
    def map(f, xs):
        return [f(x) for x in xs]


class FlatDayRefetchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fch = load_tool()

    def setUp(self):
        self.good = month_rows()
        self.calls = []

    def parse_from(self, daily):
        """``parse`` ساختگی: نشانیِ روزانه ⇒ ردیف‌های همان روز از ``daily`` (نبود ⇒ [] یعنی ۴۰۴)."""
        def parse(cli, url):
            self.calls.append(url)
            k = (dt.date.fromisoformat(url[-14:-4]) - dt.date(2023, 11, 1)).days      # «…-YYYY-MM-DD.zip»
            if daily is None or k < 0 or k >= 30:
                return []
            per = DAY // STEP
            return [tuple(r) for r in daily[k * per:(k + 1) * per]]
        return parse

    def fill(self, a, daily, check_flat=True):
        url = "https://x/BTCUSDT-5m-{d}.zip"
        return self.fch._fill(None, _Pool(), a.copy(), url, self.parse_from(daily), STEP, SPAN, check_flat)

    def test_corrupt_day_is_refetched_and_repaired(self):
        bad = freeze(self.good, 10, 180, 200)                        # ۲۰ کندلِ یخ‌زده، ردیف‌های روز کامل
        self.assertEqual(int(self.fch._flat_mask(bad).sum()), 20)
        out, added, fix = self.fill(bad, self.good)
        self.assertEqual(self.calls, ["https://x/BTCUSDT-5m-2023-11-10.zip"])   # فقط همان روز
        self.assertEqual(added, 0)
        self.assertEqual(fix["repaired"], ["2023-11-10 (20->0 flat bars)"])
        self.assertTrue(np.array_equal(out, self.good))

    def test_real_halt_in_the_daily_file_too_is_kept_and_reported(self):
        halt = freeze(self.good, 29, 75, 78)
        out, added, fix = self.fill(halt, halt)
        self.assertEqual(fix["daily_also_flat"], ["2023-11-29 (3->3 flat bars)"])
        self.assertEqual(fix["repaired"], [])
        self.assertTrue(np.array_equal(out, halt))

    def test_missing_daily_file_and_worse_daily_file_keep_the_monthly_rows(self):
        bad = freeze(self.good, 10, 180, 200)
        out, _, fix = self.fill(bad, None)
        self.assertEqual(fix["no_daily"], ["2023-11-10"])
        self.assertTrue(np.array_equal(out, bad))
        worse = freeze(self.good, 10, 100, 200)
        out, _, fix = self.fill(bad, worse)
        self.assertEqual(fix["daily_worse"], ["2023-11-10 (20->100 flat bars)"])
        self.assertTrue(np.array_equal(out, bad))

    def test_healthy_month_makes_no_request_and_flat_check_is_opt_in(self):
        out, added, fix = self.fill(self.good, self.good)
        self.assertEqual(self.calls, [])
        self.assertEqual((added, sum(len(v) for v in fix.values())), (0, 0))
        bad = freeze(self.good, 10, 180, 200)
        self.fill(bad, self.good, check_flat=False)                        # پرمیوم و مانندِ آن: بی‌ستونِ حجم
        self.assertEqual(self.calls, [])

    def test_missing_day_is_still_filled(self):
        per = DAY // STEP
        gap = np.delete(self.good, np.arange(4 * per, 5 * per), axis=0)   # ۲۰۲۳-۱۱-۰۵ غایب
        out, added, fix = self.fill(gap, self.good)
        self.assertEqual(added, per)
        self.assertTrue(np.array_equal(out, self.good))

    def test_series_saves_a_repaired_existing_file_and_reports_it(self):
        fch = self.fch
        bad = freeze(self.good, 10, 180, 200)
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "um_BTCUSDT_5m.npz")
            fch._save(path, **{k: (bad[:, i].astype(np.int64) if k == "t" else bad[:, i])
                               for i, k in enumerate(fch.KLINE_KEYS)})
            fch.STATS["repaired"].clear()
            msg = fch._series(None, _Pool(), path, fch.KLINE_KEYS, "unused", "https://x/BTCUSDT-5m-{d}.zip",
                              self.parse_from(self.good), STEP, SPAN, force=False, fill=True)
            self.assertIn("repaired 2023-11-10", msg)
            self.assertEqual(fch.STATS["repaired"], ["um_BTCUSDT_5m.npz 2023-11-10 (20->0 flat bars)"])
            with np.load(path) as z:
                self.assertEqual(int(fch._flat_mask(np.column_stack([z[k].astype(float) for k in fch.KLINE_KEYS])).sum()), 0)
            rep = fch.audit_file(path, STEP)
            self.assertEqual(rep["flat_zero_volume_bars"], 0)


class AuditReportTests(unittest.TestCase):
    def test_audit_and_crosscheck_name_the_bad_days(self):
        fch = load_tool()
        good = month_rows(1)
        bad = freeze(good, 10, 180, 200)
        with tempfile.TemporaryDirectory() as d:
            old = fch.OUT
            fch.OUT = d
            try:
                cols = {k: (bad[:, i].astype(np.int64) if k == "t" else bad[:, i]) for i, k in enumerate(fch.KLINE_KEYS)}
                fch._save(os.path.join(d, "um_BTCUSDT_5m.npz"), **cols)
                ok = {k: (good[:, i].astype(np.int64) if k == "t" else good[:, i]) for i, k in enumerate(fch.KLINE_KEYS)}
                fch._save(os.path.join(d, "um_BTCUSDT_1h.npz"), **fch.resample(ok, 3_600_000))
                rep = fch.audit_file(os.path.join(d, "um_BTCUSDT_5m.npz"), STEP)
                cc = fch.crosscheck("BTCUSDT")
            finally:
                fch.OUT = old
        self.assertEqual(rep["flat_zero_volume_bars"], 20)
        self.assertEqual(rep["flat_zero_volume_days"], ["2023-11-10"])
        self.assertGreater(cc["1h"]["ohlc_mismatches"], 0)
        self.assertEqual(cc["1h"]["mismatch_days"], ["2023-11-10"])


if __name__ == "__main__":
    unittest.main()
