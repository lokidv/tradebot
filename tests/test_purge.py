# -*- coding: utf-8 -*-
"""تستِ purge روی مرزِ پنجره‌های dev/cal/test.

قبلاً فقط مرزِ فولدهای Walk-Forward embargo داشت؛ پنجره‌های داخلیِ انتخاب،
کالیبراسیون و آزمونِ نهایی با اندیسِ ردیف بریده می‌شدند و برچسبِ آخرین
رویدادهای یک پنجره تا داخلِ پنجرهٔ بعدی ادامه داشت.
"""
import os
import sys
import unittest

import numpy as np


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "bot"))

import calib  # noqa: E402

HOUR = 3_600_000


class PurgeBoundaryTests(unittest.TestCase):
    def test_rows_whose_label_reaches_the_next_window_are_dropped(self):
        ts = np.arange(100) * HOUR
        a, b = np.arange(0, 60), np.arange(60, 100)
        kept = calib._purge_boundary(a, b, ts, 10 * HOUR)
        self.assertEqual(int(ts[kept].max()), 49 * HOUR)   # ۵۰..۵۹ برچسبشان وارد b می‌شد
        self.assertTrue(np.all(ts[kept] < ts[b].min() - 10 * HOUR + 1))

    def test_zero_purge_changes_nothing(self):
        ts = np.arange(10) * HOUR
        a, b = np.arange(0, 6), np.arange(6, 10)
        np.testing.assert_array_equal(calib._purge_boundary(a, b, ts, 0), a)

    def test_missing_timestamps_leave_rows_untouched(self):
        a, b = np.arange(0, 6), np.arange(6, 10)
        np.testing.assert_array_equal(calib._purge_boundary(a, b, None, 5 * HOUR), a)

    def test_time_partitions_leave_a_gap_at_least_as_wide_as_the_purge(self):
        ts = np.repeat(np.arange(400) * HOUR, 3)
        rows = np.arange(len(ts))
        purge = 36 * HOUR
        dev, cal, test = calib._time_partitions(rows, ts, purge_ms=purge)
        self.assertGreater(len(dev), 0)
        self.assertGreater(len(cal), 0)
        self.assertGreater(len(test), 0)
        self.assertGreaterEqual(ts[cal].min() - ts[dev].max(), purge)
        self.assertGreaterEqual(ts[test].min() - ts[cal].max(), purge)

    def test_no_timestamp_is_shared_across_partitions(self):
        ts = np.repeat(np.arange(300) * HOUR, 5)
        dev, cal, test = calib._time_partitions(np.arange(len(ts)), ts, purge_ms=10 * HOUR)
        for x, y in ((dev, cal), (cal, test), (dev, test)):
            self.assertTrue(set(ts[x]).isdisjoint(set(ts[y])))


if __name__ == "__main__":
    unittest.main()
