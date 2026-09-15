# -*- coding: utf-8 -*-
"""تست‌های پذیرشِ فاز ۲c — جهانِ نقطه‌-در-زمان.

معیار: رویدادی که ارزش در آن تاریخ جزوِ برترین‌ها نبوده، نباید وارد آموزش
شود؛ و ارزی که **بعداً** بزرگ شد نباید تاریخچهٔ پیش از رشدش را به مدل بدهد.
"""
import os
import sys
import time
import unittest

import numpy as np


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "bot"))
import _hermetic  # noqa: E402,F401  — پیش از هر ماژولِ ربات: داده به پوشهٔ موقت

import universe  # noqa: E402

DAY = universe.DAY_MS


def ts_of(y, m, d):
    return int(time.mktime((y, m, d, 0, 0, 0, 0, 0, 0)) - time.timezone) * 1000


def daily(start, n, vol):
    t = [start + k * DAY for k in range(n)]
    return {"t": t, "c": [1.0] * n, "v": list(vol) if hasattr(vol, "__len__") else [vol] * n}


class SnapshotTests(unittest.TestCase):
    def setUp(self):
        start = ts_of(2025, 1, 1)
        n = 150
        # «BIG»: همیشه بزرگ. «LATE»: تا ماهِ سوم کوچک، بعد منفجر. «DEAD»: بزرگ بود، بعد مرد.
        late = [1.0] * 70 + [1e6] * 80
        dead = [1e6] * 60 + [1.0] * 90
        self.hists = {
            "BIG": daily(start, n, 1e5),
            "LATE": daily(start, n, late),
            "DEAD": daily(start, n, dead),
        }
        self.snaps = universe.snapshots_from_histories(self.hists, top_n=2)

    def test_membership_follows_the_past_not_today(self):
        early = ts_of(2025, 2, 15)
        later = ts_of(2025, 5, 15)
        self.assertTrue(universe.in_universe(self.snaps, "DEAD", early))
        self.assertFalse(universe.in_universe(self.snaps, "LATE", early),
                         "برندهٔ بعدی نباید پیش از رشدش عضو باشد")
        self.assertTrue(universe.in_universe(self.snaps, "LATE", later))
        self.assertFalse(universe.in_universe(self.snaps, "DEAD", later))

    def test_snapshot_uses_only_volume_before_the_month_starts(self):
        # ماهِ مارس: LATE از روزِ ۷۰ (≈ ۱۱ مارس) بزرگ می‌شود؛ عکسِ اولِ مارس نباید آن را ببیند
        march = next(u for t, u in self.snaps if t == ts_of(2025, 3, 1))
        self.assertNotIn("LATE", march)

    def test_before_the_first_snapshot_membership_is_unknown(self):
        self.assertIsNone(universe.universe_at(self.snaps, ts_of(2024, 6, 1)))
        self.assertFalse(universe.in_universe(self.snaps, "BIG", ts_of(2024, 6, 1)))

    def test_summary_reports_churn(self):
        rep = universe.summary(self.snaps)
        self.assertGreater(rep["months"], 3)
        self.assertEqual(rep["symbols_ever"], 3)
        self.assertGreater(rep["avg_monthly_churn"], 0)


class RankWithinTests(unittest.TestCase):
    def test_small_population_is_neutral_instead_of_extreme(self):
        """کشِ نیمه‌خالی پس از ری‌استارت قبلاً رتبه‌های ۰ و ۱ می‌ساخت."""
        ranks = universe.rank_within({"A": 0.3, "B": -0.1, "C": 0.9})
        self.assertEqual(set(ranks.values()), {0.5})

    def test_full_population_is_ranked_zero_to_one(self):
        vals = {f"S{i}": float(i) for i in range(60)}
        ranks = universe.rank_within(vals)
        self.assertEqual(ranks["S0"], 0.0)
        self.assertEqual(ranks["S59"], 1.0)

    def test_ranking_is_restricted_to_the_universe(self):
        vals = {f"S{i}": float(i) for i in range(60)}
        vals["OUTSIDER"] = 1e9
        uni = frozenset(f"S{i}" for i in range(60))
        ranks = universe.rank_within(vals, universe=uni)
        self.assertNotIn("OUTSIDER", ranks)
        self.assertEqual(ranks["S59"], 1.0)


if __name__ == "__main__":
    unittest.main()
