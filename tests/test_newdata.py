# -*- coding: utf-8 -*-
"""فرضیه‌های دادهٔ تازه: هیچ‌کدام از آینده نمی‌خوانند (با دست‌کاریِ دادهٔ آینده آزموده می‌شود)."""
import json
import os
import sys
import unittest

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "bot"))
import _hermetic  # noqa: E402,F401  — پیش از هر ماژولِ ربات: داده به پوشهٔ موقت

import extdata  # noqa: E402
import newdata  # noqa: E402

DAY = newdata.DAY_MS
T0 = 1_483_228_800_000        # 2017-01-01
N = 1500


def _write(name, rows):
    os.makedirs(extdata.OUT, exist_ok=True)
    with open(extdata.path(name), "w", encoding="utf-8") as f:
        json.dump(rows, f)


def _fixture(seed=4, bump_from=None):
    rng = np.random.default_rng(seed)
    days = [T0 + i * DAY for i in range(N)]
    series = {
        "stablecoin_supply": np.cumsum(rng.normal(1, 3, N)) + 1000,
        "mvrv_btc": 1.5 + np.cumsum(rng.normal(0, 0.02, N)),
        "dvol_btc": 60 + np.cumsum(rng.normal(0, 1, N)),
        "spx": 3000 + np.cumsum(rng.normal(1, 20, N)),
        "dxy_broad": 110 + np.cumsum(rng.normal(0, 0.3, N)),
        "fear_greed": np.clip(50 + np.cumsum(rng.normal(0, 3, N)), 0, 100),
    }
    for name, v in series.items():
        v = v.copy()
        if bump_from is not None:
            v[bump_from:] *= 1.7
        _write(name, [[d, float(x)] for d, x in zip(days, v)])
    return np.asarray(days, np.int64)


class NoLookAheadTests(unittest.TestCase):
    def test_future_data_never_changes_past_positions(self):
        t = _fixture()
        base = {h: newdata.new_positions(h, "BTCUSDT", t) for h in newdata.NEW_HYPOTHESES}
        d = 1200
        _fixture(bump_from=d)                                    # همهٔ مقادیرِ روزِ d به بعد عوض شد
        for h in newdata.NEW_HYPOTHESES:
            after = newdata.new_positions(h, "BTCUSDT", t)
            # روزِ d فقط از d−1 (و DXY از d−8) می‌خواند، پس تا خودِ d باید دست‌نخورده بماند
            self.assertTrue(np.array_equal(np.nan_to_num(base[h][:d + 1], nan=-1),
                                           np.nan_to_num(after[:d + 1], nan=-1)), h)
            self.assertTrue(np.isfinite(base[h][1100:]).all(), h)

    def test_momentum_votes_match_the_research_rule_for_28_days(self):
        import timing
        rng = np.random.default_rng(2)
        c = 100 * np.exp(np.cumsum(rng.normal(0, 0.03, 500)))
        t = T0 + np.arange(500, dtype=np.int64) * DAY
        self.assertTrue(np.array_equal(newdata.momentum_vote(t, c, 28), timing.positions("H3_TSMOM28", t, c, None)))

    def test_exposure_costs_scale_with_the_change(self):
        expo = np.array([0, 1 / 3, 1, 0.0])
        self.assertAlmostEqual(newdata.exposure_returns(expo, np.zeros(4), cost=0.002).sum(), -0.002 * 2)


if __name__ == "__main__":
    unittest.main()
