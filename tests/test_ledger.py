# -*- coding: utf-8 -*-
"""دفترِ کاملِ معامله‌های بسته — افت و ضریبِ سود روی کلِ تاریخچه، نه ۲۰۰تای آخر."""
import json
import os
import sys
import tempfile
import unittest


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "bot"))

import paper  # noqa: E402


class LedgerTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self._old = paper.PATH
        paper.PATH = os.path.join(self.tmp.name, "positions.json")

    def tearDown(self):
        paper.PATH = self._old
        self.tmp.cleanup()

    def _trade(self, pnl_dir):
        pos = paper.open_position("BTCUSDT", "1h", "long", 100, 98, 103.6, 1_000, 2_400,
                                  cost_pct=0.0)
        price = 103.6 if pnl_dir > 0 else 98.0
        return paper.close_manual(pos["id"], price)

    def test_ledger_lives_next_to_positions_file(self):
        self.assertEqual(os.path.dirname(paper._ledger_path()), self.tmp.name)

    def test_summary_survives_the_200_row_cap(self):
        """۲۰۱ برد و بعد ۲۵۰ باخت: positions.json فقط ۲۰۰ باختِ آخر را دارد."""
        for _ in range(201):
            self._trade(+1)
        for _ in range(250):
            self._trade(-1)
        paper.refresh({})                          # همان برشِ ۲۰۰تایی را اعمال می‌کند
        db = paper.list_positions()
        self.assertEqual(len(db["closed"]), 200)
        w = paper.wallet_summary()
        self.assertEqual(w["trades"], 451)
        self.assertEqual(w["wins"], 201)
        # افتِ واقعی از اوجِ پس از ۲۰۱ برد تا انتها است، نه فقط ۲۰۰ باختِ آخر
        self.assertAlmostEqual(w["max_drawdown"], 250 * 20.0, delta=1.0)

    def test_first_new_close_seeds_existing_history(self):
        legacy = [{"id": f"old{i}", "pnl_usdt": -10.0, "closed_at": f"2026-07-0{i+1}T00:00:00+00:00"}
                  for i in range(5)]
        with open(paper.PATH, "w", encoding="utf-8") as f:
            json.dump({"open": [], "closed": list(reversed(legacy)),
                       "wallet": {"start": 10_000, "reset_at": "2026-07-01T00:00:00+00:00"}}, f)
        self.assertEqual(paper.wallet_summary()["trades"], 5)   # هنوز دفتری نیست ⇒ از positions
        self._trade(+1)
        self.assertEqual(paper.wallet_summary()["trades"], 6, "تاریخچهٔ قبلی نباید ناپدید شود")

    def test_testnet_pnl_correction_replaces_the_original_row(self):
        pos = paper.open_position("BTCUSDT", "1h", "long", 100, 98, 103.6, 1_000, 2_400,
                                  cost_pct=0.0)
        paper.close_with(pos["id"], 101.0, "test", pnl_usdt=-3.0)
        w = paper.wallet_summary()
        self.assertEqual(w["trades"], 1)
        self.assertAlmostEqual(w["realized"], -3.0)

    def test_reset_without_history_starts_fresh(self):
        self._trade(-1)
        paper.reset_wallet(10_000, keep_history=False)
        self.assertEqual(paper.wallet_summary()["trades"], 0)

    def test_reset_with_history_keeps_it(self):
        self._trade(-1)
        self._trade(+1)
        paper.reset_wallet(10_000, keep_history=True)
        self.assertEqual(paper.wallet_summary()["trades"], 2)


if __name__ == "__main__":
    unittest.main()
