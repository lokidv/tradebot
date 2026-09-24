# -*- coding: utf-8 -*-
"""هستهٔ v2.1 (core2_1): پارامترهای پیش‌ثبت، دورِ ثابت، برشِ ۲۰۲۲-۰۱، قاعدهٔ تصمیمِ «E > 0»، ویژگی‌های سنجاق‌شده،
اجرای کامل روی دادهٔ مصنوعی بی‌نگاه به پس از پنجره، و نگهبانِ holdout (سنجاق، دفترِ فقط-افزودنی، گیت)."""
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "bot"))
import _hermetic  # noqa: E402,F401  — پیش از هر ماژولِ ربات: داده به پوشهٔ موقت

import core2  # noqa: E402
import core2_1  # noqa: E402
import core_feats as cf  # noqa: E402
import decision  # noqa: E402

PREREG = os.path.join(ROOT, "bot", "data", "research", "prereg_core2_1.json")
H_MS = 3_600_000
D_MS = 86_400_000
T0 = 19_700 * D_MS
SYMS = ("BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "TRXUSDT")
HAS_GIT = shutil.which("git") is not None


def synth(n, step_ms=H_MS, seed=3, t0=T0):
    rng = np.random.RandomState(seed)
    drift = np.repeat(rng.choice([-1.0, 0.0, 1.0], n // 150 + 1), 150)[:n] * 0.0015
    c = 100 * np.exp(np.cumsum(drift + rng.normal(0, 0.006, n)))
    o = np.r_[c[0], c[:-1]] * (1 + rng.normal(0, 0.0005, n))
    wig = np.abs(rng.normal(0, 0.003, n)) * c
    v = rng.lognormal(10, 0.5, n)
    return {"t": t0 + np.arange(n, dtype=np.int64) * step_ms, "o": o, "h": np.maximum(o, c) + wig,
            "l": np.minimum(o, c) - wig, "c": c, "v": v, "qv": v * c, "n": np.floor(v / 10) + 1,
            "tbv": v * np.clip(rng.normal(0.5, 0.08, n), 0.05, 0.95)}


def load_cli():
    spec = importlib.util.spec_from_file_location("train_core2_1_cli", os.path.join(ROOT, "tools", "train_core2_1.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class PreregTests(unittest.TestCase):
    def test_constants_match_the_preregistration(self):
        pr = core2_1.check_prereg(PREREG)
        self.assertEqual(core2_1.GAP_BARS, 81)
        self.assertEqual(core2_1.window_ms("validation"), (core2_1._ms("2025-07-01"), core2_1._ms("2026-01-01")))
        self.assertEqual(core2_1.window_ms("holdout"), (core2_1._ms("2026-01-01"), core2_1._ms("2026-09-01")))
        self.assertEqual(core2_1.TRAIN_START_MS, core2_1._ms("2022-01-01"))
        self.assertIn("E_L > 0 and E_L > E_S", pr["decision_rule_fixed"])

    def test_check_prereg_rejects_a_changed_file(self):
        with open(PREREG, encoding="utf-8") as f:
            pr = json.load(f)
        pr["model"] = pr["model"].replace("lambda_l2 50", "lambda_l2 10")
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "p.json")
            with open(p, "w", encoding="utf-8") as f:
                json.dump(pr, f)
            with self.assertRaises(ValueError):
                core2_1.check_prereg(p)

    def test_lgb_params_per_timeframe(self):
        p = core2_1.lgb_params("5m", 1_000_000)
        self.assertEqual((p["objective"], p["learning_rate"], p["num_leaves"], p["max_depth"]),
                         ("regression", 0.02, 15, 5))
        self.assertNotIn("alpha", p)
        self.assertEqual(p["min_data_in_leaf"], 2500)
        self.assertEqual(core2_1.lgb_params("15m", 100_000)["min_data_in_leaf"], 500)
        self.assertEqual(core2_1.lgb_params("1h", 1)["min_data_in_leaf"], 200)
        self.assertEqual(core2_1.lgb_params("4h", 1)["min_data_in_leaf"], 100)
        d = core2_1.lgb_params("1d", 1)
        self.assertEqual((d["num_leaves"], d["min_data_in_leaf"], d["max_depth"]), (7, 60, 5))
        for q in (p, d):
            self.assertEqual((q["feature_fraction"], q["bagging_fraction"], q["bagging_freq"], q["lambda_l2"],
                              q["max_bin"], q["seed"], q["deterministic"]), (0.5, 0.7, 1, 50.0, 63, 42, True))
        self.assertEqual([core2_1.n_rounds(tf) for tf in core2_1.TFS], [300, 300, 300, 300, 150])

    def test_features_are_pinned_to_the_v2_file(self):
        self.assertEqual(core2_1.check_features(), core2_1.CORE_FEATS_SHA256_LF)
        with mock.patch.object(core2_1, "sha256_lf", return_value="0" * 64):
            with self.assertRaises(ValueError):
                core2_1.check_features()

    def test_hash_ignores_line_endings(self):
        self.assertEqual(core2_1.sha256_lf_bytes(b"a\r\nb\r\n"), core2_1.sha256_lf_bytes(b"a\nb\n"))
        self.assertNotEqual(core2_1.sha256_lf_bytes(b"a\nb\n"), core2_1.sha256_lf_bytes(b"a\nc\n"))
        with open(os.path.join(ROOT, "bot", "core_feats.py"), "rb") as f:
            raw = f.read().replace(b"\r\n", b"\n")
        self.assertEqual(core2_1.sha256_lf_bytes(raw.replace(b"\n", b"\r\n")), core2_1.CORE_FEATS_SHA256_LF)


class ModelTests(unittest.TestCase):
    def test_fit_pair_uses_a_fixed_number_of_rounds(self):
        rng = np.random.RandomState(1)
        for tf, n, rounds in (("1d", 900, 150), ("4h", 1200, 300)):
            X = rng.normal(size=(n, 4)).astype(np.float32)
            y = (0.3 * X[:, 0] + rng.normal(0, 0.5, n)).astype(np.float32)
            t = T0 + np.arange(n, dtype=np.int64) * cf.BAR_MS[tf]
            pred, meta = core2_1.fit_pair(X, y, -y, t, np.arange(n) % 3, tf, ["a", "b", "c", "d"], threads=2)
            EL, ES = pred(X[:10])
            self.assertEqual(EL.shape, (10,))
            self.assertEqual((meta["long"]["trees"], meta["short"]["trees"], meta["rounds"]), (rounds, rounds, rounds))
            self.assertNotIn("best_iter", meta["long"])

    def test_mean_objective_can_predict_positive_when_the_mean_is_positive(self):
        """v2 (huber) کنارِ میانهٔ منفی می‌ماند؛ l2 میانگین را دنبال می‌کند: میانگینِ مثبت با میانهٔ منفی."""
        rng = np.random.RandomState(2)
        n = 3000
        X = rng.normal(size=(n, 3)).astype(np.float32)
        win = rng.rand(n) < 0.40
        y = np.where(win, 1.8, -1.0).astype(np.float32)          # میانه −۱، میانگین +۰٫۱۲
        t = T0 + np.arange(n, dtype=np.int64) * D_MS
        pred, meta = core2_1.fit_pair(X, y, -y, t, np.arange(n) % 5, "1d", ["a", "b", "c"], threads=2)
        EL, _ = pred(X)
        self.assertLess(float(np.median(y)), 0)
        self.assertGreater(float(np.mean(EL)), 0)
        self.assertGreater(meta["long"]["train_pred_share_pos"], 0.5)


class WalkForwardTests(unittest.TestCase):
    TF = "1h"

    def _ds(self, n=4000, t0=T0):
        t1 = t0 + np.arange(n, dtype=np.int64) * H_MS
        t = np.r_[t1, t1]
        y = np.random.RandomState(0).normal(-0.1, 1, 2 * n).astype(np.float32)
        return {"tf": self.TF, "t": t, "sym": np.r_[np.zeros(n), np.ones(n)].astype(np.int8),
                "X": np.zeros((2 * n, 3), np.float32), "yL": y, "yS": y.copy(), "exit_close": t + 42 * H_MS,
                "symbols": ["A", "B"], "features": ["f0", "f1", "f2"]}

    def test_training_rows_start_in_2022(self):
        start = core2_1.TRAIN_START_MS
        t = start + np.arange(-500, 1500, dtype=np.int64) * H_MS
        y = np.zeros(len(t), np.float32)
        m = core2_1.train_mask(t, t + 42 * H_MS, y, y, "1h", int(t[-1]) + H_MS)
        self.assertTrue(m.any())
        self.assertTrue((t[m] >= start).all())
        self.assertTrue(m[500])                                   # اولین کندلِ ۲۰۲۲-۰۱-۰۱
        self.assertFalse(m[499])
        base = core2.train_mask(t, t + 42 * H_MS, y, y, "1h", int(t[-1]) + H_MS)
        self.assertTrue(base[:500].any())                         # بدونِ برشِ پایین همان‌ها آموزش می‌دیدند

    def test_monthly_model_only_predicts_its_own_month(self):
        t0 = core2_1.TRAIN_START_MS - 200 * H_MS
        ds = self._ds(n=4000, t0=t0)
        months = core2.month_starts(int(t0 + 3000 * H_MS), int(t0 + 3900 * H_MS))
        self.assertGreaterEqual(len(months), 2)
        seen = []

        def fit_fn(X, yL, yS, t, sym, tf, names, threads):
            k = len(seen)
            seen.append((int(t.min()), int(t.max()), len(t)))
            return (lambda Xte: (np.full(len(Xte), k, np.float32), np.full(len(Xte), -k, np.float32))), {"rounds": 1}

        oos, meta = core2_1.walk_forward(ds, months, fit_fn=fit_fn, window_end=months[-1][1])
        self.assertEqual(len(seen), len(months))
        for k, (m0, m1, ym) in enumerate(months):
            self.assertGreaterEqual(seen[k][0], core2_1.TRAIN_START_MS)
            self.assertLessEqual(seen[k][1], m0 - 81 * H_MS)
            rows = oos["month"] == ym
            self.assertTrue(((oos["t"][rows] >= m0) & (oos["t"][rows] < m1)).all())
            self.assertTrue((oos["E_L"][rows] == k).all())
            self.assertEqual(int(rows.sum()), int(((ds["t"] >= m0) & (ds["t"] < m1)).sum()))
            self.assertLessEqual(meta[k]["train_exit_close_max"], core2._date(m0))
        self.assertLess(int(oos["t"].max()), months[-1][1])

    def test_window_end_refuses_months_past_it(self):
        ds = self._ds()
        months = core2.month_starts(int(T0 + 3000 * H_MS), int(T0 + 3900 * H_MS))
        with self.assertRaises(ValueError):
            core2_1.walk_forward(ds, months, fit_fn=lambda *a: None, window_end=months[0][1])


class DecisionRuleTests(unittest.TestCase):
    def test_fixed_rule(self):
        EL = np.array([0.01, 0.00, 0.20, -0.1, -0.2, 0.30, np.nan, 0.20, 0.001, -0.01])
        ES = np.array([-0.5, -0.5, 0.10, -0.2, 0.01, 0.31, 0.30, 0.20, 0.000, -0.02])
        self.assertEqual(core2_1.decide(EL, ES).tolist(), [1, 0, 1, 0, -1, -1, 0, 0, 1, 0])

    def test_sensitivity_margins(self):
        self.assertEqual(core2_1.decide([0.06, 0.06, 0.2], [0.0, 0.02, 0.09], 0.05).tolist(), [1, 0, 1])
        self.assertEqual(core2_1.decide([0.2, 0.2], [0.11, 0.09], 0.10).tolist(), [0, 1])
        self.assertEqual(core2_1.decide([-0.3], [0.11], 0.10).tolist(), [-1])


class EvaluateTests(unittest.TestCase):
    def test_rule_and_v21_on_the_same_bars(self):
        syms = ("A", "B")
        fut = {s: synth(3000, seed=k + 9) for k, s in enumerate(syms)}
        t = fut["A"]["t"]
        w0, w1 = int(t[2400]), int(t[2900])
        rng = np.random.RandomState(0)
        ot = np.r_[t[2400:2900], t[2400:2900]]
        oos = {"sym": np.r_[np.zeros(500), np.ones(500)].astype(np.int8), "t": ot,
               "E_L": rng.normal(0, 0.1, 1000).astype(np.float32), "E_S": rng.normal(0, 0.1, 1000).astype(np.float32)}
        v21, rule, same = core2_1.evaluate("1h", oos, w0, w1, fut, symbols=syms)
        self.assertTrue(same)
        self.assertEqual(sorted(v21), [0.0, 0.05, 0.10])
        for s in syms:
            self.assertTrue(v21[0.0][s])
            self.assertTrue(all(w0 <= x["t"] < w1 for x in v21[0.0][s] + rule[s]))
        n = {m: sum(len(v21[m][s]) for s in syms) for m in v21}
        self.assertGreaterEqual(n[0.0], n[0.10])


class AdoptTests(unittest.TestCase):
    def _res(self, mean=0.2, n=80, rule=0.0, lo=0.05, coins=(0.3, 0.2, 0.1, -0.1, 0.1)):
        tot = sum(coins)
        return {"v21": {"mean": mean, "n": n, "coins_positive": sum(c > 0 for c in coins),
                        "max_coin_share": max(coins) / tot},
                "rule": {"mean": rule}, "bootstrap": {"diff_ci95": [lo, 0.4]}}

    def test_adopt_rule(self):
        self.assertTrue(core2_1.adopt_checks("1h", self._res(), True)["adopted"])
        self.assertTrue(core2_1.adopt_checks("1h", self._res(n=60), True)["adopted"])       # n ≥ 60
        self.assertFalse(core2_1.adopt_checks("1h", self._res(n=59), True)["adopted"])
        self.assertTrue(core2_1.adopt_checks("1d", self._res(n=20), True)["adopted"])       # 1d: n ≥ 20
        self.assertFalse(core2_1.adopt_checks("1d", self._res(n=19), True)["adopted"])
        self.assertFalse(core2_1.adopt_checks("1h", self._res(), False)["adopted"])         # Holm
        self.assertFalse(core2_1.adopt_checks("1h", self._res(rule=0.16), True)["adopted"])  # +0.05R
        self.assertFalse(core2_1.adopt_checks("1h", self._res(lo=-0.01), True)["adopted"])
        self.assertFalse(core2_1.adopt_checks("1h", self._res(coins=(0.3, 0.2, -0.1, -0.1, -0.1)), True)["adopted"])
        self.assertFalse(core2_1.adopt_checks("1h", self._res(coins=(0.9, 0.1, 0.1, 0.1, -0.1)), True)["adopted"])

    def test_holdout_verdict(self):
        ok = lambda tf, n, m: core2_1.holdout_verdict(tf, {"v21": {"n": n, "mean": m}})["passed"]  # noqa: E731
        self.assertTrue(ok("4h", 30, 0.01))
        self.assertFalse(ok("4h", 29, 0.5))
        self.assertFalse(ok("4h", 100, 0.0))
        self.assertTrue(ok("1d", 10, 0.01))
        self.assertFalse(ok("1d", 9, 0.5))
        self.assertFalse(ok("1d", 10, None))

    def test_assemble_holm_over_five_and_completeness(self):
        res = {"v21": {"n": 80, "mean": 0.2, "coins_positive": 4, "max_coin_share": 0.3},
               "rule": {"mean": 0.0}, "bootstrap": {"p_one_sided": 0.001, "diff_ci95": [0.05, 0.4]},
               "code_sha256_lf": core2_1.code_hashes()}
        rep = core2_1.assemble({"4h": res})
        self.assertFalse(rep["complete"])
        self.assertEqual(rep["adopted"], ["4h"])                          # 0.001 × 5 = 0.005 ≤ 0.10
        self.assertTrue(rep["timeframes"]["1d"]["missing"])
        full = core2_1.assemble({tf: res for tf in core2_1.TFS})
        self.assertTrue(full["complete"])
        stale = dict(res, code_sha256_lf={"bot/core2_1.py": "x"})
        self.assertFalse(core2_1.assemble(dict({tf: res for tf in core2_1.TFS}, **{"1d": stale}))["complete"])


class RunWindowTests(unittest.TestCase):
    """اجرای کامل روی فایل‌های مصنوعیِ 1d در micro/: هیچ کندلی پس از پایانِ پنجره (جز دُمِ برچسب) خوانده نمی‌شود."""
    TF = "1d"

    @classmethod
    def setUpClass(cls):
        cls.micro = tempfile.mkdtemp(prefix="core21-micro-")      # پوشهٔ جدا: micro/ مشترکِ تست‌ها ساخته نمی‌شود
        cls.patch = mock.patch.object(core2, "MICRO_DIR", cls.micro)
        cls.patch.start()
        t0 = core2_1._ms("2020-06-01")
        n = (core2_1._ms("2026-09-30") - t0) // D_MS
        for k, s in enumerate(SYMS):
            b = synth(int(n), D_MS, seed=40 + k, t0=t0)
            np.savez_compressed(os.path.join(cls.micro, f"spot_{s}_1d.npz"), **b)
            f = synth(int(n), D_MS, seed=40 + k, t0=t0)
            keep = f["t"] >= core2_1._ms("2021-06-01")
            np.savez_compressed(os.path.join(cls.micro, f"um_{s}_1d.npz"), **{kk: v[keep] for kk, v in f.items()})
            ft = np.arange(core2_1._ms("2021-10-01"), core2_1._ms("2026-09-30"), 8 * H_MS, dtype=np.int64)
            np.savez_compressed(os.path.join(cls.micro, f"kfund_{s}.npz"), t=ft,
                                rate=np.random.RandomState(k).normal(1e-4, 1e-4, len(ft)),
                                interval_h=np.full(len(ft), 8.0))

    @classmethod
    def tearDownClass(cls):
        cls.patch.stop()
        shutil.rmtree(cls.micro, True)

    def test_validation_run_end_to_end_without_looking_past_the_window(self):
        """ممیزی DATA-5: نه اسپات و نه فیوچرز (دیگر دُمِ ۴۵ کندلی نیست)؛ فاندینگِ کوکوین تا خودِ w1 (model-5)."""
        w0, w1 = core2_1.window_ms("validation")
        self.assertEqual(core2_1.LABEL_TAIL_BARS, 0)
        seen = []
        real_load, real_kf = core2.load_bars, core2.load_kfund

        def spy(prefix, sym, tf, t_end=None, micro_dir=None):
            seen.append((prefix, t_end))
            return real_load(prefix, sym, tf, t_end, micro_dir)

        def spy_kf(sym, t_end=None, micro_dir=None):
            seen.append(("kfund", t_end))
            return real_kf(sym, t_end, micro_dir)

        with mock.patch.object(core2, "load_bars", side_effect=spy), \
                mock.patch.object(core2, "load_kfund", side_effect=spy_kf):
            res, oos = core2_1.run_window(self.TF, "validation", threads=2)
        self.assertTrue(seen)
        self.assertIn("um", {p for p, _ in seen})
        for prefix, t_end in seen:
            self.assertIsNotNone(t_end)
            self.assertLessEqual(t_end, w1 + 1 if prefix == "kfund" else w1)
        self.assertEqual(res["label_tail_bars"], 0)
        self.assertGreater(res["oos_rows_label_unresolved"], 0)              # براکت‌های بازِ پایان ⇒ NaN
        self.assertTrue(np.isnan(oos["yL"][oos["t"] == w1 - D_MS]).all())
        self.assertIsNotNone(res["ic"]["gross_label"])
        self.assertIsNotNone(res["ic"]["partial_given_cost"])
        self.assertIn("gL", oos)
        self.assertEqual(len(res["months"]), 6)
        self.assertTrue(all(m["rounds"] == 150 for m in res["months"]))
        self.assertTrue(all(m["train_t_min"] >= "2022-01-01" for m in res["months"]))
        self.assertTrue(((oos["t"] >= w0) & (oos["t"] < w1)).all())
        self.assertEqual(res["oos_rows"], 5 * 184)                            # ۱۸۴ روز × ۵ ارز
        self.assertTrue(res["replay_equal_to_decision_replay"])
        self.assertEqual(res["core_feats_sha256_lf"], core2_1.CORE_FEATS_SHA256_LF)
        self.assertEqual(res["code_sha256_lf"], core2_1.code_hashes())
        self.assertEqual(res["rounds"], 150)
        self.assertIn("share_pos", res["predictions"]["E_L"])
        self.assertEqual(set(res["sensitivity"]), {"margin_0.00", "margin_0.05", "margin_0.10"})
        json.dumps(core2.to_json(res))

    def test_coverage_refuses_short_data(self):
        with self.assertRaises(RuntimeError):
            core2_1._coverage(self.TF, core2_1._ms("2027-01-01"), core2_1._ms("2027-01-01"))


class ReportFilesTests(unittest.TestCase):
    def test_latest_report_ignores_pin_and_erratum(self):
        """سنجاق و اِراتا همان پیشوند را دارند و در مرتب‌سازی پس از گزارشِ زمان‌دار می‌آیند."""
        with tempfile.TemporaryDirectory() as d:
            for name in ("core2_1_validation_20260924_135542.json", "core2_1_validation_pin.json",
                         "core2_1_validation_erratum.json"):
                with open(os.path.join(d, name), "w", encoding="utf-8") as f:
                    json.dump({"name": name}, f)
            rep, path = core2_1.latest_validation_report(d)
            self.assertEqual(os.path.basename(path), "core2_1_validation_20260924_135542.json")
        with tempfile.TemporaryDirectory() as d:
            with open(os.path.join(d, "core2_1_validation_erratum.json"), "w", encoding="utf-8") as f:
                f.write("{}")
            self.assertEqual(core2_1.latest_validation_report(d), (None, None))


def _git(d, *args):
    subprocess.run(["git", "-C", d, "-c", "user.name=t", "-c", "user.email=t@t", "-c", "commit.gpgsign=false",
                    *args], check=True, capture_output=True)


@unittest.skipUnless(HAS_GIT, "git نصب نیست")
class HoldoutGuardTests(unittest.TestCase):
    """مخزنِ گیتِ موقتی با research/: سنجاق ⇒ commit ⇒ holdout_start ⇒ commit ⇒ مجاز؛ هر انحراف ⇒ رد."""

    def setUp(self):
        self.d = tempfile.mkdtemp(prefix="core21-git-")
        _git(self.d, "init", "-q")
        self.rd = os.path.join(self.d, "research")
        os.makedirs(self.rd)
        self.pin = os.path.join(self.rd, "core2_1_validation_pin.json")
        self.log = os.path.join(self.rd, "core2_holdout_log.jsonl")
        self.out = os.path.join(self.d, "out")
        self.report = os.path.join(self.rd, "core2_1_validation_20990101_000000.json")
        rep = {"window": "validation", "complete": True, "adopted": ["4h"], "code_sha256_lf": core2_1.code_hashes(),
               "timeframes": {tf: {"adopted": tf == "4h"} for tf in core2_1.TFS}}
        with open(self.report, "w", encoding="utf-8") as f:
            json.dump(rep, f)
        self.patches = [mock.patch.object(core2_1, "PREREG_PATH", PREREG)]
        for p in self.patches:
            p.start()

    def tearDown(self):
        for p in self.patches:
            p.stop()
        shutil.rmtree(self.d, True)

    def guard(self, tf="4h"):
        return core2_1.holdout_guard(tf, self.pin, self.log, self.rd, self.out)

    def commit(self, msg):
        _git(self.d, "add", "-A")
        _git(self.d, "commit", "-q", "-m", msg)

    def _ready(self):
        core2_1.write_pin(self.report, self.pin)
        self.commit("validation + pin")
        core2_1.start_record("4h", self.pin, self.log, self.rd)
        self.commit("holdout start")

    def test_full_sequence(self):
        self.assertFalse(self.guard()[0])                                   # سنجاقی نیست
        pin = core2_1.write_pin(self.report, self.pin)
        self.assertEqual(pin["adopted"], ["4h"])
        with self.assertRaises(FileExistsError):
            core2_1.write_pin(self.report, self.pin)                         # فقط یک‌بار
        ok, why = self.guard()
        self.assertFalse(ok)
        self.assertIn("commit", why)                                        # سنجاق commit نشده
        with self.assertRaises(PermissionError):
            core2_1.start_record("4h", self.pin, self.log, self.rd)          # سنجاق commit نشده
        self.commit("validation + pin")
        self.assertFalse(self.guard()[0])                                   # دفتری نیست
        with self.assertRaises(PermissionError):
            core2_1.start_record("1h", self.pin, self.log, self.rd)          # پذیرفته نشده
        core2_1.start_record("4h", self.pin, self.log, self.rd)
        with self.assertRaises(PermissionError):
            core2_1.start_record("4h", self.pin, self.log, self.rd)          # یک رکورد برای هر تایم‌فریم
        self.assertFalse(self.guard()[0])                                   # رکورد commit نشده
        self.commit("holdout start")
        self.assertEqual(self.guard(), (True, ""))
        self.assertFalse(self.guard("1h")[0])                               # پذیرفته نشده
        core2_1.append_log({"event": "holdout_attempt", "tf": "4h"}, self.log)
        self.assertTrue(self.guard()[0])                                    # افزودن (commit‌نشده) مجاز است
        core2_1.append_log({"event": "holdout_done", "tf": "4h"}, self.log)
        ok, why = self.guard()
        self.assertFalse(ok)
        self.assertIn("یک‌بار", why)

    def test_changed_report_refuses(self):
        self._ready()
        with open(self.report, "a", encoding="utf-8") as f:
            f.write(" ")
        self.assertFalse(self.guard()[0])

    def test_report_committed_but_different_from_pin_refuses(self):
        self._ready()
        with open(self.report, "a", encoding="utf-8") as f:
            f.write(" ")
        self.commit("edit report")
        ok, why = self.guard()
        self.assertFalse(ok)
        self.assertIn("سنجاق", why)

    def test_rewritten_log_refuses(self):
        self._ready()
        with open(self.log, "w", encoding="utf-8") as f:
            f.write('{"event": "holdout_start", "tf": "1h"}\n')
        ok, why = self.guard()
        self.assertFalse(ok)
        self.assertIn("فقط-افزودنی", why)

    def test_crlf_checkout_is_accepted(self):
        self._ready()
        for p in (self.pin, self.log, self.report):
            with open(p, "rb") as f:
                b = f.read()
            with open(p, "wb") as f:
                f.write(b.replace(b"\n", b"\r\n"))
        self.assertEqual(self.guard(), (True, ""))

    def test_code_or_prereg_change_refuses(self):
        self._ready()
        cur = core2_1.code_hashes()
        with mock.patch.object(core2_1, "code_hashes", return_value=dict(cur, **{"bot/engine.py": "0" * 64})):
            ok, why = self.guard()
            self.assertFalse(ok)
            self.assertIn("bot/engine.py", why)
        other = os.path.join(self.d, "p.json")
        shutil.copy(PREREG, other)
        with open(other, "a", encoding="utf-8") as f:
            f.write("\n")
        with mock.patch.object(core2_1, "PREREG_PATH", other):
            self.assertFalse(self.guard()[0])

    def test_existing_result_refuses(self):
        self._ready()
        os.makedirs(self.out)
        with open(core2_1.result_path("4h", "holdout", self.out), "w") as f:
            f.write("{}")
        self.assertFalse(self.guard()[0])

    def test_start_record_for_another_pin_refuses(self):
        self._ready()
        with open(self.pin, encoding="utf-8") as f:
            pin = json.load(f)
        pin["created_utc"] = "2099-01-01 00:00:00"
        with open(self.pin, "w", encoding="utf-8", newline="\n") as f:
            json.dump(pin, f)
        self.commit("re-pin")
        ok, why = self.guard()
        self.assertFalse(ok)
        self.assertIn("holdout_start", why)

    def test_pin_without_a_report_name_refuses(self):
        self._ready()
        with open(self.pin, encoding="utf-8") as f:
            pin = json.load(f)
        pin["report"] = ""
        with open(self.pin, "w", encoding="utf-8", newline="\n") as f:
            json.dump(pin, f)
        self.commit("broken pin")
        self.assertFalse(self.guard()[0])

    def test_incomplete_report_cannot_be_pinned(self):
        with open(self.report, encoding="utf-8") as f:
            rep = json.load(f)
        rep["complete"] = False
        with open(self.report, "w", encoding="utf-8") as f:
            json.dump(rep, f)
        with self.assertRaises(ValueError):
            core2_1.write_pin(self.report, self.pin)
        self.assertFalse(os.path.exists(self.pin))


class HoldoutRefusalTests(unittest.TestCase):
    def test_run_window_refuses_holdout_before_reading_data(self):
        with mock.patch.object(core2, "build_dataset", side_effect=AssertionError("نباید داده بخواند")), \
                mock.patch.object(core2, "load_bars", side_effect=AssertionError("نباید داده بخواند")):
            with self.assertRaises(PermissionError):
                core2_1.run_window("4h", "holdout")
            with self.assertRaises(PermissionError):
                core2_1.run_window("4h", "holdout", allow_holdout=True)      # سنجاقی نیست

    def test_cli_refusals(self):
        cli = load_cli()
        with mock.patch.object(core2_1, "PREREG_PATH", PREREG), \
                mock.patch.object(core2_1, "run_window", side_effect=AssertionError("نباید اجرا شود")), \
                mock.patch("sys.stderr"):
            self.assertEqual(cli.main(["--tf", "4h", "--window", "holdout"]), 2)
            self.assertEqual(cli.main(["--tf", "4h", "--holdout"]), 2)
            self.assertEqual(cli.main(["--window", "holdout", "--holdout"]), 2)          # all
            self.assertEqual(cli.main(["--tf", "4h", "--window", "holdout", "--holdout"]), 2)  # سنجاقی نیست
            self.assertEqual(cli.main(["--pin"]), 2)                                    # گزارشی نیست
            self.assertEqual(cli.main(["--holdout-start"]), 2)                          # all
            self.assertEqual(cli.main(["--holdout-start", "--tf", "4h"]), 2)            # سنجاقی نیست
        self.assertFalse(os.path.exists(core2_1.HOLDOUT_LOG))

    def test_core2_1_never_touches_gates_or_tradeable(self):
        for rel in ("bot/core2_1.py", "tools/train_core2_1.py"):
            with open(os.path.join(ROOT, *rel.split("/")), encoding="utf-8") as f:
                src = f.read()
            self.assertNotIn("import gates", src)
            self.assertNotIn("gates.json\"", src)
            self.assertNotIn("tradeable =", src)
            self.assertNotIn("tradeable\"] =", src)


if __name__ == "__main__":
    unittest.main()
