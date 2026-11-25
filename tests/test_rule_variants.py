# -*- coding: utf-8 -*-
"""گونه‌های پیش‌ثبت‌شدهٔ قاعده (prereg_rule_variants_forward.json): هم‌ارزی با قاعده، ثبتِ جدا و فقط‌افزودنی،
داوری با همان مسیرِ دفترِ اصلی، و داوریِ یک‌بارهٔ بوت‌استرپِ بلوک‌هفتگی.

هم‌ارزی: تابعِ گونه **با** لایهٔ ستاپ باید عیناً جهتِ قاعده (engine.analyze و decision.side_series) شود —
روی دادهٔ ساختگی و، اگر آرشیوِ محلی باشد، روی آخرین کندل‌های پیش از ۲۰۲۶ِ اسپاتِ 1h/4h دو ارز.
فقط جهت‌ها مقایسه می‌شوند؛ هیچ آمارِ عملکردی روی دادهٔ واقعی حساب نمی‌شود.
"""
import json
import os
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "bot"))
import _hermetic  # noqa: E402,F401  — پیش از هر ماژولِ ربات: داده به پوشهٔ موقت

import decision  # noqa: E402
import engine  # noqa: E402
import gates  # noqa: E402

H_MS = 3_600_000
D_MS = 24 * H_MS
T0 = 472_222 * H_MS            # مرزِ ساعتِ UTC
END_2026 = 1_767_225_600_000   # فقط کندل‌های پیش از ۲۰۲۶ (holdout دست نمی‌خورد)
MICRO = os.environ.get("TRADERBOT_TEST_MICRO") or os.path.join(ROOT, "bot", "data", "hist_research", "micro")
PREREG = os.path.join(ROOT, "bot", "data", "research", "prereg_rule_variants_forward.json")
SIDE = {"long": 1, "short": -1}


def synth(n=1300, seed=11, step_ms=H_MS, drift_scale=0.0015):
    rng = np.random.RandomState(seed)
    drift = np.repeat(rng.choice([-1.0, 0.0, 1.0], n // 150 + 1), 150)[:n] * drift_scale
    c = 100 * np.exp(np.cumsum(drift + rng.normal(0, 0.006, n)))
    o = np.r_[c[0], c[:-1]] * (1 + rng.normal(0, 0.0005, n))
    wig = np.abs(rng.normal(0, 0.003, n)) * c
    return {"t": T0 + np.arange(n, dtype=np.int64) * step_ms,
            "o": o, "h": np.maximum(o, c) + wig, "l": np.minimum(o, c) - wig, "c": c,
            "v": rng.lognormal(10, 0.5, n)}


def _lists(kl):
    return {k: np.asarray(v).tolist() for k, v in kl.items()}


def zrule_reference(w):
    """لایهٔ z/رأی مستقل از engine.trade_suggestion: votes_at + kNN علّیِ پنجره (knn_series) + آستانه‌ها."""
    a = {k: np.asarray(v, float) for k, v in w.items() if k != "t"}
    cs = engine.component_series(a["o"], a["h"], a["l"], a["c"], a["v"])
    n = len(a["c"])
    b, s = engine.votes_at(cs, n - 1)
    ml = decision.knn_series(a["h"], a["l"], a["c"], cs, idx=[n - 1], window=decision.LIVE_BARS)[0]
    bull, bear = b + (ml >= decision.AI_VOTE), s + (ml <= -decision.AI_VOTE)
    z = float(cs["z"][-1])
    if z >= decision.Z_ENTRY and bull >= 3 and bear <= 1:
        return "long", float(cs["rsi"][-1])
    if z <= -decision.Z_ENTRY and bear >= 3 and bull <= 1:
        return "short", float(cs["rsi"][-1])
    return None, float(cs["rsi"][-1])


class _Gates(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self._old = gates.GATES_PATH
        gates.GATES_PATH = os.path.join(self.tmp.name, "gates.json")
        with open(gates.GATES_PATH, "w", encoding="utf-8") as f:
            json.dump(dict(gates.DEFAULTS), f)
        gates.load_gates(force=True)

    def tearDown(self):
        gates.GATES_PATH = self._old
        gates._cache.update(mtime=None, data=None)
        self.tmp.cleanup()

    def check_window(self, w, tf, sym="BTCUSDT"):
        """یک پنجرهٔ ۴۲۰ کندلی: تابعِ گونه با لایهٔ ستاپ = قاعده؛ بی‌آن = z/رأیِ مستقل. خروجی: (قاعده، z_only، ستاپ؟)."""
        a = engine.analyze(_lists(w), tf)
        rt = a["rule_trade"]
        rule = a["trade"].get("side")
        side_s, setup_s, _ = decision.side_series(w, decision.LIVE_BARS - 1)
        self.assertEqual(SIDE.get(rule, 0), int(side_s[-1]))                   # قاعده = بازپخش
        for v in decision.VARIANTS:
            with_setup = decision.variant_side(v, "1h", rt["setup_side"], rt["zrule_side"], rt["rsi14"],
                                               setup_layer=True)
            self.assertEqual(with_setup, rule, f"{v} با لایهٔ ستاپ باید خودِ قاعده باشد")
        zref, rsi_ref = zrule_reference(w)
        self.assertEqual(rt["zrule_side"], zref)
        self.assertEqual(rt["rsi14"], rsi_ref)
        self.assertEqual(rt["rsi14"], float(engine.rsi(np.asarray(w["c"], float), 14)[-1]))
        v1 = decision.variant_side("z_only", tf, rt["setup_side"], rt["zrule_side"], rt["rsi14"])
        self.assertEqual(v1, zref)
        v2 = decision.variant_side("d1_no_short_rsi30", tf, rt["setup_side"], rt["zrule_side"], rt["rsi14"])
        vetoed = tf == "1d" and rule == "short" and rt["rsi14"] < 30
        self.assertEqual(v2, None if vetoed else rule)
        d = decision.from_analysis(dict(a, symbol=sym))
        self.assertEqual(d["variants"], {"z_only": v1 or "wait", "d1_no_short_rsi30": v2 or "wait"})
        self.assertEqual(d["action"], rule or "wait")
        return rule, v1, rt["setup_side"] is not None, vetoed


class VariantParityTests(_Gates):
    def test_setup_layer_restores_the_rule_on_synthetic_data(self):
        seen = {"setup": 0, "differs": 0, "trade": 0}
        for seed, tf in ((11, "1h"), (5, "4h"), (23, "1d")):
            kl = synth(1300, seed=seed)
            for end in range(430, 1300, 13):
                w = {k: v[end - decision.LIVE_BARS:end] for k, v in kl.items()}
                rule, v1, had_setup, _ = self.check_window(w, tf)
                seen["setup"] += had_setup
                seen["differs"] += rule != v1
                seen["trade"] += rule is not None
        self.assertGreater(seen["setup"], 10)          # آزمون بی‌محتوا نیست: ستاپ رخ داد
        self.assertGreater(seen["differs"], 5)         # و گونهٔ اول واقعاً جای دیگری فرق کرد
        self.assertGreater(seen["trade"], 20)

    def test_d1_short_veto_below_rsi30(self):
        f = decision.variant_side
        self.assertIsNone(f("d1_no_short_rsi30", "1d", "short", None, 29.99))
        self.assertIsNone(f("d1_no_short_rsi30", "1d", None, "short", 12.0))
        self.assertEqual(f("d1_no_short_rsi30", "1d", "short", None, 30.0), "short")   # «زیرِ ۳۰»، نه ≤
        self.assertEqual(f("d1_no_short_rsi30", "4h", "short", None, 10.0), "short")   # فقط روزانه
        self.assertEqual(f("d1_no_short_rsi30", "1d", "long", "short", 10.0), "long")  # لانگ آزاد است
        self.assertEqual(f("d1_no_short_rsi30", "1d", None, "short", None), "short")   # RSI نامعلوم ⇒ قاعده
        self.assertEqual(f("z_only", "1d", "long", "short", 50.0), "short")            # ستاپ حذف، z/رأی
        self.assertIsNone(f("z_only", "1d", "long", None, 50.0))
        self.assertEqual(f("z_only", "1d", "long", None, 50.0, setup_layer=True), "long")
        with self.assertRaises(ValueError):
            f("other", "1d", None, None, None)

    def test_a_steep_daily_selloff_exercises_the_veto(self):
        kl = synth(1300, seed=4, step_ms=D_MS, drift_scale=0.004)
        vetoed = 0
        for end in range(430, 1300, 7):
            w = {k: v[end - decision.LIVE_BARS:end] for k, v in kl.items()}
            vetoed += self.check_window(w, "1d")[3]
        self.assertGreater(vetoed, 0)


@unittest.skipUnless(all(os.path.exists(os.path.join(MICRO, f"spot_{s}_{tf}.npz"))
                         for s in ("BTCUSDT", "ETHUSDT") for tf in ("1h", "4h")),
                     "آرشیوِ محلیِ اسپات نیست (TRADERBOT_TEST_MICRO)")
class VariantParityRealHistoryTests(_Gates):
    """آخرین کندل‌های پیش از ۲۰۲۶ِ اسپاتِ 1h/4h برای BTC و ETH — فقط جهت، هیچ آمارِ عملکرد."""

    def test_setup_layer_restores_the_rule_on_real_spot_slices(self):
        setups = trades = 0
        for sym in ("BTCUSDT", "ETHUSDT"):
            for tf in ("1h", "4h"):
                with np.load(os.path.join(MICRO, f"spot_{sym}_{tf}.npz")) as z:
                    t = np.asarray(z["t"], dtype=np.int64)
                    keep = t < END_2026
                    kl = {k: (t[keep] if k == "t" else np.asarray(z[k], float)[keep]) for k in "tohlcv"}
                n = len(kl["t"])
                self.assertLess(int(kl["t"][-1]), END_2026)
                for end in range(n - 60 * 5, n + 1, 5):
                    w = {k: v[end - decision.LIVE_BARS:end] for k, v in kl.items()}
                    rule, _, had_setup, _ = self.check_window(w, tf, sym)
                    setups += had_setup
                    trades += rule is not None
        self.assertGreater(setups, 0)
        self.assertGreater(trades, 0)


def _dec(tf="1h", ts=None, action="wait", v1="wait", v2=None, rsi=45.0, sym="BTCUSDT", entry=100.0, atr=1.0):
    ts = T0 if ts is None else ts
    v2 = action if v2 is None else v2
    d = {"sym": sym, "tf": tf, "candle_ts": ts, "action": action, "basis": "rule" if action != "wait" else None,
         "entry": entry if action != "wait" else None, "sl": 98.7 if action != "wait" else None,
         "atr": atr if action != "wait" else None,
         "variants": {"z_only": v1, "d1_no_short_rsi30": v2},
         "variant_ref": {"entry": entry, "atr": atr, "rsi14": rsi}}
    return d


class _Ledgers(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.main = os.path.join(self.tmp.name, "decision_ledger.jsonl")
        self.var = os.path.join(self.tmp.name, "decision_variants_ledger.jsonl")
        self.patches = [mock.patch.object(decision, "LEDGER_PATH", self.main),
                        mock.patch.object(decision, "VARIANTS_LEDGER_PATH", self.var)]
        for p in self.patches:
            p.start()

    def tearDown(self):
        for p in reversed(self.patches):
            p.stop()
        self.tmp.cleanup()

    def rows(self, path=None):
        return decision._read_jsonl(path or self.var)

    def text(self):
        with open(self.var, encoding="utf-8") as f:
            return f.read()

    @staticmethod
    def write(path, rows):
        with open(path, "a", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")


class VariantLedgerTests(_Ledgers):
    def test_records_variants_next_to_the_rule_append_only(self):
        now = (T0 + H_MS) / 1000
        ds = [
            _dec(action="long", v1="wait"),                                   # ستاپ: z_only صبر ⇒ skip
            _dec(sym="ETHUSDT", action="wait", v1="short"),                   # ستاپِ باطل: z_only معامله
            _dec(sym="SOLUSDT", action="long", v1="long"),                    # یکسان
            _dec(tf="1d", ts=T0 - 23 * H_MS, action="short", v1="short", v2="wait", rsi=25.0),
            _dec(tf="1d", ts=T0 - 23 * H_MS, sym="ETHUSDT", action="short", v1="wait", v2="short", rsi=35.0),
            _dec(sym="BNBUSDT", ts=T0 - 10 * H_MS, action="wait", v1="long"),  # کهنه ⇒ ثبت نمی‌شود
            {"sym": "TRXUSDT", "tf": "1h", "error": "x", "variants": {"z_only": "long"}},
        ]
        self.assertEqual(decision.record(ds, now=now), 4)                    # دفترِ اصلی: فقط قاعده
        rows = self.rows()
        self.assertEqual(rows[0]["kind"], "start")
        self.assertEqual(rows[0]["at"], round(now, 3))
        self.assertEqual(rows[0]["variants"], list(decision.VARIANTS))
        got = {(r["variant"], r["sym"], r["tf"], r["kind"], r["side"], r["rule_side"]) for r in rows[1:]}
        self.assertEqual(got, {
            ("z_only", "BTCUSDT", "1h", "skip", "wait", "long"),
            ("z_only", "ETHUSDT", "1h", "open", "short", "wait"),
            ("z_only", "SOLUSDT", "1h", "open", "long", "long"),
            ("z_only", "BTCUSDT", "1d", "open", "short", "short"),
            ("d1_no_short_rsi30", "BTCUSDT", "1d", "skip", "wait", "short"),
            ("z_only", "ETHUSDT", "1d", "skip", "wait", "short"),
            ("d1_no_short_rsi30", "ETHUSDT", "1d", "open", "short", "short"),
        })
        self.assertFalse(any(r.get("variant") == "d1_no_short_rsi30" and r["tf"] != "1d" for r in rows[1:]))
        op = next(r for r in rows if r.get("kind") == "open" and r["sym"] == "ETHUSDT" and r["tf"] == "1h")
        self.assertEqual(op["id"], f"z_only|ETHUSDT|1h|{T0}|short")
        self.assertEqual((op["venue"], op["cost_pct"], op["atr"]), ("binance-spot", 0.14, 1.0))
        self.assertAlmostEqual(op["sl"], 101.3)
        before = self.text()
        self.assertEqual(decision.record_variants(ds, now=now), 0)           # بی‌تکرار، بی start دوم
        self.assertEqual(self.text(), before)
        self.assertEqual(len(self.rows(self.main)), 4)

    def test_same_resolution_path_and_one_fetch_per_cell(self):
        now = (T0 + H_MS) / 1000
        decision.record([_dec(action="long", v1="long"), _dec(sym="ETHUSDT", action="wait", v1="short")], now=now)
        t = [T0 + k * H_MS for k in range(-2, 8)]
        kl = {"t": t, "o": [100.0] * 10, "h": [103.0] * 10, "l": [99.5] * 10, "c": [100.0] * 10,
              "v": [1.0] * 10}
        calls = []

        def gk(sym, tf):
            calls.append((sym, tf))
            if sym == "ETHUSDT":
                raise ConnectionError("down")
            return kl
        self.assertEqual(decision.resolve(gk, now=now + 9 * 3600), 1)         # شمارِ دفترِ اصلی
        self.assertEqual(sorted(calls), [("BTCUSDT", "1h"), ("ETHUSDT", "1h")])   # هر خانه یک‌بار
        main_res = [r for r in self.rows(self.main) if r["kind"] == "result"]
        var_res = [r for r in self.rows() if r.get("kind") == "result"]
        self.assertEqual(len(main_res), 1)
        self.assertEqual(len(var_res), 1)
        self.assertEqual(var_res[0]["variant"], "z_only")
        self.assertEqual(var_res[0]["id"], "z_only|" + main_res[0]["id"])
        for k in ("outcome", "net_r", "gross_r", "entry", "sl", "tp", "exit_ts", "venue"):
            self.assertEqual(var_res[0][k], main_res[0][k], k)
        st = decision.variant_stats(now=now + 9 * 3600)
        item = st["items"][0]
        self.assertEqual(item["key"], "z_only")
        self.assertEqual(item["tfs"]["1h"]["variant"]["n"], 1)
        self.assertEqual(item["tfs"]["1h"]["variant"]["open"], 1)              # ETH هنوز باز
        self.assertEqual(item["tfs"]["1h"]["rule"]["n"], 1)
        self.assertEqual(item["tfs"]["1h"]["diff"], 0.0)
        self.assertEqual(item["status"], "collecting")
        self.assertEqual(item["differs"], 1)


def _open(path_rows, variant, sym, tf, ts, side, net, recorded_at, rule_side=None, exit_ts=None):
    rid = decision.ledger_id(sym, tf, ts, side)
    op = {"kind": "open", "id": rid, "sym": sym, "tf": tf, "side": side, "candle_ts": ts,
          "entry": 100.0, "sl": 98.7, "atr": 1.0, "cost_pct": 0.14, "recorded_at": recorded_at}
    if variant:
        op.update(id=f"{variant}|{rid}", variant=variant, rule_side=rule_side or side)
    else:
        op["basis"] = "rule"
    path_rows.append(op)
    path_rows.append({"kind": "result", "id": op["id"], "net_r": net, "exit_ts": exit_ts or ts + H_MS})


class VariantStatsAndJudgementTests(_Ledgers):
    START = (T0 + 5 * D_MS) / 1000

    def _build(self, weeks, per_week, v_mean, r_mean, tf="1h", variant="z_only", seed=1, pre=0):
        rng = np.random.RandomState(seed)
        vr, mr = [{"kind": "start", "at": self.START, "prereg": decision.VARIANTS_PREREG}], []
        t_start = int(self.START * 1000)
        for w in range(weeks):
            for k in range(per_week):
                ts = t_start + w * 7 * D_MS + k * 12 * H_MS
                rec = ts / 1000 + 60
                _open(vr, variant, "BTCUSDT", tf, ts, "long", round(v_mean + rng.normal(0, 0.8), 4), rec)
                _open(mr, None, "BTCUSDT", tf, ts, "long", round(r_mean + rng.normal(0, 0.8), 4), rec)
        for k in range(pre):                        # پیش از t0 — در مقایسه نمی‌آید
            ts = t_start - (k + 1) * 12 * H_MS
            _open(mr, None, "BTCUSDT", tf, ts, "long", 5.0, ts / 1000 + 60)
        self.write(self.var, vr)
        self.write(self.main, mr)
        return t_start

    def test_stats_compare_the_same_period_only(self):
        self._build(weeks=2, per_week=3, v_mean=0.2, r_mean=-0.1, pre=4)
        st = decision.variant_stats(now=self.START + 3 * 7 * 86400)
        self.assertEqual(st["started_at"], self.START)
        self.assertAlmostEqual(st["weeks"], 3.0, places=2)
        z = st["items"][0]
        self.assertEqual((z["tfs"]["1h"]["variant"]["n"], z["tfs"]["1h"]["rule"]["n"]), (6, 6))   # بی ۴ ردیفِ پیش از t0
        self.assertEqual(z["pooled"]["variant"]["n"], 6)
        self.assertEqual(z["pooled"]["diff"],
                         round(z["pooled"]["variant"]["avg_r"] - z["pooled"]["rule"]["avg_r"], 4))
        self.assertEqual(z["progress"]["min_trades"], 300)
        self.assertEqual(z["judge_tfs"], ["15m", "1h", "4h"])
        self.assertEqual(list(z["tfs"]), list(decision.TFS))
        self.assertEqual(list(st["items"][1]["tfs"]), ["1d"])
        self.assertEqual(decision.judge_variants(now=self.START + 3 * 7 * 86400), [])   # هنوز موعد نیست
        json.dumps(st, ensure_ascii=False)

    def test_not_started(self):
        st = decision.variant_stats(now=self.START)
        self.assertIsNone(st["started_at"])
        self.assertEqual([i["status"] for i in st["items"]], ["not_started", "not_started"])
        self.assertEqual(decision.judge_variants(now=self.START), [])

    def test_z_only_is_judged_once_when_due(self):
        self._build(weeks=27, per_week=12, v_mean=0.25, r_mean=-0.15)
        early = self.START + 25 * 7 * 86400
        self.assertEqual(decision.judge_variants(now=early), [])              # < ۲۶ هفته
        due = self.START + 27 * 7 * 86400
        out = decision.judge_variants(now=due)
        self.assertEqual(len(out), 1)
        j = out[0]
        self.assertEqual((j["variant"], j["result"], j["tfs"]), ("z_only", "pass", ["15m", "1h", "4h"]))
        self.assertGreater(j["diff"], 0)
        self.assertLessEqual(j["p"], 0.05)
        self.assertEqual(j["n_variant"], 27 * 12)
        self.assertEqual(decision.judge_variants(now=due + 7 * 86400), [])    # ثابت؛ داوریِ دوم نیست
        judged = [r for r in self.rows() if r.get("kind") == "judgement"]
        self.assertEqual(len(judged), 1)
        st = decision.variant_stats(now=due)
        self.assertEqual(st["items"][0]["status"], "pass")
        self.assertEqual(st["items"][0]["judgement"]["p"], j["p"])

    def test_z_only_waits_for_300_trades_and_can_fail(self):
        self._build(weeks=27, per_week=10, v_mean=-0.2, r_mean=0.1, seed=2)
        due = self.START + 27 * 7 * 86400
        self.assertEqual(decision.judge_variants(now=due), [])               # ۲۷۰ < ۳۰۰
        more = []
        t_start = int(self.START * 1000)
        for k in range(40):
            ts = t_start + 27 * 7 * D_MS + k * 12 * H_MS
            _open(more, "z_only", "BTCUSDT", "4h", ts, "short", -0.3, ts / 1000)
        self.write(self.var, more)
        out = decision.judge_variants(now=due + 30 * 86400)
        self.assertEqual([(r["variant"], r["result"]) for r in out], [("z_only", "fail")])

    def test_d1_variant_is_inconclusive_without_enough_differing_bars(self):
        self._build(weeks=53, per_week=1, v_mean=0.3, r_mean=-0.3, tf="1d", variant="d1_no_short_rsi30")
        t_start = int(self.START * 1000)             # کندلِ پیش از t0 در دورهٔ مقایسه نیست (گاردِ کهنگی هم نمی‌گذارد)
        skips = [{"kind": "skip", "id": f"d1_no_short_rsi30|BTCUSDT|1d|{t_start + k * D_MS}|wait",
                  "variant": "d1_no_short_rsi30", "sym": "BTCUSDT", "tf": "1d", "side": "wait",
                  "candle_ts": t_start + k * D_MS, "rule_side": "short", "recorded_at": self.START}
                 for k in range(9)]
        self.write(self.var, skips)
        self.assertEqual(decision.judge_variants(now=self.START + 51 * 7 * 86400), [])
        out = decision.judge_variants(now=self.START + 53 * 7 * 86400)
        self.assertEqual([(r["variant"], r["result"], r["differs"]) for r in out],
                         [("d1_no_short_rsi30", "inconclusive", 9)])
        self.assertIsNone(out[0]["p"])

    def test_d1_variant_is_tested_with_ten_differing_bars(self):
        self._build(weeks=53, per_week=1, v_mean=0.3, r_mean=-0.3, tf="1d", variant="d1_no_short_rsi30")
        t_start = int(self.START * 1000)             # کندلِ پیش از t0 در دورهٔ مقایسه نیست (گاردِ کهنگی هم نمی‌گذارد)
        skips = [{"kind": "skip", "id": f"d1_no_short_rsi30|BTCUSDT|1d|{t_start + k * D_MS}|wait",
                  "variant": "d1_no_short_rsi30", "sym": "BTCUSDT", "tf": "1d", "side": "wait",
                  "candle_ts": t_start + k * D_MS, "rule_side": "short", "recorded_at": self.START}
                 for k in range(10)]
        self.write(self.var, skips + skips[:2])                             # تکرار شمرده نمی‌شود
        out = decision.judge_variants(now=self.START + 53 * 7 * 86400)
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0]["differs"], 10)
        self.assertIn(out[0]["result"], ("pass", "fail"))
        self.assertIsNotNone(out[0]["p"])

    def test_t0_boundary_candle_is_out_of_both_streams(self):
        """کندلِ c پیش از t0 بسته شده: دفترِ اصلی آن را پیش از t0 ثبت کرده و دورِ اولِ گونه‌ها در t0.

        V2 در هر کندل عیناً قاعده است؛ پس مقایسه باید تفاوتِ صفر بدهد — نه معاملهٔ c فقط در جریانِ گونه
        (که c+1 را هم پشتِ خودش نگه می‌دارد) و c+1 فقط در جریانِ قاعده.
        """
        v, t_start = "d1_no_short_rsi30", int(self.START * 1000)
        c = t_start - 28 * H_MS                     # بسته‌شدن: t0 − ۴ ساعت
        vr, mr = [{"kind": "start", "at": self.START, "prereg": decision.VARIANTS_PREREG}], []
        _open(mr, None, "BTCUSDT", "1d", c, "long", 1.8, self.START - 3 * 3600, exit_ts=c + 3 * D_MS)
        _open(vr, v, "BTCUSDT", "1d", c, "long", 1.8, self.START, exit_ts=c + 3 * D_MS)
        _open(mr, None, "BTCUSDT", "1d", c + D_MS, "long", -1.0, self.START + 86400)
        _open(vr, v, "BTCUSDT", "1d", c + D_MS, "long", -1.0, self.START + 86400)
        _open(vr, "z_only", "BTCUSDT", "1d", c, "long", 1.8, self.START, exit_ts=c + 3 * D_MS)
        vr.append({"kind": "skip", "id": f"{v}|BTCUSDT|1d|{c}|wait", "variant": v, "sym": "BTCUSDT",
                   "tf": "1d", "side": "wait", "candle_ts": c, "rule_side": "short", "recorded_at": self.START})
        self.write(self.var, vr)
        self.write(self.main, mr)
        st = decision.variant_stats(now=self.START + 5 * 86400)
        items = {i["key"]: i for i in st["items"]}
        cell = items[v]["tfs"]["1d"]
        self.assertEqual((cell["variant"]["n"], cell["rule"]["n"], cell["diff"]), (1, 1, 0.0))
        self.assertEqual((cell["variant"]["open"], cell["rule"]["open"]), (0, 0))
        self.assertEqual(items[v]["pooled"]["diff"], 0.0)
        self.assertEqual(items["z_only"]["tfs"]["1d"]["variant"]["n"], 0)    # c فقط در دفترِ گونه ⇒ بیرون
        self.assertEqual(items[v]["differs"], 0)                             # ردیفِ skipِ مرزی هم شمرده نمی‌شود
        self.assertEqual(sorted(decision._in_period(r, self.START) for r in vr if r.get("kind") == "open"),
                         [False, False, True])


class BootstrapTests(unittest.TestCase):
    def test_weekly_blocks_and_one_sided_p(self):
        f = decision.weekly_block_bootstrap
        monday = decision.MONDAY0_MS + 2900 * decision.WEEK_MS
        self.assertEqual(f([(monday - 1, 1.0)], [(monday, 0.0)])["weeks"], 2)   # یکشنبه ≠ دوشنبه
        self.assertEqual(f([(monday, 1.0)], [(monday + 6 * D_MS, 0.0)])["weeks"], 1)
        rng = np.random.RandomState(0)
        ts = [monday + k * 12 * H_MS for k in range(400)]
        good = [(t, 0.3 + rng.normal(0, 1)) for t in ts]
        base = [(t, -0.2 + rng.normal(0, 1)) for t in ts]
        a = f(good, base)
        self.assertEqual(a, f(good, base))                                      # قطعی (seed ثابت)
        self.assertGreater(a["diff"], 0)
        self.assertLess(a["p"], 0.01)
        self.assertEqual(a["valid"], decision.BOOT_N)
        b = f(base, good)
        self.assertGreater(b["p"], 0.99)
        same = f(good, good)
        self.assertEqual(same["diff"], 0.0)
        self.assertEqual(same["p"], 1.0)                                       # تفاوت ≤ ۰ در همهٔ تکرارها
        empty = f([], base)
        self.assertEqual((empty["diff"], empty["p"], empty["n_rule"]), (None, None, 400))


class WiringTests(unittest.TestCase):
    def test_prereg_is_committed_and_matches_the_code(self):
        with open(PREREG, encoding="utf-8") as f:
            pre = json.load(f)
        self.assertEqual(pre["registered_at_utc"], "2026-09-24")
        self.assertEqual(sorted(pre["variants"]), ["V1_z_only", "V2_d1_no_short_rsi30"])
        self.assertEqual(decision.VARIANTS, ("z_only", "d1_no_short_rsi30"))
        self.assertIn("10000", pre["test"])
        self.assertIn(str(decision.BOOT_SEED), pre["test"])
        self.assertIn("26 weeks", pre["judgement"]["V1"])
        self.assertIn("300", pre["judgement"]["V1"])
        self.assertIn("52 weeks", pre["judgement"]["V2"])
        self.assertIn("gates.json", pre["authority"])

    def test_variant_ledger_is_gitignored_and_in_the_payload(self):
        with open(os.path.join(ROOT, ".gitignore"), encoding="utf-8") as f:
            self.assertIn("bot/data/decision_variants_ledger.jsonl", f.read().split())
        with tempfile.TemporaryDirectory() as tmp, \
                mock.patch.object(decision, "VARIANTS_LEDGER_PATH", os.path.join(tmp, "v.jsonl")):
            p = decision.payload([], scorecard={"cells": {}}, live={"cells": {}, "overall": {"n": 0}})
        self.assertEqual([i["key"] for i in p["variants"]["items"]], list(decision.VARIANTS))
        self.assertIn("gates.json", p["variants"]["note"])
        json.dumps(p, ensure_ascii=False)


if __name__ == "__main__":
    unittest.main()
