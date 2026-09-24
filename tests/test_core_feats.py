# -*- coding: utf-8 -*-
"""هستهٔ v2 — ویژگی‌ها (core_feats): نام‌ها عیناً پیش‌ثبت، علّی بودن، برابریِ دُم (تاریخچه = پنجرهٔ زنده)،
NaN به‌جای جایگزینِ بی‌صدا، و برچسب = ``bracket.signal_trade``."""
import json
import os
import sys
import unittest

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "bot"))
import _hermetic  # noqa: E402,F401  — پیش از هر ماژولِ ربات: داده به پوشهٔ موقت

import bracket  # noqa: E402
import core_feats as cf  # noqa: E402
import engine  # noqa: E402

PREREG = os.path.join(ROOT, "bot", "data", "research", "prereg_core2.json")
MICRO = os.path.join(ROOT, "bot", "data", "hist_research", "micro")
COINS = ("BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "TRXUSDT")
T0 = 19_700 * cf.DAY_MS                         # مرزِ روزِ UTC


def synth(n, tf="1h", seed=0, t0=T0, drift_scale=1.0):
    rng = np.random.RandomState(seed)
    step = cf.BAR_MS[tf]
    drift = np.repeat(rng.choice([-1.0, 0.0, 1.0], n // 150 + 1), 150)[:n] * 0.0012 * drift_scale
    c = 100 * np.exp(np.cumsum(drift + rng.normal(0, 0.006, n)))
    o = np.r_[c[0], c[:-1]] * (1 + rng.normal(0, 0.0005, n))
    wig = np.abs(rng.normal(0, 0.003, n)) * c
    v = rng.lognormal(10, 0.5, n)
    nt = np.floor(v / rng.uniform(5, 20, n)) + 1
    return {"t": t0 + np.arange(n, dtype=np.int64) * step, "o": o, "h": np.maximum(o, c) + wig,
            "l": np.minimum(o, c) - wig, "c": c, "v": v, "qv": v * c * (1 + rng.normal(0, 1e-3, n)),
            "n": nt, "tbv": v * np.clip(rng.normal(0.5, 0.08, n), 0.05, 0.95)}


def synth_fund(t_lo, t_hi, seed=0, interval_h=8.0, anchor_h=4):
    """تسویه‌های کوکوین‌مانند (۰۴/۱۲/۲۰ UTC برای ۸ ساعته)."""
    rng = np.random.RandomState(seed)
    step = int(interval_h * cf.HOUR_MS)
    start = (t_lo // cf.DAY_MS) * cf.DAY_MS + anchor_h * cf.HOUR_MS - 200 * step
    ts = np.arange(start, t_hi + step, step, dtype=np.int64)
    rate = 1e-4 + rng.normal(0, 8e-5, len(ts))
    rate[rng.rand(len(ts)) < 0.3] = 1e-4                 # نرخِ قفل‌شده روی ۰٫۰۱٪ مثلِ بازارِ آرام
    return {"t": ts, "rate": rate, "interval_h": np.full(len(ts), float(interval_h))}


def world(n, tf="1h", seed=0):
    coins = {s: synth(n, tf, seed=seed + 11 * k, drift_scale=1 + 0.2 * k) for k, s in enumerate(COINS)}
    t = coins["BTCUSDT"]["t"]
    ctx = {"btc": coins["BTCUSDT"], "eth": coins["ETHUSDT"], "closes5": coins,
           "kfund": synth_fund(int(t[0]), int(t[-1]) + cf.BAR_MS[tf], seed=seed + 7)}
    return coins, ctx


def cut(b, lo, hi):
    t = np.asarray(b["t"])
    i0, i1 = np.searchsorted(t, lo), np.searchsorted(t, hi, side="right")
    return {k: np.asarray(v)[i0:i1] for k, v in b.items()}


def close_enough(a, b, rtol=1e-4, atol=1e-6):
    a, b = float(a), float(b)
    if np.isnan(a) or np.isnan(b):
        return np.isnan(a) and np.isnan(b)
    return abs(a - b) <= atol + rtol * abs(b)


def _ld(path):
    z = np.load(path)
    return {k: z[k] for k in z.files}


class TestNamesAndShape(unittest.TestCase):
    def test_names_equal_prereg_groups(self):
        with open(PREREG, encoding="utf-8") as fh:
            groups = json.load(fh)["features"]["groups"]
        want = {g: [x.split(" (")[0].strip() for x in fs] for g, fs in groups.items()}
        self.assertEqual(want, cf.GROUPS)                   # همان گروه‌ها، همان ترتیب، همان نام‌ها
        for tf in cf.TFS:
            exp = [f for g in want.values() for f in g
                   if not (f == "svwap_dist" and tf not in ("5m", "15m", "1h"))
                   and f not in cf.DROPPED.get(tf, {})]
            self.assertEqual(cf.FEATURES[tf], exp, tf)
        # هر حذف دلیلِ مکتوب دارد و فقط حذف‌های «ثابت» در 1d است
        self.assertEqual(set(cf.DROPPED), {"1d"})
        self.assertEqual(set(cf.DROPPED["1d"]), {"hour_sin", "hour_cos", "us_sess", "min_to_fund"})
        self.assertIn("svwap_dist", cf.FEATURES["5m"])
        self.assertNotIn("svwap_dist", cf.FEATURES["4h"])
        self.assertEqual(set(cf.LIVE_BARS), set(cf.TFS))
        for tf in cf.TFS:
            self.assertGreaterEqual(cf.LIVE_BARS[tf], cf.ZW[tf] + 64 + cf.CAUSAL_START // 2)

    def test_output_float32_aligned(self):
        coins, ctx = world(1500, "1h")
        F = cf.compute("1h", coins["SOLUSDT"], ctx)
        self.assertEqual(list(F), cf.FEATURES["1h"])
        for f, a in F.items():
            self.assertEqual(a.dtype, np.float32, f)
            self.assertEqual(len(a), 1500, f)
            self.assertTrue(np.all(np.isnan(a[:cf.CAUSAL_START])), f)
            self.assertTrue(np.isfinite(a[-1]), f)                      # سطرِ آخر کامل است
        self.assertEqual(cf.stack("1h", F).shape, (1500, len(cf.FEATURES["1h"])))
        with self.assertRaises(KeyError):
            cf.compute("3m", coins["SOLUSDT"], ctx)

    def test_zs_prior_window_half_present(self):
        rng = np.random.RandomState(4)
        x = rng.normal(0, 1, 400) + np.linspace(0, 5, 400)
        x[rng.rand(400) < 0.2] = np.nan
        w = 60
        got = cf.zs(x, w)
        for i in range(400):
            win = x[max(0, i - w):i]
            win = win[np.isfinite(win)]
            if not np.isfinite(x[i]) or len(win) < w // 2:
                self.assertTrue(np.isnan(got[i]), i)
            else:
                self.assertAlmostEqual(got[i], (x[i] - win.mean()) / win.std(ddof=1), places=9)
        self.assertTrue(np.all(np.isnan(cf.zs(np.full(200, 1e-4), 90)[100:])))   # پنجرهٔ ثابت ⇒ NaN

    def test_engine_parts_match_engine(self):
        b = synth(1400, "1h", seed=5)
        o, h, l, c, v = (b[k] for k in ("o", "h", "l", "c", "v"))
        cs = engine.component_series(o, h, l, c, v)
        F = cf.compute("1h", b, {})
        codes = {None: 0, "zx": 1, "pb": 2, "sq": 3, "fd": 4, "rg": 5}
        seen = set()
        for i in range(cf.CAUSAL_START, 1400):
            bb, ss = engine.votes_at(cs, i)
            self.assertEqual(F["votes_net"][i], bb - ss)
            code = codes[engine.setup_signal(cs, o, h, l, c, i)[1]]
            self.assertEqual(F["setup_code"][i], code, i)
            seen.add(code)
        self.assertGreaterEqual(len(seen), 3)


class TestCausality(unittest.TestCase):
    def _run(self, tf, n):
        coins, ctx = world(n, tf, seed=1)
        base = cf.compute(tf, coins["SOLUSDT"], ctx)
        d = n - 150
        td = int(coins["BTCUSDT"]["t"][d])
        rng = np.random.RandomState(9)

        def wreck(b):
            b = {k: np.array(v, copy=True) for k, v in b.items()}
            m = b["t"] > td
            k = m.sum()
            for f in ("o", "h", "l", "c"):
                b[f][m] *= 1.7
            b["h"][m] *= 1.05
            b["l"][m] *= 0.95
            for f in ("v", "qv", "n", "tbv"):
                b[f][m] *= rng.uniform(5, 50, k)
            return b
        coins2 = {s: wreck(b) for s, b in coins.items()}
        kf = dict(ctx["kfund"])
        kf["rate"] = np.where(kf["t"] > td + cf.BAR_MS[tf], 0.02, kf["rate"])
        ctx2 = {"btc": coins2["BTCUSDT"], "eth": coins2["ETHUSDT"], "closes5": coins2, "kfund": kf}
        pert = cf.compute(tf, coins2["SOLUSDT"], ctx2)
        changed_after = 0
        for f in cf.FEATURES[tf]:
            a, b = base[f][:d + 1], pert[f][:d + 1]
            np.testing.assert_array_equal(np.isnan(a), np.isnan(b), err_msg=f"{tf} {f}")
            np.testing.assert_allclose(a, b, rtol=1e-6, atol=1e-7, equal_nan=True, err_msg=f"{tf} {f}")
            changed_after += not np.allclose(base[f][d + 1:], pert[f][d + 1:], equal_nan=True)
        self.assertGreater(changed_after, 20)                 # اختلال واقعاً اثر داشته است

    def test_future_bars_do_not_move_past_rows(self):
        for tf, n in (("5m", cf.ZW["5m"] + 900), ("15m", 1500), ("1h", 1400), ("4h", 900), ("1d", 700)):
            with self.subTest(tf=tf):
                self._run(tf, n)


class TestTailParity(unittest.TestCase):
    """سطرِ آخرِ تاریخچهٔ بلند = سطرِ آخرِ همان‌قدر کندلی که زنده می‌گیرد (LIVE_BARS)."""

    def _data(self, tf):
        spot = {s: os.path.join(MICRO, f"spot_{s}_{tf}.npz") for s in COINS}
        if all(os.path.exists(p) for p in spot.values()):
            return {s: _ld(p) for s, p in spot.items()}, "spot"
        if tf != "5m":
            s15 = {s: os.path.join(MICRO, f"spot_{s}_15m.npz") for s in COINS}
            if all(os.path.exists(p) for p in s15.values()):
                return {s: cf.resample(_ld(p), "15m", tf) for s, p in s15.items()}, "spot15m→" + tf
        um = {s: os.path.join(MICRO, f"um_{s}_{tf}.npz") for s in COINS}
        if all(os.path.exists(p) for p in um.values()):
            return {s: _ld(p) for s, p in um.items()}, "um"
        return None, "synthetic"

    def _kfund(self, sym, t):
        p = os.path.join(MICRO, f"fund_{sym}.npz")          # جایگزینِ هم‌قالب فقط برای سازوکارِ آزمون
        if os.path.exists(p):
            z = _ld(p)
            return {"t": z["t"], "rate": z["rate"], "interval_h": z["interval_h"]}
        return synth_fund(int(t[0]), int(t[-1]) + 10 * cf.DAY_MS, seed=3)

    def _check(self, tf, coins, sym, hist_len):
        L = cf.LIVE_BARS[tf]
        n_all = len(coins[sym]["t"])
        hist_len = min(hist_len, n_all)
        lo = int(coins[sym]["t"][n_all - hist_len])
        hist = {s: cut(b, lo, 1 << 62) for s, b in coins.items()}
        B = hist[sym]
        kf = self._kfund(sym, B["t"])
        full = cf.compute(tf, B, {"btc": hist["BTCUSDT"], "eth": hist["ETHUSDT"], "closes5": hist,
                                  "kfund": kf})
        n = len(B["t"])
        for e in (n - 1, n - 1 - 97):
            if e + 1 < L:
                continue
            t_lo, t_hi = int(B["t"][e - L + 1]), int(B["t"][e])
            live = {s: cut(b, t_lo, t_hi) for s, b in hist.items()}
            close = t_hi + cf.BAR_MS[tf]
            j = np.searchsorted((np.asarray(kf["t"]) // 60_000) * 60_000, close, side="right")
            kfl = {k: np.asarray(v)[max(0, j - cf.KFUND_LIVE_SETTLEMENTS):j] for k, v in kf.items()}
            T = cf.compute(tf, live[sym], {"btc": live["BTCUSDT"], "eth": live["ETHUSDT"],
                                           "closes5": live, "kfund": kfl})
            self.assertEqual(len(T["z"]), L)
            bad = [f for f in cf.FEATURES[tf] if not close_enough(T[f][-1], full[f][e])]
            self.assertEqual(bad, [], f"{tf} {sym} end={e}: " + ", ".join(
                f"{f} live={T[f][-1]} hist={full[f][e]}" for f in bad))

    def test_tail_parity_every_tf(self):
        hist_len = {"5m": 40_000, "15m": 10 ** 9, "1h": 10 ** 9, "4h": 10 ** 9, "1d": 10 ** 9}
        for tf in cf.TFS:
            coins, src = self._data(tf)
            with self.subTest(tf=tf, source=src):
                if coins is None:
                    n = cf.LIVE_BARS[tf] * 2 + 300
                    coins, _ = world(n, tf, seed=2)
                if len(coins["SOLUSDT"]["t"]) < cf.LIVE_BARS[tf] + 100:
                    self.skipTest(f"{tf}: تاریخچهٔ محلی کوتاه‌تر از LIVE_BARS است")
                self._check(tf, coins, "SOLUSDT", hist_len[tf])

    def test_tail_parity_synthetic_all_tfs(self):
        for tf in ("5m", "1h", "1d"):
            with self.subTest(tf=tf):
                coins, _ = world(cf.LIVE_BARS[tf] + 1500, tf, seed=6)
                self._check(tf, coins, "TRXUSDT", 10 ** 9)


class TestMissingData(unittest.TestCase):
    def test_tbv_nan_blanks_order_flow_only(self):
        coins, ctx = world(1400, "1h", seed=3)
        base = cf.compute("1h", coins["BNBUSDT"], ctx)
        b = {k: np.array(v, copy=True) for k, v in coins["BNBUSDT"].items()}
        b["tbv"][1000:1010] = np.nan
        F = cf.compute("1h", b, ctx)
        g3 = cf.GROUPS["G3_spot_order_flow"]
        for f in g3:
            self.assertTrue(np.all(np.isnan(F[f][1000:1010])), f)
        for f in ("sflow_64", "sflow_16"):                      # پنجره‌ای که NaN را در بر دارد هم NaN
            self.assertTrue(np.isnan(F[f][1012]), f)
        for f in cf.FEATURES["1h"]:
            if f not in g3:
                np.testing.assert_array_equal(F[f], base[f], err_msg=f)
        self.assertTrue(np.isfinite(F["sflow_4_z"][1100]))       # پس از NaN دوباره زنده می‌شود
        b2 = {k: np.array(v, copy=True) for k, v in b.items()}
        b2["tbv"][:] = np.nan                                    # منبعِ بی‌تیکر (مثلاً OKX)
        F2 = cf.compute("1h", b2, ctx)
        for f in g3:
            self.assertTrue(np.all(np.isnan(F2[f])), f)
        btc = {k: np.array(v, copy=True) for k, v in ctx["btc"].items()}
        btc["tbv"][1200:1205] = np.nan
        F3 = cf.compute("1h", coins["BNBUSDT"], dict(ctx, btc=btc))
        self.assertTrue(np.all(np.isnan(F3["btc_flow16_z"][1200:1205])))

    def test_kfund_missing_or_stale_is_nan(self):
        coins, ctx = world(1400, "1h", seed=3)
        kcols = ("kfund_bp", "kfund_z", "kfund_persist", "min_to_fund")
        for bad in (None, {"t": [], "rate": [], "interval_h": []}):
            F = cf.compute("1h", coins["SOLUSDT"], dict(ctx, kfund=bad))
            for f in kcols:
                self.assertTrue(np.all(np.isnan(F[f])), f)
            self.assertTrue(np.isfinite(F["z"][-1]))
        t = coins["SOLUSDT"]["t"]
        kf = ctx["kfund"]
        keep = kf["t"] < t[1000]                               # فاندینگ از کندلِ ۱۰۰۰ به بعد قطع
        F = cf.compute("1h", coins["SOLUSDT"], dict(ctx, kfund={k: v[keep] for k, v in kf.items()}))
        stale = (t + cf.BAR_MS["1h"]) - kf["t"][keep][-1] > 2 * 8 * cf.HOUR_MS
        for f in kcols:
            self.assertTrue(np.all(np.isnan(F[f][stale])), f)
            self.assertTrue(np.isfinite(F[f][999]), f)

    def test_cross_asset_needs_same_close(self):
        coins, ctx = world(1400, "1h", seed=8)
        drop = 1100
        eth = {k: np.delete(v, drop) for k, v in coins["ETHUSDT"].items()}
        bnb = {k: np.delete(v, drop) for k, v in coins["BNBUSDT"].items()}
        c5 = dict(coins, BNBUSDT=bnb)
        F = cf.compute("1h", coins["SOLUSDT"], dict(ctx, eth=eth, closes5=c5))
        self.assertTrue(np.isnan(F["ethbtc_z"][drop]))
        self.assertTrue(np.isnan(F["breadth"][drop]))
        self.assertTrue(np.isfinite(F["breadth"][drop + 1]))
        four = {k: v for k, v in coins.items() if k != "TRXUSDT"}   # تعریف پنج ارز است؛ چهار ⇒ NaN
        self.assertTrue(np.all(np.isnan(cf.compute("1h", coins["SOLUSDT"], dict(ctx, closes5=four))["breadth"])))
        F0 = cf.compute("1h", coins["SOLUSDT"], {})
        for f in ("btc_roc_m", "btc_z", "btc_flow16_z", "rs_btc_m", "ethbtc_z", "breadth"):
            self.assertTrue(np.all(np.isnan(F0[f])), f)

    def test_btc_row_equals_own_values(self):
        coins, ctx = world(1400, "4h", seed=2)
        F = cf.compute("4h", coins["BTCUSDT"], ctx)
        r = slice(cf.CAUSAL_START, None)
        np.testing.assert_array_equal(F["btc_roc_m"][r], F["roc_m"][r])
        np.testing.assert_array_equal(F["btc_z"][r], F["z"][r])
        np.testing.assert_array_equal(F["btc_flow16_z"][r], F["sflow_16_z"][r])
        self.assertTrue(np.all(F["rs_btc_m"][r][np.isfinite(F["roc_m"][r])] == 0))
        # همان BTC با کپیِ جدا (نه همان شیء) هم همان نتیجه
        other = cf.compute("4h", coins["BTCUSDT"], dict(ctx, btc={k: np.array(v) for k, v in ctx["btc"].items()}))
        np.testing.assert_allclose(other["btc_z"][r], F["z"][r], rtol=1e-6)
        # breadth: سهمِ پنج ارز بالای EMA50 منهای ۰٫۵
        i = 1000
        up = sum(float(b["c"][i] > engine.ema(b["c"], 50)[i]) for b in coins.values())
        self.assertAlmostEqual(float(F["breadth"][i]), up / 5 - 0.5, places=6)


class TestKucoinFunding(unittest.TestCase):
    def test_settled_rate_normalised_and_schedule(self):
        n = 1200
        b = synth(n, "15m", seed=1)
        t = b["t"]
        step = 4 * cf.HOUR_MS                                   # بازهٔ ۴ ساعته: نرخ ×۲ برای ۸ ساعت
        ts = np.arange(int(t[0]) - 150 * step, int(t[-1]) + step, step, dtype=np.int64) + 7   # لرزشِ ms
        rng = np.random.RandomState(0)
        rate = rng.normal(1e-4, 5e-5, len(ts))
        kf = {"t": ts, "rate": rate, "interval_h": np.full(len(ts), 4.0)}
        F = cf.compute("15m", b, {"kfund": kf})
        tsm = (ts // 60_000) * 60_000
        for i in (400, 401, 555, 777, 1100):
            close = int(t[i]) + cf.BAR_MS["15m"]
            j = np.searchsorted(tsm, close, side="right") - 1
            self.assertLessEqual(tsm[j], close)
            self.assertAlmostEqual(float(F["kfund_bp"][i]), rate[j] * 2 * 1e4, places=3)
            prior = rate[j - 90:j] * 2
            self.assertAlmostEqual(float(F["kfund_z"][i]), (rate[j] * 2 - prior.mean()) / prior.std(ddof=1),
                                   places=4)
            self.assertAlmostEqual(float(F["kfund_persist"][i]), np.sign(rate[j - 8:j + 1]).mean(), places=6)
            nxt = tsm[j] + step
            self.assertAlmostEqual(float(F["min_to_fund"][i]), (nxt - close) / step, places=6)
        # تسویه دقیقاً روی بسته‌شدنِ کندل همان کندل را می‌گیرد و فاصله = یک بازهٔ کامل
        on = [i for i in range(cf.CAUSAL_START, n) if (int(t[i]) + cf.BAR_MS["15m"]) in set(tsm.tolist())]
        self.assertTrue(on)
        for i in on[:5]:
            self.assertAlmostEqual(float(F["min_to_fund"][i]), 1.0, places=6)


class TestLabels(unittest.TestCase):
    def _compare(self, b, idx, cost=0.14):
        o, h, l, c = (np.asarray(b[k], float) for k in ("o", "h", "l", "c"))
        yL, yS, ex, rp = cf.labels("1h", b, cost=cost)
        a14 = engine.atr(h, l, c, 14)
        n = len(c)
        self.assertTrue(np.isnan(yL[-1]) and np.isnan(yS[-1]))
        for i in idx:
            exits = []
            for sig, y in ((1, yL), (-1, yS)):
                res = bracket.signal_trade(o, h, l, c, float(a14[i]), i, sig, bracket.MAX_BARS)
                if res["timed_out"] and i + bracket.MAX_BARS > n - 1:
                    self.assertTrue(np.isnan(y[i]), (i, sig))           # نامعلوم ⇒ NaN
                    continue
                exits.append(res["exit_idx"])
                want = np.clip(bracket.net_r(res["gross_r"], res["risk_pct"], cost), -2, 1.8)
                self.assertAlmostEqual(float(y[i]), float(np.float32(want)), places=6, msg=(i, sig))
                self.assertAlmostEqual(float(rp[i]), res["risk_pct"], places=12)
            self.assertEqual(int(ex[i]), max(exits) if exits else -1, i)

    def test_labels_equal_signal_trade_synthetic_with_gaps(self):
        b = synth(3000, "1h", seed=12)
        rng = np.random.RandomState(1)
        jumps = rng.choice(np.arange(50, 2990), 60, replace=False)
        for j in jumps:                                           # گپ‌های بزرگ برای «gap_stop»
            f = 1 + rng.choice([-1, 1]) * rng.uniform(0.02, 0.06)
            for k in ("o", "h", "l", "c"):
                b[k][j:] = b[k][j:] * f
            b["h"][j] = max(b["h"][j], b["o"][j])
            b["l"][j] = min(b["l"][j], b["o"][j])
        idx = np.r_[rng.choice(np.arange(20, 2998), 500, replace=False), np.arange(2950, 2999)]
        self._compare(b, idx)
        self._compare(b, idx[:100], cost=0.10)
        yL, yS, _, _ = cf.labels("1h", b)
        self.assertTrue(np.nanmax(yL) <= 1.8 and np.nanmin(yL) >= -2.0)
        self.assertTrue(np.isnan(yL[-1]) and np.isfinite(yL[100]))

    def test_labels_equal_signal_trade_real_futures(self):
        p = os.path.join(MICRO, "um_TRXUSDT_5m.npz")
        if not os.path.exists(p):
            self.skipTest("um_TRXUSDT_5m.npz محلی نیست")
        z = np.load(p)
        b = {k: z[k][-20_000:] for k in ("t", "o", "h", "l", "c")}
        rng = np.random.RandomState(3)
        self._compare(b, np.r_[rng.choice(np.arange(20, 19_999), 400, replace=False), np.arange(19_955, 19_999)])


class TestResample(unittest.TestCase):
    def test_complete_buckets_only(self):
        b = synth(4 * 50, "15m", seed=1)
        b = {k: np.delete(v, [9, 101]) for k, v in b.items()}   # دو سطلِ ناقص
        b = {k: v[1:] for k, v in b.items()}                     # سطلِ اولِ ناقص
        r = cf.resample(b, "15m", "1h")
        self.assertEqual(len(r["t"]), 50 - 3)
        self.assertTrue(np.all(r["t"] % cf.BAR_MS["1h"] == 0))
        full = synth(4 * 50, "15m", seed=1)
        k = 10                                                    # سطلِ دست‌نخورده
        i = np.searchsorted(r["t"], full["t"][4 * k])
        self.assertEqual(r["o"][i], full["o"][4 * k])
        self.assertEqual(r["c"][i], full["c"][4 * k + 3])
        self.assertEqual(r["h"][i], full["h"][4 * k:4 * k + 4].max())
        self.assertAlmostEqual(r["tbv"][i], full["tbv"][4 * k:4 * k + 4].sum())


if __name__ == "__main__":
    unittest.main()
