# Trading-Bot Audit: Final Report

**Scope.** Ten audit streams covered the codebase: data ingestion, labels and bracket, indicator features, flow and funding features, the matrix rule, the model pipeline, indicator combinations, the live calib layer, the live decision path, and the display and advisor. A separate verifier re-derived each finding with its own code.

**Data rules.** Every performance number in this report uses bars before 2026-01-01. No statistic was computed on the 2026 holdout. One exception is disclosed: the live-path audit accidentally made one read-only OKX request, and no statistic was computed from it. The repository was not modified. Scratch scripts are under `/tmp/claude-0/-home-user-tradebot/8391bdd6-ea49-5c4a-b988-571e3617ef8c/scratchpad/audit/`.

---

## 1. Bottom line

No bug explains the poor research results. The losses are real.

**The research pipeline checks out end to end:**
- Labels match an independent brute-force bracket exactly.
- Refitting the saved v2.1 models reproduces their predictions bit for bit.
- Features pass causality truncation tests.
- The replay, block bootstrap, Holm correction and adoption checks all reproduce the stored reports.

**5m, 15m and 1h lose because of cost against a narrow stop.** The stop is 1.3×ATR with a 0.25% floor, and every trade pays a fixed 0.14% round-trip cost. Across the validation window that cost is 0.47R per trade at 5m, 0.34R at 15m and 0.17R at 1h. The average gross result of the bracket is about zero at every timeframe.
- The matrix rule's losses there equal those of coin-flip entries.
- v2.1's 1h loss is entirely cost: gross +0.006R, net −0.136R.
- v2.1 makes no trades at 5m because every net label averages about −0.44R.

**4h and 1d lose for a different reason.** Cost there is small (0.08R and 0.03R). v2.1 loses on gross because of which trades it picks. Its 1d model shows no stable skill.

**No fix of a real defect changed a verdict.** Each defect was fixed in a counterfactual rerun. The results moved only within seed-to-seed noise, no negative result turned positive, and no adoption decision changed.

**On the indicator-combination question:**
- No component has its sign reversed, and no combination cancels another. Flipping every signal makes every timeframe worse.
- The rule's five "votes" are mostly the same information counted several times.
- The "AI" (kNN) vote is noise.
- Re-combining the components changes net R by at most about +0.03R at 5m–1h, while the cost gap there is 0.12–0.45R.

**The real bugs are in the live, ledger and display layer.** None of them feeds the validation numbers.
- The most serious one shows a bracket win probability as a calibrated "probability of rise". That inverts the displayed direction and would make the advisor and autotrader close good trades. It is dormant in this repository.
- The heuristic "probability of rise" currently displayed has no skill.
- Several live model inputs are stale or never matched training: market dominance is pinned at +1, gold is up to 24h old, and funding timing is misaligned.

---

## Reference: the reported validation results

| TF | v2.1 (reported) | Rule (reported) | Notes |
|---|---|---|---|
| 5m | 0 trades | −0.433 (n=15930) | v2.1's highest prediction is −0.025 |
| 15m | −0.002 (n=190, SE 0.096) | −0.296 (n=6204) | |
| 1h | −0.137 (n=850, SE 0.045) | −0.173 (n=1548) | |
| 4h | −0.101 (n=591) | +0.036 (n=402, SE 0.068) | |
| 1d | −0.122 (n=112) | +0.131 (n=82, SE 0.153) | v2 (older window) 1d: +0.083 (n=100) |

Nothing was adopted (`adopted=[]`). Under the current pre-registration the 2026 holdout is therefore never run.

---

## 2. Confirmed bugs (status confirmed or partially), ranked by impact on results

**How much results move from noise alone.** Refitting v2.1 with a different seed, or dropping one unused column, already moves the results:
- 1h across seeds: −0.133 to −0.162.
- 4h seed spread: about 0.10R.
- 1d baseline across 6 seeds: −0.034 to −0.192.
- Dropping the zero-gain column `min_to_fund` alone moved 4h from −0.101 to −0.135 and 1h from −0.137 to −0.167.

Every shift below should be read against that noise.

### 2A. Issues that touch the research numbers (v2, v2.1, rule validation)

**A1. The IC and decile tables mostly measure cost, not skill.** Findings model-1 and combos-1, confirmed. Location: `bot/core2.py:502` (`ic_stats`, used at `core2_1.py:394`).
- **What happens.** Spearman IC is computed between E and the net label. The gross outcome is almost two-valued: 58–66% of rows are exactly −1R (stop) and 31–39% exactly +1.8R (target). Inside each of those tie groups, the rank of the net label is set only by the cost term, pen = 0.14/risk_pct, which is known at entry.
- **Evidence.**
  - 15m: E vs net label = +0.25 / +0.28, but E vs gross = +0.02 / +0.01. The partial IC given pen is +0.03 / −0.01. The cost term alone (−pen) scores +0.47 / +0.53, higher than the model.
  - 5m: E vs gross is about 0.00. 1h: −0.05 / 0.00.
  - Directional IC (E_L−E_S vs yL−yS): +0.03 (5m), +0.02 (15m), −0.03 (1h), −0.03 (4h), −0.20 (1d). The 1d IC is negative on gross outcomes too.
- **What it affects.** How the report is read. adopt_checks does not use IC, so no trade statistic changes.
- **Fix.** Report IC against the gross label (taken from the bracket or rounded), against the directional label, and as a partial IC given pen. Do not rebuild gross as float32 y + pen: that produced the finder's false negative gross ICs at 5m and 15m.
- **Could fixing it turn a negative result positive?** No. It only removes a false impression of skill.

**A2. Validation outcomes use bars from the sealed 2026 holdout, and the report says they do not.** Findings DATA-5, labels-3 and model-4, confirmed. Location: `bot/core2_1.py:349` (`fut_end = w1 + LABEL_TAIL_BARS*bar`, with LABEL_TAIL_BARS = 45 at `core2.py:66`). The same mechanism exists at `core2.py:576`.
- **Evidence.** Labels that depend on bars at or after 2026-01-01:

  | TF | Labels | v2.1 trades exiting in 2026 | Rule trades exiting in 2026 |
  |---|---|---|---|
  | 1d | 163 / 920 (17.7%) | 5 / 112 (3 of them enter at or after the 2026-01-01 open) | 4 / 82 |
  | 4h | 63 / 5520 | 5 / 591 | 1 / 402 |
  | 1h | 44 / 22080 | 1 / 850 | 2 / 1548 |
  | 15m | 106 / 88320 | 0 / 190 | 2 / 6204 |
  | 5m | 161 / 264960 | — | 4 / 15930 |

  At most 28 days of 2026 feed any label, and at most 14 days feed any trade. The report nevertheless writes `holdout_touched=False` and "no statistic computed on 2026-01..08". Core v2 has the same mechanism: 52 of 1825 of its 1d IC labels use July–August 2025, which became v2.1's validation window.
- **What it affects, and by how much.** With bars cut at 2026-01-01:
  - 1d: v2.1 −0.184 (n=107, CI [−0.40, 0.07]); rule +0.191 (n=78).
  - 4h: v2.1 −0.111 (n=586, CI [−0.21, −0.006]); rule +0.032.
  - 1h: v2.1 −0.136 and rule −0.172. 15m: v2.1 unchanged at −0.002.
  - A bound that uses no 2026 data puts the 1d v2.1 mean at −0.054 or lower, even if every affected trade were the worst possible.
- **Fix.** Set fut_end = w1 for validation, or start the holdout at w1 + 41 bars. Record the choice in the next pre-registration and correct the report text.
- **Could fixing it turn a negative result positive?** No, and no verdict changes. The damage is to holdout integrity.

**A3. The label contains a known cost term that the features cannot see.** labels-2 (partially), model-2 (confirmed). Location: `bot/core_feats.py:592`.
- **What happens.** The label is gross − 0.14/risk_pct, and risk_pct is known at entry. No feature exposes absolute volatility or coin identity; "coin id" is listed as not_used in the pre-registration. The model recovers only about 40% of the cost term: predictions slope −0.35 to −0.46 against cost, and a model fitted directly to the cost term reaches R² ≈ 0.48 with slope ≈ 0.40. The term is identical for long and short, so it only affects the E>0 threshold, not the choice between long and short.
- **Fix tested on the validation window:** fit gross R, then subtract the known cost.
  - 15m: −0.002 → −0.081 (n 190 → 505)
  - 1h: −0.136 → −0.037 (n 849 → 919)
  - 4h: −0.111 → −0.139
  - 5m: 0 trades → 35 trades at −0.221

  Adding log(risk_pct) and a coin id instead gives 1h −0.112 and 4h −0.143. Gross rank IC stays at about 0 either way.
- **Could fixing it turn a negative result positive?** No. The direction of change is inconsistent and every variant stays negative. This is a design flaw of low severity.

**A4. The KuCoin funding source changes regime on 2023-10-18.** DATA-1, confirmed, severity lowered to low. Not a code bug. Location: `bot/core_feats.py:454`.
- **What happens.** Before the break, 31–64% of settled rates are exactly 1e-4. After it, 0–0.9% are. About 51% of v2.1 training rows come from before the break. kfund_bp is a top-6 gain feature from 5m to 4h. At 4h, splits at 0.975 and 1.005 carry 30–54% of kfund_bp's split gain.
- **Counterfactual runs:**
  - 15m: −0.002 → −0.033 (pre-break rows set to NaN) or −0.002 (feature dropped).
  - 1h: −0.136 → −0.080 (NaN), −0.169 (dropped) or −0.142 (post-break rows only).
  - 4h: −0.111 → −0.131, −0.079 or −0.086.

  All are within about 1.2 SE, all stay at or below zero, and the shifts do not agree in direction.
- **Could fixing it turn a negative result positive?** No. Record it as a data caveat for future specs.

**A5. Zero-volume outage bars become extreme feature values.** DATA-4, feat_ind-1 and feat_flow-2, confirmed. Location: `bot/core_feats.py:375, 383, 392, 393`.
- **What happens.** During the Binance spot halt on 2023-03-24, bars have volume, trade count and taker-buy volume all at 0. The code turns these into log(1e-12) = −27.6 (for vol_z and tsize_z) or into 0 (for stbr1_z and ntrades_z).
  - At the halt bar itself, vol_z is about −12 to −78 and tsize_z about −100 to −192.
  - The bad value then stays in the z-score window for 7 days (5m), 10 days (15m) or 30 days (1h), compressing vol_z 2–5× and tsize_z about 10×.
  - It also pushes rv_ratio to −21 on 3 rows at 5m.
- **Scope.** Training rows only: 2.35% at 1h, 0.78% at 15m, 0.55% at 5m. There are none at 4h or 1d, and the validation windows are clean. Live data behaves the same as history.
- **Counterfactual runs.**
  - 1h moves from −0.136 to between −0.111 and −0.118, depending on the run. With a second seed it moves from −0.133 to −0.117.
  - 15m moves from −0.002 to −0.031 in two runs and to +0.004 in a third.
  - These moves are within noise and do not agree in direction.
- **Could fixing it turn a negative result positive?** No. The fix is to treat v==0 or n==0 bars as missing, and it needs a new pre-registration because the feature hash is pinned.

**A6. History charges only the fixed 0.14% cost: no perpetual funding and no stop slippage.** RULE-6 (partially), labels-4 (confirmed). Locations: `bot/decision.py:535`, `core2.py:371`, `core_feats.py:578`.
- **What happens.** Leaving funding out matches the pre-registration ("net R after 0.14% round-trip") but contradicts the `costs.py` docstring. The paper simulator charges more: a tier cost of 0.08/0.11/0.18/0.30, stop slippage of 3/5/10/20 bps, and funding.
- **Funding per rule trade:** −0.024R at 1d (longs −0.047R), −0.005R at 4h, −0.003R at 1h. v2.1 pays almost none (1d −0.0005R).
- **Validation effect:**
  - Rule 1d: +0.191 → +0.173. Rule 4h: +0.032 → +0.025. v2.1 moves by 0.001R or less.
  - Under paper's full cost model the change is −0.016 to +0.005R per trade. TRX is 0.05–0.07R worse and BTC 0.02–0.03R better at 4h.
  - In the older v2 window, rule 1d moves from −0.016 to −0.043. v2's own +0.08 could not be recomputed because its OOS file is not in the repository.
- **Could fixing it turn a negative result positive?** No. It narrows the rule's lead over v2.1 by about 0.018R at 1d.

**A7. Calibration of the rule's volume component (s_vol).** RULE-4 (confirmed), RULE-5 / feat_ind-4 (partially). Location: `bot/engine.py:212–214`.
- **RULE-4.** The term `0.3*clip(rvol−1,−1,1)*sign(c−o)` reverses the candle's direction whenever volume is below average (rvol<1), which is 57–64% of bars. The Pine source (`ctp-pro.pine:283-290`) only gates on rvol. Gating it instead changes rule net R by +0.001 to +0.010, and every confidence interval includes 0.
- **RULE-5.** The OBV part is mathematically capped at ±0.520, so s_vol stays within ±0.664 and the ±1 clip never binds. This is inherited from `crypto-trend-predictor.pine`. Rescaling moves the rule by 0.05R or less, within one SE: 1d +0.191 → +0.145, 4h +0.032 → +0.044, 1h −0.172 → −0.165, 15m −0.296 → −0.293. It would also make the volume vote fire on 75% of bars instead of 46%.
- **Could fixing it turn a negative result positive?** No.

**A8. KuCoin moved its funding schedule on 2025-06-17, and one stated drop reason is wrong.** DATA-2 (partially), feat_flow-3 and feat_flow-4 (confirmed). Locations: `bot/core_feats.py:458–459` and `:84`.
- **What happens.** Settlements moved from 04/12/20 to 00/08/16 UTC (BNB on 2025-06-10). The mapping from hour to min_to_fund flipped at that point. Keeping the old mapping stable changes v2.1 by 0.000R (4h) to −0.022R (1h); 5m still makes 0 trades.
- **Documentation error.** `DROPPED['1d']['min_to_fund']` says the feature is "always 0.5". After 2025-06-17 it is 1.0 at every close. The drop itself is harmless and was decided on the v2 discovery window, where the feature really was constant. The wrong text is copied into both validation reports.
- **Could fixing it turn a negative result positive?** No. Correct the text.

**A9. BTC rows are identifiable from the features.** feat_flow-5, confirmed (info). Location: `core_feats.py:484–486, 494`.
- rs_btc_m is exactly 0 on every BTC row, which gives the trees a BTC indicator even though the pre-registration excludes coin id.
- Masking it moves v2.1 from −0.18 to −0.17 (1d), −0.11 to −0.13 (4h) and −0.14 to −0.11 (1h). A seed change alone gives −0.13, −0.12 and −0.13.
- The suggested fix (NaN on BTC rows) would still single out BTC.
- **Could fixing it turn a negative result positive?** No.

**A10. The funding cut drops the settlement at the last bar's close.** model-5, confirmed. Location: `bot/core2.py:246` (keeps t < spot_end, while `_kfund` uses settlements ≤ close).
- This affects the last row of each window only.
- Exactly one v2.1 decision changes: 4h BTCUSDT at 2025-12-31 20:00 goes from wait to long. That trade's result lies in 2026 and was not computed. The effect is at most ±0.003R.
- Fix: `load_kfund(s, spot_end + 1)`.
- **Could fixing it turn a negative result positive?** No.

**A11. setup_code drops the setup's direction.** feat_ind-2, partially. Location: `core_feats.py:270`.
- The feature keeps only the setup type and is fed to LightGBM as an ordinal number. The model makes 0 splits on it at 1d. At 4h and 1h, every direction-aware encoding stays within seed noise.
- The direction is also recoverable from other features: votes_net, s_trend and z agree with it 85–93% of the time.
- The hypothesis that this explains why v2 falls short of the rule is refuted.
- **Could fixing it turn a negative result positive?** No.

**A12. Corrupt futures bars from Binance's monthly archive.** DATA-3, confirmed. Location: `tools/fetch_core_history.py:463` (`_fill`).
- The monthly zip for BTC November 2023 contains flat, zero-volume bars on 2023-11-10 (20 bars at 5m, 7 at 15m). The daily zip for that day is correct.
- `_fill()` only re-fetches days with missing rows, and nothing acts on `crosscheck()`'s output, so the bad data is kept.
- It changes 79 BTC 5m labels and 136 BTC 15m labels, all in training data. The other flat runs in the archive are real futures halts.
- **Could fixing it turn a negative result positive?** No.

**A13. vwap_dist and ichi_cloud saturate at their clips.** feat_ind-3, partially (info).
- 18% of 5m rows sit at the ±8 clip. ichi_cloud sits at its ±5 clip on 11–13% of rows at every timeframe.
- An unclipped refit has rank correlation 0.99 with the original and still produces no E>0 at 5m.
- **Could fixing it turn a negative result positive?** No.

**A14. Label ATR warm-up in January 2022.** Found during the combos-5 verification.
- The futures ATR is seeded at 2022-01-01. For the first ~40 daily bars, label ATR is 13–50% too small.
- This affects about 150 of roughly 7,200 1d training rows and no validation rows.
- It inflated one exploratory statistic (see Section 4). It does not change any result.

### 2B. Live, ledger and display bugs

None of these changes any v2, v2.1 or rule validation number, so fixing them cannot turn a research result positive. They affect what the user sees and how the live models behave. Current state: `gates.json` has `allowed_combos=[]` and `live_allowed=false`, and `calib.json` is not in the repository, so several of these paths are dormant or latent here.

**B1. The bracket win probability (p_win) is shown as a calibrated "probability of rise" (p_up).** display-1 and calib-F5, confirmed. Location: `bot/engine.py:925–927`.
- **When it happens.** When no trusted direction model exists but the setup model, or a trusted policy, supplies p_win.
- **What happens.**
  - Longs show p_up = p_win and shorts show 100 − p_win, marked "calibrated ✓".
  - Bracket base win rates are 34–41%, so a typical long reads bearish (about 36–40) and a typical short reads bullish (about 59–64).
  - The advisor then issues close_now whenever p_win is below about 43.4 (urgency 3 at 40 or below). The autotrader closes on urgency ≥2 when p_calibrated is true.
  - When the action policy supplies the side, p_up is the heuristic forecast but is still marked calibrated.
- **Status.** Dormant in the repository and in every calib build the auditors inspected. Whether it is active on the user's machine is unknown.
- **Fix.** Never map p_win to p_up. Show p_win separately and compare it with the bracket base rate (1/2.8), not with 50.

**B2. Live market-state features are built from a partly refreshed cache.** calib-F1, partially (reproduced end to end). Locations: `bot/main.py:79–116, 222–238`; `market.py:266–270`.
- **15m, 1h and 4h.** In every bar, BTC's klines are fresh while the other coins' are one bar stale. The dominance feature is therefore +1 on 96–99% of bars, against 4–9% in training, and ethbtc lags by one bar.
- **1d.** dom is simply the previous day's value (mean |diff| 0.40).
- **Separate mismatch.** Live uses 5 coins, below MIN_POPULATION=40, so breadth is always 0 and rs_rank always 0.5. Training uses the top 100 coins.
- **Affects.** The live calib outputs: direction p_up, the action policy, and setup p_win.
- **Fix.** Build the state at one bar timestamp from symbols that have that bar, require a minimum population for dom too, key the cache by bar, and align the live coin universe with training.

**B3. Funding in the live calib models is missing from training and misaligned in time.** calib-F4 and F3, confirmed. Locations: `bot/market.py:640`; `calib.py:1515`; `features.py:63–74`.
- **F4: little or no funding data in 4h/1d training.** Funding history is capped at 1000 rows (about 333 days), and the frozen windows are excluded from training. As a result funding_dir is dead or nearly dead in 4h and 1d training, while it is non-zero live.
  - If exactly dead, the effect is negligible.
  - The realistic case on the user's machine is tiny coverage, about 0.03–0.6% at the 2026-09-15 build and rising roughly 1% per week.
  - In that case live standardized values reach |z| 5–11. Predictions shift by |dp| 0.01–0.06, 14–64% of decision bands flip, and the out-of-distribution veto (zmax > 6) blocks the policy in about 25–65% of live states.
  - `features.health` flags a dead feature but does not stop the build, despite its docstring.
- **F3: alignment.** Training uses the settlement at or before the bar open. Live uses the latest cached settlement, up to 12h old. They differ on 11% of 1h bars, 43% of 4h bars and 92% of 1d bars. At 1d the live and training values correlate only about 0.25.
- **Fix.** Page the funding history back with startTime, align both paths to bar close, and filter live rows to the analysed bar.

**B4. A trusted action policy would take over almost every bar without a setup.** calib-F6 and LP-3, confirmed; latent. Locations: `bot/engine.py:627`; `calib.py:1403`; `decision.py:969–1032`.
- **Why it takes over.** policy_pass only checks zmax ≤ 6; there is no score threshold.
- **Reproduced with forged trust:**
  - "wait" dropped from 60% (1d) and 75% (4h) of cells to 0%.
  - The rule basis disappeared, and 7 of 27 4h rule decisions flipped direction.
  - About 85% of ledger rows became basis=policy.
- **Pooling and ranking.** live_stats pools all bases into one number. The cross-sectional rank filter the policy was validated with (at least 8 candidates) exists only in `main.overview`, and it can never pass with 5 coins.
- **Fix.** Keep policy rows out of the decision table and ledger, or split live_stats by basis.

**B5. The gold input is up to 24 hours stale.** calib-F7, confirmed. Locations: `bot/main.py:265`; `market.get_history` (HIST_TTL 86400).
- Measured on PAXG, the average correlation with the training definition is about 0.11 at 15m, 0.42 at 1h, 0.84 at 4h and 0.97 at 1d.
- **Fix.** Fetch PAXG with the same closed-bar freshness as the analysed timeframe.

**B6. The live cost adjustment in calib is mis-specified.** calib-F2, partially. Location: `bot/calib.py:790, 1095, 1377`.
- **What happens.** Models are trained net of each row's tier cost but store trained_cost_pct = 0.15. Because the models are pooled and blind to volatility, the real error is (0.15 + funding)/risk_live − C̄_train, where C̄_train is the average cost in R that the pooled model learned.
- **Size at 1h:** BTC +0.07R, ETH −0.03R, SOL −0.05R, TRX +0.15R. These are comparable to the trust thresholds (0.02–0.03R).
- **Fix.** The finder's proposal (drop the term) is wrong. Instead, store C̄_train in R units, or train on gross R and subtract the live cost explicitly.

**B7. Live decisions and the ledger use Binance spot bars; the scorecard uses Binance futures bars.** RULE-3, confirmed. Locations: `bot/market.py:302`; `decision.py:677–701`.
- **Divergence, driven by the volume vote:**
  - Signal bars differ on about 10% of bars at 1d and 4h, 13% at 1h, 18% at 15m and 20–30% at 5m.
  - Live trades missing from the scorecard's trade list: 12% at 1d, 19% at 4h, 22% at 1h, 28% at 15m, and 35–56% at 5m.
- The aggregate mean is unchanged within one SE. The ledger is settled on spot prices while the user trades KuCoin futures.
- **Fix.** Use one venue for both, or label the source.

**B8. The position advisor on the positions page uses the uncalibrated heuristic p_up.** display-2, partially. Location: `bot/main.py:1535`.
- **What happens.** It never checks `p_calibrated`, while `autotrader.py:731` does. Advice based on geometry alone has an expected value of exactly 0 and never says close_now. All non-zero EV figures and every close_now shown come from the heuristic. That covers 17–23% of hypothetical positions in a pre-2026 replay.
- **How good is the heuristic?** Its Brier score equals geometry-only (0.2123 vs 0.2125), and its information coefficient is weak (about +0.06 at 4h and 1d).
- **Fix.** Pass p_up only when p_calibrated is true.

**B9. The heuristic forecast p_up has no skill.** display-3, confirmed; this is a data check of a displayed quantity. Location: `bot/engine.py:460`.
- **Brier score:** worse than a constant 50% at 1h (0.2604), 4h (0.2620) and 1d (0.2546). At 1h and 4h this holds in every year.
- **Direction:** slightly inverted at 1h and 4h (recalibration slopes −0.10 and −0.09; the 1h inversion is borderline significant). Weakly positive but about 4× overconfident at 1d (slope +0.26).
- **Forecast target:** its sign matches the realised move 48.7% (1h), 49.1% (4h) and 52.3% (1d) of the time.
- **Where it is used.** It drives the UI lean buttons (p_up ≥52 / ≤48) and the advisor. The autotrader lean path is disabled (`allow_lean=False`).
- **Fix.** Label it as a heuristic score, not a probability, or recalibrate it.

**Remaining live-path issues, in descending order of impact:**

| ID (status) | What | Where | Live effect | Fix |
|---|---|---|---|---|
| LP-1 (confirmed) | `shadow.log_signal` has no caller outside tests, so the shadow log receives nothing | `bot/shadow.py:56` | Setup/TF suspension, edge pockets, the 30-day live-performance line and the drift warning are always empty. The UI implies an automatic brake that does not exist. | Feed these from the candidate or decision ledger, or remove the feature and its UI |
| LP-7 (confirmed) | Candidate resolution enters at the first bar in the 420-bar window, without checking it is the next bar | `bot/candidates.py:161` | 100% wrong once resolution lags ≥421 bars (>105h at 15m, >17.5 days at 1h); ≥30 bars late the result is uncorrelated with the truth. Feeds the proving report's shadow section. | Require the next bar; resolve in a background loop |
| LP-6 (partially) | Rule trades stored as setup 'zx' | `bot/candidates.py:114, 177` | 80–85% of 'zx' rows (BTC/ETH 4h, BTC 1d) are rule trades. This can move a pre-registered zx combo's shadow mean by 0.06–0.17R and flip its sign (BTC 1d: +0.06 blended vs −0.10 true z-cross). | Store under a separate 'rule' key; add setup_observed to GATE_FIELDS |
| LP-11 (confirmed) | OKX fallback: at most 299 bars; the '1D' candle is Hong Kong-aligned (opens 16:00 UTC) | `bot/market.py:310` | Binance-recorded 1d rows resolved during the fallback are closed permanently as no_data. About 15% of 1d decisions differ, 1–2% at 5m–4h, and about 4% at 1h because of OKX volume. | Use '1Dutc', paginate to 420 bars, tag the data source |
| LP-4 / LP-5 (confirmed) | 45s cache grace vs the 20s decision lag; one attempt per boundary | `bot/main.py:403, 1291`; `market.py:280` | After a cold cache, bars can be missed or the previous bar shown for a whole bar (about 8% of 5m bars down to 0.03% of 1d bars per cold event). An error or a cycle over 300s loses that bar. | Retry cells whose bar time is behind the expected bar |
| LP-8 (confirmed) | Candidate resolution race and quadratic file reads | `bot/candidates.py:139` | Duplicate results double-count n and sum. A call takes 5–20s at a few thousand rows while holding the lock that /api/overview needs. | Re-check under the lock; dedup by id; read the resolved ids once |
| LP-10 / display-5 (partially) | Each ledger uses a different cost | `bot/main.py:1106, 501` | Candidates and quick_backtest use 0.08/0.11/0.18/0.30 (0.30 when volume is missing); the scorecard and ledger use 0.14. BTC 1h looks about 0.08R/trade better (15m about 0.18R); TRX looks worse. | Use one shared cost source, or label it |
| display-7 (confirmed) | Advisor tilt factor 1.2 on p_up; the fitted value is 0.05–0.30 | `bot/advisor.py:62` | p_hit_tp is 5–6pp too high near entry; ev_hold_r shows +0.10 to +0.17R when the realised value is about 0 | Use the geometry base unless a calibrated model exists |
| display-4 (confirmed) | quick_backtest: median n=6, overlapping trades, setups only | `bot/engine.py:530` | The win rate on the analysis card carries no predictive value (Spearman about 0) | Show the scorecard, or add a confidence interval and a small-n flag |
| display-6 (confirmed) | Readiness 100% on 'wait' cells | `bot/engine.py:494` | 0.54–0.60% of waits (about 88% from rounding, 12% because the opposing-vote condition is ignored) | Floor instead of round; include the opposing-vote condition |
| RULE-2 (confirmed) | The kNN vote is presented as "AI" | `bot/engine.py:373` | Hit rate 0.500–0.506 at 5m–4h; counted in "X of 5 votes" and in the grade | Relabel it, or drop it from the count and grade |
| calib-F8, F9, F10, F11, F12, F13, F14 (confirmed) | Action-policy embargo of 30–36 bars vs 40-bar labels (`calib.py:1196`); diagnostic "cells" include frozen-window events and there is no purge (`:1826`); rounded z compared to ±0.3 (`main.py:480`); constant predictions produce a spurious lift of 0.548 (`calib.py:658`); two-build trust hysteresis missing for the setup/edge models and all status flags (`:2222, :2257`); dense labels truncated at the data end (`:1509`); per-coin sampling grid splits cross-sections (1d mean group 14.9 vs 49.7) | calib.py, main.py | Each is noise-level or diagnostic-only in the checks run | See each finding's fix |
| LP-2, LP-9 (confirmed, dormant) | A suspended setup stops the setup scan (`break`) instead of skipping it; edge_book reads the oldest rows as "recent", and its 40h bootstrap block is too short for 4h | `engine.py:621`; `edge_book.py:61` | No effect today because suspension is always empty (LP-1). If revived, live results would diverge from the scorecard by up to about 0.06R/trade. | Use `continue`; sort rows by time; scale the block by timeframe |
| feat_flow-1 (confirmed, latent) | Live KuCoin funding cache (10-minute TTL) ignores settlement times | `bot/market.py:493` | If v2.1 were deployed live, decisions would change on 0.8% (1h), 6.5% (4h) and 7.3% (1d) of bars. It has no live caller yet. | Make the cache settlement-aware before any live deployment |

---

## 3. Structural reasons

### 3.1 Cost against stop width

The stop is 1R = max(1.3×ATR14, 0.25% of entry), the target is 1.8R, and the round-trip cost is 0.14%, so cost in R is 0.14/risk_pct, capped at 0.56R.

For entries on every bar, gross R is about 0 at every timeframe. About 35% of resolved trades hit the target, which is the random-walk rate of 1/2.8 = 35.7%.

| TF | Median risk_pct (validation) | Floor binds (validation / all history) | Cost in R (validation / all history) | Break-even target-hit rate |
|---|---|---|---|---|
| 5m | 0.250% | 53% / 45% | 0.471 / 0.444 | 52.5% / 51.6% |
| 15m | 0.447% | 19% / 13% | 0.342 / 0.308 | 47.9% / 46.7% |
| 1h | 0.970% | — / 0.3% | 0.173 / 0.152 | 41.9% / 41.2% |
| 4h | — | 0 | 0.080 / 0.069 | about 38% |
| 1d | — | 0 | 0.029 / 0.025 | about 37% |

### 3.2 What this means for the rule (validation window 2025-07..12)

| TF | Rule gross R | Cost R | Rule net R | Random-side net R |
|---|---|---|---|---|
| 5m | +0.014 | 0.447 | −0.433 | −0.432 |
| 15m | +0.032 | 0.328 | −0.296 | −0.309 to −0.313 |
| 1h | +0.001 | 0.173 | −0.172 | −0.167 |
| 4h | +0.112 | 0.080 | +0.032 | −0.062 |
| 1d | +0.221 (±0.158) | — | +0.191 | −0.029 |

- **Cost explains the whole loss.** At 5m, 15m and 1h, cost is 101–111% of the net loss. Rule win rates (35.6%, 37.0%, 36.1%) sit at the 35.7% break-even before cost.
- **Cheaper execution would not rescue it.** The cost at which the rule's gross would break even is 0.0043% (5m), 0.0136% (15m) and 0.0007% (1h). At 0.10% (maker) the rule nets −0.306, −0.202 and −0.123; at 0.05% it nets −0.146, −0.085 and −0.061.
- **The rule's direction signal at 5m–1h is small and unstable across windows.** In the prior year every component had a positive, significant edge at 5m and 15m. In the validation window the 1h follow-minus-against edge was −0.143 (z alone −0.22).

### 3.3 What this means for v2.1

- **5m (0 trades).** The mean label is about −0.44 to −0.455. The highest prediction is −0.025.
  - v2.1's best picks do carry a small real gross edge: +0.05 to +0.12R in the top 1–20%. But the cost on those rows is 0.19–0.29R.
  - When overlapping trades are removed, none of the top slices is positive (5m: −0.09R, n=75).
  - With predictions shifted to a zero-cost target, v2.1's 5m rankings earn +0.052 ± 0.010 gross, against +0.008 for random (directional IC 0.027). That is exploratory, and cost buries it roughly ten times over.
- **Lower cost does not flip it.** At 0.05% v2.1 is still −0.085 (5m), −0.063 (15m) and −0.002 (1h). Only zero cost gives +0.04 to +0.05, and at 1h that comes from long drift.
- **15m.** Gross +0.201, cost 0.203, net −0.002 (n=190). About 43% (+0.125R) of v2.1's +0.294R advantage over the rule comes from trading higher-volatility bars, where cost is a smaller share of R. The short-side gross gain rests on 17 days, mostly ETH.
- **1h.** Gross +0.006, cost 0.142, net −0.136. About 86% of its +0.036R advantage over the rule is cost avoidance.

### 3.4 At 4h and 1d the loss is not cost

- **Drift is as large as the effects.** Label means swing ±0.1–0.3R between half-years with market direction.
- **4h (2025H2 per-side drift baseline: long +0.017, short −0.130).**
  - v2.1 is below the baseline on both sides: longs −0.064, shorts −0.178.
  - The rule is above it on both sides: longs +0.125, shorts −0.060.
- **1d (baseline: long −0.032, short +0.088).**
  - v2.1: longs +0.127 (n=59), shorts −0.399 (n=53).
  - Rule: longs +0.245 (n=46), shorts −0.014 (n=36).
  - An unverified exploratory check (labels-5) found always-long earned +0.157 in the same window, so the rule's 1d result is consistent with long market exposure.
- **The 1d model has no stable skill.** The 2025H2 IC is about −0.19. Half-year ICs over 2023–2025 range from −0.19 to +0.22, and there are only about 20 effective observations per half-year, so −0.19 is about 1 SE from zero.
  - Shortening the 81-bar gap to 41 bars leaves the IC unchanged (−0.202).
  - E_L and E_S correlate at −0.82, and about 60% of the prediction variance is shared across coins on each day.

### 3.5 Statistical power of the adoption test (model-6, confirmed)

The test uses a 6-month window, 27 weekly blocks, and Holm correction across 5 timeframes (each needs p ≤ 0.02), and the difference CI versus the rule must be above 0. The true edge needed for adoption:

| TF | ~50% chance of adoption | ~80% chance of adoption |
|---|---|---|
| 1h | +0.11R | +0.16R |
| 4h | +0.25R | +0.34R |
| 15m | +0.25R | +0.37R |
| 1d | about +0.55R | +0.66R |

- Under the null, false adoption happens 2% of the time or less.
- **A "no" is informative only at 1h and 4h.** v2.1's CIs there are [−0.235, −0.036] and [−0.200, +0.005]. At 15m ([−0.25, +0.26]) and 1d ([−0.36, +0.16]) a "no" is weak evidence against a moderate edge.

### 3.6 Wider stops (exploratory, data-mined, same trade sides)

- 1h net R: −0.116 (1.3×ATR, 40-bar cap) → −0.051 (2.6×/40) → −0.021 (2.6×/120) → +0.031 (5.2×/160, n=738).
- 15m: −0.266 → −0.119 → −0.093 → −0.012.
- Gross stays at about 0.03–0.07R; the cost in R is what shrinks.
- **Caveat.** Raising only R_MIN_PCT to 2.8% while keeping the 40-bar cap turns 92% of 5m trades and 75% of 15m trades into timeouts. The stop and the horizon have to change together.

### 3.7 Live calib layer: frozen windows (calib-F15, confirmed, info)

`calib.build` excludes the pre-registered frozen windows on every weekly rebuild, even after the judgement has been made:
- 1d window: [2024-09-28, 2026-08-03). 4h window: [2025-08-02 20:00, 2026-09-07 16:00).
- 4h and 1d models lose 25–28% of their lookback, including the most recent regime.
- Their trust tests currently run on data from before the windows. This fixes itself after about 7 months (4h) and 13 months (1d).
- It also leaves funding_dir effectively dead (see B3). It does not affect v2, v2.1 or rule validation.

---

## 4. The indicator-combination question

Every item in this section is **exploratory and data-mined** unless marked otherwise.

**How many comparisons were run.** The combinations audit alone ran:
- 628 feature × side × timeframe IC series (5024 half-year ICs), plus 942 more on other targets;
- 1080 trend × oscillator cells;
- about 184 rule-trade cells and 32 kNN cells;
- 2988 single-indicator filter cells.

The rule audit ran many counterfactual replays on top of that. Anything that looks positive here needs pre-registration and a forward test.

### 4.1 Short answer
Combining indicators does not make the system trade in the wrong direction.
- No component is sign-inverted in code.
- Setups and the z-rule rarely conflict: 11 of 7827 setup bars at 1h, 53 of 30995 at 15m.
- Flipping every signal makes every timeframe worse: −0.041 (5m), −0.079 (15m), −0.065 (1h), −0.163 (4h), −0.113 (1d).
- More agreeing votes gives better, not worse, net R. At 1h: 1 vote −0.306 up to 5 votes −0.060. At 4h, 5 votes give +0.120. It is not monotonic at 15m or 1d, but the slope is positive at every timeframe.
- The best re-combination of components improves net R by at most +0.004 (5m), +0.017 (15m) and +0.031 (1h), against cost gaps of 0.41, 0.27 and 0.12R.

### 4.2 Redundant clusters (confirmed)

**In the rule.**
- z is a z-scored, smoothed weighted sum of the same four components that vote (`engine.py:222`).
- RSI appears in several places: it is 60% of s_mom, it enters z through s_mom, it is 2 of the 4 kNN features, and it filters the fd and rg setups. RSI correlates 0.98 with s_mom.
- The four technical votes act as about 1.7–2.0 independent votes, about 2.3 once kNN is added.
- Whenever |z| ≥ 1.2, the technical votes alone already pass "3 of 4" on 77–82% of bars.
- So "3 of 5 votes agree" means roughly one technical consensus plus z, not five confirmations.

**In the v2 features.**
- 18–20 momentum and price-location features form one chain at |ρ| > 0.8. It is a chain, not a tight group: the weakest pair inside it is only 0.26–0.44.
- kdist and RSI14 are 0.996 rank-correlated. This is a real property of the two indicators, not a copy bug.
- cci20 and bb_pctb correlate at 0.98 (not independently verified).
- The cluster is about 31% of the features but holds only 12–18% of LightGBM gain.

**Impact.** Removing the kNN vote or the whole vote-count condition changes rule net R by 0.01R or less at 1h and 4h, and by 0.02–0.04R at 1d (within SE).

### 4.3 Noise and weak components

- **kNN "AI" vote (RULE-2, confirmed).**
  - It has no skill at 5m–4h and slight skill at 1d (0.52–0.53).
  - It is weakly contrarian to momentum and z (−0.10 to −0.15) and votes on about 78% of bars, which is what chance produces.
  - It rarely changes a decision: it supplies the third confirmation on 5–7% of rule signal bars and blocks a trade on about 0.3% of eligible bars. The claim that it acts as a veto is refuted.
- **Volume vote below average volume (RULE-4, confirmed).**
  - The inverted branch is weakly negative (4h −0.049, CI [−0.102, +0.004]).
  - The confirming branch is positive at 1h and 15m.
  - Gating it changes rule net R by +0.001 to +0.010.
- **The fd (trend-fade) and sq (squeeze) setups (RULE-7 confirmed, combos-6 partially).**
  - fd trades underperform z-rule trades in both 2022-07..2024-06 and 2024-07..2025-12 at most timeframes. fd's negative directional edge does not replicate in the earlier window.
  - A z-rule-only variant beats the current rule by 0.01–0.04R in most windows, but is 0.0075R worse at 1h in 2025H2.
  - Trades carrying an opposing vote did worse in validation at 1h (−0.20, z −1.6) and 4h (−0.42, z −2.1, n=29), but not at 15m (+0.05).
  - Validation counterfactual:

    | Variant | 5m (BTC+SOL) | 15m | 1h | 4h | 1d |
    |---|---|---|---|---|---|
    | Current rule | −0.420 | −0.296 | −0.172 | +0.032 | +0.191 |
    | Without fd and sq | −0.412 | −0.285 | −0.171 | +0.065 | +0.211 |
    | z-rule only | −0.415 | −0.279 | −0.180 | +0.063 | +0.217 |

    No verdict changes.
- **Forecast p_up components (display-8, not independently verified).** The drift term is anti-predictive at 1h and 4h. The "RSI > 70 bend" points against the mean realised return. Mixing terms that track the mean with terms that track up-frequency leaves the combined p_up with no skill (see B9).

### 4.4 Conflicting components: trend versus oscillator (combos-5, partially)

- **Longs.** In an uptrend, RSI > 70 ("conflict") beats neutral RSI at 15m–4h in the discovery period. Gross differences, month-clustered: +0.046, +0.100, +0.107. Validation point estimates are also positive but not significant.
  - 4h rule longs with RSI ≥ 70: +0.111 vs −0.019 in discovery, +0.215 vs +0.070 in validation.
  - A classic "avoid overbought" veto would therefore remove the better trades.
- **Shorts.** The same conflict pattern reversed in validation at 1h and 4h: conflict was worse than neutral in 81–92% of short cells.
- **1d, downtrend with RSI < 30, short.**
  - About −0.47 ± 0.12R after removing the January 2022 ATR warm-up rows; the published −0.55 was inflated by them.
  - It rests on 18 discovery episodes and 2 validation episodes, both in November 2025.
  - Rule 1d shorts with RSI ≤ 30: −0.492 (n=21) vs −0.037 for other shorts; in validation −0.552 (n=6) vs +0.267.
  - An in-sample "no 1d short when RSI < 30" filter moves the 1d rule from +0.058 to +0.097 (discovery) and from +0.191 to +0.253 (validation). It was chosen on the same data.
- **Pullbacks.** For s_trend × RSI, the "agree/pullback" cell is structurally empty.

### 4.5 Unstable features

- **Momentum block at 1h (combos-4, partially).**
  - The composite directional IC was −0.018 in 2025H2, with a CI that includes zero. 2023H1 looked the same. The positive discovery mean comes mostly from 2024H2.
  - This does not explain the 1h losses. The rule's 1h loss is −0.18R level and +0.012R direction. v2.1's is −0.14R level and −0.04R direction.
- **Momentum at 1d.** The momentum IC was +0.229 [+0.061, +0.383], while v2.1's 1d sides ran against momentum (corr −0.144). That gives a direction contribution of −0.242R. This is where the combination mattered most.
- **Funding features (combos-7, partially).** kfund_bp and kfund_persist have the largest per-side IC spread across half-years. kfund_persist's directional IC changes sign in validation at 1h, 4h and 1d. This comes on top of the 2023 regime break (A4).
- **Calendar features.** Their per-side ICs are stable, but they are pure volatility, and therefore cost, proxies: partial IC given risk_pct is about 0. They are not regime identifiers.
- **Sign flips are common.** 58% (1h), 27% (4h) and 64% (1d) of all features have a validation directional IC with the opposite sign to their training mean.
- **Post-hoc ablation.** Dropping funding, calendar and BTC/ETH features:
  - 1h: −0.136 → −0.110, inside the random-drop range of −0.155 to −0.108.
  - 4h: −0.111 → −0.051, still below the rule's +0.032.
  - 1d: −0.184 → +0.080 (n=108), driven by btc_z, ethbtc_z and btc_flow16_z. It beats all 30 random drops, but the feature set was chosen on the validation window, so it is data-mined and does not count as evidence.

### 4.6 Single-indicator filters (combos-8, not independently verified)

Of 2988 quintile cells, only 2 had a positive mean in discovery (6 of 7 halves) and in validation. Both are 1d longs, which is consistent with chance plus 1d drift. No single-indicator filter rescues any timeframe.

---

## 5. Disputed and refuted claims, unverified items, and what was verified correct

No finding was refuted outright. The partially confirmed ones had these sub-claims corrected:

**Data and labels**
- DATA-1: severity lowered from medium to low. Fixing it has no consistent effect.
- DATA-2: this is a regime break in the feature, not a live/history computation mismatch.
- DATA-5: the 1d share is 17.7%, not 22% (200 was an upper bound).
- labels-3: the overlap is at most 28 days for labels and 14 days for trades, not about 45 days.
- labels-1: "cost swamps any directional signal" overstates it for the rule at 5m and 1h, where there is no signal to swamp. It does hold for v2.1 at 5m. Raising only R_MIN_PCT makes 92% of 5m trades time out.
- labels-2: the "mechanical −1 slope" holds only when long and short are averaged; severity is low.
- labels-4: the finder's 1d funding figure (0.024R, label mean 0.069) came from a different window. Over 2022–2025 it is about 0.018R.

**Features**
- feat_ind-1: the 1h scope is 2.35% of training rows, not about 5%.
- feat_ind-2: the hypothesis that it explains why v2 falls short of the rule is refuted.
- feat_ind-3 and RULE-5: tree models ignore scale; the volume vote is not under-weighted (it fires on 46% of bars).
- feat_flow-3: "only about 2 post-switch weeks of training" holds only for July 2025.
- feat_flow-5: NaN on BTC rows would still identify BTC.

**Rule**
- RULE-1: the "correctly signed edge" does not hold at 1h in the validation window (−0.143). Median bars held is 6–7 at 5m, not 5 everywhere. At 1h cost is 3.9× gross, not an order of magnitude. The 0.04R re-combination bound holds only at 5m–1h.
- RULE-2: the vote rarely changes a decision.
- RULE-3: "roughly 10%" understates the mismatch at 1h and below.
- RULE-4: the anti-predictive evidence is weaker (the 4h CIs include 0).
- RULE-6: this is a cost-model limitation that matches the pre-registration, not a code bug.

**Model**
- model-1: the negative "E vs gross" values at 5m and 15m were a float32 artifact; the true value is about 0.
- model-2: "not better direction" is too strong at 15m.
- model-3: "stale because of the 81-bar gap" is refuted (the IC is unchanged at 41 bars). "Pure market timing" is not supported (the cross-sectional IC is −0.166).

**Combinations**
- combos-1: some of the finder's secondary numbers (15m gross ICs, 1d quintiles) did not reproduce.
- combos-3: the kNN vote is not a veto.
- combos-4: the 1h flip is not significant and does not explain the losses. LightGBM is not dominated by the cluster.
- combos-5: "the one harmful combination is 1d shorts" does not hold out of sample. The 1d magnitude was inflated by the ATR warm-up.
- combos-6: "the setup layer adds losing trades" reverses at 1h in validation.
- combos-7: calendar features are not regime identifiers, and the BTC features are not slow at 1h.

**Live calib**
- F1: 1d is not pinned at +1; breadth and rs_rank are always neutral live for a different reason.
- F2: the direction of the error is wrong (TRX is inflated, not deflated), and the proposed fix is wrong.
- F3: the live value is not always "at or after the bar close".
- F4: the finder's 0.021 effect is about 15× too large for the exactly-dead case; the tiny-coverage case is worse.

**Live path and display**
- LP-6: the edge_book and shadow consequences are moot because nothing writes the shadow log.
- LP-10: the live fallback cost is 0.30, not 0.15.
- LP-12: the z mismatch case is not a pure tie.
- display-2: "no skill" is overstated (IC about +0.06 at 4h and 1d).
- display-3: the 1d recalibration slope is +0.26; the 1h inversion is borderline significant.
- display-5: avg_r is not displayed, and TRX is shown more pessimistically.
- display-6: strength is not shown on wait cells.
- display-7: the "both-hit" explanation changes the result by less than 0.01R.

**Not independently verified (information only):** DATA-6 (spot/futures basis dislocations), labels-5 (naive baselines), feat_ind-5 (redundant cluster), feat_flow-6 (calendar/funding importance), combos-8 (quintile filters), calib-F16 (local 5-coin build: nothing trusted, direction model miscalibrated), calib-F17 (default vs real funding in labels), display-8 (forecast components).

**Verified correct**

*Data ingestion*
- Archive column mapping, header skip and µs→ms conversion.
- Series strictly increasing, on the bar grid, with no NaN and consistent OHLC.
- No missing futures bars 2022–2026-08.
- Spot resampling matches the native files.
- Live Binance API equals the archive on all 9 columns.
- KuCoin settlement semantics and units; interval_h.
- `_kfund` has no look-ahead; build_dataset aligns by exact timestamp.

*Labels*
- An independent brute-force bracket matches `core_feats.labels` on about 90k rows × 2 sides plus slices from other timeframes.
- ATR is causal and cost units are correct.
- Both-hit counted as a loss decides only 0.08–0.31% of labels, a bias under 0.005R.
- Gap stops occur on 0.02% of rows or fewer.
- Timeouts, short-side symmetry and purge semantics are correct.
- Stored validation labels equal a fresh recomputation.
- Patient-limit entries show no significant adverse selection.

*Features*
- Truncation tests show zero differences at every timeframe.
- All 38 G1/G2 indicators match reference implementations (difference ≤6e-9).
- No chikou span; session VWAP is correct; feature signs are correct.
- CAUSAL_START covers the warm-up contamination.
- G3, G5 and G6 re-implemented independently and matched.
- Live/history tail parity holds.

*Rule*
- Component signs; no cancellation between setups and the z-rule.
- Replay has no off-by-one error.
- The kNN vote has no leakage.
- The live 420-bar window matches full-history replay (at most 4 mismatches in 4292).
- z is causal, and neutral values are never counted as votes.

*Model*
- X and labels are bit-identical to independent computation.
- The walk-forward refit reproduces the saved predictions exactly.
- The train purge and embargo, coin weights, Holm, block bootstrap, decide/evaluate counts and adopt_checks all reproduce.
- The power check produces no false positives.

*Calib*
- extract_events and its maps are causal.
- Walk-forward folds share no timestamps; rows are sorted before splitting.
- Normalisation uses training rows only.
- Frozen-window rows are excluded from model training.
- The live path equals training when caches are fresh.

*Live path*
- The live decision ledger and `decision.replay` agree on the same bars: mean R differs by at most 0.003R, and matched trades are identical.
- Tradeable, suspension and combo flags never change side, entry or levels.
- The scheduler observes every bar in steady state.

*Display*
- The forecast code has no sign errors and depends only negligibly on window length.
- Trade levels equal the bracket contract.
- The displayed entry differs from the next open only at tick level.
- The advisor's driftless base probability is well calibrated.

---

## 6. Recommended next steps

The v2 validation window (2024-07..2025-06) and the v2.1 validation window (2025-07..2025-12) are now used. Nothing found in this audit is evidence for adoption. Any fixed or new model must be judged on forward data collected after a fresh pre-registration, never on a rerun of 2022–2025.

The 2026 holdout stays sealed. Do not open it to test fixes, even though the validation tails already touched up to 28 days of January 2026 at 1d (A2).

1. **Research hygiene, safe to do now (no re-judging):**
   - Set fut_end = w1 for validation, or put a 41-bar gap before the holdout. Correct the `holdout_touched` wording.
   - Report IC against rounded gross, against the directional label, and as a partial IC given cost.
   - Correct the 1d `min_to_fund` drop reason.
   - Load funding with t ≤ spot_end.
   - Treat zero-volume bars (and rv_ratio on them) as missing.
   - Make `_fill` / `crosscheck` re-fetch the daily zip when a day fails, and mask entries on v==0 bars.
   - Show always-long, always-short and random-side baselines, plus a separate funding line, next to every validation result.
2. **Decide the bracket and timeframe question first, in the new pre-registration:**
   - Under 0.14% taker cost with a 1.3×ATR / 0.25% stop, 5m and 15m, and very likely 1h, cannot be profitable. Even 0.05% cost leaves the rule negative there.
   - Either drop them, or pre-register a wider stop together with a longer horizon (not R_MIN_PCT alone), or maker execution.
   - Treat the wider-stop results in 3.6 as hypotheses only.
3. **If a v2.2 model spec is written:**
   - Model gross R or the win probability, and subtract the known cost at decision time. Tested here, this did not create an edge, so do not expect it to.
   - Build funding features only from data after 2023-10-18, or with a regime-invariant transform. Drop min_to_fund. Check every feature for exchange schedule changes before fitting.
   - Collapse the momentum cluster to 1–2 representatives. Drop the calendar features or residualize them on volatility.
   - Remove the BTC identity channel by dropping or substituting the btc_* and rs_btc_m features; NaN is not enough.
   - Plan for power: pre-register a longer forward window or a test pooled across timeframes, because 6 months only detects large edges.
   - The data-mined 1d ablation (+0.080 without the BTC/ETH features) may be listed as one pre-registered hypothesis, judged on forward data only.
4. **Rule variants for a pre-registered forward test.** Keep the list short to limit multiple testing:
   - z-rule only (no fd and sq setups) against the current rule;
   - at 1d, no short when RSI14 < 30.

   Do not add overbought or oversold vetoes on longs. Stop presenting kNN as "AI", or remove it from the vote count and the grade.
5. **Live-layer fixes.** These are not research changes and can be made now. Suggested order:
   - Separate p_win from p_up (B1).
   - Make the advisor check p_calibrated and reduce its tilt (B8, display-7).
   - Label the forecast p_up as a heuristic (B9).
   - Build market state at the bar timestamp and align the live coin universe with training (B2).
   - Page and align funding history (B3).
   - Fix gold freshness (B5) and the calib cost adjustment (B6).
   - Keep policy rows out of the ledger, or split live_stats by basis (B4).
   - Use one venue for decisions, ledger and scorecard (B7).
   - In the candidate ledger: next-bar check, race fix, a separate 'rule' key (LP-7, LP-8, LP-6).
   - OKX fallback: use '1Dutc', 420 bars, and tag the source (LP-11).
   - Retry missed scheduler bars (LP-4, LP-5).
   - Either revive the suspension feedback (using `continue`, and replaying it in the scorecard so live results stay comparable) or remove its UI (LP-1, LP-2).
   - The calib hygiene items F8–F14.
   - Make the funding cache settlement-aware before any live v2 deployment (feat_flow-1).
   - Re-register or stop excluding frozen windows that have already been judged in calib (F15).
6. **Keep the forward decision ledger clean.** It is the only uncontaminated test the rule still has. Split it by basis, use one venue, and do not lose bars, so it can judge the pre-registered rule variants later.