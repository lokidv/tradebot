# -*- coding: utf-8 -*-
"""گزارشِ دوره‌ای و ارزیابیِ گیت‌های مرحله‌ای — داورِ اثباتِ زنده (فاز ۳ و ۴).

آزمونِ منجمد فقط می‌گوید «در گذشته لبه بود». این ماژول می‌پرسد «پس از ثبت، در
شرایطِ زنده هم هست؟» و پاسخ را با همان معیارِ آماری (بوت‌استرپِ بلوکی، n مؤثر)
و اعدادی که در پلن پیش‌ثبت شده‌اند می‌دهد.

مرحله‌ها (بخشِ ۵ پلن) — هر کدام فقط اگر قبلی عبور کرده باشد:

۱. **سایه** — همهٔ کاندیدهای ترکیب‌های مجاز، دست‌کم ۹۰ روز و ۲۰۰ ردیف.
۲. **تست‌نت** — دست‌کم ۶۰ روز و ۱۰۰ پرشدنِ واقعی؛ لغزش، نرخِ پرشدن، اختلافِ
   شبیه‌ساز با صرافی، و **adverse selection** (آیا سفارش‌های پرشده بدتر از
   منقضی‌شده‌ها بودند؟).
۳. **یکپارچگیِ داده و عملیات** — صفر ردیفِ مسموم، سلامتِ سبز در ۹۹٪ نمونه‌ها.

هر تغییرِ گیت در طولِ دورهٔ اثبات، ساعت را صفر می‌کند.
"""
import json
import os
import time

import numpy as np

import bracket
import candidates
import gates
import journal
import paths
import stats

DATA_DIR = paths.DATA_DIR
REPORT_DIR = os.path.join(DATA_DIR, "reports")
HEALTH_SAMPLES = os.path.join(DATA_DIR, "health_samples.jsonl")
DAY = 86400.0

SHADOW_GATES = {
    "min_days": 90, "min_rows": 200, "lcb_alpha": 0.10,
    "min_mean_net_r": 0.10, "min_win_rate": 40.0,
    "min_brier_skill": 0.01, "max_gap_vs_frozen_r": 0.15,
}
TESTNET_GATES = {
    "min_days": 60, "min_fills": 100, "min_fill_rate": 0.60,
    "max_median_slip_pct": 0.05, "max_median_slip_pct_thin": 0.10,
    "max_sim_vs_exchange_gap": 0.20, "max_adverse_selection_gap_r": 0.10,
}
OPS_GATES = {"min_days": 30, "min_green_fraction": 0.99, "max_unhandled_errors": 0}
LIVE_SCALE = {
    "min_days": 60, "min_trades": 60, "lcb_alpha": 0.10, "min_pf": 1.10,
    "max_dd_r": 6.0, "step_up_risk_pct": 0.50, "base_risk_pct": 0.25,
    "demote_lcb_r": -0.10,
}


# ───────────────────────── نمونه‌برداریِ سلامت ─────────────────────────
def record_health_sample(status, problems=None, now=None):
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(HEALTH_SAMPLES, "a", encoding="utf-8") as f:
        f.write(json.dumps({"ts": now if now is not None else time.time(),
                            "status": status, "problems": problems or []},
                           ensure_ascii=False) + "\n")


def health_green_fraction(days=30, now=None):
    now = now if now is not None else time.time()
    if not os.path.exists(HEALTH_SAMPLES):
        return None, 0
    rows = []
    with open(HEALTH_SAMPLES, "r", encoding="utf-8") as f:
        for line in f:
            try:
                r = json.loads(line)
            except ValueError:
                continue
            if r.get("ts", 0) >= now - days * DAY:
                rows.append(r)
    if not rows:
        return None, 0
    return round(sum(1 for r in rows if r.get("status") == "green") / len(rows), 4), len(rows)


# ───────────────────────── کالیبراسیونِ احتمال ─────────────────────────
def brier_skill(p_pct, won):
    """مهارتِ احتمال نسبت به پیش‌بینیِ ثابتِ نرخِ پایه. مثبت یعنی بهتر از هیچ‌چیز."""
    p = np.asarray(p_pct, float) / 100.0
    y = np.asarray(won, float)
    if len(p) < 20:
        return None
    base = float(y.mean())
    ref = base * (1 - base)
    if ref <= 1e-12:
        return None
    return round(1.0 - float(np.mean((p - y) ** 2)) / ref, 4)


# ───────────────────────── سایه ─────────────────────────
def _candidate_index():
    cands = {c["id"]: c for c in candidates._read(candidates.CAND_PATH) if "id" in c}
    results = {r["id"]: r for r in candidates._read(candidates.RESULT_PATH)
               if "id" in r and r.get("net_r") is not None}
    return cands, results


def shadow_section(since_ts, allowed, symbols, now=None):
    """کارنامهٔ سایهٔ **همهٔ** کاندیدهای ترکیب‌های مجاز از لحظهٔ ثبت به بعد."""
    cands, results = _candidate_index()
    allowed = set(allowed or [])
    symbols = set(symbols or [])
    rows = []
    for cid, res in results.items():
        c = cands.get(cid) or {}
        key = gates.combo_key(res.get("tf"), res.get("setup"), res.get("side"))
        if allowed and key not in allowed:
            continue
        if symbols and res.get("symbol") not in symbols:
            continue
        if float(res.get("candle_ts") or 0) < since_ts * 1000:
            continue
        rows.append((res, c))
    vals = [float(r["net_r"]) for r, _c in rows]
    stamps = [float(r["candle_ts"]) for r, _c in rows]
    block = bracket.MAX_BARS * 4 * 3600 * 1000
    summ = stats.summarize(vals, stamps, block, alpha=SHADOW_GATES["lcb_alpha"], B=800)
    p_rows = [(float(c["p_win"]), float(r["net_r"]) > 0) for r, c in rows
              if c.get("p_win") is not None]
    summ["brier_skill"] = brier_skill([p for p, _ in p_rows], [w for _, w in p_rows]) if p_rows else None
    summ["days_observed"] = round((now or time.time()) / 1.0 - since_ts, 1) / DAY if since_ts else 0
    summ["days_observed"] = round(summ["days_observed"], 1)
    by_combo = {}
    for r, _c in rows:
        key = gates.combo_key(r.get("tf"), r.get("setup"), r.get("side"))
        by_combo.setdefault(key, []).append(float(r["net_r"]))
    summ["by_combo"] = {k: {"n": len(v), "mean": round(float(np.mean(v)), 4)}
                        for k, v in by_combo.items()}
    return summ


def evaluate_shadow(shadow, frozen_mean=None, gates_=SHADOW_GATES):
    reasons = []
    if (shadow.get("days_observed") or 0) < gates_["min_days"]:
        reasons.append(f"روزهای مشاهده {shadow.get('days_observed')} < {gates_['min_days']}")
    if (shadow.get("n") or 0) < gates_["min_rows"]:
        reasons.append(f"ردیف‌ها {shadow.get('n')} < {gates_['min_rows']}")
    if shadow.get("lcb") is None or shadow["lcb"] <= 0:
        reasons.append(f"LCB90={shadow.get('lcb')} ≤ 0 یا نامعلوم")
    if shadow.get("mean") is None or shadow["mean"] < gates_["min_mean_net_r"]:
        reasons.append(f"میانگین {shadow.get('mean')} < {gates_['min_mean_net_r']}")
    if shadow.get("win_rate") is None or shadow["win_rate"] < gates_["min_win_rate"]:
        reasons.append(f"وین‌ریت {shadow.get('win_rate')} < {gates_['min_win_rate']}")
    if shadow.get("brier_skill") is None or shadow["brier_skill"] < gates_["min_brier_skill"]:
        reasons.append(f"Brier skill {shadow.get('brier_skill')} < {gates_['min_brier_skill']}")
    if frozen_mean is not None and shadow.get("mean") is not None:
        gap = abs(float(shadow["mean"]) - float(frozen_mean))
        if gap > gates_["max_gap_vs_frozen_r"]:
            reasons.append(f"فاصله با آزمونِ منجمد {gap:.2f}R > {gates_['max_gap_vs_frozen_r']}R")
    return {"pass": not reasons, "reasons": reasons}


# ───────────────────────── اجرا / تست‌نت ─────────────────────────
def execution_section(since_ts):
    """کیفیتِ پرشدن و adverse selection از ژورنال."""
    evs = journal.events(since=since_ts)
    placed = [e for e in evs if e.get("kind") == journal.ORDER_PLACED]
    filled = [e for e in evs if e.get("kind") == journal.ORDER_FILLED]
    expired = [e for e in evs if e.get("kind") == journal.ORDER_EXPIRED]
    cancelled = [e for e in evs if e.get("kind") == journal.ORDER_CANCELLED]
    n_placed = len(placed)
    fill_rate = round(len(filled) / n_placed, 4) if n_placed else None
    slips = [abs(float(e["slip_r"])) for e in filled if e.get("slip_r") is not None]
    # adverse selection: سفارش‌های پرشده در برابرِ منقضی‌شده، با نتیجهٔ **براکتِ واحد**
    _cands, results = _candidate_index()
    outcome = {}
    for e in placed:
        o = e.get("order") or {}
        cid = candidates.candidate_id(e.get("symbol"), o.get("tf"), o.get("side"),
                                      o.get("signal_ts") or 0) if o.get("signal_ts") else None
        if cid:
            outcome[(e.get("symbol"), e.get("seq"))] = cid
    def _later(kind_events, sym, seq):
        return any(k.get("symbol") == sym and (k.get("seq") or 0) > seq for k in kind_events)
    filled_r, expired_r = [], []
    for (sym, seq), cid in outcome.items():
        res = results.get(cid)
        if not res:
            continue
        if _later(filled, sym, seq):
            filled_r.append(float(res["net_r"]))
        elif _later(expired, sym, seq):
            expired_r.append(float(res["net_r"]))
    gap = (round(float(np.mean(expired_r)) - float(np.mean(filled_r)), 4)
           if filled_r and expired_r else None)
    return {
        "orders_placed": n_placed, "filled": len(filled), "expired": len(expired),
        "cancelled": len(cancelled), "fill_rate": fill_rate,
        "median_slip_r": round(float(np.median(slips)), 4) if slips else None,
        "adverse_selection": {
            "filled_n": len(filled_r), "expired_n": len(expired_r),
            "filled_mean_r": round(float(np.mean(filled_r)), 4) if filled_r else None,
            "expired_mean_r": round(float(np.mean(expired_r)), 4) if expired_r else None,
            # مثبت = منقضی‌ها بهتر بودند ⇒ لیمیتِ صبور سیگنال‌های خوب را از دست می‌دهد
            "gap_expired_minus_filled_r": gap,
        },
    }


def evaluate_testnet(execution, days, sim_vs_exchange_gap=None, gates_=TESTNET_GATES):
    reasons = []
    if days < gates_["min_days"]:
        reasons.append(f"روزهای تست‌نت {days:.0f} < {gates_['min_days']}")
    if (execution.get("filled") or 0) < gates_["min_fills"]:
        reasons.append(f"پرشدن‌ها {execution.get('filled')} < {gates_['min_fills']}")
    fr = execution.get("fill_rate")
    if fr is None or fr < gates_["min_fill_rate"]:
        reasons.append(f"نرخِ پرشدن {fr} < {gates_['min_fill_rate']}")
    gap = (execution.get("adverse_selection") or {}).get("gap_expired_minus_filled_r")
    if gap is None:
        reasons.append("adverse selection هنوز قابلِ سنجش نیست")
    elif gap >= gates_["max_adverse_selection_gap_r"]:
        reasons.append(f"سفارش‌های منقضی {gap:.2f}R بهتر از پرشده‌ها بودند — "
                       "ورودِ صبور سیگنال‌های خوب را از دست می‌دهد")
    if sim_vs_exchange_gap is None:
        reasons.append("اختلافِ شبیه‌ساز با صرافی هنوز سنجیده نشده")
    elif sim_vs_exchange_gap > gates_["max_sim_vs_exchange_gap"]:
        reasons.append(f"اختلافِ شبیه‌ساز/صرافی {sim_vs_exchange_gap:.0%} > "
                       f"{gates_['max_sim_vs_exchange_gap']:.0%}")
    return {"pass": not reasons, "reasons": reasons}


def sim_vs_exchange_gap(closed_positions):
    """اختلافِ نسبیِ سود/زیانِ ناخالصِ شبیه‌ساز با صرافی روی معامله‌های تست‌نت.

    هر پوزیشنِ تست‌نت هم PnLِ صرافی دارد و هم PnLِ محاسبه‌شده از قیمت‌ها.
    """
    pairs = [(float(p.get("gross_pnl_pct") or 0) * float(p.get("size_usdt") or 0) / 100,
              float(p.get("pnl_usdt") or 0))
             for p in closed_positions if p.get("mode") == "testnet"]
    if len(pairs) < 10:
        return None
    sim = sum(a for a, _b in pairs)
    exch = sum(b for _a, b in pairs)
    denom = max(abs(sim), abs(exch), 1e-9)
    return round(abs(sim - exch) / denom, 4)


# ───────────────────────── یکپارچگیِ داده و عملیات ─────────────────────────
def integrity_section():
    cands = candidates._read(candidates.CAND_PATH)
    results = candidates._read(candidates.RESULT_PATH)
    zero_risk = sum(1 for c in cands if float(c.get("risk_pct") or 0) < candidates.MIN_RISK_PCT)
    extreme = sum(1 for r in results if r.get("net_r") is not None and abs(float(r["net_r"])) > 5)
    stale = sum(1 for c in cands if (float(c.get("logged_at") or 0) * 1000 - float(c.get("candle_ts") or 0))
                > candidates.MAX_CANDLE_AGE_BARS * candidates.TF_MINUTES.get(c.get("tf"), 60) * 60000)
    return {"zero_risk_rows": zero_risk, "extreme_r_rows": extreme, "stale_candle_rows": stale,
            "ok": zero_risk == 0 and extreme == 0 and stale == 0}


def evaluate_ops(now=None, gates_=OPS_GATES):
    frac, n = health_green_fraction(gates_["min_days"], now=now)
    reasons = []
    if frac is None:
        reasons.append("هیچ نمونهٔ سلامتی ثبت نشده")
    elif frac < gates_["min_green_fraction"]:
        reasons.append(f"سلامتِ سبز {frac:.1%} < {gates_['min_green_fraction']:.0%} ({n} نمونه)")
    return {"pass": not reasons, "reasons": reasons, "green_fraction": frac, "samples": n}


def gate_changes_since(ts):
    """همهٔ رویدادهای تغییرِ گیت از ``ts`` به بعد."""
    return [e for e in journal.events(kinds=[journal.GATE_CHANGE], since=ts)]


def _gate_action(ev):
    """نوعِ تغییرِ گیت. رویدادهای پیش از فیلدِ ``action`` از متنِ reason خوانده می‌شوند."""
    if ev.get("action"):
        return ev["action"]
    reason = str(ev.get("reason") or "")
    for prefix, action in (("preregistered trial", "preregister"), ("judged prereg", "judge"),
                           ("KILL:", "kill")):
        if reason.startswith(prefix):
            return action
    return "other"


def _event_hash(ev):
    if ev.get("prereg_hash"):
        return str(ev["prereg_hash"])
    reason = str(ev.get("reason") or "")
    return reason.split()[-1] if reason.startswith("judged prereg ") else None


def gate_changes_during_proving(prereg_hash, since_ts, judged=False):
    """تغییرهایی که شواهدِ دورهٔ اثبات را باطل می‌کنند — ساعت از نو.

    پیش‌ثبتِ همین آزمایش و **یک** نوشتنِ حکمش خودِ پروتکل‌اند، نه دست‌کاری؛ پلهٔ
    اندازهٔ ریسک (scale) هم فهرستِ مجاز را عوض نمی‌کند. هر چیزِ دیگر — قطعِ اضطراری،
    پیش‌ثبتِ تازه، نوشتنِ فهرستِ مجاز بدونِ حکمِ ثبت‌شده، تغییرِ ناشناخته — شمرده می‌شود.
    بدونِ این تفکیک، خودِ حکم «تغییر در دورهٔ اثبات» شمرده می‌شد و مرحلهٔ سایه
    هیچ‌وقت نمی‌توانست قبول شود.
    """
    out, judgement_seen = [], False
    for e in gate_changes_since(since_ts):
        action, h = _gate_action(e), _event_hash(e)
        ours = bool(h and prereg_hash and str(prereg_hash).startswith(h))
        if action in ("scale", "testnet_research"):     # هیچ‌کدام مجوزِ پول را عوض نمی‌کنند
            continue
        if action == "preregister" and ours:
            continue
        if action == "judge" and ours and judged and not judgement_seen:
            judgement_seen = True
            continue
        out.append(e)
    return out


# ───────────────────────── مقیاسِ سرمایهٔ واقعی ─────────────────────────
def live_scale_decision(closed_live_trades, current_risk_pct, now=None, rules=LIVE_SCALE):
    """تصمیمِ پیش‌ثبت‌شده برای اندازهٔ ریسکِ لایو (بخشِ ۵ پلن).

    ``closed_live_trades``: فهرستِ ``{"closed_at_ts", "r"}`` از معامله‌های واقعیِ ربات.
    """
    now = now if now is not None else time.time()
    if not closed_live_trades:
        return {"action": "hold", "risk_pct": current_risk_pct, "reasons": ["هنوز معاملهٔ لایو نیست"]}
    rs = np.asarray([float(t["r"]) for t in closed_live_trades], float)
    ts = np.asarray([float(t["closed_at_ts"]) * 1000 for t in closed_live_trades], float)
    first = float(min(t["closed_at_ts"] for t in closed_live_trades))
    days = (now - first) / DAY
    block = 7 * 86400 * 1000
    lcb = stats.block_bootstrap_lcb(rs, ts, block, alpha=rules["lcb_alpha"], B=800)
    gains, losses = float(rs[rs > 0].sum()), float(-rs[rs <= 0].sum())
    pf = gains / losses if losses > 1e-12 else (99.0 if gains > 0 else 0.0)
    eq = np.cumsum(rs)
    dd = float(np.max(np.maximum.accumulate(np.concatenate([[0.0], eq]))[1:] - eq)) if len(eq) else 0.0
    # افت: هر ماه با کرانِ پایینِ ضعیف ⇒ برگشت به ریسکِ پایه
    month = rs[ts >= (now - 30 * DAY) * 1000]
    mts = ts[ts >= (now - 30 * DAY) * 1000]
    month_lcb = stats.block_bootstrap_lcb(month, mts, block, alpha=rules["lcb_alpha"], B=500) \
        if len(month) >= 10 else None
    month_mean = float(month.mean()) if len(month) >= 10 else None
    # عدم‌تقارنِ ایمنی: **افزایش** اثبات می‌خواهد (کرانِ پایین > ۰)، ولی **کاهش** نه.
    # یک ماهِ کاملاً زیان‌ده (۲۵ از ۲۵) کمتر از ۵ بلوکِ هفتگی دارد، پس بوت‌استرپ
    # «نامعلوم» برمی‌گرداند؛ اگر فقط به آن تکیه کنیم، ترمز هرگز نمی‌گیرد.
    demote_reason = None
    if month_lcb is not None and month_lcb < rules["demote_lcb_r"]:
        demote_reason = f"LCB ماهِ اخیر {month_lcb} < {rules['demote_lcb_r']}"
    elif month_mean is not None and month_mean < rules["demote_lcb_r"]:
        demote_reason = f"میانگینِ ماهِ اخیر {month_mean:.2f}R < {rules['demote_lcb_r']}"
    if demote_reason:
        return {"action": "demote", "risk_pct": rules["base_risk_pct"],
                "reasons": [demote_reason],
                "lcb": lcb, "pf": round(pf, 3), "max_dd_r": round(dd, 2)}
    ok = (days >= rules["min_days"] and len(rs) >= rules["min_trades"]
          and lcb is not None and lcb > 0 and pf >= rules["min_pf"] and dd < rules["max_dd_r"])
    if ok and current_risk_pct < rules["step_up_risk_pct"]:
        return {"action": "step_up", "risk_pct": rules["step_up_risk_pct"], "reasons": [],
                "lcb": lcb, "pf": round(pf, 3), "max_dd_r": round(dd, 2), "days": round(days, 1)}
    reasons = []
    if days < rules["min_days"]:
        reasons.append(f"روزهای لایو {days:.0f} < {rules['min_days']}")
    if len(rs) < rules["min_trades"]:
        reasons.append(f"معامله‌ها {len(rs)} < {rules['min_trades']}")
    if lcb is None or lcb <= 0:
        reasons.append(f"LCB={lcb}")
    if pf < rules["min_pf"]:
        reasons.append(f"PF={pf:.2f}")
    if dd >= rules["max_dd_r"]:
        reasons.append(f"DD={dd:.1f}R")
    return {"action": "hold", "risk_pct": current_risk_pct, "reasons": reasons,
            "lcb": lcb, "pf": round(pf, 3), "max_dd_r": round(dd, 2)}


# ───────────────────────── گزارشِ کامل ─────────────────────────
def build(now=None, closed_positions=None, testnet_started_at=None):
    """گزارشِ کامل + حکمِ هر مرحله. هیچ گیتی را خودکار عوض نمی‌کند؛ فقط می‌گوید."""
    import research
    now = now if now is not None else time.time()
    g = gates.load_gates(force=True)
    doc = research.load_prereg() or {}
    judged = research.last_judgement() or {}
    # حکمِ **همین** پیش‌ثبت؛ حکمِ آزمایشِ قبلی دربارهٔ فهرستِ فعلی چیزی نمی‌گوید
    judged_this = bool(judged.get("judged")) and bool(doc.get("hash")) \
        and judged.get("prereg_hash") == doc.get("hash")
    if not judged_this:
        judged = {}
    # ساعتِ اثبات با حکم شروع می‌شود (نه با پیش‌ثبت): پیش از حکم چیزی مجاز نبود
    since = float(judged.get("judged_at") or doc.get("registered_at") or now)
    allowed = g.get("allowed_combos") or []
    symbols = g.get("allowed_symbols") or []
    frozen_means = [h.get("mean") for k, h in (judged.get("hypotheses") or {}).items()
                    if k in allowed and h.get("mean") is not None]
    frozen_mean = float(np.mean(frozen_means)) if frozen_means else None
    # gates.json را با دست هم می‌شود ویرایش کرد و آن ردی در ژورنال نمی‌گذارد؛ پس
    # فهرستِ فعلی با خودِ حکم مقایسه می‌شود. کمتر از حکم (پس از قطعِ اضطراری) امن
    # است؛ هر چیزِ **بیشتر** از حکم مجوزی است که هیچ آزمونی نداده.
    beyond = sorted(set(allowed) - set(judged.get("passing") or []))
    if allowed:
        beyond += sorted(set(symbols) - set(judged.get("symbols") or []))

    shadow = shadow_section(since, allowed, symbols, now=now)
    all_cands = shadow_section(since, [], [], now=now)       # همهٔ کاندیدها، برای مقایسه
    execution = execution_section(since)
    closed = closed_positions or []
    gap = sim_vs_exchange_gap(closed)
    tn_days = (now - testnet_started_at) / DAY if testnet_started_at else 0.0
    changes = gate_changes_during_proving(doc.get("hash"), since, judged=judged_this)
    rep = {
        "generated_at": now,
        "prereg_hash": doc.get("hash"),
        "registered_at": doc.get("registered_at"),
        "proving_started_at": judged.get("judged_at"),
        "allowed_combos": allowed,
        "allowed_beyond_judgement": beyond,
        "frozen_test_mean_r": frozen_mean,
        "shadow_allowed": shadow,
        "shadow_all_candidates": {k: all_cands.get(k) for k in
                                  ("n", "n_eff", "mean", "lcb", "win_rate", "brier_skill")},
        "execution": execution,
        "sim_vs_exchange_gap": gap,
        "integrity": integrity_section(),
        "gate_changes_during_proving": len(changes),
        "stages": {},
    }
    if not allowed:
        rep["stages"]["frozen_test"] = {
            "pass": False,
            "reasons": ["هیچ ترکیبی از آزمونِ منجمد عبور نکرده — مراحلِ بعد موضوعیت ندارند"]}
    elif beyond:
        rep["stages"]["frozen_test"] = {
            "pass": False,
            "reasons": [f"gates.json چیزی را مجاز کرده که حکمِ این پیش‌ثبت قبول نکرده: {beyond}"]}
    else:
        rep["stages"]["frozen_test"] = {"pass": True, "reasons": []}
    rep["stages"]["shadow"] = evaluate_shadow(shadow, frozen_mean)
    if changes:
        rep["stages"]["shadow"]["pass"] = False
        rep["stages"]["shadow"]["reasons"].append(
            f"{len(changes)} تغییرِ گیت در دورهٔ اثبات — ساعت باید از نو شروع شود")
    rep["stages"]["testnet"] = evaluate_testnet(execution, tn_days, gap)
    rep["stages"]["ops"] = evaluate_ops(now=now)
    rep["stages"]["integrity"] = {"pass": rep["integrity"]["ok"],
                                  "reasons": [] if rep["integrity"]["ok"] else ["ردیفِ مسموم در دفتر"]}
    order = ["frozen_test", "shadow", "testnet", "ops", "integrity"]
    rep["ready_for_live"] = bool(allowed) and all(
        rep["stages"][k]["pass"] for k in order if k in rep["stages"])
    return rep


def write(rep, now=None):
    os.makedirs(REPORT_DIR, exist_ok=True)
    day = time.strftime("%Y-%m-%d", time.gmtime(now if now is not None else time.time()))
    path = os.path.join(REPORT_DIR, f"{day}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(rep, f, ensure_ascii=False, indent=1, default=float)
    return path
