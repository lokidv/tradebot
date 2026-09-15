# -*- coding: utf-8 -*-
"""آمارِ صادقانه برای دادهٔ معاملاتی — کرانِ پایین، اندازهٔ نمونهٔ مؤثر، چند-مقایسه‌ای.

سه فرضِ نادرست در کدِ قبلی، همهٔ گیت‌های اعتماد را از نویز عبور می‌داد:

۱. **استقلالِ نمونه‌ها.** کرانِ پایین با ``avg − z·sd/√n`` حساب می‌شد. ولی
   برچسب‌های جهت هر ۴ کندل با افقِ ۳۰ تا ۴۸ کندل ساخته می‌شوند، یعنی هر نتیجه
   در ۸ تا ۱۲ نمونه تکرار شده؛ و همبستگیِ مقطعیِ بازده بین ارزهای بزرگ روی ۴
   ساعته و روزانه حدودِ ۰٫۶ است، یعنی ۱۰۰ ارز تقریباً ۱٫۶ ارزِ مستقل‌اند.
   ``effective_n`` هر دو را حساب می‌کند.

۲. **توزیعِ نرمال.** توزیعِ R در ساختارِ ۱٫۸R/−۱R دوقله‌ای و کج است؛
   ``block_bootstrap_lcb`` هیچ فرضی دربارهٔ شکلِ توزیع نمی‌گذارد و بلوک‌های
   زمانی را دست‌نخورده نمونه‌برداری می‌کند تا خودهمبستگی حفظ شود.

۳. **یک آزمون.** در عمل ده‌ها ترکیب هم‌زمان آزموده می‌شد (۴ تایم‌فریم × چند
   خانوادهٔ مدل × شبکهٔ پارامتر × ۱۲ سطلِ جیب). ``romano_wolf`` مقدارِ p را
   برای کلِ خانواده تصحیح می‌کند، وگرنه «بهترینِ ۱۰۰ تلاش» همیشه خوب به‌نظر می‌رسد.
"""
import math

import numpy as np

DEFAULT_RHO = 0.60         # همبستگیِ مقطعیِ اندازه‌گیری‌شدهٔ بازده بین ارزهای بزرگ (4h/1d)
DEFAULT_B = 2000
SEED = 20260914            # ثابت: «یک‌بار قضاوت» باید تکرارپذیر باشد
MIN_BLOCKS = 5             # کمتر از این تعداد بلوکِ زمانی، عدم‌قطعیت قابلِ برآورد نیست


def _blocks(ts, block_ms):
    """شمارهٔ بلوکِ زمانی هر رویداد — رویدادهای یک بلوک با هم جابه‌جا می‌شوند."""
    ts = np.asarray(ts, dtype=np.float64)
    if block_ms <= 0:
        return np.arange(len(ts))
    return np.floor((ts - ts.min()) / float(block_ms)).astype(np.int64)


def effective_n(values, ts, block_ms, rho=DEFAULT_RHO):
    """اندازهٔ نمونهٔ مؤثر با احتسابِ هم‌پوشانیِ زمانی و همبستگیِ مقطعی.

    درونِ هر بلوکِ زمانی، ``m`` رویدادِ هم‌زمان ارزشِ ``m / (1 + (m−1)ρ)`` نمونهٔ
    مستقل دارند. جمعِ این مقادیر روی بلوک‌ها، ``n`` مؤثر است.
    """
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return 0.0
    blk = _blocks(ts, block_ms)
    total = 0.0
    for b in np.unique(blk):
        m = int(np.sum(blk == b))
        total += m / (1.0 + (m - 1) * float(rho)) if m > 1 else 1.0
    return round(float(total), 3)


def block_bootstrap(values, ts, block_ms, B=DEFAULT_B, seed=SEED, statistic=np.mean):
    """توزیعِ بوت‌استرپِ آماره با نمونه‌برداریِ بلوکیِ زمانی."""
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return np.array([])
    blk = _blocks(ts, block_ms)
    groups = [values[blk == b] for b in np.unique(blk)]
    k = len(groups)
    rng = np.random.default_rng(seed)
    out = np.empty(B, dtype=np.float64)
    for i in range(B):
        pick = rng.integers(0, k, size=k)
        sample = np.concatenate([groups[j] for j in pick])
        out[i] = statistic(sample)
    return out


def n_blocks(ts, block_ms):
    return int(np.unique(_blocks(ts, block_ms)).size) if len(ts) else 0


def block_bootstrap_lcb(values, ts, block_ms, alpha=0.05, B=DEFAULT_B, seed=SEED,
                        min_blocks=MIN_BLOCKS):
    """کرانِ پایینِ یک‌طرفهٔ ``1−alpha`` برای میانگین، بدونِ فرضِ نرمال‌بودن.

    اگر رویدادها در کمتر از ``min_blocks`` بلوکِ زمانی جا بگیرند، ``None``
    برمی‌گردد: با یک خوشه نمی‌توان عدم‌قطعیت را برآورد کرد و بوت‌استرپ همان
    میانگین را پس می‌دهد — یعنی دقیقاً همان اطمینانِ کاذبی که «جیبِ زنده» با
    n=۱۸ در یک بازهٔ کوتاه می‌ساخت. «نامعلوم» صادقانه‌تر از «اثبات‌شده» است.
    """
    if len(values) == 0:
        return None
    if n_blocks(ts, block_ms) < int(min_blocks):
        return None
    dist = block_bootstrap(values, ts, block_ms, B=B, seed=seed)
    if dist.size == 0:
        return None
    return round(float(np.quantile(dist, alpha)), 4)


def summarize(values, ts, block_ms, alpha=0.05, baseline=0.0, rho=DEFAULT_RHO, B=DEFAULT_B):
    """خلاصهٔ کاملِ یک سطل: میانگین، کرانِ پایین، n مؤثر، t با n مؤثر، PF."""
    values = np.asarray(values, dtype=np.float64)
    n = int(values.size)
    if n == 0:
        return {"n": 0, "n_eff": 0.0, "n_blocks": 0, "mean": None, "lcb": None,
                "t_stat": None, "win_rate": None, "profit_factor": None,
                "sd": None, "uplift": None}
    mean = float(values.mean())
    sd = float(values.std(ddof=1)) if n > 1 else 0.0
    n_eff = effective_n(values, ts, block_ms, rho=rho)
    t_stat = (mean - float(baseline)) / (sd / math.sqrt(n_eff)) if sd > 0 and n_eff > 1 else None
    gains = float(values[values > 0].sum())
    losses = float(-values[values <= 0].sum())
    return {
        "n": n,
        "n_eff": n_eff,
        "n_blocks": n_blocks(ts, block_ms),
        "mean": round(mean, 4),
        "sd": round(sd, 4),
        "lcb": block_bootstrap_lcb(values, ts, block_ms, alpha=alpha, B=B),
        "t_stat": round(t_stat, 3) if t_stat is not None else None,
        "win_rate": round(float((values > 0).mean()) * 100, 2),
        "profit_factor": round(gains / losses, 4) if losses > 1e-12 else (99.0 if gains > 0 else 0.0),
        "uplift": round(mean - float(baseline), 4),
        "alpha": alpha,
    }


def _studentized(sample, baseline):
    sd = sample.std(ddof=1)
    if sd <= 1e-12:
        return 0.0
    return float((sample.mean() - baseline) / (sd / math.sqrt(sample.size)))


def romano_wolf(hypotheses, block_ms, alpha=0.05, B=DEFAULT_B, seed=SEED, baseline=0.0):
    """مقدارِ p تصحیح‌شده برای کلِ خانوادهٔ فرضیه‌ها (روشِ گام‌به‌گامِ maxT).

    ``hypotheses``: دیکشنریِ ``{نام: (values, ts)}``. برای هر فرضیه آمارهٔ
    studentized محاسبه و با توزیعِ بوت‌استرپِ **مرکزشده** مقایسه می‌شود؛ سپس
    گام‌به‌گام از قوی‌ترین فرضیه به ضعیف‌ترین، سقفِ توزیعِ باقی‌مانده ملاک است.

    آزمون یک‌طرفه است: فرضِ صفر «لبه ≤ baseline».
    """
    names = list(hypotheses)
    if not names:
        return {}
    obs, dists = {}, {}
    for name in names:
        values, ts = hypotheses[name]
        values = np.asarray(values, dtype=np.float64)
        if values.size < 2:
            obs[name] = 0.0
            dists[name] = np.zeros(B)
            continue
        obs[name] = _studentized(values, baseline)
        boot = block_bootstrap(values, ts, block_ms, B=B, seed=seed,
                               statistic=lambda s: _studentized(s, values.mean()))
        dists[name] = np.nan_to_num(boot)
    order = sorted(names, key=lambda k: obs[k], reverse=True)
    adjusted, running, remaining = {}, 0.0, list(order)
    for name in order:
        block = np.max(np.column_stack([dists[k] for k in remaining]), axis=1)
        p = float(np.mean(block >= obs[name]))
        running = max(running, p)              # یکنواختی: p تصحیح‌شده نزولی نمی‌شود
        adjusted[name] = round(running, 4)
        remaining.remove(name)
    return {n: {"t_stat": round(obs[n], 3), "p_adj": adjusted[n],
                "reject_null": adjusted[n] < alpha} for n in names}


def deflated_mean_threshold(n_trials, sd, n_eff, alpha=0.05):
    """حداقلِ میانگینی که پس از ``n_trials`` تلاش هنوز معنادار است.

    کاربرد: وقتی ۱۰۰ ترکیب آزموده‌ای، «بهترین» باید از این آستانه رد شود نه از
    آستانهٔ یک آزمونِ تکی.
    """
    if n_eff <= 1 or sd <= 0:
        return None
    eff_alpha = alpha / max(int(n_trials), 1)          # Bonferroni به‌عنوان کفِ محافظه‌کار
    # تقریبِ معکوسِ نرمالِ استاندارد (Acklam ساده‌شده برای دُمِ راست)
    z = _norm_ppf(1.0 - eff_alpha)
    return round(z * sd / math.sqrt(n_eff), 4)


def _norm_ppf(p):
    """معکوسِ CDF نرمالِ استاندارد (تقریبِ Moro/Beasley-Springer)."""
    p = min(max(p, 1e-12), 1 - 1e-12)
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]
    plow, phigh = 0.02425, 1 - 0.02425
    if p < plow:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
               ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    if p > phigh:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
               ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    q = p - 0.5
    r = q * q
    return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / \
           (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)


def power_sample_size(edge_r, sd=1.25, alpha=0.05, power=0.80):
    """چند رویدادِ **مستقل** برای تشخیصِ لبهٔ ``edge_r`` لازم است؟

    با sd ≈ ۱٫۲۵R و لبهٔ +۰٫۱R جواب ~۹۵۰ می‌شود — به همین دلیل سطلی با n=۱۸
    هرگز نمی‌تواند مرجعِ تصمیم باشد، فقط پایشگر.
    """
    if edge_r <= 0:
        return None
    z_a, z_b = _norm_ppf(1 - alpha), _norm_ppf(power)
    return int(math.ceil(((z_a + z_b) * sd / edge_r) ** 2))
