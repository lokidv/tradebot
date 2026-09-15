# -*- coding: utf-8 -*-
"""پچِ meta_gate — بلاکِ سخت با کدِ دلیل، نه با جست‌وجوی زیررشتهٔ فارسی.

تشخیصِ «سخت» بودنِ یک بلاک با ``"شدید" in x or "شلوغ" in x or "اهرم" in x`` انجام
می‌شد؛ یعنی ویرایشِ یک واژه در پیامِ نمایشی، رفتارِ ریسک را بی‌صدا عوض می‌کرد.
حالا هر بلاک یک کدِ ثابت دارد و مجموعهٔ کدهای سخت صریح است.
"""
import io
import os

P = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "bot", "meta_gate.py")
s = io.open(P, encoding="utf-8").read()


def sub1(old, new, label):
    global s
    n = s.count(old)
    assert n == 1, f"{label}: found {n}"
    s = s.replace(old, new)


sub1('''def evaluate(side: str, extras: dict, cs=None) -> dict[str, Any]:
    """خروجی: approve, score∈[0,1], reasons, size_mult, blocks."""
    d = 1 if side == "long" else -1
    reasons = []
    blocks = []
    score = 0.55''',
     '''# کدهای بلاکی که بدونِ توجه به امتیاز، ورود را رد می‌کنند. صریح و ثابت —
# نه برداشت از متنِ پیام که با هر ویرایشِ واژه عوض می‌شد.
HARD_BLOCKS = frozenset({
    "macro_extreme_against", "breadth_against", "funding_crowded", "oi_leverage_building",
})


def evaluate(side: str, extras: dict, cs=None) -> dict[str, Any]:
    """خروجی: approve, score∈[0,1], reasons, size_mult, blocks, block_codes."""
    d = 1 if side == "long" else -1
    reasons = []
    blocks = []
    codes = []

    def block(code, text):
        codes.append(code)
        blocks.append(text)

    score = 0.55''', "evaluate header")

replacements = [
    ('        blocks.append("رژیم کلان نزولیِ شدید BTC — لانگ آلت رد شد")',
     '        block("macro_extreme_against", "رژیم کلان نزولیِ شدید BTC — لانگ آلت رد شد")'),
    ('        blocks.append("رژیم کلان صعودیِ شدید BTC — شورت رد شد")',
     '        block("macro_extreme_against", "رژیم کلان صعودیِ شدید BTC — شورت رد شد")'),
    ('        blocks.append(f"پهنای بازار ضعیف است ({breadth:+.2f}) — لانگ خلاف جریان")',
     '        block("breadth_against", f"پهنای بازار ضعیف است ({breadth:+.2f}) — لانگ خلاف جریان")'),
    ('        blocks.append(f"پهنای بازار قوی است ({breadth:+.2f}) — شورت خلاف جریان")',
     '        block("breadth_against", f"پهنای بازار قوی است ({breadth:+.2f}) — شورت خلاف جریان")'),
    ('        blocks.append(f"لانگ‌ها شلوغ‌اند (funding z={fz:+.1f}, persist={persist})")',
     '        block("funding_crowded", f"لانگ‌ها شلوغ‌اند (funding z={fz:+.1f}, persist={persist})")'),
    ('        blocks.append(f"شورت‌ها شلوغ‌اند (funding z={fz:+.1f}, persist={persist})")',
     '        block("funding_crowded", f"شورت‌ها شلوغ‌اند (funding z={fz:+.1f}, persist={persist})")'),
    ('            blocks.append(f"اهرم لانگ در حال انباشت (OI z={oi_z:+.1f}, Δ{oi_chg:+.1f}٪)")',
     '            block("oi_leverage_building", f"اهرم لانگ در حال انباشت (OI z={oi_z:+.1f}, Δ{oi_chg:+.1f}٪)")'),
    ('            blocks.append(f"اهرم شورت در حال انباشت (OI z={oi_z:+.1f}, Δ{oi_chg:+.1f}٪)")',
     '            block("oi_leverage_building", f"اهرم شورت در حال انباشت (OI z={oi_z:+.1f}, Δ{oi_chg:+.1f}٪)")'),
    ('''            blocks.append(f"رژیم محلی ناپایدار ({'/'.join(hyst.get('recent') or [])})")''',
     '''            block("regime_unstable", f"رژیم محلی ناپایدار ({'/'.join(hyst.get('recent') or [])})")'''),
    ('            blocks.append(f"مخالف تکانهٔ BTC (z={bz:+.1f})")',
     '            block("btc_momentum_against", f"مخالف تکانهٔ BTC (z={bz:+.1f})")'),
]
for old, new in replacements:
    sub1(old, new, old[:40])

sub1('''    hard = any(
        "شدید" in x or "شلوغ" in x or "اهرم" in x or (x.startswith("پهنای بازار") and "خلاف" in x)
        for x in blocks
    )''',
     '''    hard = bool(HARD_BLOCKS.intersection(codes))''', "hard detection")

sub1('''        "blocks": blocks[:4],''', '''        "blocks": blocks[:4],
        "block_codes": list(codes),''', "return codes")

assert s.count("blocks.append") == 1, "raw blocks.append remains outside the block() helper"
io.open(P, "w", encoding="utf-8", newline="\n").write(s)
print("meta_gate patch applied")
