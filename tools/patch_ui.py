# -*- coding: utf-8 -*-
"""پچِ رابطِ کاربری — عددِ بی‌مجوز نباید مثلِ مجوز دیده شود.

جدولِ اصلی، خروجیِ CSV و کارتِ معامله «احتمالِ برد» و «EV» را خام نشان می‌دادند،
حتی وقتی از مدلی آمده بود که دادگاه ردش کرده بود. کاربر عدد را می‌دید و فکر می‌کرد
تأیید شده است. حالا هر عددِ بی‌مجوز برچسبِ «تشخیصی» می‌گیرد و وضعیتِ قفل و
سلامتِ سیستم بالای صفحه نمایش داده می‌شود.
"""
import io
import os

P = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "bot", "static", "index.html")
s = io.open(P, encoding="utf-8").read()


def sub1(old, new, label):
    global s
    n = s.count(old)
    assert n == 1, f"{label}: found {n}"
    s = s.replace(old, new)


# ── بنرِ قفل و سلامت ──
sub1('  <div id="calibBanner"></div>',
     '  <div id="gateBanner"></div>\n  <div id="calibBanner"></div>',
     "gate banner slot")

sub1('''/* ═══════════ بنرِ ساختِ کالیبراسیون ═══════════ */''',
     '''/* ═══════════ بنرِ قفلِ ایمنی و سلامت ═══════════ */
async function loadGateBanner(){
  try{
    const [g, h, r] = await Promise.all([
      (await fetch("/api/gates", {cache:"no-store"})).json(),
      (await fetch("/api/health", {cache:"no-store"})).json(),
      (await fetch("/api/research", {cache:"no-store"})).json(),
    ]);
    const el = $("#gateBanner");
    const combos = (g.allowed_combos||[]);
    const color = h.status==="green" ? "var(--up)" : h.status==="amber" ? "var(--brass)" : "var(--dn)";
    const judged = r.judgement && r.judgement.judged;
    const evid = judged ? Object.entries(r.judgement.hypotheses||{})
        .map(([k,v])=>`${k}: ${({edge_proven:"اثبات شد",no_edge:"لبه نیست",
          weak_edge:"لبهٔ ضعیف",insufficient_evidence:"شواهد ناکافی"})[v.evidence]||v.verdict||"—"}`).join(" · ") : "";
    el.innerHTML = `<div class="calbanner" style="border-color:${color}">
      <div class="top"><span>🔒 <b>قفلِ ایمنیِ سرمایه</b> — ${combos.length
          ? `ترکیب‌های مجاز: ${combos.join("، ")} (فقط روی ${(g.allowed_symbols||[]).length} نماد)`
          : "هیچ ترکیبی مجاز نیست؛ هیچ معامله‌ای باز نمی‌شود"}
        · پول واقعی: <b>${g.live_effective?"فعال":"خاموش"}</b></span>
        <span style="color:${color}">سلامت: <b>${h.status}</b></span></div>
      ${(h.problems||[]).length?`<div class="sub" style="color:var(--dn)">${h.problems.join(" · ")}</div>`:""}
      ${judged?`<div class="sub">حکمِ آزمونِ منجمد: ${evid}</div>`
        : `<div class="sub">آزمونِ منجمد هنوز قضاوت نشده (${r.integrity})</div>`}
    </div>`;
  }catch(e){}
}

/* ═══════════ بنرِ ساختِ کالیبراسیون ═══════════ */''',
     "gate banner fn")

sub1('loadVersion(); loadOverview(); loadPositions(); loadConfig(false); loadEngineCred(); loadCalibBanner(); loadEdgeHealth(); loadBot();',
     'loadVersion(); loadGateBanner(); loadOverview(); loadPositions(); loadConfig(false); loadEngineCred(); loadCalibBanner(); loadEdgeHealth(); loadBot();\nsetInterval(loadGateBanner, 60000);',
     "gate banner init")

# ── جدول: وین‌ریتِ بی‌مجوز خاکستری و برچسب‌دار ──
sub1('''    <td class="num">${c.p_win!=null ? `${fmt(c.p_win,0)}٪ <small>(${c.n_hist})</small>''',
     '''    <td class="num" ${c.p_win!=null && !c.tradeable ? 'style="opacity:.55" title="از مدلی که مجوزِ ورود ندارد — فقط تشخیصی"' : ""}>${c.p_win!=null ? `${fmt(c.p_win,0)}٪${!c.tradeable?' <small>تشخیصی</small>':''} <small>(${c.n_hist})</small>''',
     "table p_win diagnostic")

# ── CSV: ستونِ «مجاز» تا عددِ خروجی با مجوز اشتباه گرفته نشود ──
sub1('''  const head = "symbol,price,chg24h,score,z,side,setup,grade,p_up,target_pct,rr,ev_pct,ev_lcb_pct,p_win,p_win_low,p_win_high,edge_r,edge_lcb_r,market_rank_pct,n_hist,regime,status";''',
     '''  const head = "symbol,price,chg24h,score,z,side,setup,grade,authorized,p_up,target_pct,rr,ev_pct,ev_lcb_pct,p_win,p_win_low,p_win_high,edge_r,edge_lcb_r,market_rank_pct,n_hist,regime,status";''',
     "csv head")
sub1('''  const body = rows.map(c=>[c.symbol,c.price,c.chg24h,c.score,c.z,c.side||"",c.setup_fa||"",c.grade||"",c.p_up,''',
     '''  const body = rows.map(c=>[c.symbol,c.price,c.chg24h,c.score,c.z,c.side||"",c.setup_fa||"",c.grade||"",c.tradeable?"yes":"no",c.p_up,''',
     "csv body")

# ── کارتِ معامله ──
sub1('''  $("#tmStats").innerHTML = `${c.p_win!=null?fmt(c.p_win,0)+"٪ برد":"—"} · EV ${c.ev_pct!=null?pct(c.ev_pct):"—"}` +''',
     '''  $("#tmStats").innerHTML = (c.tradeable ? "" : `<b style="color:var(--dn)">بدون مجوز ورود — اعداد فقط تشخیصی‌اند</b> · `) +
    `${c.p_win!=null?fmt(c.p_win,0)+"٪ برد":"—"} · EV ${c.ev_pct!=null?pct(c.ev_pct):"—"}` +''',
     "trade card diagnostic")

# ── زیرعنوانِ گمراه‌کننده ──
sub1('<div class="sub">پایش ۲۰۰ ارز برتر · مدل آماری کالیبره</div>',
     '<div class="sub">پایش ارزهای برتر · ورود فقط با ترکیبِ پیش‌ثبت‌شده و قضاوت‌شده</div>',
     "subtitle")

io.open(P, "w", encoding="utf-8", newline="\n").write(s)
print("ui patch applied")
