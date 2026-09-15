# -*- coding: utf-8 -*-
"""اتصال به حساب دموی تست‌نت بایننس فیوچرز (پول مجازی — بدون هیچ معامله واقعی).
آدرس پایه عمداً فقط تست‌نت است و قابل تغییر به شبکه اصلی نیست."""
import hashlib
import hmac
import json
import os
import threading
import time
from urllib.parse import urlencode

import httpx

import log

TESTNET_BASE = "https://testnet.binancefuture.com"   # ⚠️ فقط تست‌نت
CFG_PATH = os.path.join(os.path.dirname(__file__), "data", "config.json")
DEFAULT_CFG = {"broker": "local", "api_key": "", "api_secret": "", "leverage": 2}
_cfg_lock = threading.Lock()


def load_cfg():
    with _cfg_lock:
        if not os.path.exists(CFG_PATH):
            return dict(DEFAULT_CFG)
        with open(CFG_PATH, "r", encoding="utf-8") as f:
            cfg = dict(DEFAULT_CFG)
            cfg.update(json.load(f))
            return cfg


def save_cfg(cfg):
    with _cfg_lock:
        os.makedirs(os.path.dirname(CFG_PATH), exist_ok=True)
        tmp = CFG_PATH + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(cfg, f, ensure_ascii=False, indent=1)
        os.replace(tmp, CFG_PATH)


class TestnetError(RuntimeError):
    pass


class BinanceTestnet:
    """کلاینت مینیمال فیوچرز تست‌نت: موجودی، سفارش مارکت + براکت حدضرر/هدف، پوزیشن‌ها، بستن."""

    def __init__(self, api_key, api_secret, leverage=2):
        self.key, self.secret = api_key.strip(), api_secret.strip()
        self.leverage = max(1, min(int(leverage), 10))
        self.client = httpx.Client(base_url=TESTNET_BASE, timeout=15.0,
                                   headers={"X-MBX-APIKEY": self.key})
        self._info = None
        self._lev_set: set = set()

    # ── زیرساخت امضا ──
    def _signed(self, method, path, **params):
        params["timestamp"] = int(time.time() * 1000)
        params["recvWindow"] = 10000
        qs = urlencode(params)
        sig = hmac.new(self.secret.encode(), qs.encode(), hashlib.sha256).hexdigest()
        url = f"{path}?{qs}&signature={sig}"
        r = self.client.request(method, url)
        if r.status_code >= 400:
            try:
                msg = r.json().get("msg", r.text)
            except Exception:  # noqa: BLE001
                msg = r.text
            raise TestnetError(f"تست‌نت بایننس: {msg}")
        return r.json()

    def _public(self, path, **params):
        r = self.client.get(path, params=params)
        r.raise_for_status()
        return r.json()

    # ── اطلاعات حساب و نماد ──
    def balance_usdt(self):
        for b in self._signed("GET", "/fapi/v2/balance"):
            if b.get("asset") == "USDT":
                return float(b["availableBalance"]), float(b["balance"])
        return 0.0, 0.0

    def _sym_info(self, symbol):
        if self._info is None:
            data = self._public("/fapi/v1/exchangeInfo")
            self._info = {s["symbol"]: s for s in data["symbols"]}
        info = self._info.get(symbol)
        if not info:
            raise TestnetError(f"نماد {symbol} در فیوچرز تست‌نت موجود نیست")
        step = tick = min_qty = 0.0
        min_notional = 5.0
        for f in info["filters"]:
            if f["filterType"] == "LOT_SIZE":
                step, min_qty = float(f["stepSize"]), float(f["minQty"])
            elif f["filterType"] == "PRICE_FILTER":
                tick = float(f["tickSize"])
            elif f["filterType"] in ("MIN_NOTIONAL", "NOTIONAL"):
                min_notional = float(f.get("notional", f.get("minNotional", 5)))
        return step, tick, min_qty, min_notional

    @staticmethod
    def _round_step(x, step):
        if step <= 0:
            return x
        n = int(x / step + 1e-9)
        return round(n * step, 12)

    def mark_price(self, symbol):
        return float(self._public("/fapi/v1/ticker/price", symbol=symbol)["price"])

    def _ensure_leverage(self, symbol):
        if symbol in self._lev_set:
            return
        try:
            self._signed("POST", "/fapi/v1/leverage", symbol=symbol, leverage=self.leverage)
            self._lev_set.add(symbol)
        except TestnetError:
            pass  # اهرم پیش‌فرض حساب استفاده می‌شود

    # ── معامله ──
    def open_bracket(self, symbol, side, size_usdt, sl, tp):
        """ورود مارکت با نُشنالِ درخواستی + دو سفارش محافظ.

        ``size_usdt`` ارزش خود پوزیشن است، نه مارجین. اهرم فقط مارجین موردنیاز
        صرافی را تغییر می‌دهد و نباید دوباره در quantity ضرب شود.
        """
        step, tick, min_qty, min_notional = self._sym_info(symbol)
        price = self.mark_price(symbol)
        qty = self._round_step(size_usdt / price, step)
        if qty < min_qty or qty * price < min_notional:
            need = max(min_qty * price, min_notional)
            raise TestnetError(f"حجم خیلی کم است — حداقل ~{need:.0f}$ برای {symbol}")
        self._ensure_leverage(symbol)
        entry_side = "BUY" if side == "long" else "SELL"
        exit_side = "SELL" if side == "long" else "BUY"
        order = self._signed("POST", "/fapi/v1/order", symbol=symbol, side=entry_side,
                             type="MARKET", quantity=qty, newOrderRespType="RESULT")
        fill = float(order.get("avgPrice") or 0) or price
        direction = 1 if side == "long" else -1
        sl = fill - direction * abs(price - sl)
        tp = fill + direction * abs(tp - price)
        sl_r = self._round_step(sl, tick)
        tp_r = self._round_step(tp, tick)
        try:
            self._signed("POST", "/fapi/v1/order", symbol=symbol, side=exit_side,
                         type="STOP_MARKET", stopPrice=sl_r, closePosition="true",
                         workingType="MARK_PRICE")
            self._signed("POST", "/fapi/v1/order", symbol=symbol, side=exit_side,
                         type="TAKE_PROFIT_MARKET", stopPrice=tp_r, closePosition="true",
                         workingType="MARK_PRICE")
        except TestnetError:
            # اگر براکت ثبت نشد، پوزیشن بی‌محافظ نماند
            self.close_symbol(symbol)
            raise
        return {"order_id": order.get("orderId"), "qty": qty, "price": fill,
                "sl": sl_r, "tp": tp_r}

    def positions(self):
        out = []
        for p in self._signed("GET", "/fapi/v2/positionRisk"):
            amt = float(p.get("positionAmt", 0) or 0)
            if abs(amt) < 1e-12:
                continue
            out.append({
                "symbol": p["symbol"],
                "side": "long" if amt > 0 else "short",
                "qty": abs(amt),
                "entry": float(p["entryPrice"]),
                "mark": float(p["markPrice"]),
                "pnl_usdt": float(p["unRealizedProfit"]),
                "leverage": float(p.get("leverage", self.leverage) or self.leverage),
            })
        return out

    def close_symbol(self, symbol):
        try:
            self._signed("DELETE", "/fapi/v1/allOpenOrders", symbol=symbol)
        except TestnetError:
            log.exc()
        for p in self.positions():
            if p["symbol"] == symbol:
                side = "SELL" if p["side"] == "long" else "BUY"
                self._signed("POST", "/fapi/v1/order", symbol=symbol, side=side,
                             type="MARKET", quantity=p["qty"], reduceOnly="true")
                return True
        return False

    def realized_pnl_since(self, symbol, start_ms):
        """سود/زیان خالص پس از کمیسیون و فاندینگِ ثبت‌شده در income history."""
        try:
            rows = self._signed("GET", "/fapi/v1/income", symbol=symbol,
                                startTime=int(start_ms), limit=100)
            included = {"REALIZED_PNL", "COMMISSION", "FUNDING_FEE"}
            return sum(float(r["income"]) for r in rows if r.get("incomeType") in included)
        except TestnetError:
            return 0.0


SAFE_BROKERS = ("local", "binance_testnet")


def make_broker(cfg=None):
    """تنها نقطهٔ ساختِ بروکر در کل سیستم.

    هر چیزی جز شبیه‌سازِ محلی و تست‌نت، بدون مجوزِ صریحِ gates رد می‌شود؛ و آداپتورِ
    شبکهٔ اصلی عمداً هنوز نوشته نشده است (فاز ۴ پلن)، پس حتی با مجوز هم استثنا می‌دهد.
    """
    import gates
    cfg = cfg or load_cfg()
    kind = cfg.get("broker", "local")
    if kind not in SAFE_BROKERS:
        if not gates.live_allowed():
            raise gates.GateError(
                f"بروکر «{kind}» مجاز نیست: gates.live_allowed خاموش است یا "
                f"متغیرِ {gates.LIVE_ACK_ENV} با نسخهٔ گیت نمی‌خواند")
        raise NotImplementedError("آداپتورِ شبکهٔ اصلی هنوز پیاده‌سازی نشده است (فاز ۴ پلن)")
    if kind == "binance_testnet" and cfg.get("api_key") and cfg.get("api_secret"):
        return BinanceTestnet(cfg["api_key"], cfg["api_secret"], cfg.get("leverage", 2))
    return None   # حالت شبیه‌ساز محلی
