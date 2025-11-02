import asyncio
import json
import requests
from eth_account import Account
from core.config import get_settings
from hyperliquid.info import Info
from hyperliquid.utils import constants
from hyperliquid.exchange import Exchange

class HyperliquidRiskService:
    """Service to monitor Hyperliquid positions and enforce risk limits"""
    def __init__(self, settings=None):
        self.settings = settings or get_settings()
        self.demo = self.settings.HYPERLIQUID_RISK_DEMO_MODE
        self.symbol = getattr(self.settings, 'HYPERLIQUID_SYMBOL', None) or 'WIF'
        self.target = self.settings.HYPERLIQUID_RISK_TARGET
        self.max_loss = self.settings.HYPERLIQUID_RISK_MAX_LOSS
        self.acct_min = self.settings.HYPERLIQUID_RISK_ACCT_MIN
        self.check_interval = self.settings.HYPERLIQUID_RISK_CHECK_INTERVAL
        if not self.demo and not self.settings.HYPERLIQUID_PRIVATE_KEY:
            raise ValueError("HYPERLIQUID_PRIVATE_KEY required for Hyperliquid risk service")
        self.info = Info(constants.MAINNET_API_URL, skip_ws=True)
        if not self.demo:
            self.account = Account.from_key(self.settings.HYPERLIQUID_PRIVATE_KEY)
            self.exchange = Exchange(self.account, constants.MAINNET_API_URL)
        else:
            self.account = None
            self.exchange = None

    async def start(self):
        print(f"🔒 Starting HyperliquidRiskService (demo={self.demo})")
        while True:
            try:
                await self._check_pnl_hl()
                await self._check_acct_min()
            except Exception as e:
                print(f"❌ Error in HyperliquidRiskService: {e}")
            await asyncio.sleep(self.check_interval)

    async def _check_pnl_hl(self):
        if self.demo:
            print("[DEMO] Hyperliquid PnL check - no real positions")
            return
        try:
            state = await asyncio.to_thread(lambda: self.info.user_state(self.info.address))
            positions = state.get('positions', [])
            for p in positions:
                coin = p['position']['coin']
                if coin != self.symbol:
                    continue
                pnl = float(p['position'].get('returnOnEquity',0)) * 100
                print(f"Hyperliquid PnL for {coin}: {pnl:.2f}%")
                if pnl >= self.target or pnl <= self.max_loss:
                    print(f"⚠️ HL Risk: PnL {pnl:.2f}% out of bounds, kill switch...")
                    await self._kill_switch(coin)
        except Exception as e:
            print(f"Error in HL PnL check: {e}")

    async def _check_acct_min(self):
        if self.demo:
            print("[DEMO] Hyperliquid acct min check")
            return
        try:
            state = await asyncio.to_thread(lambda: self.info.user_state(self.info.address))
            acct_val = float(state['marginSummary'].get('accountValue',0))
            print(f"Hyperliquid account value: {acct_val}")
            if acct_val < self.acct_min:
                print(f"⚠️ HL Risk: Account value {acct_val} < {self.acct_min}, kill switch...")
                await self._kill_switch(self.symbol)
        except Exception as e:
            print(f"Error in HL acct check: {e}")

    async def _kill_switch(self, symbol):
        """Close position for symbol with retries"""
        if self.demo:
            print(f"[DEMO] Hyperliquid kill switch for {symbol}")
            return
        max_attempts = 5
        attempt = 0
        while attempt < max_attempts:
            attempt += 1
            print(f"⚠️ HL Risk: kill switch attempt {attempt}/{max_attempts} for {symbol}")
            # cancel existing orders
            await self._cancel_orders()
            # fetch position info
            _, has_pos, size, is_long = await self._get_position_info(symbol)
            if not has_pos:
                print(f"✅ Position closed for {symbol}")
                return
            ask, bid = await self._get_ask_bid(symbol)
            if ask == 0 or bid == 0:
                print("⚠️ Invalid prices, retry in 5s")
                await asyncio.sleep(self.check_interval)
                continue
            if is_long:
                print(f"🔄 Closing long: SELL {symbol} {size}@{ask}")
                await asyncio.to_thread(self.exchange.order, symbol, False, size, ask, {"limit":{"tif":"Gtc"}}, True)
            else:
                print(f"🔄 Closing short: BUY {symbol} {size}@{bid}")
                await asyncio.to_thread(self.exchange.order, symbol, True, size, bid, {"limit":{"tif":"Gtc"}}, True)
            await asyncio.sleep(self.check_interval)
        print(f"❌ Failed to close position for {symbol} after {max_attempts} attempts")

    async def _cancel_orders(self):
        """Cancel all open orders for this account"""
        try:
            orders = await asyncio.to_thread(self.info.open_orders, self.account.address)
            for order in orders:
                try:
                    await asyncio.to_thread(self.exchange.cancel, order['coin'], order['oid'])
                    print(f"Cancelled {order['coin']} order {order['oid']}")
                except Exception as e:
                    print(f"Error cancelling order {order['oid']}: {e}")
        except Exception as e:
            print(f"Error fetching open orders: {e}")

    async def _get_position_info(self, symbol):
        """Fetch current position info for a symbol"""
        if self.demo:
            return [], False, 0, False
        state = await asyncio.to_thread(self.info.user_state, self.account.address)
        positions = state.get('positions', [])
        for p in positions:
            pos = p.get('position', {})
            if pos.get('coin') == symbol and float(pos.get('szi', 0)) != 0:
                size = abs(float(pos.get('szi')))
                # assume long side for positive size
                return positions, True, size, True
        return positions, False, 0, False

    async def _get_ask_bid(self, symbol):
        """Fetch ask/bid via Hyperliquid L2 book"""
        try:
            url = 'https://api.hyperliquid.xyz/info'
            headers = {'Content-Type': 'application/json'}
            data = json.dumps({"type": "l2Book", "coin": symbol})
            resp = await asyncio.to_thread(requests.post, url, headers=headers, data=data)
            resp.raise_for_status()
            result = resp.json()
            levels = result.get('levels', [])
            bid = float(levels[0][0]['px'])
            ask = float(levels[0][1]['px'])
            return ask, bid
        except Exception as e:
            print(f"Error fetching L2 book: {e}")
            return 0, 0 