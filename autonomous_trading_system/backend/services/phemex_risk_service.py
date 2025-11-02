import asyncio
import ccxt
from core.config import get_settings

class PhemexRiskService:
    """Service to monitor Phemex positions and enforce risk limits"""
    def __init__(self, settings=None):
        self.settings = settings or get_settings()
        self.demo = self.settings.PHEMEX_RISK_DEMO_MODE
        self.symbol = self.settings.PHEMEX_RISK_SYMBOL
        self.check_interval = self.settings.PHEMEX_RISK_CHECK_INTERVAL
        self.target = self.settings.PHEMEX_RISK_TARGET
        self.max_loss = self.settings.PHEMEX_RISK_MAX_LOSS
        self.max_risk = self.settings.PHEMEX_RISK_MAX_RISK

        if not self.demo:
            if not self.settings.PHEMEX_API_KEY or not self.settings.PHEMEX_SECRET_KEY:
                raise ValueError("PHEMEX_API_KEY/SECRET_KEY required for Phemex risk service")
            self.phemex = ccxt.phemex({
                'enableRateLimit': True,
                'apiKey': self.settings.PHEMEX_API_KEY,
                'secret': self.settings.PHEMEX_SECRET_KEY
            })
        else:
            self.phemex = None

    async def start(self):
        print(f"🔒 Starting PhemexRiskService (demo={self.demo})")
        while True:
            try:
                await self._check_size_risk()
                await self._check_pnl()
            except Exception as e:
                print(f"❌ Error in PhemexRiskService: {e}")
            await asyncio.sleep(self.check_interval)

    async def _check_size_risk(self):
        if self.demo:
            print("[DEMO] Checking size risk - no real positions")
            return
        try:
            bal = await asyncio.to_thread(lambda: self.phemex.fetch_balance({'type':'swap','code':'USD'}))
            positions = bal['info']['data']['positions']
            for pos in positions:
                cost = float(pos.get('posCost', 0))
                symbol = pos.get('symbol')
                if cost > self.max_risk:
                    print(f"⚠️ PhemexRisk: {symbol} cost {cost} exceeds {self.max_risk}, closing...")
                    await self._kill_switch(symbol)
        except Exception as e:
            print(f"Error checking size risk: {e}")

    async def _check_pnl(self):
        if self.demo:
            print("[DEMO] Checking PnL risk - no real positions")
            return
        try:
            # fetch positions
            bal = await asyncio.to_thread(lambda: self.phemex.fetch_positions({'type':'swap','code':'USD'}))
            if not bal:
                return
            positions = bal
            # find our symbol position
            for pos in positions:
                if pos.get('symbol') == self.symbol:
                    side = pos.get('side')
                    entry = float(pos.get('entryPrice',0))
                    leverage = float(pos.get('leverage',1))
                    contracts = float(pos.get('contracts',0))
                    # get current price
                    ob = await asyncio.to_thread(lambda: self.phemex.fetch_ticker(self.symbol))
                    current = float(ob['last'])
                    # calc pnl%
                    diff = (current-entry) if side=='long' else (entry-current)
                    pnl_pct = (diff/entry)*leverage*100 if entry>0 else 0
                    print(f"PnL for {self.symbol}: {pnl_pct:.2f}%")
                    if pnl_pct >= self.target or pnl_pct <= self.max_loss:
                        print(f"⚠️ PhemexRisk: PnL {pnl_pct:.2f}% out of bounds, closing...")
                        await self._kill_switch(self.symbol)
        except Exception as e:
            print(f"Error checking PnL risk: {e}")

    async def _get_position_info(self, symbol):
        """Fetch and parse position info for a symbol"""
        if self.demo:
            return None, False, 0, None
        bal = await asyncio.to_thread(self.phemex.fetch_balance, {'type':'swap','code':'USD'})
        positions = bal['info']['data']['positions']
        for pos in positions:
            if pos.get('symbol') == symbol:
                side = pos.get('side')
                size = float(pos.get('size', 0))
                has_pos = size != 0
                is_long = True if side == 'Buy' else False
                return positions, has_pos, size, is_long
        return positions, False, 0, None

    async def _get_market_data(self, symbol):
        """Fetch current ask and bid prices"""
        try:
            ob = await asyncio.to_thread(self.phemex.fetch_order_book, symbol)
            bid = ob.get('bids', [[0,0]])[0][0]
            ask = ob.get('asks', [[0,0]])[0][0]
            return ask, bid
        except Exception:
            return 0, 0

    async def _kill_switch(self, symbol):
        """Close open positions for symbol with retries"""
        if self.demo:
            print(f"[DEMO] Phemex kill switch for {symbol}")
            return
        max_attempts = 5
        attempt = 0
        while attempt < max_attempts:
            attempt += 1
            print(f"⚠️ PhemexRisk: kill switch attempt {attempt}/{max_attempts} for {symbol}")
            # cancel existing orders
            await asyncio.to_thread(self.phemex.cancel_all_orders, symbol)
            # fetch updated position info
            _, has_pos, size, is_long = await self._get_position_info(symbol)
            if not has_pos or size == 0:
                print(f"✅ Position closed for {symbol}")
                return
            # get market data
            ask, bid = await self._get_market_data(symbol)
            if ask == 0 or bid == 0:
                print("⚠️ Invalid market data, retrying in 5s")
                await asyncio.sleep(5)
                continue
            # place closing order
            if is_long:
                print(f"🔄 Closing long position: SELL {size}@{ask}")
                await asyncio.to_thread(self.phemex.create_limit_sell_order, symbol, size, ask, {'timeInForce':'PostOnly'})
            else:
                print(f"🔄 Closing short position: BUY {size}@{bid}")
                await asyncio.to_thread(self.phemex.create_limit_buy_order, symbol, size, bid, {'timeInForce':'PostOnly'})
            # wait for fill
            await asyncio.sleep(30)
        print(f"❌ Failed to close position after {max_attempts} attempts for {symbol}") 