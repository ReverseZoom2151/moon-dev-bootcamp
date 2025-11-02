import asyncio
import ccxt
import pandas as pd
import logging
from core.config import get_settings

class SMATradingService:
    """Service to run the Day 6 SMA trading bot natively in the ATS"""
    def __init__(self):
        settings = get_settings()
        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(settings.SMA_LOG_FILE),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger('sma_trader')
        self.demo = settings.SMA_DEMO_MODE
        self.symbols = settings.SMA_SYMBOLS
        self.timeframe = settings.SMA_TIMEFRAME
        self.limit = settings.SMA_LIMIT
        self.period = settings.SMA_PERIOD
        self.target = settings.SMA_TARGET_PROFIT_PCT
        self.max_loss = settings.SMA_MAX_LOSS_PCT
        self.max_risk = settings.SMA_MAX_RISK_AMOUNT
        self.order_params = settings.SMA_ORDER_PARAMS
        self.risk_interval = settings.SMA_RISK_CHECK_INTERVAL
        self.trade_interval = settings.SMA_TRADE_CYCLE_INTERVAL
        # Heartbeat and manual wait settings
        self.heartbeat_interval = settings.SMA_HEARTBEAT_INTERVAL
        self.wait_time = settings.SMA_WAIT_TIME

        if not self.demo:
            # Initialize Phemex exchange via CCXT
            if not (settings.PHEMEX_API_KEY and settings.PHEMEX_SECRET_KEY):
                raise ValueError("PHEMEX_API_KEY and PHEMEX_SECRET_KEY required for SMA trading")
            self.exchange = ccxt.phemex({
                'enableRateLimit': True,
                'apiKey': settings.PHEMEX_API_KEY,
                'secret': settings.PHEMEX_SECRET_KEY
            })
        else:
            self.exchange = None

    async def start(self):
        self.logger.info(f"🚀 Starting SMA Trading Service (demo={self.demo})")
        if not self.demo:
            await self._test_api_connection()
            # Initial risk check
            await asyncio.to_thread(self._check_risk_exposure)
            # Initial SMA analysis
            await asyncio.to_thread(self._initial_analysis)
        # Launch risk, trading, and heartbeat loops
        await asyncio.gather(
            self._risk_loop(),
            self._trade_loop(),
            self._heartbeat_loop()
        )

    async def _test_api_connection(self):
        try:
            # Test public endpoint
            symbol = self.symbols[0]
            ticker = await asyncio.to_thread(self.exchange.fetch_ticker, symbol)
            print(f"✅ Public API OK: {ticker['symbol']}")
            # Test private endpoint
            bal = await asyncio.to_thread(self.exchange.fetch_balance)
            print("✅ Private API OK: balance fetched")
        except Exception as e:
            print(f"❌ SMA API connection failed: {e}")
            raise

    async def _risk_loop(self):
        while True:
            if not self.demo:
                await asyncio.to_thread(self._check_risk_exposure)
            else:
                print("[DEMO] Skipping SMA risk check")
            await asyncio.sleep(self.risk_interval)

    def _check_risk_exposure(self):
        # Sum position costs and enforce max_risk
        bal = self.exchange.fetch_balance({'type':'swap','code':'USD'})
        positions = bal['info']['data']['positions']
        total = sum(float(p.get('posCost', 0)) for p in positions)
        print(f"🔒 SMA Risk Exposure: ${total:.2f} (max ${self.max_risk})")
        if total > self.max_risk:
            print("⚠️ Exposure exceeded, closing all positions")
            for sym in self.symbols:
                self._close_position(sym)

    async def _trade_loop(self):
        while True:
            for sym in self.symbols:
                await asyncio.to_thread(self._run_cycle, sym)
            await asyncio.sleep(self.trade_interval)

    def _run_cycle(self, symbol):
        # Determine open position
        positions, has_pos, size, is_long = self._get_position(symbol)
        if has_pos:
            self._check_pnl(symbol)
        else:
            self._enter_trade(symbol)

    def _get_position(self, symbol):
        if self.demo:
            # Demo: no positions
            return [], False, 0, False
        bal = self.exchange.fetch_balance({'type':'swap','code':'USD'})
        positions = bal['info']['data']['positions']
        for pos in positions:
            if pos.get('symbol') == symbol:
                sz = float(pos.get('size', 0))
                if sz:
                    return positions, True, sz, pos.get('side') == 'Buy'
        return positions, False, 0, False

    def _check_pnl(self, symbol):
        # Check PnL percent and close if thresholds hit
        positions = self.exchange.fetch_positions({'type':'swap','code':'USD'})
        for pos in positions:
            if pos.get('symbol') == symbol:
                side = pos.get('side')
                entry = float(pos.get('entryPrice', 0))
                leverage = float(pos.get('leverage', 1))
                current = float(self.exchange.fetch_ticker(symbol)['last'])
                diff = (current - entry) if side == 'long' else (entry - current)
                perc = (diff / entry) * leverage * 100 if entry else 0
                print(f"🔍 SMA P&L for {symbol}: {perc:.2f}%")
                if perc >= self.target or perc <= self.max_loss:
                    print(f"⚠️ SMA P&L out of bounds ({perc:.2f}%), closing {symbol}")
                    self._close_position(symbol)
                return

    def _enter_trade(self, symbol):
        # Calculate SMA and place limit order
        print(f"📈 SMA enter trade for {symbol}")
        bars = self.exchange.fetch_ohlcv(symbol, self.timeframe, limit=self.limit)
        df = pd.DataFrame(bars, columns=['ts','o','h','l','c','v'])
        df['sma'] = df['c'].rolling(self.period).mean()
        last_sma = df['sma'].iloc[-1]
        bid = self.exchange.fetch_order_book(symbol)['bids'][0][0]
        ask = self.exchange.fetch_order_book(symbol)['asks'][0][0]
        signal = 'BUY' if last_sma < bid else ('SELL' if last_sma > ask else None)
        if signal == 'BUY':
            print(f"🔄 Placing BUY {symbol} @ {bid}")
            self.exchange.create_limit_buy_order(symbol, self.settings.SMA_ORDER_PARAMS.get('size',1), bid, self.order_params)
        elif signal == 'SELL':
            print(f"🔄 Placing SELL {symbol} @ {ask}")
            self.exchange.create_limit_sell_order(symbol, self.settings.SMA_ORDER_PARAMS.get('size',1), ask, self.order_params)

    def _close_position(self, symbol):
        # Cancel orders and place opposite order
        orders = self.exchange.fetch_open_orders(symbol)
        for o in orders:
            self.exchange.cancel_order(o['id'], symbol)
        _, has_pos, size, is_long = self._get_position(symbol)
        if not has_pos:
            print(f"✅ {symbol} already closed")
            return
        bid = self.exchange.fetch_order_book(symbol)['bids'][0][0]
        ask = self.exchange.fetch_order_book(symbol)['asks'][0][0]
        if is_long:
            print(f"🔄 Closing long: SELL {size}@{ask}")
            self.exchange.create_limit_sell_order(symbol, size, ask, self.order_params)
        else:
            print(f"🔄 Closing short: BUY {size}@{bid}")
            self.exchange.create_limit_buy_order(symbol, size, bid, self.order_params)

    def _initial_analysis(self):
        """Perform initial SMA calculation for all symbols"""
        for symbol in self.symbols:
            try:
                bars = self.exchange.fetch_ohlcv(symbol, self.timeframe, limit=self.limit)
                df = pd.DataFrame(bars, columns=['ts','o','h','l','c','v'])
                df['sma'] = df['c'].rolling(self.period).mean()
                if not df['sma'].isna().all():
                    last = df['sma'].iloc[-1]
                    self.logger.info(f"Initial SMA for {symbol}: {last:.2f}")
            except Exception as e:
                self.logger.error(f"Initial SMA analysis failed for {symbol}: {e}")

    async def _heartbeat_loop(self):
        """Periodic heartbeat log to indicate service is alive"""
        while True:
            self.logger.info("SMA heartbeat - service alive")
            await asyncio.sleep(self.heartbeat_interval) 