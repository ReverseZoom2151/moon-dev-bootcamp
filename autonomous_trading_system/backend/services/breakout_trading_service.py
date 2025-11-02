import asyncio
import time
import logging
import ccxt
import pandas as pd
from core.config import get_settings

class BreakoutTradingService:
    """Service to run Day 11's Breakout trading logic in the ATS"""
    def __init__(self):
        settings = get_settings()
        # configure logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[logging.FileHandler('breakout_trader.log'), logging.StreamHandler()]
        )
        self.logger = logging.getLogger('breakout_trader')

        # configuration from settings
        self.loop_interval = settings.BREAKOUT_LOOP_INTERVAL
        self.symbol = settings.BREAKOUT_SYMBOL
        self.pos_size = settings.BREAKOUT_POS_SIZE
        self.target_gain = settings.BREAKOUT_TARGET_GAIN
        self.max_loss = settings.BREAKOUT_MAX_LOSS
        self.timeframe = settings.BREAKOUT_TIMEFRAME
        self.limit = settings.BREAKOUT_LIMIT
        self.pause_time = settings.BREAKOUT_PAUSE_TIME
        self.price_threshold = getattr(settings, 'BREAKOUT_PRICE_THRESHOLD', 0.0)
        self.params = settings.BREAKOUT_ORDER_PARAMS

        # rate limiting
        self._last_api_call = 0
        self._min_call_interval = getattr(settings, 'BREAKOUT_MIN_CALL_INTERVAL', 1.0)

        # initialize CCXT Phemex exchange
        if not settings.DEBUG:
            if not (settings.PHEMEX_API_KEY and settings.PHEMEX_SECRET_KEY):
                raise ValueError("PHEMEX_API_KEY and PHEMEX_SECRET_KEY required for Breakout trading")
            self.exchange = ccxt.phemex({
                'enableRateLimit': True,
                'apiKey': settings.PHEMEX_API_KEY,
                'secret': settings.PHEMEX_SECRET_KEY
            })
        else:
            self.exchange = None

    async def start(self):
        self.logger.info(f"🚀 Starting Breakout Trading Service (symbol={self.symbol})")
        while True:
            try:
                await asyncio.to_thread(self._bot_iteration)
            except Exception as e:
                self.logger.error(f"Error in Breakout iteration: {e}")
            await asyncio.sleep(self.loop_interval)

    def _api_call(self, func, *args, **kwargs):
        now = time.time()
        if now - self._last_api_call < self._min_call_interval:
            time.sleep(self._min_call_interval - (now - self._last_api_call))
        result = func(*args, **kwargs)
        self._last_api_call = time.time()
        return result

    def ask_bid(self):
        ob = self._api_call(self.exchange.fetch_order_book, self.symbol)
        bid = ob.get('bids', [[None]])[0][0]
        ask = ob.get('asks', [[None]])[0][0]
        return ask, bid

    def open_positions(self):
        bal = self._api_call(self.exchange.fetch_balance, {'type':'swap','code':'USD'})
        positions = bal['info']['data'].get('positions', [])
        if positions:
            pos = positions[0]
            side = pos.get('side')
            size = float(pos.get('size', 0))
            has = side in ['Buy', 'Sell']
            is_long = side == 'Buy'
        else:
            has, size, is_long = False, 0, None
        return positions, has, size, is_long

    def pnl_close(self):
        # Monitor PnL and close position when thresholds are hit
        pos_list = self._api_call(self.exchange.fetch_positions, {'type':'swap','code':'USD'})
        if not pos_list:
            return False
        pos = pos_list[0]
        side = pos.get('side')
        entry = float(pos.get('entryPrice', 0))
        lev = float(pos.get('leverage', 1))
        ask, bid = self.ask_bid()
        current = bid if side.lower() == 'buy' else ask
        diff = (current - entry) if side.lower() == 'buy' else (entry - current)
        perc = (diff / entry) * lev * 100 if entry else 0
        if perc >= self.target_gain or perc <= self.max_loss:
            self.logger.info(f"PnL threshold reached: {perc:.2f}% - closing position")
            self._api_call(self.exchange.cancel_all_orders, self.symbol)
            # close position
            if side.lower() == 'buy':
                self._api_call(self.exchange.create_limit_sell_order, self.symbol, pos.get('size'), ask, self.params)
            else:
                self._api_call(self.exchange.create_limit_buy_order, self.symbol, pos.get('size'), bid, self.params)
            return True
        return False

    def _bot_iteration(self):
        # Check for existing position
        positions, in_pos, size, is_long = self.open_positions()
        if in_pos:
            closed = self.pnl_close()
            if closed:
                time.sleep(self.pause_time * 60)
            return

        # Not in position: fetch market data and compute support/resistance
        ask, bid = self.ask_bid()
        bars = self._api_call(self.exchange.fetch_ohlcv, self.symbol, self.timeframe, limit=self.limit)
        df = pd.DataFrame(bars, columns=['timestamp','open','high','low','close','volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        support = df['close'].min()
        resistance = df['close'].max()
        self.logger.info(f"{self.symbol} bid={bid}, support={support}, resistance={resistance}")

        # Check for breakout and place orders
        if bid > resistance:
            price = resistance * (1 + self.price_threshold)
            self.logger.info(f"BREAKOUT detected. Placing BUY @ {price}")
            self._api_call(self.exchange.cancel_all_orders, self.symbol)
            self._api_call(self.exchange.create_limit_buy_order, self.symbol, self.pos_size, price, self.params)
            time.sleep(self.pause_time * 60)
        elif bid < support:
            price = support * (1 - self.price_threshold)
            self.logger.info(f"BREAKDOWN detected. Placing SELL @ {price}")
            self._api_call(self.exchange.cancel_all_orders, self.symbol)
            self._api_call(self.exchange.create_limit_sell_order, self.symbol, self.pos_size, price, self.params)
            time.sleep(self.pause_time * 60)
        else:
            self.logger.info("No breakout/breakdown - waiting next cycle") 