import asyncio
import time
import logging
import ccxt
import pandas as pd
from core.config import get_settings
from ta.volatility import BollingerBands

class BollingerTradingService:
    """Service to run Day 10's Bollinger Band strategy in the ATS"""
    def __init__(self):
        settings = get_settings()
        # configure logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[logging.FileHandler('bollinger_trader.log'), logging.StreamHandler()]
        )
        self.logger = logging.getLogger('bollinger_trader')

        # config
        self.loop_interval = settings.BOLLINGER_LOOP_INTERVAL
        self.symbol = settings.BOLLINGER_SYMBOL
        self.timeframe = settings.BOLLINGER_TIMEFRAME
        self.window = settings.BOLLINGER_SMA_WINDOW
        self.lookback_days = settings.BOLLINGER_LOOKBACK_DAYS
        self.size = settings.BOLLINGER_ORDER_SIZE
        self.target = settings.BOLLINGER_TARGET_PROFIT
        self.max_loss = settings.BOLLINGER_MAX_LOSS
        self.leverage = settings.BOLLINGER_LEVERAGE
        self.max_positions = settings.BOLLINGER_MAX_POSITIONS
        self.params = {'timeInForce': 'PostOnly'}

        # rate limiting
        self._last_api_call = 0
        self._min_interval = settings.BOT1_MIN_CALL_INTERVAL

        # initialize exchange
        if not settings.DEBUG_MODE:
            if not (settings.PHEMEX_API_KEY and settings.PHEMEX_SECRET_KEY):
                raise ValueError("PHEMEX_API_KEY and PHEMEX_SECRET_KEY required for Bollinger trading")
            self.exchange = ccxt.phemex({
                'enableRateLimit': True,
                'apiKey': settings.PHEMEX_API_KEY,
                'secret': settings.PHEMEX_SECRET_KEY
            })
        else:
            self.exchange = None

    async def start(self):
        self.logger.info(f"🚀 Starting Bollinger Trading Service (symbol={self.symbol})")
        while True:
            try:
                await asyncio.to_thread(self._bot_iteration)
            except Exception as e:
                self.logger.error(f"Error in Bollinger iteration: {e}")
            await asyncio.sleep(self.loop_interval)

    def _api_call(self, func, *args, **kwargs):
        """Rate-limited wrapper"""
        now = time.time()
        if now - self._last_api_call < self._min_interval:
            time.sleep(self._min_interval - (now - self._last_api_call))
        result = func(*args, **kwargs)
        self._last_api_call = time.time()
        return result

    def fetch_ohlcv(self):
        # placeholder: fetch recent candles (lookback_days conversion needed)
        return self._api_call(self.exchange.fetch_ohlcv, self.symbol, self.timeframe, limit=self.window * self.lookback_days * 24)

    def calculate_bands(self, df: pd.DataFrame):
        bb = BollingerBands(df['close'], window=self.window, window_dev=2)
        df['bb_upper'] = bb.bollinger_hband()
        df['bb_mid'] = bb.bollinger_mavg()
        df['bb_lower'] = bb.bollinger_lband()
        df['bandwidth'] = df['bb_upper'] - df['bb_lower']
        # compression = bandwidth below some quantile
        thresh_tight = df['bandwidth'].quantile(0.2)
        return df, df['bandwidth'].iloc[-1] <= thresh_tight

    def open_positions(self):
        """Check for open positions"""
        bal = self._api_call(self.exchange.fetch_balance, {'type':'swap','code':'USD'})
        positions = bal['info']['data']['positions']
        # use first position entry for this symbol
        pos = positions[0]
        side = pos.get('side')
        size = float(pos.get('size', 0))
        has = side in ['Buy', 'Sell']
        is_long = True if side == 'Buy' else False if side == 'Sell' else None
        return positions, has, size, is_long

    def kill_switch(self):
        """Close all open positions gracefully"""
        self.logger.info(f"Starting kill switch for {self.symbol}")
        try:
            _, has, size, is_long = self.open_positions()
            while has:
                self._api_call(self.exchange.cancel_all_orders, self.symbol)
                _, has, size, is_long = self.open_positions()
                ask, bid = self.ask_bid()
                if not is_long:
                    self._api_call(self.exchange.create_limit_buy_order, self.symbol, size, bid, self.params)
                else:
                    self._api_call(self.exchange.create_limit_sell_order, self.symbol, size, ask, self.params)
                time.sleep(30)
        except Exception as e:
            self.logger.error(f"Error in Bollinger kill switch: {e}")

    def pnl_close(self):
        """Monitor PnL and close positions if target or stop loss hit"""
        try:
            pos_list = self._api_call(self.exchange.fetch_positions, {'type':'swap','code':'USD'})
            pos = pos_list[0]
            side = pos.get('side')
            contracts = pos.get('contracts', 0)
            entry = float(pos.get('entryPrice', 0))
            lev = float(pos.get('leverage', 1))
            ask, bid = self.ask_bid()
            current = bid if side.lower() == 'long' else ask
            diff = (current - entry) if side.lower() == 'long' else (entry - current)
            perc = (diff / entry) * lev * 100 if entry else 0
            self.logger.info(f"Current PnL: {perc:.2f}%")
            if perc >= self.target or perc <= self.max_loss:
                self.logger.info("PnL threshold hit, exiting position")
                self._api_call(self.exchange.cancel_all_orders, self.symbol)
                self.kill_switch()
                return True, True, contracts, (side.lower() == 'long')
            return False, False, contracts, (side.lower() == 'long')
        except Exception as e:
            self.logger.error(f"Error in Bollinger pnl_close: {e}")
            return False, False, 0, None

    def place_entry_orders(self, pos_size):
        """Place both buy and sell limit orders at depth-10 prices if available"""
        ob = self._api_call(self.exchange.fetch_order_book, self.symbol)
        bids, asks = ob.get('bids', []), ob.get('asks', [])
        if len(bids) > 10 and len(asks) > 10:
            bid_price = bids[10][0]
            ask_price = asks[10][0]
        else:
            ask_price, bid_price = self.ask_bid()
        self._api_call(self.exchange.cancel_all_orders, self.symbol)
        self._api_call(self.exchange.create_limit_buy_order, self.symbol, pos_size, bid_price, self.params)
        self._api_call(self.exchange.create_limit_sell_order, self.symbol, pos_size, ask_price, self.params)

    def _bot_iteration(self):
        """One iteration: PnL check, band compression entry/exit"""
        # Monitor PnL if in a position
        positions, in_pos, _, _ = self.open_positions()
        if in_pos:
            self.logger.info("Currently in position, monitoring PnL targets")
            self._api_call(self.exchange.cancel_all_orders, self.symbol)
            self.pnl_close()
            return
        # Not in position: check bands
        df = pd.DataFrame(self.fetch_ohlcv(), columns=['timestamp','open','high','low','close','volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df, tight = self.calculate_bands(df)
        if tight:
            self.logger.info("Bands tight and no open position, placing entry orders")
            self.place_entry_orders(self.size)
        else:
            self.logger.info("Bands not tight, cancelling orders and closing any positions")
            self._api_call(self.exchange.cancel_all_orders, self.symbol)
            self.kill_switch() 