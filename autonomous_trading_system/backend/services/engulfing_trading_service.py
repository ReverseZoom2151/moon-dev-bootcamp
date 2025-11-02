import asyncio
import time
import logging
import ccxt
import pandas as pd
from core.config import get_settings
from services.trading_utils import ask_bid, df_sma, validate_symbol
from datetime import datetime

class EngulfingTradingService:
    """Service to run Day 12 Engulfing Candle strategy in the ATS"""
    def __init__(self):
        settings = get_settings()
        # Trading parameters from config
        self.loop_interval = settings.ENGULFING_LOOP_INTERVAL
        self.symbol = validate_symbol(settings.ENGULFING_SYMBOL)
        self.pos_size = settings.ENGULFING_POS_SIZE
        self.target = settings.ENGULFING_TARGET_PROFIT
        self.max_loss = settings.ENGULFING_MAX_LOSS
        self.timeframe = settings.ENGULFING_TIMEFRAME
        self.limit = settings.ENGULFING_LIMIT
        self.sma_period = settings.ENGULFING_SMA_PERIOD
        self.retry_delay = settings.ENGULFING_RETRY_DELAY
        self.params = settings.ENGULFING_ORDER_PARAMS

        # Recent order tracking
        self.recent_order = False
        self.recent_order_time = None

        # Initialize Phemex via CCXT
        if not settings.DEBUG_MODE:
            if not (settings.PHEMEX_API_KEY and settings.PHEMEX_SECRET_KEY):
                raise ValueError("PHEMEX_API_KEY and PHEMEX_SECRET_KEY must be set for Engulfing bot")
            self.exchange = ccxt.phemex({
                'enableRateLimit': True,
                'apiKey': settings.PHEMEX_API_KEY,
                'secret': settings.PHEMEX_SECRET_KEY
            })
        else:
            self.exchange = None

        # Set up logger
        self.logger = logging.getLogger('engulfing_trader')
        self.logger.info(f"Configured Engulfing bot for {self.symbol}")

    async def start(self):
        """Main loop for the Engulfing trading service"""
        self.logger.info("🚀 Starting Engulfing Trading Service...")
        while True:
            try:
                await asyncio.to_thread(self._bot_iteration)
            except Exception as e:
                self.logger.error(f"Error in Engulfing iteration: {e}")
                time.sleep(self.retry_delay)
            await asyncio.sleep(self.loop_interval)

    def _api_call(self, func, *args, **kwargs):
        now = time.time()
        # Basic rate limiter reuse if needed
        result = func(*args, **kwargs)
        return result

    def open_positions(self):
        """Fetch current positions for the symbol"""
        try:
            positions = self._api_call(self.exchange.fetch_positions, [self.symbol])
            for pos in positions:
                if pos.get('symbol') == self.symbol and float(pos.get('contracts', 0)) > 0:
                    size = float(pos['contracts'])
                    is_long = pos.get('side') == 'long'
                    entry = float(pos.get('entryPrice', 0))
                    pnl = float(pos.get('unrealizedPnl', 0))
                    return positions, True, size, is_long, entry, pnl
        except Exception as e:
            self.logger.error(f"Error fetching positions: {e}")
        return [], False, 0, False, 0.0, 0.0

    def pnl_close(self):
        """Close position on PnL target or stop loss using market orders"""
        positions, in_pos, size, is_long, entry, pnl = self.open_positions()
        if not in_pos or entry == 0:
            return False
        try:
            ticker = self._api_call(self.exchange.fetch_ticker, self.symbol)
            current = ticker.get('last')
            if is_long:
                pnl_pct = ((current / entry) - 1) * 100
            else:
                pnl_pct = ((entry / current) - 1) * 100
            if pnl_pct >= self.target or pnl_pct <= self.max_loss:
                self.logger.info(f"PnL {pnl_pct:.2f}% hit, closing position")
                # Cancel existing orders
                self.exchange.cancel_all_orders(self.symbol)
                # Market close
                if is_long:
                    self.exchange.create_market_sell_order(self.symbol, size)
                else:
                    self.exchange.create_market_buy_order(self.symbol, size)
                return True
        except Exception as e:
            self.logger.error(f"Error in PnL close: {e}")
        return False

    def detect_engulfing_pattern(self, df: pd.DataFrame) -> str:
        """Detect bullish/bearish engulfing in the last two completed candles"""
        if len(df) < 3:
            return None
        prev = df.iloc[-3]
        curr = df.iloc[-2]
        prev_body = abs(prev['close'] - prev['open'])
        curr_body = abs(curr['close'] - curr['open'])
        prev_bull = prev['close'] > prev['open']
        curr_bull = curr['close'] > curr['open']
        if (not prev_bull and curr_bull and curr_body > prev_body
                and curr['open'] <= prev['close'] and curr['close'] >= prev['open']):
            return 'bullish'
        if (prev_bull and not curr_bull and curr_body > prev_body
                and curr['open'] >= prev['close'] and curr['close'] <= prev['open']):
            return 'bearish'
        return None

    def create_order(self, order_type: str, price: float):
        """Place a limit order for entry"""
        try:
            self.exchange.cancel_all_orders(self.symbol)
            if order_type == 'buy':
                self.exchange.create_limit_buy_order(self.symbol, self.pos_size, price, self.params)
            else:
                self.exchange.create_limit_sell_order(self.symbol, self.pos_size, price, self.params)
            self.recent_order = True
            self.recent_order_time = datetime.now()
        except Exception as e:
            self.logger.error(f"Error placing {order_type} order: {e}")

    def _bot_iteration(self):
        """One iteration of the trading logic"""
        # Enforce 2-minute cooldown after recent order
        if self.recent_order and self.recent_order_time:
            elapsed = (datetime.now() - self.recent_order_time).total_seconds()
            if elapsed < 120:
                return
            self.recent_order = False

        # PnL management
        closed = self.pnl_close()
        if closed:
            time.sleep(self.retry_delay)
            return

        # Fetch indicators and price data
        df = df_sma(self.symbol, self.timeframe, self.limit, self.sma_period)
        if df.empty:
            return
        sma_val = df[f'sma_{self.sma_period}'].iloc[-1]
        ask, bid = ask_bid(self.symbol)
        pattern = self.detect_engulfing_pattern(df)
        # If in position, skip entry
        _, in_pos, size, _, _, _ = self.open_positions()
        if in_pos and size > 0:
            return

        # Entry logic
        if pattern == 'bullish' and bid > sma_val:
            price = bid * 0.999
            self.logger.info(f"Bullish engulfing + SMA, BUY @ {price}")
            self.create_order('buy', price)
        elif pattern == 'bearish' and ask < sma_val:
            price = ask * 1.001
            self.logger.info(f"Bearish engulfing + SMA, SELL @ {price}")
            self.create_order('sell', price) 