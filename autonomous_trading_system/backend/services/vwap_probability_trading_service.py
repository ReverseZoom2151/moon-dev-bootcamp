import asyncio
import time
import random
import logging
from eth_account import Account
from core.config import get_settings
from services.trading_utils import (
    ask_bid, calculate_vwap_with_symbol,
    get_position_andmaxpos, adjust_leverage_size_signal,
    cancel_all_orders, limit_order, pnl_close
)

class VwapProbabilityTradingService:
    """Service to run Day 12 VWAP probability bot logic in the ATS"""
    def __init__(self):
        settings = get_settings()
        if not settings.ENABLE_VWAP_PROBABILITY_BOT:
            raise ValueError("ENABLE_VWAP_PROBABILITY_BOT must be True to start VWAP Probability bot")
        self.loop_interval = settings.VWAP_PROBABILITY_LOOP_INTERVAL
        self.symbol = settings.VWAP_PROBABILITY_SYMBOL
        self.timeframe = settings.VWAP_PROBABILITY_TIMEFRAME
        self.sma_window = settings.VWAP_PROBABILITY_SMA_WINDOW
        self.lookback_days = settings.VWAP_PROBABILITY_LOOKBACK_DAYS
        self.base_pos_size = settings.VWAP_PROBABILITY_POS_SIZE
        self.target = settings.VWAP_PROBABILITY_TARGET_PROFIT
        self.max_loss = settings.VWAP_PROBABILITY_MAX_LOSS
        self.leverage = settings.VWAP_PROBABILITY_LEVERAGE
        self.max_positions = settings.VWAP_PROBABILITY_MAX_POSITIONS
        self.prob_above = settings.VWAP_PROBABILITY_LONG_PROB_ABOVE_VWAP
        self.prob_below = settings.VWAP_PROBABILITY_LONG_PROB_BELOW_VWAP
        self.error_sleep = settings.VWAP_PROBABILITY_ERROR_SLEEP_TIME
        self.params = settings.VWAP_PROBABILITY_ORDER_PARAMS

        # Set up logging
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger('vwap_probability_trader')

        # Initialize account for Hyperliquid
        self.account = Account.from_key(settings.HYPERLIQUID_PRIVATE_KEY)

    async def start(self):
        """Main loop for the VWAP Probability trading service"""
        self.logger.info(f"🚀 Starting VWAP Probability Trading Service (symbol={self.symbol})")
        while True:
            try:
                await asyncio.to_thread(self._bot_iteration)
            except Exception as e:
                self.logger.error(f"Error in VWAP Probability iteration: {e}")
                time.sleep(self.error_sleep)
            await asyncio.sleep(self.loop_interval)

    def _bot_iteration(self):
        # Check positions and enforce max positions
        positions, in_pos, size, pos_sym, entry_px, pnl_pct, is_long, num_pos = \
            get_position_andmaxpos(self.symbol, self.account, self.max_positions)
        self.logger.info(f"Positions: {positions}, in_pos={in_pos}, size={size}, num_pos={num_pos}")
        if in_pos:
            # PnL management
            closed = pnl_close(self.symbol, self.target, self.max_loss, self.account)
            if closed:
                time.sleep(self.error_sleep)
            return

        # Adjust leverage and calculate position size
        try:
            lev, pos_size = adjust_leverage_size_signal(self.symbol, self.leverage, self.account)
            if pos_size <= 0:
                self.logger.warning(f"Calculated pos_size {pos_size} invalid, using base {self.base_pos_size}")
                pos_size = self.base_pos_size
        except Exception as e:
            self.logger.error(f"Error adjusting leverage: {e}")
            pos_size = self.base_pos_size

        # Cancel any open orders
        cancel_all_orders(self.account)

        # Fetch market data
        ask, bid, levels = ask_bid(self.symbol)
        if ask is None or bid is None or not levels:
            self.logger.error("Failed to fetch order book levels")
            time.sleep(self.error_sleep)
            return
        # Extract 11th level prices
        try:
            bid11 = float(levels[0][10]['px'])
            ask11 = float(levels[1][10]['px'])
        except Exception:
            self.logger.error("Insufficient order book depth for 11th level")
            time.sleep(self.error_sleep)
            return

        # Calculate VWAP
        _, latest_vwap = calculate_vwap_with_symbol(self.symbol)
        if latest_vwap is None:
            self.logger.error("Failed to calculate VWAP")
            time.sleep(self.error_sleep)
            return

        # Determine trade direction by probability
        rand = random.random()
        if bid > latest_vwap:
            going_long = rand <= self.prob_above
            self.logger.info(f"Price {bid} > VWAP {latest_vwap}, rand={rand:.2f}, long={going_long}")
        else:
            going_long = rand <= self.prob_below
            self.logger.info(f"Price {bid} <= VWAP {latest_vwap}, rand={rand:.2f}, long={going_long}")

        # Execute order if not in position
        if not in_pos:
            if going_long:
                self.logger.info(f"Placing BUY order size={pos_size} @ {bid11}")
                limit_order(self.symbol, True, pos_size, bid11, False, self.account)
            else:
                self.logger.info(f"Placing SELL order size={pos_size} @ {ask11}")
                limit_order(self.symbol, False, pos_size, ask11, False, self.account) 