import asyncio
import time
import logging
from eth_account import Account
from core.config import get_settings
from services.trading_utils import (
    ask_bid, get_position, pnl_close, cancel_all_orders,
    limit_order, supply_demand_zones_hl, get_latest_sma, adjust_leverage_size_signal
)

class SupplyDemandTradingService:
    """Service to run Day 11 Supply/Demand Zone trading logic in the ATS"""
    def __init__(self):
        settings = get_settings()
        if not settings.ENABLE_SDZ_BOT:
            raise ValueError("ENABLE_SDZ_BOT must be True to start Supply/Demand trading service")
        if not settings.HYPERLIQUID_API_KEY or not settings.HYPERLIQUID_PRIVATE_KEY:
            raise ValueError("HYPERLIQUID_API_KEY and HYPERLIQUID_PRIVATE_KEY must be set in .env for Supply/Demand trading")
        # configure logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[logging.FileHandler('sdz_trader.log'), logging.StreamHandler()]
        )
        self.logger = logging.getLogger('sdz_trader')

        # settings
        self.loop_interval = settings.SDZ_LOOP_INTERVAL
        self.symbol = settings.SDZ_SYMBOL
        self.timeframe = settings.SDZ_TIMEFRAME
        self.sma_window = settings.SDZ_SMA_WINDOW
        self.sma_lookback_days = settings.SDZ_SMA_LOOKBACK_DAYS
        self.limit_days = settings.SDZ_LIMIT
        self.pos_size = settings.SDZ_POS_SIZE
        self.target = settings.SDZ_TARGET_PROFIT
        self.max_loss = settings.SDZ_MAX_LOSS
        self.leverage = settings.SDZ_LEVERAGE
        self.max_positions = settings.SDZ_MAX_POSITIONS
        self.pause_time = settings.SDZ_PAUSE_TIME  # minutes
        self.params = settings.SDZ_ORDER_PARAMS

        # initialize account for Hyperliquid
        self.account = Account.from_key(settings.HYPERLIQUID_PRIVATE_KEY)
        # set leverage
        try:
            adjust_leverage_size_signal(self.symbol, self.leverage, self.account)
            self.logger.info(f"Leverage set to {self.leverage}")
        except Exception as e:
            self.logger.error(f"Error setting leverage: {e}")

    async def start(self):
        self.logger.info(f"🚀 Starting Supply/Demand Trading Service (symbol={self.symbol})")
        while True:
            try:
                await asyncio.to_thread(self._bot_iteration)
            except Exception as e:
                self.logger.error(f"Error in SDZ iteration: {e}")
            await asyncio.sleep(self.loop_interval)

    def _bot_iteration(self):
        # Check existing position
        positions, in_pos, size, pos_sym, entry_px, pnl_perc, long = get_position(self.symbol, self.account)
        self.logger.info(f"Current positions: {positions}")
        if in_pos:
            # Manage PnL
            closed = pnl_close(self.symbol, self.target, self.max_loss, self.account)
            if closed:
                time.sleep(self.pause_time * 60)
            return

        # Handle partial position
        if 0 < size < self.pos_size:
            remaining = self.pos_size - size
            self.logger.info(f"Partial position detected ({size}), adjusting size to {remaining}")
            self.pos_size = remaining

        # Optionally log latest SMA
        try:
            sma = get_latest_sma(self.symbol, self.timeframe, self.sma_window, self.sma_lookback_days)
            self.logger.info(f"Latest SMA: {sma}")
        except Exception:
            pass

        # Get current market price (ask)
        ask, bid = ask_bid(self.symbol)
        current_price = ask

        # Calculate supply/demand zones
        try:
            zones = supply_demand_zones_hl(self.symbol, self.timeframe, self.limit_days)
            self.logger.info(f"Supply/Demand zones:\n{zones}")
            buy_zone = float(zones[f'{self.timeframe}_dz'].mean())
            sell_zone = float(zones[f'{self.timeframe}_sz'].mean())
        except Exception as e:
            self.logger.error(f"Error calculating zones: {e}")
            return

        # Determine closest zone and place order
        diff_buy = abs(current_price - buy_zone)
        diff_sell = abs(current_price - sell_zone)
        # Cancel existing orders
        cancel_all_orders(self.account)
        if diff_buy < diff_sell:
            self.logger.info(f"Placing BUY order at {buy_zone}")
            limit_order(self.symbol, True, self.pos_size, buy_zone, False, self.account)
        else:
            self.logger.info(f"Placing SELL order at {sell_zone}")
            limit_order(self.symbol, False, self.pos_size, sell_zone, False, self.account) 