"""
Supply/Demand Zone (SDZ) Trading Service V2
---
This service implements the trading logic from the Day 50 `sdz.py` script.
It periodically checks for trading opportunities based on supply and demand zones
for a configured list of tokens.
"""
import asyncio
import logging
import time
import os
from ..utils import nice_funcs_v2 as n

logger = logging.getLogger(__name__)

class SupplyDemandZoneServiceV2:
    def __init__(self, settings):
        self.settings = settings
        self.is_running = False
        self._create_data_dirs()

    def _create_data_dirs(self):
        """Ensure that the data directories for this service exist."""
        os.makedirs(self.settings.SDZ_V2_CSV_DIR, exist_ok=True)
        os.makedirs(self.settings.SDZ_V2_DATA_DIR, exist_ok=True)
        # Ensure the closed positions file exists
        if not os.path.exists(self.settings.SDZ_V2_CLOSED_POSITIONS_TXT):
            with open(self.settings.SDZ_V2_CLOSED_POSITIONS_TXT, 'w') as f:
                pass # Create empty file

    async def start(self):
        """Starts the background trading loop for the service."""
        if not self.settings.ENABLE_SDZ_BOT_V2:
            logger.info("SDZ Trading Service V2 is disabled in settings.")
            return
        
        logger.info("🚀 Starting Supply/Demand Zone Trading Service V2...")
        self.is_running = True
        while self.is_running:
            try:
                await self._run_trade_cycle()
            except Exception as e:
                logger.error(f"Major error in SDZ V2 trade cycle: {e}", exc_info=True)
            
            await asyncio.sleep(30) # Run every 30 seconds

    async def stop(self):
        """Stops the trading service."""
        self.is_running = False
        logger.info("🛑 Stopping Supply/Demand Zone Trading Service V2...")

    async def _run_trade_cycle(self):
        """Executes one full cycle of the trading logic for all configured symbols."""
        logger.info(f"🌙 SDZ V2 Bot Run at {time.strftime('%Y-%m-%d %H:%M:%S')}")
        for symbol in self.settings.SDZ_V2_SYMBOLS:
            try:
                await self._check_zone_and_trade(symbol)
            except Exception as e:
                logger.error(f"Error processing symbol {symbol[:6]} in SDZ V2 cycle: {e}", exc_info=True)
            await asyncio.sleep(5) # Small delay between symbols

    async def _check_zone_and_trade(self, symbol):
        """The core trading logic for a single symbol."""
        logger.info(f"Checking zones for symbol: {symbol}")
        
        trend = await self._get_trend(symbol)
        sell_percentage = self.settings.SDZ_V2_SELL_PERCENTAGE_TRENDING_UP if trend == 'up' else self.settings.SDZ_V2_SELL_PERCENTAGE_TRENDING_DOWN

        position_tokens = n.get_token_balance(self.settings, self.settings.SDZ_V2_TRADING_WALLET_ADDRESS, symbol)
        price = n.get_token_price(self.settings, symbol)
        
        if price is None:
            logger.warning(f"Could not get price for {symbol}, skipping trade check.")
            return

        current_value_usd = position_tokens * price
        zones = n.get_supply_demand_zones(self.settings, symbol, self.settings.SDZ_V2_DAYS_BACK_4_DATA, self.settings.SDZ_V2_TIMEFRAME)

        if zones is None:
            logger.warning(f"Not enough data for S/D zones for {symbol}, skipping.")
            return

        position_pct = (current_value_usd / self.settings.SDZ_V2_POSITION_SIZE_USD) if self.settings.SDZ_V2_POSITION_SIZE_USD > 0 else 0
        
        # Decision Logic
        in_demand_zone = zones['dz'].min() <= price <= zones['dz'].max()
        in_supply_zone = zones['sz'].min() <= price <= zones['sz'].max()

        if in_demand_zone:
            logger.info(f"Price for {symbol} is in demand zone.")
            await self._handle_demand_zone(symbol, trend, current_value_usd)
        elif in_supply_zone:
            logger.info(f"Price for {symbol} is in supply zone.")
            await self._handle_supply_zone(symbol, trend, position_tokens, current_value_usd, sell_percentage)
        else:
            logger.info(f"Price for {symbol} is not in a defined zone.")

    async def _get_trend(self, symbol):
        """Determines the trend for a symbol based on SMA."""
        price = n.get_token_price(self.settings, symbol)
        if price is None: return 'down'

        df = n.get_historical_data(self.settings, symbol, self.settings.SDZ_V2_SMA_DAYS_BACK, self.settings.SDZ_V2_SMA_TIMEFRAME)
        if df.empty or len(df) < self.settings.SDZ_V2_SMA_BARS:
            return 'down'
        
        sma = df['Close'].tail(self.settings.SDZ_V2_SMA_BARS).mean()
        sma_lower_bound = sma * (1 - self.settings.SDZ_V2_SMA_BUFFER_PCT)
        
        return 'up' if price > sma_lower_bound else 'down'

    async def _handle_demand_zone(self, symbol, trend, current_value_usd):
        """Logic for when price is in the demand zone."""
        if trend == 'up' and current_value_usd < (self.settings.SDZ_V2_POSITION_SIZE_USD * (1 - self.settings.SDZ_V2_BUFFER_PCT)):
            amount_to_buy_usd = self.settings.SDZ_V2_POSITION_SIZE_USD - current_value_usd
            logger.info(f"Trend is UP. Buying ${amount_to_buy_usd:.2f} of {symbol}.")
            for _ in range(self.settings.SDZ_V2_ORDERS_PER_OPEN):
                n.market_buy(self.settings, symbol, amount_to_buy_usd)
                await asyncio.sleep(1)
        else:
            logger.info(f"Not buying {symbol}. Trend: {trend}, Current Value: ${current_value_usd:.2f}")

    async def _handle_supply_zone(self, symbol, trend, position_tokens, current_value_usd, sell_percentage):
        """Logic for when price is in the supply zone."""
        if position_tokens > 0 and current_value_usd > (self.settings.SDZ_V2_POSITION_SIZE_USD * self.settings.SDZ_V2_MINIMUM_POSITION_PCT):
            decimals = n.get_token_decimals(self.settings, symbol)
            if decimals is None:
                logger.error(f"Cannot sell {symbol}, failed to get decimals.")
                return
            
            sell_amount_tokens = position_tokens * sell_percentage
            sell_amount_atomic = int(sell_amount_tokens * (10**decimals))
            
            logger.info(f"Trend is {trend}. Selling {sell_percentage*100}% of {symbol} position.")
            
            for _ in range(self.settings.SDZ_V2_ORDERS_PER_SELL):
                n.market_sell(self.settings, symbol, sell_amount_atomic)
                await asyncio.sleep(1)
        else:
            logger.info(f"Not selling {symbol}. Position Value: ${current_value_usd:.2f}")
