# autonomous_trading_system/backend/services/liquidation_hunter_service.py

import asyncio
import logging
import pandas as pd
from datetime import datetime, timedelta

from .hyperliquid_service import HyperliquidService
from .whale_tracking_service import WhaleTrackingService

logger = logging.getLogger(__name__)

class LiquidationHunterService:
    def __init__(self, config, hyperliquid_service: HyperliquidService, whale_tracking_service: WhaleTrackingService):
        self.config = config
        self.hyperliquid_service = hyperliquid_service
        self.whale_tracking_service = whale_tracking_service
        
        self.trading_bias = None  # 'long' or 'short'
        self.target_coin = None # The coin with the most promising liquidation imbalance
        self.last_analysis_time = None
        self.last_analysis_report = {}
        self.analysis_interval = timedelta(minutes=self.config.get("analysis_interval_minutes", 15))
        self.tokens_to_analyze = self.config.get("liquidation_hunter_tokens", ['BTC', 'ETH', 'SOL', 'WIF'])
        
        logger.info("LiquidationHunterService initialized.")

    async def analyze_market_bias(self):
        logger.info("Starting advanced market bias analysis...")
        try:
            positions_df = self.whale_tracking_service.get_whale_positions(source="hyperdash")
            if positions_df.empty:
                logger.warning("No whale positions data available to determine market bias.")
                return False

            # Fetch current prices for all tokens to analyze
            current_prices = {}
            for token in self.tokens_to_analyze:
                ask, bid, _ = self.hyperliquid_service.get_order_book(token)
                if ask and bid:
                    current_prices[token] = (ask + bid) / 2
                else:
                    logger.warning(f"Could not fetch price for {token}. It will be excluded from this analysis cycle.")
            
            if not current_prices:
                logger.error("Could not fetch any token prices. Aborting analysis.")
                return False

            # --- 3% Liquidation Impact Analysis ---
            total_long_liquidations_value = 0
            total_short_liquidations_value = 0
            coin_liquidation_imbalance = {}

            for coin, price in current_prices.items():
                coin_positions = positions_df[positions_df['coin'] == coin].copy()
                if coin_positions.empty:
                    continue
                
                price_move_3_percent = price * 0.03
                
                # Longs liquidated if price moves DOWN
                long_liq_df = coin_positions[
                    (coin_positions['is_long']) & 
                    (coin_positions['liquidation_price'] >= (price - price_move_3_percent))
                ]
                long_liq_value = long_liq_df['position_value'].sum()
                
                # Shorts liquidated if price moves UP
                short_liq_df = coin_positions[
                    (~coin_positions['is_long']) & 
                    (coin_positions['liquidation_price'] <= (price + price_move_3_percent))
                ]
                short_liq_value = short_liq_df['position_value'].sum()

                total_long_liquidations_value += long_liq_value
                total_short_liquidations_value += short_liq_value
                coin_liquidation_imbalance[coin] = {
                    "long_liquidation_value": long_liq_value,
                    "short_liquidation_value": short_liq_value,
                    "imbalance": abs(long_liq_value - short_liq_value)
                }
            
            # Determine overall market bias
            if total_long_liquidations_value > total_short_liquidations_value:
                self.trading_bias = 'short'
                logger.info(f"Market Bias: SHORT (Longs at risk: ${total_long_liquidations_value:,.2f} vs Shorts at risk: ${total_short_liquidations_value:,.2f})")
            else:
                self.trading_bias = 'long'
                logger.info(f"Market Bias: LONG (Shorts at risk: ${total_short_liquidations_value:,.2f} vs Longs at risk: ${total_long_liquidations_value:,.2f})")
                
            # Determine the best coin to target
            if coin_liquidation_imbalance:
                # Find coin with the largest imbalance
                best_coin = max(coin_liquidation_imbalance, key=lambda k: coin_liquidation_imbalance[k]['imbalance'])
                self.target_coin = best_coin
                logger.info(f"Targeting {best_coin} due to highest liquidation imbalance (${coin_liquidation_imbalance[best_coin]['imbalance']:,.2f}).")

            # Store report for potential API endpoint
            self.last_analysis_report = {
                "timestamp": datetime.now().isoformat(),
                "market_bias": self.trading_bias,
                "target_coin": self.target_coin,
                "total_long_liquidation_risk_usd": total_long_liquidations_value,
                "total_short_liquidation_risk_usd": total_short_liquidations_value,
                "coin_analysis": coin_liquidation_imbalance
            }
            self.last_analysis_time = datetime.now()
            return True

        except Exception as e:
            logger.error(f"Error during market analysis: {e}", exc_info=True)
            return False

    async def check_for_trade_entry(self, symbol: str):
        if not self.trading_bias or not symbol:
            logger.info("No trading bias or target symbol set. Skipping trade entry check.")
            return

        logger.info(f"Checking for liquidation-based trade entries for {symbol} with bias: {self.trading_bias.upper()}")
        
        liquidations_df = self.hyperliquid_service.get_liquidations(symbol, lookback_hours=1)
        if liquidations_df.empty:
            return

        # 'side' is 'Sell' for long liquidations, 'Buy' for short liquidations
        long_liq_amount = liquidations_df[liquidations_df['side'] == 'Sell']['usd_size'].sum()
        short_liq_amount = liquidations_df[liquidations_df['side'] == 'Buy']['usd_size'].sum()

        trigger_amount = self.config.get("liquidation_trigger_amount", 100000)

        position = self.hyperliquid_service.get_position(symbol)
        if position.get("in_pos"):
             logger.info(f"Already in position for {symbol}. Skipping new entry.")
             return

        # If bias is LONG, we enter when there's a cascade of LONG liquidations (price dip opportunity)
        if self.trading_bias == 'long' and long_liq_amount > trigger_amount:
            logger.info(f"Significant long liquidations detected (${long_liq_amount:,.2f}). Favorable for LONG entry on {symbol}.")
            await self.execute_trade(symbol, is_buy=True)
            
        # If bias is SHORT, we enter when there's a cascade of SHORT liquidations (price spike opportunity)
        elif self.trading_bias == 'short' and short_liq_amount > trigger_amount:
            logger.info(f"Significant short liquidations detected (${short_liq_amount:,.2f}). Favorable for SHORT entry on {symbol}.")
            await self.execute_trade(symbol, is_buy=False)

    async def execute_trade(self, symbol: str, is_buy: bool):
        logger.info(f"Executing {'BUY' if is_buy else 'SELL'} trade for {symbol}.")
        
        leverage = self.config.get("leverage", 5)
        position_size_usd = self.config.get("position_size_usd", 10)
        
        # Adjust leverage and calculate size
        _, size = self.hyperliquid_service.adjust_leverage(symbol, leverage, usd_size=position_size_usd)
        
        if not size:
            logger.error("Could not calculate position size. Aborting trade.")
            return
            
        # Determine price and place order
        ask, bid, _ = self.hyperliquid_service.get_order_book(symbol)
        if not ask or not bid:
            logger.error("Could not get order book. Aborting trade.")
            return

        price = bid if is_buy else ask
        
        self.hyperliquid_service.limit_order(symbol, is_buy, size, price, reduce_only=False)
        
        # SL/TP logic remains the same, but would ideally be placed after order fill confirmation.
        logger.info("Entry order placed. PnL management will handle exit conditions.")


    async def manage_positions(self):
        # Manage positions for all analyzed tokens, not just the target
        for symbol in self.tokens_to_analyze:
            position = self.hyperliquid_service.get_position(symbol)
            if not position.get("in_pos"):
                continue
                
            sl_percent = self.config.get("stop_loss_percent", -6.0)
            tp_percent = self.config.get("take_profit_percent", 1.0)
            
            pnl_perc = position.get("pnl_perc", 0)
            
            if pnl_perc >= tp_percent or pnl_perc <= sl_percent:
                logger.info(f"Closing position for {symbol} due to PnL ({pnl_perc:.2f}%).")
                self.hyperliquid_service.kill_switch(symbol, market=True)

    async def run_cycle(self):
        # 1. Analyze market bias if interval has passed
        if not self.last_analysis_time or (datetime.now() - self.last_analysis_time) > self.analysis_interval:
            await self.analyze_market_bias()

        # 2. Manage all existing positions
        await self.manage_positions()

        # 3. Check for new trade entries on the dynamically targeted coin
        if self.target_coin:
            await self.check_for_trade_entry(self.target_coin)
        else:
            logger.info("No target coin identified in this cycle.")

    async def start(self):
        logger.info("Starting Liquidation Hunter Service with Advanced Analysis...")
        while True:
            try:
                await self.run_cycle()
            except Exception as e:
                logger.error(f"Error in LiquidationHunterService run_cycle: {e}", exc_info=True)
            
            await asyncio.sleep(self.config.get("cycle_delay_seconds", 60))

    def get_last_analysis(self):
        return self.last_analysis_report
