# binance_rrs_bot.py - Binance RRS Automated Trading Bot
"""
This bot implements a market neutral RRS strategy:
- Longs the top RRS performer 
- Shorts the bottom RRS performer (via selling and rebuying)
- Maintains risk management with stop losses and take profits
- Automatically rebalances based on RRS changes
"""

import os
import time
import logging
import pandas as pd
import json
import binance_nice_funcs as n
from binance_rrs_config import SYMBOLS, DATA_DIR, RESULTS_DIR, DEFAULT_BENCHMARK
from binance_rrs_data_fetcher import fetch_data
from binance_rrs_data_processor import calculate_returns_and_volatility, calculate_volume_metrics
from binance_rrs_calculator import calculate_rrs
from datetime import datetime, timedelta
from pathlib import Path
from dataclasses import dataclass

# Configure logging
logger = logging.getLogger(__name__)
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('binance_rrs_bot.log')
    ]
)

# Bot Configuration
@dataclass(frozen=True)
class BinanceTradingConfig:
    """Configuration for Binance RRS trading bot."""
    # Position sizing
    USDT_SIZE: float = 100  # Per position size in USDT
    MAX_POSITIONS: int = 2  # Long + Short position
    
    # Risk management
    STOP_LOSS_PCT: float = -3.0  # -3% stop loss
    TAKE_PROFIT_PCT: float = 6.0  # 6% take profit
    MAX_DAILY_LOSS: float = -5.0  # -5% max daily loss
    
    # Execution timing
    REBALANCE_INTERVAL: int = 900  # 15 minutes in seconds
    RRS_CALCULATION_INTERVAL: int = 1800  # 30 minutes for RRS recalc
    CACHE_EXPIRY_MINUTES: int = 15
    
    # RRS thresholds
    MIN_RRS_THRESHOLD: float = 0.005  # Minimum RRS difference to trade
    RRS_LOOKBACK_DAYS: int = 7
    TIMEFRAME: str = '15m'

CONFIG = BinanceTradingConfig()

class BinanceRRSBot:
    """Binance RRS automated trading bot."""
    
    def __init__(self):
        self.current_positions = {'long': None, 'short': None}
        self.daily_pnl = 0.0
        self.last_rrs_calculation = datetime.min
        self.rrs_cache = {}
        self.bot_start_time = datetime.utcnow()
        
        # Ensure directories exist
        Path(DATA_DIR).mkdir(parents=True, exist_ok=True)
        Path(RESULTS_DIR).mkdir(parents=True, exist_ok=True)
        
        logger.info("🚀 Binance RRS Bot initialized")
    
    def check_api_connection(self) -> bool:
        """Check if Binance API is accessible."""
        try:
            return n.test_connection()
        except Exception as e:
            logger.error(f"API connection check failed: {e}")
            return False
    
    def get_account_status(self) -> dict:
        """Get current account status and balances."""
        try:
            balances = n.get_account_balance()
            usdt_balance = balances.get('USDT', {}).get('total', 0)
            
            # Check current positions
            positions_summary = {}
            total_position_value = 0
            
            for symbol in SYMBOLS.keys():
                if symbol == DEFAULT_BENCHMARK:
                    continue
                
                position = n.get_position_size(f"{symbol}USDT")
                if position.get('size', 0) > 0:
                    positions_summary[symbol] = position
                    total_position_value += position.get('position_value_usdt', 0)
            
            return {
                'usdt_balance': usdt_balance,
                'total_position_value': total_position_value,
                'positions': positions_summary,
                'available_for_trading': usdt_balance >= CONFIG.USDT_SIZE * 2
            }
            
        except Exception as e:
            logger.error(f"Failed to get account status: {e}")
            return {}
    
    def should_recalculate_rrs(self) -> bool:
        """Check if RRS should be recalculated."""
        time_since_last = datetime.utcnow() - self.last_rrs_calculation
        return time_since_last.total_seconds() > CONFIG.RRS_CALCULATION_INTERVAL
    
    def get_cached_rrs(self) -> dict:
        """Get RRS scores from cache if still valid."""
        cache_file = Path(RESULTS_DIR) / 'rrs_cache.json'
        
        try:
            if cache_file.exists():
                with open(cache_file, 'r') as f:
                    cache_data = json.load(f)
                
                cache_time = datetime.fromisoformat(cache_data.get('timestamp', ''))
                if (datetime.utcnow() - cache_time).total_seconds() < (CONFIG.CACHE_EXPIRY_MINUTES * 60):
                    return cache_data.get('rrs_scores', {})
        
        except Exception as e:
            logger.warning(f"Failed to load RRS cache: {e}")
        
        return {}
    
    def calculate_current_rrs(self) -> dict:
        """Calculate current RRS for all symbols against benchmark."""
        try:
            logger.info("📊 Calculating current RRS scores...")
            
            # Calculate time range
            end_time = datetime.utcnow()
            start_time = end_time - timedelta(days=CONFIG.RRS_LOOKBACK_DAYS)
            
            # Get benchmark data
            benchmark_symbol = DEFAULT_BENCHMARK
            benchmark_code = SYMBOLS[benchmark_symbol]
            
            logger.info(f"Fetching benchmark data: {benchmark_symbol}")
            benchmark_df = fetch_data(benchmark_code, CONFIG.TIMEFRAME, start_time, end_time)
            
            if benchmark_df.empty:
                logger.error("Failed to fetch benchmark data")
                return {}
            
            # Process benchmark data
            benchmark_df = calculate_returns_and_volatility(benchmark_df)
            benchmark_df = calculate_volume_metrics(benchmark_df)
            
            rrs_scores = {}
            
            # Calculate RRS for each symbol
            for symbol_name, symbol_code in SYMBOLS.items():
                if symbol_name == benchmark_symbol:
                    continue
                
                try:
                    logger.debug(f"Processing {symbol_name}...")
                    
                    # Fetch symbol data
                    symbol_df = fetch_data(symbol_code, CONFIG.TIMEFRAME, start_time, end_time)
                    if symbol_df.empty:
                        continue
                    
                    # Process symbol data
                    symbol_df = calculate_returns_and_volatility(symbol_df)
                    symbol_df = calculate_volume_metrics(symbol_df)
                    
                    # Calculate RRS
                    rrs_df = calculate_rrs(symbol_df, benchmark_df, symbol_name, benchmark_symbol)
                    
                    if not rrs_df.empty:
                        current_rrs = rrs_df['current_rrs'].iloc[-1]
                        momentum = rrs_df.get('current_momentum', pd.Series([0])).iloc[-1]
                        
                        rrs_scores[symbol_name] = {
                            'rrs_score': current_rrs,
                            'momentum': momentum,
                            'symbol_code': symbol_code,
                            'timestamp': datetime.utcnow().isoformat()
                        }
                
                except Exception as e:
                    logger.warning(f"Failed to calculate RRS for {symbol_name}: {e}")
                    continue
            
            # Cache the results
            cache_data = {
                'timestamp': datetime.utcnow().isoformat(),
                'rrs_scores': rrs_scores
            }
            
            cache_file = Path(RESULTS_DIR) / 'rrs_cache.json'
            with open(cache_file, 'w') as f:
                json.dump(cache_data, f, indent=2)
            
            self.last_rrs_calculation = datetime.utcnow()
            logger.info(f"✅ RRS calculation completed for {len(rrs_scores)} symbols")
            
            return rrs_scores
            
        except Exception as e:
            logger.error(f"RRS calculation failed: {e}")
            return {}
    
    def get_top_bottom_performers(self, rrs_scores: dict) -> tuple:
        """Get top and bottom RRS performers."""
        if not rrs_scores:
            return None, None
        
        # Sort by RRS score
        sorted_symbols = sorted(rrs_scores.items(), 
                              key=lambda x: x[1]['rrs_score'], 
                              reverse=True)
        
        top_symbol = sorted_symbols[0][0] if sorted_symbols else None
        bottom_symbol = sorted_symbols[-1][0] if sorted_symbols else None
        
        # Check minimum threshold
        if (top_symbol and bottom_symbol and 
            abs(rrs_scores[top_symbol]['rrs_score'] - rrs_scores[bottom_symbol]['rrs_score']) < CONFIG.MIN_RRS_THRESHOLD):
            logger.info("RRS difference below threshold, no trade signal")
            return None, None
        
        return top_symbol, bottom_symbol
    
    def check_pnl_and_close(self, symbol: str, position_type: str) -> bool:
        """Check PnL and close position if targets hit."""
        try:
            pnl_result = n.pnl_monitor(f"{symbol}USDT", 
                                     target_profit=CONFIG.TAKE_PROFIT_PCT, 
                                     max_loss=CONFIG.STOP_LOSS_PCT)
            
            if pnl_result['action'] in ['close_profit', 'close_loss']:
                logger.info(f"💰 Position closed for {symbol}: {pnl_result['action']} at {pnl_result['pnl_pct']:.2f}%")
                self.current_positions[position_type] = None
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"PnL check failed for {symbol}: {e}")
            return False
    
    def open_position(self, symbol: str, position_type: str) -> bool:
        """Open a new position."""
        try:
            symbol_pair = f"{symbol}USDT"
            
            if position_type == 'long':
                result = n.market_buy(symbol_pair, CONFIG.USDT_SIZE)
                if result.get('success'):
                    self.current_positions['long'] = symbol
                    logger.info(f"📈 Opened LONG position: {symbol} for ${CONFIG.USDT_SIZE}")
                    return True
            
            elif position_type == 'short':
                # For spot trading, we simulate short by not holding the asset
                # This is a simplified approach - real shorting requires margin
                self.current_positions['short'] = symbol
                logger.info(f"📉 Marked SHORT position: {symbol} (spot trading simulation)")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Failed to open {position_type} position for {symbol}: {e}")
            return False
    
    def close_position(self, symbol: str, position_type: str) -> bool:
        """Close an existing position."""
        try:
            if position_type == 'long':
                symbol_pair = f"{symbol}USDT"
                result = n.market_sell(symbol_pair)
                if result.get('success'):
                    logger.info(f"🔄 Closed LONG position: {symbol}")
                    return True
            
            elif position_type == 'short':
                # For simulated short, just clear the position
                logger.info(f"🔄 Closed SHORT position: {symbol}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Failed to close {position_type} position for {symbol}: {e}")
            return False
    
    def adjust_positions(self, top_symbol: str, bottom_symbol: str) -> dict:
        """Adjust positions based on current RRS rankings."""
        adjustments = {'long': False, 'short': False}
        
        try:
            # Check if we need to adjust long position
            current_long = self.current_positions.get('long')
            if current_long != top_symbol:
                # Close current long if exists
                if current_long:
                    if self.close_position(current_long, 'long'):
                        self.current_positions['long'] = None
                
                # Open new long position
                if top_symbol and self.open_position(top_symbol, 'long'):
                    adjustments['long'] = True
            
            # Check if we need to adjust short position  
            current_short = self.current_positions.get('short')
            if current_short != bottom_symbol:
                # Close current short if exists
                if current_short:
                    if self.close_position(current_short, 'short'):
                        self.current_positions['short'] = None
                
                # Open new short position
                if bottom_symbol and self.open_position(bottom_symbol, 'short'):
                    adjustments['short'] = True
            
            return adjustments
            
        except Exception as e:
            logger.error(f"Position adjustment failed: {e}")
            return adjustments
    
    def check_daily_limits(self) -> bool:
        """Check if daily loss limits are exceeded."""
        # This is a simplified version - in production you'd track actual P&L
        try:
            account_status = self.get_account_status()
            # Simplified daily limit check based on position values
            if account_status.get('total_position_value', 0) < (CONFIG.USDT_SIZE * 2 * 0.95):  # 5% loss threshold
                logger.warning("⚠️ Daily loss limit approaching")
                return False
            return True
        except:
            return True
    
    def emergency_close_all(self) -> dict:
        """Emergency close all positions."""
        results = {}
        
        try:
            for position_type in ['long', 'short']:
                symbol = self.current_positions.get(position_type)
                if symbol:
                    kill_result = n.kill_switch(f"{symbol}USDT")
                    results[f"{position_type}_{symbol}"] = kill_result
                    self.current_positions[position_type] = None
            
            logger.warning("🚨 Emergency close all positions executed")
            return results
            
        except Exception as e:
            logger.error(f"Emergency close failed: {e}")
            return {}
    
    def run_single_cycle(self):
        """Run a single trading cycle."""
        try:
            logger.info("🔄 Starting RRS bot cycle...")
            
            # Check API connection
            if not self.check_api_connection():
                logger.error("❌ API connection failed, skipping cycle")
                return
            
            # Check account status
            account_status = self.get_account_status()
            if not account_status.get('available_for_trading', False):
                logger.warning("⚠️ Insufficient funds for trading")
                return
            
            # Check daily limits
            if not self.check_daily_limits():
                logger.warning("⚠️ Daily limits exceeded, emergency close")
                self.emergency_close_all()
                return
            
            # Check current positions for PnL management
            for position_type in ['long', 'short']:
                symbol = self.current_positions.get(position_type)
                if symbol and position_type == 'long':  # Only monitor actual positions
                    self.check_pnl_and_close(symbol, position_type)
            
            # Get or calculate RRS scores
            rrs_scores = self.get_cached_rrs() if not self.should_recalculate_rrs() else self.calculate_current_rrs()
            
            if not rrs_scores:
                logger.warning("⚠️ No RRS scores available, skipping position adjustment")
                return
            
            # Get top and bottom performers
            top_symbol, bottom_symbol = self.get_top_bottom_performers(rrs_scores)
            
            if not top_symbol or not bottom_symbol:
                logger.info("📊 No clear top/bottom performers, maintaining positions")
                return
            
            logger.info(f"📊 Top performer: {top_symbol} (RRS: {rrs_scores[top_symbol]['rrs_score']:.4f})")
            logger.info(f"📊 Bottom performer: {bottom_symbol} (RRS: {rrs_scores[bottom_symbol]['rrs_score']:.4f})")
            
            # Adjust positions
            adjustments = self.adjust_positions(top_symbol, bottom_symbol)
            
            if any(adjustments.values()):
                logger.info(f"✅ Position adjustments completed: {adjustments}")
            else:
                logger.info("📊 No position adjustments needed")
            
            # Log current status
            logger.info(f"📈 Current Long: {self.current_positions.get('long', 'None')}")
            logger.info(f"📉 Current Short: {self.current_positions.get('short', 'None')}")
            
        except Exception as e:
            logger.error(f"❌ Bot cycle failed: {e}")
            # In case of critical error, consider emergency close
            if "critical" in str(e).lower():
                self.emergency_close_all()
    
    def run(self):
        """Main bot loop."""
        logger.info("🚀 Starting Binance RRS Bot...")
        logger.info(f"💰 Position size: ${CONFIG.USDT_SIZE} each")
        logger.info(f"⏰ Rebalance interval: {CONFIG.REBALANCE_INTERVAL}s")
        logger.info(f"📊 RRS timeframe: {CONFIG.TIMEFRAME}")
        
        try:
            while True:
                self.run_single_cycle()
                
                logger.info(f"😴 Sleeping for {CONFIG.REBALANCE_INTERVAL} seconds...")
                time.sleep(CONFIG.REBALANCE_INTERVAL)
                
        except KeyboardInterrupt:
            logger.info("🛑 Bot stopped by user")
            self.emergency_close_all()
        
        except Exception as e:
            logger.error(f"❌ Fatal bot error: {e}")
            self.emergency_close_all()
            raise

def main():
    """Main function to start the bot."""
    bot = BinanceRRSBot()
    bot.run()

if __name__ == "__main__":
    main()
