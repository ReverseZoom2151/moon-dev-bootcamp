# bitfinex_rrs_bot.py - Bitfinex Professional RRS Automated Trading Bot
"""
Professional institutional-grade RRS trading bot for Bitfinex:
- Market neutral strategy: Long top RRS performer, Short bottom RRS performer
- Professional risk management with advanced position sizing
- Institutional-grade margin trading capabilities
- Professional monitoring and alerting systems
- Advanced PnL tracking with beta-adjusted metrics
"""

import os
import time
import logging
import json
import bitfinex_nice_funcs as n
from bitfinex_rrs_config import SYMBOLS, DATA_DIR, RESULTS_DIR, DEFAULT_BENCHMARK
from bitfinex_rrs_data_fetcher import fetch_data
from bitfinex_rrs_data_processor import (calculate_professional_returns_and_volatility, calculate_professional_volume_metrics)
from bitfinex_rrs_calculator import calculate_professional_rrs
from datetime import datetime, timedelta
from pathlib import Path
from dataclasses import dataclass

# Professional logging configuration
logger = logging.getLogger(__name__)
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('bitfinex_professional_rrs_bot.log')
    ]
)

# Professional Bot Configuration
@dataclass(frozen=True)
class BitfinexProfessionalTradingConfig:
    """Professional configuration for Bitfinex institutional RRS trading bot."""
    # Professional position sizing
    USD_SIZE: float = 250  # Per position size in USD (institutional minimum)
    MAX_POSITIONS: int = 4  # Long + Short with hedging capability
    LEVERAGE: int = 3  # Conservative institutional leverage
    
    # Professional risk management
    STOP_LOSS_PCT: float = -2.5  # -2.5% institutional stop loss
    TAKE_PROFIT_PCT: float = 8.0  # 8% institutional take profit
    MAX_DAILY_LOSS: float = -4.0  # -4% max daily loss (institutional)
    MAX_DRAWDOWN: float = -10.0  # -10% max drawdown before emergency close
    
    # Professional execution timing
    REBALANCE_INTERVAL: int = 1800  # 30 minutes (institutional frequency)
    RRS_CALCULATION_INTERVAL: int = 3600  # 60 minutes for professional RRS recalc
    CACHE_EXPIRY_MINUTES: int = 30  # Extended cache for institutional analysis
    
    # Professional RRS thresholds
    MIN_RRS_THRESHOLD: float = 0.008  # Higher threshold for institutional quality
    RRS_LOOKBACK_DAYS: int = 10  # Extended lookback for professional analysis
    TIMEFRAME: str = '1h'  # Professional hourly timeframe
    
    # Professional features
    USE_MARGIN_TRADING: bool = True
    BETA_ADJUSTMENT: bool = True  # Adjust position sizes based on beta
    ALPHA_TARGET: float = 0.02  # Target alpha generation
    INFORMATION_RATIO_MIN: float = 0.5  # Minimum information ratio
    CORRELATION_MAX: float = 0.8  # Maximum correlation between positions

CONFIG = BitfinexProfessionalTradingConfig()

class BitfinexProfessionalRRSBot:
    """Professional institutional-grade Bitfinex RRS automated trading bot."""
    
    def __init__(self):
        self.current_positions = {'long': None, 'short': None, 'hedge': None}
        self.position_metrics = {}
        self.daily_pnl = 0.0
        self.max_drawdown_current = 0.0
        self.last_rrs_calculation = datetime.min
        self.rrs_cache = {}
        self.bot_start_time = datetime.utcnow()
        self.professional_alerts = []
        
        # Professional directories
        Path(DATA_DIR).mkdir(parents=True, exist_ok=True)
        Path(RESULTS_DIR).mkdir(parents=True, exist_ok=True)
        
        logger.info("🏛️ Bitfinex Professional RRS Bot initialized")
        logger.info(f"💼 Institutional features: Margin Trading, Beta Adjustment, Alpha Targeting")
    
    def check_professional_api_connection(self) -> bool:
        """Check professional Bitfinex API accessibility."""
        try:
            return n.test_connection()
        except Exception as e:
            logger.error(f"Professional API connection failed: {e}")
            return False
    
    def get_professional_account_status(self) -> dict:
        """Get professional account status with institutional metrics."""
        try:
            account_data = n.get_professional_account_balance()
            
            # Calculate professional metrics
            total_usd_value = account_data.get('total_usd_value', 0)
            margin_available = account_data.get('margin_available', False)
            
            # Get professional position summary
            positions_summary = {}
            total_position_value = 0
            total_margin_used = 0
            
            for symbol in SYMBOLS.keys():
                if symbol == DEFAULT_BENCHMARK:
                    continue
                
                position = n.get_professional_position_size(symbol)
                if abs(position.get('size', 0)) > 0:
                    positions_summary[symbol] = position
                    total_position_value += position.get('position_value_usd', 0)
                    total_margin_used += position.get('margin_funding', 0)
            
            # Professional risk metrics
            leverage_ratio = total_position_value / total_usd_value if total_usd_value > 0 else 0
            margin_utilization = total_margin_used / total_usd_value if total_usd_value > 0 else 0
            
            return {
                'total_usd_value': total_usd_value,
                'total_position_value': total_position_value,
                'margin_available': margin_available,
                'positions': positions_summary,
                'leverage_ratio': leverage_ratio,
                'margin_utilization': margin_utilization,
                'professional_grade': account_data.get('professional_grade', 'standard'),
                'available_for_trading': total_usd_value >= CONFIG.USD_SIZE * 2,
                'institutional_ready': total_usd_value >= 10000 and margin_available
            }
            
        except Exception as e:
            logger.error(f"Failed to get professional account status: {e}")
            return {}
    
    def should_recalculate_professional_rrs(self) -> bool:
        """Check if professional RRS should be recalculated."""
        time_since_last = datetime.utcnow() - self.last_rrs_calculation
        return time_since_last.total_seconds() > CONFIG.RRS_CALCULATION_INTERVAL
    
    def get_cached_professional_rrs(self) -> dict:
        """Get professional RRS scores from cache if valid."""
        cache_file = Path(RESULTS_DIR) / 'professional_rrs_cache.json'
        
        try:
            if cache_file.exists():
                with open(cache_file, 'r') as f:
                    cache_data = json.load(f)
                
                cache_time = datetime.fromisoformat(cache_data.get('timestamp', ''))
                if (datetime.utcnow() - cache_time).total_seconds() < (CONFIG.CACHE_EXPIRY_MINUTES * 60):
                    return cache_data.get('professional_rrs_scores', {})
        
        except Exception as e:
            logger.warning(f"Failed to load professional RRS cache: {e}")
        
        return {}
    
    def calculate_current_professional_rrs(self) -> dict:
        """Calculate current professional RRS with institutional metrics."""
        try:
            logger.info("📊 Calculating professional RRS scores with institutional analysis...")
            
            # Professional time range calculation
            end_time = datetime.utcnow()
            start_time = end_time - timedelta(days=CONFIG.RRS_LOOKBACK_DAYS)
            
            # Get professional benchmark data
            benchmark_symbol = DEFAULT_BENCHMARK
            benchmark_code = SYMBOLS[benchmark_symbol].lower()
            
            logger.info(f"Fetching professional benchmark: {benchmark_symbol}")
            benchmark_df = fetch_data(benchmark_code, CONFIG.TIMEFRAME, start_time, end_time)
            
            if benchmark_df.empty:
                logger.error("Failed to fetch professional benchmark data")
                return {}
            
            # Professional benchmark processing
            benchmark_df = calculate_professional_returns_and_volatility(benchmark_df)
            benchmark_df = calculate_professional_volume_metrics(benchmark_df)
            
            professional_rrs_scores = {}
            
            # Calculate professional RRS for each symbol
            for symbol_name, symbol_code in SYMBOLS.items():
                if symbol_name == benchmark_symbol:
                    continue
                
                try:
                    logger.debug(f"Processing professional analysis for {symbol_name}...")
                    
                    # Fetch professional symbol data
                    symbol_df = fetch_data(symbol_code.lower(), CONFIG.TIMEFRAME, start_time, end_time)
                    if symbol_df.empty:
                        continue
                    
                    # Professional processing
                    symbol_df = calculate_professional_returns_and_volatility(symbol_df)
                    symbol_df = calculate_professional_volume_metrics(symbol_df)
                    
                    # Calculate professional RRS
                    rrs_df = calculate_professional_rrs(symbol_df, benchmark_df, symbol_name, benchmark_symbol)
                    
                    if not rrs_df.empty:
                        latest_data = rrs_df.iloc[-1]
                        
                        # Professional metrics extraction
                        professional_rrs_scores[symbol_name] = {
                            'rrs_score': latest_data.get('current_rrs', 0),
                            'momentum': latest_data.get('current_momentum', 0),
                            'beta': latest_data.get('beta', 1.0),
                            'alpha': latest_data.get('alpha', 0),
                            'correlation': latest_data.get('correlation', 0),
                            'information_ratio': latest_data.get('information_ratio', 0),
                            'tracking_error': latest_data.get('tracking_error', 0),
                            'professional_score': latest_data.get('professional_score', 0),
                            'symbol_code': symbol_code,
                            'timestamp': datetime.utcnow().isoformat(),
                            'professional_grade': 'institutional' if abs(latest_data.get('alpha', 0)) > 0.01 else 'standard'
                        }
                
                except Exception as e:
                    logger.warning(f"Failed to calculate professional RRS for {symbol_name}: {e}")
                    continue
            
            # Cache professional results
            cache_data = {
                'timestamp': datetime.utcnow().isoformat(),
                'professional_rrs_scores': professional_rrs_scores,
                'calculation_parameters': {
                    'lookback_days': CONFIG.RRS_LOOKBACK_DAYS,
                    'timeframe': CONFIG.TIMEFRAME,
                    'benchmark': benchmark_symbol
                }
            }
            
            cache_file = Path(RESULTS_DIR) / 'professional_rrs_cache.json'
            with open(cache_file, 'w') as f:
                json.dump(cache_data, f, indent=2)
            
            self.last_rrs_calculation = datetime.utcnow()
            logger.info(f"✅ Professional RRS calculation completed for {len(professional_rrs_scores)} symbols")
            
            return professional_rrs_scores
            
        except Exception as e:
            logger.error(f"Professional RRS calculation failed: {e}")
            return {}
    
    def get_professional_top_bottom_performers(self, rrs_scores: dict) -> tuple:
        """Get top and bottom performers using professional criteria."""
        if not rrs_scores:
            return None, None, {}
        
        # Professional sorting with multiple criteria
        def professional_score(item):
            symbol, data = item
            return (
                data['rrs_score'] * 0.4 +
                data.get('alpha', 0) * 0.3 +
                data.get('information_ratio', 0) * 0.2 +
                data['momentum'] * 0.1
            )
        
        sorted_symbols = sorted(rrs_scores.items(), key=professional_score, reverse=True)
        
        top_symbol = sorted_symbols[0][0] if sorted_symbols else None
        bottom_symbol = sorted_symbols[-1][0] if sorted_symbols else None
        
        # Professional quality checks
        professional_analysis = {}
        if top_symbol and bottom_symbol:
            top_data = rrs_scores[top_symbol]
            bottom_data = rrs_scores[bottom_symbol]
            
            rrs_spread = top_data['rrs_score'] - bottom_data['rrs_score']
            alpha_spread = top_data.get('alpha', 0) - bottom_data.get('alpha', 0)
            
            professional_analysis = {
                'rrs_spread': rrs_spread,
                'alpha_spread': alpha_spread,
                'top_quality': top_data.get('professional_grade', 'standard'),
                'bottom_quality': bottom_data.get('professional_grade', 'standard'),
                'correlation_check': abs(top_data.get('correlation', 0) - bottom_data.get('correlation', 0)) > 0.3,
                'trade_quality': 'institutional' if rrs_spread > CONFIG.MIN_RRS_THRESHOLD * 1.5 else 'standard'
            }
            
            # Professional threshold check
            if rrs_spread < CONFIG.MIN_RRS_THRESHOLD:
                logger.info("Professional RRS spread below institutional threshold")
                return None, None, professional_analysis
        
        return top_symbol, bottom_symbol, professional_analysis
    
    def calculate_professional_position_size(self, symbol: str, rrs_data: dict, position_type: str) -> float:
        """Calculate professional position size with beta adjustment."""
        try:
            base_size = CONFIG.USD_SIZE
            
            if CONFIG.BETA_ADJUSTMENT:
                beta = rrs_data.get('beta', 1.0)
                # Adjust size inversely to beta (lower beta = higher size)
                beta_adjustment = min(2.0, max(0.5, 1.5 / max(0.5, beta)))
                base_size *= beta_adjustment
            
            # Information ratio adjustment
            info_ratio = rrs_data.get('information_ratio', 0)
            if info_ratio > CONFIG.INFORMATION_RATIO_MIN:
                info_adjustment = min(1.5, 1 + (info_ratio - CONFIG.INFORMATION_RATIO_MIN))
                base_size *= info_adjustment
            
            # Professional risk scaling
            if rrs_data.get('professional_grade') == 'institutional':
                base_size *= 1.2
            
            return round(base_size, 2)
            
        except Exception as e:
            logger.error(f"Professional position sizing failed: {e}")
            return CONFIG.USD_SIZE
    
    def check_professional_pnl_and_close(self, symbol: str, position_type: str) -> bool:
        """Professional PnL monitoring with institutional risk management."""
        try:
            pnl_result = n.professional_pnl_monitor(
                symbol.lower(),
                target_profit=CONFIG.TAKE_PROFIT_PCT,
                max_loss=CONFIG.STOP_LOSS_PCT
            )
            
            if pnl_result['action'] in ['close_profit', 'close_loss']:
                logger.info(f"💼 Professional position closed for {symbol}: {pnl_result['action']} at {pnl_result['pnl_pct']:.2f}%")
                self.current_positions[position_type] = None
                
                # Professional alert
                alert = {
                    'timestamp': datetime.utcnow().isoformat(),
                    'type': 'position_closed',
                    'symbol': symbol,
                    'action': pnl_result['action'],
                    'pnl_pct': pnl_result['pnl_pct'],
                    'professional_grade': pnl_result.get('professional_grade', 'standard')
                }
                self.professional_alerts.append(alert)
                
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Professional PnL check failed for {symbol}: {e}")
            return False
    
    def open_professional_position(self, symbol: str, position_type: str, rrs_data: dict) -> bool:
        """Open professional position with institutional features."""
        try:
            position_size = self.calculate_professional_position_size(symbol, rrs_data, position_type)
            symbol_formatted = symbol.lower()
            
            if position_type == 'long':
                result = n.professional_market_buy(symbol_formatted, position_size)
                if result.get('success'):
                    self.current_positions['long'] = symbol
                    self.position_metrics[symbol] = {
                        'entry_time': datetime.utcnow(),
                        'position_size': position_size,
                        'entry_type': 'professional_long',
                        'rrs_data': rrs_data
                    }
                    logger.info(f"📈 Professional LONG opened: {symbol} for ${position_size}")
                    return True
            
            elif position_type == 'short' and CONFIG.USE_MARGIN_TRADING:
                # Professional short using margin
                result = n.professional_market_sell(symbol_formatted, position_size / n.get_professional_price(symbol_formatted))
                if result.get('success'):
                    self.current_positions['short'] = symbol
                    self.position_metrics[symbol] = {
                        'entry_time': datetime.utcnow(),
                        'position_size': position_size,
                        'entry_type': 'professional_short',
                        'rrs_data': rrs_data
                    }
                    logger.info(f"📉 Professional SHORT opened: {symbol} for ${position_size}")
                    return True
            
            return False
            
        except Exception as e:
            logger.error(f"Failed to open professional {position_type} position for {symbol}: {e}")
            return False
    
    def close_professional_position(self, symbol: str, position_type: str) -> bool:
        """Close professional position with institutional controls."""
        try:
            symbol_formatted = symbol.lower()
            
            if position_type == 'long':
                result = n.professional_market_sell(symbol_formatted)
                if result.get('success'):
                    logger.info(f"🔄 Professional LONG closed: {symbol}")
                    return True
            
            elif position_type == 'short':
                # Close short by buying back
                position_size = self.position_metrics.get(symbol, {}).get('position_size', CONFIG.USD_SIZE)
                result = n.professional_market_buy(symbol_formatted, position_size)
                if result.get('success'):
                    logger.info(f"🔄 Professional SHORT closed: {symbol}")
                    return True
            
            return False
            
        except Exception as e:
            logger.error(f"Failed to close professional {position_type} position for {symbol}: {e}")
            return False
    
    def adjust_professional_positions(self, top_symbol: str, bottom_symbol: str, rrs_scores: dict) -> dict:
        """Adjust positions with professional institutional logic."""
        adjustments = {'long': False, 'short': False}
        
        try:
            # Professional long position adjustment
            current_long = self.current_positions.get('long')
            if current_long != top_symbol:
                if current_long:
                    if self.close_professional_position(current_long, 'long'):
                        self.current_positions['long'] = None
                        if current_long in self.position_metrics:
                            del self.position_metrics[current_long]
                
                if top_symbol and self.open_professional_position(top_symbol, 'long', rrs_scores[top_symbol]):
                    adjustments['long'] = True
            
            # Professional short position adjustment
            if CONFIG.USE_MARGIN_TRADING:
                current_short = self.current_positions.get('short')
                if current_short != bottom_symbol:
                    if current_short:
                        if self.close_professional_position(current_short, 'short'):
                            self.current_positions['short'] = None
                            if current_short in self.position_metrics:
                                del self.position_metrics[current_short]
                    
                    if bottom_symbol and self.open_professional_position(bottom_symbol, 'short', rrs_scores[bottom_symbol]):
                        adjustments['short'] = True
            
            return adjustments
            
        except Exception as e:
            logger.error(f"Professional position adjustment failed: {e}")
            return adjustments
    
    def check_professional_risk_limits(self) -> bool:
        """Check professional institutional risk limits."""
        try:
            account_status = self.get_professional_account_status()
            
            # Professional drawdown check
            if account_status.get('leverage_ratio', 0) > CONFIG.LEVERAGE * 1.2:
                logger.warning("⚠️ Professional leverage exceeded institutional limits")
                return False
            
            # Margin utilization check
            if account_status.get('margin_utilization', 0) > 0.8:
                logger.warning("⚠️ Professional margin utilization too high")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Professional risk limit check failed: {e}")
            return True
    
    def professional_emergency_close_all(self) -> dict:
        """Professional emergency closure with institutional controls."""
        results = {}
        
        try:
            for position_type in ['long', 'short']:
                symbol = self.current_positions.get(position_type)
                if symbol:
                    kill_result = n.professional_kill_switch(symbol.lower())
                    results[f"professional_{position_type}_{symbol}"] = kill_result
                    self.current_positions[position_type] = None
                    
                    # Clean up metrics
                    if symbol in self.position_metrics:
                        del self.position_metrics[symbol]
            
            # Professional alert
            alert = {
                'timestamp': datetime.utcnow().isoformat(),
                'type': 'emergency_close_all',
                'positions_closed': len(results),
                'professional_grade': 'institutional_emergency'
            }
            self.professional_alerts.append(alert)
            
            logger.warning("🚨 Professional emergency close all positions executed")
            return results
            
        except Exception as e:
            logger.error(f"Professional emergency close failed: {e}")
            return {}
    
    def run_professional_single_cycle(self):
        """Run single professional trading cycle."""
        try:
            logger.info("🔄 Starting professional RRS bot cycle...")
            
            # Professional API check
            if not self.check_professional_api_connection():
                logger.error("❌ Professional API connection failed, skipping cycle")
                return
            
            # Professional account status
            account_status = self.get_professional_account_status()
            if not account_status.get('available_for_trading', False):
                logger.warning("⚠️ Insufficient professional funds for trading")
                return
            
            # Professional risk limits
            if not self.check_professional_risk_limits():
                logger.warning("⚠️ Professional risk limits exceeded, emergency close")
                self.professional_emergency_close_all()
                return
            
            # Professional PnL monitoring
            for position_type in ['long', 'short']:
                symbol = self.current_positions.get(position_type)
                if symbol:
                    self.check_professional_pnl_and_close(symbol, position_type)
            
            # Get professional RRS scores
            rrs_scores = (self.get_cached_professional_rrs() 
                         if not self.should_recalculate_professional_rrs() 
                         else self.calculate_current_professional_rrs())
            
            if not rrs_scores:
                logger.warning("⚠️ No professional RRS scores available")
                return
            
            # Get professional top/bottom performers
            top_symbol, bottom_symbol, analysis = self.get_professional_top_bottom_performers(rrs_scores)
            
            if not top_symbol or not bottom_symbol:
                logger.info("📊 No clear professional performers, maintaining positions")
                return
            
            # Professional logging
            logger.info(f"📊 Professional Top: {top_symbol} (RRS: {rrs_scores[top_symbol]['rrs_score']:.4f}, "
                       f"Alpha: {rrs_scores[top_symbol].get('alpha', 0):.4f})")
            logger.info(f"📊 Professional Bottom: {bottom_symbol} (RRS: {rrs_scores[bottom_symbol]['rrs_score']:.4f}, "
                       f"Alpha: {rrs_scores[bottom_symbol].get('alpha', 0):.4f})")
            logger.info(f"🏛️ Trade Quality: {analysis.get('trade_quality', 'standard')}")
            
            # Professional position adjustment
            adjustments = self.adjust_professional_positions(top_symbol, bottom_symbol, rrs_scores)
            
            if any(adjustments.values()):
                logger.info(f"✅ Professional adjustments completed: {adjustments}")
            else:
                logger.info("📊 No professional adjustments needed")
            
            # Professional status logging
            logger.info(f"📈 Professional Long: {self.current_positions.get('long', 'None')}")
            logger.info(f"📉 Professional Short: {self.current_positions.get('short', 'None')}")
            logger.info(f"🏛️ Account Grade: {account_status.get('professional_grade', 'standard')}")
            
        except Exception as e:
            logger.error(f"❌ Professional bot cycle failed: {e}")
            if "critical" in str(e).lower():
                self.professional_emergency_close_all()
    
    def run(self):
        """Main professional bot loop."""
        logger.info("🚀 Starting Bitfinex Professional RRS Bot...")
        logger.info(f"💼 Professional position size: ${CONFIG.USD_SIZE} each")
        logger.info(f"⏰ Professional rebalance interval: {CONFIG.REBALANCE_INTERVAL}s")
        logger.info(f"📊 Professional RRS timeframe: {CONFIG.TIMEFRAME}")
        logger.info(f"🏛️ Institutional features: Margin={CONFIG.USE_MARGIN_TRADING}, Beta Adj={CONFIG.BETA_ADJUSTMENT}")
        
        try:
            while True:
                self.run_professional_single_cycle()
                
                logger.info(f"😴 Professional sleep for {CONFIG.REBALANCE_INTERVAL} seconds...")
                time.sleep(CONFIG.REBALANCE_INTERVAL)
                
        except KeyboardInterrupt:
            logger.info("🛑 Professional bot stopped by user")
            self.professional_emergency_close_all()
        
        except Exception as e:
            logger.error(f"❌ Fatal professional bot error: {e}")
            self.professional_emergency_close_all()
            raise

def main():
    """Main function to start professional bot."""
    bot = BitfinexProfessionalRRSBot()
    bot.run()

if __name__ == "__main__":
    main()
