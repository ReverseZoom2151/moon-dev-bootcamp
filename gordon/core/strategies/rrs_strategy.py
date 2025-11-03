"""
RRS-Based Trading Strategy
==========================
Trading strategy based on Relative Rotation Strength (RRS) analysis.
"""

import asyncio
import logging
from typing import Dict, Optional, Any
from datetime import datetime

from .base import BaseStrategy

try:
    from ...research.rrs import RRSManager
    RRS_AVAILABLE = True
except ImportError:
    RRS_AVAILABLE = False
    RRSManager = None

logger = logging.getLogger(__name__)


class RRSStrategy(BaseStrategy):
    """
    RRS-based trading strategy.
    
    Uses Relative Rotation Strength analysis to identify
    strong/weak assets relative to a benchmark.
    """
    
    def __init__(self, name: str = "RRS_Strategy"):
        """Initialize RRS strategy."""
        super().__init__(name)
        
        if not RRS_AVAILABLE:
            logger.error("RRS module not available. Install required dependencies.")
        
        self.rrs_manager = None
        
        # Default configuration
        self.default_config = {
            'symbols': ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'SOLUSDT'],
            'benchmark': 'BTCUSDT',
            'timeframe': '1h',
            'lookback_days': 7,
            'strong_threshold': 1.0,
            'weak_threshold': -1.0,
            'min_confidence': 0.6,
            'position_size': 1.0,
            'max_positions': 5,
            'schedule_interval_minutes': 60
        }
    
    async def initialize(self, config: Dict):
        """Initialize strategy with configuration."""
        await super().initialize(config)
        
        if not RRS_AVAILABLE:
            raise RuntimeError("RRS module not available")
        
        # Merge with default config
        for key, value in self.default_config.items():
            if key not in self.config:
                self.config[key] = value
        
        # Initialize RRS manager
        data_fetcher = config.get('data_fetcher')
        if not data_fetcher:
            logger.error("Data fetcher required for RRS strategy")
            raise ValueError("data_fetcher required")
        
        self.rrs_manager = RRSManager(
            data_fetcher=data_fetcher,
            config={
                'strong_threshold': self.config['strong_threshold'],
                'weak_threshold': self.config['weak_threshold'],
                'default_benchmark': self.config['benchmark'],
                'timeframes': {
                    self.config['timeframe']: self.config['lookback_days']
                }
            }
        )
        
        self.logger.info(
            f"RRS Strategy initialized with {len(self.config['symbols'])} symbols, "
            f"benchmark={self.config['benchmark']}"
        )
    
    async def execute(self) -> Optional[Dict]:
        """
        Execute RRS strategy logic.
        
        Returns:
            Trading signal or None
        """
        if not self.rrs_manager:
            self.logger.error("RRS manager not initialized")
            return None
        
        try:
            # Run RRS analysis
            symbols = self.config.get('symbols', [])
            timeframe = self.config.get('timeframe', '1h')
            
            self.logger.info(f"Running RRS analysis for {len(symbols)} symbols")
            
            # Analyze symbols
            rrs_results = self.rrs_manager.analyze_multiple_symbols(
                symbols=symbols,
                timeframe=timeframe,
                lookback_days=self.config.get('lookback_days', 7),
                benchmark=self.config.get('benchmark', 'BTCUSDT')
            )
            
            if not rrs_results:
                self.logger.warning("No RRS results generated")
                return None
            
            # Generate rankings and signals
            signals_df = self.rrs_manager.generate_rankings_and_signals(
                rrs_results,
                timeframe
            )
            
            if signals_df.empty:
                self.logger.warning("No signals generated")
                return None
            
            # Find best signal
            strong_buy_signals = signals_df[
                signals_df['primary_signal'] == 'STRONG_BUY'
            ]
            
            if strong_buy_signals.empty:
                # Fall back to BUY signals
                buy_signals = signals_df[signals_df['primary_signal'] == 'BUY']
                
                if buy_signals.empty:
                    self.logger.info("No buy signals found")
                    return None
                
                best_signal = buy_signals.iloc[0]
            else:
                # Get highest confidence strong buy
                best_signal = strong_buy_signals.iloc[0]
            
            # Check minimum confidence
            min_confidence = self.config.get('min_confidence', 0.6)
            if best_signal['signal_confidence'] < min_confidence:
                self.logger.info(
                    f"Signal confidence {best_signal['signal_confidence']:.2f} "
                    f"below threshold {min_confidence}"
                )
                return None
            
            # Generate trading signal
            symbol = best_signal['symbol']
            confidence = float(best_signal['signal_confidence'])
            
            return {
                'action': 'BUY',
                'symbol': symbol,
                'size': self.config.get('position_size', 1.0),
                'confidence': min(confidence, 1.0),
                'metadata': {
                    'strategy': 'RRS',
                    'rrs_score': float(best_signal['current_rrs']),
                    'rrs_rank': int(best_signal['rank']),
                    'rrs_momentum': float(best_signal['rrs_momentum']),
                    'rrs_trend': float(best_signal['rrs_trend']),
                    'risk_level': str(best_signal['risk_level']),
                    'timeframe': timeframe,
                    'benchmark': self.config.get('benchmark', 'BTCUSDT')
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error in RRS strategy execution: {e}")
            return None
    
    async def get_rankings(self) -> Optional[Dict]:
        """
        Get current RRS rankings without executing a trade.
        
        Returns:
            Dictionary with rankings data
        """
        if not self.rrs_manager:
            return None
        
        try:
            symbols = self.config.get('symbols', [])
            timeframe = self.config.get('timeframe', '1h')
            
            # Analyze symbols
            rrs_results = self.rrs_manager.analyze_multiple_symbols(
                symbols=symbols,
                timeframe=timeframe,
                lookback_days=self.config.get('lookback_days', 7),
                benchmark=self.config.get('benchmark', 'BTCUSDT')
            )
            
            if not rrs_results:
                return None
            
            # Generate rankings
            rankings_df = self.rrs_manager.calculator.rank_symbols_by_rrs(rrs_results)
            
            return {
                'rankings': rankings_df.to_dict('records'),
                'timestamp': datetime.now().isoformat(),
                'timeframe': timeframe,
                'benchmark': self.config.get('benchmark', 'BTCUSDT')
            }
            
        except Exception as e:
            self.logger.error(f"Error getting rankings: {e}")
            return None

