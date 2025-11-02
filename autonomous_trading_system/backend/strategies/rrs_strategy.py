"""
Real-time Recommendation System (RRS) Strategy
Dynamic asset selection and ranking based on relative performance
Based on Day 37 RRS implementation
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
from dataclasses import dataclass
from strategies.base_strategy import BaseStrategy, StrategySignal, SignalAction, TechnicalIndicatorMixin

logger = logging.getLogger(__name__)

@dataclass
class RRSResult:
    """RRS calculation result for a symbol"""
    symbol: str
    rrs_score: float
    smoothed_rrs: float
    rank: int
    volatility: float
    volume_ratio: float
    timestamp: datetime


class RRSCalculator:
    """Calculates Relative Ranking System scores"""
    
    def __init__(self, smoothing_factor: float = 0.1):
        self.smoothing_factor = smoothing_factor
        self.previous_rrs = {}
    
    def calculate_returns_and_volatility(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate log returns and volatility"""
        df = df.copy()
        
        # Calculate log returns
        df['log_return'] = np.log(df['close'] / df['close'].shift(1))
        
        # Calculate rolling volatility (20-period)
        df['volatility'] = df['log_return'].rolling(window=20).std()
        
        return df
    
    def calculate_volume_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate volume-based metrics"""
        df = df.copy()
        
        # Volume moving average
        df['volume_ma'] = df['volume'].rolling(window=20).mean()
        
        # Volume ratio (current vs average)
        df['volume_ratio'] = df['volume'] / df['volume_ma']
        
        return df
    
    def calculate_rrs(self, symbol_df: pd.DataFrame, benchmark_df: pd.DataFrame) -> pd.DataFrame:
        """Calculate RRS score relative to benchmark"""
        try:
            # Ensure both dataframes have the same time index
            merged_df = pd.merge(
                symbol_df[['timestamp', 'log_return', 'volatility', 'volume_ratio']],
                benchmark_df[['timestamp', 'log_return']],
                on='timestamp',
                suffixes=('_symbol', '_benchmark'),
                how='inner'
            )
            
            if merged_df.empty:
                logger.warning(f"No overlapping data for RRS calculation")
                return pd.DataFrame()
            
            # Calculate relative performance
            merged_df['relative_return'] = merged_df['log_return_symbol'] - merged_df['log_return_benchmark']
            
            # Calculate cumulative relative performance
            merged_df['cumulative_relative'] = merged_df['relative_return'].cumsum()
            
            # Calculate RRS score (normalized cumulative relative performance)
            lookback_window = min(50, len(merged_df))
            merged_df['rrs_raw'] = merged_df['cumulative_relative'].rolling(window=lookback_window).mean()
            
            # Normalize RRS to [-1, 1] range
            rrs_std = merged_df['rrs_raw'].rolling(window=lookback_window).std()
            merged_df['rrs_normalized'] = merged_df['rrs_raw'] / (rrs_std + 1e-8)
            merged_df['rrs_normalized'] = np.tanh(merged_df['rrs_normalized'])  # Bound to [-1, 1]
            
            # Apply exponential smoothing
            merged_df['smoothed_rrs'] = merged_df['rrs_normalized'].ewm(alpha=self.smoothing_factor).mean()
            
            # Add volume and volatility adjustments
            merged_df['volume_adjusted_rrs'] = merged_df['smoothed_rrs'] * np.log1p(merged_df['volume_ratio'])
            merged_df['final_rrs'] = merged_df['volume_adjusted_rrs'] / (1 + merged_df['volatility'])
            
            return merged_df[['timestamp', 'final_rrs', 'smoothed_rrs', 'volatility', 'volume_ratio']].copy()
            
        except Exception as e:
            logger.error(f"Error calculating RRS: {e}")
            return pd.DataFrame()


class RRSStrategy(BaseStrategy, TechnicalIndicatorMixin):
    """
    Real-time Recommendation System Strategy
    
    Features:
    - Dynamic asset ranking based on relative performance
    - Benchmark-relative scoring
    - Volume and volatility adjustments
    - Automatic symbol rotation
    - Cross-asset correlation analysis
    """
    
    def __init__(self, config: Dict[str, Any], market_data_manager, name: str = "RRS"):
        super().__init__(config, market_data_manager, name)
        
        # RRS Configuration
        self.benchmark_symbol = config.get("benchmark_symbol", "BTC")
        self.lookback_days = config.get("lookback_days", 30)
        self.min_rrs_score = config.get("min_rrs_score", 0.3)
        self.max_positions = config.get("max_positions", 3)
        self.rebalance_interval = config.get("rebalance_interval", 6)  # hours
        self.smoothing_factor = config.get("smoothing_factor", 0.1)
        
        # RRS Calculator
        self.rrs_calculator = RRSCalculator(self.smoothing_factor)
        
        # Tracking
        self.current_rankings = []
        self.last_rebalance = None
        self.rrs_history = {}
        self.correlation_matrix = {}
        
        # Remove benchmark from trading symbols
        self.trading_symbols = [s for s in self.symbols if s != self.benchmark_symbol]
        
        logger.info(f"📊 RRS Strategy initialized:")
        logger.info(f"   Benchmark: {self.benchmark_symbol}")
        logger.info(f"   Trading Symbols: {self.trading_symbols}")
        logger.info(f"   Min RRS Score: {self.min_rrs_score}")
        logger.info(f"   Max Positions: {self.max_positions}")
    
    async def _initialize_strategy(self):
        """Initialize RRS strategy"""
        try:
            # Initialize RRS history for each symbol
            for symbol in self.trading_symbols:
                self.rrs_history[symbol] = []
            
            logger.info("✅ RRS strategy validation complete")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize RRS strategy: {e}")
            raise
    
    async def generate_signal(self) -> Optional[StrategySignal]:
        """Generate RRS-based trading signal"""
        try:
            # Check if rebalancing is needed
            if await self._should_rebalance():
                await self._calculate_rrs_rankings()
                signal = await self._generate_rebalance_signal()
                return signal
            
            return None
            
        except Exception as e:
            logger.error(f"❌ Error generating RRS signal: {e}", exc_info=True)
            return None
    
    async def _should_rebalance(self) -> bool:
        """Check if portfolio rebalancing is needed"""
        if self.last_rebalance is None:
            return True
        
        hours_since_rebalance = (datetime.utcnow() - self.last_rebalance).total_seconds() / 3600
        return hours_since_rebalance >= self.rebalance_interval
    
    async def _calculate_rrs_rankings(self):
        """Calculate RRS scores and rank all symbols"""
        try:
            logger.info("📊 Calculating RRS rankings...")
            
            # Get benchmark data
            benchmark_data = await self._get_market_data(
                self.benchmark_symbol, 
                limit=self.lookback_days * 24
            )
            
            if benchmark_data is None:
                logger.warning(f"No benchmark data available for {self.benchmark_symbol}")
                return
            
            # Process benchmark data
            benchmark_df = pd.DataFrame(benchmark_data)
            benchmark_df = self.rrs_calculator.calculate_returns_and_volatility(benchmark_df)
            
            # Calculate RRS for each trading symbol
            rrs_results = []
            
            for symbol in self.trading_symbols:
                symbol_data = await self._get_market_data(symbol, limit=self.lookback_days * 24)
                
                if symbol_data is None:
                    continue
                
                # Process symbol data
                symbol_df = pd.DataFrame(symbol_data)
                symbol_df = self.rrs_calculator.calculate_returns_and_volatility(symbol_df)
                symbol_df = self.rrs_calculator.calculate_volume_metrics(symbol_df)
                
                # Calculate RRS
                rrs_df = self.rrs_calculator.calculate_rrs(symbol_df, benchmark_df)
                
                if rrs_df.empty:
                    continue
                
                # Get latest RRS score
                latest_rrs = rrs_df.iloc[-1]
                
                rrs_result = RRSResult(
                    symbol=symbol,
                    rrs_score=latest_rrs['final_rrs'],
                    smoothed_rrs=latest_rrs['smoothed_rrs'],
                    rank=0,  # Will be set after sorting
                    volatility=latest_rrs['volatility'],
                    volume_ratio=latest_rrs['volume_ratio'],
                    timestamp=datetime.utcnow()
                )
                
                rrs_results.append(rrs_result)
                
                # Store in history
                self.rrs_history[symbol].append(rrs_result)
                
                # Keep only recent history
                cutoff_time = datetime.utcnow() - timedelta(days=7)
                self.rrs_history[symbol] = [
                    r for r in self.rrs_history[symbol] 
                    if r.timestamp > cutoff_time
                ]
            
            # Sort by RRS score and assign ranks
            rrs_results.sort(key=lambda x: x.rrs_score, reverse=True)
            for i, result in enumerate(rrs_results):
                result.rank = i + 1
            
            self.current_rankings = rrs_results
            
            # Calculate correlation matrix
            await self._calculate_correlation_matrix()
            
            # Log rankings
            logger.info("📊 RRS Rankings:")
            for result in self.current_rankings[:5]:  # Top 5
                logger.info(f"   {result.rank}. {result.symbol}: {result.rrs_score:.4f}")
            
        except Exception as e:
            logger.error(f"❌ Error calculating RRS rankings: {e}")
    
    async def _calculate_correlation_matrix(self):
        """Calculate correlation matrix between symbols"""
        try:
            if len(self.current_rankings) < 2:
                return
            
            # Get recent price data for correlation calculation
            price_data = {}
            
            for result in self.current_rankings:
                data = await self._get_market_data(result.symbol, limit=100)
                if data:
                    df = pd.DataFrame(data)
                    price_data[result.symbol] = df['close'].pct_change().dropna()
            
            if len(price_data) < 2:
                return
            
            # Create correlation matrix
            combined_df = pd.DataFrame(price_data)
            self.correlation_matrix = combined_df.corr().to_dict()
            
        except Exception as e:
            logger.error(f"❌ Error calculating correlation matrix: {e}")
    
    async def _generate_rebalance_signal(self) -> Optional[StrategySignal]:
        """Generate rebalancing signal based on RRS rankings"""
        try:
            if not self.current_rankings:
                return None
            
            # Select top performers with low correlation
            selected_symbols = self._select_diversified_symbols()
            
            if not selected_symbols:
                return None
            
            # Create signal for the top-ranked symbol
            top_symbol = selected_symbols[0]
            
            # Get current price
            market_data = await self._get_market_data(top_symbol.symbol, limit=1)
            if not market_data:
                return None
            
            current_price = float(market_data[0]['close'])
            
            # Calculate confidence based on RRS score and rank
            confidence = min(
                abs(top_symbol.rrs_score),
                1.0 / top_symbol.rank,  # Higher rank = lower confidence
                1.0
            )
            
            # Only generate signal if RRS score is above threshold
            if top_symbol.rrs_score < self.min_rrs_score:
                return None
            
            # Determine action based on RRS score
            action = SignalAction.BUY if top_symbol.rrs_score > 0 else SignalAction.SELL
            
            # Create metadata
            metadata = {
                'rrs_score': top_symbol.rrs_score,
                'smoothed_rrs': top_symbol.smoothed_rrs,
                'rank': top_symbol.rank,
                'volatility': top_symbol.volatility,
                'volume_ratio': top_symbol.volume_ratio,
                'selected_symbols': [s.symbol for s in selected_symbols],
                'benchmark': self.benchmark_symbol,
                'strategy_type': 'rrs'
            }
            
            # Create signal
            signal = self._create_signal(
                symbol=top_symbol.symbol,
                action=action,
                price=current_price,
                confidence=confidence,
                metadata=metadata
            )
            
            # Update rebalance time
            self.last_rebalance = datetime.utcnow()
            
            logger.info(f"📊 RRS Signal: {action.value} {top_symbol.symbol} @ {current_price:.4f}")
            logger.info(f"   RRS Score: {top_symbol.rrs_score:.4f}, Rank: {top_symbol.rank}")
            logger.info(f"   Selected symbols: {[s.symbol for s in selected_symbols]}")
            
            return signal
            
        except Exception as e:
            logger.error(f"❌ Error generating rebalance signal: {e}")
            return None
    
    def _select_diversified_symbols(self) -> List[RRSResult]:
        """Select symbols with good RRS scores and low correlation"""
        if not self.current_rankings:
            return []
        
        selected = []
        
        for result in self.current_rankings:
            if len(selected) >= self.max_positions:
                break
            
            # Check RRS score threshold
            if result.rrs_score < self.min_rrs_score:
                continue
            
            # Check correlation with already selected symbols
            if self._is_diversified(result.symbol, [s.symbol for s in selected]):
                selected.append(result)
        
        return selected
    
    def _is_diversified(self, symbol: str, selected_symbols: List[str], max_correlation: float = 0.7) -> bool:
        """Check if symbol is diversified relative to selected symbols"""
        if not selected_symbols or not self.correlation_matrix:
            return True
        
        symbol_correlations = self.correlation_matrix.get(symbol, {})
        
        for selected_symbol in selected_symbols:
            correlation = abs(symbol_correlations.get(selected_symbol, 0))
            if correlation > max_correlation:
                return False
        
        return True
    
    def get_rrs_metrics(self) -> Dict[str, Any]:
        """Get RRS performance metrics"""
        if not self.current_rankings:
            return {}
        
        return {
            'total_symbols_ranked': len(self.current_rankings),
            'top_performer': {
                'symbol': self.current_rankings[0].symbol,
                'rrs_score': self.current_rankings[0].rrs_score,
                'rank': self.current_rankings[0].rank
            },
            'bottom_performer': {
                'symbol': self.current_rankings[-1].symbol,
                'rrs_score': self.current_rankings[-1].rrs_score,
                'rank': self.current_rankings[-1].rank
            },
            'avg_rrs_score': np.mean([r.rrs_score for r in self.current_rankings]),
            'rrs_score_std': np.std([r.rrs_score for r in self.current_rankings]),
            'last_rebalance': self.last_rebalance.isoformat() if self.last_rebalance else None,
            'correlation_matrix_size': len(self.correlation_matrix)
        }
    
    def get_strategy_info(self) -> Dict[str, Any]:
        """Get strategy information and performance metrics"""
        return {
            "name": self.name,
            "type": "rrs_ranking",
            "benchmark_symbol": self.benchmark_symbol,
            "trading_symbols": self.trading_symbols,
            "min_rrs_score": self.min_rrs_score,
            "max_positions": self.max_positions,
            "current_rankings": [
                {
                    'symbol': r.symbol,
                    'rrs_score': r.rrs_score,
                    'rank': r.rank,
                    'volatility': r.volatility
                }
                for r in self.current_rankings[:10]  # Top 10
            ],
            "rrs_metrics": self.get_rrs_metrics(),
            "status": self.status.value,
            "enabled": self.enabled
        } 