"""
RRS Manager
===========
Orchestrates RRS analysis across multiple exchanges and timeframes.
"""

import pandas as pd
import logging
from typing import Dict, List, Optional
from datetime import datetime, timedelta
from pathlib import Path

from .calculator import RRSCalculator
from .data_processor import RRSDataProcessor
from .signal_generator import RRSSignalGenerator

logger = logging.getLogger(__name__)


class RRSManager:
    """
    Manages RRS analysis across exchanges and timeframes.
    
    Coordinates data fetching, processing, RRS calculation, and signal generation.
    """
    
    def __init__(
        self,
        data_fetcher=None,
        config: Optional[Dict] = None
    ):
        """
        Initialize RRS manager.
        
        Args:
            data_fetcher: Data fetcher instance (from backtesting.data)
            config: Configuration dictionary
        """
        self.data_fetcher = data_fetcher
        self.config = config or {}
        
        # Initialize components
        self.calculator = RRSCalculator(
            smoothing_window=self.config.get('rrs_smoothing_window', 5),
            volatility_lookback=self.config.get('volatility_lookback', 20),
            volume_lookback=self.config.get('volume_lookback', 10),
            min_data_points=self.config.get('min_data_points', 50)
        )
        
        self.processor = RRSDataProcessor(
            volatility_lookback=self.config.get('volatility_lookback', 20),
            volume_lookback=self.config.get('volume_lookback', 10)
        )
        
        self.signal_generator = RRSSignalGenerator(
            strong_threshold=self.config.get('strong_threshold', 1.0),
            weak_threshold=self.config.get('weak_threshold', -1.0)
        )
        
        # Default benchmark
        self.default_benchmark = self.config.get('default_benchmark', 'BTCUSDT')
        
        # Timeframes
        self.timeframes = self.config.get('timeframes', {
            '1h': 7,   # 1 hour data, 7 days lookback
            '4h': 30,  # 4 hour data, 30 days lookback
            '1d': 90,  # 1 day data, 90 days lookback
        })
        
        # Results directory
        self.results_dir = Path(self.config.get('results_dir', './rrs_results'))
        self.results_dir.mkdir(parents=True, exist_ok=True)
    
    def analyze_symbol(
        self,
        symbol: str,
        timeframe: str = '1h',
        lookback_days: Optional[int] = None,
        benchmark: Optional[str] = None
    ) -> Optional[pd.DataFrame]:
        """
        Analyze a single symbol's RRS.
        
        Args:
            symbol: Trading symbol
            timeframe: Timeframe string (e.g., '1h', '4h', '1d')
            lookback_days: Number of days to look back
            benchmark: Benchmark symbol (defaults to config default)
            
        Returns:
            DataFrame with RRS metrics or None
        """
        if not self.data_fetcher:
            logger.error("Data fetcher not available")
            return None
        
        benchmark = benchmark or self.default_benchmark
        lookback_days = lookback_days or self.timeframes.get(timeframe, 7)
        
        logger.info(f"Analyzing {symbol} vs {benchmark} on {timeframe} timeframe")
        
        # Calculate time range
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(days=lookback_days)
        
        try:
            # Fetch symbol data
            symbol_data = self.data_fetcher.fetch(
                symbol=symbol,
                timeframe=timeframe,
                start=start_time,
                end=end_time
            )
            
            if symbol_data.empty:
                logger.warning(f"No data fetched for {symbol}")
                return None
            
            # Fetch benchmark data
            benchmark_data = self.data_fetcher.fetch(
                symbol=benchmark,
                timeframe=timeframe,
                start=start_time,
                end=end_time
            )
            
            if benchmark_data.empty:
                logger.warning(f"No data fetched for benchmark {benchmark}")
                return None
            
            # Process data
            symbol_processed = self.processor.process_data(symbol_data)
            benchmark_processed = self.processor.process_data(benchmark_data)
            
            # Calculate RRS
            rrs_result = self.calculator.calculate_rrs(
                symbol_processed,
                benchmark_processed,
                symbol_name=symbol,
                benchmark_name=benchmark
            )
            
            return rrs_result
            
        except Exception as e:
            logger.error(f"Error analyzing {symbol}: {e}")
            return None
    
    def analyze_multiple_symbols(
        self,
        symbols: List[str],
        timeframe: str = '1h',
        lookback_days: Optional[int] = None,
        benchmark: Optional[str] = None
    ) -> Dict[str, pd.DataFrame]:
        """
        Analyze multiple symbols' RRS.
        
        Args:
            symbols: List of trading symbols
            timeframe: Timeframe string
            lookback_days: Number of days to look back
            benchmark: Benchmark symbol
            
        Returns:
            Dictionary mapping symbols to RRS DataFrames
        """
        results = {}
        
        logger.info(f"Analyzing {len(symbols)} symbols on {timeframe} timeframe")
        
        for symbol in symbols:
            try:
                rrs_result = self.analyze_symbol(
                    symbol=symbol,
                    timeframe=timeframe,
                    lookback_days=lookback_days,
                    benchmark=benchmark
                )
                
                if rrs_result is not None and not rrs_result.empty:
                    results[symbol] = rrs_result
                    logger.info(f"✅ Completed RRS analysis for {symbol}")
                else:
                    logger.warning(f"⚠️ Failed to analyze {symbol}")
                    
            except Exception as e:
                logger.error(f"❌ Error analyzing {symbol}: {e}")
                continue
        
        logger.info(f"Successfully analyzed {len(results)} symbols")
        
        return results
    
    def generate_rankings_and_signals(
        self,
        rrs_results: Dict[str, pd.DataFrame],
        timeframe: str = '1h'
    ) -> pd.DataFrame:
        """
        Generate rankings and trading signals from RRS results.
        
        Args:
            rrs_results: Dictionary mapping symbols to RRS DataFrames
            timeframe: Timeframe string
            
        Returns:
            DataFrame with rankings and signals
        """
        if not rrs_results:
            logger.warning("No RRS results provided")
            return pd.DataFrame()
        
        # Generate rankings
        rankings_df = self.calculator.rank_symbols_by_rrs(rrs_results)
        
        if rankings_df.empty:
            logger.warning("No rankings generated")
            return pd.DataFrame()
        
        # Generate signals
        signals_df = self.signal_generator.generate_signals(rankings_df)
        
        # Save results
        self._save_results(signals_df, timeframe)
        
        # Log top performers
        logger.info(f"🚀 Top 5 Performers ({timeframe}):")
        for i, (_, row) in enumerate(signals_df.head(5).iterrows(), 1):
            logger.info(
                f"  {i}. {row['symbol']}: RRS={row['current_rrs']:.3f} | "
                f"Signal={row['primary_signal']} | Risk={row['risk_level']}"
            )
        
        return signals_df
    
    def run_full_analysis(
        self,
        symbols: List[str],
        timeframes: Optional[List[str]] = None,
        benchmark: Optional[str] = None
    ) -> Dict[str, pd.DataFrame]:
        """
        Run full RRS analysis across multiple timeframes.
        
        Args:
            symbols: List of symbols to analyze
            timeframes: List of timeframes (defaults to config)
            benchmark: Benchmark symbol
            
        Returns:
            Dictionary mapping timeframes to signal DataFrames
        """
        timeframes = timeframes or list(self.timeframes.keys())
        benchmark = benchmark or self.default_benchmark
        
        logger.info(f"Running full RRS analysis for {len(symbols)} symbols")
        logger.info(f"Timeframes: {timeframes}")
        logger.info(f"Benchmark: {benchmark}")
        
        all_results = {}
        
        for timeframe in timeframes:
            lookback_days = self.timeframes.get(timeframe, 7)
            
            logger.info(f"Processing {timeframe} timeframe ({lookback_days} days lookback)")
            
            # Analyze all symbols
            rrs_results = self.analyze_multiple_symbols(
                symbols=symbols,
                timeframe=timeframe,
                lookback_days=lookback_days,
                benchmark=benchmark
            )
            
            if rrs_results:
                # Generate rankings and signals
                signals_df = self.generate_rankings_and_signals(
                    rrs_results,
                    timeframe
                )
                
                if not signals_df.empty:
                    all_results[timeframe] = signals_df
        
        logger.info(f"✅ Completed full RRS analysis across {len(all_results)} timeframes")
        
        return all_results
    
    def _save_results(self, signals_df: pd.DataFrame, timeframe: str):
        """Save results to CSV."""
        try:
            filename = self.results_dir / f"rrs_signals_{timeframe}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            signals_df.to_csv(filename, index=False)
            logger.info(f"Saved results to {filename}")
        except Exception as e:
            logger.error(f"Failed to save results: {e}")

