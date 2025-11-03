"""
Genetic Programming Evolution Runner
=====================================
Day 29: Runner for evolving trading strategies using GP.

Provides high-level interface for strategy evolution with integration
into Gordon's backtesting framework.
"""

import logging
import pandas as pd
from typing import Dict, List, Optional, Any
from pathlib import Path

from .gp_engine import GPEvolutionEngine, GPConfig
from .evolved_strategy import EvolvedStrategy
from ..data.fetcher import DataFetcher

logger = logging.getLogger(__name__)


class GPEvolutionRunner:
    """
    Runner for Genetic Programming strategy evolution.
    
    Provides high-level interface for evolving trading strategies.
    """

    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize GP evolution runner.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.data_fetcher = DataFetcher()
        self.engine = GPEvolutionEngine()
        
    def evolve_strategy(
        self,
        symbol: str,
        timeframe: str = '1h',
        data_path: Optional[str] = None,
        gp_config: Optional[GPConfig] = None,
        use_multiprocessing: bool = True,
        save_results: bool = True
    ) -> Dict[str, Any]:
        """
        Evolve a trading strategy for a symbol using GP.
        
        Args:
            symbol: Trading symbol (e.g., 'BTCUSDT')
            timeframe: Data timeframe (e.g., '1h', '4h', '1d')
            data_path: Optional path to CSV file (will fetch if not provided)
            gp_config: GP configuration (uses defaults if None)
            use_multiprocessing: Whether to use multiprocessing
            save_results: Whether to save results to CSV
            
        Returns:
            Dictionary with evolution results
        """
        logger.info(f"Starting GP evolution for {symbol} ({timeframe})")
        
        try:
            # Load or fetch data
            if data_path and Path(data_path).exists():
                logger.info(f"Loading data from {data_path}")
                data = self._load_data(data_path)
            else:
                logger.info(f"Fetching data for {symbol} ({timeframe})")
                data = self.data_fetcher.fetch_for_backtesting_lib(symbol, timeframe, limit=2000)
            
            if data.empty:
                logger.error("No data available for evolution")
                return {'error': 'No data available'}
            
            logger.info(f"Data loaded: {data.shape[0]} rows")
            
            # Setup GP configuration
            if gp_config is None:
                gp_config = GPConfig(
                    initial_cash=self.config.get('initial_cash', 1_000_000),
                    commission_pct=self.config.get('commission', 0.002),
                    population_size=self.config.get('population_size', 200),
                    generations=self.config.get('generations', 30)
                )
            
            # Create primitive set
            pset = self.engine.create_primitive_set()
            
            # Setup toolbox
            toolbox = self.engine.setup_toolbox(
                pset=pset,
                data=data,
                strategy_class=EvolvedStrategy,
                config=gp_config,
                use_multiprocessing=use_multiprocessing
            )
            
            # Run evolution
            pop, logbook, hof = self.engine.run_evolution(toolbox, gp_config)
            
            # Extract results
            results = {
                'symbol': symbol,
                'timeframe': timeframe,
                'generations': gp_config.generations,
                'population_size': gp_config.population_size,
                'hall_of_fame_size': len(hof),
                'logbook': logbook,
                'best_strategies': []
            }
            
            # Extract best strategies
            for i, individual in enumerate(hof):
                try:
                    fitness = individual.fitness.values[0] if individual.fitness.valid else None
                    compiled_func = self.engine.compile_individual(individual, pset)
                    
                    results['best_strategies'].append({
                        'rank': i + 1,
                        'fitness': fitness,
                        'size': len(individual),
                        'height': individual.height if hasattr(individual, 'height') else None,
                        'strategy_tree': str(individual),
                        'compiled_function': compiled_func
                    })
                except Exception as e:
                    logger.error(f"Error processing Hall of Fame individual {i}: {e}")
            
            # Save results if requested
            if save_results:
                output_path = self.config.get(
                    'results_path',
                    f'./evolved_strategies_{symbol}_{timeframe}.csv'
                )
                self.engine.save_results(hof, output_path, symbol)
                results['results_file'] = output_path
            
            logger.info(f"Evolution completed. Found {len(hof)} best strategies")
            return results
            
        except Exception as e:
            logger.error(f"Error during GP evolution: {e}", exc_info=True)
            return {'error': str(e)}

    def _load_data(self, filepath: str) -> pd.DataFrame:
        """
        Load and prepare data from CSV file.
        
        Args:
            filepath: Path to CSV file
            
        Returns:
            Prepared DataFrame
        """
        try:
            data = pd.read_csv(filepath)
            
            # Find date column
            date_cols = ['datetime', 'date', 'timestamp', 'time']
            date_col = None
            for col in date_cols:
                if col in data.columns:
                    date_col = col
                    break
            
            if date_col:
                data[date_col] = pd.to_datetime(data[date_col], errors='coerce')
                data.dropna(subset=[date_col], inplace=True)
                data.set_index(date_col, inplace=True)
            
            # Clean column names
            data.columns = data.columns.str.strip().str.capitalize()
            
            # Ensure required columns
            required = {'Open', 'High', 'Low', 'Close', 'Volume'}
            missing = required - set(col.lower() for col in data.columns)
            if missing:
                raise ValueError(f"Missing required columns: {missing}")
            
            # Convert to numeric
            for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
                col_actual = [c for c in data.columns if c.lower() == col.lower()][0]
                data[col_actual] = pd.to_numeric(data[col_actual], errors='coerce')
            
            data.dropna(inplace=True)
            
            return data
            
        except Exception as e:
            logger.error(f"Error loading data from {filepath}: {e}")
            return pd.DataFrame()

    def test_evolved_strategy(
        self,
        strategy_tree: str,
        symbol: str,
        timeframe: str = '1h',
        data_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Test a specific evolved strategy on data.
        
        Args:
            strategy_tree: String representation of GP tree
            symbol: Trading symbol
            timeframe: Data timeframe
            data_path: Optional path to CSV file
            
        Returns:
            Dictionary with backtest results
        """
        from backtesting import Backtest
        
        try:
            # Load data
            if data_path and Path(data_path).exists():
                data = self._load_data(data_path)
            else:
                data = self.data_fetcher.fetch_for_backtesting_lib(symbol, timeframe, limit=2000)
            
            if data.empty:
                return {'error': 'No data available'}
            
            # Parse and compile strategy tree
            # This is simplified - in production, you'd need proper tree parsing
            pset = self.engine.create_primitive_set()
            
            # For now, this is a placeholder
            # In production, you'd reconstruct the tree from string
            logger.warning("Strategy tree parsing not fully implemented")
            
            return {'error': 'Strategy tree parsing not implemented'}
            
        except Exception as e:
            logger.error(f"Error testing evolved strategy: {e}")
            return {'error': str(e)}

    def get_evolution_summary(self, logbook: Any) -> Dict[str, Any]:
        """
        Get summary statistics from evolution logbook.
        
        Args:
            logbook: DEAP logbook
            
        Returns:
            Dictionary with summary statistics
        """
        if not logbook:
            return {}
        
        try:
            # Extract statistics
            gen_data = logbook.select("gen", "fitness")
            
            if not gen_data:
                return {}
            
            # Get final generation stats
            final_gen = gen_data[-1]
            
            return {
                'final_generation': final_gen.get('gen', 0),
                'final_avg_fitness': final_gen.get('fitness', {}).get('avg', 0),
                'final_max_fitness': final_gen.get('fitness', {}).get('max', 0),
                'final_min_fitness': final_gen.get('fitness', {}).get('min', 0),
                'final_std_fitness': final_gen.get('fitness', {}).get('std', 0),
                'evolution_progress': gen_data
            }
            
        except Exception as e:
            logger.error(f"Error extracting evolution summary: {e}")
            return {}

