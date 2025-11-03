"""
Alpha Decay Testing Utilities
==============================
Day 22: Utilities for testing how strategy performance degrades with entry delays.

Alpha Decay: The degradation of strategy returns as execution delays increase.
This helps determine optimal execution speed and slippage tolerance.
"""

import logging
from typing import List, Dict, Any, Type
import pandas as pd
# Import external backtesting package, avoiding conflict with gordon.backtesting
from .backtesting_import import get_backtesting_strategy

# Import Backtest from backtest_runner which handles the import correctly
import sys
from pathlib import Path
parent_dir = str(Path(__file__).parent.parent.parent)
if parent_dir in sys.path:
    sys.path.remove(parent_dir)
try:
    from backtesting import Backtest
except ImportError:
    Backtest = None
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

Strategy = get_backtesting_strategy()

if Strategy is None or Backtest is None:
    raise ImportError("External 'backtesting' package not installed. Install with: pip install backtesting")

logger = logging.getLogger(__name__)


def run_alpha_decay_test(
    data: pd.DataFrame,
    strategy_class: Type[Strategy],
    delays: List[int],
    initial_cash: float = 100000,
    commission: float = 0.002,
    maximize: str = 'Equity Final [$]'
) -> Dict[int, Dict[str, Any]]:
    """
    Run backtests for a given strategy class with varying entry delays.
    
    Measures alpha decay - how strategy performance degrades with execution delays.
    
    Args:
        data: Prepared OHLCV data with liquidation features
        strategy_class: Strategy class (must have a `delay_minutes` attribute)
        delays: List of delay values (in minutes) to test
        initial_cash: Starting capital
        commission: Trading commission
        maximize: Metric to maximize for optimization
        
    Returns:
        Dictionary mapping delay to backtest results
    """
    results = {}
    
    for delay in delays:
        logger.info(f"Running backtest with {delay}-minute delay")
        
        # Modify the class variable directly
        # Note: This is simple for sequential tests but not suitable for parallel execution
        # For parallel execution, consider creating strategy instances with parameters
        original_delay = getattr(strategy_class, 'delay_minutes', 0)
        strategy_class.delay_minutes = delay
        
        try:
            # Create and configure the backtest
            bt = Backtest(
                data,
                strategy_class,
                cash=initial_cash,
                commission=commission
            )
            
            # Run the backtest
            stats = bt.run()
            
            # Extract key metrics
            results[delay] = {
                'equity_final': stats.get('Equity Final [$]', 0),
                'return_pct': stats.get('Return [%]', 0),
                'sharpe_ratio': stats.get('Sharpe Ratio', 0),
                'max_drawdown': stats.get('Max. Drawdown [%]', 0),
                'num_trades': stats.get('# Trades', 0),
                'win_rate': stats.get('Win Rate [%]', 0),
                'stats': stats
            }
            
            logger.info(f"Delay {delay}min: Return={results[delay]['return_pct']:.2f}%, "
                       f"Sharpe={results[delay]['sharpe_ratio']:.2f}")
            
        except Exception as e:
            logger.error(f"Error during backtest run with delay {delay}: {e}")
            results[delay] = {'error': str(e)}
        
        finally:
            # Restore original delay value
            strategy_class.delay_minutes = original_delay
    
    return results


def analyze_alpha_decay(results: Dict[int, Dict[str, Any]], metric: str = 'return_pct') -> Dict[str, Any]:
    """
    Analyze alpha decay results and calculate decay metrics.
    
    Args:
        results: Results dictionary from run_alpha_decay_test
        metric: Metric to analyze ('return_pct', 'sharpe_ratio', 'equity_final')
        
    Returns:
        Dictionary with decay analysis
    """
    if not results:
        return {}
    
    # Extract metric values
    delays = sorted([d for d in results.keys() if 'error' not in results[d]])
    metric_values = [results[d].get(metric, 0) for d in delays]
    
    if not delays or not metric_values:
        return {}
    
    baseline_value = metric_values[0]  # Value at delay=0
    
    # Calculate decay rates
    decay_rates = []
    for i, delay in enumerate(delays):
        if i == 0:
            decay_rates.append(0.0)  # No decay at baseline
        else:
            decay = ((baseline_value - metric_values[i]) / abs(baseline_value)) * 100 if baseline_value != 0 else 0
            decay_rates.append(decay)
    
    # Find maximum tolerable delay (where decay < 10%)
    max_tolerable_delay = None
    for i, decay in enumerate(decay_rates):
        if decay < 10.0:
            max_tolerable_delay = delays[i]
        else:
            break
    
    analysis = {
        'baseline_value': baseline_value,
        'delays': delays,
        'metric_values': metric_values,
        'decay_rates': decay_rates,
        'max_tolerable_delay': max_tolerable_delay,
        'decay_at_1min': decay_rates[1] if len(decay_rates) > 1 else None,
        'decay_at_5min': decay_rates[delays.index(5)] if 5 in delays else None,
        'decay_at_15min': decay_rates[delays.index(15)] if 15 in delays else None,
        'decay_at_60min': decay_rates[delays.index(60)] if 60 in delays else None
    }
    
    return analysis


def print_alpha_decay_report(results: Dict[int, Dict[str, Any]], metric: str = 'return_pct'):
    """
    Print formatted alpha decay analysis report.
    
    Args:
        results: Results dictionary from run_alpha_decay_test
        metric: Metric to analyze
    """
    analysis = analyze_alpha_decay(results, metric)
    
    if not analysis:
        logger.warning("No analysis data available")
        return
    
    print("\n" + "=" * 60)
    print("ALPHA DECAY ANALYSIS REPORT")
    print("=" * 60)
    print(f"\nBaseline Performance (0 min delay): {analysis['baseline_value']:.2f}")
    print(f"\n{'Delay (min)':<15} {'Value':<15} {'Decay %':<15}")
    print("-" * 45)
    
    for delay, value, decay in zip(
        analysis['delays'],
        analysis['metric_values'],
        analysis['decay_rates']
    ):
        print(f"{delay:<15} {value:<15.2f} {decay:<15.2f}")
    
    if analysis['max_tolerable_delay'] is not None:
        print(f"\nMaximum Tolerable Delay: {analysis['max_tolerable_delay']} minutes")
    
    print("\nDecay Milestones:")
    if analysis['decay_at_1min'] is not None:
        print(f"  1 min delay: {analysis['decay_at_1min']:.2f}% decay")
    if analysis['decay_at_5min'] is not None:
        print(f"  5 min delay: {analysis['decay_at_5min']:.2f}% decay")
    if analysis['decay_at_15min'] is not None:
        print(f"  15 min delay: {analysis['decay_at_15min']:.2f}% decay")
    if analysis['decay_at_60min'] is not None:
        print(f"  60 min delay: {analysis['decay_at_60min']:.2f}% decay")
    
    print("=" * 60 + "\n")

