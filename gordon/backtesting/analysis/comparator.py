"""Strategy comparator for multi-strategy analysis"""

import logging
from typing import List, Dict
import pandas as pd
from ..base import BacktestResult

logger = logging.getLogger(__name__)


class StrategyComparator:
    """Compare multiple backtest results"""
    
    @staticmethod
    def compare(results: List[BacktestResult]) -> pd.DataFrame:
        """
        Compare multiple backtest results
        
        Args:
            results: List of BacktestResult objects
            
        Returns:
            DataFrame with comparison data
        """
        if not results:
            return pd.DataFrame()
        
        comparison_data = []
        for result in results:
            metrics = result.metrics
            comparison_data.append({
                'Strategy': result.strategy_name,
                'Framework': result.framework,
                'Return %': metrics.total_return,
                'Sharpe Ratio': metrics.sharpe_ratio,
                'Max DD %': metrics.max_drawdown,
                'Win Rate %': metrics.win_rate,
                'Trades': metrics.num_trades,
                'Final Value': metrics.final_value,
                'Risk/Reward': metrics.risk_reward_ratio
            })
        
        df = pd.DataFrame(comparison_data)
        return df.sort_values('Return %', ascending=False)
    
    @staticmethod
    def rank_by_metric(results: List[BacktestResult], 
                      metric: str = 'Return %') -> List[tuple]:
        """
        Rank strategies by specific metric
        
        Args:
            results: List of BacktestResult objects
            metric: Metric name to rank by
            
        Returns:
            List of (strategy_name, metric_value) tuples
        """
        df = StrategyComparator.compare(results)
        if metric not in df.columns:
            logger.error(f"Metric '{metric}' not found in comparison data")
            return []
        
        ranked = df[['Strategy', metric]].sort_values(metric, ascending=False)
        return list(zip(ranked['Strategy'], ranked[metric]))
    
    @staticmethod
    def find_best_strategy(results: List[BacktestResult]) -> BacktestResult:
        """
        Find best performing strategy by return
        
        Args:
            results: List of BacktestResult objects
            
        Returns:
            Best BacktestResult
        """
        if not results:
            return None
        
        return max(results, key=lambda r: r.metrics.total_return)
    
    @staticmethod
    def print_comparison(results: List[BacktestResult]):
        """Print formatted comparison table"""
        df = StrategyComparator.compare(results)
        
        print(f"\n{'='*80}")
        print("STRATEGY COMPARISON")
        print(f"{'='*80}\n")
        
        print(df.to_string(index=False))
        
        if len(results) > 0:
            best = StrategyComparator.find_best_strategy(results)
            best_return = best.metrics.total_return
            print(f"\n{'-'*80}")
            print(f"Best Strategy: {best.strategy_name} with {best_return:.2f}% return")
            print(f"{'-'*80}\n")
