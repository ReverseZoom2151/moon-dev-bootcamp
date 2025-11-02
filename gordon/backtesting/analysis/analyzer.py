"""Results analyzer for individual backtest results"""

import logging
from typing import Dict, List
import pandas as pd
from ..base import BacktestResult, BacktestMetrics

logger = logging.getLogger(__name__)


class ResultsAnalyzer:
    """Analyze individual backtest results"""
    
    @staticmethod
    def analyze(result: BacktestResult) -> Dict:
        """
        Analyze backtest results comprehensively
        
        Args:
            result: BacktestResult object
            
        Returns:
            Dictionary with detailed analysis
        """
        metrics = result.metrics
        trades = result.trades
        
        analysis = {
            'strategy': result.strategy_name,
            'framework': result.framework,
            'performance': {
                'total_return': metrics.total_return,
                'final_value': metrics.final_value,
                'initial_value': metrics.initial_value,
                'sharpe_ratio': metrics.sharpe_ratio,
                'max_drawdown': metrics.max_drawdown,
                'risk_reward_ratio': metrics.risk_reward_ratio,
            },
            'trading_activity': {
                'total_trades': metrics.num_trades,
                'win_rate': metrics.win_rate,
                'avg_trade_return': metrics.avg_trade,
                'profit_factor': metrics.profit_factor,
                'max_consecutive_losses': metrics.max_consecutive_losses,
            },
            'trade_analysis': ResultsAnalyzer._analyze_trades(trades),
            'execution': {
                'execution_time': result.execution_time,
                'timestamp': result.timestamp.isoformat()
            }
        }
        
        return analysis
    
    @staticmethod
    def _analyze_trades(trades: List) -> Dict:
        """Analyze trade statistics"""
        if not trades:
            return {'total': 0, 'wins': 0, 'losses': 0, 'data': []}
        
        df_trades = pd.DataFrame([t.to_dict() for t in trades])
        
        winners = df_trades[df_trades['pnl'] > 0]
        losers = df_trades[df_trades['pnl'] <= 0]
        
        return {
            'total': len(df_trades),
            'wins': len(winners),
            'losses': len(losers),
            'avg_win': winners['pnl'].mean() if len(winners) > 0 else 0,
            'avg_loss': losers['pnl'].mean() if len(losers) > 0 else 0,
            'largest_win': winners['pnl'].max() if len(winners) > 0 else 0,
            'largest_loss': losers['pnl'].min() if len(losers) > 0 else 0,
        }
    
    @staticmethod
    def print_summary(result: BacktestResult):
        """Print formatted results summary"""
        metrics = result.metrics
        
        print(f"\n{'='*60}")
        print(f"Backtest Results: {result.strategy_name}")
        print(f"{'='*60}")
        
        print(f"\nPerformance:")
        print(f"  Initial Capital:    ${metrics.initial_value:,.2f}")
        print(f"  Final Value:        ${metrics.final_value:,.2f}")
        print(f"  Total Return:       {metrics.total_return:.2f}%")
        print(f"  Sharpe Ratio:       {metrics.sharpe_ratio:.3f}")
        print(f"  Max Drawdown:       {metrics.max_drawdown:.2f}%")
        print(f"  Risk/Reward Ratio:  {metrics.risk_reward_ratio:.3f}")
        
        print(f"\nTrading Activity:")
        print(f"  Total Trades:       {metrics.num_trades}")
        print(f"  Win Rate:           {metrics.win_rate:.2f}%")
        if metrics.avg_trade is not None:
            print(f"  Avg Trade Return:   {metrics.avg_trade:.2f}%")
        
        print(f"\nExecution:")
        print(f"  Execution Time:     {result.execution_time:.2f}s")
        print(f"  Framework:          {result.framework}")
