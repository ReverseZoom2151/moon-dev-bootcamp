"""
Backtesting Tools for Gordon
============================
Backtest and optimize trading strategies.
"""

from typing import Dict, Any, Optional, List
from langchain_core.tools import tool
import sys
from pathlib import Path
from datetime import datetime, timedelta

# Add parent directories to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
from gordon.backtesting.backtest_main import ComprehensiveBacktester


@tool
def backtest_strategy(
    strategy_name: str,
    symbol: str,
    start_date: str,
    end_date: str,
    initial_capital: float = 10000,
    position_size: float = 0.1,
    **strategy_params
) -> Dict[str, Any]:
    """Backtest a trading strategy on historical data.

    Args:
        strategy_name: Name of strategy to backtest (sma, rsi, vwap, etc.)
        symbol: Trading pair (e.g., "BTC/USDT")
        start_date: Start date for backtest (YYYY-MM-DD)
        end_date: End date for backtest (YYYY-MM-DD)
        initial_capital: Starting capital for backtest
        position_size: Position size as percentage of capital
        **strategy_params: Additional strategy-specific parameters

    Returns:
        Backtest results including performance metrics
    """
    try:
        backtester = ComprehensiveBacktester({
            'initial_capital': initial_capital,
            'position_size': position_size
        })

        # Run backtest
        results = backtester.run_backtest(
            strategy=strategy_name,
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
            params=strategy_params
        )

        # Calculate key metrics
        total_return = results.get('total_return', 0)
        sharpe_ratio = results.get('sharpe_ratio', 0)
        max_drawdown = results.get('max_drawdown', 0)
        win_rate = results.get('win_rate', 0)
        total_trades = results.get('total_trades', 0)

        return {
            'status': 'success',
            'strategy': strategy_name,
            'symbol': symbol,
            'period': f"{start_date} to {end_date}",
            'metrics': {
                'total_return': f"{total_return:.2%}",
                'sharpe_ratio': f"{sharpe_ratio:.2f}",
                'max_drawdown': f"{max_drawdown:.2%}",
                'win_rate': f"{win_rate:.2%}",
                'total_trades': total_trades,
                'final_capital': results.get('final_capital', initial_capital)
            },
            'trades': results.get('trades', [])[:10],  # First 10 trades
            'details': results
        }
    except Exception as e:
        return {
            'status': 'error',
            'message': str(e)
        }


@tool
def optimize_strategy_parameters(
    strategy_name: str,
    symbol: str,
    start_date: str,
    end_date: str,
    param_ranges: Dict[str, List],
    optimization_metric: str = "sharpe_ratio",
    initial_capital: float = 10000
) -> Dict[str, Any]:
    """Optimize strategy parameters using historical data.

    Args:
        strategy_name: Name of strategy to optimize
        symbol: Trading pair (e.g., "BTC/USDT")
        start_date: Start date for optimization
        end_date: End date for optimization
        param_ranges: Dictionary of parameter ranges to test
        optimization_metric: Metric to optimize ("sharpe_ratio", "return", "win_rate")
        initial_capital: Starting capital for backtest

    Returns:
        Optimal parameters and performance metrics
    """
    try:
        backtester = ComprehensiveBacktester({
            'initial_capital': initial_capital
        })

        # Run optimization
        results = backtester.optimize_strategy(
            strategy=strategy_name,
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
            param_ranges=param_ranges,
            metric=optimization_metric
        )

        best_params = results.get('best_params', {})
        best_performance = results.get('best_performance', {})

        return {
            'status': 'success',
            'strategy': strategy_name,
            'symbol': symbol,
            'optimization_metric': optimization_metric,
            'best_parameters': best_params,
            'performance': {
                'total_return': f"{best_performance.get('total_return', 0):.2%}",
                'sharpe_ratio': f"{best_performance.get('sharpe_ratio', 0):.2f}",
                'max_drawdown': f"{best_performance.get('max_drawdown', 0):.2%}",
                'win_rate': f"{best_performance.get('win_rate', 0):.2%}"
            },
            'iterations_tested': results.get('iterations', 0),
            'details': results
        }
    except Exception as e:
        return {
            'status': 'error',
            'message': str(e)
        }


@tool
def get_backtest_results(
    backtest_id: Optional[str] = None
) -> Dict[str, Any]:
    """Retrieve detailed results from a previous backtest.

    Args:
        backtest_id: ID of the backtest to retrieve (latest if not specified)

    Returns:
        Detailed backtest results and analysis
    """
    try:
        backtester = ComprehensiveBacktester()

        # Get results
        if backtest_id:
            results = backtester.get_results(backtest_id)
        else:
            results = backtester.get_latest_results()

        if not results:
            return {
                'status': 'error',
                'message': 'No backtest results found'
            }

        # Format detailed analysis
        return {
            'status': 'success',
            'backtest_id': results.get('id'),
            'strategy': results.get('strategy'),
            'symbol': results.get('symbol'),
            'period': results.get('period'),
            'performance_summary': {
                'total_return': f"{results.get('total_return', 0):.2%}",
                'annualized_return': f"{results.get('annualized_return', 0):.2%}",
                'sharpe_ratio': f"{results.get('sharpe_ratio', 0):.2f}",
                'sortino_ratio': f"{results.get('sortino_ratio', 0):.2f}",
                'max_drawdown': f"{results.get('max_drawdown', 0):.2%}",
                'calmar_ratio': f"{results.get('calmar_ratio', 0):.2f}"
            },
            'trade_statistics': {
                'total_trades': results.get('total_trades', 0),
                'winning_trades': results.get('winning_trades', 0),
                'losing_trades': results.get('losing_trades', 0),
                'win_rate': f"{results.get('win_rate', 0):.2%}",
                'avg_win': f"{results.get('avg_win', 0):.2%}",
                'avg_loss': f"{results.get('avg_loss', 0):.2%}",
                'profit_factor': f"{results.get('profit_factor', 0):.2f}"
            },
            'risk_metrics': {
                'value_at_risk': f"{results.get('var_95', 0):.2%}",
                'conditional_var': f"{results.get('cvar_95', 0):.2%}",
                'max_consecutive_losses': results.get('max_consecutive_losses', 0),
                'recovery_time': f"{results.get('recovery_time_days', 0)} days"
            },
            'equity_curve': results.get('equity_curve', [])[:20],  # First 20 points
            'monthly_returns': results.get('monthly_returns', {}),
            'details': results
        }
    except Exception as e:
        return {
            'status': 'error',
            'message': str(e)
        }