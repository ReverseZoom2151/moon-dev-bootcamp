# Trading tools for Gordon
from .strategies import (
    run_sma_strategy,
    run_rsi_strategy,
    run_vwap_strategy,
    run_mean_reversion_strategy,
    run_bollinger_strategy
)
from .orders import (
    place_market_order,
    place_limit_order,
    execute_algo_order,
    execute_twap,
    execute_vwap_order
)

# Backtesting tools - made optional due to import conflicts
try:
    from .backtest import (
        backtest_strategy,
        optimize_strategy_parameters,
        get_backtest_results
    )
    BACKTEST_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Backtesting tools not available: {e}")
    # Create dummy functions
    def backtest_strategy(*args, **kwargs):
        return {'status': 'error', 'message': 'Backtesting not available - install required packages'}
    def optimize_strategy_parameters(*args, **kwargs):
        return {'status': 'error', 'message': 'Backtesting not available - install required packages'}
    def get_backtest_results(*args, **kwargs):
        return {'status': 'error', 'message': 'Backtesting not available - install required packages'}
    BACKTEST_AVAILABLE = False

from .risk import (
    check_risk_limits,
    calculate_position_size,
    get_portfolio_risk_metrics
)
from .market_data import (
    get_live_price,
    get_orderbook,
    get_recent_trades,
    stream_market_data
)

TRADING_TOOLS = [
    # Strategy execution
    run_sma_strategy,
    run_rsi_strategy,
    run_vwap_strategy,
    run_mean_reversion_strategy,
    run_bollinger_strategy,

    # Order execution
    place_market_order,
    place_limit_order,
    execute_algo_order,
    execute_twap,
    execute_vwap_order,

    # Backtesting
    backtest_strategy,
    optimize_strategy_parameters,
    get_backtest_results,

    # Risk management
    check_risk_limits,
    calculate_position_size,
    get_portfolio_risk_metrics,

    # Market data
    get_live_price,
    get_orderbook,
    get_recent_trades,
    stream_market_data
]