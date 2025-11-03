# This file makes the directory a Python package
from typing_extensions import Callable

# Financial research tools (from Dexter)
from .finance.filings import get_filings
from .finance.filings import get_10K_filing_items
from .finance.filings import get_10Q_filing_items
from .finance.filings import get_8K_filing_items
from .finance.fundamentals import get_income_statements
from .finance.fundamentals import get_balance_sheets
from .finance.fundamentals import get_cash_flow_statements
from .finance.metrics import get_financial_metrics_snapshot
from .finance.metrics import get_financial_metrics
from .finance.prices import get_price_snapshot
from .finance.prices import get_prices
from .finance.news import get_news
from .finance.estimates import get_analyst_estimates
from .search.google import search_google_news

# Trading tools (our advanced system)
from .trading import (
    # Strategies
    run_sma_strategy,
    run_rsi_strategy,
    run_vwap_strategy,
    run_mean_reversion_strategy,
    run_bollinger_strategy,
    # Orders
    place_market_order,
    place_limit_order,
    execute_algo_order,
    execute_twap,
    execute_vwap_order,
    # Backtesting
    backtest_strategy,
    optimize_strategy_parameters,
    get_backtest_results,
    # Risk
    check_risk_limits,
    calculate_position_size,
    get_portfolio_risk_metrics,
    # Market Data
    get_live_price,
    get_orderbook,
    get_recent_trades,
    stream_market_data
)

# Combined tools list - Gordon's full arsenal!
TOOLS: list[Callable[..., any]] = [
    # === FINANCIAL RESEARCH TOOLS ===
    get_income_statements,
    get_balance_sheets,
    get_cash_flow_statements,
    get_10K_filing_items,
    get_10Q_filing_items,
    get_8K_filing_items,
    get_filings,
    get_price_snapshot,
    get_prices,
    get_financial_metrics_snapshot,
    get_financial_metrics,
    get_news,
    get_analyst_estimates,
    search_google_news,

    # === TRADING STRATEGY TOOLS ===
    run_sma_strategy,
    run_rsi_strategy,
    run_vwap_strategy,
    run_mean_reversion_strategy,
    run_bollinger_strategy,

    # === ORDER EXECUTION TOOLS ===
    place_market_order,
    place_limit_order,
    execute_algo_order,
    execute_twap,
    execute_vwap_order,

    # === BACKTESTING TOOLS ===
    backtest_strategy,
    optimize_strategy_parameters,
    get_backtest_results,

    # === RISK MANAGEMENT TOOLS ===
    check_risk_limits,
    calculate_position_size,
    get_portfolio_risk_metrics,

    # === MARKET DATA TOOLS ===
    get_live_price,
    get_orderbook,
    get_recent_trades,
    stream_market_data
]
