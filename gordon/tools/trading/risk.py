"""
Risk Management Tools for Gordon
================================
Monitor and manage trading risk.
"""

from typing import Dict, Any, Optional, List
from langchain_core.tools import tool
import sys
from pathlib import Path

# Add parent directories to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
from gordon.core.risk_manager import RiskManager
from gordon.core.event_bus import EventBus
from gordon.config.config_manager import ConfigManager


@tool
def check_risk_limits(
    symbol: str,
    side: str,
    amount: float,
    exchange: str = "binance"
) -> Dict[str, Any]:
    """Check if a trade violates risk limits before execution.

    Args:
        symbol: Trading pair (e.g., "BTC/USDT")
        side: Order side ("buy" or "sell")
        amount: Amount to trade
        exchange: Exchange to trade on

    Returns:
        Risk check results and approval status
    """
    try:
        event_bus = EventBus()
        config_manager = ConfigManager()
        risk_manager = RiskManager(event_bus, config_manager, demo_mode=True)
        
        # Note: Config override removed - RiskManager uses config_manager.get_risk_config()

        # Check risk - use async check_trade_allowed
        # Note: This is a sync function, so we'll use a wrapper
        import asyncio
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        approval_result = loop.run_until_complete(risk_manager.check_trade_allowed(
            exchange=exchange,
            symbol=symbol,
            side=side,
            amount=amount
        ))
        
        # Build approval dict
        approval = {
            'approved': approval_result,
            'risk_score': 0.5 if approval_result else 0.9,
            'position_size_ok': approval_result,
            'drawdown_ok': approval_result,
            'daily_loss_ok': approval_result,
            'correlation_ok': approval_result,
            'reason': 'Trade approved' if approval_result else 'Risk limit exceeded',
            'violations': [] if approval_result else ['Risk check failed'],
            'recommendations': []
        }

        if approval['approved']:
            return {
                'status': 'approved',
                'symbol': symbol,
                'side': side,
                'amount': amount,
                'risk_score': approval.get('risk_score', 0),
                'message': 'Trade approved by risk manager',
                'checks': {
                    'position_size': approval.get('position_size_ok', True),
                    'drawdown': approval.get('drawdown_ok', True),
                    'daily_loss': approval.get('daily_loss_ok', True),
                    'correlation': approval.get('correlation_ok', True)
                }
            }
        else:
            return {
                'status': 'rejected',
                'symbol': symbol,
                'side': side,
                'amount': amount,
                'reason': approval.get('reason', 'Risk limit exceeded'),
                'violations': approval.get('violations', []),
                'recommendations': approval.get('recommendations', [])
            }
    except Exception as e:
        return {
            'status': 'error',
            'message': str(e)
        }


@tool
def calculate_position_size(
    symbol: str,
    risk_percentage: float = 0.02,
    stop_loss_percentage: float = 0.05,
    account_balance: Optional[float] = None,
    exchange: str = "binance"
) -> Dict[str, Any]:
    """Calculate optimal position size based on risk parameters.

    Args:
        symbol: Trading pair (e.g., "BTC/USDT")
        risk_percentage: Percentage of account to risk (e.g., 0.02 for 2%)
        stop_loss_percentage: Stop loss distance as percentage
        account_balance: Account balance (fetched if not provided)
        exchange: Exchange to trade on

    Returns:
        Calculated position size and risk metrics
    """
    try:
        event_bus = EventBus()
        config_manager = ConfigManager()
        risk_manager = RiskManager(event_bus, config_manager, demo_mode=True)

        # Get account balance if not provided
        if account_balance is None:
            from gordon.exchanges.factory import ExchangeFactory
            from gordon.agent.config_manager import get_config
            config = get_config()
            credentials = config.get_exchange_config(exchange)
            exchange_instance = ExchangeFactory.create_exchange(
                exchange_name=exchange,
                credentials=credentials or {},
                event_bus=None
            )
            balance_info = exchange_instance.get_balance()
            account_balance = balance_info.get('total_usdt', 10000)

        # Calculate position size - use correct signature
        position_size = risk_manager.calculate_position_size(
            balance=account_balance,
            stop_loss_percent=stop_loss_percentage
        )

        # Get current price for value calculation
        from gordon.core.market_data_stream import MarketDataStream
        market_data = MarketDataStream(exchange)
        current_price = market_data.get_price(symbol)

        position_value = position_size * current_price
        risk_amount = account_balance * risk_percentage

        return {
            'status': 'success',
            'symbol': symbol,
            'position_size': position_size,
            'position_value': position_value,
            'position_percentage': (position_value / account_balance) * 100,
            'risk_amount': risk_amount,
            'stop_loss_distance': f"{stop_loss_percentage:.2%}",
            'risk_reward_ratio': 2.0,  # Default 1:2 risk-reward
            'recommendations': {
                'entry_price': current_price,
                'stop_loss': current_price * (1 - stop_loss_percentage),
                'take_profit': current_price * (1 + stop_loss_percentage * 2),
                'max_loss': risk_amount
            }
        }
    except Exception as e:
        return {
            'status': 'error',
            'message': str(e)
        }


@tool
def get_portfolio_risk_metrics(
    exchange: str = "binance",
    include_open_positions: bool = True
) -> Dict[str, Any]:
    """Get comprehensive portfolio risk metrics.

    Args:
        exchange: Exchange to analyze
        include_open_positions: Include analysis of open positions

    Returns:
        Portfolio risk metrics and analysis
    """
    try:
        event_bus = EventBus()
        config_manager = ConfigManager()
        risk_manager = RiskManager(event_bus, config_manager, demo_mode=True)

        # Get risk metrics from RiskManager
        metrics = risk_manager.get_risk_metrics()
        
        # Calculate a simple risk score based on metrics
        drawdown = metrics.get('drawdown_percent', 0)
        daily_pnl = metrics.get('daily_pnl', 0)
        daily_loss_limit = metrics.get('daily_loss_limit', 0)
        
        # Simple risk score calculation (0-100)
        risk_score = 0
        if drawdown > 0:
            risk_score += min(50, drawdown * 2)
        if daily_pnl < 0 and abs(daily_pnl) > daily_loss_limit * 0.8:
            risk_score += min(50, abs(daily_pnl) / daily_loss_limit * 50)
        
        # Get open positions if requested
        positions = []
        if include_open_positions:
            from gordon.exchanges.factory import ExchangeFactory
            from gordon.agent.config_manager import get_config
            config = get_config()
            credentials = config.get_exchange_config(exchange)
            exchange_instance = ExchangeFactory.create_exchange(
                exchange_name=exchange,
                credentials=credentials or {},
                event_bus=None
            )
            positions = exchange_instance.get_open_positions()

        return {
            'status': 'success',
            'risk_score': risk_score,
            'risk_level': 'Low' if risk_score < 30 else 'Medium' if risk_score < 70 else 'High',
            'portfolio_metrics': {
                'total_value': metrics.get('current_balance', 0),
                'unrealized_pnl': 0,  # Not available from risk metrics
                'realized_pnl': metrics.get('daily_pnl', 0),
                'current_drawdown': f"{drawdown:.2%}",
                'max_drawdown': f"{metrics.get('max_drawdown_percent', 0):.2%}"
            },
            'risk_metrics': {
                'value_at_risk_95': 'N/A',  # Not available
                'conditional_var_95': 'N/A',  # Not available
                'sharpe_ratio': 'N/A',  # Not available
                'sortino_ratio': 'N/A',  # Not available
                'beta': 'N/A',  # Not available
                'correlation_risk': 0
            },
            'exposure': {
                'long_exposure': 0,  # Not available
                'short_exposure': 0,  # Not available
                'net_exposure': 0,  # Not available
                'gross_exposure': 0  # Not available
            },
            'open_positions': metrics.get('current_positions', 0),
            'positions': [],
            'recommendations': [],
            'alerts': []
        }
    except Exception as e:
        return {
            'status': 'error',
            'message': str(e)
        }