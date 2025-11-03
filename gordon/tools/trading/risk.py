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
        risk_manager = RiskManager({
            'max_position_size': 0.1,  # 10% of portfolio
            'max_drawdown': 0.2,  # 20% max drawdown
            'daily_loss_limit': 0.05,  # 5% daily loss limit
            'correlation_limit': 0.7  # Max correlation between positions
        })

        # Check risk
        approval = risk_manager.check_trade(
            symbol=symbol,
            side=side,
            amount=amount,
            exchange=exchange
        )

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
        risk_manager = RiskManager()

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

        # Calculate position size
        position_size = risk_manager.calculate_position_size(
            account_balance=account_balance,
            risk_percentage=risk_percentage,
            stop_loss_percentage=stop_loss_percentage
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
        risk_manager = RiskManager()

        # Get portfolio metrics
        metrics = risk_manager.get_portfolio_metrics(exchange)

        # Calculate risk scores
        var_95 = metrics.get('value_at_risk_95', 0)
        cvar_95 = metrics.get('conditional_var_95', 0)
        sharpe = metrics.get('sharpe_ratio', 0)
        sortino = metrics.get('sortino_ratio', 0)
        max_dd = metrics.get('max_drawdown', 0)

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

        # Calculate overall risk score (0-100)
        risk_score = risk_manager.calculate_risk_score(metrics)

        return {
            'status': 'success',
            'risk_score': risk_score,
            'risk_level': 'Low' if risk_score < 30 else 'Medium' if risk_score < 70 else 'High',
            'portfolio_metrics': {
                'total_value': metrics.get('total_value', 0),
                'unrealized_pnl': metrics.get('unrealized_pnl', 0),
                'realized_pnl': metrics.get('realized_pnl', 0),
                'current_drawdown': f"{metrics.get('current_drawdown', 0):.2%}",
                'max_drawdown': f"{max_dd:.2%}"
            },
            'risk_metrics': {
                'value_at_risk_95': f"{var_95:.2%}",
                'conditional_var_95': f"{cvar_95:.2%}",
                'sharpe_ratio': f"{sharpe:.2f}",
                'sortino_ratio': f"{sortino:.2f}",
                'beta': f"{metrics.get('beta', 0):.2f}",
                'correlation_risk': metrics.get('correlation_risk', 0)
            },
            'exposure': {
                'long_exposure': metrics.get('long_exposure', 0),
                'short_exposure': metrics.get('short_exposure', 0),
                'net_exposure': metrics.get('net_exposure', 0),
                'gross_exposure': metrics.get('gross_exposure', 0)
            },
            'open_positions': len(positions),
            'positions': positions[:5] if positions else [],  # First 5 positions
            'recommendations': risk_manager.get_risk_recommendations(metrics),
            'alerts': risk_manager.get_risk_alerts(metrics)
        }
    except Exception as e:
        return {
            'status': 'error',
            'message': str(e)
        }