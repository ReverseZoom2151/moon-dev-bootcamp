"""
Order Execution Tools for Gordon
=================================
Execute various order types on exchanges.
"""

from typing import Dict, Any, Optional
from langchain_core.tools import tool
import sys
from pathlib import Path

# Add parent directories to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from gordon.exchanges.factory import ExchangeFactory
from gordon.agent.config_manager import get_config


@tool
def place_market_order(
    symbol: str,
    side: str,
    amount: float,
    exchange: str = "binance",
    dry_run: bool = True
) -> Dict[str, Any]:
    """Place a market order on an exchange.

    Args:
        symbol: Trading pair (e.g., "BTC/USDT")
        side: Order side ("buy" or "sell")
        amount: Amount to trade
        exchange: Exchange to trade on
        dry_run: If True, simulate order without executing

    Returns:
        Order execution results
    """
    try:
        if dry_run:
            return {
                'status': 'simulated',
                'order_type': 'market',
                'symbol': symbol,
                'side': side,
                'amount': amount,
                'message': 'Dry run - order not executed'
            }

        # Get exchange credentials from config
        config = get_config()
        credentials = config.get_exchange_config(exchange)

        if not credentials:
            return {
                'status': 'error',
                'message': f'No credentials configured for {exchange}'
            }

        # Create exchange instance
        exchange_instance = ExchangeFactory.create_exchange(
            exchange_name=exchange,
            credentials=credentials,
            event_bus=None  # Optional event bus
        )

        # Place order
        order = exchange_instance.place_order(
            symbol=symbol,
            order_type='market',
            side=side,
            amount=amount
        )

        return {
            'status': 'success',
            'order_id': order.get('id'),
            'symbol': symbol,
            'side': side,
            'amount': amount,
            'executed_price': order.get('price'),
            'details': order
        }
    except Exception as e:
        return {
            'status': 'error',
            'message': str(e)
        }


@tool
def place_limit_order(
    symbol: str,
    side: str,
    amount: float,
    price: float,
    exchange: str = "binance",
    dry_run: bool = True
) -> Dict[str, Any]:
    """Place a limit order on an exchange.

    Args:
        symbol: Trading pair (e.g., "BTC/USDT")
        side: Order side ("buy" or "sell")
        amount: Amount to trade
        price: Limit price
        exchange: Exchange to trade on
        dry_run: If True, simulate order without executing

    Returns:
        Order execution results
    """
    try:
        if dry_run:
            return {
                'status': 'simulated',
                'order_type': 'limit',
                'symbol': symbol,
                'side': side,
                'amount': amount,
                'price': price,
                'message': 'Dry run - order not executed'
            }

        # Get exchange credentials from config
        config = get_config()
        credentials = config.get_exchange_config(exchange)

        if not credentials:
            return {
                'status': 'error',
                'message': f'No credentials configured for {exchange}'
            }

        # Create exchange instance
        exchange_instance = ExchangeFactory.create_exchange(
            exchange_name=exchange,
            credentials=credentials,
            event_bus=None
        )

        order = exchange_instance.place_order(
            symbol=symbol,
            order_type='limit',
            side=side,
            amount=amount,
            price=price
        )

        return {
            'status': 'success',
            'order_id': order.get('id'),
            'symbol': symbol,
            'side': side,
            'amount': amount,
            'limit_price': price,
            'details': order
        }
    except Exception as e:
        return {
            'status': 'error',
            'message': str(e)
        }


@tool
def execute_algo_order(
    symbol: str,
    side: str,
    amount: float,
    algo_type: str = "twap",
    duration_minutes: int = 60,
    exchange: str = "binance",
    dry_run: bool = True
) -> Dict[str, Any]:
    """Execute an algorithmic order (TWAP, VWAP, etc.).

    Args:
        symbol: Trading pair (e.g., "BTC/USDT")
        side: Order side ("buy" or "sell")
        amount: Total amount to trade
        algo_type: Algorithm type ("twap", "vwap", "iceberg")
        duration_minutes: Duration for algo execution
        exchange: Exchange to trade on
        dry_run: If True, simulate order without executing

    Returns:
        Algo order execution results
    """
    try:
        from gordon.core.algo_orders import AlgoOrderManager

        manager = AlgoOrderManager(exchange)

        if dry_run:
            return {
                'status': 'simulated',
                'algo_type': algo_type,
                'symbol': symbol,
                'side': side,
                'amount': amount,
                'duration': duration_minutes,
                'message': 'Dry run - algo order not executed'
            }

        result = manager.execute_algo_order(
            algo_type=algo_type,
            symbol=symbol,
            side=side,
            amount=amount,
            duration_minutes=duration_minutes
        )

        return {
            'status': 'success',
            'algo_id': result.get('id'),
            'algo_type': algo_type,
            'symbol': symbol,
            'side': side,
            'amount': amount,
            'slices': result.get('slices'),
            'details': result
        }
    except Exception as e:
        return {
            'status': 'error',
            'message': str(e)
        }


@tool
def execute_twap(
    symbol: str,
    side: str,
    amount: float,
    duration_minutes: int = 60,
    slices: int = 10,
    exchange: str = "binance",
    dry_run: bool = True
) -> Dict[str, Any]:
    """Execute a Time-Weighted Average Price (TWAP) order.

    Args:
        symbol: Trading pair (e.g., "BTC/USDT")
        side: Order side ("buy" or "sell")
        amount: Total amount to trade
        duration_minutes: Duration for TWAP execution
        slices: Number of order slices
        exchange: Exchange to trade on
        dry_run: If True, simulate order without executing

    Returns:
        TWAP order execution results
    """
    try:
        from gordon.core.algo_orders.twap import TWAPOrder

        twap = TWAPOrder({
            'symbol': symbol,
            'side': side,
            'amount': amount,
            'duration_minutes': duration_minutes,
            'slices': slices,
            'exchange': exchange
        })

        if dry_run:
            return {
                'status': 'simulated',
                'algo_type': 'TWAP',
                'symbol': symbol,
                'side': side,
                'amount': amount,
                'duration': duration_minutes,
                'slices': slices,
                'message': 'Dry run - TWAP not executed'
            }

        result = twap.execute()

        return {
            'status': 'success',
            'algo_type': 'TWAP',
            'symbol': symbol,
            'side': side,
            'amount': amount,
            'avg_price': result.get('avg_price'),
            'slices_executed': result.get('slices_executed'),
            'details': result
        }
    except Exception as e:
        return {
            'status': 'error',
            'message': str(e)
        }


@tool
def execute_vwap_order(
    symbol: str,
    side: str,
    amount: float,
    lookback_minutes: int = 30,
    exchange: str = "binance",
    dry_run: bool = True
) -> Dict[str, Any]:
    """Execute a Volume-Weighted Average Price (VWAP) order.

    Args:
        symbol: Trading pair (e.g., "BTC/USDT")
        side: Order side ("buy" or "sell")
        amount: Total amount to trade
        lookback_minutes: Minutes to look back for volume profile
        exchange: Exchange to trade on
        dry_run: If True, simulate order without executing

    Returns:
        VWAP order execution results
    """
    try:
        from gordon.core.algo_orders.vwap import VWAPOrder

        vwap = VWAPOrder({
            'symbol': symbol,
            'side': side,
            'amount': amount,
            'lookback_minutes': lookback_minutes,
            'exchange': exchange
        })

        if dry_run:
            return {
                'status': 'simulated',
                'algo_type': 'VWAP',
                'symbol': symbol,
                'side': side,
                'amount': amount,
                'message': 'Dry run - VWAP not executed'
            }

        result = vwap.execute()

        return {
            'status': 'success',
            'algo_type': 'VWAP',
            'symbol': symbol,
            'side': side,
            'amount': amount,
            'avg_price': result.get('avg_price'),
            'vwap_price': result.get('vwap'),
            'slippage': result.get('slippage'),
            'details': result
        }
    except Exception as e:
        return {
            'status': 'error',
            'message': str(e)
        }