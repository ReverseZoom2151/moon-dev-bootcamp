"""
Market Data Tools for Gordon
============================
Access real-time market data and analytics.
"""

from typing import Dict, Any, Optional, List
from langchain_core.tools import tool
import sys
from pathlib import Path

# Add parent directories to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from gordon.core.market_data_stream import MarketDataStream
from gordon.exchanges.factory import ExchangeFactory
from gordon.agent.config_manager import get_config


@tool
def get_live_price(
    symbol: str,
    exchange: str = "binance"
) -> Dict[str, Any]:
    """Get the current live price of a trading pair.

    Args:
        symbol: Trading pair (e.g., "BTC/USDT")
        exchange: Exchange to get price from

    Returns:
        Current price and related market data
    """
    try:
        market_stream = MarketDataStream(exchange)
        price_data = market_stream.get_ticker(symbol)

        return {
            'status': 'success',
            'symbol': symbol,
            'exchange': exchange,
            'price': price_data.get('last', 0),
            'bid': price_data.get('bid', 0),
            'ask': price_data.get('ask', 0),
            'spread': price_data.get('ask', 0) - price_data.get('bid', 0),
            'volume_24h': price_data.get('volume', 0),
            'change_24h': f"{price_data.get('percentage', 0):.2%}",
            'high_24h': price_data.get('high', 0),
            'low_24h': price_data.get('low', 0),
            'timestamp': price_data.get('timestamp')
        }
    except Exception as e:
        return {
            'status': 'error',
            'message': str(e)
        }


@tool
def get_orderbook(
    symbol: str,
    depth: int = 20,
    exchange: str = "binance"
) -> Dict[str, Any]:
    """Get the order book for a trading pair.

    Args:
        symbol: Trading pair (e.g., "BTC/USDT")
        depth: Number of levels to retrieve
        exchange: Exchange to get order book from

    Returns:
        Order book with bids and asks
    """
    try:
        # Get exchange instance
        config = get_config()
        credentials = config.get_exchange_config(exchange)

        exchange_instance = ExchangeFactory.create_exchange(
            exchange_name=exchange,
            credentials=credentials or {},
            event_bus=None
        )

        orderbook = exchange_instance.get_orderbook(symbol, depth)

        # Calculate market metrics
        best_bid = orderbook['bids'][0][0] if orderbook['bids'] else 0
        best_ask = orderbook['asks'][0][0] if orderbook['asks'] else 0
        spread = best_ask - best_bid
        spread_percentage = (spread / best_bid * 100) if best_bid > 0 else 0

        # Calculate depth metrics
        bid_volume = sum(bid[1] for bid in orderbook['bids'][:depth])
        ask_volume = sum(ask[1] for ask in orderbook['asks'][:depth])

        return {
            'status': 'success',
            'symbol': symbol,
            'exchange': exchange,
            'best_bid': best_bid,
            'best_ask': best_ask,
            'spread': spread,
            'spread_percentage': f"{spread_percentage:.3%}",
            'bid_volume': bid_volume,
            'ask_volume': ask_volume,
            'imbalance': (bid_volume - ask_volume) / (bid_volume + ask_volume) if (bid_volume + ask_volume) > 0 else 0,
            'bids': orderbook['bids'][:10],  # Top 10 bids
            'asks': orderbook['asks'][:10],  # Top 10 asks
            'timestamp': orderbook.get('timestamp')
        }
    except Exception as e:
        return {
            'status': 'error',
            'message': str(e)
        }


@tool
def get_recent_trades(
    symbol: str,
    limit: int = 50,
    exchange: str = "binance"
) -> Dict[str, Any]:
    """Get recent trades for a trading pair.

    Args:
        symbol: Trading pair (e.g., "BTC/USDT")
        limit: Number of trades to retrieve
        exchange: Exchange to get trades from

    Returns:
        Recent trades and analysis
    """
    try:
        # Get exchange instance
        config = get_config()
        credentials = config.get_exchange_config(exchange)

        exchange_instance = ExchangeFactory.create_exchange(
            exchange_name=exchange,
            credentials=credentials or {},
            event_bus=None
        )

        trades = exchange_instance.get_recent_trades(symbol, limit)

        # Analyze trades
        buy_volume = sum(t['amount'] for t in trades if t['side'] == 'buy')
        sell_volume = sum(t['amount'] for t in trades if t['side'] == 'sell')
        total_volume = buy_volume + sell_volume

        buy_count = sum(1 for t in trades if t['side'] == 'buy')
        sell_count = len(trades) - buy_count

        avg_buy_price = sum(t['price'] * t['amount'] for t in trades if t['side'] == 'buy') / buy_volume if buy_volume > 0 else 0
        avg_sell_price = sum(t['price'] * t['amount'] for t in trades if t['side'] == 'sell') / sell_volume if sell_volume > 0 else 0

        # Detect large trades
        avg_trade_size = total_volume / len(trades) if trades else 0
        large_trades = [t for t in trades if t['amount'] > avg_trade_size * 3]

        return {
            'status': 'success',
            'symbol': symbol,
            'exchange': exchange,
            'trade_count': len(trades),
            'analysis': {
                'buy_volume': buy_volume,
                'sell_volume': sell_volume,
                'net_volume': buy_volume - sell_volume,
                'buy_sell_ratio': buy_volume / sell_volume if sell_volume > 0 else float('inf'),
                'buy_count': buy_count,
                'sell_count': sell_count,
                'avg_buy_price': avg_buy_price,
                'avg_sell_price': avg_sell_price,
                'price_trend': 'Bullish' if avg_buy_price > avg_sell_price else 'Bearish'
            },
            'large_trades': {
                'count': len(large_trades),
                'volume': sum(t['amount'] for t in large_trades),
                'trades': large_trades[:5]  # Top 5 large trades
            },
            'recent_trades': trades[:20],  # Most recent 20 trades
            'timestamp': trades[0]['timestamp'] if trades else None
        }
    except Exception as e:
        return {
            'status': 'error',
            'message': str(e)
        }


@tool
def stream_market_data(
    symbol: str,
    data_types: List[str] = ["trades", "orderbook"],
    duration_seconds: int = 30,
    exchange: str = "binance"
) -> Dict[str, Any]:
    """Stream real-time market data for a trading pair.

    Args:
        symbol: Trading pair (e.g., "BTC/USDT")
        data_types: Types of data to stream (trades, orderbook, ticker)
        duration_seconds: How long to stream data
        exchange: Exchange to stream from

    Returns:
        Streamed market data and analysis
    """
    try:
        market_stream = MarketDataStream(exchange)

        # Start streaming
        stream_data = market_stream.stream_data(
            symbol=symbol,
            data_types=data_types,
            duration=duration_seconds
        )

        # Analyze streamed data
        analysis = {}

        if 'trades' in stream_data:
            trades = stream_data['trades']
            total_volume = sum(t['amount'] for t in trades)
            analysis['trade_analysis'] = {
                'total_trades': len(trades),
                'total_volume': total_volume,
                'avg_trade_size': total_volume / len(trades) if trades else 0,
                'price_range': {
                    'high': max(t['price'] for t in trades) if trades else 0,
                    'low': min(t['price'] for t in trades) if trades else 0
                }
            }

        if 'orderbook' in stream_data:
            orderbook_updates = stream_data['orderbook']
            analysis['orderbook_analysis'] = {
                'updates_received': len(orderbook_updates),
                'avg_spread': sum(u['spread'] for u in orderbook_updates) / len(orderbook_updates) if orderbook_updates else 0,
                'liquidity_changes': orderbook_updates[-1]['total_liquidity'] - orderbook_updates[0]['total_liquidity'] if len(orderbook_updates) > 1 else 0
            }

        if 'ticker' in stream_data:
            ticker_updates = stream_data['ticker']
            analysis['price_analysis'] = {
                'updates_received': len(ticker_updates),
                'price_change': ticker_updates[-1]['price'] - ticker_updates[0]['price'] if len(ticker_updates) > 1 else 0,
                'volatility': stream_data.get('volatility', 0)
            }

        return {
            'status': 'success',
            'symbol': symbol,
            'exchange': exchange,
            'duration': duration_seconds,
            'data_types': data_types,
            'analysis': analysis,
            'summary': {
                'market_activity': 'High' if analysis.get('trade_analysis', {}).get('total_trades', 0) > 100 else 'Medium' if analysis.get('trade_analysis', {}).get('total_trades', 0) > 50 else 'Low',
                'trend': 'Up' if analysis.get('price_analysis', {}).get('price_change', 0) > 0 else 'Down',
                'liquidity': 'Improving' if analysis.get('orderbook_analysis', {}).get('liquidity_changes', 0) > 0 else 'Declining'
            },
            'raw_data_samples': {
                'trades': stream_data.get('trades', [])[:5],
                'orderbook': stream_data.get('orderbook', [])[:2],
                'ticker': stream_data.get('ticker', [])[:3]
            }
        }
    except Exception as e:
        return {
            'status': 'error',
            'message': str(e)
        }