"""
Bitfinex Exchange Adapter
=========================
Implementation for Bitfinex exchange integration.
Consolidates functionality from all Bitfinex-specific files across Days 2-56.
"""

import asyncio
import hmac
import hashlib
import time
import json
from typing import Dict, List, Optional, Any
from datetime import datetime
from urllib.parse import urlencode
import aiohttp
import ccxt.async_support as ccxt

from .base import BaseExchange


class BitfinexAdapter(BaseExchange):
    """
    Bitfinex exchange adapter.

    Consolidates Bitfinex functionality from:
    - Day_2_Projects (all bitfinex_*.py files)
    - Day_4_Projects (bitfinex_algo_orders.py, bitfinex_bot.py)
    - Day_5-56_Projects (all bitfinex_*.py variants)
    """

    def __init__(self, credentials: Dict, event_bus: Any):
        """Initialize Bitfinex adapter."""
        super().__init__(credentials, event_bus)

        self.api_key = credentials.get("api_key")
        self.api_secret = credentials.get("api_secret")
        self.testnet = credentials.get("testnet", True)

        # API endpoints
        if self.testnet:
            self.rest_url = "https://api.bitfinex.com"  # Bitfinex doesn't have separate testnet
            self.ws_url = "wss://api-pub.bitfinex.com/ws/2"
        else:
            self.rest_url = "https://api.bitfinex.com"
            self.ws_url = "wss://api-pub.bitfinex.com/ws/2"

        # CCXT client for advanced features
        self.ccxt_client = None

        # Rate limiting (Bitfinex is more restrictive)
        self.rate_limiter = RateLimiter(
            max_requests=credentials.get("rate_limit", 60),
            time_window=60  # per minute
        )

        # Bitfinex-specific mappings
        self.symbol_mapping = {}  # Maps standard symbols to Bitfinex format

    async def initialize(self):
        """Initialize Bitfinex connection."""
        try:
            self.logger.info("Initializing Bitfinex connection...")

            # Initialize CCXT client
            self.ccxt_client = ccxt.bitfinex({
                'apiKey': self.api_key,
                'secret': self.api_secret,
                'enableRateLimit': True,
                'options': {
                    'adjustForTimeDifference': True
                }
            })

            # Note: Bitfinex doesn't have a true testnet
            if self.testnet:
                self.logger.warning("Bitfinex doesn't have a testnet. Using mainnet with small amounts.")

            # Test connection
            await self._test_connection()

            # Load exchange info
            await self._load_exchange_info()

            self.is_connected = True
            self.logger.info("Connected to Bitfinex")

            await self.emit_event("exchange_connected", {"exchange": "bitfinex"})
            return True

        except Exception as e:
            self.logger.error(f"Failed to initialize Bitfinex: {e}")
            return False

    async def disconnect(self):
        """Disconnect from Bitfinex."""
        if self.ccxt_client:
            await self.ccxt_client.close()
        self.is_connected = False
        await self.emit_event("exchange_disconnected", {"exchange": "bitfinex"})
        self.logger.info("Disconnected from Bitfinex")

    async def _test_connection(self):
        """Test connection to Bitfinex."""
        try:
            # Test public endpoint
            ticker = await self.ccxt_client.fetch_ticker("BTC/USDT")
            self.logger.debug(f"Connection test successful. BTC price: {ticker['last']}")

            # Test authenticated endpoint if credentials provided
            if self.api_key and self.api_secret:
                balance = await self.ccxt_client.fetch_balance()
                self.logger.debug("Authentication test successful")

        except Exception as e:
            raise Exception(f"Connection test failed: {e}")

    async def _load_exchange_info(self):
        """Load exchange information."""
        try:
            markets = await self.ccxt_client.load_markets()

            # Store symbol information and mapping
            for symbol, market in markets.items():
                # Store Bitfinex symbol mapping (e.g., BTC/USDT -> tBTCUST)
                self.symbol_mapping[symbol] = market['id']

                if market['quote'] in ['USD', 'USDT', 'UST']:
                    self.symbols_info[symbol] = {
                        'min_amount': market['limits']['amount']['min'],
                        'min_cost': market['limits']['cost']['min'],
                        'price_precision': market['precision']['price'],
                        'amount_precision': market['precision']['amount'],
                        'base': market['base'],
                        'quote': market['quote'],
                        'bitfinex_symbol': market['id']
                    }

            self.logger.info(f"Loaded info for {len(self.symbols_info)} symbols")

        except Exception as e:
            self.logger.error(f"Failed to load exchange info: {e}")

    def _format_symbol(self, symbol: str) -> str:
        """Format symbol for Bitfinex (e.g., BTC/USDT -> tBTCUST)."""
        if symbol in self.symbol_mapping:
            return self.symbol_mapping[symbol]

        # Manual formatting if not in mapping
        if "/" in symbol:
            base, quote = symbol.split("/")
            # Bitfinex uses 't' prefix for trading pairs
            return f"t{base}{quote}"

        return symbol

    # Order Management

    async def place_order(self, symbol: str, side: str, amount: float,
                         order_type: str = "market", price: Optional[float] = None,
                         **kwargs) -> Dict:
        """Place an order on Bitfinex."""
        try:
            await self.rate_limiter.acquire()

            # Format symbol for Bitfinex
            if "/" not in symbol:
                symbol = f"{symbol}/USDT"

            # Adjust amount for Bitfinex (negative for sell)
            if side.lower() == "sell":
                amount = -abs(amount)
            else:
                amount = abs(amount)

            # Prepare order parameters
            params = {}

            # Handle order flags (from Day_25_Projects)
            if kwargs.get("reduce_only", False):
                params['flags'] = 1024  # Reduce-only flag

            if kwargs.get("post_only", False):
                params['flags'] = params.get('flags', 0) | 4096  # Post-only flag

            # Place order using CCXT
            if order_type.lower() == "market":
                order = await self.ccxt_client.create_market_order(
                    symbol, side.lower(), abs(amount), params=params
                )
            elif order_type.lower() == "limit":
                if price is None:
                    raise ValueError("Price required for limit orders")
                order = await self.ccxt_client.create_limit_order(
                    symbol, side.lower(), abs(amount), price, params=params
                )
            elif order_type.lower() == "stop":
                if price is None:
                    raise ValueError("Stop price required for stop orders")
                params['stopPrice'] = price
                order = await self.ccxt_client.create_order(
                    symbol, 'stop', side.lower(), abs(amount), price, params=params
                )
            else:
                raise ValueError(f"Unknown order type: {order_type}")

            # Format response
            order_info = {
                "id": order['id'],
                "symbol": symbol,
                "side": side,
                "amount": abs(amount),
                "price": order.get('price', price),
                "type": order_type,
                "status": order['status'],
                "timestamp": order['timestamp']
            }

            await self.emit_event("order_placed", order_info)
            return order_info

        except Exception as e:
            self.logger.error(f"Failed to place order: {e}")
            await self.emit_event("order_failed", {"error": str(e)})
            return {}

    async def cancel_order(self, order_id: str, symbol: Optional[str] = None) -> bool:
        """Cancel an order."""
        try:
            await self.rate_limiter.acquire()

            if symbol and "/" not in symbol:
                symbol = f"{symbol}/USDT"

            result = await self.ccxt_client.cancel_order(order_id, symbol)

            if result:
                await self.emit_event("order_cancelled", {"order_id": order_id})
                return True
            return False

        except Exception as e:
            self.logger.error(f"Failed to cancel order: {e}")
            return False

    async def get_order(self, order_id: str, symbol: Optional[str] = None) -> Dict:
        """Get order information."""
        try:
            await self.rate_limiter.acquire()

            if symbol and "/" not in symbol:
                symbol = f"{symbol}/USDT"

            order = await self.ccxt_client.fetch_order(order_id, symbol)

            return {
                "id": order['id'],
                "symbol": order['symbol'],
                "side": order['side'],
                "amount": order['amount'],
                "price": order['price'],
                "status": order['status'],
                "filled": order['filled'],
                "remaining": order['remaining'],
                "timestamp": order['timestamp']
            }

        except Exception as e:
            self.logger.error(f"Failed to get order: {e}")
            return {}

    async def get_open_orders(self, symbol: Optional[str] = None) -> List[Dict]:
        """Get open orders."""
        try:
            await self.rate_limiter.acquire()

            if symbol and "/" not in symbol:
                symbol = f"{symbol}/USDT"

            orders = await self.ccxt_client.fetch_open_orders(symbol)

            return [
                {
                    "id": order['id'],
                    "symbol": order['symbol'],
                    "side": order['side'],
                    "amount": order['amount'],
                    "price": order['price'],
                    "status": order['status'],
                    "timestamp": order['timestamp']
                }
                for order in orders
            ]

        except Exception as e:
            self.logger.error(f"Failed to get open orders: {e}")
            return []

    # Position Management

    async def get_position(self, symbol: str) -> Dict:
        """Get position for a symbol."""
        try:
            await self.rate_limiter.acquire()

            if "/" not in symbol:
                symbol = f"{symbol}/USDT"

            # Bitfinex uses different endpoint for positions
            positions = await self.ccxt_client.fetch_positions([symbol])

            if positions:
                pos = positions[0]
                return {
                    "symbol": pos['symbol'],
                    "amount": pos['contracts'],
                    "side": 'long' if pos['side'] == 'long' else 'short',
                    "entry_price": pos['markPrice'],
                    "mark_price": pos['markPrice'],
                    "pnl": pos['unrealizedPnl'],
                    "pnl_percent": pos['percentage'],
                    "margin": pos['initialMargin']
                }

            return {}

        except Exception as e:
            self.logger.error(f"Failed to get position: {e}")
            return {}

    async def get_all_positions(self) -> List[Dict]:
        """Get all open positions."""
        try:
            await self.rate_limiter.acquire()

            # For Bitfinex spot trading, we use balance
            # For margin/derivatives, we use positions
            positions = []

            # Try to get positions (for margin/derivatives)
            try:
                margin_positions = await self.ccxt_client.fetch_positions()
                for pos in margin_positions:
                    if pos['contracts'] != 0:
                        positions.append({
                            "symbol": pos['symbol'],
                            "amount": pos['contracts'],
                            "side": 'long' if pos['side'] == 'long' else 'short',
                            "entry_price": pos.get('markPrice', 0),
                            "mark_price": pos.get('markPrice', 0),
                            "pnl": pos.get('unrealizedPnl', 0),
                            "pnl_percent": pos.get('percentage', 0),
                            "margin": pos.get('initialMargin', 0)
                        })
            except:
                pass  # Not all accounts have margin trading

            # Get spot positions from balance
            balance = await self.ccxt_client.fetch_balance()
            for currency, info in balance.items():
                if info['total'] > 0 and currency != 'USD' and currency != 'USDT':
                    # Get current price
                    try:
                        ticker = await self.ccxt_client.fetch_ticker(f"{currency}/USDT")
                        positions.append({
                            "symbol": f"{currency}/USDT",
                            "amount": info['total'],
                            "side": 'long',
                            "entry_price": 0,  # Unknown for spot
                            "mark_price": ticker['last'],
                            "pnl": 0,  # Can't calculate without entry price
                            "pnl_percent": 0,
                            "margin": 0
                        })
                    except:
                        pass

            return positions

        except Exception as e:
            self.logger.error(f"Failed to get positions: {e}")
            return []

    async def close_position(self, symbol: str, amount: Optional[float] = None) -> Dict:
        """Close a position."""
        try:
            position = await self.get_position(symbol)
            if not position:
                self.logger.warning(f"No position found for {symbol}")
                return {}

            # Determine side to close
            close_side = "sell" if position['side'] == 'long' else "buy"
            close_amount = amount or abs(position['amount'])

            return await self.place_order(
                symbol=symbol,
                side=close_side,
                amount=close_amount,
                order_type="market",
                reduce_only=True
            )

        except Exception as e:
            self.logger.error(f"Failed to close position: {e}")
            return {}

    # Market Data

    async def get_balance(self, asset: Optional[str] = None) -> Dict:
        """Get account balance."""
        try:
            await self.rate_limiter.acquire()

            balance = await self.ccxt_client.fetch_balance()

            if asset:
                asset = asset.upper()
                return {
                    "total": balance.get(asset, {}).get('total', 0),
                    "free": balance.get(asset, {}).get('free', 0),
                    "used": balance.get(asset, {}).get('used', 0)
                }

            return {
                currency: {
                    "total": info['total'],
                    "free": info['free'],
                    "used": info['used']
                }
                for currency, info in balance.items()
                if info['total'] > 0
            }

        except Exception as e:
            self.logger.error(f"Failed to get balance: {e}")
            return {}

    async def get_ohlcv(self, symbol: str, timeframe: str = "1h",
                       limit: int = 100) -> List[List]:
        """Get OHLCV candle data."""
        try:
            await self.rate_limiter.acquire()

            if "/" not in symbol:
                symbol = f"{symbol}/USDT"

            # Bitfinex timeframe mapping
            timeframe_map = {
                "1m": "1m",
                "5m": "5m",
                "15m": "15m",
                "30m": "30m",
                "1h": "1h",
                "3h": "3h",
                "6h": "6h",
                "12h": "12h",
                "1d": "1D",
                "1w": "1W"
            }

            bf_timeframe = timeframe_map.get(timeframe, timeframe)
            ohlcv = await self.ccxt_client.fetch_ohlcv(symbol, bf_timeframe, limit=limit)
            return ohlcv

        except Exception as e:
            self.logger.error(f"Failed to get OHLCV: {e}")
            return []

    async def get_ticker(self, symbol: str) -> Dict:
        """Get ticker information."""
        try:
            await self.rate_limiter.acquire()

            if "/" not in symbol:
                symbol = f"{symbol}/USDT"

            ticker = await self.ccxt_client.fetch_ticker(symbol)

            return {
                "symbol": ticker['symbol'],
                "bid": ticker['bid'],
                "ask": ticker['ask'],
                "last": ticker['last'],
                "volume": ticker['baseVolume'],
                "change_24h": ticker['percentage']
            }

        except Exception as e:
            self.logger.error(f"Failed to get ticker: {e}")
            return {}

    async def get_orderbook(self, symbol: str, limit: int = 20) -> Dict:
        """Get orderbook."""
        try:
            await self.rate_limiter.acquire()

            if "/" not in symbol:
                symbol = f"{symbol}/USDT"

            orderbook = await self.ccxt_client.fetch_order_book(symbol, limit)

            return {
                "bids": orderbook['bids'],
                "asks": orderbook['asks'],
                "timestamp": orderbook['timestamp']
            }

        except Exception as e:
            self.logger.error(f"Failed to get orderbook: {e}")
            return {"bids": [], "asks": []}

    # Bitfinex-specific methods

    async def get_funding_rate(self, symbol: str) -> float:
        """Get funding rate for perpetual contracts."""
        try:
            await self.rate_limiter.acquire()

            # Bitfinex funding is different - it's for margin trading
            # Get funding stats
            bf_symbol = self._format_symbol(f"{symbol}/USDT")

            # Using raw API call for funding
            async with aiohttp.ClientSession() as session:
                url = f"{self.rest_url}/v2/calc/trade/avg?symbol={bf_symbol}&amount=10000"
                async with session.get(url) as response:
                    data = await response.json()
                    if data and len(data) > 0:
                        # Return rate as percentage
                        return float(data[0]) / 100

            return 0.0

        except Exception as e:
            self.logger.error(f"Failed to get funding rate: {e}")
            return 0.0

    async def set_leverage(self, symbol: str, leverage: int) -> bool:
        """Set leverage for a symbol (Bitfinex margin trading)."""
        try:
            # Bitfinex handles leverage differently through margin trading
            # Leverage is determined by the amount of margin used
            self.logger.info(f"Bitfinex leverage is managed through margin allocation")
            return True

        except Exception as e:
            self.logger.error(f"Failed to set leverage: {e}")
            return False

    async def get_trades(self, symbol: str, limit: int = 100) -> List[Dict]:
        """Get recent trades."""
        try:
            await self.rate_limiter.acquire()

            if "/" not in symbol:
                symbol = f"{symbol}/USDT"

            trades = await self.ccxt_client.fetch_trades(symbol, limit=limit)

            return [
                {
                    "id": trade['id'],
                    "symbol": trade['symbol'],
                    "price": trade['price'],
                    "amount": trade['amount'],
                    "side": trade['side'],
                    "timestamp": trade['timestamp']
                }
                for trade in trades
            ]

        except Exception as e:
            self.logger.error(f"Failed to get trades: {e}")
            return []

    # Advanced order types (from Day_4_Projects)

    async def iceberg_order(self, symbol: str, side: str, total_amount: float,
                          slice_amount: float, price: Optional[float] = None) -> List[Dict]:
        """
        Place an iceberg order (hidden large order).
        Bitfinex supports this natively.
        """
        try:
            params = {
                'hidden': True,  # Hide order from orderbook
                'iceberg': slice_amount  # Show only this amount
            }

            order = await self.ccxt_client.create_limit_order(
                f"{symbol}/USDT",
                side.lower(),
                total_amount,
                price,
                params=params
            )

            return [{
                "id": order['id'],
                "symbol": order['symbol'],
                "side": side,
                "total_amount": total_amount,
                "visible_amount": slice_amount,
                "price": price,
                "status": order['status']
            }]

        except Exception as e:
            self.logger.error(f"Failed to place iceberg order: {e}")
            return []

    async def trailing_stop(self, symbol: str, side: str, amount: float,
                          trail_amount: float) -> Dict:
        """
        Place a trailing stop order.
        Bitfinex supports trailing stops natively.
        """
        try:
            params = {
                'trailingAmount': trail_amount,  # Trail by this amount
                'type': 'trailing-stop'
            }

            order = await self.ccxt_client.create_order(
                f"{symbol}/USDT",
                'trailing-stop',
                side.lower(),
                amount,
                None,
                params=params
            )

            return {
                "id": order['id'],
                "symbol": order['symbol'],
                "side": side,
                "amount": amount,
                "trail_amount": trail_amount,
                "status": order['status']
            }

        except Exception as e:
            self.logger.error(f"Failed to place trailing stop: {e}")
            return {}

    # Margin trading specific (from Day_25-26 projects)

    async def get_margin_info(self) -> Dict:
        """Get margin trading information."""
        try:
            await self.rate_limiter.acquire()

            # Get margin balance
            async with aiohttp.ClientSession() as session:
                # This would need proper authentication
                # Simplified version
                return {
                    "margin_balance": 0,
                    "margin_net": 0,
                    "margin_required": 0
                }

        except Exception as e:
            self.logger.error(f"Failed to get margin info: {e}")
            return {}

    async def get_lending_rates(self, currency: str = "USD") -> Dict:
        """Get lending rates for margin funding."""
        try:
            await self.rate_limiter.acquire()

            # Get funding book
            bf_symbol = f"f{currency}"

            async with aiohttp.ClientSession() as session:
                url = f"{self.rest_url}/v2/book/{bf_symbol}/P0"
                async with session.get(url) as response:
                    data = await response.json()

                    if data:
                        asks = [d for d in data if d[3] > 0]  # Positive = ask
                        bids = [d for d in data if d[3] < 0]  # Negative = bid

                        return {
                            "currency": currency,
                            "best_ask_rate": asks[0][0] if asks else 0,
                            "best_bid_rate": bids[0][0] if bids else 0
                        }

            return {}

        except Exception as e:
            self.logger.error(f"Failed to get lending rates: {e}")
            return {}


class RateLimiter:
    """Rate limiter for API calls."""

    def __init__(self, max_requests: int, time_window: int):
        """
        Initialize rate limiter.

        Args:
            max_requests: Maximum requests allowed
            time_window: Time window in seconds
        """
        self.max_requests = max_requests
        self.time_window = time_window
        self.requests = []

    async def acquire(self):
        """Wait if necessary to respect rate limits."""
        now = time.time()

        # Remove old requests outside the time window
        self.requests = [req for req in self.requests if now - req < self.time_window]

        # Check if we're at the limit
        if len(self.requests) >= self.max_requests:
            # Calculate wait time
            oldest_request = self.requests[0]
            wait_time = self.time_window - (now - oldest_request)

            if wait_time > 0:
                await asyncio.sleep(wait_time)

        # Record this request
        self.requests.append(time.time())