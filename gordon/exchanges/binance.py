"""
Binance Exchange Adapter
========================
Implementation for Binance exchange integration.
Consolidates functionality from all Binance-specific files across Days 2-56.
"""

import asyncio
import time
import ccxt.async_support as ccxt
from typing import Dict, List, Optional, Any
from .base import BaseExchange


class BinanceAdapter(BaseExchange):
    """
    Binance exchange adapter.

    Consolidates Binance functionality from:
    - Day_2_Projects (all binance_*.py files)
    - Day_4_Projects (binance_algo_orders.py, binance_bot.py)
    - Day_5-56_Projects (all binance_*.py variants)
    """

    def __init__(self, credentials: Dict, event_bus: Any):
        """Initialize Binance adapter."""
        super().__init__(credentials, event_bus)

        self.api_key = credentials.get("api_key")
        self.api_secret = credentials.get("api_secret")
        self.testnet = credentials.get("testnet", True)

        # API endpoints
        if self.testnet:
            self.rest_url = "https://testnet.binancefuture.com"
            self.ws_url = "wss://stream.binancefuture.com/ws/"
        else:
            self.rest_url = "https://fapi.binance.com"
            self.ws_url = "wss://fstream.binance.com/ws/"

        # CCXT client for advanced features
        self.ccxt_client = None

        # Rate limiting
        self.rate_limiter = RateLimiter(
            max_requests=credentials.get("rate_limit", 1200),
            time_window=60  # per minute
        )

    async def initialize(self):
        """Initialize Binance connection."""
        try:
            self.logger.info("Initializing Binance connection...")

            # Initialize CCXT client
            self.ccxt_client = ccxt.binance({
                'apiKey': self.api_key,
                'secret': self.api_secret,
                'enableRateLimit': True,
                'options': {
                    'defaultType': 'future',  # Use futures by default
                    'adjustForTimeDifference': True
                }
            })

            if self.testnet:
                self.ccxt_client.set_sandbox_mode(True)

            # Test connection
            await self._test_connection()

            # Load exchange info
            await self._load_exchange_info()

            self.is_connected = True
            self.logger.info(f"Connected to Binance ({'testnet' if self.testnet else 'mainnet'})")

            await self.emit_event("exchange_connected", {"exchange": "binance"})
            return True

        except Exception as e:
            self.logger.error(f"Failed to initialize Binance: {e}")
            return False

    async def disconnect(self):
        """Disconnect from Binance."""
        if self.ccxt_client:
            await self.ccxt_client.close()
        self.is_connected = False
        await self.emit_event("exchange_disconnected", {"exchange": "binance"})
        self.logger.info("Disconnected from Binance")

    async def _test_connection(self):
        """Test connection to Binance."""
        try:
            # Test public endpoint
            ticker = await self.ccxt_client.fetch_ticker("BTC/USDT")
            self.logger.debug(f"Connection test successful. BTC price: {ticker['last']}")

            # Test authenticated endpoint if credentials provided
            if self.api_key and self.api_secret:
                balance = await self.ccxt_client.fetch_balance()
                self.logger.debug(f"Authentication test successful. Balance: {balance['USDT']['total']}")

        except Exception as e:
            raise Exception(f"Connection test failed: {e}")

    async def _load_exchange_info(self):
        """Load exchange information."""
        try:
            markets = await self.ccxt_client.load_markets()

            # Store symbol information
            for symbol, market in markets.items():
                if market['quote'] == 'USDT':
                    self.symbols_info[symbol] = {
                        'min_amount': market['limits']['amount']['min'],
                        'min_cost': market['limits']['cost']['min'],
                        'price_precision': market['precision']['price'],
                        'amount_precision': market['precision']['amount'],
                        'base': market['base'],
                        'quote': market['quote']
                    }

            self.logger.info(f"Loaded info for {len(self.symbols_info)} symbols")

        except Exception as e:
            self.logger.error(f"Failed to load exchange info: {e}")

    # Order Management (from Day_4, Day_25, Day_26, etc.)

    async def place_order(self, symbol: str, side: str, amount: float,
                         order_type: str = "market", price: Optional[float] = None,
                         **kwargs) -> Dict:
        """Place an order on Binance."""
        try:
            await self.rate_limiter.acquire()

            # Format symbol for Binance
            if not symbol.endswith("/USDT"):
                symbol = f"{symbol}/USDT"

            # Prepare order parameters
            params = {}

            # Add stop loss / take profit (from Day_4_Projects)
            if "stop_loss" in kwargs:
                params['stopLoss'] = kwargs['stop_loss']
            if "take_profit" in kwargs:
                params['takeProfit'] = kwargs['take_profit']

            # Handle reduce only (from Day_25_Projects)
            if kwargs.get("reduce_only", False):
                params['reduceOnly'] = True

            # Place order using CCXT
            if order_type.lower() == "market":
                order = await self.ccxt_client.create_market_order(
                    symbol, side.lower(), amount, params=params
                )
            elif order_type.lower() == "limit":
                if price is None:
                    raise ValueError("Price required for limit orders")
                order = await self.ccxt_client.create_limit_order(
                    symbol, side.lower(), amount, price, params=params
                )
            elif order_type.lower() == "stop":
                if price is None:
                    raise ValueError("Stop price required for stop orders")
                params['stopPrice'] = price
                order = await self.ccxt_client.create_order(
                    symbol, 'stop', side.lower(), amount, price, params=params
                )
            else:
                raise ValueError(f"Unknown order type: {order_type}")

            # Format response
            order_info = {
                "id": order['id'],
                "symbol": symbol,
                "side": side,
                "amount": amount,
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

            if symbol and not symbol.endswith("/USDT"):
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

            if symbol and not symbol.endswith("/USDT"):
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

            if symbol and not symbol.endswith("/USDT"):
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

            if not symbol.endswith("/USDT"):
                symbol = f"{symbol}/USDT"

            positions = await self.ccxt_client.fetch_positions([symbol])

            if positions:
                pos = positions[0]
                return {
                    "symbol": pos['symbol'],
                    "amount": pos['contracts'],
                    "side": pos['side'],
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

            positions = await self.ccxt_client.fetch_positions()

            return [
                {
                    "symbol": pos['symbol'],
                    "amount": pos['contracts'],
                    "side": pos['side'],
                    "entry_price": pos['markPrice'],
                    "mark_price": pos['markPrice'],
                    "pnl": pos['unrealizedPnl'],
                    "pnl_percent": pos['percentage'],
                    "margin": pos['initialMargin']
                }
                for pos in positions if pos['contracts'] != 0
            ]

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
                    "total": balance[asset]['total'],
                    "free": balance[asset]['free'],
                    "used": balance[asset]['used']
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

            if not symbol.endswith("/USDT"):
                symbol = f"{symbol}/USDT"

            ohlcv = await self.ccxt_client.fetch_ohlcv(symbol, timeframe, limit=limit)
            return ohlcv

        except Exception as e:
            self.logger.error(f"Failed to get OHLCV: {e}")
            return []

    async def get_ticker(self, symbol: str) -> Dict:
        """Get ticker information."""
        try:
            await self.rate_limiter.acquire()

            if not symbol.endswith("/USDT"):
                symbol = f"{symbol}/USDT"

            ticker = await self.ccxt_client.fetch_ticker(symbol)

            return {
                "symbol": ticker['symbol'],
                "bid": ticker['bid'],
                "ask": ticker['ask'],
                "last": ticker['last'],
                "volume": ticker['volume'],
                "change_24h": ticker['percentage']
            }

        except Exception as e:
            self.logger.error(f"Failed to get ticker: {e}")
            return {}

    async def get_orderbook(self, symbol: str, limit: int = 20) -> Dict:
        """Get orderbook."""
        try:
            await self.rate_limiter.acquire()

            if not symbol.endswith("/USDT"):
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

    # Binance-specific methods (from Day projects)

    async def get_funding_rate(self, symbol: str) -> float:
        """Get funding rate for perpetual contracts."""
        try:
            await self.rate_limiter.acquire()

            if not symbol.endswith("/USDT"):
                symbol = f"{symbol}/USDT"

            # Get funding rate using CCXT
            funding = await self.ccxt_client.fetch_funding_rate(symbol)
            return funding['fundingRate']

        except Exception as e:
            self.logger.error(f"Failed to get funding rate: {e}")
            return 0.0

    async def set_leverage(self, symbol: str, leverage: int) -> bool:
        """Set leverage for a symbol."""
        try:
            await self.rate_limiter.acquire()

            if not symbol.endswith("/USDT"):
                symbol = f"{symbol}/USDT"

            result = await self.ccxt_client.set_leverage(leverage, symbol)
            self.logger.info(f"Set leverage to {leverage}x for {symbol}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to set leverage: {e}")
            return False

    async def get_liquidations(self, symbol: Optional[str] = None) -> List[Dict]:
        """Get recent liquidations (via REST API)."""
        try:
            await self.rate_limiter.acquire()

            # Binance doesn't provide historical liquidations via REST
            # This would typically come from WebSocket stream
            # Return empty for REST implementation
            return []

        except Exception as e:
            self.logger.error(f"Failed to get liquidations: {e}")
            return []

    async def get_trades(self, symbol: str, limit: int = 100) -> List[Dict]:
        """Get recent trades."""
        try:
            await self.rate_limiter.acquire()

            if not symbol.endswith("/USDT"):
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

    async def bracket_order(self, symbol: str, side: str, amount: float,
                          stop_loss: float, take_profit: float) -> Dict:
        """
        Place a bracket order (entry + stop loss + take profit).
        From Day_4_Projects/binance_algo_orders.py
        """
        try:
            # Place main order
            main_order = await self.place_order(
                symbol=symbol,
                side=side,
                amount=amount,
                order_type="market"
            )

            if main_order:
                # Place stop loss
                sl_side = "sell" if side == "buy" else "buy"
                sl_order = await self.place_order(
                    symbol=symbol,
                    side=sl_side,
                    amount=amount,
                    order_type="stop",
                    price=stop_loss
                )

                # Place take profit
                tp_order = await self.place_order(
                    symbol=symbol,
                    side=sl_side,
                    amount=amount,
                    order_type="limit",
                    price=take_profit
                )

                return {
                    "main_order": main_order,
                    "stop_loss": sl_order,
                    "take_profit": tp_order
                }

            return {}

        except Exception as e:
            self.logger.error(f"Failed to place bracket order: {e}")
            return {}

    async def trailing_stop(self, symbol: str, side: str, amount: float,
                          callback_rate: float) -> Dict:
        """
        Place a trailing stop order.
        From Day_4_Projects
        """
        try:
            params = {
                'callbackRate': callback_rate,  # Percentage for trailing
                'type': 'TRAILING_STOP_MARKET'
            }

            order = await self.ccxt_client.create_order(
                symbol=f"{symbol}/USDT",
                type='trailing_stop_market',
                side=side.lower(),
                amount=amount,
                price=None,
                params=params
            )

            return {
                "id": order['id'],
                "symbol": order['symbol'],
                "side": side,
                "amount": amount,
                "callback_rate": callback_rate,
                "status": order['status']
            }

        except Exception as e:
            self.logger.error(f"Failed to place trailing stop: {e}")
            return {}

    # Helper methods from nice_funcs.py files

    async def get_24h_ticker_stats(self, symbol: str) -> Dict:
        """Get 24h ticker statistics."""
        try:
            ticker = await self.get_ticker(symbol)
            return {
                "high_24h": ticker.get("high"),
                "low_24h": ticker.get("low"),
                "volume_24h": ticker.get("volume"),
                "change_24h": ticker.get("change_24h")
            }
        except Exception as e:
            self.logger.error(f"Failed to get 24h stats: {e}")
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