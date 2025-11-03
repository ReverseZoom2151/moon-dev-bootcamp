"""
HyperLiquid Exchange Adapter
============================
Implementation for HyperLiquid exchange integration.
Consolidates functionality from multiple Day projects.
"""

import asyncio
import json
import time
from typing import Dict, List, Optional, Any
from datetime import datetime
try:
    from hyperliquid.info import Info
    from hyperliquid.exchange import Exchange as HyperLiquidExchange
    from hyperliquid.utils import constants
    HYPERLIQUID_AVAILABLE = True
except ImportError:
    HYPERLIQUID_AVAILABLE = False
    Info = None
    HyperLiquidExchange = None
    constants = None

try:
    from eth_account import Account
except ImportError:
    Account = None

from .base import BaseExchange


class HyperLiquidAdapter(BaseExchange):
    """
    HyperLiquid exchange adapter.

    Consolidates HyperLiquid functionality from:
    - Day_4_Projects (test_hyperliquid.py)
    - Day_5_Projects (5_risk_mgmt_hl.py)
    - Day_31_Projects (hl_data.py)
    - Day_37_Projects (RRS for hyperliquid/)
    - Day_43_Projects (various order types)
    - Day_44_Projects (ppls_positions.py)
    """

    def __init__(self, credentials: Dict, event_bus: Any):
        """Initialize HyperLiquid adapter."""
        super().__init__(credentials, event_bus)

        if not HYPERLIQUID_AVAILABLE:
            self.logger.warning("HyperLiquid package not installed. Install with: pip install hyperliquid-python")
            self.private_key = None
            self.account_address = None
            self.info = None
            self.exchange = None
            self.agent_private_key = None
            self.is_mainnet = True
            return

        self.private_key = credentials.get("private_key")
        self.account_address = None
        self.info = None
        self.exchange = None
        self.agent_private_key = credentials.get("agent_private_key")
        self.is_mainnet = credentials.get("is_mainnet", True)

    async def initialize(self):
        """Initialize HyperLiquid connection."""
        if not HYPERLIQUID_AVAILABLE:
            self.logger.warning("Cannot initialize HyperLiquid - package not installed")
            return False

        try:
            self.logger.info("Initializing HyperLiquid connection...")

            # Create account from private key
            if self.private_key:
                account = Account.from_key(self.private_key)
                self.account_address = account.address

                # Initialize Info API (public data)
                self.info = Info(
                    base_url=constants.MAINNET_API_URL if self.is_mainnet else constants.TESTNET_API_URL,
                    skip_ws=True
                )

                # Initialize Exchange API (trading)
                if self.agent_private_key:
                    # Use agent for trading
                    self.exchange = HyperLiquidExchange(
                        Account.from_key(self.agent_private_key),
                        base_url=constants.MAINNET_API_URL if self.is_mainnet else constants.TESTNET_API_URL,
                        account_address=self.account_address
                    )
                else:
                    # Direct trading
                    self.exchange = HyperLiquidExchange(
                        account,
                        base_url=constants.MAINNET_API_URL if self.is_mainnet else constants.TESTNET_API_URL
                    )

                self.is_connected = True
                self.logger.info(f"Connected to HyperLiquid ({'mainnet' if self.is_mainnet else 'testnet'})")

                # Load market info
                await self._load_market_info()

                await self.emit_event("exchange_connected", {"exchange": "hyperliquid"})
                return True

            else:
                self.logger.error("No private key provided")
                return False

        except Exception as e:
            self.logger.error(f"Failed to initialize HyperLiquid: {e}")
            return False

    async def disconnect(self):
        """Disconnect from HyperLiquid."""
        self.is_connected = False
        await self.emit_event("exchange_disconnected", {"exchange": "hyperliquid"})
        self.logger.info("Disconnected from HyperLiquid")

    async def _load_market_info(self):
        """Load market information."""
        try:
            meta = self.info.meta()
            self.symbols_info = {
                asset["name"]: {
                    "min_size": float(asset["szDecimals"]),
                    "price_decimals": asset.get("priceDecimals", 2),
                    "asset_id": asset["assetId"]
                }
                for asset in meta["universe"]
            }
            self.logger.info(f"Loaded info for {len(self.symbols_info)} symbols")
        except Exception as e:
            self.logger.error(f"Failed to load market info: {e}")

    async def get_balance(self, asset: Optional[str] = None) -> Dict:
        """Get account balance."""
        try:
            user_state = self.info.user_state(self.account_address)

            balances = {}
            if "marginSummary" in user_state:
                margin = user_state["marginSummary"]
                balances["USDC"] = {
                    "total": float(margin.get("accountValue", 0)),
                    "free": float(margin.get("availableMargin", 0)),
                    "used": float(margin.get("totalMarginUsed", 0))
                }

            if asset:
                return balances.get(asset, {"total": 0, "free": 0, "used": 0})

            return balances

        except Exception as e:
            self.logger.error(f"Failed to get balance: {e}")
            return {}

    async def place_order(self, symbol: str, side: str, amount: float,
                         order_type: str = "market", price: Optional[float] = None,
                         **kwargs) -> Dict:
        """Place an order on HyperLiquid."""
        try:
            # Format symbol (e.g., BTC -> BTC-USD)
            if not symbol.endswith("-USD"):
                symbol = f"{symbol}-USD"

            # Get current price if needed
            if order_type == "market" and not price:
                ask, bid = await self.get_ask_bid(symbol)
                price = ask if side.lower() == "buy" else bid

            # Build order request
            order_request = {
                "coin": symbol.replace("-USD", ""),
                "is_buy": side.lower() == "buy",
                "sz": amount,
                "limit_px": price,
                "order_type": {"limit": {"tif": "Ioc"}} if order_type == "market" else {"limit": {"tif": "Gtc"}},
                "reduce_only": kwargs.get("reduce_only", False)
            }

            # Add stop loss / take profit if provided
            if "stop_loss" in kwargs:
                order_request["stop_loss"] = kwargs["stop_loss"]
            if "take_profit" in kwargs:
                order_request["take_profit"] = kwargs["take_profit"]

            # Place the order
            result = self.exchange.order(
                order_request["coin"],
                order_request["is_buy"],
                order_request["sz"],
                order_request["limit_px"],
                order_request["order_type"],
                order_request.get("reduce_only", False)
            )

            if result and result.get("status") == "ok":
                order_data = result["response"]["data"]
                order_info = {
                    "id": order_data.get("oid"),
                    "symbol": symbol,
                    "side": side,
                    "amount": amount,
                    "price": price,
                    "type": order_type,
                    "status": "open",
                    "timestamp": datetime.now().isoformat()
                }

                await self.emit_event("order_placed", order_info)
                return order_info

            else:
                self.logger.error(f"Order failed: {result}")
                return {}

        except Exception as e:
            self.logger.error(f"Failed to place order: {e}")
            return {}

    async def cancel_order(self, order_id: str, symbol: Optional[str] = None) -> bool:
        """Cancel an order."""
        try:
            if symbol and not symbol.endswith("-USD"):
                symbol = f"{symbol}-USD"

            coin = symbol.replace("-USD", "") if symbol else None
            result = self.exchange.cancel(coin, order_id)

            if result and result.get("status") == "ok":
                await self.emit_event("order_cancelled", {"order_id": order_id})
                return True

            return False

        except Exception as e:
            self.logger.error(f"Failed to cancel order: {e}")
            return False

    async def get_order(self, order_id: str, symbol: Optional[str] = None) -> Dict:
        """Get order information."""
        try:
            orders = await self.get_open_orders(symbol)
            for order in orders:
                if order["id"] == order_id:
                    return order
            return {}
        except Exception as e:
            self.logger.error(f"Failed to get order: {e}")
            return {}

    async def get_open_orders(self, symbol: Optional[str] = None) -> List[Dict]:
        """Get open orders."""
        try:
            user_state = self.info.user_state(self.account_address)
            open_orders = user_state.get("openOrders", [])

            orders = []
            for order in open_orders:
                order_info = {
                    "id": order["oid"],
                    "symbol": f"{order['coin']}-USD",
                    "side": "buy" if order["side"] == "B" else "sell",
                    "amount": float(order["sz"]),
                    "price": float(order["limitPx"]),
                    "filled": float(order.get("filledSz", 0)),
                    "timestamp": order["timestamp"]
                }

                if not symbol or order_info["symbol"] == symbol:
                    orders.append(order_info)

            return orders

        except Exception as e:
            self.logger.error(f"Failed to get open orders: {e}")
            return []

    async def get_position(self, symbol: str) -> Dict:
        """Get position for a symbol."""
        try:
            if not symbol.endswith("-USD"):
                symbol = f"{symbol}-USD"

            positions = await self.get_all_positions()
            for pos in positions:
                if pos["symbol"] == symbol:
                    return pos

            return {}

        except Exception as e:
            self.logger.error(f"Failed to get position: {e}")
            return {}

    async def get_all_positions(self) -> List[Dict]:
        """Get all open positions."""
        try:
            user_state = self.info.user_state(self.account_address)
            asset_positions = user_state.get("assetPositions", [])

            positions = []
            for pos in asset_positions:
                if float(pos["position"]["szi"]) != 0:
                    position_info = {
                        "symbol": f"{pos['position']['coin']}-USD",
                        "amount": float(pos["position"]["szi"]),
                        "entry_price": float(pos["position"]["entryPx"]),
                        "mark_price": float(pos["position"]["markPx"]),
                        "pnl": float(pos["position"]["pnl"]),
                        "margin": float(pos["position"]["marginUsed"]),
                        "leverage": int(pos["position"].get("leverage", 1))
                    }
                    positions.append(position_info)

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

            close_amount = amount or abs(position["amount"])
            side = "sell" if position["amount"] > 0 else "buy"

            return await self.place_order(
                symbol=symbol,
                side=side,
                amount=close_amount,
                order_type="market",
                reduce_only=True
            )

        except Exception as e:
            self.logger.error(f"Failed to close position: {e}")
            return {}

    async def get_ohlcv(self, symbol: str, timeframe: str = "1h",
                       limit: int = 100) -> List[List]:
        """Get OHLCV candle data."""
        try:
            if not symbol.endswith("-USD"):
                symbol = f"{symbol}-USD"

            coin = symbol.replace("-USD", "")

            # Convert timeframe to minutes
            timeframe_map = {
                "1m": 1, "5m": 5, "15m": 15, "30m": 30,
                "1h": 60, "4h": 240, "1d": 1440
            }
            interval = timeframe_map.get(timeframe, 60)

            # Get candles
            end_time = int(time.time() * 1000)
            start_time = end_time - (interval * 60 * 1000 * limit)

            candles = self.info.candles(coin, interval, start_time, end_time)

            ohlcv = []
            for candle in candles:
                ohlcv.append([
                    candle["t"],  # timestamp
                    float(candle["o"]),  # open
                    float(candle["h"]),  # high
                    float(candle["l"]),  # low
                    float(candle["c"]),  # close
                    float(candle["v"])  # volume
                ])

            return ohlcv

        except Exception as e:
            self.logger.error(f"Failed to get OHLCV: {e}")
            return []

    async def get_ticker(self, symbol: str) -> Dict:
        """Get ticker information."""
        try:
            if not symbol.endswith("-USD"):
                symbol = f"{symbol}-USD"

            coin = symbol.replace("-USD", "")
            l2_data = self.info.l2_book(coin)

            return {
                "symbol": symbol,
                "bid": float(l2_data["bids"][0][0]) if l2_data["bids"] else 0,
                "ask": float(l2_data["asks"][0][0]) if l2_data["asks"] else 0,
                "last": (float(l2_data["bids"][0][0]) + float(l2_data["asks"][0][0])) / 2
                        if l2_data["bids"] and l2_data["asks"] else 0
            }

        except Exception as e:
            self.logger.error(f"Failed to get ticker: {e}")
            return {}

    async def get_orderbook(self, symbol: str, limit: int = 20) -> Dict:
        """Get orderbook."""
        try:
            if not symbol.endswith("-USD"):
                symbol = f"{symbol}-USD"

            coin = symbol.replace("-USD", "")
            l2_data = self.info.l2_book(coin)

            return {
                "bids": [[float(b[0]), float(b[1])] for b in l2_data.get("bids", [])[:limit]],
                "asks": [[float(a[0]), float(a[1])] for a in l2_data.get("asks", [])[:limit]]
            }

        except Exception as e:
            self.logger.error(f"Failed to get orderbook: {e}")
            return {"bids": [], "asks": []}

    async def get_funding_rate(self, symbol: str) -> float:
        """Get funding rate for perpetual contracts."""
        try:
            if not symbol.endswith("-USD"):
                symbol = f"{symbol}-USD"

            coin = symbol.replace("-USD", "")
            meta_and_asset = self.info.meta_and_asset(coin)

            if meta_and_asset and "funding" in meta_and_asset:
                return float(meta_and_asset["funding"])

            return 0.0

        except Exception as e:
            self.logger.error(f"Failed to get funding rate: {e}")
            return 0.0

    async def set_leverage(self, symbol: str, leverage: int) -> bool:
        """Set leverage for a symbol."""
        try:
            if not symbol.endswith("-USD"):
                symbol = f"{symbol}-USD"

            coin = symbol.replace("-USD", "")
            result = self.exchange.update_leverage(leverage, coin)

            if result and result.get("status") == "ok":
                self.logger.info(f"Set leverage to {leverage}x for {symbol}")
                return True

            return False

        except Exception as e:
            self.logger.error(f"Failed to set leverage: {e}")
            return False

    async def get_liquidations(self, symbol: Optional[str] = None) -> List[Dict]:
        """Get recent liquidations."""
        try:
            # HyperLiquid-specific liquidation data
            snapshot = self.info.l2_snapshot(symbol.replace("-USD", "") if symbol else "BTC")

            liquidations = []
            if "liquidations" in snapshot:
                for liq in snapshot["liquidations"]:
                    liquidations.append({
                        "symbol": symbol or "BTC-USD",
                        "side": liq.get("side"),
                        "amount": float(liq.get("sz", 0)),
                        "price": float(liq.get("px", 0)),
                        "timestamp": liq.get("time")
                    })

            return liquidations

        except Exception as e:
            self.logger.error(f"Failed to get liquidations: {e}")
            return []

    # HyperLiquid-specific methods

    async def get_user_fills(self, limit: int = 100) -> List[Dict]:
        """Get user's trade fills."""
        try:
            fills = self.info.user_fills(self.account_address)
            return fills[:limit] if fills else []
        except Exception as e:
            self.logger.error(f"Failed to get user fills: {e}")
            return []

    async def get_vault_details(self, vault_address: str) -> Dict:
        """Get vault details (if using vault trading)."""
        try:
            return self.info.vault_details(vault_address)
        except Exception as e:
            self.logger.error(f"Failed to get vault details: {e}")
            return {}

    async def twap_order(self, symbol: str, side: str, total_amount: float,
                        duration_minutes: int, num_slices: int) -> List[Dict]:
        """
        Execute TWAP (Time-Weighted Average Price) order.

        Args:
            symbol: Trading symbol
            side: Buy or sell
            total_amount: Total amount to trade
            duration_minutes: Duration over which to execute
            num_slices: Number of order slices

        Returns:
            List of placed orders
        """
        slice_amount = total_amount / num_slices
        interval_seconds = (duration_minutes * 60) / num_slices
        orders = []

        for i in range(num_slices):
            order = await self.place_order(symbol, side, slice_amount, "market")
            if order:
                orders.append(order)
            await asyncio.sleep(interval_seconds)

        return orders