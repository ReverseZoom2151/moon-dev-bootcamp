"""
Exchange Utilities Module
=========================
Contains exchange-specific utilities for trading operations.
Includes position sizing, PnL calculations, risk management, and order management.
"""

import numpy as np
from typing import Dict, List


class ExchangeUtils:
    """Utilities for exchange-specific trading operations."""

    # ===========================================
    # POSITION & RISK MANAGEMENT
    # ===========================================

    @staticmethod
    def calculate_position_size(balance: float, risk_percent: float = 1.0,
                              stop_loss_percent: float = 2.0,
                              leverage: int = 1) -> float:
        """
        Calculate position size based on risk management.

        Args:
            balance: Account balance
            risk_percent: Risk per trade (%)
            stop_loss_percent: Stop loss distance (%)
            leverage: Leverage to use

        Returns:
            Position size
        """
        risk_amount = balance * (risk_percent / 100)
        position_size = (risk_amount / (stop_loss_percent / 100)) * leverage
        return round(position_size, 2)

    @staticmethod
    def calculate_stop_loss(entry_price: float, side: str,
                          atr_value: float, multiplier: float = 2.0) -> float:
        """
        Calculate stop loss price using ATR.

        Args:
            entry_price: Entry price
            side: 'buy' or 'sell'
            atr_value: ATR value
            multiplier: ATR multiplier

        Returns:
            Stop loss price
        """
        if side.lower() == 'buy':
            return entry_price - (atr_value * multiplier)
        else:
            return entry_price + (atr_value * multiplier)

    @staticmethod
    def calculate_take_profit(entry_price: float, side: str,
                            risk_reward_ratio: float = 2.0,
                            stop_loss_price: float = None) -> float:
        """
        Calculate take profit price.

        Args:
            entry_price: Entry price
            side: 'buy' or 'sell'
            risk_reward_ratio: Risk/reward ratio
            stop_loss_price: Stop loss price

        Returns:
            Take profit price
        """
        if stop_loss_price:
            risk = abs(entry_price - stop_loss_price)
            if side.lower() == 'buy':
                return entry_price + (risk * risk_reward_ratio)
            else:
                return entry_price - (risk * risk_reward_ratio)
        else:
            # Default 5% take profit
            if side.lower() == 'buy':
                return entry_price * 1.05
            else:
                return entry_price * 0.95

    @staticmethod
    def calculate_pnl(entry_price: float, exit_price: float, amount: float,
                     side: str, leverage: int = 1, fees: float = 0.001) -> Dict:
        """
        Calculate PnL for a trade.

        Args:
            entry_price: Entry price
            exit_price: Exit price
            amount: Trade amount
            side: 'buy' or 'sell'
            leverage: Leverage used
            fees: Fee percentage

        Returns:
            Dictionary with PnL details
        """
        if side.lower() == 'buy':
            gross_pnl = (exit_price - entry_price) * amount
        else:
            gross_pnl = (entry_price - exit_price) * amount

        gross_pnl *= leverage

        # Calculate fees
        entry_fee = entry_price * amount * fees
        exit_fee = exit_price * amount * fees
        total_fees = entry_fee + exit_fee

        net_pnl = gross_pnl - total_fees
        pnl_percent = (net_pnl / (entry_price * amount)) * 100

        return {
            "gross_pnl": round(gross_pnl, 2),
            "fees": round(total_fees, 2),
            "net_pnl": round(net_pnl, 2),
            "pnl_percent": round(pnl_percent, 2)
        }

    # ===========================================
    # LIQUIDATION ANALYSIS
    # ===========================================

    @staticmethod
    def calculate_liquidation_levels(price: float, leverage: int,
                                   side: str = "long") -> Dict[str, float]:
        """
        Calculate liquidation levels.

        Args:
            price: Current price
            leverage: Leverage used
            side: 'long' or 'short'

        Returns:
            Dictionary with liquidation levels
        """
        maintenance_margin = 0.5  # 0.5% for most exchanges

        if side == "long":
            liquidation_price = price * (1 - (1 / leverage) + (maintenance_margin / 100))
        else:
            liquidation_price = price * (1 + (1 / leverage) - (maintenance_margin / 100))

        return {
            "liquidation_price": round(liquidation_price, 2),
            "distance_percent": round(abs(price - liquidation_price) / price * 100, 2),
            "leverage": leverage,
            "side": side
        }

    @staticmethod
    def find_liquidation_clusters(orderbook: Dict, price: float,
                                threshold_percent: float = 5.0) -> List[Dict]:
        """
        Find potential liquidation clusters in orderbook.

        Args:
            orderbook: Orderbook data
            price: Current price
            threshold_percent: Distance threshold

        Returns:
            List of liquidation clusters
        """
        clusters = []
        common_leverages = [3, 5, 10, 20, 25, 50, 75, 100]

        for leverage in common_leverages:
            # Calculate long liquidation level
            long_liq = price * (1 - (1 / leverage) + 0.005)
            # Calculate short liquidation level
            short_liq = price * (1 + (1 / leverage) - 0.005)

            # Check if significant orders exist near these levels
            for level, side in [(long_liq, "long"), (short_liq, "short")]:
                distance_percent = abs(price - level) / price * 100

                if distance_percent <= threshold_percent:
                    clusters.append({
                        "level": round(level, 2),
                        "leverage": leverage,
                        "side": side,
                        "distance_percent": round(distance_percent, 2)
                    })

        return sorted(clusters, key=lambda x: x["distance_percent"])

    # ===========================================
    # MARKET MAKING
    # ===========================================

    @staticmethod
    def calculate_spread(ask: float, bid: float) -> float:
        """Calculate bid-ask spread percentage."""
        if bid == 0:
            return 0
        return ((ask - bid) / bid) * 100

    @staticmethod
    def calculate_fair_value(orderbook: Dict, depth: int = 5) -> float:
        """
        Calculate fair value from orderbook.

        Args:
            orderbook: Orderbook data
            depth: Depth to consider

        Returns:
            Fair value price
        """
        bids = orderbook.get("bids", [])[:depth]
        asks = orderbook.get("asks", [])[:depth]

        if not bids or not asks:
            return 0

        # Volume-weighted average
        bid_value = sum(float(b[0]) * float(b[1]) for b in bids)
        bid_volume = sum(float(b[1]) for b in bids)

        ask_value = sum(float(a[0]) * float(a[1]) for a in asks)
        ask_volume = sum(float(a[1]) for a in asks)

        total_value = bid_value + ask_value
        total_volume = bid_volume + ask_volume

        if total_volume == 0:
            return 0

        return total_value / total_volume

    @staticmethod
    def calculate_order_imbalance(orderbook: Dict, depth: int = 5) -> float:
        """
        Calculate order book imbalance.

        Returns:
            Imbalance ratio (positive = more buying pressure)
        """
        bids = orderbook.get("bids", [])[:depth]
        asks = orderbook.get("asks", [])[:depth]

        bid_volume = sum(float(b[1]) for b in bids)
        ask_volume = sum(float(a[1]) for a in asks)

        total_volume = bid_volume + ask_volume
        if total_volume == 0:
            return 0

        return (bid_volume - ask_volume) / total_volume

    # ===========================================
    # SMART ORDER ROUTING
    # ===========================================

    @staticmethod
    def calculate_order_slices(total_amount: float, max_slice_size: float = None,
                             num_slices: int = None) -> List[float]:
        """
        Calculate order slices for large orders.

        Args:
            total_amount: Total order amount
            max_slice_size: Maximum size per slice
            num_slices: Number of slices (alternative to max_slice_size)

        Returns:
            List of slice amounts
        """
        if num_slices:
            base_slice = total_amount / num_slices
            # Add some randomness to avoid detection
            slices = []
            remaining = total_amount
            for i in range(num_slices - 1):
                variation = base_slice * 0.2  # 20% variation
                slice_size = base_slice + np.random.uniform(-variation, variation)
                slice_size = min(slice_size, remaining)
                slices.append(round(slice_size, 8))
                remaining -= slice_size
            slices.append(round(remaining, 8))  # Last slice gets remainder
            return slices

        elif max_slice_size:
            slices = []
            remaining = total_amount
            while remaining > 0:
                slice_size = min(max_slice_size, remaining)
                slices.append(round(slice_size, 8))
                remaining -= slice_size
            return slices

        else:
            return [total_amount]


# Create singleton instance
exchange_utils = ExchangeUtils()
