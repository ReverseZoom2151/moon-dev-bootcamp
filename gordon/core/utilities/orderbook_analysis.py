"""
Advanced Order Book Utilities
==============================
Day 45: Enhanced order book analysis utilities.

Features:
- Whale order detection
- Bid/ask spread analysis
- Order book depth analysis
- Market impact estimation
"""

import logging
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class OrderBookAnalyzer:
    """
    Advanced order book analysis utilities.
    
    Provides functions for analyzing order book depth,
    detecting whale orders, and estimating market impact.
    """

    def __init__(self, whale_threshold_usd: float = 50000):
        """
        Initialize order book analyzer.
        
        Args:
            whale_threshold_usd: Minimum USD value to consider an order a "whale" order
        """
        self.whale_threshold_usd = whale_threshold_usd

    def analyze_whale_orders(
        self,
        bids: List[List[float]],
        asks: List[List[float]],
        current_price: float
    ) -> Dict[str, any]:
        """
        Analyze whale orders in the order book.
        
        Args:
            bids: List of [price, quantity] pairs for bids
            asks: List of [price, quantity] pairs for asks
            current_price: Current market price
            
        Returns:
            Dictionary with bias, strength, and whale order details
        """
        try:
            if not bids or not asks:
                return {'bias': 'neutral', 'strength': 0, 'whale_bids_value': 0, 'whale_asks_value': 0}
            
            whale_bids_value = 0
            whale_asks_value = 0
            whale_bids_count = 0
            whale_asks_count = 0
            
            # Analyze bids (buy orders)
            for bid in bids:
                if len(bid) >= 2:
                    price, quantity = float(bid[0]), float(bid[1])
                    value = price * quantity
                    if value >= self.whale_threshold_usd:
                        whale_bids_value += value
                        whale_bids_count += 1
            
            # Analyze asks (sell orders)
            for ask in asks:
                if len(ask) >= 2:
                    price, quantity = float(ask[0]), float(ask[1])
                    value = price * quantity
                    if value >= self.whale_threshold_usd:
                        whale_asks_value += value
                        whale_asks_count += 1
            
            # Determine bias
            total_whale_value = whale_bids_value + whale_asks_value
            if total_whale_value > 0:
                bid_percentage = whale_bids_value / total_whale_value
                
                if bid_percentage > 0.6:
                    bias = 'bullish'
                    strength = bid_percentage
                elif bid_percentage < 0.4:
                    bias = 'bearish'
                    strength = 1 - bid_percentage
                else:
                    bias = 'neutral'
                    strength = 0.5
            else:
                bias = 'neutral'
                strength = 0
            
            return {
                'bias': bias,
                'strength': float(strength),
                'whale_bids_value': whale_bids_value,
                'whale_asks_value': whale_asks_value,
                'whale_bids_count': whale_bids_count,
                'whale_asks_count': whale_asks_count,
                'total_whale_value': total_whale_value
            }
            
        except Exception as e:
            logger.error(f"Error analyzing whale orders: {e}")
            return {'bias': 'neutral', 'strength': 0, 'whale_bids_value': 0, 'whale_asks_value': 0}

    def calculate_spread(
        self,
        bids: List[List[float]],
        asks: List[List[float]]
    ) -> Dict[str, float]:
        """
        Calculate bid/ask spread metrics.
        
        Args:
            bids: List of [price, quantity] pairs for bids
            asks: List of [price, quantity] pairs for asks
            
        Returns:
            Dictionary with spread metrics
        """
        try:
            if not bids or not asks:
                return {'spread': 0, 'spread_percent': 0, 'best_bid': 0, 'best_ask': 0}
            
            best_bid = float(bids[0][0]) if len(bids[0]) > 0 else 0
            best_ask = float(asks[0][0]) if len(asks[0]) > 0 else 0
            
            if best_bid == 0 or best_ask == 0:
                return {'spread': 0, 'spread_percent': 0, 'best_bid': best_bid, 'best_ask': best_ask}
            
            spread = best_ask - best_bid
            mid_price = (best_bid + best_ask) / 2
            spread_percent = (spread / mid_price) * 100 if mid_price > 0 else 0
            
            return {
                'spread': spread,
                'spread_percent': spread_percent,
                'best_bid': best_bid,
                'best_ask': best_ask,
                'mid_price': mid_price
            }
            
        except Exception as e:
            logger.error(f"Error calculating spread: {e}")
            return {'spread': 0, 'spread_percent': 0, 'best_bid': 0, 'best_ask': 0}

    def calculate_order_book_depth(
        self,
        bids: List[List[float]],
        asks: List[List[float]],
        depth_levels: int = 10
    ) -> Dict[str, float]:
        """
        Calculate order book depth metrics.
        
        Args:
            bids: List of [price, quantity] pairs for bids
            asks: List of [price, quantity] pairs for asks
            depth_levels: Number of levels to analyze
            
        Returns:
            Dictionary with depth metrics
        """
        try:
            if not bids or not asks:
                return {'bid_depth': 0, 'ask_depth': 0, 'depth_imbalance': 0}
            
            # Calculate cumulative depth
            bid_depth = 0
            ask_depth = 0
            
            for i, bid in enumerate(bids[:depth_levels]):
                if len(bid) >= 2:
                    price, quantity = float(bid[0]), float(bid[1])
                    bid_depth += price * quantity
            
            for i, ask in enumerate(asks[:depth_levels]):
                if len(ask) >= 2:
                    price, quantity = float(ask[0]), float(ask[1])
                    ask_depth += price * quantity
            
            total_depth = bid_depth + ask_depth
            depth_imbalance = (bid_depth - ask_depth) / total_depth if total_depth > 0 else 0
            
            return {
                'bid_depth': bid_depth,
                'ask_depth': ask_depth,
                'total_depth': total_depth,
                'depth_imbalance': depth_imbalance,
                'bid_ask_ratio': bid_depth / ask_depth if ask_depth > 0 else 0
            }
            
        except Exception as e:
            logger.error(f"Error calculating order book depth: {e}")
            return {'bid_depth': 0, 'ask_depth': 0, 'depth_imbalance': 0}

    def estimate_market_impact(
        self,
        bids: List[List[float]],
        asks: List[List[float]],
        order_size_usd: float,
        side: str = 'BUY'
    ) -> Dict[str, float]:
        """
        Estimate market impact of a trade.
        
        Args:
            bids: List of [price, quantity] pairs for bids
            asks: List of [price, quantity] pairs for asks
            order_size_usd: Size of order in USD
            side: 'BUY' or 'SELL'
            
        Returns:
            Dictionary with impact estimates
        """
        try:
            if not bids or not asks:
                return {'impact_price': 0, 'slippage_percent': 0, 'levels_consumed': 0}
            
            if side == 'BUY':
                levels = asks
                best_price = float(asks[0][0]) if len(asks[0]) > 0 else 0
            else:
                levels = bids
                best_price = float(bids[0][0]) if len(bids[0]) > 0 else 0
            
            if best_price == 0:
                return {'impact_price': 0, 'slippage_percent': 0, 'levels_consumed': 0}
            
            remaining_size = order_size_usd
            levels_consumed = 0
            total_cost = 0
            
            for level in levels:
                if len(level) < 2:
                    continue
                
                price, quantity = float(level[0]), float(level[1])
                level_value = price * quantity
                
                if remaining_size <= level_value:
                    total_cost += remaining_size
                    levels_consumed += 1
                    break
                else:
                    total_cost += level_value
                    remaining_size -= level_value
                    levels_consumed += 1
            
            if remaining_size > 0:
                # Couldn't fill entire order with available liquidity
                avg_price = best_price * 1.1 if side == 'BUY' else best_price * 0.9  # Estimate
            else:
                avg_price = total_cost / order_size_usd if order_size_usd > 0 else best_price
            
            slippage = abs(avg_price - best_price) / best_price * 100 if best_price > 0 else 0
            
            return {
                'impact_price': avg_price,
                'slippage_percent': slippage,
                'levels_consumed': levels_consumed,
                'best_price': best_price,
                'fill_percentage': 1.0 if remaining_size == 0 else (order_size_usd - remaining_size) / order_size_usd
            }
            
        except Exception as e:
            logger.error(f"Error estimating market impact: {e}")
            return {'impact_price': 0, 'slippage_percent': 0, 'levels_consumed': 0}

    def analyze_order_book(
        self,
        bids: List[List[float]],
        asks: List[List[float]],
        current_price: float
    ) -> Dict[str, any]:
        """
        Comprehensive order book analysis.
        
        Args:
            bids: List of [price, quantity] pairs for bids
            asks: List of [price, quantity] pairs for asks
            current_price: Current market price
            
        Returns:
            Dictionary with all analysis metrics
        """
        whale_analysis = self.analyze_whale_orders(bids, asks, current_price)
        spread_metrics = self.calculate_spread(bids, asks)
        depth_metrics = self.calculate_order_book_depth(bids, asks)
        
        return {
            'whale_analysis': whale_analysis,
            'spread_metrics': spread_metrics,
            'depth_metrics': depth_metrics,
            'current_price': current_price
        }

