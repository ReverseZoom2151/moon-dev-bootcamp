"""
Enhanced Market Making Strategy
================================
Day 26 (Base) + Day 45 (Enhanced): Advanced market making strategy with order book analysis.

Strategy Logic:
- Buy when price < BUY_UNDER threshold
- Sell when price > SELL_OVER threshold
- Maintains position within target size
- Uses order book depth to adjust spread dynamically
- Considers whale orders for better execution
"""

import logging
from typing import Dict, Optional, List
from datetime import datetime

from .base import BaseStrategy
from ..utilities.orderbook_analysis import OrderBookAnalyzer

logger = logging.getLogger(__name__)


class MarketMakerStrategy(BaseStrategy):
    """
    Simple market making strategy.
    
    Buys when price is below threshold, sells when price is above threshold.
    """

    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize enhanced market maker strategy.
        
        Args:
            config: Configuration dictionary
        """
        super().__init__("EnhancedMarketMaker")
        self.config = config or {}
        self.buy_under = self.config.get('buy_under', 0.0004)
        self.sell_over = self.config.get('sell_over', 0.001)
        self.target_size = self.config.get('target_size', 350)  # USD
        self.position_threshold = self.config.get('position_threshold', 0.97)  # 97% of target
        self.min_position_value = self.config.get('min_position_value', 1.0)  # USD
        self.sleep_time = self.config.get('sleep_time', 15)  # seconds
        
        # Enhanced features (Day 45)
        self.use_orderbook_analysis = self.config.get('use_orderbook_analysis', True)
        self.dynamic_spread_adjustment = self.config.get('dynamic_spread_adjustment', True)
        self.whale_threshold_usd = self.config.get('whale_threshold_usd', 50000)
        
        # Initialize order book analyzer
        if self.use_orderbook_analysis:
            self.orderbook_analyzer = OrderBookAnalyzer(whale_threshold_usd=self.whale_threshold_usd)
        
        # Exchange adapter (will be set via set_exchange_connection)
        self.exchange_adapter = None

    async def initialize(self, config: Dict):
        """Initialize strategy with configuration."""
        await super().initialize(config)
        
        # Get exchange adapter if available
        exchange_name = self.config.get('exchange', 'binance')
        if exchange_name in self.exchange_connections:
            self.exchange_adapter = self.exchange_connections[exchange_name]
        
        logger.info(f"Enhanced Market Maker Strategy initialized for {exchange_name}")

    async def _get_order_book(self, symbol: str) -> Optional[Dict]:
        """Get order book from exchange."""
        if not self.exchange_adapter:
            return None
        
        try:
            orderbook = await self.exchange_adapter.get_order_book(symbol)
            return orderbook
        except Exception as e:
            logger.error(f"Error fetching order book: {e}")
            return None

    def _adjust_spread_with_orderbook(self, price: float, orderbook: Optional[Dict]) -> tuple[float, float]:
        """
        Adjust buy/sell thresholds based on order book analysis.
        
        Args:
            price: Current price
            orderbook: Order book data
            
        Returns:
            Tuple of (adjusted_buy_under, adjusted_sell_over)
        """
        buy_under = self.buy_under
        sell_over = self.sell_over
        
        if not self.dynamic_spread_adjustment or not orderbook or not self.use_orderbook_analysis:
            return buy_under, sell_over
        
        try:
            bids = orderbook.get('bids', [])
            asks = orderbook.get('asks', [])
            
            if not bids or not asks:
                return buy_under, sell_over
            
            # Analyze order book
            analysis = self.orderbook_analyzer.analyze_order_book(bids, asks, price)
            
            # Adjust spread based on depth imbalance
            depth_imbalance = analysis['depth_metrics'].get('depth_imbalance', 0)
            
            # If bid depth >> ask depth, widen buy spread (less aggressive buying)
            if depth_imbalance > 0.3:
                buy_under = buy_under * 0.95  # More conservative buying
            # If ask depth >> bid depth, widen sell spread (less aggressive selling)
            elif depth_imbalance < -0.3:
                sell_over = sell_over * 1.05  # More conservative selling
            
            # Adjust based on whale orders
            whale_bias = analysis['whale_analysis'].get('bias', 'neutral')
            if whale_bias == 'bullish':
                # More whales buying, be more aggressive on sells
                sell_over = sell_over * 0.98
            elif whale_bias == 'bearish':
                # More whales selling, be more aggressive on buys
                buy_under = buy_under * 1.02
            
            return buy_under, sell_over
            
        except Exception as e:
            logger.error(f"Error adjusting spread: {e}")
            return buy_under, sell_over

    def check_signals(
        self,
        price: float,
        position_value: float,
        orderbook: Optional[Dict] = None
    ) -> Dict[str, any]:
        """
        Check market making signals with order book analysis.
        
        Args:
            price: Current price
            position_value: Current position value in USD
            orderbook: Optional order book data for enhanced analysis
            
        Returns:
            Dictionary with signals
        """
        result = {
            'action': 'HOLD',
            'reason': None,
            'price': price,
            'position_value': position_value,
            'metadata': {}
        }
        
        # Adjust thresholds based on order book
        buy_under, sell_over = self._adjust_spread_with_orderbook(price, orderbook)
        
        if orderbook and self.use_orderbook_analysis:
            # Add order book analysis to metadata
            try:
                bids = orderbook.get('bids', [])
                asks = orderbook.get('asks', [])
                analysis = self.orderbook_analyzer.analyze_order_book(bids, asks, price)
                result['metadata']['orderbook_analysis'] = analysis
            except Exception as e:
                logger.debug(f"Error analyzing order book: {e}")
        
        # Sell signal: Price > SELL_OVER and have position
        if price > sell_over:
            if position_value > self.min_position_value:
                result['action'] = 'SELL'
                result['reason'] = f"Price {price} > SELL_OVER {sell_over:.6f}"
                result['metadata']['sell_over'] = sell_over
            else:
                result['reason'] = f"Price {price} > SELL_OVER but no significant position"
        
        # Buy signal: Price < BUY_UNDER and position not full
        elif price < buy_under:
            target_threshold = self.target_size * self.position_threshold
            if position_value < target_threshold:
                result['action'] = 'BUY'
                result['reason'] = (
                    f"Price {price} < BUY_UNDER {buy_under:.6f} "
                    f"and position ${position_value:.2f} < target ${target_threshold:.2f}"
                )
                result['metadata']['buy_under'] = buy_under
            else:
                result['reason'] = f"Price {price} < BUY_UNDER but position already full"
        
        return result

    async def execute(self) -> Optional[Dict]:
        """
        Execute market making strategy.
        
        Returns:
            Trading signal dictionary or None
        """
        try:
            symbol = self.config.get('symbol', 'BTCUSDT')
            
            # Get current price
            if not self.exchange_adapter:
                logger.warning("No exchange adapter available")
                return None
            
            ticker = await self.exchange_adapter.get_ticker(symbol)
            if not ticker:
                return None
            
            current_price = float(ticker.get('last', 0))
            if current_price == 0:
                return None
            
            # Get current position
            position_value = 0  # Would get from position manager in production
            
            # Get order book if using enhanced analysis
            orderbook = None
            if self.use_orderbook_analysis:
                orderbook = await self._get_order_book(symbol)
            
            # Check signals
            signal = self.check_signals(current_price, position_value, orderbook)
            
            if signal['action'] == 'HOLD':
                return None
            
            # Calculate size
            if signal['action'] == 'BUY':
                size = self.calculate_buy_size(position_value)
            else:
                size = position_value * 0.95  # Sell 95% of position
            
            return {
                'action': signal['action'],
                'symbol': symbol,
                'size': size,
                'confidence': 0.7,
                'metadata': {
                    'strategy': 'enhanced_market_maker',
                    'reason': signal['reason'],
                    **signal.get('metadata', {})
                }
            }
            
        except Exception as e:
            logger.error(f"Error executing market maker strategy: {e}")
            return None

    def calculate_buy_size(self, current_value: float) -> float:
        """
        Calculate buy size needed.
        
        Args:
            current_value: Current position value in USD
            
        Returns:
            Buy size in USD
        """
        size_needed = self.target_size - current_value
        return max(0, size_needed)

    def should_sell(self, price: float, position_value: float) -> bool:
        """
        Check if should sell.
        
        Args:
            price: Current price
            position_value: Current position value
            
        Returns:
            True if should sell
        """
        return price > self.sell_over and position_value > self.min_position_value

    def should_buy(self, price: float, position_value: float) -> bool:
        """
        Check if should buy.
        
        Args:
            price: Current price
            position_value: Current position value
            
        Returns:
            True if should buy
        """
        target_threshold = self.target_size * self.position_threshold
        return price < self.buy_under and position_value < target_threshold

