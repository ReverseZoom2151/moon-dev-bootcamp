"""
Enhanced Supply/Demand Zone Strategy
====================================
Day 50: Enhanced Supply/Demand Zone Strategy with trend-based position management.

Features:
- Trend-based sell percentages (50% uptrend, 95% downtrend)
- Position buffers and verification logic
- Multi-order execution support
- Enhanced zone detection
- Position size management
"""

import logging
import asyncio
from typing import Dict, Optional, Tuple
from datetime import datetime

logger = logging.getLogger(__name__)


class EnhancedSupplyDemandStrategy:
    """
    Enhanced Supply/Demand Zone Strategy (Day 50).
    
    Features:
    - Trend-based sell percentages
    - Position buffers for verification
    - Multi-order execution
    - Enhanced zone detection
    """
    
    def __init__(
        self,
        exchange_adapter=None,
        config: Optional[Dict] = None
    ):
        """
        Initialize Enhanced Supply/Demand Zone Strategy.
        
        Args:
            exchange_adapter: Exchange adapter instance
            config: Configuration dictionary
        """
        self.exchange_adapter = exchange_adapter
        self.config = config or {}
        
        # Strategy parameters
        self.position_size_usd = self.config.get('position_size_usd', 150.0)
        self.minimum_position_pct = self.config.get('minimum_position_pct', 0.05)  # 5%
        self.buffer_pct = self.config.get('buffer_pct', 0.05)  # 5%
        
        # Trend-based sell percentages
        self.sell_pct_trending_up = self.config.get('sell_pct_trending_up', 0.50)  # 50%
        self.sell_pct_trending_down = self.config.get('sell_pct_trending_down', 0.95)  # 95%
        
        # Order execution
        self.orders_per_sell = self.config.get('orders_per_sell', 3)  # Triple-sell
        self.orders_per_open = self.config.get('orders_per_open', 1)
        
        # SMA settings for trend detection
        self.sma_timeframe = self.config.get('sma_timeframe', '1h')
        self.sma_bars = self.config.get('sma_bars', 10)
        self.sma_buffer_pct = self.config.get('sma_buffer_pct', 0.15)  # 15%
        
        # Zone detection
        self.timeframe = self.config.get('timeframe', '15m')
        self.days_back = self.config.get('days_back', 0.2)
        
        logger.info(
            f"Enhanced S/D Zone Strategy initialized: "
            f"Position Size=${self.position_size_usd}, "
            f"Sell Up={self.sell_pct_trending_up*100}%, "
            f"Sell Down={self.sell_pct_trending_down*100}%"
        )
    
    async def check_trend(
        self,
        symbol: str
    ) -> str:
        """
        Check trend using SMA (Day 50 method).
        
        Args:
            symbol: Trading symbol
            
        Returns:
            'up' or 'down'
        """
        if not self.exchange_adapter:
            return 'down'  # Default to down trend
        
        try:
            # Get historical data
            ohlcv_data = await self.exchange_adapter.get_ohlcv(
                symbol=symbol,
                timeframe=self.sma_timeframe,
                limit=self.sma_bars + 10
            )
            
            if not ohlcv_data or len(ohlcv_data) < self.sma_bars:
                logger.warning(f"Insufficient data for trend check: {symbol}")
                return 'down'
            
            # Convert to DataFrame
            import pandas as pd
            df = pd.DataFrame(ohlcv_data, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume'
            ])
            
            # Calculate SMA
            from gordon.core.utilities import EnhancedTradingUtils
            utils = EnhancedTradingUtils()
            sma = utils.calculate_sma(df['close'], window=self.sma_bars)
            
            if sma.empty:
                return 'down'
            
            current_price = float(df['close'].iloc[-1])
            current_sma = float(sma.iloc[-1])
            
            # Check if price is within buffer zone
            buffer_amount = current_sma * self.sma_buffer_pct
            
            if current_price > (current_sma + buffer_amount):
                return 'up'
            elif current_price < (current_sma - buffer_amount):
                return 'down'
            else:
                # Within buffer zone - use previous trend
                if len(sma) > 1:
                    prev_price = float(df['close'].iloc[-2])
                    if prev_price > current_sma:
                        return 'up'
                return 'down'
                
        except Exception as e:
            logger.error(f"Error checking trend: {e}")
            return 'down'
    
    async def get_supply_demand_zones(
        self,
        symbol: str
    ) -> Optional[Dict]:
        """
        Get supply and demand zones.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Dict with 'dz' (demand zone) and 'sz' (supply zone) DataFrames or None
        """
        try:
            # Get historical data
            ohlcv_data = await self.exchange_adapter.get_ohlcv(
                symbol=symbol,
                timeframe=self.timeframe,
                limit=1000
            )
            
            if not ohlcv_data:
                return None
            
            # Convert to DataFrame
            import pandas as pd
            df = pd.DataFrame(ohlcv_data, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume'
            ])
            
            # Calculate supply/demand zones using pivot method
            from gordon.core.utilities import EnhancedTradingUtils
            utils = EnhancedTradingUtils()
            
            sd_data = utils.calculate_supply_demand_zones_pivot(
                df,
                days_back=self.days_back,
                timeframe=self.timeframe
            )
            
            if not sd_data:
                return None
            
            # Create zone ranges
            demand_zone = pd.Series([sd_data['support'], sd_data['support2']])
            supply_zone = pd.Series([sd_data['resistance'], sd_data['resistance2']])
            
            return {
                'dz': demand_zone,
                'sz': supply_zone,
                'current_price': sd_data['current_price'],
                'support': sd_data['support'],
                'resistance': sd_data['resistance']
            }
            
        except Exception as e:
            logger.error(f"Error calculating S/D zones: {e}")
            return None
    
    async def get_current_position_value(
        self,
        symbol: str
    ) -> Tuple[float, float]:
        """
        Get current position size and value.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Tuple of (position_quantity, position_value_usd)
        """
        if not self.exchange_adapter:
            return 0.0, 0.0
        
        try:
            position = await self.exchange_adapter.get_position(symbol)
            if not position:
                return 0.0, 0.0
            
            quantity = float(position.get('quantity', 0))
            if abs(quantity) < 0.0001:
                return 0.0, 0.0
            
            # Get current price
            ticker = await self.exchange_adapter.get_ticker(symbol)
            if not ticker:
                return quantity, 0.0
            
            current_price = float(ticker.get('last', 0))
            position_value_usd = abs(quantity) * current_price
            
            return quantity, position_value_usd
            
        except Exception as e:
            logger.error(f"Error getting position: {e}")
            return 0.0, 0.0
    
    async def execute_multiple_sells(
        self,
        symbol: str,
        sell_size: float,
        num_orders: int = None
    ) -> bool:
        """
        Execute multiple sell orders (Day 50 triple-sell).
        
        Args:
            symbol: Trading symbol
            sell_size: Size to sell per order
            num_orders: Number of orders to place
            
        Returns:
            Success status
        """
        if not self.exchange_adapter:
            return False
        
        num_orders = num_orders or self.orders_per_sell
        
        try:
            for i in range(num_orders):
                order_result = await self.exchange_adapter.place_order(
                    symbol=symbol,
                    side='sell',
                    amount=sell_size,
                    order_type='market'
                )
                
                if order_result:
                    logger.info(f"Sell order {i+1}/{num_orders} executed")
                else:
                    logger.warning(f"Sell order {i+1}/{num_orders} failed")
                
                # Small delay between orders
                if i < num_orders - 1:
                    await asyncio.sleep(1)
            
            # Longer delay after all orders
            await asyncio.sleep(5)
            return True
            
        except Exception as e:
            logger.error(f"Error executing multiple sells: {e}")
            return False
    
    async def verify_position_reduction(
        self,
        symbol: str,
        expected_max_value: float,
        max_attempts: int = 3
    ) -> bool:
        """
        Verify that position was reduced (Day 50 verification logic).
        
        Args:
            symbol: Trading symbol
            expected_max_value: Maximum expected position value after sell
            max_attempts: Maximum verification attempts
            
        Returns:
            True if position reduced, False otherwise
        """
        for attempt in range(max_attempts):
            quantity, current_value = await self.get_current_position_value(symbol)
            
            if current_value <= expected_max_value:
                logger.info(f"Position verified: ${current_value:.2f} <= ${expected_max_value:.2f}")
                return True
            
            logger.warning(
                f"Position not reduced enough (attempt {attempt+1}/{max_attempts}): "
                f"${current_value:.2f} > ${expected_max_value:.2f}"
            )
            await asyncio.sleep(5)
        
        return False
    
    async def handle_demand_zone(
        self,
        symbol: str,
        trend: str,
        current_value: float,
        price: float
    ) -> Dict:
        """
        Handle trading logic when price is in demand zone.
        
        Args:
            symbol: Trading symbol
            trend: Current trend ('up' or 'down')
            current_value: Current position value in USD
            price: Current price
            
        Returns:
            Dict with action results
        """
        position_pct = current_value / self.position_size_usd if self.position_size_usd > 0 else 0
        
        # Only buy in uptrend and if position is below target
        if trend == 'up' and position_pct < (1 - self.buffer_pct):
            if current_value < self.position_size_usd:
                amount_to_buy_usd = self.position_size_usd - current_value
                
                logger.info(f"Buying ${amount_to_buy_usd:.2f} worth of {symbol} in demand zone")
                
                try:
                    # Place buy order
                    quantity = amount_to_buy_usd / price
                    order_result = await self.exchange_adapter.place_order(
                        symbol=symbol,
                        side='buy',
                        amount=quantity,
                        order_type='market'
                    )
                    
                    if order_result:
                        return {
                            'action': 'BOUGHT',
                            'amount_usd': amount_to_buy_usd,
                            'success': True
                        }
                    else:
                        return {
                            'action': 'BUY_FAILED',
                            'success': False
                        }
                except Exception as e:
                    logger.error(f"Error buying in demand zone: {e}")
                    return {
                        'action': 'BUY_ERROR',
                        'error': str(e),
                        'success': False
                    }
            else:
                return {
                    'action': 'ALREADY_AT_TARGET',
                    'message': 'Already at or above target position size'
                }
        else:
            return {
                'action': 'NO_BUY_SIGNAL',
                'reason': f"Trend not up or position too large (trend={trend}, pct={position_pct:.2%})"
            }
    
    async def handle_supply_zone(
        self,
        symbol: str,
        trend: str,
        position: float,
        current_value: float
    ) -> Dict:
        """
        Handle trading logic when price is in supply zone.
        
        Args:
            symbol: Trading symbol
            trend: Current trend ('up' or 'down')
            position: Current position quantity
            current_value: Current position value in USD
            
        Returns:
            Dict with action results
        """
        # Determine sell percentage based on trend
        if trend == 'up':
            sell_percentage = self.sell_pct_trending_up  # 50%
        else:
            sell_percentage = self.sell_pct_trending_down  # 95%
        
        # Check if position is meaningful
        min_position_value = self.position_size_usd * self.minimum_position_pct
        
        if position <= 0 or current_value < min_position_value:
            return {
                'action': 'NO_SELL_SIGNAL',
                'reason': 'No position or position too small'
            }
        
        # Calculate expected value after sell
        expected_value_after_sell = current_value * (1 - sell_percentage)
        verification_max_value = expected_value_after_sell * (1 + self.buffer_pct)
        
        logger.info(
            f"Selling {sell_percentage*100:.0f}% of position (Trend: {trend.upper()})"
        )
        logger.info(f"Expected value after sell: ${expected_value_after_sell:.2f}")
        
        # Calculate sell size
        sell_size_tokens = position * sell_percentage
        
        # Execute multiple sells
        success = await self.execute_multiple_sells(
            symbol=symbol,
            sell_size=sell_size_tokens / self.orders_per_sell,  # Divide by number of orders
            num_orders=self.orders_per_sell
        )
        
        if success:
            # Verify position reduction
            verified = await self.verify_position_reduction(
                symbol=symbol,
                expected_max_value=verification_max_value
            )
            
            return {
                'action': 'SOLD',
                'sell_percentage': sell_percentage,
                'sell_size_tokens': sell_size_tokens,
                'expected_value_after_sell': expected_value_after_sell,
                'verified': verified,
                'success': True
            }
        else:
            return {
                'action': 'SELL_FAILED',
                'success': False
            }
    
    async def execute(
        self,
        symbol: str
    ) -> Optional[Dict]:
        """
        Execute enhanced supply/demand zone strategy.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Dict with execution results
        """
        try:
            # Get trend
            trend = await self.check_trend(symbol)
            
            # Get current position
            position, current_value = await self.get_current_position_value(symbol)
            
            # Get supply/demand zones
            zones = await self.get_supply_demand_zones(symbol)
            if not zones:
                return {
                    'action': 'NO_ZONES',
                    'error': 'Could not calculate supply/demand zones'
                }
            
            current_price = zones['current_price']
            demand_zone_min = zones['dz'].min()
            demand_zone_max = zones['dz'].max()
            supply_zone_min = zones['sz'].min()
            supply_zone_max = zones['sz'].max()
            
            # Check which zone price is in
            if demand_zone_min <= current_price <= demand_zone_max:
                # In demand zone - buy logic
                result = await self.handle_demand_zone(
                    symbol=symbol,
                    trend=trend,
                    current_value=current_value,
                    price=current_price
                )
                result['zone'] = 'DEMAND'
                result['symbol'] = symbol
                return result
            
            elif supply_zone_min <= current_price <= supply_zone_max:
                # In supply zone - sell logic
                result = await self.handle_supply_zone(
                    symbol=symbol,
                    trend=trend,
                    position=position,
                    current_value=current_value
                )
                result['zone'] = 'SUPPLY'
                result['symbol'] = symbol
                return result
            
            else:
                return {
                    'action': 'NO_ZONE',
                    'message': 'Price not in any defined zone',
                    'symbol': symbol,
                    'price': current_price,
                    'demand_zone': (demand_zone_min, demand_zone_max),
                    'supply_zone': (supply_zone_min, supply_zone_max)
                }
                
        except Exception as e:
            logger.error(f"Error executing strategy: {e}")
            return {
                'action': 'ERROR',
                'error': str(e),
                'symbol': symbol
            }

