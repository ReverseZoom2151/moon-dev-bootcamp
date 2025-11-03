"""
Easy Entry Strategy
===================
Day 48: Averaging in at demand zones and technical levels.

Features:
- Market buy loops with averaging
- Supply/demand zone entry
- SMA trend following entry
- Position management
"""

import logging
import asyncio
from typing import Dict, Optional, Tuple
from datetime import datetime

logger = logging.getLogger(__name__)


class EasyEntryStrategy:
    """
    Easy entry strategy for averaging in at good prices.
    
    Features:
    - Market buy loops with chunking
    - Supply/demand zone entry
    - SMA trend following
    - Target position sizing
    """
    
    def __init__(
        self,
        exchange_adapter=None,
        config: Optional[Dict] = None
    ):
        """
        Initialize easy entry strategy.
        
        Args:
            exchange_adapter: Exchange adapter instance
            config: Configuration dictionary
        """
        self.exchange_adapter = exchange_adapter
        self.config = config or {}
        
        # Default configuration
        self.default_total_position_size_usd = self.config.get('total_position_size_usd', 1000.0)
        self.default_max_order_size_usd = self.config.get('max_order_size_usd', 500.0)
        self.default_target_fill_ratio = self.config.get('target_fill_ratio', 0.97)  # 97%
        self.default_sleep_after_buy = self.config.get('sleep_after_buy', 10)  # seconds
        self.default_min_buy_threshold_usd = self.config.get('min_buy_threshold_usd', 10.0)
        self.default_max_attempts = self.config.get('max_attempts', 20)
    
    async def get_current_position_usd(self, symbol: str) -> Tuple[float, float]:
        """
        Get current position size in units and USD value.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Tuple of (position_units, position_value_usd)
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
            logger.error(f"Error getting position for {symbol}: {e}")
            return 0.0, 0.0
    
    async def market_buy_loop(
        self,
        symbol: str,
        target_position_usd: float,
        max_order_size_usd: Optional[float] = None,
        sleep_after_buy: Optional[int] = None
    ) -> Dict:
        """
        Execute market buy loop to reach target position size.
        
        Args:
            symbol: Trading symbol
            target_position_usd: Target position size in USD
            max_order_size_usd: Maximum order size per chunk
            sleep_after_buy: Sleep duration after each buy
            
        Returns:
            Dict with execution results
        """
        max_order_size_usd = max_order_size_usd or self.default_max_order_size_usd
        sleep_after_buy = sleep_after_buy or self.default_sleep_after_buy
        
        target_fill_usd = target_position_usd * self.default_target_fill_ratio
        
        logger.info(
            f"Starting market buy loop for {symbol}. "
            f"Target: ${target_position_usd:.2f} (aiming for ${target_fill_usd:.2f})"
        )
        
        attempts = []
        attempt_num = 0
        
        while attempt_num < self.default_max_attempts:
            attempt_num += 1
            
            # Get current position
            pos_units, pos_value_usd = await self.get_current_position_usd(symbol)
            
            # Check if target reached
            if pos_value_usd >= target_fill_usd:
                logger.info(
                    f"Target reached! Position: ${pos_value_usd:.2f} >= ${target_fill_usd:.2f}"
                )
                break
            
            # Calculate remaining needed
            remaining_usd = target_position_usd - pos_value_usd
            
            if remaining_usd < self.default_min_buy_threshold_usd:
                logger.info(
                    f"Position very close to target (need ${remaining_usd:.2f}, "
                    f"threshold ${self.default_min_buy_threshold_usd:.2f}). Stopping."
                )
                break
            
            # Calculate buy chunk size
            buy_chunk_usd = min(remaining_usd, max_order_size_usd)
            
            logger.info(
                f"Attempt {attempt_num}: Buying ${buy_chunk_usd:.2f} worth of {symbol}"
            )
            
            try:
                # Place market buy order
                order_result = await self.exchange_adapter.place_order(
                    symbol=symbol,
                    side='buy',
                    amount=buy_chunk_usd,  # Assuming exchange adapter accepts USD amount
                    order_type='market',
                    quote_quantity=buy_chunk_usd  # Alternative parameter name
                )
                
                if order_result:
                    attempts.append({
                        'attempt': attempt_num,
                        'amount_usd': buy_chunk_usd,
                        'success': True,
                        'order_id': order_result.get('id', 'N/A'),
                        'timestamp': datetime.now().isoformat()
                    })
                    logger.info(f"Buy order {attempt_num} executed successfully")
                else:
                    attempts.append({
                        'attempt': attempt_num,
                        'amount_usd': buy_chunk_usd,
                        'success': False,
                        'error': 'Order execution failed',
                        'timestamp': datetime.now().isoformat()
                    })
                    logger.warning(f"Buy order {attempt_num} failed")
                
            except Exception as e:
                logger.error(f"Error in buy attempt {attempt_num}: {e}")
                attempts.append({
                    'attempt': attempt_num,
                    'amount_usd': buy_chunk_usd,
                    'success': False,
                    'error': str(e),
                    'timestamp': datetime.now().isoformat()
                })
            
            # Sleep before next attempt
            if attempt_num < self.default_max_attempts:
                await asyncio.sleep(sleep_after_buy)
        
        # Final position check
        final_units, final_value_usd = await self.get_current_position_usd(symbol)
        
        successful_attempts = [a for a in attempts if a.get('success')]
        total_bought_usd = sum(a['amount_usd'] for a in successful_attempts)
        
        return {
            'success': len(successful_attempts) > 0,
            'symbol': symbol,
            'target_position_usd': target_position_usd,
            'final_position_usd': final_value_usd,
            'total_bought_usd': total_bought_usd,
            'attempts': attempts,
            'successful_attempts': len(successful_attempts),
            'timestamp': datetime.now().isoformat()
        }
    
    async def supply_demand_zone_entry(
        self,
        symbol: str,
        target_position_usd: float,
        max_order_size_usd: Optional[float] = None,
        sd_data: Optional[Dict] = None,
        timeframe: str = '15m',
        days_back: int = 1
    ) -> Dict:
        """
        Enter position at supply/demand zone.
        
        Args:
            symbol: Trading symbol
            target_position_usd: Target position size in USD
            max_order_size_usd: Maximum order size per chunk
            sd_data: Pre-calculated supply/demand data (optional)
            timeframe: Timeframe for analysis
            days_back: Days to look back
            
        Returns:
            Dict with entry results
        """
        from .trading_utils import EnhancedTradingUtils
        
        # Get supply/demand zones if not provided
        if sd_data is None:
            try:
                # Get historical data
                ohlcv_data = await self.exchange_adapter.get_ohlcv(
                    symbol=symbol,
                    timeframe=timeframe,
                    limit=1000
                )
                
                if not ohlcv_data:
                    return {
                        'success': False,
                        'error': 'Could not fetch historical data'
                    }
                
                # Convert to DataFrame
                import pandas as pd
                df = pd.DataFrame(ohlcv_data, columns=[
                    'timestamp', 'open', 'high', 'low', 'close', 'volume'
                ])
                
                # Calculate supply/demand zones
                utils = EnhancedTradingUtils()
                sd_data = utils.calculate_supply_demand_zones_pivot(
                    df,
                    days_back=days_back,
                    timeframe=timeframe
                )
                
            except Exception as e:
                logger.error(f"Error calculating supply/demand zones: {e}")
                return {
                    'success': False,
                    'error': f'Supply/demand calculation failed: {e}'
                }
        
        if not sd_data:
            return {
                'success': False,
                'error': 'Could not calculate supply/demand zones'
            }
        
        # Check if price is near support/demand zone
        if sd_data.get('near_support'):
            logger.info(
                f"Price is near support/demand zone for {symbol}. "
                f"Executing buy..."
            )
            
            # Check current position
            pos_units, pos_value_usd = await self.get_current_position_usd(symbol)
            target_fill_usd = target_position_usd * self.default_target_fill_ratio
            
            if pos_value_usd >= target_fill_usd:
                return {
                    'success': True,
                    'message': 'Already at target position size',
                    'position_usd': pos_value_usd
                }
            
            # Calculate buy amount
            buy_amount = min(
                max_order_size_usd or self.default_max_order_size_usd,
                target_position_usd - pos_value_usd
            )
            
            # Execute buy
            try:
                order_result = await self.exchange_adapter.place_order(
                    symbol=symbol,
                    side='buy',
                    amount=buy_amount,
                    order_type='market',
                    quote_quantity=buy_amount
                )
                
                if order_result:
                    return {
                        'success': True,
                        'entry_type': 'supply_demand_zone',
                        'buy_amount_usd': buy_amount,
                        'order_id': order_result.get('id', 'N/A'),
                        'sd_data': sd_data
                    }
                else:
                    return {
                        'success': False,
                        'error': 'Order execution failed'
                    }
                    
            except Exception as e:
                logger.error(f"Error executing demand zone entry: {e}")
                return {
                    'success': False,
                    'error': str(e)
                }
        else:
            return {
                'success': False,
                'message': 'Not near demand zone',
                'sd_data': sd_data,
                'support_distance': sd_data.get('support_distance', 0)
            }
    
    async def sma_trend_entry(
        self,
        symbol: str,
        target_position_usd: float,
        max_order_size_usd: Optional[float] = None,
        sma_periods: int = 20,
        timeframe: str = '5m'
    ) -> Dict:
        """
        Enter position based on SMA trend following.
        
        Args:
            symbol: Trading symbol
            target_position_usd: Target position size in USD
            max_order_size_usd: Maximum order size per chunk
            sma_periods: SMA period
            timeframe: Timeframe for analysis
            
        Returns:
            Dict with entry results
        """
        from .trading_utils import EnhancedTradingUtils
        
        try:
            # Get historical data
            ohlcv_data = await self.exchange_adapter.get_ohlcv(
                symbol=symbol,
                timeframe=timeframe,
                limit=sma_periods + 10
            )
            
            if not ohlcv_data:
                return {
                    'success': False,
                    'error': 'Could not fetch historical data'
                }
            
            # Convert to DataFrame
            import pandas as pd
            df = pd.DataFrame(ohlcv_data, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume'
            ])
            
            # Check trend
            utils = EnhancedTradingUtils()
            trend_data = utils.check_trend_sma(df, periods=sma_periods)
            
            if not trend_data:
                return {
                    'success': False,
                    'error': 'Could not calculate trend'
                }
            
            # Check if uptrend
            if trend_data.get('is_uptrend'):
                logger.info(
                    f"Uptrend detected for {symbol}. Trend strength: "
                    f"{trend_data.get('trend_strength', 0):.2f}%"
                )
                
                # Check current position
                pos_units, pos_value_usd = await self.get_current_position_usd(symbol)
                target_fill_usd = target_position_usd * self.default_target_fill_ratio
                
                if pos_value_usd >= target_fill_usd:
                    return {
                        'success': True,
                        'message': 'Already at target position size',
                        'position_usd': pos_value_usd
                    }
                
                # Calculate buy amount
                buy_amount = min(
                    max_order_size_usd or self.default_max_order_size_usd,
                    target_position_usd - pos_value_usd
                )
                
                # Execute buy
                order_result = await self.exchange_adapter.place_order(
                    symbol=symbol,
                    side='buy',
                    amount=buy_amount,
                    order_type='market',
                    quote_quantity=buy_amount
                )
                
                if order_result:
                    return {
                        'success': True,
                        'entry_type': 'sma_trend',
                        'buy_amount_usd': buy_amount,
                        'order_id': order_result.get('id', 'N/A'),
                        'trend_data': trend_data
                    }
                else:
                    return {
                        'success': False,
                        'error': 'Order execution failed'
                    }
            else:
                return {
                    'success': False,
                    'message': 'Downtrend detected, no buy signal',
                    'trend_data': trend_data
                }
                
        except Exception as e:
            logger.error(f"Error in SMA trend entry: {e}")
            return {
                'success': False,
                'error': str(e)
            }

