"""
Market Making Strategy
Based on Day_45_Projects mm.py implementation
"""

import logging
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List
from .base_strategy import BaseStrategy

logger = logging.getLogger(__name__)

class MarketMakingStrategy(BaseStrategy):
    """
    Market Making Strategy for providing liquidity
    
    Strategy Logic:
    - Places buy and sell orders around current market price
    - Maintains inventory balance
    - Adjusts spreads based on volatility
    - Manages risk through position limits
    """
    
    def __init__(self, market_data_manager, portfolio_manager, risk_manager, config):
        super().__init__(market_data_manager, portfolio_manager, risk_manager, config)
        
        self.name = "market_making"
        self.description = "Automated Market Making Strategy"
        
        # Market making parameters
        self.base_spread_pct = config.get('MM_SPREAD_PCT', 0.002)  # 0.2% base spread
        self.inventory_target = config.get('MM_INVENTORY_TARGET', 0.5)  # 50% target
        self.max_position_size = config.get('MM_MAX_POSITION_SIZE', 1000.0)
        self.min_order_size = config.get('MM_MIN_ORDER_SIZE', 10.0)
        self.max_order_size = config.get('MM_MAX_ORDER_SIZE', 100.0)
        
        # Risk management
        self.max_inventory_deviation = config.get('MM_MAX_INVENTORY_DEVIATION', 0.3)  # 30%
        self.volatility_adjustment = config.get('MM_VOLATILITY_ADJUSTMENT', True)
        self.max_spread_pct = config.get('MM_MAX_SPREAD_PCT', 0.01)  # 1% max spread
        
        # Order management
        self.order_refresh_interval = config.get('MM_ORDER_REFRESH_INTERVAL', 30)  # seconds
        self.price_update_threshold = config.get('MM_PRICE_UPDATE_THRESHOLD', 0.001)  # 0.1%
        
        # State tracking
        self.active_orders = {}
        self.inventory_positions = {}
        self.last_price_update = {}
        self.volatility_cache = {}
        
        # Performance tracking
        self.total_volume = 0.0
        self.total_profit = 0.0
        self.orders_filled = 0
        self.inventory_turns = 0
    
    async def initialize(self):
        """Initialize the market making strategy"""
        try:
            logger.info(f"🏭 Initializing Market Making Strategy")
            logger.info(f"📊 Base Spread: {self.base_spread_pct:.3%}")
            logger.info(f"🎯 Inventory Target: {self.inventory_target:.1%}")
            logger.info(f"💰 Max Position: ${self.max_position_size:,.2f}")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to initialize Market Making Strategy: {e}")
            return False
    
    async def should_trade(self, symbol: str, data: Dict) -> bool:
        """Check if we should provide liquidity for this symbol"""
        try:
            # Get market data
            historical_data = await self.market_data_manager.get_historical_data(
                symbol, 
                timeframe='1m', 
                limit=100
            )
            
            if historical_data is None or len(historical_data) < 20:
                return False
            
            # Calculate volatility
            await self._calculate_volatility(symbol, historical_data)
            
            # Check if market conditions are suitable for market making
            current_price = data.get('price', 0)
            if current_price <= 0:
                return False
            
            # Check volume requirements
            recent_volume = historical_data['volume'].tail(10).mean()
            if recent_volume < 1000:  # Minimum volume threshold
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Error checking if should trade {symbol}: {e}")
            return False
    
    async def generate_signals(self, symbol: str, data: Dict) -> Dict:
        """Generate market making orders"""
        try:
            current_price = data.get('price', 0)
            if current_price <= 0:
                return {'action': 'hold', 'confidence': 0.0, 'reason': 'Invalid price'}
            
            # Check if we need to refresh orders
            if not await self._should_refresh_orders(symbol, current_price):
                return {'action': 'hold', 'confidence': 0.0, 'reason': 'Orders still valid'}
            
            # Calculate optimal spread
            spread = await self._calculate_optimal_spread(symbol, current_price)
            
            # Get current inventory
            inventory_ratio = await self._get_inventory_ratio(symbol, current_price)
            
            # Generate buy and sell orders
            orders = await self._generate_market_making_orders(
                symbol, current_price, spread, inventory_ratio
            )
            
            if not orders:
                return {'action': 'hold', 'confidence': 0.0, 'reason': 'No orders generated'}
            
            return {
                'action': 'market_make',
                'confidence': 0.8,
                'reason': f'Market making with {spread:.3%} spread',
                'orders': orders,
                'spread': spread,
                'inventory_ratio': inventory_ratio
            }
            
        except Exception as e:
            logger.error(f"❌ Error generating signals for {symbol}: {e}")
            return {'action': 'hold', 'confidence': 0.0, 'reason': f'Error: {str(e)}'}
    
    async def _should_refresh_orders(self, symbol: str, current_price: float) -> bool:
        """Check if orders need to be refreshed"""
        try:
            # Check if enough time has passed
            if symbol in self.last_price_update:
                time_since_update = (datetime.now() - self.last_price_update[symbol]).total_seconds()
                if time_since_update < self.order_refresh_interval:
                    return False
            
            # Check if price has moved significantly
            if symbol in self.last_price_update:
                last_price = getattr(self, f'last_price_{symbol}', current_price)
                price_change = abs(current_price - last_price) / last_price
                
                if price_change > self.price_update_threshold:
                    return True
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Error checking order refresh for {symbol}: {e}")
            return True
    
    async def _calculate_optimal_spread(self, symbol: str, current_price: float) -> float:
        """Calculate optimal spread based on market conditions"""
        try:
            base_spread = self.base_spread_pct
            
            # Adjust for volatility if enabled
            if self.volatility_adjustment and symbol in self.volatility_cache:
                volatility = self.volatility_cache[symbol]
                # Higher volatility = wider spread
                volatility_adjustment = min(volatility * 2, 0.005)  # Max 0.5% adjustment
                base_spread += volatility_adjustment
            
            # Adjust for inventory imbalance
            inventory_ratio = await self._get_inventory_ratio(symbol, current_price)
            inventory_deviation = abs(inventory_ratio - self.inventory_target)
            
            if inventory_deviation > self.max_inventory_deviation:
                # Widen spread when inventory is imbalanced
                inventory_adjustment = inventory_deviation * 0.001  # 0.1% per 10% deviation
                base_spread += inventory_adjustment
            
            # Cap the spread
            final_spread = min(base_spread, self.max_spread_pct)
            
            return final_spread
            
        except Exception as e:
            logger.error(f"❌ Error calculating optimal spread for {symbol}: {e}")
            return self.base_spread_pct
    
    async def _get_inventory_ratio(self, symbol: str, current_price: float) -> float:
        """Get current inventory ratio (0 = all cash, 1 = all asset)"""
        try:
            # Get current position
            current_position = await self.portfolio_manager.get_position(symbol)
            
            if not current_position:
                return 0.0  # No position = all cash
            
            # Calculate position value
            position_value = current_position.quantity * current_price
            
            # Get total allocated capital for this symbol
            total_capital = self.max_position_size
            
            # Calculate inventory ratio
            inventory_ratio = position_value / total_capital
            
            return max(0.0, min(1.0, inventory_ratio))  # Clamp between 0 and 1
            
        except Exception as e:
            logger.error(f"❌ Error getting inventory ratio for {symbol}: {e}")
            return 0.5  # Default to balanced
    
    async def _generate_market_making_orders(self, symbol: str, price: float, spread: float, inventory_ratio: float) -> List[Dict]:
        """Generate buy and sell orders for market making"""
        try:
            orders = []
            
            # Calculate bid and ask prices
            half_spread = spread / 2
            bid_price = price * (1 - half_spread)
            ask_price = price * (1 + half_spread)
            
            # Adjust order sizes based on inventory
            base_order_size = (self.min_order_size + self.max_order_size) / 2
            
            # Buy order (more aggressive when inventory is low)
            if inventory_ratio < (self.inventory_target + self.max_inventory_deviation):
                buy_size_multiplier = 1 + (self.inventory_target - inventory_ratio)
                buy_size = min(base_order_size * buy_size_multiplier, self.max_order_size)
                
                orders.append({
                    'side': 'buy',
                    'price': bid_price,
                    'quantity': buy_size / bid_price,  # Convert to asset quantity
                    'order_type': 'limit',
                    'time_in_force': 'GTC'
                })
            
            # Sell order (more aggressive when inventory is high)
            if inventory_ratio > (self.inventory_target - self.max_inventory_deviation):
                sell_size_multiplier = 1 + (inventory_ratio - self.inventory_target)
                sell_size = min(base_order_size * sell_size_multiplier, self.max_order_size)
                
                orders.append({
                    'side': 'sell',
                    'price': ask_price,
                    'quantity': sell_size / ask_price,  # Convert to asset quantity
                    'order_type': 'limit',
                    'time_in_force': 'GTC'
                })
            
            return orders
            
        except Exception as e:
            logger.error(f"❌ Error generating market making orders for {symbol}: {e}")
            return []
    
    async def _calculate_volatility(self, symbol: str, data: pd.DataFrame):
        """Calculate recent volatility for spread adjustment"""
        try:
            if len(data) < 20:
                return
            
            # Calculate returns
            returns = data['close'].pct_change().dropna()
            
            # Calculate rolling volatility (20-period)
            volatility = returns.rolling(window=20).std().iloc[-1]
            
            # Annualize volatility (assuming 1-minute data)
            annualized_volatility = volatility * np.sqrt(525600)  # 525600 minutes in a year
            
            # Store volatility
            self.volatility_cache[symbol] = annualized_volatility
            
        except Exception as e:
            logger.error(f"❌ Error calculating volatility for {symbol}: {e}")
    
    async def calculate_position_size(self, symbol: str, signal: Dict) -> float:
        """Calculate position size for market making orders"""
        try:
            orders = signal.get('orders', [])
            if not orders:
                return 0
            
            # For market making, return the order quantity directly
            # This will be handled by the order management system
            return orders[0].get('quantity', 0)
            
        except Exception as e:
            logger.error(f"❌ Error calculating position size for {symbol}: {e}")
            return 0
    
    async def on_trade_executed(self, symbol: str, side: str, quantity: float, price: float):
        """Handle trade execution for market making"""
        try:
            # Update performance metrics
            trade_value = quantity * price
            self.total_volume += trade_value
            self.orders_filled += 1
            
            # Estimate profit (simplified)
            if side == 'sell':
                # Assume we bought at a lower price (spread/2 below current)
                estimated_buy_price = price * (1 - self.base_spread_pct / 2)
                profit = (price - estimated_buy_price) * quantity
                self.total_profit += profit
                
                logger.info(f"🏭 MM SELL: {symbol} @ {price:.6f} | Size: {quantity:.4f} | Est. Profit: ${profit:.2f}")
            else:
                logger.info(f"🏭 MM BUY: {symbol} @ {price:.6f} | Size: {quantity:.4f}")
            
            # Update last price
            setattr(self, f'last_price_{symbol}', price)
            self.last_price_update[symbol] = datetime.now()
            
        except Exception as e:
            logger.error(f"❌ Error handling trade execution for {symbol}: {e}")
    
    async def get_strategy_stats(self) -> Dict:
        """Get market making strategy performance statistics"""
        try:
            # Calculate average profit per trade
            avg_profit_per_trade = (self.total_profit / self.orders_filled) if self.orders_filled > 0 else 0
            
            # Calculate profit margin
            profit_margin = (self.total_profit / self.total_volume * 100) if self.total_volume > 0 else 0
            
            return {
                'strategy_name': self.name,
                'total_volume': self.total_volume,
                'total_profit': self.total_profit,
                'orders_filled': self.orders_filled,
                'avg_profit_per_trade': avg_profit_per_trade,
                'profit_margin_pct': profit_margin,
                'active_symbols': len(self.active_orders),
                'parameters': {
                    'base_spread_pct': self.base_spread_pct,
                    'inventory_target': self.inventory_target,
                    'max_position_size': self.max_position_size,
                    'volatility_adjustment': self.volatility_adjustment
                }
            }
            
        except Exception as e:
            logger.error(f"❌ Error getting strategy stats: {e}")
            return {'strategy_name': self.name, 'error': str(e)}
    
    async def update_parameters(self, new_params: Dict):
        """Update strategy parameters"""
        try:
            if 'base_spread_pct' in new_params:
                self.base_spread_pct = new_params['base_spread_pct']
            if 'inventory_target' in new_params:
                self.inventory_target = new_params['inventory_target']
            if 'max_position_size' in new_params:
                self.max_position_size = new_params['max_position_size']
            if 'volatility_adjustment' in new_params:
                self.volatility_adjustment = new_params['volatility_adjustment']
            
            logger.info(f"🏭 Market Making parameters updated: {new_params}")
            
        except Exception as e:
            logger.error(f"❌ Error updating parameters: {e}")
    
    async def cleanup(self):
        """Cleanup strategy resources"""
        try:
            # Cancel all active orders
            for symbol in list(self.active_orders.keys()):
                # In production, this would cancel orders via exchange API
                logger.info(f"🏭 Cancelling market making orders for {symbol}")
            
            # Clear data structures
            self.active_orders.clear()
            self.inventory_positions.clear()
            self.last_price_update.clear()
            self.volatility_cache.clear()
            
            logger.info("🧹 Market Making strategy cleaned up")
            
        except Exception as e:
            logger.error(f"❌ Error cleaning up Market Making strategy: {e}") 