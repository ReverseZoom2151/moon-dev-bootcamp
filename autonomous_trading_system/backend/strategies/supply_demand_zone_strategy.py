"""
Supply/Demand Zone Strategy
Based on Day_50_Projects sdz.py implementation
"""

import logging
import pandas as pd
from datetime import datetime
from typing import Dict
from .base_strategy import BaseStrategy

logger = logging.getLogger(__name__)

class SupplyDemandZoneStrategy(BaseStrategy):
    """
    Supply/Demand Zone Strategy for accumulation and distribution
    
    Strategy Logic:
    - Identifies supply and demand zones from price action
    - Buys in demand zones (support levels)
    - Sells in supply zones (resistance levels)
    - Uses trend analysis for position sizing and exit timing
    """
    
    def __init__(self, market_data_manager, portfolio_manager, risk_manager, config):
        super().__init__(market_data_manager, portfolio_manager, risk_manager, config)
        
        self.name = "supply_demand_zone"
        self.description = "Supply/Demand Zone Strategy for accumulation"
        
        # Strategy parameters from config
        self.timeframe = config.get('SDZ_TIMEFRAME', '15m')
        self.days_back = config.get('SDZ_DAYS_BACK', 0.2)
        self.position_size_usd = config.get('SDZ_POSITION_SIZE_USD', 150)
        self.buffer_pct = config.get('SDZ_BUFFER_PCT', 0.05)  # 5% buffer for zones
        self.minimum_position_pct = config.get('SDZ_MINIMUM_POSITION_PCT', 0.05)
        
        # Trend analysis parameters
        self.sma_timeframe = config.get('SDZ_SMA_TIMEFRAME', '1H')
        self.sma_days_back = config.get('SDZ_SMA_DAYS_BACK', 2)
        self.sma_bars = config.get('SDZ_SMA_BARS', 10)
        self.sma_buffer_pct = config.get('SDZ_SMA_BUFFER_PCT', 0.15)
        
        # Sell percentages based on trend
        self.sell_pct_trending_up = config.get('SDZ_SELL_PCT_TRENDING_UP', 0.50)
        self.sell_pct_trending_down = config.get('SDZ_SELL_PCT_TRENDING_DOWN', 0.95)
        
        # Order execution settings
        self.orders_per_open = config.get('SDZ_ORDERS_PER_OPEN', 1)
        self.orders_per_sell = config.get('SDZ_ORDERS_PER_SELL', 3)
        
        # Data storage
        self.zone_data = {}
        self.trend_data = {}
        self.position_tracking = {}
        
        # Performance tracking
        self.trades_count = 0
        self.successful_entries = 0
        self.successful_exits = 0
    
    async def initialize(self):
        """Initialize the strategy"""
        try:
            logger.info(f"🏗️ Initializing Supply/Demand Zone Strategy")
            logger.info(f"📊 Position Size: ${self.position_size_usd}")
            logger.info(f"🎯 Buffer: {self.buffer_pct:.1%}, Timeframe: {self.timeframe}")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to initialize S/D Zone Strategy: {e}")
            return False
    
    async def should_trade(self, symbol: str, data: Dict) -> bool:
        """Check if we should trade this symbol"""
        try:
            # Get historical data for zone analysis
            historical_data = await self.market_data_manager.get_historical_data(
                symbol, 
                timeframe=self.timeframe, 
                limit=100
            )
            
            if historical_data is None or len(historical_data) < 20:
                return False
            
            # Calculate supply/demand zones
            await self._calculate_zones(symbol, historical_data)
            
            # Get trend data
            await self._analyze_trend(symbol)
            
            return symbol in self.zone_data and symbol in self.trend_data
            
        except Exception as e:
            logger.error(f"❌ Error checking if should trade {symbol}: {e}")
            return False
    
    async def generate_signals(self, symbol: str, data: Dict) -> Dict:
        """Generate trading signals based on supply/demand zones"""
        try:
            if symbol not in self.zone_data or symbol not in self.trend_data:
                return {'action': 'hold', 'confidence': 0.0, 'reason': 'No zone/trend data'}
            
            current_price = data.get('price', 0)
            if current_price <= 0:
                return {'action': 'hold', 'confidence': 0.0, 'reason': 'Invalid price'}
            
            zones = self.zone_data[symbol]
            trend = self.trend_data[symbol]['trend']
            
            # Get current position
            current_position = await self.portfolio_manager.get_position(symbol)
            position_value = 0
            if current_position:
                position_value = current_position.quantity * current_price
            
            position_pct = position_value / self.position_size_usd if self.position_size_usd > 0 else 0
            
            # Check for demand zone entry (buy signal)
            demand_signal = await self._check_demand_zone_entry(
                symbol, current_price, zones, trend, position_pct
            )
            if demand_signal['action'] != 'hold':
                return demand_signal
            
            # Check for supply zone exit (sell signal)
            supply_signal = await self._check_supply_zone_exit(
                symbol, current_price, zones, trend, position_value, position_pct
            )
            if supply_signal['action'] != 'hold':
                return supply_signal
            
            return {
                'action': 'hold',
                'confidence': 0.0,
                'reason': 'No zone signals',
                'zone_info': {
                    'trend': trend,
                    'position_pct': position_pct,
                    'zones': len(zones.get('demand', [])) + len(zones.get('supply', []))
                }
            }
            
        except Exception as e:
            logger.error(f"❌ Error generating signals for {symbol}: {e}")
            return {'action': 'hold', 'confidence': 0.0, 'reason': f'Error: {str(e)}'}
    
    async def _calculate_zones(self, symbol: str, data: pd.DataFrame):
        """Calculate supply and demand zones from price data"""
        try:
            if len(data) < 20:
                return
            
            # Identify swing highs and lows
            swing_highs = []
            swing_lows = []
            
            # Look for swing points (simplified approach)
            for i in range(2, len(data) - 2):
                # Swing high: higher than 2 bars before and after
                if (data['high'].iloc[i] > data['high'].iloc[i-1] and 
                    data['high'].iloc[i] > data['high'].iloc[i-2] and
                    data['high'].iloc[i] > data['high'].iloc[i+1] and 
                    data['high'].iloc[i] > data['high'].iloc[i+2]):
                    swing_highs.append({
                        'price': data['high'].iloc[i],
                        'index': i,
                        'timestamp': data.index[i] if hasattr(data.index[i], 'timestamp') else i
                    })
                
                # Swing low: lower than 2 bars before and after
                if (data['low'].iloc[i] < data['low'].iloc[i-1] and 
                    data['low'].iloc[i] < data['low'].iloc[i-2] and
                    data['low'].iloc[i] < data['low'].iloc[i+1] and 
                    data['low'].iloc[i] < data['low'].iloc[i+2]):
                    swing_lows.append({
                        'price': data['low'].iloc[i],
                        'index': i,
                        'timestamp': data.index[i] if hasattr(data.index[i], 'timestamp') else i
                    })
            
            # Create supply zones from swing highs (resistance)
            supply_zones = []
            for swing in swing_highs[-5:]:  # Use last 5 swing highs
                zone_high = swing['price'] * (1 + self.buffer_pct)
                zone_low = swing['price'] * (1 - self.buffer_pct)
                supply_zones.append({
                    'high': zone_high,
                    'low': zone_low,
                    'center': swing['price'],
                    'strength': 1,  # Could be enhanced with volume/touch analysis
                    'timestamp': swing['timestamp']
                })
            
            # Create demand zones from swing lows (support)
            demand_zones = []
            for swing in swing_lows[-5:]:  # Use last 5 swing lows
                zone_high = swing['price'] * (1 + self.buffer_pct)
                zone_low = swing['price'] * (1 - self.buffer_pct)
                demand_zones.append({
                    'high': zone_high,
                    'low': zone_low,
                    'center': swing['price'],
                    'strength': 1,
                    'timestamp': swing['timestamp']
                })
            
            # Store zones
            self.zone_data[symbol] = {
                'supply': supply_zones,
                'demand': demand_zones,
                'last_updated': datetime.now()
            }
            
            logger.info(f"📊 {symbol[:8]}... zones: {len(supply_zones)} supply, {len(demand_zones)} demand")
            
        except Exception as e:
            logger.error(f"❌ Error calculating zones for {symbol}: {e}")
    
    async def _analyze_trend(self, symbol: str):
        """Analyze trend using SMA"""
        try:
            # Get SMA data
            sma_data = await self.market_data_manager.get_historical_data(
                symbol,
                timeframe=self.sma_timeframe,
                limit=self.sma_bars + 5
            )
            
            if sma_data is None or len(sma_data) < self.sma_bars:
                self.trend_data[symbol] = {'trend': 'down', 'sma': 0}
                return
            
            # Calculate SMA
            sma = sma_data['close'].tail(self.sma_bars).mean()
            current_price = sma_data['close'].iloc[-1]
            
            # Determine trend with buffer
            sma_lower = sma * (1 - self.sma_buffer_pct)
            trend = 'up' if current_price > sma_lower else 'down'
            
            self.trend_data[symbol] = {
                'trend': trend,
                'sma': sma,
                'current_price': current_price,
                'last_updated': datetime.now()
            }
            
            logger.info(f"📈 {symbol[:8]}... trend: {trend.upper()}, SMA: {sma:.6f}")
            
        except Exception as e:
            logger.error(f"❌ Error analyzing trend for {symbol}: {e}")
            self.trend_data[symbol] = {'trend': 'down', 'sma': 0}
    
    async def _check_demand_zone_entry(self, symbol: str, price: float, zones: Dict, trend: str, position_pct: float) -> Dict:
        """Check if price is in a demand zone for entry"""
        try:
            demand_zones = zones.get('demand', [])
            
            # Check if price is in any demand zone
            in_demand_zone = False
            zone_strength = 0
            
            for zone in demand_zones:
                if zone['low'] <= price <= zone['high']:
                    in_demand_zone = True
                    zone_strength = max(zone_strength, zone['strength'])
                    break
            
            if not in_demand_zone:
                return {'action': 'hold', 'confidence': 0.0, 'reason': 'Not in demand zone'}
            
            # Check if we need more position
            if position_pct >= 1.0:
                return {'action': 'hold', 'confidence': 0.0, 'reason': 'Position already full'}
            
            # Calculate confidence based on trend and zone strength
            confidence = 0.6  # Base confidence
            if trend == 'up':
                confidence += 0.2  # Higher confidence in uptrend
            
            confidence += (zone_strength - 1) * 0.1  # Adjust for zone strength
            confidence = min(confidence, 1.0)
            
            return {
                'action': 'buy',
                'confidence': confidence,
                'reason': f'In demand zone, trend: {trend}, position: {position_pct:.1%}',
                'zone_type': 'demand',
                'zone_strength': zone_strength
            }
            
        except Exception as e:
            logger.error(f"❌ Error checking demand zone entry: {e}")
            return {'action': 'hold', 'confidence': 0.0, 'reason': f'Error: {str(e)}'}
    
    async def _check_supply_zone_exit(self, symbol: str, price: float, zones: Dict, trend: str, position_value: float, position_pct: float) -> Dict:
        """Check if price is in a supply zone for exit"""
        try:
            if position_value <= 0:
                return {'action': 'hold', 'confidence': 0.0, 'reason': 'No position to sell'}
            
            supply_zones = zones.get('supply', [])
            
            # Check if price is in any supply zone
            in_supply_zone = False
            zone_strength = 0
            
            for zone in supply_zones:
                if zone['low'] <= price <= zone['high']:
                    in_supply_zone = True
                    zone_strength = max(zone_strength, zone['strength'])
                    break
            
            if not in_supply_zone:
                return {'action': 'hold', 'confidence': 0.0, 'reason': 'Not in supply zone'}
            
            # Determine sell percentage based on trend
            sell_percentage = self.sell_pct_trending_up if trend == 'up' else self.sell_pct_trending_down
            
            # Calculate confidence
            confidence = 0.7  # Base confidence for supply zone
            if trend == 'down':
                confidence += 0.2  # Higher confidence to sell in downtrend
            
            confidence += (zone_strength - 1) * 0.1
            confidence = min(confidence, 1.0)
            
            return {
                'action': 'sell',
                'confidence': confidence,
                'reason': f'In supply zone, trend: {trend}, sell {sell_percentage:.0%}',
                'zone_type': 'supply',
                'zone_strength': zone_strength,
                'sell_percentage': sell_percentage
            }
            
        except Exception as e:
            logger.error(f"❌ Error checking supply zone exit: {e}")
            return {'action': 'hold', 'confidence': 0.0, 'reason': f'Error: {str(e)}'}
    
    async def calculate_position_size(self, symbol: str, signal: Dict) -> float:
        """Calculate position size for the trade"""
        try:
            if signal['action'] == 'buy':
                # For demand zone entries, calculate based on target position size
                current_position = await self.portfolio_manager.get_position(symbol)
                current_value = 0
                if current_position:
                    current_price = signal.get('price', 0)
                    current_value = current_position.quantity * current_price
                
                # Calculate how much more we need to reach target
                remaining_target = self.position_size_usd - current_value
                remaining_target = max(remaining_target, 0)
                
                # Limit to reasonable increments
                increment_size = self.position_size_usd * 0.2  # 20% increments
                buy_amount = min(remaining_target, increment_size)
                
                return buy_amount
                
            elif signal['action'] == 'sell':
                # For supply zone exits, calculate based on sell percentage
                current_position = await self.portfolio_manager.get_position(symbol)
                if not current_position:
                    return 0
                
                sell_percentage = signal.get('sell_percentage', 0.5)
                sell_quantity = current_position.quantity * sell_percentage
                
                return sell_quantity
            
            return 0
            
        except Exception as e:
            logger.error(f"❌ Error calculating position size for {symbol}: {e}")
            return 0
    
    async def on_trade_executed(self, symbol: str, side: str, quantity: float, price: float):
        """Handle trade execution"""
        try:
            if side in ['buy', 'long']:
                self.successful_entries += 1
                logger.info(f"🟢 S/D Zone BUY: {symbol[:8]}... @ {price:.6f} | Size: ${quantity * price:.2f}")
                
            elif side in ['sell', 'short']:
                self.successful_exits += 1
                logger.info(f"🔴 S/D Zone SELL: {symbol[:8]}... @ {price:.6f} | Size: ${quantity * price:.2f}")
            
            self.trades_count += 1
            
        except Exception as e:
            logger.error(f"❌ Error handling trade execution for {symbol}: {e}")
    
    async def get_strategy_stats(self) -> Dict:
        """Get strategy performance statistics"""
        try:
            entry_success_rate = (self.successful_entries / max(self.trades_count, 1)) * 100
            exit_success_rate = (self.successful_exits / max(self.trades_count, 1)) * 100
            
            return {
                'strategy_name': self.name,
                'total_trades': self.trades_count,
                'successful_entries': self.successful_entries,
                'successful_exits': self.successful_exits,
                'entry_success_rate': entry_success_rate,
                'exit_success_rate': exit_success_rate,
                'active_symbols': len(self.zone_data),
                'parameters': {
                    'position_size_usd': self.position_size_usd,
                    'buffer_pct': self.buffer_pct,
                    'timeframe': self.timeframe,
                    'sell_pct_up': self.sell_pct_trending_up,
                    'sell_pct_down': self.sell_pct_trending_down
                }
            }
            
        except Exception as e:
            logger.error(f"❌ Error getting strategy stats: {e}")
            return {'strategy_name': self.name, 'error': str(e)}
    
    async def update_parameters(self, new_params: Dict):
        """Update strategy parameters"""
        try:
            if 'position_size_usd' in new_params:
                self.position_size_usd = new_params['position_size_usd']
            if 'buffer_pct' in new_params:
                self.buffer_pct = new_params['buffer_pct']
            if 'sell_pct_trending_up' in new_params:
                self.sell_pct_trending_up = new_params['sell_pct_trending_up']
            if 'sell_pct_trending_down' in new_params:
                self.sell_pct_trending_down = new_params['sell_pct_trending_down']
            
            logger.info(f"📊 S/D Zone parameters updated: {new_params}")
            
        except Exception as e:
            logger.error(f"❌ Error updating parameters: {e}")
    
    async def cleanup(self):
        """Cleanup strategy resources"""
        try:
            self.zone_data.clear()
            self.trend_data.clear()
            self.position_tracking.clear()
            
            logger.info("🧹 Supply/Demand Zone strategy cleaned up")
            
        except Exception as e:
            logger.error(f"❌ Error cleaning up S/D Zone strategy: {e}") 