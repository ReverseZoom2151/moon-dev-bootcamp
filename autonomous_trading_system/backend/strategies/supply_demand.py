"""
Supply/Demand Zones Strategy - Based on Day_11_Projects implementation
Identifies and trades based on supply and demand zones
"""

import logging
import pandas as pd
from typing import Dict, Any, Optional, List
from strategies.base_strategy import BaseStrategy, StrategySignal, SignalAction, TechnicalIndicatorMixin

logger = logging.getLogger(__name__)

class SupplyDemandStrategy(BaseStrategy, TechnicalIndicatorMixin):
    """
    Supply/Demand Zones Strategy Implementation
    
    Strategy Logic:
    - Identify supply zones (resistance levels where selling pressure is high)
    - Identify demand zones (support levels where buying pressure is high)
    - Place orders when price approaches these zones
    - Use zone strength and proximity to determine entry signals
    """
    
    def __init__(self, config: Dict[str, Any], market_data_manager, name: str = "SupplyDemand"):
        super().__init__(config, market_data_manager, name)
        
        # Strategy-specific configuration
        self.zone_strength_min = config.get("zone_strength_min", 3)
        self.zone_proximity_pct = config.get("zone_proximity_pct", 0.5)  # 0.5% from zone
        self.lookback_periods = config.get("lookback_periods", 100)
        
        # Minimum data points required
        self.min_data_points = max(self.lookback_periods + 20, 120)
        
        logger.info(f"🔧 Supply/Demand Strategy initialized:")
        logger.info(f"   Zone Strength Min: {self.zone_strength_min}")
        logger.info(f"   Zone Proximity: {self.zone_proximity_pct}%")
        logger.info(f"   Lookback Periods: {self.lookback_periods}")
        logger.info(f"   Symbols: {self.symbols}")
    
    async def _initialize_strategy(self):
        """Initialize strategy-specific components"""
        # Validate configuration
        if self.zone_strength_min <= 0:
            raise ValueError("Zone strength minimum must be positive")
        
        if not 0 < self.zone_proximity_pct < 10:
            raise ValueError("Zone proximity percentage must be between 0 and 10")
        
        if self.lookback_periods < 50:
            raise ValueError("Lookback periods must be at least 50")
        
        logger.info("✅ Supply/Demand strategy validation complete")
    
    async def generate_signal(self) -> Optional[StrategySignal]:
        """Generate trading signal based on supply/demand zones"""
        try:
            # Process each symbol
            for symbol in self.symbols:
                signal = await self._analyze_symbol(symbol)
                if signal and signal.action != SignalAction.HOLD:
                    return signal
            
            return None  # No signals generated
            
        except Exception as e:
            logger.error(f"❌ Error generating signal: {e}", exc_info=True)
            return None
    
    async def _analyze_symbol(self, symbol: str) -> Optional[StrategySignal]:
        """Analyze a specific symbol for supply/demand opportunities"""
        try:
            # Get market data
            data = await self._get_market_data(symbol, limit=self.min_data_points)
            if data is None or len(data) < self.min_data_points:
                logger.warning(f"⚠️ Insufficient data for {symbol}: {len(data) if data is not None else 0} candles")
                return None
            
            # Calculate indicators
            indicators = await self._calculate_indicators(symbol, data)
            if not indicators:
                return None
            
            # Generate signal based on zone proximity
            signal = await self._evaluate_supply_demand(symbol, data, indicators)
            return signal
            
        except Exception as e:
            logger.error(f"❌ Error analyzing {symbol}: {e}")
            return None
    
    async def _calculate_indicators(self, symbol: str, data: Any) -> Dict[str, Any]:
        """Calculate supply and demand zones"""
        try:
            # Convert to DataFrame if needed
            if not isinstance(data, pd.DataFrame):
                df = pd.DataFrame(data)
            else:
                df = data.copy()
            
            # Identify supply and demand zones
            supply_zones = self._identify_supply_zones(df)
            demand_zones = self._identify_demand_zones(df)
            
            # Get current price
            current_price = float(df['close'].iloc[-1])
            
            # Calculate zone strengths and distances
            supply_analysis = self._analyze_zones(supply_zones, current_price, 'supply')
            demand_analysis = self._analyze_zones(demand_zones, current_price, 'demand')
            
            indicators = {
                'current_price': current_price,
                'supply_zones': supply_zones,
                'demand_zones': demand_zones,
                'supply_analysis': supply_analysis,
                'demand_analysis': demand_analysis,
                'nearest_supply': supply_analysis.get('nearest_zone'),
                'nearest_demand': demand_analysis.get('nearest_zone'),
                'supply_distance': supply_analysis.get('nearest_distance'),
                'demand_distance': demand_analysis.get('nearest_distance')
            }
            
            logger.debug(f"📊 {symbol} S/D zones: Supply={len(supply_zones)}, Demand={len(demand_zones)}, "
                        f"Price={current_price:.4f}")
            
            return indicators
            
        except Exception as e:
            logger.error(f"❌ Error calculating indicators for {symbol}: {e}")
            return {}
    
    def _identify_supply_zones(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Identify supply zones (resistance levels)"""
        supply_zones = []
        
        try:
            # Look for swing highs
            highs = df['high'].values
            lows = df['low'].values
            closes = df['close'].values
            
            # Find local maxima (swing highs)
            for i in range(2, len(highs) - 2):
                if (highs[i] > highs[i-1] and highs[i] > highs[i-2] and 
                    highs[i] > highs[i+1] and highs[i] > highs[i+2]):
                    
                    # Calculate zone strength (how many times price tested this level)
                    zone_price = highs[i]
                    zone_range = zone_price * 0.002  # 0.2% range around the level
                    
                    # Count touches within the zone
                    touches = 0
                    for j in range(max(0, i-20), min(len(highs), i+20)):
                        if abs(highs[j] - zone_price) <= zone_range:
                            touches += 1
                    
                    if touches >= self.zone_strength_min:
                        supply_zones.append({
                            'price': zone_price,
                            'strength': touches,
                            'index': i,
                            'type': 'supply'
                        })
            
            # Sort by strength (strongest zones first)
            supply_zones.sort(key=lambda x: x['strength'], reverse=True)
            
        except Exception as e:
            logger.error(f"❌ Error identifying supply zones: {e}")
        
        return supply_zones[:10]  # Return top 10 zones
    
    def _identify_demand_zones(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Identify demand zones (support levels)"""
        demand_zones = []
        
        try:
            # Look for swing lows
            highs = df['high'].values
            lows = df['low'].values
            closes = df['close'].values
            
            # Find local minima (swing lows)
            for i in range(2, len(lows) - 2):
                if (lows[i] < lows[i-1] and lows[i] < lows[i-2] and 
                    lows[i] < lows[i+1] and lows[i] < lows[i+2]):
                    
                    # Calculate zone strength
                    zone_price = lows[i]
                    zone_range = zone_price * 0.002  # 0.2% range around the level
                    
                    # Count touches within the zone
                    touches = 0
                    for j in range(max(0, i-20), min(len(lows), i+20)):
                        if abs(lows[j] - zone_price) <= zone_range:
                            touches += 1
                    
                    if touches >= self.zone_strength_min:
                        demand_zones.append({
                            'price': zone_price,
                            'strength': touches,
                            'index': i,
                            'type': 'demand'
                        })
            
            # Sort by strength (strongest zones first)
            demand_zones.sort(key=lambda x: x['strength'], reverse=True)
            
        except Exception as e:
            logger.error(f"❌ Error identifying demand zones: {e}")
        
        return demand_zones[:10]  # Return top 10 zones
    
    def _analyze_zones(self, zones: List[Dict[str, Any]], current_price: float, zone_type: str) -> Dict[str, Any]:
        """Analyze zones and find the nearest one"""
        if not zones:
            return {'nearest_zone': None, 'nearest_distance': float('inf')}
        
        # Find nearest zone
        nearest_zone = None
        nearest_distance = float('inf')
        
        for zone in zones:
            distance = abs(zone['price'] - current_price) / current_price * 100  # Percentage distance
            if distance < nearest_distance:
                nearest_distance = distance
                nearest_zone = zone
        
        return {
            'nearest_zone': nearest_zone,
            'nearest_distance': nearest_distance,
            'total_zones': len(zones),
            'avg_strength': sum(z['strength'] for z in zones) / len(zones) if zones else 0
        }
    
    async def _evaluate_supply_demand(self, symbol: str, data: Any, indicators: Dict[str, Any]) -> Optional[StrategySignal]:
        """Evaluate supply/demand conditions and generate signal"""
        try:
            current_price = indicators['current_price']
            supply_distance = indicators.get('supply_distance', float('inf'))
            demand_distance = indicators.get('demand_distance', float('inf'))
            nearest_supply = indicators.get('nearest_supply')
            nearest_demand = indicators.get('nearest_demand')
            
            # Determine action based on proximity to zones
            action = SignalAction.HOLD
            confidence = 0.0
            target_zone = None
            
            # Check if price is near a demand zone (buy signal)
            if (demand_distance < self.zone_proximity_pct and 
                nearest_demand and 
                demand_distance < supply_distance):
                
                action = SignalAction.BUY
                target_zone = nearest_demand
                # Confidence based on zone strength and proximity
                strength_factor = min(nearest_demand['strength'] / 10, 1.0)
                proximity_factor = max(0, 1 - (demand_distance / self.zone_proximity_pct))
                confidence = (strength_factor + proximity_factor) / 2
            
            # Check if price is near a supply zone (sell signal)
            elif (supply_distance < self.zone_proximity_pct and 
                  nearest_supply and 
                  supply_distance < demand_distance):
                
                action = SignalAction.SELL
                target_zone = nearest_supply
                # Confidence based on zone strength and proximity
                strength_factor = min(nearest_supply['strength'] / 10, 1.0)
                proximity_factor = max(0, 1 - (supply_distance / self.zone_proximity_pct))
                confidence = (strength_factor + proximity_factor) / 2
            
            # Only generate signal if confidence is above threshold
            min_confidence = 0.3
            if confidence < min_confidence or not target_zone:
                return None
            
            # Create metadata with strategy details
            metadata = {
                'target_zone': target_zone,
                'supply_zones_count': len(indicators.get('supply_zones', [])),
                'demand_zones_count': len(indicators.get('demand_zones', [])),
                'supply_distance': supply_distance,
                'demand_distance': demand_distance,
                'zone_proximity_threshold': self.zone_proximity_pct,
                'strategy_type': 'supply_demand'
            }
            
            # Create and return signal
            if action != SignalAction.HOLD:
                signal = self._create_signal(
                    symbol=symbol,
                    action=action,
                    price=current_price,
                    confidence=confidence,
                    metadata=metadata
                )
                
                logger.info(f"🎯 Supply/Demand Signal: {action.value} {symbol} @ {current_price:.4f} "
                           f"(confidence: {confidence:.2f}, zone: {target_zone['price']:.4f}, "
                           f"strength: {target_zone['strength']})")
                
                return signal
            
            return None
            
        except Exception as e:
            logger.error(f"❌ Error evaluating supply/demand for {symbol}: {e}")
            return None
    
    def _calculate_stop_loss(self, entry_price: float, action: SignalAction) -> float:
        """Calculate stop loss based on zone levels"""
        # More conservative stop loss for zone trading
        stop_loss_pct = 0.015  # 1.5% stop loss
        
        if action == SignalAction.BUY:
            return entry_price * (1 - stop_loss_pct)
        elif action == SignalAction.SELL:
            return entry_price * (1 + stop_loss_pct)
        
        return entry_price
    
    def _calculate_take_profit(self, entry_price: float, action: SignalAction) -> float:
        """Calculate take profit for zone trading"""
        # Target the opposite zone or reasonable profit
        take_profit_pct = 0.03  # 3% take profit
        
        if action == SignalAction.BUY:
            return entry_price * (1 + take_profit_pct)
        elif action == SignalAction.SELL:
            return entry_price * (1 - take_profit_pct)
        
        return entry_price
    
    async def _get_zone_summary(self, symbol: str) -> Dict[str, Any]:
        """Get a summary of current zones for monitoring"""
        try:
            data = await self._get_market_data(symbol, limit=self.min_data_points)
            if data is None:
                return {}
            
            indicators = await self._calculate_indicators(symbol, data)
            
            return {
                'symbol': symbol,
                'current_price': indicators.get('current_price'),
                'supply_zones': len(indicators.get('supply_zones', [])),
                'demand_zones': len(indicators.get('demand_zones', [])),
                'nearest_supply_distance': indicators.get('supply_distance'),
                'nearest_demand_distance': indicators.get('demand_distance')
            }
            
        except Exception as e:
            logger.error(f"❌ Error getting zone summary for {symbol}: {e}")
            return {}
    
    def get_strategy_info(self) -> Dict[str, Any]:
        """Get detailed strategy information"""
        base_info = self.get_status()
        
        strategy_specific = {
            'strategy_type': 'supply_demand',
            'parameters': {
                'zone_strength_min': self.zone_strength_min,
                'zone_proximity_pct': self.zone_proximity_pct,
                'lookback_periods': self.lookback_periods
            },
            'description': 'Identifies and trades based on supply and demand zones'
        }
        
        base_info.update(strategy_specific)
        return base_info 