# hyperliquid_rrs_bot_service.py
"""
Hyperliquid RRS Trading Bot Service
Implements Day 37 RRS Bot functionality for automated trading based on RRS analysis

This service provides automated trading capabilities using Hyperliquid RRS analysis,
including position management, risk controls, and dynamic leverage adjustment.
"""

import asyncio
import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from core.config import get_settings
from services.hyperliquid_rrs_analysis_service import hyperliquid_rrs_service

logger = logging.getLogger(__name__)
settings = get_settings()


@dataclass
class Position:
    """Represents a trading position"""
    symbol: str
    side: str  # 'long' or 'short'
    size: float
    entry_price: float
    current_price: float
    pnl: float
    pnl_percent: float
    leverage: int
    timestamp: datetime


@dataclass
class TradingState:
    """Represents the current trading state"""
    positions: List[Position]
    total_pnl: float
    total_pnl_percent: float
    account_value: float
    available_balance: float
    last_rrs_update: Optional[datetime]
    rrs_cache: Dict[str, float]
    is_running: bool
    last_trade_time: Optional[datetime]


class HyperliquidRRSBotService:
    """Main Hyperliquid RRS trading bot service"""
    
    def __init__(self):
        self.is_running = False
        self.trading_state = TradingState(
            positions=[],
            total_pnl=0.0,
            total_pnl_percent=0.0,
            account_value=0.0,
            available_balance=0.0,
            last_rrs_update=None,
            rrs_cache={},
            is_running=False,
            last_trade_time=None
        )
        
        # Bot configuration from settings
        self.usdc_size = settings.HYPERLIQUID_RRS_BOT_USDC_SIZE
        self.leverage = settings.HYPERLIQUID_RRS_BOT_LEVERAGE
        self.sleep_seconds = settings.HYPERLIQUID_RRS_BOT_SLEEP_SECONDS
        self.cache_timeout = settings.HYPERLIQUID_RRS_BOT_CACHE_TIMEOUT
        self.take_profit = settings.HYPERLIQUID_RRS_BOT_TAKE_PROFIT
        self.stop_loss = settings.HYPERLIQUID_RRS_BOT_STOP_LOSS
        self.auto_adjust_leverage = settings.HYPERLIQUID_RRS_BOT_AUTO_ADJUST_LEVERAGE
        self.limit_order_buffer = settings.HYPERLIQUID_RRS_BOT_LIMIT_ORDER_BUFFER
        
        # Ensure bot data directory exists
        self._ensure_bot_directory()
    
    def _ensure_bot_directory(self):
        """Create bot data directory if it doesn't exist"""
        try:
            Path(settings.HYPERLIQUID_RRS_BOT_DATA_DIR).mkdir(parents=True, exist_ok=True)
            logger.info("Ensured bot data directory exists")
        except OSError as e:
            logger.error(f"Error creating bot data directory: {e}")
    
    def _save_trading_state(self):
        """Save current trading state to file"""
        try:
            state_file = Path(settings.HYPERLIQUID_RRS_BOT_DATA_DIR) / "trading_state.json"
            state_data = {
                "positions": [asdict(pos) for pos in self.trading_state.positions],
                "total_pnl": self.trading_state.total_pnl,
                "total_pnl_percent": self.trading_state.total_pnl_percent,
                "account_value": self.trading_state.account_value,
                "available_balance": self.trading_state.available_balance,
                "last_rrs_update": self.trading_state.last_rrs_update.isoformat() if self.trading_state.last_rrs_update else None,
                "rrs_cache": self.trading_state.rrs_cache,
                "is_running": self.trading_state.is_running,
                "last_trade_time": self.trading_state.last_trade_time.isoformat() if self.trading_state.last_trade_time else None,
                "timestamp": datetime.utcnow().isoformat()
            }
            
            with open(state_file, 'w') as f:
                json.dump(state_data, f, indent=2)
            
            logger.debug("Trading state saved")
        except Exception as e:
            logger.error(f"Error saving trading state: {e}")
    
    def _load_trading_state(self):
        """Load trading state from file"""
        try:
            state_file = Path(settings.HYPERLIQUID_RRS_BOT_DATA_DIR) / "trading_state.json"
            if not state_file.exists():
                logger.info("No existing trading state found")
                return
            
            with open(state_file, 'r') as f:
                state_data = json.load(f)
            
            # Reconstruct positions
            positions = []
            for pos_data in state_data.get("positions", []):
                pos_data["timestamp"] = datetime.fromisoformat(pos_data["timestamp"])
                positions.append(Position(**pos_data))
            
            self.trading_state.positions = positions
            self.trading_state.total_pnl = state_data.get("total_pnl", 0.0)
            self.trading_state.total_pnl_percent = state_data.get("total_pnl_percent", 0.0)
            self.trading_state.account_value = state_data.get("account_value", 0.0)
            self.trading_state.available_balance = state_data.get("available_balance", 0.0)
            self.trading_state.rrs_cache = state_data.get("rrs_cache", {})
            
            if state_data.get("last_rrs_update"):
                self.trading_state.last_rrs_update = datetime.fromisoformat(state_data["last_rrs_update"])
            
            if state_data.get("last_trade_time"):
                self.trading_state.last_trade_time = datetime.fromisoformat(state_data["last_trade_time"])
            
            logger.info("Trading state loaded successfully")
        except Exception as e:
            logger.error(f"Error loading trading state: {e}")
    
    async def _get_fresh_rrs_data(self) -> Dict[str, float]:
        """Get fresh RRS data from analysis service"""
        try:
            # Check if we need to refresh RRS cache
            now = datetime.utcnow()
            if (self.trading_state.last_rrs_update and 
                (now - self.trading_state.last_rrs_update).total_seconds() < self.cache_timeout * 60):
                logger.debug("Using cached RRS data")
                return self.trading_state.rrs_cache
            
            logger.info("Fetching fresh RRS data...")
            
            # Get consolidated rankings from RRS service
            rankings_data = hyperliquid_rrs_service.get_rankings()
            
            if "error" in rankings_data:
                logger.error(f"Error getting RRS rankings: {rankings_data['error']}")
                return self.trading_state.rrs_cache  # Return cached data on error
            
            # Extract RRS scores
            rrs_scores = {}
            if "rankings" in rankings_data:
                for ranking in rankings_data["rankings"]:
                    symbol = ranking.get("symbol")
                    rrs_score = ranking.get("rrs")
                    if symbol and rrs_score is not None:
                        rrs_scores[symbol] = float(rrs_score)
            
            # Update cache
            self.trading_state.rrs_cache = rrs_scores
            self.trading_state.last_rrs_update = now
            
            logger.info(f"Updated RRS cache with {len(rrs_scores)} symbols")
            return rrs_scores
            
        except Exception as e:
            logger.error(f"Error getting fresh RRS data: {e}")
            return self.trading_state.rrs_cache
    
    def _calculate_dynamic_leverage(self, symbol: str, rrs_score: float) -> int:
        """Calculate dynamic leverage based on RRS score"""
        if not self.auto_adjust_leverage:
            return self.leverage
        
        try:
            # Adjust leverage based on RRS strength
            if rrs_score >= settings.HYPERLIQUID_RRS_EXCEPTIONAL_THRESHOLD:
                return min(5, self.leverage + 2)  # Higher leverage for exceptional strength
            elif rrs_score >= settings.HYPERLIQUID_RRS_STRONG_THRESHOLD:
                return min(4, self.leverage + 1)  # Slightly higher leverage for strong
            elif rrs_score >= settings.HYPERLIQUID_RRS_MODERATE_THRESHOLD:
                return self.leverage  # Default leverage for moderate
            elif rrs_score >= settings.HYPERLIQUID_RRS_WEAK_THRESHOLD:
                return max(1, self.leverage - 1)  # Lower leverage for weak
            else:
                return max(1, self.leverage - 2)  # Minimal leverage for underperforming
        except Exception as e:
            logger.error(f"Error calculating dynamic leverage for {symbol}: {e}")
            return self.leverage
    
    async def _simulate_get_account_info(self) -> Dict[str, Any]:
        """Simulate getting account information (placeholder for actual Hyperliquid API)"""
        # This would be replaced with actual Hyperliquid API calls
        return {
            "account_value": 10000.0,
            "available_balance": 8000.0,
            "positions": []
        }
    
    async def _simulate_get_current_price(self, symbol: str) -> float:
        """Simulate getting current price (placeholder for actual Hyperliquid API)"""
        # This would be replaced with actual Hyperliquid API calls
        # For now, return a simulated price
        base_prices = {
            "BTC": 45000.0,
            "ETH": 3000.0,
            "SOL": 100.0,
            "WIF": 2.5,
            "ADA": 0.5,
            "DOT": 7.0,
            "LINK": 15.0,
            "XRP": 0.6,
            "LTC": 70.0,
            "BCH": 250.0
        }
        
        base_price = base_prices.get(symbol, 1.0)
        # Add some random variation
        import random
        variation = random.uniform(-0.02, 0.02)  # ±2% variation
        return base_price * (1 + variation)
    
    async def _simulate_place_order(self, symbol: str, side: str, size: float, price: float, leverage: int) -> Dict[str, Any]:
        """Simulate placing an order (placeholder for actual Hyperliquid API)"""
        # This would be replaced with actual Hyperliquid API calls
        order_id = f"order_{symbol}_{int(time.time())}"
        
        logger.info(f"SIMULATED ORDER: {side.upper()} {size} {symbol} at ${price:.4f} with {leverage}x leverage")
        
        return {
            "order_id": order_id,
            "symbol": symbol,
            "side": side,
            "size": size,
            "price": price,
            "leverage": leverage,
            "status": "filled",
            "timestamp": datetime.utcnow()
        }
    
    async def _simulate_close_position(self, symbol: str) -> Dict[str, Any]:
        """Simulate closing a position (placeholder for actual Hyperliquid API)"""
        # This would be replaced with actual Hyperliquid API calls
        logger.info(f"SIMULATED CLOSE: Closing position for {symbol}")
        
        return {
            "symbol": symbol,
            "status": "closed",
            "timestamp": datetime.utcnow()
        }
    
    async def _update_positions(self):
        """Update position information"""
        try:
            # Get account info (simulated)
            account_info = await self._simulate_get_account_info()
            
            self.trading_state.account_value = account_info["account_value"]
            self.trading_state.available_balance = account_info["available_balance"]
            
            # Update existing positions with current prices
            updated_positions = []
            total_pnl = 0.0
            
            for position in self.trading_state.positions:
                current_price = await self._simulate_get_current_price(position.symbol)
                
                # Calculate PnL
                if position.side == "long":
                    pnl = (current_price - position.entry_price) * position.size
                else:  # short
                    pnl = (position.entry_price - current_price) * position.size
                
                pnl_percent = (pnl / (position.entry_price * position.size)) * 100
                
                updated_position = Position(
                    symbol=position.symbol,
                    side=position.side,
                    size=position.size,
                    entry_price=position.entry_price,
                    current_price=current_price,
                    pnl=pnl,
                    pnl_percent=pnl_percent,
                    leverage=position.leverage,
                    timestamp=position.timestamp
                )
                
                updated_positions.append(updated_position)
                total_pnl += pnl
            
            self.trading_state.positions = updated_positions
            self.trading_state.total_pnl = total_pnl
            
            if self.trading_state.account_value > 0:
                self.trading_state.total_pnl_percent = (total_pnl / self.trading_state.account_value) * 100
            
            logger.debug(f"Updated {len(updated_positions)} positions. Total PnL: ${total_pnl:.2f}")
            
        except Exception as e:
            logger.error(f"Error updating positions: {e}")
    
    async def _check_exit_conditions(self):
        """Check if any positions should be closed based on PnL thresholds"""
        try:
            positions_to_close = []
            
            # Check individual position exit conditions
            for position in self.trading_state.positions:
                should_close = False
                reason = ""
                
                # Take profit check
                if position.pnl_percent >= self.take_profit:
                    should_close = True
                    reason = f"Take profit hit: {position.pnl_percent:.2f}%"
                
                # Stop loss check
                elif position.pnl_percent <= self.stop_loss:
                    should_close = True
                    reason = f"Stop loss hit: {position.pnl_percent:.2f}%"
                
                if should_close:
                    positions_to_close.append((position, reason))
            
            # Check overall portfolio exit conditions
            if self.trading_state.total_pnl_percent >= self.take_profit:
                logger.info(f"Portfolio take profit hit: {self.trading_state.total_pnl_percent:.2f}%")
                positions_to_close = [(pos, "Portfolio take profit") for pos in self.trading_state.positions]
            
            elif self.trading_state.total_pnl_percent <= self.stop_loss:
                logger.info(f"Portfolio stop loss hit: {self.trading_state.total_pnl_percent:.2f}%")
                positions_to_close = [(pos, "Portfolio stop loss") for pos in self.trading_state.positions]
            
            # Close positions
            for position, reason in positions_to_close:
                await self._close_position(position, reason)
            
        except Exception as e:
            logger.error(f"Error checking exit conditions: {e}")
    
    async def _close_position(self, position: Position, reason: str):
        """Close a specific position"""
        try:
            logger.info(f"Closing position {position.symbol} ({position.side}): {reason}")
            
            # Simulate closing the position
            close_result = await self._simulate_close_position(position.symbol)
            
            if close_result["status"] == "closed":
                # Remove from positions list
                self.trading_state.positions = [p for p in self.trading_state.positions if p.symbol != position.symbol]
                self.trading_state.last_trade_time = datetime.utcnow()
                
                logger.info(f"Successfully closed {position.symbol} position. PnL: ${position.pnl:.2f} ({position.pnl_percent:.2f}%)")
            else:
                logger.error(f"Failed to close position for {position.symbol}")
                
        except Exception as e:
            logger.error(f"Error closing position {position.symbol}: {e}")
    
    async def _execute_rrs_strategy(self, rrs_data: Dict[str, float]):
        """Execute trading strategy based on RRS data"""
        try:
            if not rrs_data:
                logger.warning("No RRS data available for trading")
                return
            
            # Sort symbols by RRS score (highest first)
            sorted_symbols = sorted(rrs_data.items(), key=lambda x: x[1], reverse=True)
            
            # Get top performers
            top_symbols = sorted_symbols[:3]  # Top 3 performers
            bottom_symbols = sorted_symbols[-3:]  # Bottom 3 performers
            
            logger.info("Top RRS performers:")
            for symbol, rrs_score in top_symbols:
                interpretation = hyperliquid_rrs_service.interpret_score(rrs_score)
                logger.info(f"  {symbol}: {rrs_score:.4f} ({interpretation['interpretation']})")
            
            # Check if we should open new positions
            current_symbols = {pos.symbol for pos in self.trading_state.positions}
            
            # Long the strongest symbols
            for symbol, rrs_score in top_symbols:
                if symbol in current_symbols:
                    continue  # Already have position
                
                if rrs_score >= settings.HYPERLIQUID_RRS_STRONG_THRESHOLD:
                    await self._open_long_position(symbol, rrs_score)
            
            # Short the weakest symbols (if RRS is significantly negative)
            for symbol, rrs_score in bottom_symbols:
                if symbol in current_symbols:
                    continue  # Already have position
                
                if rrs_score < -settings.HYPERLIQUID_RRS_MODERATE_THRESHOLD:
                    await self._open_short_position(symbol, rrs_score)
            
        except Exception as e:
            logger.error(f"Error executing RRS strategy: {e}")
    
    async def _open_long_position(self, symbol: str, rrs_score: float):
        """Open a long position"""
        try:
            current_price = await self._simulate_get_current_price(symbol)
            leverage = self._calculate_dynamic_leverage(symbol, rrs_score)
            
            # Calculate position size
            position_value = self.usdc_size * leverage
            size = position_value / current_price
            
            # Place buy order slightly above market price
            order_price = current_price * (1 + self.limit_order_buffer)
            
            order_result = await self._simulate_place_order(
                symbol=symbol,
                side="long",
                size=size,
                price=order_price,
                leverage=leverage
            )
            
            if order_result["status"] == "filled":
                # Add to positions
                new_position = Position(
                    symbol=symbol,
                    side="long",
                    size=size,
                    entry_price=order_result["price"],
                    current_price=current_price,
                    pnl=0.0,
                    pnl_percent=0.0,
                    leverage=leverage,
                    timestamp=datetime.utcnow()
                )
                
                self.trading_state.positions.append(new_position)
                self.trading_state.last_trade_time = datetime.utcnow()
                
                logger.info(f"Opened LONG position: {symbol} at ${order_result['price']:.4f} with {leverage}x leverage")
            
        except Exception as e:
            logger.error(f"Error opening long position for {symbol}: {e}")
    
    async def _open_short_position(self, symbol: str, rrs_score: float):
        """Open a short position"""
        try:
            current_price = await self._simulate_get_current_price(symbol)
            leverage = self._calculate_dynamic_leverage(symbol, abs(rrs_score))
            
            # Calculate position size
            position_value = self.usdc_size * leverage
            size = position_value / current_price
            
            # Place sell order slightly below market price
            order_price = current_price * (1 - self.limit_order_buffer)
            
            order_result = await self._simulate_place_order(
                symbol=symbol,
                side="short",
                size=size,
                price=order_price,
                leverage=leverage
            )
            
            if order_result["status"] == "filled":
                # Add to positions
                new_position = Position(
                    symbol=symbol,
                    side="short",
                    size=size,
                    entry_price=order_result["price"],
                    current_price=current_price,
                    pnl=0.0,
                    pnl_percent=0.0,
                    leverage=leverage,
                    timestamp=datetime.utcnow()
                )
                
                self.trading_state.positions.append(new_position)
                self.trading_state.last_trade_time = datetime.utcnow()
                
                logger.info(f"Opened SHORT position: {symbol} at ${order_result['price']:.4f} with {leverage}x leverage")
            
        except Exception as e:
            logger.error(f"Error opening short position for {symbol}: {e}")
    
    async def _trading_loop(self):
        """Main trading loop"""
        logger.info("🚀 Starting Hyperliquid RRS Trading Bot...")
        
        while self.is_running:
            try:
                logger.info("=== Trading Loop Iteration ===")
                
                # Update positions
                await self._update_positions()
                
                # Check exit conditions
                await self._check_exit_conditions()
                
                # Get fresh RRS data
                rrs_data = await self._get_fresh_rrs_data()
                
                # Execute strategy
                await self._execute_rrs_strategy(rrs_data)
                
                # Save state
                self._save_trading_state()
                
                # Log current status
                logger.info(f"Positions: {len(self.trading_state.positions)}")
                logger.info(f"Total PnL: ${self.trading_state.total_pnl:.2f} ({self.trading_state.total_pnl_percent:.2f}%)")
                logger.info(f"Account Value: ${self.trading_state.account_value:.2f}")
                
                # Sleep before next iteration
                logger.info(f"Sleeping for {self.sleep_seconds} seconds...")
                await asyncio.sleep(self.sleep_seconds)
                
            except Exception as e:
                logger.error(f"Error in trading loop: {e}")
                await asyncio.sleep(self.sleep_seconds)
        
        logger.info("🛑 Hyperliquid RRS Trading Bot stopped")
    
    async def start(self):
        """Start the trading bot"""
        if self.is_running:
            logger.warning("Trading bot is already running")
            return
        
        if not settings.ENABLE_HYPERLIQUID_RRS_BOT:
            logger.warning("Hyperliquid RRS bot is disabled in settings")
            return
        
        logger.info("Starting Hyperliquid RRS Trading Bot...")
        
        # Load previous state
        self._load_trading_state()
        
        self.is_running = True
        self.trading_state.is_running = True
        
        # Start trading loop
        await self._trading_loop()
    
    async def stop(self):
        """Stop the trading bot"""
        logger.info("Stopping Hyperliquid RRS Trading Bot...")
        
        self.is_running = False
        self.trading_state.is_running = False
        
        # Save final state
        self._save_trading_state()
        
        logger.info("Hyperliquid RRS Trading Bot stopped")
    
    def get_status(self) -> Dict[str, Any]:
        """Get bot status"""
        return {
            "service": "Hyperliquid RRS Trading Bot",
            "enabled": settings.ENABLE_HYPERLIQUID_RRS_BOT,
            "is_running": self.is_running,
            "positions_count": len(self.trading_state.positions),
            "total_pnl": self.trading_state.total_pnl,
            "total_pnl_percent": self.trading_state.total_pnl_percent,
            "account_value": self.trading_state.account_value,
            "available_balance": self.trading_state.available_balance,
            "last_rrs_update": self.trading_state.last_rrs_update.isoformat() if self.trading_state.last_rrs_update else None,
            "last_trade_time": self.trading_state.last_trade_time.isoformat() if self.trading_state.last_trade_time else None,
            "rrs_cache_size": len(self.trading_state.rrs_cache),
            "configuration": {
                "usdc_size": self.usdc_size,
                "leverage": self.leverage,
                "sleep_seconds": self.sleep_seconds,
                "cache_timeout": self.cache_timeout,
                "take_profit": self.take_profit,
                "stop_loss": self.stop_loss,
                "auto_adjust_leverage": self.auto_adjust_leverage,
                "limit_order_buffer": self.limit_order_buffer
            }
        }
    
    def get_positions(self) -> List[Dict[str, Any]]:
        """Get current positions"""
        return [asdict(pos) for pos in self.trading_state.positions]
    
    def get_trading_state(self) -> Dict[str, Any]:
        """Get complete trading state"""
        return {
            "positions": [asdict(pos) for pos in self.trading_state.positions],
            "total_pnl": self.trading_state.total_pnl,
            "total_pnl_percent": self.trading_state.total_pnl_percent,
            "account_value": self.trading_state.account_value,
            "available_balance": self.trading_state.available_balance,
            "last_rrs_update": self.trading_state.last_rrs_update.isoformat() if self.trading_state.last_rrs_update else None,
            "rrs_cache": self.trading_state.rrs_cache,
            "is_running": self.trading_state.is_running,
            "last_trade_time": self.trading_state.last_trade_time.isoformat() if self.trading_state.last_trade_time else None
        }
    
    async def force_rrs_update(self):
        """Force update of RRS data"""
        self.trading_state.last_rrs_update = None  # Clear cache
        return await self._get_fresh_rrs_data()
    
    async def close_all_positions(self, reason: str = "Manual close"):
        """Close all open positions"""
        positions_to_close = self.trading_state.positions.copy()
        
        for position in positions_to_close:
            await self._close_position(position, reason)
        
        return len(positions_to_close)


# Global service instance
hyperliquid_rrs_bot_service = HyperliquidRRSBotService() 