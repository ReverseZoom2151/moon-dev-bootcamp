"""
Portfolio Manager - Manages trading positions, executions, and portfolio state
"""

import asyncio
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum
from core.config import Settings
from strategies.base_strategy import StrategySignal

logger = logging.getLogger(__name__)

class OrderStatus(Enum):
    PENDING = "pending"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


@dataclass
class Position:
    """Trading position"""
    symbol: str
    side: str  # "long" or "short"
    size: float
    entry_price: float
    current_price: float
    pnl: float
    pnl_pct: float
    timestamp: datetime
    strategy: str


@dataclass
class Trade:
    """Executed trade"""
    id: str
    symbol: str
    side: str
    size: float
    price: float
    timestamp: datetime
    strategy: str
    pnl: float = 0.0
    fees: float = 0.0


@dataclass
class ExecutionResult:
    """Result of trade execution"""
    success: bool
    trade_id: Optional[str] = None
    error: Optional[str] = None
    executed_price: Optional[float] = None
    executed_size: Optional[float] = None


class PortfolioManager:
    """
    Manages trading portfolio, positions, and executions
    """
    
    def __init__(self, settings: Settings, risk_manager):
        self.settings = settings
        self.risk_manager = risk_manager
        
        # Portfolio state
        self.positions: Dict[str, Position] = {}
        self.trades: List[Trade] = []
        self.orders: Dict[str, Dict] = {}
        
        # Portfolio metrics
        self.total_value = 100000.0  # Starting capital
        self.available_balance = 100000.0
        self.unrealized_pnl = 0.0
        self.realized_pnl = 0.0
        
        # Monitoring
        self.is_running = False
        self.last_update = datetime.utcnow()
        
        logger.info("🔧 Portfolio Manager initialized")
    
    async def start_monitoring(self):
        """Start portfolio monitoring"""
        self.is_running = True
        asyncio.create_task(self._monitoring_loop())
        logger.info("✅ Portfolio monitoring started")
    
    async def stop(self):
        """Stop portfolio manager"""
        self.is_running = False
        logger.info("✅ Portfolio Manager stopped")
    
    async def _monitoring_loop(self):
        """Main monitoring loop"""
        while self.is_running:
            try:
                await self._update_positions()
                await self._update_portfolio_metrics()
                await asyncio.sleep(10)  # Update every 10 seconds
            except Exception as e:
                logger.error(f"❌ Error in portfolio monitoring: {e}")
                await asyncio.sleep(30)
    
    async def execute_trade(self, signal: StrategySignal, position_size: float) -> ExecutionResult:
        """Execute a trading signal"""
        try:
            # Validate the trade
            if not await self._validate_trade(signal, position_size):
                return ExecutionResult(success=False, error="Trade validation failed")
            
            # Simulate trade execution
            trade_id = f"trade_{datetime.utcnow().timestamp()}"
            executed_price = signal.price * (1 + (0.001 if signal.action.value == "BUY" else -0.001))  # Small slippage
            
            # Create trade record
            trade = Trade(
                id=trade_id,
                symbol=signal.symbol,
                side="long" if signal.action.value == "BUY" else "short",
                size=position_size,
                price=executed_price,
                timestamp=datetime.utcnow(),
                strategy=signal.strategy_name,
                fees=position_size * executed_price * 0.001  # 0.1% fee
            )
            
            self.trades.append(trade)
            
            # Update or create position
            await self._update_position_from_trade(trade)
            
            # Update available balance
            trade_value = position_size * executed_price
            self.available_balance -= trade_value + trade.fees
            
            logger.info(f"✅ Executed trade: {trade.side} {trade.size} {trade.symbol} @ {trade.price}")
            
            return ExecutionResult(
                success=True,
                trade_id=trade_id,
                executed_price=executed_price,
                executed_size=position_size
            )
            
        except Exception as e:
            logger.error(f"❌ Error executing trade: {e}")
            return ExecutionResult(success=False, error=str(e))
    
    async def _validate_trade(self, signal: StrategySignal, position_size: float) -> bool:
        """Validate if trade can be executed"""
        try:
            # Check available balance
            trade_value = position_size * signal.price
            if trade_value > self.available_balance:
                logger.warning(f"⚠️ Insufficient balance: need {trade_value}, have {self.available_balance}")
                return False
            
            # Check position limits
            if signal.symbol in self.positions:
                existing_position = self.positions[signal.symbol]
                # Don't allow opposing positions for now
                if ((signal.action.value == "BUY" and existing_position.side == "short") or
                    (signal.action.value == "SELL" and existing_position.side == "long")):
                    logger.warning(f"⚠️ Cannot open opposing position for {signal.symbol}")
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Error validating trade: {e}")
            return False
    
    async def _update_position_from_trade(self, trade: Trade):
        """Update position based on executed trade"""
        try:
            if trade.symbol in self.positions:
                # Update existing position
                position = self.positions[trade.symbol]
                
                if position.side == trade.side:
                    # Add to position
                    total_size = position.size + trade.size
                    weighted_price = ((position.size * position.entry_price) + 
                                    (trade.size * trade.price)) / total_size
                    position.size = total_size
                    position.entry_price = weighted_price
                else:
                    # Reduce or close position
                    if trade.size >= position.size:
                        # Close position
                        del self.positions[trade.symbol]
                    else:
                        # Reduce position
                        position.size -= trade.size
            else:
                # Create new position
                position = Position(
                    symbol=trade.symbol,
                    side=trade.side,
                    size=trade.size,
                    entry_price=trade.price,
                    current_price=trade.price,
                    pnl=0.0,
                    pnl_pct=0.0,
                    timestamp=trade.timestamp,
                    strategy=trade.strategy
                )
                self.positions[trade.symbol] = position
                
        except Exception as e:
            logger.error(f"❌ Error updating position: {e}")
    
    async def _update_positions(self):
        """Update current prices and PnL for all positions"""
        try:
            for symbol, position in self.positions.items():
                # Simulate price updates (in real implementation, get from market data)
                # For now, add small random movement
                import random
                price_change = random.uniform(-0.02, 0.02)  # ±2% random movement
                position.current_price = position.entry_price * (1 + price_change)
                
                # Calculate PnL
                if position.side == "long":
                    position.pnl = (position.current_price - position.entry_price) * position.size
                else:
                    position.pnl = (position.entry_price - position.current_price) * position.size
                
                position.pnl_pct = (position.pnl / (position.entry_price * position.size)) * 100
                
        except Exception as e:
            logger.error(f"❌ Error updating positions: {e}")
    
    async def _update_portfolio_metrics(self):
        """Update overall portfolio metrics"""
        try:
            # Calculate unrealized PnL
            self.unrealized_pnl = sum(pos.pnl for pos in self.positions.values())
            
            # Calculate realized PnL from closed trades
            self.realized_pnl = sum(trade.pnl for trade in self.trades)
            
            # Update total value
            self.total_value = self.available_balance + self.unrealized_pnl
            
            self.last_update = datetime.utcnow()
            
        except Exception as e:
            logger.error(f"❌ Error updating portfolio metrics: {e}")
    
    async def can_execute_trade(self, signal: StrategySignal) -> bool:
        """Check if trade can be executed"""
        return await self._validate_trade(signal, self.settings.DEFAULT_POSITION_SIZE)
    
    async def get_open_positions(self) -> List[Position]:
        """Get all open positions"""
        return list(self.positions.values())
    
    async def get_position(self, symbol: str) -> Optional[Position]:
        """Get position for specific symbol"""
        return self.positions.get(symbol)
    
    async def get_total_value(self) -> float:
        """Get total portfolio value"""
        return self.total_value
    
    async def get_daily_pnl(self) -> float:
        """Get daily PnL"""
        # Calculate PnL for today's trades
        today = datetime.utcnow().date()
        daily_trades = [trade for trade in self.trades if trade.timestamp.date() == today]
        return sum(trade.pnl for trade in daily_trades)
    
    async def get_strategy_trades(self, strategy_name: str) -> List[Trade]:
        """Get trades for specific strategy"""
        return [trade for trade in self.trades if trade.strategy == strategy_name]
    
    async def close_position(self, symbol: str) -> bool:
        """Close position for symbol"""
        try:
            if symbol not in self.positions:
                return False
            
            position = self.positions[symbol]
            
            # Create closing trade
            trade = Trade(
                id=f"close_{datetime.utcnow().timestamp()}",
                symbol=symbol,
                side="short" if position.side == "long" else "long",
                size=position.size,
                price=position.current_price,
                timestamp=datetime.utcnow(),
                strategy="manual_close",
                pnl=position.pnl,
                fees=position.size * position.current_price * 0.001
            )
            
            self.trades.append(trade)
            
            # Update balance
            trade_value = position.size * position.current_price
            self.available_balance += trade_value - trade.fees + position.pnl
            
            # Remove position
            del self.positions[symbol]
            
            logger.info(f"✅ Closed position: {symbol}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error closing position {symbol}: {e}")
            return False
    
    async def close_all_positions(self):
        """Close all open positions"""
        symbols = list(self.positions.keys())
        for symbol in symbols:
            await self.close_position(symbol)
    
    async def cancel_all_orders(self):
        """Cancel all pending orders"""
        # In a real implementation, this would cancel orders on exchanges
        self.orders.clear()
        logger.info("✅ Cancelled all orders")
    
    async def get_portfolio_summary(self) -> Dict[str, Any]:
        """Get portfolio summary"""
        return {
            'total_value': self.total_value,
            'available_balance': self.available_balance,
            'unrealized_pnl': self.unrealized_pnl,
            'realized_pnl': self.realized_pnl,
            'open_positions': len(self.positions),
            'total_trades': len(self.trades),
            'last_update': self.last_update.isoformat()
        } 