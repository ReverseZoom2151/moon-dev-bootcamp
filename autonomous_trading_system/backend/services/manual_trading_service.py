"""
Manual Trading Service

Provides manual buy/sell functionality with position management.
Based on ez_buy_sell.py with enterprise enhancements.
"""

import asyncio
import logging
import time
from datetime import datetime
from typing import Optional, Dict, List
from dataclasses import dataclass
from enum import Enum

# Setup logging
logger = logging.getLogger(__name__)

class TradeAction(Enum):
    """Trade action types"""
    CLOSE_POSITION = "close_position"
    OPEN_POSITION = "open_position"
    PARTIAL_CLOSE = "partial_close"
    ADD_TO_POSITION = "add_to_position"

class OrderStatus(Enum):
    """Order execution status"""
    PENDING = "pending"
    EXECUTED = "executed"
    FAILED = "failed"
    CANCELLED = "cancelled"

@dataclass
class PositionDetails:
    """Current position information"""
    symbol: str
    tokens: float
    price: float
    usd_value: float
    timestamp: datetime

@dataclass
class TradeRequest:
    """Manual trade request structure"""
    action: TradeAction
    symbol: str
    target_usd_size: Optional[float] = None
    max_chunk_usd: Optional[float] = None
    orders_per_burst: int = 1
    slippage_bps: int = 50
    tx_delay_seconds: float = 2.0
    user_id: Optional[str] = None
    reason: Optional[str] = None

@dataclass
class TradeResult:
    """Trade execution result"""
    request_id: str
    action: TradeAction
    symbol: str
    success: bool
    initial_position: PositionDetails
    final_position: PositionDetails
    orders_executed: int
    total_volume_usd: float
    execution_time_seconds: float
    error_message: Optional[str] = None
    trade_details: List[Dict] = None

class ManualTradingService:
    """
    Manual Trading Service for position management
    
    Features:
    - Manual position opening to target USD size
    - Full position closing with chunked selling
    - Partial position management
    - Risk controls and position monitoring
    - Integration with existing order management
    - Async execution with progress tracking
    """
    
    def __init__(self, market_data_manager=None, order_manager=None, config=None):
        self.market_data_manager = market_data_manager
        self.order_manager = order_manager
        self.config = config
        self.active_trades: Dict[str, TradeRequest] = {}
        self.trade_history: List[TradeResult] = []
        self.is_running = False
        
        # Default configuration
        self.default_config = {
            'max_usd_order_size': 1000.0,
            'orders_per_burst': 1,
            'tx_sleep_seconds': 2.0,
            'slippage_bps': 50,
            'position_close_threshold_usd': 0.50,
            'target_reach_tolerance': 0.97,
            'max_retries': 2,
            'retry_delay_multiplier': 2.0
        }
        
        logger.info("📋 Manual Trading Service initialized")
    
    async def start(self):
        """Start the manual trading service"""
        self.is_running = True
        logger.info("🚀 Manual Trading Service started")
    
    async def stop(self):
        """Stop the manual trading service"""
        self.is_running = False
        # Cancel any active trades
        for trade_id in list(self.active_trades.keys()):
            await self.cancel_trade(trade_id)
        logger.info("🛑 Manual Trading Service stopped")
    
    async def get_position_details(self, symbol: str) -> PositionDetails:
        """Get current position details for a symbol"""
        try:
            # Get position from order manager or portfolio manager
            position_size = 0.0
            if self.order_manager and hasattr(self.order_manager, 'get_position'):
                position_info = await self.order_manager.get_position(symbol)
                position_size = position_info.get('size', 0.0) if position_info else 0.0
            
            # Get current price from market data manager
            current_price = 0.0
            if self.market_data_manager:
                price_data = await self.market_data_manager.get_current_price(symbol)
                current_price = price_data.get('price', 0.0) if price_data else 0.0
            
            # Simulate some data for demo if no real data
            if current_price == 0.0:
                if symbol.upper() == 'BTC':
                    current_price = 45000.0  # Demo BTC price
                elif symbol.upper() == 'ETH':
                    current_price = 3000.0   # Demo ETH price
                else:
                    current_price = 100.0    # Demo price
            
            # Calculate USD value
            usd_value = position_size * current_price
            
            return PositionDetails(
                symbol=symbol,
                tokens=position_size,
                price=current_price,
                usd_value=usd_value,
                timestamp=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"Error getting position details for {symbol}: {e}")
            return PositionDetails(
                symbol=symbol,
                tokens=0.0,
                price=0.0,
                usd_value=0.0,
                timestamp=datetime.now()
            )
    
    def calculate_buy_chunk_size(self, target_usd: float, current_usd: float, max_chunk_usd: float) -> float:
        """Calculate the size of the next buy chunk in USD"""
        size_needed = target_usd - current_usd
        if size_needed <= 0:
            return 0.0
        
        return min(size_needed, max_chunk_usd)
    
    def calculate_sell_chunk_size(self, current_usd: float, max_chunk_usd: float) -> float:
        """Calculate the size of the next sell chunk in USD"""
        return min(current_usd, max_chunk_usd)
    
    async def execute_market_buy(self, symbol: str, usd_amount: float, slippage_bps: int, orders_per_burst: int = 1) -> bool:
        """Execute market buy order(s)"""
        try:
            if usd_amount <= 0:
                return True
            
            # Calculate order size per burst
            order_size = usd_amount / orders_per_burst
            
            for i in range(orders_per_burst):
                if self.order_manager:
                    # Create market buy order
                    order_request = {
                        'symbol': symbol,
                        'side': 'buy',
                        'type': 'market',
                        'quantity': order_size,  # USD amount for market orders
                        'slippage_bps': slippage_bps
                    }
                    
                    order_result = await self.order_manager.create_order(**order_request)
                    logger.info(f"Market buy order submitted: {symbol} ${order_size:.2f}")
                    
                    if i < orders_per_burst - 1:
                        await asyncio.sleep(1)  # Small delay between burst orders
                else:
                    # Simulate order execution for testing
                    logger.info(f"Simulated market buy: {symbol} ${order_size:.2f}")
                    await asyncio.sleep(0.1)
            
            return True
            
        except Exception as e:
            logger.error(f"Error executing market buy for {symbol}: {e}")
            return False
    
    async def execute_market_sell(self, symbol: str, usd_amount: float, slippage_bps: int) -> bool:
        """Execute market sell order"""
        try:
            if usd_amount <= 0:
                return True
            
            if self.order_manager:
                # Create market sell order
                order_request = {
                    'symbol': symbol,
                    'side': 'sell',
                    'type': 'market',
                    'quantity': usd_amount,  # USD amount for market orders
                    'slippage_bps': slippage_bps
                }
                
                order_result = await self.order_manager.create_order(**order_request)
                logger.info(f"Market sell order submitted: {symbol} ${usd_amount:.2f}")
            else:
                # Simulate order execution for testing
                logger.info(f"Simulated market sell: {symbol} ${usd_amount:.2f}")
                await asyncio.sleep(0.1)
            
            return True
            
        except Exception as e:
            logger.error(f"Error executing market sell for {symbol}: {e}")
            return False
    
    async def open_position_to_target(self, request: TradeRequest) -> TradeResult:
        """Open position to target USD size with chunked buying"""
        request_id = f"open_{request.symbol}_{int(time.time())}"
        start_time = datetime.now()
        
        try:
            logger.info(f"Opening position for {request.symbol} to ${request.target_usd_size:.2f}")
            
            initial_position = await self.get_position_details(request.symbol)
            orders_executed = 0
            total_volume = 0.0
            
            config = self.config or self.default_config
            max_chunk_usd = request.max_chunk_usd or config.get('max_usd_order_size', 1000.0)
            tx_delay = request.tx_delay_seconds or config.get('tx_sleep_seconds', 2.0)
            max_retries = config.get('max_retries', 2)
            tolerance = config.get('target_reach_tolerance', 0.97)
            
            while True:
                current_position = await self.get_position_details(request.symbol)
                
                logger.info(f"Current position: {current_position.tokens:.6f} tokens (${current_position.usd_value:.2f}) | Target: ${request.target_usd_size:.2f}")
                
                # Check if target reached
                if current_position.usd_value >= request.target_usd_size * tolerance:
                    logger.info(f"Target position size reached for {request.symbol} (${current_position.usd_value:.2f})")
                    break
                
                # Calculate next chunk
                chunk_size = self.calculate_buy_chunk_size(
                    request.target_usd_size, 
                    current_position.usd_value, 
                    max_chunk_usd
                )
                
                if chunk_size <= 0:
                    logger.info("Calculated chunk size is 0, target likely reached")
                    break
                
                logger.info(f"Buying chunk of ${chunk_size:.2f}...")
                
                # Attempt buy with retry logic
                buy_successful = False
                for attempt in range(max_retries + 1):
                    buy_successful = await self.execute_market_buy(
                        request.symbol, 
                        chunk_size, 
                        request.slippage_bps, 
                        request.orders_per_burst
                    )
                    
                    if buy_successful:
                        orders_executed += request.orders_per_burst
                        total_volume += chunk_size
                        break
                    else:
                        if attempt < max_retries:
                            retry_delay = tx_delay * config.get('retry_delay_multiplier', 2.0)
                            logger.warning(f"Buy attempt {attempt + 1} failed, retrying in {retry_delay}s...")
                            await asyncio.sleep(retry_delay)
                        else:
                            logger.error("All buy attempts failed, exiting")
                            break
                
                if not buy_successful:
                    break
                
                # Wait between chunks
                await asyncio.sleep(tx_delay)
            
            final_position = await self.get_position_details(request.symbol)
            execution_time = (datetime.now() - start_time).total_seconds()
            
            result = TradeResult(
                request_id=request_id,
                action=request.action,
                symbol=request.symbol,
                success=True,
                initial_position=initial_position,
                final_position=final_position,
                orders_executed=orders_executed,
                total_volume_usd=total_volume,
                execution_time_seconds=execution_time
            )
            
            self.trade_history.append(result)
            logger.info(f"Position opening completed: {initial_position.usd_value:.2f} → ${final_position.usd_value:.2f}")
            
            return result
            
        except Exception as e:
            error_msg = f"Error opening position for {request.symbol}: {str(e)}"
            logger.error(error_msg)
            
            final_position = await self.get_position_details(request.symbol)
            execution_time = (datetime.now() - start_time).total_seconds()
            
            return TradeResult(
                request_id=request_id,
                action=request.action,
                symbol=request.symbol,
                success=False,
                initial_position=initial_position if 'initial_position' in locals() else final_position,
                final_position=final_position,
                orders_executed=orders_executed if 'orders_executed' in locals() else 0,
                total_volume_usd=total_volume if 'total_volume' in locals() else 0.0,
                execution_time_seconds=execution_time,
                error_message=error_msg
            )
    
    async def close_position_fully(self, request: TradeRequest) -> TradeResult:
        """Close entire position with chunked selling"""
        request_id = f"close_{request.symbol}_{int(time.time())}"
        start_time = datetime.now()
        
        try:
            logger.info(f"Closing position fully for {request.symbol}")
            
            initial_position = await self.get_position_details(request.symbol)
            orders_executed = 0
            total_volume = 0.0
            
            config = self.config or self.default_config
            max_chunk_usd = request.max_chunk_usd or config.get('max_usd_order_size', 1000.0)
            tx_delay = request.tx_delay_seconds or config.get('tx_sleep_seconds', 2.0)
            close_threshold = config.get('position_close_threshold_usd', 0.50)
            max_retries = config.get('max_retries', 2)
            
            while True:
                current_position = await self.get_position_details(request.symbol)
                
                logger.info(f"Current position: {current_position.tokens:.6f} tokens (${current_position.usd_value:.2f})")
                
                # Check if position is effectively closed
                if current_position.usd_value < close_threshold:
                    logger.info("Position considered closed")
                    break
                
                # Calculate sell chunk size
                chunk_size = self.calculate_sell_chunk_size(current_position.usd_value, max_chunk_usd)
                
                logger.info(f"Selling chunk of ${chunk_size:.2f}...")
                
                # Attempt sell with retry logic
                sell_successful = False
                for attempt in range(max_retries + 1):
                    sell_successful = await self.execute_market_sell(
                        request.symbol, 
                        chunk_size, 
                        request.slippage_bps
                    )
                    
                    if sell_successful:
                        orders_executed += 1
                        total_volume += chunk_size
                        break
                    else:
                        if attempt < max_retries:
                            retry_delay = tx_delay * config.get('retry_delay_multiplier', 2.0)
                            logger.warning(f"Sell attempt {attempt + 1} failed, retrying in {retry_delay}s...")
                            await asyncio.sleep(retry_delay)
                        else:
                            logger.error("All sell attempts failed, exiting")
                            break
                
                if not sell_successful:
                    break
                
                # Wait between chunks
                await asyncio.sleep(tx_delay)
            
            final_position = await self.get_position_details(request.symbol)
            execution_time = (datetime.now() - start_time).total_seconds()
            
            result = TradeResult(
                request_id=request_id,
                action=request.action,
                symbol=request.symbol,
                success=True,
                initial_position=initial_position,
                final_position=final_position,
                orders_executed=orders_executed,
                total_volume_usd=total_volume,
                execution_time_seconds=execution_time
            )
            
            self.trade_history.append(result)
            logger.info(f"Position closing completed: ${initial_position.usd_value:.2f} → ${final_position.usd_value:.2f}")
            
            return result
            
        except Exception as e:
            error_msg = f"Error closing position for {request.symbol}: {str(e)}"
            logger.error(error_msg)
            
            final_position = await self.get_position_details(request.symbol)
            execution_time = (datetime.now() - start_time).total_seconds()
            
            return TradeResult(
                request_id=request_id,
                action=request.action,
                symbol=request.symbol,
                success=False,
                initial_position=initial_position if 'initial_position' in locals() else final_position,
                final_position=final_position,
                orders_executed=orders_executed if 'orders_executed' in locals() else 0,
                total_volume_usd=total_volume if 'total_volume' in locals() else 0.0,
                execution_time_seconds=execution_time,
                error_message=error_msg
            )
    
    async def execute_trade(self, request: TradeRequest) -> TradeResult:
        """Execute a manual trade request"""
        try:
            # Validate request
            if request.action not in [TradeAction.OPEN_POSITION, TradeAction.CLOSE_POSITION]:
                raise ValueError(f"Unsupported trade action: {request.action}")
            
            if request.action == TradeAction.OPEN_POSITION and not request.target_usd_size:
                raise ValueError("target_usd_size required for open_position action")
            
            # Store active trade
            trade_id = f"{request.action.value}_{request.symbol}_{int(time.time())}"
            self.active_trades[trade_id] = request
            
            try:
                # Execute based on action
                if request.action == TradeAction.OPEN_POSITION:
                    result = await self.open_position_to_target(request)
                elif request.action == TradeAction.CLOSE_POSITION:
                    result = await self.close_position_fully(request)
                
                return result
                
            finally:
                # Remove from active trades
                self.active_trades.pop(trade_id, None)
                
        except Exception as e:
            error_msg = f"Error executing trade: {str(e)}"
            logger.error(error_msg)
            
            # Create error result
            current_position = await self.get_position_details(request.symbol)
            return TradeResult(
                request_id=f"error_{int(time.time())}",
                action=request.action,
                symbol=request.symbol,
                success=False,
                initial_position=current_position,
                final_position=current_position,
                orders_executed=0,
                total_volume_usd=0.0,
                execution_time_seconds=0.0,
                error_message=error_msg
            )
    
    async def cancel_trade(self, trade_id: str) -> bool:
        """Cancel an active trade"""
        try:
            if trade_id in self.active_trades:
                del self.active_trades[trade_id]
                logger.info(f"Trade {trade_id} cancelled")
                return True
            return False
        except Exception as e:
            logger.error(f"Error cancelling trade {trade_id}: {e}")
            return False
    
    async def get_service_status(self) -> Dict:
        """Get service status and statistics"""
        total_trades = len(self.trade_history)
        successful_trades = sum(1 for t in self.trade_history if t.success)
        
        return {
            "service_name": "Manual Trading Service",
            "status": "active" if self.is_running else "inactive",
            "features": [
                "Position opening to target USD size",
                "Full position closing with chunked selling", 
                "Configurable order burst sizes",
                "Retry logic for failed orders",
                "Real-time position monitoring",
                "Risk controls and thresholds"
            ],
            "active_trades": len(self.active_trades),
            "statistics": {
                "total_trades": total_trades,
                "successful_trades": successful_trades,
                "failed_trades": total_trades - successful_trades,
                "success_rate": (successful_trades / total_trades * 100) if total_trades > 0 else 0
            },
            "configuration": self.config or self.default_config,
            "recent_trades": [
                {
                    "request_id": t.request_id,
                    "action": t.action.value,
                    "symbol": t.symbol,
                    "success": t.success,
                    "initial_usd": t.initial_position.usd_value,
                    "final_usd": t.final_position.usd_value,
                    "orders_executed": t.orders_executed,
                    "total_volume_usd": t.total_volume_usd,
                    "execution_time_seconds": t.execution_time_seconds
                }
                for t in self.trade_history[-10:]
            ]
        }
    
    def get_recent_trades(self, limit: int = 10) -> List[Dict]:
        """Get recent trade results"""
        return [
            {
                "request_id": t.request_id,
                "action": t.action.value,
                "symbol": t.symbol,
                "success": t.success,
                "initial_position": {
                    "tokens": t.initial_position.tokens,
                    "price": t.initial_position.price,
                    "usd_value": t.initial_position.usd_value
                },
                "final_position": {
                    "tokens": t.final_position.tokens,
                    "price": t.final_position.price,
                    "usd_value": t.final_position.usd_value
                },
                "orders_executed": t.orders_executed,
                "total_volume_usd": t.total_volume_usd,
                "execution_time_seconds": t.execution_time_seconds,
                "error_message": t.error_message
            }
            for t in self.trade_history[-limit:]
        ]

# Convenience functions for API integration
async def execute_manual_buy(
    symbol: str,
    target_usd_size: float,
    max_chunk_usd: float = 1000.0,
    orders_per_burst: int = 1,
    slippage_bps: int = 50,
    tx_delay_seconds: float = 2.0,
    service: ManualTradingService = None
) -> TradeResult:
    """Convenience function for manual buy operations"""
    if not service:
        service = ManualTradingService()
    
    request = TradeRequest(
        action=TradeAction.OPEN_POSITION,
        symbol=symbol,
        target_usd_size=target_usd_size,
        max_chunk_usd=max_chunk_usd,
        orders_per_burst=orders_per_burst,
        slippage_bps=slippage_bps,
        tx_delay_seconds=tx_delay_seconds
    )
    
    return await service.execute_trade(request)

async def execute_manual_sell(
    symbol: str,
    max_chunk_usd: float = 1000.0,
    slippage_bps: int = 50,
    tx_delay_seconds: float = 2.0,
    service: ManualTradingService = None
) -> TradeResult:
    """Convenience function for manual sell operations"""
    if not service:
        service = ManualTradingService()
    
    request = TradeRequest(
        action=TradeAction.CLOSE_POSITION,
        symbol=symbol,
        max_chunk_usd=max_chunk_usd,
        slippage_bps=slippage_bps,
        tx_delay_seconds=tx_delay_seconds
    )
    
    return await service.execute_trade(request) 