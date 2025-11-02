import uuid
import logging
from typing import Dict, List, Optional
from enum import Enum
from dataclasses import dataclass
from datetime import datetime

logger = logging.getLogger(__name__)

class OrderType(Enum):
    MARKET = "market"
    LIMIT = "limit"
    STOP_LOSS = "stop_loss"
    TAKE_PROFIT = "take_profit"
    TRAILING_STOP = "trailing_stop"
    BRACKET = "bracket"
    POST_ONLY = "post_only"

class OrderStatus(Enum):
    PENDING = "pending"
    SUBMITTED = "submitted"
    FILLED = "filled"
    PARTIALLY_FILLED = "partially_filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    EXPIRED = "expired"

class OrderSide(Enum):
    BUY = "buy"
    SELL = "sell"

@dataclass
class Order:
    """Order data structure"""
    id: str
    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: float
    price: Optional[float] = None
    stop_price: Optional[float] = None
    time_in_force: str = "GTC"
    status: OrderStatus = OrderStatus.PENDING
    filled_quantity: float = 0.0
    average_fill_price: float = 0.0
    created_at: datetime = None
    updated_at: datetime = None
    exchange_order_id: Optional[str] = None
    parent_order_id: Optional[str] = None
    child_orders: List[str] = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.utcnow()
        if self.updated_at is None:
            self.updated_at = datetime.utcnow()
        if self.child_orders is None:
            self.child_orders = []

@dataclass
class BracketOrder:
    """Bracket order with main order, stop loss, and take profit"""
    main_order: Order
    stop_loss_order: Optional[Order] = None
    take_profit_order: Optional[Order] = None
    trailing_stop_distance: Optional[float] = None

class OrderManager:
    """Advanced order management system"""
    
    def __init__(self, strategy_engine):
        self.strategy_engine = strategy_engine
        self.orders: Dict[str, Order] = {}
        self.bracket_orders: Dict[str, BracketOrder] = {}
        self.active_orders: Dict[str, Order] = {}
        self.order_history: List[Order] = []
        self.trailing_stops: Dict[str, dict] = {}
        
    async def create_order(
        self,
        symbol: str,
        side: OrderSide,
        order_type: OrderType,
        quantity: float,
        price: Optional[float] = None,
        stop_price: Optional[float] = None,
        time_in_force: str = "GTC",
        reduce_only: bool = False
    ) -> Order:
        """Create a new order"""
        
        order_id = str(uuid.uuid4())
        
        order = Order(
            id=order_id,
            symbol=symbol,
            side=side,
            order_type=order_type,
            quantity=quantity,
            price=price,
            stop_price=stop_price,
            time_in_force=time_in_force
        )
        
        self.orders[order_id] = order
        
        # Submit order to exchange
        try:
            await self._submit_order(order, reduce_only)
            self.active_orders[order_id] = order
            logger.info(f"Order created: {order_id} - {symbol} {side.value} {quantity}")
            
        except Exception as e:
            order.status = OrderStatus.REJECTED
            logger.error(f"Failed to submit order {order_id}: {e}")
            
        return order
    
    async def create_bracket_order(
        self,
        symbol: str,
        side: OrderSide,
        quantity: float,
        entry_price: Optional[float] = None,
        stop_loss_price: Optional[float] = None,
        take_profit_price: Optional[float] = None,
        trailing_stop_distance: Optional[float] = None
    ) -> BracketOrder:
        """Create a bracket order with stop loss and take profit"""
        
        # Create main order
        main_order_type = OrderType.LIMIT if entry_price else OrderType.MARKET
        main_order = await self.create_order(
            symbol=symbol,
            side=side,
            order_type=main_order_type,
            quantity=quantity,
            price=entry_price
        )
        
        bracket_order = BracketOrder(
            main_order=main_order,
            trailing_stop_distance=trailing_stop_distance
        )
        
        # Create stop loss order (opposite side)
        if stop_loss_price:
            stop_side = OrderSide.SELL if side == OrderSide.BUY else OrderSide.BUY
            stop_order = Order(
                id=str(uuid.uuid4()),
                symbol=symbol,
                side=stop_side,
                order_type=OrderType.STOP_LOSS,
                quantity=quantity,
                stop_price=stop_loss_price,
                parent_order_id=main_order.id
            )
            bracket_order.stop_loss_order = stop_order
            main_order.child_orders.append(stop_order.id)
            
        # Create take profit order (opposite side)
        if take_profit_price:
            tp_side = OrderSide.SELL if side == OrderSide.BUY else OrderSide.BUY
            tp_order = Order(
                id=str(uuid.uuid4()),
                symbol=symbol,
                side=tp_side,
                order_type=OrderType.TAKE_PROFIT,
                quantity=quantity,
                price=take_profit_price,
                parent_order_id=main_order.id
            )
            bracket_order.take_profit_order = tp_order
            main_order.child_orders.append(tp_order.id)
            
        self.bracket_orders[main_order.id] = bracket_order
        
        logger.info(f"Bracket order created: {main_order.id}")
        return bracket_order
    
    async def create_trailing_stop(
        self,
        symbol: str,
        side: OrderSide,
        quantity: float,
        trail_distance: float,
        trail_percent: Optional[float] = None
    ) -> Order:
        """Create a trailing stop order"""
        
        order = await self.create_order(
            symbol=symbol,
            side=side,
            order_type=OrderType.TRAILING_STOP,
            quantity=quantity
        )
        
        # Get current price
        current_price = await self._get_current_price(symbol)
        
        self.trailing_stops[order.id] = {
            'trail_distance': trail_distance,
            'trail_percent': trail_percent,
            'best_price': current_price,
            'stop_price': current_price - trail_distance if side == OrderSide.SELL else current_price + trail_distance
        }
        
        logger.info(f"Trailing stop created: {order.id}")
        return order
    
    async def cancel_order(self, order_id: str) -> bool:
        """Cancel an order"""
        if order_id not in self.orders:
            logger.warning(f"Order {order_id} not found")
            return False
            
        order = self.orders[order_id]
        
        try:
            await self._cancel_order_on_exchange(order)
            order.status = OrderStatus.CANCELLED
            order.updated_at = datetime.utcnow()
            
            if order_id in self.active_orders:
                del self.active_orders[order_id]
                
            logger.info(f"Order cancelled: {order_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to cancel order {order_id}: {e}")
            return False
    
    async def cancel_all_orders(self, symbol: Optional[str] = None) -> int:
        """Cancel all orders or all orders for a specific symbol"""
        cancelled_count = 0
        
        orders_to_cancel = []
        for order_id, order in self.active_orders.items():
            if symbol is None or order.symbol == symbol:
                orders_to_cancel.append(order_id)
                
        for order_id in orders_to_cancel:
            if await self.cancel_order(order_id):
                cancelled_count += 1
                
        logger.info(f"Cancelled {cancelled_count} orders")
        return cancelled_count
    
    async def modify_order(
        self,
        order_id: str,
        new_quantity: Optional[float] = None,
        new_price: Optional[float] = None
    ) -> bool:
        """Modify an existing order"""
        if order_id not in self.orders:
            logger.warning(f"Order {order_id} not found")
            return False
            
        order = self.orders[order_id]
        
        try:
            await self._modify_order_on_exchange(order, new_quantity, new_price)
            
            if new_quantity:
                order.quantity = new_quantity
            if new_price:
                order.price = new_price
                
            order.updated_at = datetime.utcnow()
            logger.info(f"Order modified: {order_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to modify order {order_id}: {e}")
            return False
    
    async def update_trailing_stops(self):
        """Update trailing stop orders based on current prices"""
        for order_id, trail_data in self.trailing_stops.items():
            if order_id not in self.active_orders:
                continue
                
            order = self.active_orders[order_id]
            current_price = await self._get_current_price(order.symbol)
            
            if current_price is None:
                continue
                
            # Update trailing stop logic
            if order.side == OrderSide.SELL:
                # For sell orders, trail when price goes up
                if current_price > trail_data['best_price']:
                    trail_data['best_price'] = current_price
                    new_stop = current_price - trail_data['trail_distance']
                    trail_data['stop_price'] = max(trail_data['stop_price'], new_stop)
                    
                # Trigger if price falls below stop
                if current_price <= trail_data['stop_price']:
                    await self._trigger_trailing_stop(order)
                    
            else:  # BUY orders
                # For buy orders, trail when price goes down
                if current_price < trail_data['best_price']:
                    trail_data['best_price'] = current_price
                    new_stop = current_price + trail_data['trail_distance']
                    trail_data['stop_price'] = min(trail_data['stop_price'], new_stop)
                    
                # Trigger if price rises above stop
                if current_price >= trail_data['stop_price']:
                    await self._trigger_trailing_stop(order)
    
    async def handle_fill(self, order_id: str, fill_quantity: float, fill_price: float):
        """Handle order fill notification"""
        if order_id not in self.orders:
            logger.warning(f"Received fill for unknown order: {order_id}")
            return
            
        order = self.orders[order_id]
        order.filled_quantity += fill_quantity
        
        # Update average fill price
        total_filled_value = order.average_fill_price * (order.filled_quantity - fill_quantity)
        total_filled_value += fill_price * fill_quantity
        order.average_fill_price = total_filled_value / order.filled_quantity
        
        # Update status
        if order.filled_quantity >= order.quantity:
            order.status = OrderStatus.FILLED
            if order_id in self.active_orders:
                del self.active_orders[order_id]
                
            # Handle bracket order completion
            await self._handle_bracket_order_fill(order)
            
        else:
            order.status = OrderStatus.PARTIALLY_FILLED
            
        order.updated_at = datetime.utcnow()
        logger.info(f"Order fill: {order_id} - {fill_quantity} @ {fill_price}")
    
    async def get_order_status(self, order_id: str) -> Optional[OrderStatus]:
        """Get order status"""
        if order_id in self.orders:
            return self.orders[order_id].status
        return None
    
    async def get_active_orders(self, symbol: Optional[str] = None) -> List[Order]:
        """Get active orders"""
        if symbol:
            return [order for order in self.active_orders.values() if order.symbol == symbol]
        return list(self.active_orders.values())
    
    async def get_order_history(self, symbol: Optional[str] = None, limit: int = 100) -> List[Order]:
        """Get order history"""
        history = self.order_history
        if symbol:
            history = [order for order in history if order.symbol == symbol]
        return history[-limit:]
    
    # Private methods for exchange integration
    async def _submit_order(self, order: Order, reduce_only: bool = False):
        """Submit order to exchange"""
        # Implementation depends on exchange
        # This would integrate with the exchange APIs
        pass
    
    async def _cancel_order_on_exchange(self, order: Order):
        """Cancel order on exchange"""
        # Implementation depends on exchange
        pass
    
    async def _modify_order_on_exchange(self, order: Order, new_quantity: Optional[float], new_price: Optional[float]):
        """Modify order on exchange"""
        # Implementation depends on exchange
        pass
    
    async def _get_current_price(self, symbol: str) -> Optional[float]:
        """Get current market price"""
        # This would integrate with price feed
        return None
    
    async def _trigger_trailing_stop(self, order: Order):
        """Trigger trailing stop order"""
        # Convert to market order
        order.order_type = OrderType.MARKET
        order.price = None
        await self._submit_order(order)
        
        # Remove from trailing stops
        if order.id in self.trailing_stops:
            del self.trailing_stops[order.id]
            
        logger.info(f"Trailing stop triggered: {order.id}")
    
    async def _handle_bracket_order_fill(self, order: Order):
        """Handle bracket order fill and activate child orders"""
        if order.id in self.bracket_orders:
            bracket = self.bracket_orders[order.id]
            
            # Submit stop loss and take profit orders
            if bracket.stop_loss_order:
                await self._submit_order(bracket.stop_loss_order, reduce_only=True)
                self.active_orders[bracket.stop_loss_order.id] = bracket.stop_loss_order
                
            if bracket.take_profit_order:
                await self._submit_order(bracket.take_profit_order, reduce_only=True)
                self.active_orders[bracket.take_profit_order.id] = bracket.take_profit_order
                
            logger.info(f"Bracket order child orders activated: {order.id}")

class PositionManager:
    """Manages trading positions"""
    
    def __init__(self, order_manager: OrderManager):
        self.order_manager = order_manager
        self.positions: Dict[str, dict] = {}
        
    async def get_position(self, symbol: str) -> dict:
        """Get current position for symbol"""
        if symbol not in self.positions:
            self.positions[symbol] = {
                'symbol': symbol,
                'size': 0.0,
                'side': None,
                'entry_price': 0.0,
                'unrealized_pnl': 0.0,
                'realized_pnl': 0.0
            }
        return self.positions[symbol]
    
    async def update_position(self, symbol: str, fill_quantity: float, fill_price: float, side: OrderSide):
        """Update position based on fill"""
        position = await self.get_position(symbol)
        
        if side == OrderSide.BUY:
            if position['size'] < 0:  # Closing short position
                close_quantity = min(abs(position['size']), fill_quantity)
                # Calculate realized PnL for closed portion
                realized_pnl = close_quantity * (position['entry_price'] - fill_price)
                position['realized_pnl'] += realized_pnl
                position['size'] += close_quantity
                fill_quantity -= close_quantity
                
            if fill_quantity > 0:  # Opening/adding to long position
                total_cost = position['size'] * position['entry_price'] + fill_quantity * fill_price
                position['size'] += fill_quantity
                position['entry_price'] = total_cost / position['size'] if position['size'] > 0 else fill_price
                position['side'] = 'long'
                
        else:  # SELL
            if position['size'] > 0:  # Closing long position
                close_quantity = min(position['size'], fill_quantity)
                # Calculate realized PnL for closed portion
                realized_pnl = close_quantity * (fill_price - position['entry_price'])
                position['realized_pnl'] += realized_pnl
                position['size'] -= close_quantity
                fill_quantity -= close_quantity
                
            if fill_quantity > 0:  # Opening/adding to short position
                total_cost = abs(position['size']) * position['entry_price'] + fill_quantity * fill_price
                position['size'] -= fill_quantity
                position['entry_price'] = total_cost / abs(position['size']) if position['size'] != 0 else fill_price
                position['side'] = 'short'
        
        # Update side based on final size
        if position['size'] > 0:
            position['side'] = 'long'
        elif position['size'] < 0:
            position['side'] = 'short'
        else:
            position['side'] = None
            position['entry_price'] = 0.0
    
    async def close_position(self, symbol: str, percentage: float = 100.0) -> bool:
        """Close position (partially or fully)"""
        position = await self.get_position(symbol)
        
        if position['size'] == 0:
            logger.info(f"No position to close for {symbol}")
            return True
            
        close_quantity = abs(position['size']) * (percentage / 100.0)
        close_side = OrderSide.SELL if position['size'] > 0 else OrderSide.BUY
        
        order = await self.order_manager.create_order(
            symbol=symbol,
            side=close_side,
            order_type=OrderType.MARKET,
            quantity=close_quantity,
            reduce_only=True
        )
        
        logger.info(f"Position close order created: {order.id} for {symbol}")
        return True 