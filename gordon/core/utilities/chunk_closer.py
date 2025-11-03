"""
Chunk Position Closer
======================
Day 48: Close large positions in chunks to minimize market impact.

Features:
- Close positions in multiple smaller orders
- Configurable chunk sizes
- Sleep between orders
- Progress tracking
"""

import logging
import asyncio
from typing import Dict, Optional, List
from datetime import datetime

logger = logging.getLogger(__name__)


class ChunkPositionCloser:
    """
    Close large positions in chunks to minimize market impact.
    
    Useful for:
    - Large positions that could move the market
    - Reducing slippage
    - Better execution prices
    """
    
    def __init__(
        self,
        exchange_adapter=None,
        config: Optional[Dict] = None
    ):
        """
        Initialize chunk position closer.
        
        Args:
            exchange_adapter: Exchange adapter instance
            config: Configuration dictionary
        """
        self.exchange_adapter = exchange_adapter
        self.config = config or {}
        
        # Default configuration
        self.default_max_order_size_usd = self.config.get('max_order_size_usd', 1000.0)
        self.default_sleep_between_orders = self.config.get('sleep_between_orders', 60)  # seconds
        self.min_chunk_size_usd = self.config.get('min_chunk_size_usd', 50.0)
    
    async def close_position_in_chunks(
        self,
        symbol: str,
        total_quantity: float,
        max_order_size_usd: Optional[float] = None,
        sleep_between_orders: Optional[int] = None,
        current_price: Optional[float] = None
    ) -> Dict:
        """
        Close a position in chunks.
        
        Args:
            symbol: Trading symbol
            total_quantity: Total position quantity to close (positive for long, negative for short)
            max_order_size_usd: Maximum USD value per chunk
            sleep_between_orders: Seconds to sleep between orders
            current_price: Current price (will fetch if not provided)
            
        Returns:
            Dict with execution results
        """
        if not self.exchange_adapter:
            logger.error("Exchange adapter not available")
            return {'success': False, 'error': 'Exchange adapter not available'}
        
        max_order_size_usd = max_order_size_usd or self.default_max_order_size_usd
        sleep_between_orders = sleep_between_orders or self.default_sleep_between_orders
        
        # Get current price if not provided
        if current_price is None:
            try:
                ticker = await self.exchange_adapter.get_ticker(symbol)
                if ticker:
                    current_price = float(ticker.get('last', 0))
                else:
                    return {'success': False, 'error': 'Could not get current price'}
            except Exception as e:
                logger.error(f"Error getting price: {e}")
                return {'success': False, 'error': f'Price fetch error: {e}'}
        
        if current_price <= 0:
            return {'success': False, 'error': 'Invalid current price'}
        
        # Calculate chunk size in quantity
        max_quantity_per_chunk = max_order_size_usd / current_price
        
        # Determine if long or short
        is_long = total_quantity > 0
        remaining_quantity = abs(total_quantity)
        total_value_usd = remaining_quantity * current_price
        
        # Calculate number of chunks
        num_chunks = int(remaining_quantity / max_quantity_per_chunk) + (
            1 if remaining_quantity % max_quantity_per_chunk > 0 else 0
        )
        
        logger.info(
            f"Closing {symbol} position in {num_chunks} chunks: "
            f"{remaining_quantity:.6f} units (${total_value_usd:.2f})"
        )
        
        results = []
        chunk_num = 0
        total_closed = 0
        
        while remaining_quantity > 0:
            chunk_num += 1
            
            # Calculate chunk quantity
            chunk_quantity = min(remaining_quantity, max_quantity_per_chunk)
            chunk_value_usd = chunk_quantity * current_price
            
            # Skip if chunk too small
            if chunk_value_usd < self.min_chunk_size_usd:
                logger.warning(
                    f"Chunk size ${chunk_value_usd:.2f} below minimum "
                    f"${self.min_chunk_size_usd:.2f}. Closing remaining position..."
                )
                chunk_quantity = remaining_quantity
                chunk_value_usd = chunk_quantity * current_price
            
            logger.info(
                f"Chunk {chunk_num}/{num_chunks}: Closing {chunk_quantity:.6f} units "
                f"(${chunk_value_usd:.2f})"
            )
            
            try:
                # Place order
                if is_long:
                    # Long position - sell
                    order_result = await self.exchange_adapter.place_order(
                        symbol=symbol,
                        side='sell',
                        amount=chunk_quantity,
                        order_type='market'
                    )
                else:
                    # Short position - buy to close
                    order_result = await self.exchange_adapter.place_order(
                        symbol=symbol,
                        side='buy',
                        amount=chunk_quantity,
                        order_type='market'
                    )
                
                if order_result:
                    total_closed += chunk_quantity
                    remaining_quantity -= chunk_quantity
                    
                    results.append({
                        'chunk_num': chunk_num,
                        'quantity': chunk_quantity,
                        'value_usd': chunk_value_usd,
                        'order_id': order_result.get('id', 'N/A'),
                        'success': True,
                        'timestamp': datetime.now().isoformat()
                    })
                    
                    logger.info(
                        f"Chunk {chunk_num} executed successfully. "
                        f"Remaining: {remaining_quantity:.6f} units"
                    )
                else:
                    logger.error(f"Chunk {chunk_num} failed to execute")
                    results.append({
                        'chunk_num': chunk_num,
                        'quantity': chunk_quantity,
                        'value_usd': chunk_value_usd,
                        'success': False,
                        'error': 'Order execution failed',
                        'timestamp': datetime.now().isoformat()
                    })
                    
                    # Continue with next chunk even if one fails
                    remaining_quantity -= chunk_quantity
                    total_closed += chunk_quantity
                
            except Exception as e:
                logger.error(f"Error executing chunk {chunk_num}: {e}")
                results.append({
                    'chunk_num': chunk_num,
                    'quantity': chunk_quantity,
                    'value_usd': chunk_value_usd,
                    'success': False,
                    'error': str(e),
                    'timestamp': datetime.now().isoformat()
                })
            
            # Sleep between orders (except for last chunk)
            if remaining_quantity > 0 and chunk_num < num_chunks:
                logger.info(f"Sleeping {sleep_between_orders}s before next chunk...")
                await asyncio.sleep(sleep_between_orders)
        
        # Calculate summary
        successful_chunks = [r for r in results if r.get('success')]
        failed_chunks = [r for r in results if not r.get('success')]
        total_closed_usd = sum(r['value_usd'] for r in successful_chunks)
        
        return {
            'success': len(failed_chunks) == 0,
            'symbol': symbol,
            'total_chunks': num_chunks,
            'successful_chunks': len(successful_chunks),
            'failed_chunks': len(failed_chunks),
            'total_closed_quantity': total_closed,
            'total_closed_usd': total_closed_usd,
            'original_value_usd': total_value_usd,
            'chunks': results,
            'timestamp': datetime.now().isoformat()
        }
    
    async def close_all_positions_chunked(
        self,
        max_order_size_usd: Optional[float] = None,
        sleep_between_orders: Optional[int] = None
    ) -> Dict:
        """
        Close all open positions in chunks.
        
        Args:
            max_order_size_usd: Maximum USD value per chunk
            sleep_between_orders: Seconds to sleep between orders
            
        Returns:
            Dict with results for all positions
        """
        if not self.exchange_adapter:
            return {'success': False, 'error': 'Exchange adapter not available'}
        
        try:
            # Get all positions
            positions = await self.exchange_adapter.get_all_positions()
            
            if not positions:
                return {
                    'success': True,
                    'message': 'No positions to close',
                    'results': []
                }
            
            all_results = []
            
            for position in positions:
                symbol = position.get('symbol')
                quantity = float(position.get('quantity', 0))
                current_price = float(position.get('price', 0))
                
                if abs(quantity) < 0.0001:  # Skip dust positions
                    continue
                
                result = await self.close_position_in_chunks(
                    symbol=symbol,
                    total_quantity=quantity,
                    max_order_size_usd=max_order_size_usd,
                    sleep_between_orders=sleep_between_orders,
                    current_price=current_price
                )
                
                all_results.append(result)
            
            return {
                'success': True,
                'total_positions': len(positions),
                'results': all_results,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error closing all positions: {e}")
            return {'success': False, 'error': str(e)}

