"""
EZ2 Solana Trading Service

A comprehensive service that implements all trading modes from the ez_2.py script:
- Mode 0: Close position (in chunks)
- Mode 1: Open buying position (in chunks)  
- Mode 2: ETH SMA based strategy
- Mode 4: Close positions based on PnL thresholds
- Mode 5: Simple market making (buy under/sell over)

Integrates with the autonomous trading system architecture and configuration.
"""

import asyncio
import logging
from datetime import datetime
from typing import Optional, Dict, Any
from enum import IntEnum
from dataclasses import dataclass
from termcolor import cprint
from core.config import get_settings
from services.solana_trading_utils_service import SolanaTradingUtilsService

logger = logging.getLogger(__name__)

class EZ2TradingMode(IntEnum):
    """Trading mode constants from ez_2.py"""
    CLOSE_MODE = 0
    BUY_MODE = 1
    ETH_TRADE_MODE = 2
    PNL_CLOSE_MODE = 4
    MARKET_MAKER_MODE = 5

@dataclass
class EZ2TradeResult:
    """Result of an EZ2 trading operation"""
    success: bool
    mode: EZ2TradingMode
    symbol: Optional[str] = None
    message: str = ""
    error: Optional[str] = None
    final_position_usd: float = 0.0
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None

@dataclass
class PositionDetails:
    """Position details structure"""
    tokens: float
    price: float
    usd_value: float

class EZ2SolanaTradingService:
    """
    EZ2 Solana Trading Service
    
    Implements all trading modes from ez_2.py with enterprise-grade error handling,
    logging, and integration with the autonomous trading system.
    """
    
    def __init__(self, config=None):
        """Initialize the EZ2 trading service"""
        self.config = config or get_settings()
        self.solana_service = SolanaTradingUtilsService(config)
        self.is_running = False
        self.active_market_maker_task = None
        
        # Configuration from autonomous trading system
        self.primary_symbol_mint = self.config.PRIMARY_SYMBOL_MINT
        self.usd_size = self.config.USD_SIZE
        self.max_usd_order_size = self.config.MAX_USD_ORDER_SIZE
        self.orders_per_open = self.config.ORDERS_PER_OPEN
        self.tx_sleep = self.config.TX_SLEEP
        self.slippage = self.config.SLIPPAGE
        self.token_batch = self.config.TOKEN_BATCH
        self.wallet_address = getattr(self.config, 'SOLANA_WALLET_ADDRESS', '')
        self.do_not_trade_list = self.config.SOLANA_DO_NOT_TRADE_LIST
        self.lowest_balance = self.config.LOWEST_BALANCE
        self.target_balance = self.config.TARGET_BALANCE
        self.buy_under = self.config.BUY_UNDER
        self.sell_over = self.config.SELL_OVER
        
        logger.info("🚀 EZ2 Solana Trading Service initialized")
    
    async def start(self):
        """Start the EZ2 trading service"""
        try:
            self.is_running = True
            await self.solana_service.start()
            logger.info("✅ EZ2 Solana Trading Service started successfully")
        except Exception as e:
            logger.error(f"❌ Failed to start EZ2 Solana Trading Service: {e}")
            self.is_running = False
            raise
    
    async def stop(self):
        """Stop the EZ2 trading service"""
        try:
            self.is_running = False
            
            # Stop market maker if running
            if self.active_market_maker_task and not self.active_market_maker_task.done():
                self.active_market_maker_task.cancel()
                try:
                    await self.active_market_maker_task
                except asyncio.CancelledError:
                    pass
            
            await self.solana_service.stop()
            logger.info("🛑 EZ2 Solana Trading Service stopped")
        except Exception as e:
            logger.error(f"❌ Error stopping EZ2 Solana Trading Service: {e}")
    
    async def _get_position_details(self, symbol: str) -> PositionDetails:
        """Fetch position size, price, and USD value"""
        try:
            pos = await self.solana_service.get_position(symbol)
            price = await self.solana_service.get_token_price(symbol)
            pos_usd = pos * price if price else 0.0
            return PositionDetails(tokens=pos, price=price or 0.0, usd_value=pos_usd)
        except Exception as e:
            logger.error(f"Error getting position details for {symbol}: {e}")
            return PositionDetails(tokens=0.0, price=0.0, usd_value=0.0)
    
    def _calculate_chunk_size(self, size_needed: float, max_chunk: float) -> str:
        """Calculate chunk size for an order, respecting the max limit"""
        chunk = max_chunk if size_needed > max_chunk else size_needed
        chunk_lamports = int(chunk * 10**6)  # Assuming USDC has 6 decimals
        return str(chunk_lamports)
    
    async def _attempt_market_buy(self, symbol: str, chunk_size_str: str, orders_count: int, sleep_time: int) -> bool:
        """Attempt to execute market buy orders with retries"""
        try:
            for _ in range(orders_count):
                result = await self.solana_service.market_buy(symbol, chunk_size_str, self.slippage)
                if result:
                    cprint(f'Chunk buy submitted for {symbol[-4:]} size: {chunk_size_str} lamports', 'white', 'on_blue')
                    await asyncio.sleep(1)  # Small delay between individual orders
                else:
                    logger.warning(f"Market buy failed for {symbol}")
                    return False
            
            await asyncio.sleep(sleep_time)  # Longer delay after a burst
            return True
        except Exception as e:
            cprint(f"Error during market buy attempt: {e}", 'light_yellow', 'on_red')
            return False

    async def _retry_market_buy(self, symbol: str, chunk_size_str: str, orders_count: int, sleep_time: int, retry_delay: int = 30) -> bool:
        """Retry market buy after a delay"""
        cprint(f'Retrying market buy in {retry_delay} seconds...', 'light_blue', 'on_light_magenta')
        await asyncio.sleep(retry_delay)
        
        try:
            for _ in range(orders_count):
                result = await self.solana_service.market_buy(symbol, chunk_size_str, self.slippage)
                if result:
                    cprint(f'Retry chunk buy submitted for {symbol[-4:]} size: {chunk_size_str} lamports', 'white', 'on_blue')
                    await asyncio.sleep(1)
                else:
                    logger.warning(f"Retry market buy failed for {symbol}")
                    return False
            
            await asyncio.sleep(sleep_time)
            return True
        except Exception as e:
            cprint(f"Error during retry market buy attempt: {e}", 'white', 'on_red')
            return False
    
    # ============== MODE IMPLEMENTATIONS ==============
    
    async def run_close_mode(self, symbol: str) -> EZ2TradeResult:
        """Mode 0: Close the position for the given symbol in chunks"""
        start_time = datetime.now()
        logger.info(f'--- Mode {EZ2TradingMode.CLOSE_MODE}: Closing Position for {symbol[-4:]} ---')
        
        try:
            pos_details = await self._get_position_details(symbol)
            
            while pos_details.tokens > 0.1:  # Using a small threshold instead of exactly 0
                logger.info(f"Current position: {pos_details.tokens}. Closing in chunks...")
                
                try:
                    # Use the chunk_sell_position method from solana service
                    success = await self.solana_service.chunk_sell_position(
                        symbol, self.max_usd_order_size, self.slippage
                    )
                    
                    if success:
                        cprint(f'Chunk kill order sent for {symbol[-4:]}', 'white', 'on_magenta')
                    else:
                        logger.warning(f"Chunk sell failed for {symbol}, retrying in 5 sec...")
                        await asyncio.sleep(5)
                    
                except Exception as e:
                    cprint(f"Error during chunk_kill: {e}. Retrying in 5 sec...", 'red')
                    await asyncio.sleep(5)
                
                await asyncio.sleep(1)  # Wait a bit before checking position again
                pos_details = await self._get_position_details(symbol)  # Refresh position status
            
            cprint(f'Position for {symbol[-4:]} closed.', 'white', 'on_green')
            
            return EZ2TradeResult(
                success=True,
                mode=EZ2TradingMode.CLOSE_MODE,
                symbol=symbol,
                message=f"Position closed for {symbol[-4:]}",
                final_position_usd=pos_details.usd_value,
                start_time=start_time,
                end_time=datetime.now()
            )
            
        except Exception as e:
            error_msg = f"Error in close mode for {symbol}: {e}"
            logger.error(error_msg)
            return EZ2TradeResult(
                success=False,
                mode=EZ2TradingMode.CLOSE_MODE,
                symbol=symbol,
                error=error_msg,
                start_time=start_time,
                end_time=datetime.now()
            )
    
    async def run_buy_mode(self, symbol: str, target_usd_size: Optional[float] = None) -> EZ2TradeResult:
        """Mode 1: Open a buying position up to USD_SIZE for the given symbol"""
        start_time = datetime.now()
        target_size = target_usd_size or self.usd_size
        logger.info(f'--- Mode {EZ2TradingMode.BUY_MODE}: Opening Position for {symbol[-4:]} ---')
        
        try:
            # Initial check
            pos_details = await self._get_position_details(symbol)
            logger.info(f'Initial State - Position: {round(pos_details.tokens, 2)}, '
                       f'Price: {round(pos_details.price, 8)}, '
                       f'Value: ${round(pos_details.usd_value, 2)}, Target: ${target_size}')
            
            if pos_details.usd_value >= (0.97 * target_size):
                cprint('Position already filled or close to target size.', 'yellow')
                return EZ2TradeResult(
                    success=True,
                    mode=EZ2TradingMode.BUY_MODE,
                    symbol=symbol,
                    message="Position already filled",
                    final_position_usd=pos_details.usd_value,
                    start_time=start_time,
                    end_time=datetime.now()
                )
            
            while pos_details.usd_value < (0.97 * target_size):
                size_needed = target_size - pos_details.usd_value
                chunk_size_str = self._calculate_chunk_size(size_needed, self.max_usd_order_size)
                
                logger.info(f'Need ${round(size_needed, 2)}. Buying chunk: {chunk_size_str} lamports.')
                
                # Attempt buy
                if not await self._attempt_market_buy(symbol, chunk_size_str, self.orders_per_open, self.tx_sleep):
                    # Attempt 2 (Retry)
                    if not await self._retry_market_buy(symbol, chunk_size_str, self.orders_per_open, self.tx_sleep):
                        error_msg = f'FINAL ERROR in buy process for {symbol[-4:]}. Exiting buy mode.'
                        cprint(error_msg, 'white', 'on_red')
                        return EZ2TradeResult(
                            success=False,
                            mode=EZ2TradingMode.BUY_MODE,
                            symbol=symbol,
                            error=error_msg,
                            final_position_usd=pos_details.usd_value,
                            start_time=start_time,
                            end_time=datetime.now()
                        )
                
                # Refresh position status after attempts
                try:
                    pos_details = await self._get_position_details(symbol)
                    logger.info(f'Updated State - Position: {round(pos_details.tokens, 2)}, '
                               f'Price: {round(pos_details.price, 8)}, '
                               f'Value: ${round(pos_details.usd_value, 2)}')
                except Exception as e:
                    error_msg = f"Error fetching position details after buy attempt: {e}. Exiting."
                    cprint(error_msg, 'red')
                    return EZ2TradeResult(
                        success=False,
                        mode=EZ2TradingMode.BUY_MODE,
                        symbol=symbol,
                        error=error_msg,
                        final_position_usd=pos_details.usd_value,
                        start_time=start_time,
                        end_time=datetime.now()
                    )
            
            success_msg = f'Position filled for {symbol[-4:]}, total value: ${round(pos_details.usd_value, 2)}'
            cprint(success_msg, 'white', 'on_green')
            
            return EZ2TradeResult(
                success=True,
                mode=EZ2TradingMode.BUY_MODE,
                symbol=symbol,
                message=success_msg,
                final_position_usd=pos_details.usd_value,
                start_time=start_time,
                end_time=datetime.now()
            )
            
        except Exception as e:
            error_msg = f"Error in buy mode for {symbol}: {e}"
            logger.error(error_msg)
            return EZ2TradeResult(
                success=False,
                mode=EZ2TradingMode.BUY_MODE,
                symbol=symbol,
                error=error_msg,
                start_time=start_time,
                end_time=datetime.now()
            )
    
    async def run_eth_trade_mode(self) -> EZ2TradeResult:
        """Mode 2: Execute trades based on ETH SMA strategy"""
        start_time = datetime.now()
        logger.info(f'--- Mode {EZ2TradingMode.ETH_TRADE_MODE}: Starting ETH SMA Trade Logic ---')
        
        try:
            # Note: This would need to be adapted to fetch ETH data from a suitable source
            # For now, we'll use a placeholder implementation
            logger.warning("ETH SMA strategy not fully implemented - requires ETH data source integration")
            
            # Placeholder for ETH data fetch - would need to integrate with Binance or similar
            # eth_df = await self.solana_service.get_ohlcv_data('ETH', days_back=200, timeframe='1d')
            
            return EZ2TradeResult(
                success=False,
                mode=EZ2TradingMode.ETH_TRADE_MODE,
                message="ETH SMA strategy requires additional data source integration",
                error="Not implemented - needs ETH data source",
                start_time=start_time,
                end_time=datetime.now()
            )
            
        except Exception as e:
            error_msg = f"Error in ETH trade mode: {e}"
            logger.error(error_msg)
            return EZ2TradeResult(
                success=False,
                mode=EZ2TradingMode.ETH_TRADE_MODE,
                error=error_msg,
                start_time=start_time,
                end_time=datetime.now()
            )
    
    async def run_pnl_close_mode(self) -> EZ2TradeResult:
        """Mode 4: Close positions if total balance is outside LOWEST_BALANCE or TARGET_BALANCE"""
        start_time = datetime.now()
        logger.info(f'--- Mode {EZ2TradingMode.PNL_CLOSE_MODE}: PNL Close Logic ---')
        
        try:
            # Get wallet holdings
            positions_df = await self.solana_service.get_wallet_holdings(self.wallet_address)
            
            if positions_df.empty:
                logger.info("No wallet holdings found or error fetching.")
                return EZ2TradeResult(
                    success=True,
                    mode=EZ2TradingMode.PNL_CLOSE_MODE,
                    message="No positions found",
                    start_time=start_time,
                    end_time=datetime.now()
                )
            
            # Calculate total portfolio value
            total_pos_value = positions_df['USD Value'].sum() if 'USD Value' in positions_df.columns else 0.0
            
            logger.info(f"Total Portfolio Value: ${round(total_pos_value, 2)}")
            logger.info(f"PNL Thresholds: Lowest=${self.lowest_balance}, Target=${self.target_balance}")
            
            # Check if balance is outside thresholds
            if total_pos_value < self.lowest_balance or total_pos_value > self.target_balance:
                action = "LOSS" if total_pos_value < self.lowest_balance else "PROFIT"
                logger.info(f"{action} Target Hit! Evaluating positions to close...")
                
                # Filter positions to close (value > 2, not in DO_NOT_TRADE_LIST)
                if 'USD Value' in positions_df.columns and 'Mint Address' in positions_df.columns:
                    positions_to_close = positions_df[
                        (positions_df['USD Value'] > 2) &
                        (~positions_df['Mint Address'].isin(self.do_not_trade_list))
                    ]
                    
                    if positions_to_close.empty:
                        logger.info("No eligible positions to close based on PNL trigger.")
                        return EZ2TradeResult(
                            success=True,
                            mode=EZ2TradingMode.PNL_CLOSE_MODE,
                            message="No eligible positions to close",
                            start_time=start_time,
                            end_time=datetime.now()
                        )
                    
                    logger.info(f"Found {len(positions_to_close)} positions to close.")
                    
                    closed_count = 0
                    for _, row in positions_to_close.iterrows():
                        symbol = row['Mint Address']
                        usd_value = row['USD Value']
                        
                        reason = (f"total value ${round(total_pos_value, 2)} < lowest ${self.lowest_balance}" 
                                if action == "LOSS" 
                                else f"total value ${round(total_pos_value, 2)} > target ${self.target_balance}")
                        
                        cprint(f'{action} KILL - Closing {symbol[-4:]} (Value: ${round(usd_value, 2)}) because {reason}', 'yellow')
                        
                        try:
                            # Close using chunk_sell_position
                            success = await self.solana_service.chunk_sell_position(
                                symbol, min(usd_value, self.max_usd_order_size), self.slippage
                            )
                            
                            if success:
                                cprint(f"PNL Close order sent for {symbol[-4:]}", 'white', 'on_magenta')
                                closed_count += 1
                            
                            await asyncio.sleep(2)  # Small delay between closes
                            
                        except Exception as e:
                            cprint(f"Error during PNL close for {symbol[-4:]}: {e}", 'red')
                            await asyncio.sleep(5)  # Pause if error
                    
                    return EZ2TradeResult(
                        success=True,
                        mode=EZ2TradingMode.PNL_CLOSE_MODE,
                        message=f"{action} trigger: closed {closed_count} positions",
                        start_time=start_time,
                        end_time=datetime.now()
                    )
            else:
                logger.info("Portfolio value is within PNL thresholds. No action taken.")
                return EZ2TradeResult(
                    success=True,
                    mode=EZ2TradingMode.PNL_CLOSE_MODE,
                    message="Portfolio within thresholds, no action taken",
                    start_time=start_time,
                    end_time=datetime.now()
                )
                
        except Exception as e:
            error_msg = f"Error during PNL Close execution: {e}"
            cprint(error_msg, 'red')
            return EZ2TradeResult(
                success=False,
                mode=EZ2TradingMode.PNL_CLOSE_MODE,
                error=error_msg,
                start_time=start_time,
                end_time=datetime.now()
            )
    
    async def run_market_maker_mode(self, symbol: str, duration_minutes: Optional[int] = None) -> EZ2TradeResult:
        """Mode 5: Execute simple buy under/sell over logic"""
        start_time = datetime.now()
        logger.info(f'--- Mode {EZ2TradingMode.MARKET_MAKER_MODE}: Market Maker for {symbol[-4:]} ---')
        logger.info(f"Using BUY_UNDER: {self.buy_under}, SELL_OVER: {self.sell_over}")
        
        try:
            iterations = 0
            max_iterations = (duration_minutes * 60 // 30) if duration_minutes else None  # 30-second cycles
            
            while self.is_running and (max_iterations is None or iterations < max_iterations):
                try:
                    pos_details = await self._get_position_details(symbol)
                    logger.info(f'State: Position: {round(pos_details.tokens, 2)}, '
                               f'Price: {round(pos_details.price, 8)}, '
                               f'Value: ${round(pos_details.usd_value, 2)}, Target Size: ${self.usd_size}')
                    
                    # --- Sell Logic ---
                    if pos_details.price > self.sell_over:
                        if pos_details.usd_value > 1:  # Only sell if we have a meaningful position
                            logger.info(f'Selling {symbol[-4:]}: Price {pos_details.price} > SELL_OVER {self.sell_over}')
                            
                            # Close the position
                            close_result = await self.run_close_mode(symbol)
                            if close_result.success:
                                cprint(f'Market maker sell triggered and position closed for {symbol[-4:]}.', 'white', 'on_magenta')
                            
                            await asyncio.sleep(15)  # Wait after selling
                        else:
                            logger.info(f"Price {pos_details.price} > SELL_OVER {self.sell_over}, but no significant position to sell (${round(pos_details.usd_value, 2)}).")
                            await asyncio.sleep(15)  # Still wait if condition met but no action
                    
                    # --- Buy Logic ---
                    elif pos_details.price < self.buy_under:
                        if pos_details.usd_value < (self.usd_size * 0.97):  # Only buy if position is not already full
                            logger.info(f'Buying {symbol[-4:]}: Price {pos_details.price} < BUY_UNDER {self.buy_under} and Position Value ${round(pos_details.usd_value, 2)} < Target ${self.usd_size}')
                            
                            try:
                                # Calculate buy amount needed
                                needed_amount = self.usd_size - pos_details.usd_value
                                chunk_size = min(needed_amount, self.max_usd_order_size)
                                chunk_lamports = str(int(chunk_size * 10**6))
                                
                                # Execute market buy
                                result = await self.solana_service.market_buy(symbol, chunk_lamports, self.slippage)
                                if result:
                                    cprint(f'Market maker buy executed for {symbol[-4:]}', 'cyan')
                                else:
                                    logger.warning(f"Market maker buy failed for {symbol[-4:]}")
                                
                                await asyncio.sleep(15)  # Wait after buying attempt
                                
                            except Exception as e:
                                cprint(f"Error during market maker buy for {symbol[-4:]}: {e}", 'red')
                                await asyncio.sleep(15)  # Wait even if error
                        else:
                            logger.info(f"Price {pos_details.price} < BUY_UNDER {self.buy_under}, but position already near target size (${round(pos_details.usd_value, 2)}).")
                            await asyncio.sleep(15)  # Still wait if condition met but no action
                    
                    # --- No Action ---
                    else:
                        logger.info(f'Price {pos_details.price} is between BUY_UNDER ({self.buy_under}) and SELL_OVER ({self.sell_over}). No action.')
                        await asyncio.sleep(30)  # Longer wait if no action needed
                    
                    iterations += 1
                    
                except asyncio.CancelledError:
                    logger.info("Market maker mode cancelled")
                    break
                except Exception as e:
                    cprint(f"Error during Market Maker execution for {symbol[-4:]}: {e}", 'red')
                    await asyncio.sleep(30)  # Wait after error
            
            return EZ2TradeResult(
                success=True,
                mode=EZ2TradingMode.MARKET_MAKER_MODE,
                symbol=symbol,
                message=f"Market maker completed {iterations} iterations",
                start_time=start_time,
                end_time=datetime.now()
            )
            
        except Exception as e:
            error_msg = f"Error in market maker mode for {symbol}: {e}"
            logger.error(error_msg)
            return EZ2TradeResult(
                success=False,
                mode=EZ2TradingMode.MARKET_MAKER_MODE,
                symbol=symbol,
                error=error_msg,
                start_time=start_time,
                end_time=datetime.now()
            )
    
    async def start_continuous_market_maker(self, symbol: str) -> str:
        """Start continuous market maker mode in background"""
        if self.active_market_maker_task and not self.active_market_maker_task.done():
            return "Market maker already running"
        
        self.active_market_maker_task = asyncio.create_task(
            self.run_market_maker_mode(symbol)
        )
        
        logger.info(f"🤖 Started continuous market maker for {symbol[-4:]}")
        return f"Market maker started for {symbol[-4:]}"
    
    async def stop_market_maker(self) -> str:
        """Stop the active market maker"""
        if self.active_market_maker_task and not self.active_market_maker_task.done():
            self.active_market_maker_task.cancel()
            try:
                await self.active_market_maker_task
            except asyncio.CancelledError:
                pass
            logger.info("🛑 Market maker stopped")
            return "Market maker stopped"
        else:
            return "No active market maker to stop"
    
    # ============== SERVICE STATUS AND UTILITIES ==============
    
    async def get_service_status(self) -> Dict[str, Any]:
        """Get comprehensive service status"""
        market_maker_status = "inactive"
        if self.active_market_maker_task and not self.active_market_maker_task.done():
            market_maker_status = "active"
        elif self.active_market_maker_task and self.active_market_maker_task.done():
            market_maker_status = "completed"
        
        return {
            "service_name": "EZ2 Solana Trading Service",
            "status": "active" if self.is_running else "inactive",
            "market_maker_status": market_maker_status,
            "available_modes": {
                "0": "Close position (in chunks)",
                "1": "Open buying position (in chunks)", 
                "2": "ETH SMA based strategy",
                "4": "Close positions based on PnL thresholds",
                "5": "Simple market making (buy under/sell over)"
            },
            "configuration": {
                "primary_symbol": self.primary_symbol_mint,
                "usd_size": self.usd_size,
                "max_usd_order_size": self.max_usd_order_size,
                "orders_per_open": self.orders_per_open,
                "tx_sleep": self.tx_sleep,
                "slippage": self.slippage,
                "lowest_balance": self.lowest_balance,
                "target_balance": self.target_balance,
                "buy_under": self.buy_under,
                "sell_over": self.sell_over
            }
        }
    
    async def execute_mode(self, mode: int, symbol: str = None, **kwargs) -> EZ2TradeResult:
        """Execute a specific trading mode"""
        symbol = symbol or self.primary_symbol_mint
        
        try:
            if mode == EZ2TradingMode.CLOSE_MODE:
                return await self.run_close_mode(symbol)
            elif mode == EZ2TradingMode.BUY_MODE:
                target_usd_size = kwargs.get('target_usd_size')
                return await self.run_buy_mode(symbol, target_usd_size)
            elif mode == EZ2TradingMode.ETH_TRADE_MODE:
                return await self.run_eth_trade_mode()
            elif mode == EZ2TradingMode.PNL_CLOSE_MODE:
                return await self.run_pnl_close_mode()
            elif mode == EZ2TradingMode.MARKET_MAKER_MODE:
                duration_minutes = kwargs.get('duration_minutes')
                return await self.run_market_maker_mode(symbol, duration_minutes)
            else:
                return EZ2TradeResult(
                    success=False,
                    mode=mode,
                    error=f"Unknown mode: {mode}",
                    start_time=datetime.now(),
                    end_time=datetime.now()
                )
        except Exception as e:
            logger.error(f"Error executing mode {mode}: {e}")
            return EZ2TradeResult(
                success=False,
                mode=mode,
                symbol=symbol,
                error=str(e),
                start_time=datetime.now(),
                end_time=datetime.now()
            )

# Convenience function for creating service instance
def create_ez2_service(config=None) -> EZ2SolanaTradingService:
    """Create and return an EZ2 Solana Trading Service instance"""
    return EZ2SolanaTradingService(config) 