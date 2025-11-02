"""
Quick Execution Strategy - File-based instant trading
Based on Day_51_Projects quick_buysell.py implementation
"""

import asyncio
import logging
import os
from typing import Dict
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from .base_strategy import BaseStrategy

logger = logging.getLogger(__name__)

class TokenFileHandler(FileSystemEventHandler):
    """File system event handler for token address monitoring"""
    
    def __init__(self, strategy):
        self.strategy = strategy
        self.processed_tokens = set()
    
    def on_modified(self, event):
        """Handle file modification events"""
        if not event.is_directory and event.src_path.endswith('token_addresses.txt'):
            asyncio.create_task(self.strategy.process_file_changes())


class QuickExecutionStrategy(BaseStrategy):
    """
    Quick execution strategy that monitors a file for instant buy/sell commands
    
    File format:
    - Contract address only = BUY
    - Contract address + space + 'c' or 'x' = SELL
    """
    
    def __init__(self, market_data_manager, portfolio_manager, risk_manager, config):
        super().__init__(market_data_manager, portfolio_manager, risk_manager, config)
        
        self.name = "quick_execution"
        self.description = "File-based instant trading for arbitrage opportunities"
        
        # Configuration
        self.token_file_path = config.get('QBS_TOKEN_ADDRESSES_FILE', 'token_addresses.txt')
        self.usdc_size = config.get('QBS_USDC_SIZE', 1)
        self.slippage_bps = config.get('QBS_SLIPPAGE_BPS', 49)
        self.priority_fee = config.get('QBS_PRIORITY_FEE_LAMPORTS', 20000)
        self.check_delay = config.get('QBS_CHECK_DELAY_SECONDS', 7)
        self.max_retries = config.get('QBS_MAX_RETRIES', 3)
        self.buys_per_batch = config.get('QBS_BUYS_PER_BATCH', 1)
        self.sells_per_batch = config.get('QBS_SELLS_PER_BATCH', 3)
        
        # File monitoring
        self.observer = None
        self.file_handler = None
        self.processed_tokens = set()
        
        # Execution tracking
        self.pending_orders = {}
        self.execution_stats = {
            'total_buys': 0,
            'total_sells': 0,
            'successful_buys': 0,
            'successful_sells': 0,
            'failed_executions': 0
        }
    
    async def initialize(self):
        """Initialize the strategy"""
        try:
            # Ensure token file exists
            if not os.path.exists(self.token_file_path):
                with open(self.token_file_path, 'w') as f:
                    f.write("")
                logger.info(f"Created token addresses file: {self.token_file_path}")
            
            # Set up file monitoring
            self.file_handler = TokenFileHandler(self)
            self.observer = Observer()
            self.observer.schedule(
                self.file_handler, 
                path=os.path.dirname(os.path.abspath(self.token_file_path)) or '.', 
                recursive=False
            )
            self.observer.start()
            
            logger.info(f"🚀 Quick execution strategy initialized - monitoring {self.token_file_path}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize quick execution strategy: {e}")
            return False
    
    async def cleanup(self):
        """Cleanup strategy resources"""
        if self.observer:
            self.observer.stop()
            self.observer.join()
        logger.info("🛑 Quick execution strategy cleaned up")
    
    async def process_file_changes(self):
        """Process changes to the token addresses file"""
        try:
            if not os.path.exists(self.token_file_path):
                return
            
            with open(self.token_file_path, 'r') as f:
                lines = [line.strip() for line in f.readlines() if line.strip()]
            
            for line in lines:
                if line not in self.processed_tokens:
                    await self.process_token_command(line)
                    self.processed_tokens.add(line)
            
            # Clear processed tokens periodically to allow re-processing
            if len(self.processed_tokens) > 1000:
                self.processed_tokens.clear()
                
        except Exception as e:
            logger.error(f"❌ Error processing file changes: {e}")
    
    async def process_token_command(self, command: str):
        """Process a single token command"""
        try:
            parts = command.split()
            token_address = parts[0]
            
            # Validate token address format (basic Solana address validation)
            if len(token_address) < 32 or len(token_address) > 44:
                logger.warning(f"⚠️ Invalid token address format: {token_address}")
                return
            
            if len(parts) == 1:
                # Buy command
                await self.execute_quick_buy(token_address)
            elif len(parts) == 2 and parts[1].lower() in ['c', 'x']:
                # Sell command
                await self.execute_quick_sell(token_address)
            else:
                logger.warning(f"⚠️ Invalid command format: {command}")
                
        except Exception as e:
            logger.error(f"❌ Error processing token command '{command}': {e}")
    
    async def execute_quick_buy(self, token_address: str):
        """Execute quick buy for token"""
        try:
            logger.info(f"🚀 QUICK BUY: {token_address[:8]}...")
            
            # Check if we already have a position
            current_position = await self.portfolio_manager.get_position(token_address)
            if current_position and current_position.quantity > 0:
                logger.info(f"⚠️ Already have position in {token_address[:8]}...")
                return
            
            # Execute buy orders in batch
            success_count = 0
            for i in range(self.buys_per_batch):
                try:
                    # Calculate USDC amount in atomic units (6 decimals for USDC)
                    usdc_amount_atomic = int(self.usdc_size * 1_000_000)
                    
                    # Create buy order
                    order = await self.portfolio_manager.create_market_order(
                        symbol=token_address,
                        side='buy',
                        quantity=usdc_amount_atomic,
                        order_type='market'
                    )
                    
                    if order:
                        success_count += 1
                        logger.info(f"✅ Buy order {i+1} executed: {order.id}")
                    
                    # Small delay between orders
                    await asyncio.sleep(0.1)
                    
                except Exception as e:
                    logger.error(f"❌ Buy order {i+1} failed: {e}")
            
            # Update stats
            self.execution_stats['total_buys'] += self.buys_per_batch
            self.execution_stats['successful_buys'] += success_count
            
            # Wait before checking position
            await asyncio.sleep(self.check_delay)
            
            # Verify position
            final_position = await self.portfolio_manager.get_position(token_address)
            if final_position and final_position.quantity > 0:
                logger.info(f"🎯 Quick buy successful - Position: {final_position.quantity}")
            else:
                logger.warning(f"⚠️ Quick buy may have failed - No position detected")
                
        except Exception as e:
            logger.error(f"❌ Quick buy failed for {token_address}: {e}")
            self.execution_stats['failed_executions'] += 1
    
    async def execute_quick_sell(self, token_address: str):
        """Execute quick sell for token"""
        try:
            logger.info(f"💰 QUICK SELL: {token_address[:8]}...")
            
            # Get current position
            current_position = await self.portfolio_manager.get_position(token_address)
            if not current_position or current_position.quantity <= 0:
                logger.warning(f"⚠️ No position to sell for {token_address[:8]}...")
                return
            
            # Calculate sell amount (70% of position by default)
            sell_percentage = self.config.get('SELL_AMOUNT_PERC', 0.7)
            sell_quantity = int(current_position.quantity * sell_percentage)
            
            # Execute sell orders in batch
            success_count = 0
            remaining_quantity = sell_quantity
            
            for i in range(self.sells_per_batch):
                if remaining_quantity <= 0:
                    break
                
                try:
                    # Calculate quantity for this order
                    order_quantity = remaining_quantity // (self.sells_per_batch - i)
                    
                    # Create sell order
                    order = await self.portfolio_manager.create_market_order(
                        symbol=token_address,
                        side='sell',
                        quantity=order_quantity,
                        order_type='market'
                    )
                    
                    if order:
                        success_count += 1
                        remaining_quantity -= order_quantity
                        logger.info(f"✅ Sell order {i+1} executed: {order.id}")
                    
                    # Small delay between orders
                    await asyncio.sleep(0.1)
                    
                except Exception as e:
                    logger.error(f"❌ Sell order {i+1} failed: {e}")
            
            # Update stats
            self.execution_stats['total_sells'] += self.sells_per_batch
            self.execution_stats['successful_sells'] += success_count
            
            # Wait before checking position
            await asyncio.sleep(self.check_delay)
            
            # Verify remaining position
            final_position = await self.portfolio_manager.get_position(token_address)
            remaining_after_sell = final_position.quantity if final_position else 0
            
            logger.info(f"🎯 Quick sell completed - Remaining position: {remaining_after_sell}")
                
        except Exception as e:
            logger.error(f"❌ Quick sell failed for {token_address}: {e}")
            self.execution_stats['failed_executions'] += 1
    
    async def should_trade(self, symbol: str, data: Dict) -> bool:
        """Quick execution strategy doesn't use traditional signals"""
        return False  # File-based execution only
    
    async def generate_signals(self, symbol: str, data: Dict) -> Dict:
        """Generate trading signals - not used for file-based execution"""
        return {
            'action': 'hold',
            'confidence': 0.0,
            'reason': 'File-based execution only'
        }
    
    async def calculate_position_size(self, symbol: str, signal: Dict) -> float:
        """Calculate position size"""
        return self.usdc_size
    
    async def get_strategy_stats(self) -> Dict:
        """Get strategy execution statistics"""
        return {
            'strategy_name': self.name,
            'execution_stats': self.execution_stats,
            'file_path': self.token_file_path,
            'processed_tokens_count': len(self.processed_tokens),
            'configuration': {
                'usdc_size': self.usdc_size,
                'slippage_bps': self.slippage_bps,
                'buys_per_batch': self.buys_per_batch,
                'sells_per_batch': self.sells_per_batch
            }
        }
    
    async def update_parameters(self, new_params: Dict):
        """Update strategy parameters"""
        if 'usdc_size' in new_params:
            self.usdc_size = new_params['usdc_size']
        if 'slippage_bps' in new_params:
            self.slippage_bps = new_params['slippage_bps']
        if 'buys_per_batch' in new_params:
            self.buys_per_batch = new_params['buys_per_batch']
        if 'sells_per_batch' in new_params:
            self.sells_per_batch = new_params['sells_per_batch']
        
        logger.info(f"📊 Quick execution strategy parameters updated: {new_params}") 