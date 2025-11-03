"""
Quick Buy/Sell Strategy (Day 51)
==================================
Rapid execution strategy that monitors a file for instant buy/sell commands.

File format:
- Token symbol only = BUY
- Token symbol + 'x' or 'c' = SELL

Features:
- Instant market order execution
- Real-time file monitoring
- Position tracking and verification
- Supports Binance and Bitfinex
"""

import asyncio
import logging
import time
from typing import Dict, Optional, Any
from pathlib import Path

from ..strategies.base import BaseStrategy
from ...exchanges.factory import ExchangeFactory
from ..utilities.file_monitor import QuickBuySellMonitor


class QuickBuySellStrategy(BaseStrategy):
    """
    Quick Buy/Sell execution strategy.
    
    Monitors a text file for token symbols and executes trades instantly.
    Perfect for rapid arbitrage opportunities or new token launches.
    """
    
    def __init__(self, exchange_adapter=None, config: Optional[Dict] = None):
        """
        Initialize Quick Buy/Sell strategy.
        
        Args:
            exchange_adapter: Exchange adapter instance
            config: Strategy configuration
        """
        super().__init__(name="QuickBuySell")
        self.exchange_adapter = exchange_adapter
        self.config = config or {}
        self.monitor: Optional[QuickBuySellMonitor] = None
        self.is_monitoring = False
        
        # Configuration
        self.file_path = self.config.get('token_file_path', './token_addresses.txt')
        self.usd_size = self.config.get('usd_size', 10.0)
        self.check_delay = self.config.get('check_delay', 3)
        self.max_retries = self.config.get('max_retries', 3)
        self.buys_per_batch = self.config.get('buys_per_batch', 1)
        self.sells_per_batch = self.config.get('sells_per_batch', 2)
        self.sell_percentage = self.config.get('sell_percentage', 0.8)
        self.min_position_value = self.config.get('min_position_value', 5.0)
    
    async def initialize(self, config: Dict):
        """Initialize strategy."""
        await super().initialize(config)
        self.config.update(config)
        
        # Initialize exchange adapter if not provided
        if not self.exchange_adapter:
            exchange_name = self.config.get('exchange', 'binance')
            exchange_config = self.config.get('exchange_config', {})
            event_bus = self.config.get('event_bus')
            
            self.exchange_adapter = ExchangeFactory.create_exchange(
                exchange_name,
                exchange_config,
                event_bus=event_bus
            )
            
            await self.exchange_adapter.initialize()
        
        self.logger.info(f"Quick Buy/Sell Strategy initialized for {self.exchange_adapter.__class__.__name__}")
    
    async def execute(self) -> Optional[Dict]:
        """
        Execute strategy - start monitoring if not already running.
        
        Returns:
            None (this strategy runs continuously)
        """
        if not self.is_monitoring:
            await self.start_monitoring()
        
        return None
    
    async def start_monitoring(self):
        """Start monitoring the token file."""
        if self.is_monitoring:
            self.logger.warning("Monitor is already running")
            return
        
        try:
            self.monitor = QuickBuySellMonitor(
                file_path=self.file_path,
                on_token_added=self._on_token_added,
                config=self.config
            )
            
            # Start monitoring in background
            self.monitor.start()
            self.is_monitoring = True
            
            self.logger.info(f"Started monitoring file: {self.file_path}")
            self.logger.info("Add token symbols to file to trigger trades:")
            self.logger.info("  - 'BTCUSDT' = BUY")
            self.logger.info("  - 'BTCUSDT x' = SELL")
        
        except Exception as e:
            self.logger.error(f"Failed to start monitoring: {e}")
            raise
    
    async def stop_monitoring(self):
        """Stop monitoring the token file."""
        if self.monitor:
            self.monitor.stop()
        self.is_monitoring = False
        self.logger.info("Stopped monitoring file")
    
    def _on_token_added(self, token_symbol: str, command: Optional[str]):
        """
        Handle token added to file.
        
        Args:
            token_symbol: Token symbol
            command: 'BUY' or 'SELL'
        """
        # Run in async context
        asyncio.create_task(self._process_token_command(token_symbol, command))
    
    async def _process_token_command(self, token_symbol: str, command: Optional[str]):
        """
        Process token command.
        
        Args:
            token_symbol: Token symbol
            command: 'BUY' or 'SELL'
        """
        try:
            self.logger.info(f"📝 Processing command: {command} {token_symbol}")
            
            if command == 'BUY':
                await self._quick_buy(token_symbol)
            elif command == 'SELL':
                await self._quick_sell(token_symbol)
            else:
                self.logger.warning(f"Unknown command: {command}")
        
        except Exception as e:
            self.logger.error(f"Error processing token command: {e}")
    
    async def _quick_buy(self, symbol: str):
        """
        Execute quick buy.
        
        Args:
            symbol: Trading symbol
        """
        try:
            self.logger.info(f"🟢 QUICK BUY: {symbol} for ${self.usd_size:.2f}")
            
            # Check if we already have a position
            position = await self.exchange_adapter.get_position(symbol)
            if position and abs(float(position.get('quantity', 0))) > 0.0001:
                self.logger.info(f"⚠️ Already have position in {symbol}")
                return
            
            # Get current price to calculate quantity
            ticker = await self.exchange_adapter.get_ticker(symbol)
            if not ticker:
                self.logger.error(f"Could not get price for {symbol}")
                return
            
            current_price = float(ticker.get('last', 0))
            if current_price == 0:
                self.logger.error(f"Invalid price for {symbol}")
                return
            
            # Calculate quantity
            quantity = self.usd_size / current_price
            
            # Execute buy orders in batch
            success_count = 0
            for i in range(self.buys_per_batch):
                try:
                    order = await self.exchange_adapter.place_order(
                        symbol=symbol,
                        side='buy',
                        amount=quantity / self.buys_per_batch,
                        order_type='market'
                    )
                    
                    if order:
                        success_count += 1
                        self.logger.info(f"✅ Buy order {i+1}/{self.buys_per_batch} executed")
                    
                    # Small delay between orders
                    await asyncio.sleep(0.1)
                
                except Exception as e:
                    self.logger.error(f"❌ Buy order {i+1} failed: {e}")
            
            # Wait before checking position
            await asyncio.sleep(self.check_delay)
            
            # Verify position
            final_position = await self.exchange_adapter.get_position(symbol)
            if final_position:
                position_value = float(final_position.get('quantity', 0)) * current_price
                self.logger.info(f"💰 Position value: ${position_value:.2f}")
        
        except Exception as e:
            self.logger.error(f"Error in quick_buy: {e}")
    
    async def _quick_sell(self, symbol: str):
        """
        Execute quick sell.
        
        Args:
            symbol: Trading symbol
        """
        try:
            self.logger.info(f"🔴 QUICK SELL: {symbol}")
            
            # Check if we have a position
            position = await self.exchange_adapter.get_position(symbol)
            if not position or abs(float(position.get('quantity', 0))) < 0.0001:
                self.logger.warning(f"⚠️ No position to sell for {symbol}")
                return
            
            # Get current price
            ticker = await self.exchange_adapter.get_ticker(symbol)
            if not ticker:
                self.logger.error(f"Could not get price for {symbol}")
                return
            
            current_price = float(ticker.get('last', 0))
            position_value = abs(float(position.get('quantity', 0))) * current_price
            
            if position_value < self.min_position_value:
                self.logger.warning(f"⚠️ Position too small to sell: ${position_value:.2f}")
                return
            
            # Execute sell orders in batch
            quantity = abs(float(position.get('quantity', 0)))
            sell_quantity_per_order = (quantity * self.sell_percentage) / self.sells_per_batch
            
            for i in range(self.sells_per_batch):
                try:
                    order = await self.exchange_adapter.place_order(
                        symbol=symbol,
                        side='sell',
                        amount=sell_quantity_per_order,
                        order_type='market'
                    )
                    
                    if order:
                        self.logger.info(f"✅ Sell order {i+1}/{self.sells_per_batch} executed")
                    
                    # Brief pause between orders
                    await asyncio.sleep(0.5)
                
                except Exception as e:
                    self.logger.error(f"❌ Sell order {i+1} failed: {e}")
            
            # Wait and check remaining position
            await asyncio.sleep(self.check_delay)
            
            remaining_position = await self.exchange_adapter.get_position(symbol)
            if remaining_position:
                remaining_value = abs(float(remaining_position.get('quantity', 0))) * current_price
                self.logger.info(f"💰 Remaining position: ${remaining_value:.2f}")
        
        except Exception as e:
            self.logger.error(f"Error in quick_sell: {e}")
    
    async def cleanup(self):
        """Cleanup strategy resources."""
        await self.stop_monitoring()
        await super().cleanup()
    
    async def quick_buy_manual(self, symbol: str, usd_amount: Optional[float] = None):
        """
        Manually trigger quick buy.
        
        Args:
            symbol: Trading symbol
            usd_amount: USD amount to buy (uses config default if not provided)
        """
        original_size = self.usd_size
        if usd_amount:
            self.usd_size = usd_amount
        try:
            await self._quick_buy(symbol)
        finally:
            self.usd_size = original_size
    
    async def quick_sell_manual(self, symbol: str):
        """
        Manually trigger quick sell.
        
        Args:
            symbol: Trading symbol
        """
        await self._quick_sell(symbol)

