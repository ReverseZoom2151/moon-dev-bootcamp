"""
Quick Buy/Sell Service
---
This service monitors a text file for token addresses and executes rapid
buy or sell orders based on the file's content. It's designed for quick,
manual trading interventions.

Adapted from the Day 51 `quick_buysell.py` script.
"""
import asyncio
import logging
import os
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from services.solana_jupiter_service import SolanaJupiterService

logger = logging.getLogger(__name__)

class QuickBuySellService:
    def __init__(self, settings):
        self.settings = settings
        self.solana_jupiter_service = SolanaJupiterService(private_key=self.settings.SOL_KEY2)
        
        # Absolute path for the file to watch
        self.file_to_watch = os.path.join(self.settings.BASE_DIR, self.settings.QBS_TOKEN_ADDRESSES_FILE)
        self.ensure_token_file_exists()

        self.last_processed = set()
        self.observer = Observer()
        self.is_running = False

    def ensure_token_file_exists(self):
        """Creates the token address file if it doesn't exist."""
        if not os.path.exists(self.file_to_watch):
            logger.info(f"Token file not found at {self.file_to_watch}. Creating it now.")
            try:
                with open(self.file_to_watch, 'w') as f:
                    f.write("# Quick Buy/Sell Bot Token List\n")
                    f.write("# To buy, add a token address on a new line.\n")
                    f.write("# To sell, add the token address followed by ' x' or ' c'.\n")
                    f.write("# Example (BUY): 7i5KKsX2p2UpfDV4i3t212syT5H52TfG551AbZ3W4xY\n")
                    f.write("# Example (SELL): 7i5KKsX2p2UpfDV4i3t212syT5H52TfG551AbZ3W4xY x\n")
                logger.info(f"Successfully created {self.file_to_watch}")
            except IOError as e:
                logger.error(f"Failed to create token file: {e}")
                
    async def quick_buy(self, token_address: str):
        """Executes a quick buy operation."""
        logger.info(f"🚀 INITIATING QUICK BUY for token: {token_address}")
        
        usdc_size = self.settings.QBS_USDC_SIZE
        # USDC has 6 decimals
        amount_atomic = int(usdc_size * 1_000_000)

        # Initial buy attempt
        tx_id = self.solana_jupiter_service.market_swap(
            input_mint="EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v", # USDC
            output_mint=token_address,
            amount_atomic=amount_atomic,
            slippage_bps=self.settings.QBS_SLIPPAGE_BPS,
            priority_fee=self.settings.QBS_PRIORITY_FEE_LAMPORTS
        )

        if not tx_id:
            logger.error(f"❌ Initial buy failed for {token_address}")
            return

        for i in range(self.settings.QBS_MAX_RETRIES):
            await asyncio.sleep(self.settings.QBS_CHECK_DELAY_SECONDS)
            balance_info = self.solana_jupiter_service.get_wallet_token_balance(token_address)
            
            # Check if we have at least 90% of the intended buy size in USD value
            # Note: This requires a price check, which adds complexity.
            # A simpler check is just if we have a non-zero balance.
            if balance_info['uiAmount'] > 0:
                 logger.info(f"✅ Position successfully opened for {token_address} with {balance_info['uiAmount']} tokens.")
                 return
            
            logger.warning(f"Position not yet detected for {token_address}. Retrying buy ({i+1}/{self.settings.QBS_MAX_RETRIES}).")
            for _ in range(self.settings.QBS_BUYS_PER_BATCH):
                 self.solana_jupiter_service.market_swap("EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v", token_address, amount_atomic, self.settings.QBS_SLIPPAGE_BPS, self.settings.QBS_PRIORITY_FEE_LAMPORTS)

        logger.error(f"❌ BUY FAILED for {token_address} after all retries.")

    async def quick_sell(self, token_address: str):
        """Executes a quick sell operation."""
        logger.info(f"🚀 INITIATING QUICK SELL for token: {token_address}")

        for i in range(self.settings.QBS_MAX_RETRIES):
            balance_info = self.solana_jupiter_service.get_wallet_token_balance(token_address)
            atomic_balance = balance_info['amount']

            if atomic_balance <= 0:
                logger.info(f"✅ Position for {token_address} is successfully closed.")
                return

            logger.warning(f"Attempting to sell {balance_info['uiAmount']} of {token_address} (Attempt {i+1}/{self.settings.QBS_MAX_RETRIES})")

            for _ in range(self.settings.QBS_SELLS_PER_BATCH):
                self.solana_jupiter_service.market_swap(token_address, "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v", atomic_balance, self.settings.QBS_SLIPPAGE_BPS, self.settings.QBS_PRIORITY_FEE_LAMPORTS)
            
            await asyncio.sleep(self.settings.QBS_CHECK_DELAY_SECONDS)
        
        logger.error(f"❌ SELL FAILED for {token_address}. Position may still be open.")

    def process_token_file(self):
        """Reads the token file and processes new entries."""
        try:
            with open(self.file_to_watch, 'r') as f:
                lines = [line.strip() for line in f if line.strip() and not line.startswith('#')]
            
            current_entries = set(lines)
            new_entries = current_entries - self.last_processed

            if not new_entries:
                return

            logger.info(f"Detected {len(new_entries)} new entries in token file.")

            for entry in new_entries:
                parts = entry.split()
                token_address = parts[0]
                command = 'buy' # Default command
                
                if len(parts) > 1 and parts[1].lower() in ['x', 'c']:
                    command = 'sell'
                
                logger.info(f"Processing command '{command}' for token {token_address}")
                if command == 'buy':
                    asyncio.create_task(self.quick_buy(token_address))
                else:
                    asyncio.create_task(self.quick_sell(token_address))
            
            self.last_processed = current_entries

        except Exception as e:
            logger.error(f"Error processing token file: {e}", exc_info=True)
            
    async def start(self):
        """Starts the file monitoring service."""
        if not self.settings.ENABLE_QUICK_BUY_SELL_BOT:
            logger.info("Quick Buy/Sell Bot is disabled in settings.")
            return

        logger.info("Starting Quick Buy/Sell Bot Service...")
        self.is_running = True
        
        # Initial check of the file
        self.process_token_file()

        event_handler = FileSystemEventHandler()
        event_handler.on_modified = lambda event: self.on_file_modified(event)
        
        watch_path = os.path.dirname(self.file_to_watch)
        self.observer.schedule(event_handler, watch_path, recursive=False)
        self.observer.start()
        
        logger.info(f"👀 Monitoring {self.file_to_watch} for changes...")
        
        while self.is_running:
            await asyncio.sleep(1) # Keep the service alive

    def on_file_modified(self, event):
        if event.src_path == self.file_to_watch:
            logger.info("Token file modified. Processing changes...")
            self.process_token_file()

    def stop(self):
        """Stops the file monitoring service."""
        self.is_running = False
        self.observer.stop()
        self.observer.join()
        logger.info("Quick Buy/Sell Bot Service stopped.")
