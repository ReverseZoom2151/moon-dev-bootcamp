"""
Solana Token Scanner Service
Monitors Raydium for new token launches and provides comprehensive token discovery
Based on Day_27_Projects/solscanner.py with enterprise enhancements
"""

import asyncio
import json
import csv
import os
import platform
import logging
import aiohttp
import websockets
from datetime import datetime, timezone
from typing import Optional, Dict, List, Set, Any
from dataclasses import dataclass, asdict
from websockets.exceptions import ConnectionClosed, WebSocketException
from core.config import get_settings

logger = logging.getLogger(__name__)

@dataclass
class NewTokenData:
    """Data structure for newly discovered tokens"""
    token_address: str
    signature: str
    timestamp: datetime
    time_found: str
    epoch_time: int
    solscan_link: str
    dexscreener_link: str
    birdeye_link: str
    block_time: Optional[int] = None
    slot: Optional[int] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        return data

@dataclass
class ScannerStats:
    """Scanner performance statistics"""
    total_signatures_processed: int = 0
    total_tokens_discovered: int = 0
    initialize2_events_detected: int = 0
    failed_transaction_fetches: int = 0
    websocket_reconnections: int = 0
    uptime_seconds: float = 0.0
    last_heartbeat: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        data = asdict(self)
        if self.last_heartbeat:
            data['last_heartbeat'] = self.last_heartbeat.isoformat()
        return data

class SolanaTokenScannerService:
    """
    Enterprise Solana Token Scanner Service
    
    Features:
    - Real-time monitoring of Raydium program for new token launches
    - Automatic reconnection and error recovery
    - CSV data persistence with comprehensive token information
    - Performance statistics and monitoring
    - Integration with autonomous trading system
    - Configurable output formats and destinations
    - Rate limiting and API optimization
    """
    
    def __init__(self, websocket_manager=None, config=None):
        self.settings = config or get_settings()
        self.websocket_manager = websocket_manager
        
        # Core configuration
        self.helius_api_key = self.settings.HELIUS_API_KEY
        if not self.helius_api_key:
            raise ValueError("HELIUS_API_KEY is required for token scanner")
            
        self.helius_wss_url = f"wss://mainnet.helius-rpc.com/?api-key={self.helius_api_key}"
        self.raydium_program_id = "675kPX9MHTjS2zt1qfr1NYHuzeLXfQM9H24wFSUt1Mp8"
        self.solana_rpc_url = self.settings.SOLANA_RPC_URL or "https://api.mainnet-beta.solana.com"
        
        # File configuration
        self.csv_file = os.environ.get("SOLSCANNER_CSV_FILE", "./data/new_sol_tokens.csv")
        self.ensure_data_directory()
        
        # State management
        self.seen_signatures: Set[str] = set()
        self.discovered_tokens: List[NewTokenData] = []
        self.stats = ScannerStats()
        self.is_running = False
        self.start_time: Optional[datetime] = None
        
        # Configuration
        self.heartbeat_interval = 30  # seconds
        self.max_seen_signatures = 10000  # Prevent memory bloat
        self.websocket_timeout = 60  # seconds
        self.transaction_fetch_timeout = 10  # seconds
        self.max_reconnect_attempts = 5
        self.reconnect_delay = 5  # seconds
        
        # Configure event loop for Windows compatibility
        self._configure_event_loop()
        
        logger.info("🔍 Solana Token Scanner Service initialized")
        logger.info(f"📁 Output file: {self.csv_file}")
        logger.info(f"🔗 Monitoring Raydium Program: {self.raydium_program_id}")
        logger.info(f"🌐 Using Solana RPC: {self.solana_rpc_url}")
    
    def _configure_event_loop(self):
        """Configure event loop for Windows compatibility"""
        if platform.system() == 'Windows':
            asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
            logger.info("🪟 Configured Windows SelectorEventLoop for compatibility")
    
    def ensure_data_directory(self):
        """Ensure the data directory exists"""
        try:
            os.makedirs(os.path.dirname(self.csv_file), exist_ok=True)
        except Exception as e:
            logger.error(f"❌ Error creating data directory: {e}")
    
    def ensure_csv_file_exists(self):
        """Ensure the CSV file exists with proper headers"""
        try:
            if not os.path.exists(self.csv_file) or os.path.getsize(self.csv_file) == 0:
                with open(self.csv_file, 'w', newline='', encoding='utf-8') as file:
                    writer = csv.writer(file)
                    writer.writerow([
                        "Token Address", "Time Found", "Epoch Time", 
                        "Solscan Link", "DexScreener Link", "Birdeye Link",
                        "Transaction Signature", "Block Time", "Slot"
                    ])
                logger.info(f"📄 Created CSV file: {self.csv_file}")
        except IOError as e:
            logger.error(f"❌ Error ensuring CSV file exists: {e}")
            raise
    
    async def start(self):
        """Start the token scanner service"""
        if self.is_running:
            logger.warning("⚠️ Token scanner is already running")
            return
            
        self.is_running = True
        self.start_time = datetime.now(timezone.utc)
        self.stats = ScannerStats()
        
        logger.info("🚀 Starting Solana Token Scanner Service")
        
        # Ensure CSV file is ready
        self.ensure_csv_file_exists()
        
        # Start main scanner loop
        await self._run_scanner_loop()
    
    async def stop(self):
        """Stop the token scanner service"""
        self.is_running = False
        logger.info("🛑 Solana Token Scanner Service stopped")
        
        # Log final statistics
        if self.start_time:
            self.stats.uptime_seconds = (datetime.now(timezone.utc) - self.start_time).total_seconds()
        
        logger.info(f"📊 Final Statistics: {self.stats.to_dict()}")
    
    async def _run_scanner_loop(self):
        """Main scanner loop with reconnection logic"""
        reconnect_attempts = 0
        
        while self.is_running:
            try:
                await self._run_websocket_scanner()
                reconnect_attempts = 0  # Reset on successful connection
                
            except Exception as e:
                reconnect_attempts += 1
                self.stats.websocket_reconnections += 1
                
                logger.error(f"❌ Scanner error (attempt {reconnect_attempts}): {e}")
                
                if reconnect_attempts >= self.max_reconnect_attempts:
                    logger.error(f"💥 Max reconnection attempts ({self.max_reconnect_attempts}) reached")
                    break
                
                logger.info(f"🔄 Reconnecting in {self.reconnect_delay} seconds...")
                await asyncio.sleep(self.reconnect_delay)
    
    async def _run_websocket_scanner(self):
        """Run the WebSocket scanner with heartbeat"""
        async with websockets.connect(self.helius_wss_url) as websocket:
            logger.info("🔗 Connected to Helius WebSocket")
            
            # Subscribe to Raydium program logs
            subscribe_msg = {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "logsSubscribe",
                "params": [
                    {"mentions": [self.raydium_program_id]},
                    {"commitment": "finalized"}
                ]
            }
            
            await websocket.send(json.dumps(subscribe_msg))
            logger.info("📡 Subscription message sent to Helius")
            
            # Start heartbeat task
            heartbeat_task = asyncio.create_task(self._heartbeat_loop())
            
            try:
                async with aiohttp.ClientSession() as session:
                    while self.is_running:
                        try:
                            # Wait for message with timeout
                            message = await asyncio.wait_for(
                                websocket.recv(), 
                                timeout=self.websocket_timeout
                            )
                            await self._process_message(message, session)
                            
                        except asyncio.TimeoutError:
                            logger.debug("⏰ WebSocket timeout, sending ping")
                            await websocket.ping()
                            
                        except (ConnectionClosed, WebSocketException) as e:
                            logger.warning(f"🔌 WebSocket connection issue: {e}")
                            break
                            
                        except Exception as e:
                            logger.error(f"❌ Error processing message: {e}")
                            await asyncio.sleep(1)  # Prevent tight error loop
                            
            finally:
                heartbeat_task.cancel()
                logger.info("🔌 WebSocket connection closed")
    
    async def _heartbeat_loop(self):
        """Periodic heartbeat to confirm scanner is running"""
        counter = 0
        
        while self.is_running:
            try:
                await asyncio.sleep(self.heartbeat_interval)
                counter += 1
                
                # Update statistics
                if self.start_time:
                    self.stats.uptime_seconds = (datetime.now(timezone.utc) - self.start_time).total_seconds()
                self.stats.last_heartbeat = datetime.now(timezone.utc)
                
                logger.info(
                    f"💓 Scanner heartbeat #{counter} | "
                    f"Tokens discovered: {self.stats.total_tokens_discovered} | "
                    f"Signatures processed: {self.stats.total_signatures_processed} | "
                    f"Uptime: {self.stats.uptime_seconds:.0f}s"
                )
                
                # Emit heartbeat via WebSocket if available
                if self.websocket_manager:
                    await self.websocket_manager.emit('scanner_heartbeat', {
                        'counter': counter,
                        'stats': self.stats.to_dict(),
                        'timestamp': datetime.now(timezone.utc).isoformat()
                    })
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"❌ Error in heartbeat loop: {e}")
    
    async def _process_message(self, message: str, session: aiohttp.ClientSession):
        """Process a single WebSocket message"""
        try:
            data = json.loads(message)
        except json.JSONDecodeError:
            logger.warning(f"⚠️ Failed to decode JSON: {message[:100]}...")
            return
        
        # Extract transaction data
        if not ("params" in data and "result" in data["params"]):
            return
            
        result_value = data["params"]["result"]["value"]
        signature = result_value.get("signature")
        
        if not signature or signature in self.seen_signatures:
            return
        
        # Add signature to seen set (with memory management)
        self.seen_signatures.add(signature)
        if len(self.seen_signatures) > self.max_seen_signatures:
            # Remove oldest signatures (simple approach)
            self.seen_signatures = set(list(self.seen_signatures)[-self.max_seen_signatures//2:])
        
        self.stats.total_signatures_processed += 1
        
        # Check for initialize2 events
        logs = result_value.get("logs", [])
        if any("initialize2" in log for log in logs):
            self.stats.initialize2_events_detected += 1
            logger.info(f"🎯 Detected 'initialize2' in signature: {signature}")
            
            # Extract token information
            token_address = await self._get_new_token(session, signature)
            if token_address:
                await self._handle_new_token_discovery(token_address, signature, result_value)
    
    async def _get_new_token(self, session: aiohttp.ClientSession, signature: str) -> Optional[str]:
        """Extract new token address from transaction"""
        request_body = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "getTransaction",
            "params": [
                signature,
                {
                    "encoding": "jsonParsed",
                    "maxSupportedTransactionVersion": 0
                }
            ]
        }
        
        try:
            async with session.post(
                self.solana_rpc_url,
                json=request_body,
                timeout=self.transaction_fetch_timeout
            ) as response:
                response.raise_for_status()
                data = await response.json()
                
                if "error" in data:
                    logger.error(f"❌ RPC error for {signature}: {data['error']}")
                    self.stats.failed_transaction_fetches += 1
                    return None
                
                # Extract token address from transaction
                transaction = data.get("result", {}).get("transaction", {})
                instructions = transaction.get("message", {}).get("instructions", [])
                
                for instruction in instructions:
                    if (instruction.get("programId") == self.raydium_program_id and 
                        instruction.get("data", "").startswith("16a40b14bb677619")):  # initialize2 discriminator
                        
                        accounts = instruction.get("accounts", [])
                        if len(accounts) > 8:
                            token_address = accounts[8]  # Base token account position
                            logger.debug(f"✅ Extracted token: {token_address} from {signature}")
                            return token_address
                        else:
                            logger.warning(f"⚠️ Insufficient accounts in instruction for {signature}")
                
        except aiohttp.ClientError as e:
            logger.error(f"❌ HTTP error fetching transaction {signature}: {e}")
            self.stats.failed_transaction_fetches += 1
        except asyncio.TimeoutError:
            logger.error(f"⏰ Timeout fetching transaction {signature}")
            self.stats.failed_transaction_fetches += 1
        except Exception as e:
            logger.error(f"❌ Unexpected error fetching transaction {signature}: {e}")
            self.stats.failed_transaction_fetches += 1
        
        return None
    
    async def _handle_new_token_discovery(self, token_address: str, signature: str, result_value: Dict):
        """Handle discovery of a new token"""
        now = datetime.now(timezone.utc)
        
        # Create token data
        token_data = NewTokenData(
            token_address=token_address,
            signature=signature,
            timestamp=now,
            time_found=now.strftime("%Y-%m-%d %H:%M:%S"),
            epoch_time=int(now.timestamp()),
            solscan_link=f"https://solscan.io/tx/{signature}",
            dexscreener_link=f"https://dexscreener.com/solana/{token_address}",
            birdeye_link=f"https://birdeye.so/token/{token_address}?chain=solana",
            block_time=result_value.get("blockTime"),
            slot=result_value.get("slot")
        )
        
        # Update statistics
        self.stats.total_tokens_discovered += 1
        self.discovered_tokens.append(token_data)
        
        # Log discovery
        logger.info(
            f"🎉 NEW TOKEN DISCOVERED: {token_address} | "
            f"Tx: https://solscan.io/tx/{signature} | "
            f"Total discovered: {self.stats.total_tokens_discovered}"
        )
        
        # Save to CSV
        await self._save_token_to_csv(token_data)
        
        # Emit via WebSocket if available
        if self.websocket_manager:
            await self.websocket_manager.emit('new_token_discovered', token_data.to_dict())
        
        # Keep memory usage reasonable
        if len(self.discovered_tokens) > 1000:
            self.discovered_tokens = self.discovered_tokens[-500:]  # Keep last 500
    
    async def _save_token_to_csv(self, token_data: NewTokenData):
        """Save token data to CSV file"""
        try:
            # Use asyncio to avoid blocking
            await asyncio.get_event_loop().run_in_executor(
                None, self._write_csv_row, token_data
            )
        except Exception as e:
            logger.error(f"❌ Error saving token to CSV: {e}")
    
    def _write_csv_row(self, token_data: NewTokenData):
        """Write a single row to CSV (synchronous)"""
        try:
            with open(self.csv_file, 'a', newline='', encoding='utf-8') as file:
                writer = csv.writer(file)
                writer.writerow([
                    token_data.token_address,
                    token_data.time_found,
                    token_data.epoch_time,
                    token_data.solscan_link,
                    token_data.dexscreener_link,
                    token_data.birdeye_link,
                    token_data.signature,
                    token_data.block_time,
                    token_data.slot
                ])
        except IOError as e:
            logger.error(f"❌ Error writing to CSV: {e}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get current scanner statistics"""
        if self.start_time:
            self.stats.uptime_seconds = (datetime.now(timezone.utc) - self.start_time).total_seconds()
        
        return {
            'stats': self.stats.to_dict(),
            'config': {
                'csv_file': self.csv_file,
                'raydium_program_id': self.raydium_program_id,
                'solana_rpc_url': self.solana_rpc_url,
                'heartbeat_interval': self.heartbeat_interval
            },
            'recent_tokens': [token.to_dict() for token in self.discovered_tokens[-10:]]  # Last 10
        }
    
    def get_recent_tokens(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get recently discovered tokens"""
        return [token.to_dict() for token in self.discovered_tokens[-limit:]]
    
    async def export_tokens_json(self, filepath: str) -> bool:
        """Export discovered tokens to JSON file"""
        try:
            data = {
                'export_timestamp': datetime.now(timezone.utc).isoformat(),
                'total_tokens': len(self.discovered_tokens),
                'statistics': self.stats.to_dict(),
                'tokens': [token.to_dict() for token in self.discovered_tokens]
            }
            
            await asyncio.get_event_loop().run_in_executor(
                None, self._write_json_file, filepath, data
            )
            
            logger.info(f"📤 Exported {len(self.discovered_tokens)} tokens to {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error exporting tokens to JSON: {e}")
            return False
    
    def _write_json_file(self, filepath: str, data: Dict):
        """Write JSON file (synchronous)"""
        with open(filepath, 'w', encoding='utf-8') as file:
            json.dump(data, file, indent=2, ensure_ascii=False) 