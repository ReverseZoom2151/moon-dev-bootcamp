"""
Token Scanner Utility
=====================
Day 27: WebSocket-based scanner for new token launches (Solana).

Features:
- Monitors Raydium program for new token initializations
- Saves new tokens to CSV with metadata
- Provides links to Solscan, DexScreener, Birdeye
- Detects 'initialize2' events
"""

import asyncio
import json
import csv
import os
import logging
import platform
from datetime import datetime
from typing import Optional, Set, Dict
import aiohttp
from websockets import connect

logger = logging.getLogger(__name__)


class TokenScanner:
    """
    Token scanner for detecting new token launches.
    
    Monitors Solana blockchain for new token initializations via WebSocket.
    """

    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize token scanner.
        
        Args:
            config: Configuration dictionary with:
                - helius_ws_api_key: Helius WebSocket API key
                - csv_file: Path to CSV file for saving tokens
                - raydium_program_id: Raydium program ID to monitor
                - solana_rpc_url: Solana RPC URL
        """
        self.config = config or {}
        self.helius_api_key = self.config.get('helius_ws_api_key')
        self.csv_file = self.config.get('csv_file', './new_sol_tokens.csv')
        self.raydium_program_id = self.config.get(
            'raydium_program_id',
            '675kPX9MHTjS2zt1qfr1NYHuzeLXfQM9H24wFSUt1Mp8'
        )
        self.solana_rpc_url = self.config.get(
            'solana_rpc_url',
            'https://api.mainnet-beta.solana.com'
        )
        
        self.wss_url = f"wss://mainnet.helius-rpc.com/?api-key={self.helius_api_key}" if self.helius_api_key else None
        self.seen_signatures: Set[str] = set()
        
        # Ensure CSV file exists
        self._ensure_csv_file()

    def _ensure_csv_file(self):
        """Ensure CSV file exists with headers."""
        try:
            if not os.path.exists(self.csv_file) or os.path.getsize(self.csv_file) == 0:
                with open(self.csv_file, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        'Token Address',
                        'Time Found',
                        'Epoch Time',
                        'Solscan Link',
                        'DexScreener Link',
                        'Birdeye Link'
                    ])
                logger.info(f"Created CSV file: {self.csv_file}")
        except Exception as e:
            logger.error(f"Error ensuring CSV file exists: {e}")

    def _save_to_csv(self, token_address: str, signature: str):
        """
        Save new token to CSV.
        
        Args:
            token_address: Token mint address
            signature: Transaction signature
        """
        try:
            with open(self.csv_file, 'a', newline='') as f:
                writer = csv.writer(f)
                now = datetime.utcnow()
                time_found = now.strftime("%Y-%m-%d %H:%M:%S")
                epoch_time = int(now.timestamp())
                
                solscan_link = f"https://solscan.io/tx/{signature}"
                dexscreener_link = f"https://dexscreener.com/solana/{token_address}"
                birdeye_link = f"https://birdeye.so/token/{token_address}?chain=solana"
                
                writer.writerow([
                    token_address,
                    time_found,
                    epoch_time,
                    solscan_link,
                    dexscreener_link,
                    birdeye_link
                ])
                
            logger.info(f"Saved token {token_address} to CSV")
        except Exception as e:
            logger.error(f"Error writing to CSV: {e}")

    async def _get_new_token(
        self,
        session: aiohttp.ClientSession,
        signature: str
    ) -> Optional[str]:
        """
        Fetch transaction details and extract new token address.
        
        Args:
            session: aiohttp session
            signature: Transaction signature
            
        Returns:
            Token address or None
        """
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
                timeout=10
            ) as response:
                response.raise_for_status()
                data = await response.json()
                
                if "error" in data:
                    logger.error(f"Error in getTransaction response: {data['error']}")
                    return None
                
                instructions = (
                    data.get("result", {})
                    .get("transaction", {})
                    .get("message", {})
                    .get("instructions", [])
                )
                
                for instruction in instructions:
                    program_id = instruction.get("programId")
                    data_hex = instruction.get("data", "")
                    
                    # Check for initialize2 instruction
                    if (
                        program_id == self.raydium_program_id and
                        data_hex.startswith("16a40b14bb677619")  # initialize2 discriminator
                    ):
                        accounts = instruction.get("accounts", [])
                        if len(accounts) > 8:
                            token_address = accounts[8]  # Base token account index
                            logger.debug(f"Extracted token address {token_address}")
                            return token_address
                        else:
                            logger.warning(f"Instruction accounts length <= 8 for {signature}")
                
        except aiohttp.ClientError as e:
            logger.error(f"HTTP Client error fetching transaction {signature}: {e}")
        except asyncio.TimeoutError:
            logger.error(f"Timeout fetching transaction {signature}")
        except Exception as e:
            logger.error(f"Unexpected error fetching transaction {signature}: {e}")
        
        return None

    async def _process_message(
        self,
        message: str,
        session: aiohttp.ClientSession
    ):
        """
        Process WebSocket message.
        
        Args:
            message: WebSocket message
            session: aiohttp session
        """
        try:
            data = json.loads(message)
        except json.JSONDecodeError:
            logger.warning(f"Failed to decode JSON message: {message[:100]}...")
            return
        
        if "params" in data and "result" in data["params"]:
            result_value = data["params"]["result"]["value"]
            signature = result_value.get("signature")
            
            if signature and signature not in self.seen_signatures:
                self.seen_signatures.add(signature)
                logs = result_value.get("logs", [])
                
                # Check for initialize2 log
                if any("initialize2" in log for log in logs):
                    logger.info(f"Detected 'initialize2' in logs for signature: {signature}")
                    new_token = await self._get_new_token(session, signature)
                    
                    if new_token:
                        logger.info(
                            f"New token found: {new_token} "
                            f"(Tx: https://solscan.io/tx/{signature})"
                        )
                        self._save_to_csv(new_token, signature)

    async def _heartbeat(self):
        """Periodic heartbeat to confirm scanner is running."""
        counter = 0
        while True:
            await asyncio.sleep(30)
            counter += 1
            logger.info(f"Still listening for new tokens... (heartbeat #{counter})")

    async def start(self):
        """Start the token scanner."""
        if not self.wss_url:
            logger.error("Helius WebSocket API key not configured")
            return
        
        logger.info(f"Starting TokenScanner. Output: {self.csv_file}")
        logger.info(f"Monitoring Raydium Program ID: {self.raydium_program_id}")
        
        # Configure event loop for Windows
        if platform.system() == 'Windows':
            asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
            logger.info("Configured Windows to use SelectorEventLoop")
        
        while True:
            try:
                await self._run_websocket()
            except Exception as e:
                logger.error(f"Error in WebSocket connection: {e}", exc_info=True)
                logger.info("Attempting to reconnect in 5 seconds...")
                await asyncio.sleep(5)

    async def _run_websocket(self):
        """Run WebSocket connection."""
        async with connect(self.wss_url) as websocket:
            logger.info("Connected to WebSocket")
            
            subscribe_msg = {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "logsSubscribe",
                "params": [
                    {
                        "mentions": [self.raydium_program_id]
                    },
                    {
                        "commitment": "finalized"
                    }
                ]
            }
            
            await websocket.send(json.dumps(subscribe_msg))
            logger.info("Subscription message sent")
            
            async with aiohttp.ClientSession() as session:
                heartbeat_task = asyncio.create_task(self._heartbeat())
                
                try:
                    while True:
                        try:
                            message = await asyncio.wait_for(websocket.recv(), timeout=60)
                            await self._process_message(message, session)
                        except asyncio.TimeoutError:
                            logger.debug("WebSocket timeout, sending ping")
                            await websocket.ping()
                        except Exception as e:
                            logger.error(f"Error processing message: {e}", exc_info=True)
                            await asyncio.sleep(1)
                
                finally:
                    heartbeat_task.cancel()
                    logger.info("WebSocket loop terminated")

