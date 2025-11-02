"""
WebSocket Manager for real-time data streaming
"""

import asyncio
import json
import logging
import websockets
import aiohttp
from typing import Dict, List, Callable, Optional
from datetime import datetime
from core.config import get_settings

logger = logging.getLogger(__name__)

class WebSocketManager:
    """Manages WebSocket connections for real-time data streaming"""
    
    def __init__(self):
        self.connections: Dict[str, websockets.WebSocketServerProtocol] = {}
        self.subscriptions: Dict[str, List[Callable]] = {}
        self.running = False
        
    async def start(self):
        """Start the WebSocket manager"""
        self.running = True
        logger.info("WebSocket manager started")
        
    async def stop(self):
        """Stop the WebSocket manager"""
        self.running = False
        for connection in self.connections.values():
            await connection.close()
        logger.info("WebSocket manager stopped")
        
    def subscribe(self, event_type: str, callback: Callable):
        """Subscribe to WebSocket events"""
        if event_type not in self.subscriptions:
            self.subscriptions[event_type] = []
        self.subscriptions[event_type].append(callback)
        
    async def emit(self, event_type: str, data: dict):
        """Emit data to subscribers"""
        if event_type in self.subscriptions:
            for callback in self.subscriptions[event_type]:
                try:
                    await callback(data)
                except Exception as e:
                    logger.error(f"Error in WebSocket callback: {e}")

class BinanceLiquidationStream:
    """Streams Binance liquidation data"""
    
    def __init__(self, websocket_manager: WebSocketManager):
        self.ws_manager = websocket_manager
        self.url = 'wss://fstream.binance.com/ws/!forceOrder@arr'
        
    async def start_stream(self):
        """Start streaming liquidation data"""
        while self.ws_manager.running:
            try:
                async with websockets.connect(self.url) as websocket:
                    logger.info("Connected to Binance liquidation stream")
                    
                    while self.ws_manager.running:
                        try:
                            msg = await websocket.recv()
                            await self._process_liquidation(json.loads(msg))
                        except Exception as e:
                            logger.error(f"Error processing liquidation: {e}")
                            await asyncio.sleep(1)
                            
            except Exception as e:
                logger.error(f"Liquidation stream error: {e}")
                await asyncio.sleep(5)
                
    async def _process_liquidation(self, data: dict):
        """Process liquidation data"""
        try:
            order_data = data['o']
            symbol = order_data['s'].replace('USDT', '')
            side = order_data['S']
            timestamp = int(order_data['T'])
            filled_quantity = float(order_data['z'])
            price = float(order_data['p'])
            usd_size = filled_quantity * price
            
            if usd_size > get_settings().MIN_LIQUIDATION_SIZE:
                liquidation_data = {
                    'symbol': symbol,
                    'side': side,
                    'price': price,
                    'size': usd_size,
                    'timestamp': timestamp,
                    'type': 'liquidation'
                }
                
                await self.ws_manager.emit('liquidation', liquidation_data)
                
        except Exception as e:
            logger.error(f"Error processing liquidation data: {e}")

class SolanaTokenScanner:
    """Scans for new Solana tokens"""
    
    def __init__(self, websocket_manager: WebSocketManager):
        self.ws_manager = websocket_manager
        self.helius_url = f"wss://mainnet.helius-rpc.com/?api-key={get_settings().HELIUS_API_KEY}"
        self.raydium_program_id = "675kPX9MHTjS2zt1qfr1NYHuzeLXfQM9H24wFSUt1Mp8"
        self.seen_signatures = set()
        
    async def start_scanner(self):
        """Start scanning for new tokens"""
        while self.ws_manager.running:
            try:
                async with websockets.connect(self.helius_url) as websocket:
                    logger.info("Connected to Solana token scanner")
                    
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
                    
                    while self.ws_manager.running:
                        try:
                            message = await asyncio.wait_for(websocket.recv(), timeout=60)
                            await self._process_token_message(message)
                        except asyncio.TimeoutError:
                            await websocket.ping()
                        except Exception as e:
                            logger.error(f"Error processing token message: {e}")
                            
            except Exception as e:
                logger.error(f"Token scanner error: {e}")
                await asyncio.sleep(5)
                
    async def _process_token_message(self, message: str):
        """Process token discovery message"""
        try:
            data = json.loads(message)
            
            if "params" in data and "result" in data["params"]:
                result_value = data["params"]["result"]["value"]
                signature = result_value.get("signature")
                
                if signature and signature not in self.seen_signatures:
                    self.seen_signatures.add(signature)
                    logs = result_value.get("logs", [])
                    
                    if any("initialize2" in log for log in logs):
                        token_address = await self._get_new_token(signature)
                        if token_address:
                            token_data = {
                                'address': token_address,
                                'signature': signature,
                                'timestamp': datetime.utcnow().isoformat(),
                                'type': 'new_token'
                            }
                            
                            await self.ws_manager.emit('new_token', token_data)
                            
        except Exception as e:
            logger.error(f"Error processing token message: {e}")
            
    async def _get_new_token(self, signature: str) -> Optional[str]:
        """Extract new token address from transaction"""
        try:
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
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    "https://api.mainnet-beta.solana.com",
                    json=request_body,
                    timeout=10
                ) as response:
                    data = await response.json()
                    
                    if "error" in data:
                        return None
                        
                    instructions = data.get("result", {}).get("transaction", {}).get("message", {}).get("instructions", [])
                    
                    for instruction in instructions:
                        if (instruction.get("programId") == self.raydium_program_id and 
                            instruction.get("data", "").startswith("16a40b14bb677619")):
                            accounts = instruction.get("accounts", [])
                            if len(accounts) > 8:
                                return accounts[8]
                                
        except Exception as e:
            logger.error(f"Error getting new token: {e}")
            
        return None

class PriceStreamManager:
    """Manages real-time price streams"""
    
    def __init__(self, websocket_manager: WebSocketManager):
        self.ws_manager = websocket_manager
        self.price_streams = {}
        
    async def subscribe_to_price(self, symbol: str, exchange: str = "binance"):
        """Subscribe to price updates for a symbol"""
        if exchange == "binance":
            await self._start_binance_price_stream(symbol)
        elif exchange == "hyperliquid":
            await self._start_hyperliquid_price_stream(symbol)
            
    async def _start_binance_price_stream(self, symbol: str):
        """Start Binance price stream"""
        url = f"wss://stream.binance.com:9443/ws/{symbol.lower()}@ticker"
        
        async with websockets.connect(url) as websocket:
            while self.ws_manager.running:
                try:
                    msg = await websocket.recv()
                    data = json.loads(msg)
                    
                    price_data = {
                        'symbol': data['s'],
                        'price': float(data['c']),
                        'change': float(data['P']),
                        'volume': float(data['v']),
                        'timestamp': int(data['E']),
                        'exchange': 'binance',
                        'type': 'price_update'
                    }
                    
                    await self.ws_manager.emit('price_update', price_data)
                    
                except Exception as e:
                    logger.error(f"Binance price stream error: {e}")
                    
    async def _start_hyperliquid_price_stream(self, symbol: str):
        """Start HyperLiquid price stream"""
        # Implementation for HyperLiquid WebSocket
        pass 