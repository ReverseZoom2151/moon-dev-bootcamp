"""
WebSocket Manager
=================
Manages WebSocket connections for real-time market data.
Exchange-agnostic WebSocket handling.
"""

import asyncio, json, logging
from datetime import datetime
from typing import Dict, List, Optional, Any
from websockets import connect, WebSocketClientProtocol
from websockets.exceptions import WebSocketException
    
class WebSocketManager:
    """
    Unified WebSocket connection manager for all exchanges.

    Handles:
    - Connection management
    - Reconnection logic
    - Message routing
    - Error handling
    """

    def __init__(self, event_bus: Any, market_data_stream: Any):
        """
        Initialize WebSocket manager.

        Args:
            event_bus: Event bus for publishing events
            market_data_stream: Market data stream handler
        """
        self.event_bus = event_bus
        self.market_data_stream = market_data_stream
        self.logger = logging.getLogger(__name__)

        # Active connections
        self.connections: Dict[str, WebSocketClientProtocol] = {}
        self.connection_tasks: Dict[str, asyncio.Task] = {}

        # Exchange-specific configurations
        self.exchange_configs = {
            'binance': {
                'base_url': 'wss://fstream.binance.com/ws/',
                'streams': {
                    'liquidations': '!forceOrder@arr',
                    'trades': '@aggTrade',
                    'funding': '@markPrice',
                    'ticker': '@ticker',
                    'orderbook': '@depth20@100ms'
                }
            },
            'bitfinex': {
                'base_url': 'wss://api-pub.bitfinex.com/ws/2',
                'streams': {
                    'trades': 'trades',
                    'ticker': 'ticker',
                    'orderbook': 'book'
                }
            },
            'hyperliquid': {
                'base_url': 'wss://api.hyperliquid.xyz/ws',
                'streams': {
                    'trades': 'trades',
                    'orderbook': 'book',
                    'liquidations': 'liquidations'
                }
            }
        }

        # Message handlers by exchange
        self.message_handlers = {
            'binance': self._handle_binance_message,
            'bitfinex': self._handle_bitfinex_message,
            'hyperliquid': self._handle_hyperliquid_message
        }

        self.is_running = False
        self.reconnect_delays = {}  # Track reconnection delays per connection

    async def connect_stream(self, exchange: str, stream_type: str,
                            symbols: Optional[List[str]] = None):
        """
        Connect to a specific stream.

        Args:
            exchange: Exchange name
            stream_type: Type of stream (liquidations, trades, funding, etc.)
            symbols: Optional list of symbols to subscribe to
        """
        if exchange not in self.exchange_configs:
            self.logger.error(f"Unknown exchange: {exchange}")
            return

        config = self.exchange_configs[exchange]

        # Build WebSocket URL based on exchange and stream type
        if exchange == 'binance':
            url = await self._build_binance_url(config, stream_type, symbols)
        elif exchange == 'bitfinex':
            url = config['base_url']  # Bitfinex uses channel subscriptions
        elif exchange == 'hyperliquid':
            url = config['base_url']
        else:
            url = config['base_url']

        # Create connection ID
        connection_id = f"{exchange}_{stream_type}"
        if symbols:
            connection_id += f"_{'_'.join(symbols[:3])}"  # Add first 3 symbols to ID

        # Start connection task
        task = asyncio.create_task(
            self._maintain_connection(connection_id, url, exchange, stream_type, symbols)
        )
        self.connection_tasks[connection_id] = task

        self.logger.info(f"Started stream: {connection_id}")

    async def _maintain_connection(self, connection_id: str, url: str,
                                  exchange: str, stream_type: str,
                                  symbols: Optional[List[str]]):
        """
        Maintain a WebSocket connection with auto-reconnect.

        Args:
            connection_id: Unique connection identifier
            url: WebSocket URL
            exchange: Exchange name
            stream_type: Stream type
            symbols: Symbols to subscribe to
        """
        self.reconnect_delays[connection_id] = 5  # Initial reconnect delay

        while self.is_running:
            try:
                self.logger.info(f"Connecting to {connection_id}: {url}")

                async with connect(url) as websocket:
                    self.connections[connection_id] = websocket

                    # Subscribe to channels if needed (Bitfinex, HyperLiquid)
                    if exchange in ['bitfinex', 'hyperliquid']:
                        await self._subscribe_to_channels(
                            websocket, exchange, stream_type, symbols
                        )

                    # Reset reconnect delay on successful connection
                    self.reconnect_delays[connection_id] = 5

                    # Emit connection event
                    await self.event_bus.emit("websocket_connected", {
                        'connection_id': connection_id,
                        'exchange': exchange,
                        'stream_type': stream_type
                    })

                    # Process messages
                    await self._process_messages(
                        websocket, connection_id, exchange, stream_type
                    )

            except WebSocketException as e:
                self.logger.error(f"WebSocket error on {connection_id}: {e}")
            except Exception as e:
                self.logger.error(f"Unexpected error on {connection_id}: {e}")

            # Connection lost, clean up and reconnect
            if connection_id in self.connections:
                del self.connections[connection_id]

            # Emit disconnection event
            await self.event_bus.emit("websocket_disconnected", {
                'connection_id': connection_id,
                'exchange': exchange,
                'stream_type': stream_type
            })

            if self.is_running:
                # Exponential backoff for reconnection
                delay = self.reconnect_delays[connection_id]
                self.logger.info(f"Reconnecting {connection_id} in {delay} seconds...")
                await asyncio.sleep(delay)

                # Increase delay for next attempt (max 60 seconds)
                self.reconnect_delays[connection_id] = min(delay * 2, 60)

    async def _process_messages(self, websocket: WebSocketClientProtocol,
                               connection_id: str, exchange: str,
                               stream_type: str):
        """
        Process incoming WebSocket messages.

        Args:
            websocket: WebSocket connection
            connection_id: Connection identifier
            exchange: Exchange name
            stream_type: Stream type
        """
        async for message in websocket:
            try:
                # Parse message
                if isinstance(message, str):
                    data = json.loads(message)
                else:
                    data = message

                # Route to appropriate handler
                if exchange in self.message_handlers:
                    await self.message_handlers[exchange](data, stream_type)
                else:
                    await self._handle_generic_message(data, exchange, stream_type)

            except json.JSONDecodeError as e:
                self.logger.error(f"Failed to parse message from {connection_id}: {e}")
            except Exception as e:
                self.logger.error(f"Error processing message from {connection_id}: {e}")

    async def _build_binance_url(self, config: Dict, stream_type: str,
                                symbols: Optional[List[str]]) -> str:
        """Build Binance WebSocket URL."""
        base_url = config['base_url']
        stream_suffix = config['streams'].get(stream_type, '')

        if stream_type == 'liquidations':
            # Liquidations use a global stream
            return base_url + stream_suffix

        elif symbols:
            # Symbol-specific streams
            streams = []
            for symbol in symbols:
                symbol_lower = symbol.lower()
                streams.append(f"{symbol_lower}{stream_suffix}")
            return base_url + '/'.join(streams)

        else:
            # Default to BTC for single symbol streams
            return base_url + f"btcusdt{stream_suffix}"

    async def _subscribe_to_channels(self, websocket: WebSocketClientProtocol,
                                    exchange: str, stream_type: str,
                                    symbols: Optional[List[str]]):
        """Subscribe to channels for exchanges that require it."""
        if exchange == 'bitfinex':
            # Bitfinex subscription format
            for symbol in symbols or ['BTCUSD']:
                subscribe_msg = {
                    "event": "subscribe",
                    "channel": stream_type,
                    "symbol": f"t{symbol}"
                }
                await websocket.send(json.dumps(subscribe_msg))

        elif exchange == 'hyperliquid':
            # HyperLiquid subscription format
            subscribe_msg = {
                "method": "subscribe",
                "subscription": {
                    "type": stream_type,
                    "coins": symbols or ["BTC"]
                }
            }
            await websocket.send(json.dumps(subscribe_msg))

    # Message handlers for different exchanges

    async def _handle_binance_message(self, data: Dict, stream_type: str):
        """Handle Binance WebSocket messages."""
        if stream_type == 'liquidations':
            if 'o' in data:  # Liquidation order data
                await self.market_data_stream.process_liquidation('binance', data)

        elif stream_type == 'trades':
            if 'e' in data and data['e'] == 'aggTrade':
                await self.market_data_stream.process_trade('binance', data)

        elif stream_type == 'funding':
            if 'e' in data and data['e'] == 'markPriceUpdate':
                await self.market_data_stream.process_funding('binance', data)

    async def _handle_bitfinex_message(self, data: Any, stream_type: str):
        """Handle Bitfinex WebSocket messages."""
        if isinstance(data, list):
            # Bitfinex sends arrays for channel data
            if len(data) > 2:
                channel_id = data[0]
                event_type = data[1]

                if event_type == 'te':  # Trade execution
                    trade_data = {
                        'id': data[2],
                        'timestamp': data[3],
                        'amount': data[4],
                        'price': data[5]
                    }
                    await self.market_data_stream.process_trade('bitfinex', trade_data)

    async def _handle_hyperliquid_message(self, data: Dict, stream_type: str):
        """Handle HyperLiquid WebSocket messages."""
        if 'channel' in data:
            channel = data['channel']

            if channel == 'trades':
                for trade in data.get('data', []):
                    await self.market_data_stream.process_trade('hyperliquid', trade)

            elif channel == 'liquidations':
                for liq in data.get('data', []):
                    await self.market_data_stream.process_liquidation('hyperliquid', liq)

    async def _handle_generic_message(self, data: Dict, exchange: str, stream_type: str):
        """Handle generic WebSocket messages."""
        # Emit raw message for custom handling
        await self.event_bus.emit(f"{exchange}_{stream_type}_message", {
            'exchange': exchange,
            'stream_type': stream_type,
            'data': data,
            'timestamp': datetime.now().isoformat()
        })

    async def start(self):
        """Start the WebSocket manager."""
        self.is_running = True
        self.logger.info("WebSocket manager started")

    async def stop(self):
        """Stop all WebSocket connections."""
        self.is_running = False

        # Cancel all connection tasks
        for task in self.connection_tasks.values():
            task.cancel()

        # Wait for tasks to complete
        await asyncio.gather(*self.connection_tasks.values(), return_exceptions=True)

        # Close all connections
        for connection in self.connections.values():
            await connection.close()

        self.connections.clear()
        self.connection_tasks.clear()

        self.logger.info("WebSocket manager stopped")

    async def disconnect_stream(self, connection_id: str):
        """Disconnect a specific stream."""
        if connection_id in self.connection_tasks:
            self.connection_tasks[connection_id].cancel()
            await self.connection_tasks[connection_id]
            del self.connection_tasks[connection_id]

        if connection_id in self.connections:
            await self.connections[connection_id].close()
            del self.connections[connection_id]

        self.logger.info(f"Disconnected stream: {connection_id}")

    def get_connection_status(self) -> Dict:
        """Get status of all connections."""
        return {
            'active_connections': list(self.connections.keys()),
            'total_connections': len(self.connections),
            'is_running': self.is_running,
            'reconnect_delays': self.reconnect_delays
        }

    async def setup_default_streams(self, exchange: str, symbols: List[str]):
        """
        Setup default streams for an exchange.
        This replicates the functionality from Day 2 projects.

        Args:
            exchange: Exchange name
            symbols: List of symbols to monitor
        """
        # Setup streams similar to Day 2 projects
        if exchange == 'binance':
            # Liquidations (binance_liqs.py / binance_big_liqs.py)
            await self.connect_stream(exchange, 'liquidations')

            # Large trades (binance_huge_trades.py / binance_recent_trades.py)
            await self.connect_stream(exchange, 'trades', symbols)

            # Funding rates (binance_funding.py)
            await self.connect_stream(exchange, 'funding', symbols)

        elif exchange == 'bitfinex':
            # Similar setup for Bitfinex
            await self.connect_stream(exchange, 'trades', symbols)

        elif exchange == 'hyperliquid':
            # Similar setup for HyperLiquid
            await self.connect_stream(exchange, 'trades', symbols)
            await self.connect_stream(exchange, 'liquidations')

        self.logger.info(f"Setup default streams for {exchange} with symbols: {symbols}")