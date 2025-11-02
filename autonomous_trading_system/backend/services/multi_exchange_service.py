"""
Multi-Exchange Integration Service
Supports trading across multiple exchanges simultaneously
"""

import asyncio
import logging
import aiohttp
from datetime import datetime
from typing import Dict, List, Optional, Union
from dataclasses import dataclass
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

@dataclass
class ExchangeConfig:
    """Exchange configuration"""
    name: str
    api_key: Optional[str]
    secret_key: Optional[str]
    base_url: str
    testnet_url: Optional[str]
    rate_limit: int  # requests per minute
    supported_symbols: List[str]
    fees: Dict[str, float]  # maker, taker fees
    min_order_size: float
    max_leverage: int
    enabled: bool = True


@dataclass
class OrderBook:
    """Order book data"""
    symbol: str
    exchange: str
    bids: List[List[float]]  # [price, quantity]
    asks: List[List[float]]  # [price, quantity]
    timestamp: datetime


@dataclass
class ExchangeBalance:
    """Exchange balance data"""
    exchange: str
    symbol: str
    free: float
    locked: float
    total: float
    timestamp: datetime


@dataclass
class ArbitrageOpportunity:
    """Arbitrage opportunity data"""
    symbol: str
    buy_exchange: str
    sell_exchange: str
    buy_price: float
    sell_price: float
    profit_pct: float
    max_quantity: float
    estimated_profit: float
    timestamp: datetime


class BaseExchange(ABC):
    """Base class for exchange implementations"""
    
    def __init__(self, config: ExchangeConfig):
        self.config = config
        self.session = None
        self.rate_limiter = {}
        
    @abstractmethod
    async def connect(self):
        """Connect to exchange"""
        pass
    
    @abstractmethod
    async def disconnect(self):
        """Disconnect from exchange"""
        pass
    
    @abstractmethod
    async def get_balance(self, symbol: str = None) -> Union[ExchangeBalance, List[ExchangeBalance]]:
        """Get account balance"""
        pass
    
    @abstractmethod
    async def get_order_book(self, symbol: str, limit: int = 100) -> OrderBook:
        """Get order book"""
        pass
    
    @abstractmethod
    async def place_order(self, symbol: str, side: str, order_type: str, quantity: float, price: float = None) -> Dict:
        """Place order"""
        pass
    
    @abstractmethod
    async def cancel_order(self, symbol: str, order_id: str) -> bool:
        """Cancel order"""
        pass
    
    @abstractmethod
    async def get_ticker(self, symbol: str) -> Dict:
        """Get ticker data"""
        pass


class BinanceExchange(BaseExchange):
    """Binance exchange implementation"""
    
    async def connect(self):
        """Connect to Binance"""
        try:
            self.session = aiohttp.ClientSession()
            logger.info(f"✅ Connected to Binance")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to connect to Binance: {e}")
            return False
    
    async def disconnect(self):
        """Disconnect from Binance"""
        if self.session:
            await self.session.close()
            logger.info("🔌 Disconnected from Binance")
    
    async def get_balance(self, symbol: str = None) -> Union[ExchangeBalance, List[ExchangeBalance]]:
        """Get Binance balance"""
        try:
            # Simulate balance data
            if symbol:
                return ExchangeBalance(
                    exchange="binance",
                    symbol=symbol,
                    free=1000.0,
                    locked=0.0,
                    total=1000.0,
                    timestamp=datetime.utcnow()
                )
            else:
                # Return all balances
                symbols = ['USDT', 'BTC', 'ETH', 'SOL']
                balances = []
                for sym in symbols:
                    balances.append(ExchangeBalance(
                        exchange="binance",
                        symbol=sym,
                        free=1000.0 if sym == 'USDT' else 0.1,
                        locked=0.0,
                        total=1000.0 if sym == 'USDT' else 0.1,
                        timestamp=datetime.utcnow()
                    ))
                return balances
                
        except Exception as e:
            logger.error(f"❌ Error getting Binance balance: {e}")
            return None
    
    async def get_order_book(self, symbol: str, limit: int = 100) -> OrderBook:
        """Get Binance order book"""
        try:
            # Simulate order book data
            base_price = 50000.0 if symbol == 'BTCUSDT' else 3000.0
            
            bids = []
            asks = []
            
            for i in range(min(limit, 20)):
                bid_price = base_price - (i * 0.01 * base_price)
                ask_price = base_price + (i * 0.01 * base_price)
                quantity = 0.1 + (i * 0.01)
                
                bids.append([bid_price, quantity])
                asks.append([ask_price, quantity])
            
            return OrderBook(
                symbol=symbol,
                exchange="binance",
                bids=bids,
                asks=asks,
                timestamp=datetime.utcnow()
            )
            
        except Exception as e:
            logger.error(f"❌ Error getting Binance order book: {e}")
            return None
    
    async def place_order(self, symbol: str, side: str, order_type: str, quantity: float, price: float = None) -> Dict:
        """Place Binance order"""
        try:
            # Simulate order placement
            order_id = f"binance_{datetime.utcnow().timestamp()}"
            
            return {
                'order_id': order_id,
                'symbol': symbol,
                'side': side,
                'type': order_type,
                'quantity': quantity,
                'price': price,
                'status': 'NEW',
                'exchange': 'binance',
                'timestamp': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"❌ Error placing Binance order: {e}")
            return None
    
    async def cancel_order(self, symbol: str, order_id: str) -> bool:
        """Cancel Binance order"""
        try:
            # Simulate order cancellation
            logger.info(f"📝 Cancelled Binance order {order_id}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error cancelling Binance order: {e}")
            return False
    
    async def get_ticker(self, symbol: str) -> Dict:
        """Get Binance ticker"""
        try:
            # Simulate ticker data
            base_price = 50000.0 if 'BTC' in symbol else 3000.0
            
            return {
                'symbol': symbol,
                'price': base_price,
                'bid': base_price * 0.999,
                'ask': base_price * 1.001,
                'volume': 1000.0,
                'change_24h': 2.5,
                'exchange': 'binance',
                'timestamp': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"❌ Error getting Binance ticker: {e}")
            return None


class HyperliquidExchange(BaseExchange):
    """Hyperliquid exchange implementation"""
    
    async def connect(self):
        """Connect to Hyperliquid"""
        try:
            self.session = aiohttp.ClientSession()
            logger.info(f"✅ Connected to Hyperliquid")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to connect to Hyperliquid: {e}")
            return False
    
    async def disconnect(self):
        """Disconnect from Hyperliquid"""
        if self.session:
            await self.session.close()
            logger.info("🔌 Disconnected from Hyperliquid")
    
    async def get_balance(self, symbol: str = None) -> Union[ExchangeBalance, List[ExchangeBalance]]:
        """Get Hyperliquid balance"""
        try:
            if symbol:
                return ExchangeBalance(
                    exchange="hyperliquid",
                    symbol=symbol,
                    free=500.0,
                    locked=0.0,
                    total=500.0,
                    timestamp=datetime.utcnow()
                )
            else:
                symbols = ['USDC', 'BTC', 'ETH', 'SOL']
                balances = []
                for sym in symbols:
                    balances.append(ExchangeBalance(
                        exchange="hyperliquid",
                        symbol=sym,
                        free=500.0 if sym == 'USDC' else 0.05,
                        locked=0.0,
                        total=500.0 if sym == 'USDC' else 0.05,
                        timestamp=datetime.utcnow()
                    ))
                return balances
                
        except Exception as e:
            logger.error(f"❌ Error getting Hyperliquid balance: {e}")
            return None
    
    async def get_order_book(self, symbol: str, limit: int = 100) -> OrderBook:
        """Get Hyperliquid order book"""
        try:
            base_price = 50100.0 if symbol == 'BTC' else 3010.0
            
            bids = []
            asks = []
            
            for i in range(min(limit, 20)):
                bid_price = base_price - (i * 0.01 * base_price)
                ask_price = base_price + (i * 0.01 * base_price)
                quantity = 0.1 + (i * 0.01)
                
                bids.append([bid_price, quantity])
                asks.append([ask_price, quantity])
            
            return OrderBook(
                symbol=symbol,
                exchange="hyperliquid",
                bids=bids,
                asks=asks,
                timestamp=datetime.utcnow()
            )
            
        except Exception as e:
            logger.error(f"❌ Error getting Hyperliquid order book: {e}")
            return None
    
    async def place_order(self, symbol: str, side: str, order_type: str, quantity: float, price: float = None) -> Dict:
        """Place Hyperliquid order"""
        try:
            order_id = f"hyperliquid_{datetime.utcnow().timestamp()}"
            
            return {
                'order_id': order_id,
                'symbol': symbol,
                'side': side,
                'type': order_type,
                'quantity': quantity,
                'price': price,
                'status': 'NEW',
                'exchange': 'hyperliquid',
                'timestamp': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"❌ Error placing Hyperliquid order: {e}")
            return None
    
    async def cancel_order(self, symbol: str, order_id: str) -> bool:
        """Cancel Hyperliquid order"""
        try:
            logger.info(f"📝 Cancelled Hyperliquid order {order_id}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error cancelling Hyperliquid order: {e}")
            return False
    
    async def get_ticker(self, symbol: str) -> Dict:
        """Get Hyperliquid ticker"""
        try:
            base_price = 50100.0 if symbol == 'BTC' else 3010.0
            
            return {
                'symbol': symbol,
                'price': base_price,
                'bid': base_price * 0.999,
                'ask': base_price * 1.001,
                'volume': 800.0,
                'change_24h': 2.8,
                'exchange': 'hyperliquid',
                'timestamp': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"❌ Error getting Hyperliquid ticker: {e}")
            return None


class MultiExchangeService:
    """
    Multi-Exchange Integration Service
    
    Features:
    - Unified interface for multiple exchanges
    - Cross-exchange arbitrage detection
    - Optimal execution routing
    - Risk management across exchanges
    - Real-time price aggregation
    """
    
    def __init__(self, config):
        self.config = config
        
        # Exchange configurations
        self.exchange_configs = self._load_exchange_configs()
        
        # Exchange instances
        self.exchanges = {}
        
        # Data aggregation
        self.order_books = {}
        self.balances = {}
        self.tickers = {}
        
        # Arbitrage tracking
        self.arbitrage_opportunities = []
        self.min_profit_pct = config.get('ARB_MIN_PROFIT_BPS', 10) / 10000  # 0.1%
        self.max_slippage_pct = config.get('ARB_MAX_SLIPPAGE_BPS', 50) / 10000  # 0.5%
        
        # Performance tracking
        self.total_arbitrage_profit = 0.0
        self.successful_arbitrages = 0
        self.failed_arbitrages = 0
        
        logger.info(f"🔗 Multi-Exchange Service initialized")
        logger.info(f"📊 Configured exchanges: {list(self.exchange_configs.keys())}")
    
    def _load_exchange_configs(self) -> Dict[str, ExchangeConfig]:
        """Load exchange configurations"""
        configs = {}
        
        # Binance configuration
        if self.config.get('BINANCE_API_KEY'):
            configs['binance'] = ExchangeConfig(
                name='binance',
                api_key=self.config.get('BINANCE_API_KEY'),
                secret_key=self.config.get('BINANCE_SECRET_KEY'),
                base_url='https://api.binance.com',
                testnet_url='https://testnet.binance.vision',
                rate_limit=1200,  # requests per minute
                supported_symbols=['BTCUSDT', 'ETHUSDT', 'SOLUSDT'],
                fees={'maker': 0.001, 'taker': 0.001},
                min_order_size=10.0,
                max_leverage=125,
                enabled=True
            )
        
        # Hyperliquid configuration
        if self.config.get('HYPERLIQUID_API_KEY'):
            configs['hyperliquid'] = ExchangeConfig(
                name='hyperliquid',
                api_key=self.config.get('HYPERLIQUID_API_KEY'),
                secret_key=None,
                base_url='https://api.hyperliquid.xyz',
                testnet_url='https://api.hyperliquid-testnet.xyz',
                rate_limit=600,
                supported_symbols=['BTC', 'ETH', 'SOL', 'WIF'],
                fees={'maker': 0.0002, 'taker': 0.0005},
                min_order_size=1.0,
                max_leverage=50,
                enabled=True
            )
        
        return configs
    
    async def start(self):
        """Start the multi-exchange service"""
        try:
            logger.info("🚀 Starting Multi-Exchange Service...")
            
            # Initialize exchanges
            await self._initialize_exchanges()
            
            # Start background tasks
            tasks = [
                asyncio.create_task(self._price_aggregator()),
                asyncio.create_task(self._balance_monitor()),
                asyncio.create_task(self._arbitrage_scanner()),
                asyncio.create_task(self._order_book_updater())
            ]
            
            logger.info(f"✅ Started {len(tasks)} multi-exchange monitoring tasks")
            
            # Run all tasks concurrently
            await asyncio.gather(*tasks, return_exceptions=True)
            
        except Exception as e:
            logger.error(f"❌ Error starting multi-exchange service: {e}")
    
    async def _initialize_exchanges(self):
        """Initialize exchange connections"""
        try:
            for name, config in self.exchange_configs.items():
                if not config.enabled:
                    continue
                
                # Create exchange instance
                if name == 'binance':
                    exchange = BinanceExchange(config)
                elif name == 'hyperliquid':
                    exchange = HyperliquidExchange(config)
                else:
                    logger.warning(f"⚠️ Unknown exchange: {name}")
                    continue
                
                # Connect to exchange
                if await exchange.connect():
                    self.exchanges[name] = exchange
                    logger.info(f"✅ {name.title()} exchange initialized")
                else:
                    logger.error(f"❌ Failed to initialize {name} exchange")
            
        except Exception as e:
            logger.error(f"❌ Error initializing exchanges: {e}")
    
    async def _price_aggregator(self):
        """Aggregate prices from all exchanges"""
        try:
            logger.info("💰 Starting price aggregation...")
            
            while True:
                try:
                    await self._update_all_tickers()
                    await asyncio.sleep(5)  # Update every 5 seconds
                    
                except Exception as e:
                    logger.error(f"❌ Error in price aggregation: {e}")
                    await asyncio.sleep(10)
                    
        except Exception as e:
            logger.error(f"❌ Fatal error in price aggregator: {e}")
    
    async def _update_all_tickers(self):
        """Update tickers from all exchanges"""
        try:
            # Get common symbols across exchanges
            common_symbols = self._get_common_symbols()
            
            for symbol in common_symbols:
                symbol_tickers = {}
                
                for exchange_name, exchange in self.exchanges.items():
                    try:
                        # Convert symbol format for each exchange
                        exchange_symbol = self._convert_symbol_format(symbol, exchange_name)
                        ticker = await exchange.get_ticker(exchange_symbol)
                        
                        if ticker:
                            symbol_tickers[exchange_name] = ticker
                    
                    except Exception as e:
                        logger.error(f"❌ Error getting ticker for {symbol} on {exchange_name}: {e}")
                
                if symbol_tickers:
                    self.tickers[symbol] = symbol_tickers
            
        except Exception as e:
            logger.error(f"❌ Error updating tickers: {e}")
    
    def _get_common_symbols(self) -> List[str]:
        """Get symbols common across exchanges"""
        if not self.exchanges:
            return []
        
        # Start with symbols from first exchange
        common_symbols = set(list(self.exchange_configs.values())[0].supported_symbols)
        
        # Find intersection with other exchanges
        for config in self.exchange_configs.values():
            exchange_symbols = set(config.supported_symbols)
            common_symbols = common_symbols.intersection(exchange_symbols)
        
        # Convert to standard format
        standard_symbols = []
        for symbol in common_symbols:
            if 'BTC' in symbol:
                standard_symbols.append('BTC')
            elif 'ETH' in symbol:
                standard_symbols.append('ETH')
            elif 'SOL' in symbol:
                standard_symbols.append('SOL')
        
        return list(set(standard_symbols))
    
    def _convert_symbol_format(self, symbol: str, exchange: str) -> str:
        """Convert symbol to exchange-specific format"""
        if exchange == 'binance':
            return f"{symbol}USDT"
        elif exchange == 'hyperliquid':
            return symbol
        else:
            return symbol
    
    async def _balance_monitor(self):
        """Monitor balances across exchanges"""
        try:
            logger.info("💼 Starting balance monitoring...")
            
            while True:
                try:
                    await self._update_all_balances()
                    await asyncio.sleep(30)  # Update every 30 seconds
                    
                except Exception as e:
                    logger.error(f"❌ Error in balance monitoring: {e}")
                    await asyncio.sleep(60)
                    
        except Exception as e:
            logger.error(f"❌ Fatal error in balance monitor: {e}")
    
    async def _update_all_balances(self):
        """Update balances from all exchanges"""
        try:
            for exchange_name, exchange in self.exchanges.items():
                try:
                    balances = await exchange.get_balance()
                    
                    if balances:
                        if isinstance(balances, list):
                            self.balances[exchange_name] = {b.symbol: b for b in balances}
                        else:
                            if exchange_name not in self.balances:
                                self.balances[exchange_name] = {}
                            self.balances[exchange_name][balances.symbol] = balances
                
                except Exception as e:
                    logger.error(f"❌ Error updating balances for {exchange_name}: {e}")
            
        except Exception as e:
            logger.error(f"❌ Error updating balances: {e}")
    
    async def _arbitrage_scanner(self):
        """Scan for arbitrage opportunities"""
        try:
            logger.info("🔍 Starting arbitrage scanning...")
            
            while True:
                try:
                    await self._scan_arbitrage_opportunities()
                    await asyncio.sleep(10)  # Scan every 10 seconds
                    
                except Exception as e:
                    logger.error(f"❌ Error in arbitrage scanning: {e}")
                    await asyncio.sleep(30)
                    
        except Exception as e:
            logger.error(f"❌ Fatal error in arbitrage scanner: {e}")
    
    async def _scan_arbitrage_opportunities(self):
        """Scan for arbitrage opportunities across exchanges"""
        try:
            opportunities = []
            
            for symbol in self.tickers:
                symbol_tickers = self.tickers[symbol]
                
                if len(symbol_tickers) < 2:
                    continue
                
                # Find best bid and ask across exchanges
                best_bid = {'price': 0, 'exchange': None}
                best_ask = {'price': float('inf'), 'exchange': None}
                
                for exchange_name, ticker in symbol_tickers.items():
                    bid_price = ticker.get('bid', 0)
                    ask_price = ticker.get('ask', float('inf'))
                    
                    if bid_price > best_bid['price']:
                        best_bid = {'price': bid_price, 'exchange': exchange_name}
                    
                    if ask_price < best_ask['price']:
                        best_ask = {'price': ask_price, 'exchange': exchange_name}
                
                # Check for arbitrage opportunity
                if (best_bid['exchange'] != best_ask['exchange'] and 
                    best_bid['price'] > best_ask['price']):
                    
                    profit_pct = (best_bid['price'] - best_ask['price']) / best_ask['price']
                    
                    if profit_pct > self.min_profit_pct:
                        # Calculate maximum quantity based on balances
                        max_quantity = await self._calculate_max_arbitrage_quantity(
                            symbol, best_ask['exchange'], best_bid['exchange']
                        )
                        
                        if max_quantity > 0:
                            opportunity = ArbitrageOpportunity(
                                symbol=symbol,
                                buy_exchange=best_ask['exchange'],
                                sell_exchange=best_bid['exchange'],
                                buy_price=best_ask['price'],
                                sell_price=best_bid['price'],
                                profit_pct=profit_pct,
                                max_quantity=max_quantity,
                                estimated_profit=profit_pct * best_ask['price'] * max_quantity,
                                timestamp=datetime.utcnow()
                            )
                            
                            opportunities.append(opportunity)
                            
                            logger.info(f"💰 ARBITRAGE OPPORTUNITY: {symbol}")
                            logger.info(f"   Buy {max_quantity:.4f} on {best_ask['exchange']} @ {best_ask['price']:.2f}")
                            logger.info(f"   Sell on {best_bid['exchange']} @ {best_bid['price']:.2f}")
                            logger.info(f"   Profit: {profit_pct:.2%} (${opportunity.estimated_profit:.2f})")
            
            # Update opportunities list
            self.arbitrage_opportunities = opportunities
            
        except Exception as e:
            logger.error(f"❌ Error scanning arbitrage opportunities: {e}")
    
    async def _calculate_max_arbitrage_quantity(self, symbol: str, buy_exchange: str, sell_exchange: str) -> float:
        """Calculate maximum quantity for arbitrage trade"""
        try:
            max_quantity = 0.0
            
            # Check balance on buy exchange (need quote currency)
            buy_balances = self.balances.get(buy_exchange, {})
            quote_symbol = 'USDT' if buy_exchange == 'binance' else 'USDC'
            quote_balance = buy_balances.get(quote_symbol)
            
            if quote_balance:
                # Get current price to calculate max quantity
                ticker = self.tickers.get(symbol, {}).get(buy_exchange)
                if ticker:
                    price = ticker.get('ask', 0)
                    if price > 0:
                        max_quantity_buy = quote_balance.free / price
                    else:
                        max_quantity_buy = 0
                else:
                    max_quantity_buy = 0
            else:
                max_quantity_buy = 0
            
            # Check balance on sell exchange (need base currency)
            sell_balances = self.balances.get(sell_exchange, {})
            base_balance = sell_balances.get(symbol)
            
            if base_balance:
                max_quantity_sell = base_balance.free
            else:
                max_quantity_sell = 0
            
            # Take minimum of both
            max_quantity = min(max_quantity_buy, max_quantity_sell)
            
            # Apply safety factor
            max_quantity *= 0.9  # Use 90% of available balance
            
            return max(0, max_quantity)
            
        except Exception as e:
            logger.error(f"❌ Error calculating max arbitrage quantity: {e}")
            return 0.0
    
    async def _order_book_updater(self):
        """Update order books from exchanges"""
        try:
            logger.info("📊 Starting order book updates...")
            
            while True:
                try:
                    await self._update_order_books()
                    await asyncio.sleep(2)  # Update every 2 seconds
                    
                except Exception as e:
                    logger.error(f"❌ Error in order book updates: {e}")
                    await asyncio.sleep(10)
                    
        except Exception as e:
            logger.error(f"❌ Fatal error in order book updater: {e}")
    
    async def _update_order_books(self):
        """Update order books for all symbols"""
        try:
            common_symbols = self._get_common_symbols()
            
            for symbol in common_symbols:
                symbol_books = {}
                
                for exchange_name, exchange in self.exchanges.items():
                    try:
                        exchange_symbol = self._convert_symbol_format(symbol, exchange_name)
                        order_book = await exchange.get_order_book(exchange_symbol, limit=20)
                        
                        if order_book:
                            symbol_books[exchange_name] = order_book
                    
                    except Exception as e:
                        logger.error(f"❌ Error getting order book for {symbol} on {exchange_name}: {e}")
                
                if symbol_books:
                    self.order_books[symbol] = symbol_books
            
        except Exception as e:
            logger.error(f"❌ Error updating order books: {e}")
    
    # Public API methods
    
    async def get_best_price(self, symbol: str, side: str) -> Optional[Dict]:
        """Get best price across all exchanges"""
        try:
            if symbol not in self.tickers:
                return None
            
            best_price = None
            best_exchange = None
            
            for exchange_name, ticker in self.tickers[symbol].items():
                price = ticker.get('bid' if side == 'sell' else 'ask')
                
                if price and (best_price is None or 
                             (side == 'sell' and price > best_price) or
                             (side == 'buy' and price < best_price)):
                    best_price = price
                    best_exchange = exchange_name
            
            if best_price and best_exchange:
                return {
                    'price': best_price,
                    'exchange': best_exchange,
                    'symbol': symbol,
                    'side': side
                }
            
            return None
            
        except Exception as e:
            logger.error(f"❌ Error getting best price: {e}")
            return None
    
    async def execute_arbitrage(self, opportunity: ArbitrageOpportunity) -> bool:
        """Execute arbitrage opportunity"""
        try:
            logger.info(f"🚀 Executing arbitrage for {opportunity.symbol}")
            
            # Place buy order
            buy_exchange = self.exchanges[opportunity.buy_exchange]
            buy_symbol = self._convert_symbol_format(opportunity.symbol, opportunity.buy_exchange)
            
            buy_order = await buy_exchange.place_order(
                symbol=buy_symbol,
                side='buy',
                order_type='market',
                quantity=opportunity.max_quantity
            )
            
            if not buy_order:
                logger.error(f"❌ Failed to place buy order on {opportunity.buy_exchange}")
                return False
            
            # Place sell order
            sell_exchange = self.exchanges[opportunity.sell_exchange]
            sell_symbol = self._convert_symbol_format(opportunity.symbol, opportunity.sell_exchange)
            
            sell_order = await sell_exchange.place_order(
                symbol=sell_symbol,
                side='sell',
                order_type='market',
                quantity=opportunity.max_quantity
            )
            
            if not sell_order:
                logger.error(f"❌ Failed to place sell order on {opportunity.sell_exchange}")
                # TODO: Cancel buy order
                return False
            
            # Update statistics
            self.successful_arbitrages += 1
            self.total_arbitrage_profit += opportunity.estimated_profit
            
            logger.info(f"✅ Arbitrage executed successfully!")
            logger.info(f"   Estimated profit: ${opportunity.estimated_profit:.2f}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Error executing arbitrage: {e}")
            self.failed_arbitrages += 1
            return False
    
    async def get_arbitrage_opportunities(self) -> List[ArbitrageOpportunity]:
        """Get current arbitrage opportunities"""
        return self.arbitrage_opportunities.copy()
    
    async def get_exchange_balances(self) -> Dict:
        """Get balances across all exchanges"""
        return self.balances.copy()
    
    async def get_aggregated_prices(self) -> Dict:
        """Get aggregated prices from all exchanges"""
        return self.tickers.copy()
    
    async def place_order_on_exchange(self, exchange_name: str, symbol: str, side: str, 
                                    order_type: str, quantity: float, price: float = None) -> Dict:
        """Place order on specific exchange"""
        try:
            if exchange_name not in self.exchanges:
                raise ValueError(f"Exchange {exchange_name} not available")
            
            exchange = self.exchanges[exchange_name]
            exchange_symbol = self._convert_symbol_format(symbol, exchange_name)
            
            return await exchange.place_order(exchange_symbol, side, order_type, quantity, price)
            
        except Exception as e:
            logger.error(f"❌ Error placing order on {exchange_name}: {e}")
            return None
    
    async def get_service_stats(self) -> Dict:
        """Get multi-exchange service statistics"""
        try:
            return {
                'connected_exchanges': len(self.exchanges),
                'total_arbitrage_profit': self.total_arbitrage_profit,
                'successful_arbitrages': self.successful_arbitrages,
                'failed_arbitrages': self.failed_arbitrages,
                'current_opportunities': len(self.arbitrage_opportunities),
                'tracked_symbols': len(self.tickers),
                'exchange_configs': {name: config.name for name, config in self.exchange_configs.items()}
            }
            
        except Exception as e:
            logger.error(f"❌ Error getting service stats: {e}")
            return {}
    
    async def cleanup(self):
        """Cleanup multi-exchange service"""
        try:
            # Disconnect from all exchanges
            for exchange_name, exchange in self.exchanges.items():
                await exchange.disconnect()
            
            # Clear data structures
            self.exchanges.clear()
            self.order_books.clear()
            self.balances.clear()
            self.tickers.clear()
            self.arbitrage_opportunities.clear()
            
            logger.info("🧹 Multi-exchange service cleaned up")
            
        except Exception as e:
            logger.error(f"❌ Error cleaning up multi-exchange service: {e}") 