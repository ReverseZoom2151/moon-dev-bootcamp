"""
Multi-Exchange Arbitrage Strategy
Identifies and exploits price differences across multiple exchanges
"""

import logging
import asyncio
import numpy as np
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime
from dataclasses import dataclass
from strategies.base_strategy import BaseStrategy, StrategySignal, SignalAction, TechnicalIndicatorMixin

logger = logging.getLogger(__name__)

@dataclass
class ExchangePrice:
    """Price information from an exchange"""
    exchange: str
    symbol: str
    bid: float
    ask: float
    timestamp: datetime
    volume_24h: float = 0
    fees: float = 0.001  # Default 0.1% fee


@dataclass
class ArbitrageOpportunity:
    """Arbitrage opportunity between exchanges"""
    symbol: str
    buy_exchange: str
    sell_exchange: str
    buy_price: float
    sell_price: float
    profit_pct: float
    profit_after_fees: float
    volume_available: float
    confidence: float
    timestamp: datetime


class ExchangeConnector:
    """Mock exchange connector for demonstration"""
    
    def __init__(self, name: str, base_prices: Dict[str, float]):
        self.name = name
        self.base_prices = base_prices
        self.fees = {
            'binance': 0.001,
            'coinbase': 0.005,
            'kraken': 0.0026,
            'hyperliquid': 0.0002,
            'bybit': 0.001
        }.get(name.lower(), 0.001)
    
    async def get_ticker(self, symbol: str) -> Optional[ExchangePrice]:
        """Get current ticker price (simulated)"""
        try:
            if symbol not in self.base_prices:
                return None
            
            base_price = self.base_prices[symbol]
            
            # Add some random variation to simulate real exchange differences
            variation = np.random.normal(0, 0.002)  # 0.2% standard deviation
            price = base_price * (1 + variation)
            
            # Simulate bid-ask spread
            spread = base_price * 0.0005  # 0.05% spread
            bid = price - spread / 2
            ask = price + spread / 2
            
            # Simulate volume
            volume_24h = np.random.uniform(1000000, 10000000)
            
            return ExchangePrice(
                exchange=self.name,
                symbol=symbol,
                bid=bid,
                ask=ask,
                timestamp=datetime.utcnow(),
                volume_24h=volume_24h,
                fees=self.fees
            )
            
        except Exception as e:
            logger.error(f"❌ Error getting ticker for {symbol} on {self.name}: {e}")
            return None
    
    async def get_order_book(self, symbol: str, depth: int = 10) -> Dict[str, List[Tuple[float, float]]]:
        """Get order book (simulated)"""
        try:
            ticker = await self.get_ticker(symbol)
            if not ticker:
                return {'bids': [], 'asks': []}
            
            # Simulate order book
            bids = []
            asks = []
            
            for i in range(depth):
                bid_price = ticker.bid - (i * ticker.bid * 0.0001)
                ask_price = ticker.ask + (i * ticker.ask * 0.0001)
                
                bid_volume = np.random.uniform(0.1, 10.0)
                ask_volume = np.random.uniform(0.1, 10.0)
                
                bids.append((bid_price, bid_volume))
                asks.append((ask_price, ask_volume))
            
            return {'bids': bids, 'asks': asks}
            
        except Exception as e:
            logger.error(f"❌ Error getting order book for {symbol} on {self.name}: {e}")
            return {'bids': [], 'asks': []}


class ArbitrageStrategy(BaseStrategy, TechnicalIndicatorMixin):
    """
    Multi-Exchange Arbitrage Trading Strategy
    
    Features:
    - Real-time price monitoring across exchanges
    - Arbitrage opportunity detection
    - Risk assessment and filtering
    - Execution optimization
    - Latency and slippage consideration
    """
    
    def __init__(self, config: Dict[str, Any], market_data_manager, name: str = "Arbitrage"):
        super().__init__(config, market_data_manager, name)
        
        # Arbitrage Configuration
        self.exchanges = config.get("exchanges", ["binance", "coinbase", "kraken", "hyperliquid"])
        self.min_profit_pct = config.get("min_profit_pct", 0.005)  # 0.5% minimum profit
        self.max_position_size = config.get("max_position_size", 10000)  # $10k max position
        self.execution_timeout = config.get("execution_timeout", 5)  # 5 seconds
        
        # Risk parameters
        self.max_slippage = config.get("max_slippage", 0.001)  # 0.1% max slippage
        self.min_volume_ratio = config.get("min_volume_ratio", 0.01)  # 1% of 24h volume
        self.latency_buffer = config.get("latency_buffer", 0.0005)  # 0.05% latency buffer
        
        # Exchange connectors
        self.connectors = {}
        self._initialize_connectors()
        
        # Opportunity tracking
        self.opportunities = []
        self.executed_arbitrages = []
        self.price_cache = {}
        
        # Performance metrics
        self.total_arbitrages = 0
        self.successful_arbitrages = 0
        self.total_profit = 0
        
        logger.info(f"🔄 Arbitrage Strategy initialized:")
        logger.info(f"   Exchanges: {self.exchanges}")
        logger.info(f"   Min Profit: {self.min_profit_pct*100:.2f}%")
        logger.info(f"   Symbols: {self.symbols}")
    
    def _initialize_connectors(self):
        """Initialize exchange connectors"""
        # Base prices for simulation
        base_prices = {
            'BTC': 45000,
            'ETH': 2800,
            'SOL': 100,
            'WIF': 2.5,
            'POPCAT': 1.2
        }
        
        for exchange in self.exchanges:
            self.connectors[exchange] = ExchangeConnector(exchange, base_prices)
            logger.info(f"📡 Initialized connector for {exchange}")
    
    async def _initialize_strategy(self):
        """Initialize arbitrage strategy"""
        try:
            # Test all exchange connections
            for exchange, connector in self.connectors.items():
                for symbol in self.symbols[:2]:  # Test first 2 symbols
                    ticker = await connector.get_ticker(symbol)
                    if ticker:
                        logger.info(f"✅ {exchange} connection test successful for {symbol}")
                    else:
                        logger.warning(f"⚠️ {exchange} connection test failed for {symbol}")
            
            logger.info("✅ Arbitrage strategy validation complete")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize arbitrage strategy: {e}")
            raise
    
    async def generate_signal(self) -> Optional[StrategySignal]:
        """Generate arbitrage trading signal"""
        try:
            # Scan for arbitrage opportunities
            opportunities = await self._scan_arbitrage_opportunities()
            
            if not opportunities:
                return None
            
            # Select best opportunity
            best_opportunity = max(opportunities, key=lambda x: x.profit_after_fees)
            
            # Validate opportunity
            if await self._validate_opportunity(best_opportunity):
                signal = await self._create_arbitrage_signal(best_opportunity)
                return signal
            
            return None
            
        except Exception as e:
            logger.error(f"❌ Error generating arbitrage signal: {e}", exc_info=True)
            return None
    
    async def _scan_arbitrage_opportunities(self) -> List[ArbitrageOpportunity]:
        """Scan all exchanges for arbitrage opportunities"""
        opportunities = []
        
        try:
            # Get prices from all exchanges
            all_prices = await self._fetch_all_prices()
            
            # Compare prices across exchanges
            for symbol in self.symbols:
                if symbol not in all_prices:
                    continue
                
                symbol_prices = all_prices[symbol]
                if len(symbol_prices) < 2:
                    continue
                
                # Find arbitrage opportunities
                symbol_opportunities = self._find_arbitrage_pairs(symbol, symbol_prices)
                opportunities.extend(symbol_opportunities)
            
            # Filter and rank opportunities
            filtered_opportunities = self._filter_opportunities(opportunities)
            
            logger.debug(f"📊 Found {len(opportunities)} raw opportunities, {len(filtered_opportunities)} after filtering")
            
            return filtered_opportunities
            
        except Exception as e:
            logger.error(f"❌ Error scanning arbitrage opportunities: {e}")
            return []
    
    async def _fetch_all_prices(self) -> Dict[str, List[ExchangePrice]]:
        """Fetch prices from all exchanges"""
        all_prices = {}
        
        # Create tasks for parallel fetching
        tasks = []
        for symbol in self.symbols:
            for exchange, connector in self.connectors.items():
                task = asyncio.create_task(connector.get_ticker(symbol))
                tasks.append((symbol, exchange, task))
        
        # Wait for all tasks
        results = await asyncio.gather(*[task for _, _, task in tasks], return_exceptions=True)
        
        # Process results
        for (symbol, exchange, _), result in zip(tasks, results):
            if isinstance(result, ExchangePrice):
                if symbol not in all_prices:
                    all_prices[symbol] = []
                all_prices[symbol].append(result)
                
                # Update price cache
                cache_key = f"{symbol}_{exchange}"
                self.price_cache[cache_key] = result
        
        return all_prices
    
    def _find_arbitrage_pairs(self, symbol: str, prices: List[ExchangePrice]) -> List[ArbitrageOpportunity]:
        """Find arbitrage opportunities for a symbol"""
        opportunities = []
        
        # Compare all exchange pairs
        for i, buy_price in enumerate(prices):
            for j, sell_price in enumerate(prices):
                if i == j:  # Same exchange
                    continue
                
                # Calculate potential profit
                profit_pct = (sell_price.bid - buy_price.ask) / buy_price.ask
                
                # Calculate profit after fees
                total_fees = buy_price.fees + sell_price.fees
                profit_after_fees = profit_pct - total_fees
                
                if profit_after_fees > self.min_profit_pct:
                    # Estimate available volume
                    volume_available = min(
                        buy_price.volume_24h * self.min_volume_ratio,
                        sell_price.volume_24h * self.min_volume_ratio,
                        self.max_position_size / buy_price.ask
                    )
                    
                    # Calculate confidence
                    confidence = self._calculate_opportunity_confidence(
                        profit_after_fees, volume_available, buy_price, sell_price
                    )
                    
                    opportunity = ArbitrageOpportunity(
                        symbol=symbol,
                        buy_exchange=buy_price.exchange,
                        sell_exchange=sell_price.exchange,
                        buy_price=buy_price.ask,
                        sell_price=sell_price.bid,
                        profit_pct=profit_pct,
                        profit_after_fees=profit_after_fees,
                        volume_available=volume_available,
                        confidence=confidence,
                        timestamp=datetime.utcnow()
                    )
                    
                    opportunities.append(opportunity)
        
        return opportunities
    
    def _calculate_opportunity_confidence(
        self, 
        profit_after_fees: float, 
        volume_available: float,
        buy_price: ExchangePrice,
        sell_price: ExchangePrice
    ) -> float:
        """Calculate confidence score for arbitrage opportunity"""
        
        # Base confidence from profit margin
        profit_confidence = min(profit_after_fees / (self.min_profit_pct * 2), 1.0)
        
        # Volume confidence
        volume_confidence = min(volume_available / 1000, 1.0)  # Normalize to $1000
        
        # Exchange reliability (simulated)
        exchange_scores = {
            'binance': 0.95,
            'coinbase': 0.90,
            'kraken': 0.85,
            'hyperliquid': 0.80,
            'bybit': 0.75
        }
        
        buy_reliability = exchange_scores.get(buy_price.exchange.lower(), 0.7)
        sell_reliability = exchange_scores.get(sell_price.exchange.lower(), 0.7)
        reliability_confidence = (buy_reliability + sell_reliability) / 2
        
        # Time freshness
        now = datetime.utcnow()
        buy_age = (now - buy_price.timestamp).total_seconds()
        sell_age = (now - sell_price.timestamp).total_seconds()
        max_age = max(buy_age, sell_age)
        freshness_confidence = max(0, 1 - max_age / 10)  # Decay over 10 seconds
        
        # Combined confidence
        confidence = (
            profit_confidence * 0.4 +
            volume_confidence * 0.2 +
            reliability_confidence * 0.2 +
            freshness_confidence * 0.2
        )
        
        return confidence
    
    def _filter_opportunities(self, opportunities: List[ArbitrageOpportunity]) -> List[ArbitrageOpportunity]:
        """Filter and rank arbitrage opportunities"""
        filtered = []
        
        for opp in opportunities:
            # Minimum profit check
            if opp.profit_after_fees < self.min_profit_pct:
                continue
            
            # Minimum volume check
            if opp.volume_available < 100:  # $100 minimum
                continue
            
            # Confidence check
            if opp.confidence < 0.5:
                continue
            
            # Latency buffer check
            if opp.profit_after_fees < self.latency_buffer * 2:
                continue
            
            filtered.append(opp)
        
        # Sort by profit after fees
        filtered.sort(key=lambda x: x.profit_after_fees, reverse=True)
        
        return filtered[:5]  # Top 5 opportunities
    
    async def _validate_opportunity(self, opportunity: ArbitrageOpportunity) -> bool:
        """Validate arbitrage opportunity before execution"""
        try:
            # Re-fetch current prices to ensure opportunity still exists
            buy_connector = self.connectors[opportunity.buy_exchange]
            sell_connector = self.connectors[opportunity.sell_exchange]
            
            buy_ticker = await buy_connector.get_ticker(opportunity.symbol)
            sell_ticker = await sell_connector.get_ticker(opportunity.symbol)
            
            if not buy_ticker or not sell_ticker:
                return False
            
            # Recalculate profit
            current_profit = (sell_ticker.bid - buy_ticker.ask) / buy_ticker.ask
            total_fees = buy_ticker.fees + sell_ticker.fees
            current_profit_after_fees = current_profit - total_fees
            
            # Check if opportunity still exists
            if current_profit_after_fees < self.min_profit_pct:
                logger.debug(f"⚠️ Opportunity expired: {opportunity.symbol} profit now {current_profit_after_fees*100:.3f}%")
                return False
            
            # Check price movement (slippage protection)
            buy_price_change = abs(buy_ticker.ask - opportunity.buy_price) / opportunity.buy_price
            sell_price_change = abs(sell_ticker.bid - opportunity.sell_price) / opportunity.sell_price
            
            if buy_price_change > self.max_slippage or sell_price_change > self.max_slippage:
                logger.debug(f"⚠️ Excessive slippage: {opportunity.symbol}")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Error validating opportunity: {e}")
            return False
    
    async def _create_arbitrage_signal(self, opportunity: ArbitrageOpportunity) -> StrategySignal:
        """Create trading signal from arbitrage opportunity"""
        try:
            # For arbitrage, we'll create a BUY signal with special metadata
            # The execution engine should handle the simultaneous buy/sell
            
            metadata = {
                'arbitrage_type': 'cross_exchange',
                'buy_exchange': opportunity.buy_exchange,
                'sell_exchange': opportunity.sell_exchange,
                'buy_price': opportunity.buy_price,
                'sell_price': opportunity.sell_price,
                'profit_pct': opportunity.profit_pct * 100,
                'profit_after_fees_pct': opportunity.profit_after_fees * 100,
                'volume_available': opportunity.volume_available,
                'execution_timeout': self.execution_timeout,
                'strategy_type': 'arbitrage'
            }
            
            # Calculate position size
            position_size = min(
                opportunity.volume_available,
                self.max_position_size / opportunity.buy_price
            )
            
            signal = self._create_signal(
                symbol=opportunity.symbol,
                action=SignalAction.BUY,  # Arbitrage buy signal
                price=opportunity.buy_price,
                confidence=opportunity.confidence,
                metadata=metadata,
                position_size=position_size
            )
            
            # Store opportunity
            self.opportunities.append(opportunity)
            
            logger.info(f"🔄 Arbitrage Signal: {opportunity.symbol}")
            logger.info(f"   Buy: {opportunity.buy_exchange} @ {opportunity.buy_price:.4f}")
            logger.info(f"   Sell: {opportunity.sell_exchange} @ {opportunity.sell_price:.4f}")
            logger.info(f"   Profit: {opportunity.profit_after_fees*100:.3f}%")
            logger.info(f"   Volume: ${opportunity.volume_available:.0f}")
            
            return signal
            
        except Exception as e:
            logger.error(f"❌ Error creating arbitrage signal: {e}")
            return None
    
    async def execute_arbitrage(self, opportunity: ArbitrageOpportunity) -> bool:
        """Execute arbitrage trade (simulation)"""
        try:
            logger.info(f"🚀 Executing arbitrage for {opportunity.symbol}...")
            
            # Simulate execution
            execution_success = np.random.random() > 0.1  # 90% success rate
            
            if execution_success:
                # Record successful arbitrage
                self.executed_arbitrages.append({
                    'opportunity': opportunity,
                    'executed_at': datetime.utcnow(),
                    'profit_realized': opportunity.profit_after_fees * opportunity.volume_available * opportunity.buy_price
                })
                
                self.successful_arbitrages += 1
                self.total_profit += opportunity.profit_after_fees * opportunity.volume_available * opportunity.buy_price
                
                logger.info(f"✅ Arbitrage executed successfully")
                logger.info(f"   Profit: ${opportunity.profit_after_fees * opportunity.volume_available * opportunity.buy_price:.2f}")
            else:
                logger.warning(f"❌ Arbitrage execution failed")
            
            self.total_arbitrages += 1
            return execution_success
            
        except Exception as e:
            logger.error(f"❌ Error executing arbitrage: {e}")
            return False
    
    def get_arbitrage_metrics(self) -> Dict[str, Any]:
        """Get arbitrage performance metrics"""
        success_rate = self.successful_arbitrages / max(self.total_arbitrages, 1)
        avg_profit = self.total_profit / max(self.successful_arbitrages, 1)
        
        return {
            'total_arbitrages': self.total_arbitrages,
            'successful_arbitrages': self.successful_arbitrages,
            'success_rate': success_rate,
            'total_profit': self.total_profit,
            'avg_profit_per_trade': avg_profit,
            'opportunities_found': len(self.opportunities),
            'active_exchanges': len(self.connectors),
            'last_opportunity': self.opportunities[-1].timestamp if self.opportunities else None
        }
    
    def get_strategy_info(self) -> Dict[str, Any]:
        """Get strategy information and performance metrics"""
        return {
            "name": self.name,
            "type": "arbitrage",
            "exchanges": self.exchanges,
            "symbols": self.symbols,
            "min_profit_pct": self.min_profit_pct * 100,
            "max_position_size": self.max_position_size,
            "metrics": self.get_arbitrage_metrics(),
            "recent_opportunities": [
                {
                    'symbol': opp.symbol,
                    'profit_pct': opp.profit_after_fees * 100,
                    'buy_exchange': opp.buy_exchange,
                    'sell_exchange': opp.sell_exchange,
                    'timestamp': opp.timestamp.isoformat()
                }
                for opp in self.opportunities[-5:]  # Last 5 opportunities
            ],
            "status": self.status.value,
            "enabled": self.enabled
        } 