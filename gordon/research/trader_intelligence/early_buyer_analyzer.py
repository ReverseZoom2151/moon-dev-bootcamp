"""
Early Buyer Analyzer
====================
Analyzes historical trades to identify early buyers and large volume participants.
Supports Binance and Bitfinex exchanges.
"""

import pandas as pd
import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta, timezone
from pathlib import Path

logger = logging.getLogger(__name__)


class EarlyBuyerAnalyzer:
    """
    Analyzes historical trades to find early buyers and significant traders.
    
    Identifies accounts that bought early or traded large volumes,
    useful for discovering "smart money" to follow.
    """
    
    def __init__(
        self,
        exchange_adapter=None,
        config: Optional[Dict] = None
    ):
        """
        Initialize early buyer analyzer.
        
        Args:
            exchange_adapter: Exchange adapter instance (Binance/Bitfinex)
            config: Configuration dictionary
        """
        self.exchange_adapter = exchange_adapter
        self.config = config or {}
        
        # Analysis parameters
        self.min_trade_size_usd = self.config.get('min_trade_size_usd', 1000.0)
        self.max_trade_size_usd = self.config.get('max_trade_size_usd', 1000000.0)
        self.fetch_limit = self.config.get('fetch_limit', 1000)
        self.max_requests = self.config.get('max_requests', 100)
        
        # Output directory
        self.output_dir = Path(self.config.get('output_dir', './trader_intelligence_output'))
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def analyze_early_buyers(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        sort_by: str = 'timestamp'  # 'timestamp' or 'trade_size'
    ) -> pd.DataFrame:
        """
        Analyze early buyers for a given symbol.
        
        Args:
            symbol: Trading symbol (e.g., 'BTCUSDT', 'tBTCUSD')
            start_date: Start date for analysis
            end_date: End date for analysis
            sort_by: How to sort results ('timestamp' or 'trade_size')
            
        Returns:
            DataFrame with early buyer analysis
        """
        if not self.exchange_adapter:
            logger.error("Exchange adapter not available")
            return pd.DataFrame()
        
        logger.info(f"Analyzing early buyers for {symbol}")
        logger.info(f"Date range: {start_date} to {end_date}")
        logger.info(f"Trade size filter: ${self.min_trade_size_usd:,.2f} - ${self.max_trade_size_usd:,.2f}")
        
        # Fetch historical trades
        trades = self._fetch_historical_trades(symbol, start_date, end_date)
        
        if not trades:
            logger.warning(f"No trades found for {symbol}")
            return pd.DataFrame()
        
        # Process and filter trades
        processed_trades = []
        for trade in trades:
            processed = self._process_trade(trade, symbol, start_date, end_date)
            if processed:
                processed_trades.append(processed)
        
        if not processed_trades:
            logger.warning("No trades matched the filters")
            return pd.DataFrame()
        
        # Create DataFrame
        df = pd.DataFrame(processed_trades)
        
        # Sort by specified field
        if sort_by == 'timestamp':
            df = df.sort_values('timestamp').reset_index(drop=True)
        elif sort_by == 'trade_size':
            df = df.sort_values('usd_value', ascending=False).reset_index(drop=True)
        
        # Add early buyer ranking (by timestamp)
        df_sorted_by_time = df.sort_values('timestamp').reset_index(drop=True)
        df_sorted_by_time['early_buyer_rank'] = range(1, len(df_sorted_by_time) + 1)
        
        # Merge rank back
        df = df.merge(
            df_sorted_by_time[['trade_id', 'early_buyer_rank']],
            on='trade_id',
            how='left'
        )
        
        logger.info(f"Found {len(df)} qualifying trades")
        logger.info(f"Early buyers identified: {df['early_buyer_rank'].notna().sum()}")
        
        return df
    
    def _fetch_historical_trades(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime
    ) -> List[Dict]:
        """
        Fetch historical trades from exchange.
        
        Args:
            symbol: Trading symbol
            start_date: Start date
            end_date: End date
            
        Returns:
            List of trade dictionaries
        """
        trades = []
        requests_made = 0
        
        # Determine exchange type from adapter
        exchange_type = None
        if hasattr(self.exchange_adapter, '__class__'):
            class_name = self.exchange_adapter.__class__.__name__.lower()
            if 'binance' in class_name:
                exchange_type = 'binance'
            elif 'bitfinex' in class_name:
                exchange_type = 'bitfinex'
        
        if exchange_type == 'binance':
            trades = self._fetch_binance_trades(symbol, start_date, end_date)
        elif exchange_type == 'bitfinex':
            trades = self._fetch_bitfinex_trades(symbol, start_date, end_date)
        else:
            logger.warning(f"Unknown exchange type, using generic method")
            # Try generic approach
            trades = self._fetch_generic_trades(symbol, start_date, end_date)
        
        return trades
    
    def _fetch_binance_trades(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime
    ) -> List[Dict]:
        """Fetch trades from Binance."""
        trades = []
        from_id = 0
        consecutive_errors = 0
        max_errors = 3
        
        start_time_ms = int(start_date.timestamp() * 1000)
        end_time_ms = int(end_date.timestamp() * 1000)
        
        logger.info(f"Fetching Binance trades for {symbol}")
        
        # Use CCXT client directly if available
        if hasattr(self.exchange_adapter, 'ccxt_client'):
            import asyncio
            
            async def fetch():
                all_trades = []
                since = from_id
                while len(all_trades) < self.fetch_limit * 10:
                    try:
                        batch = await self.exchange_adapter.ccxt_client.fetch_trades(
                            symbol=symbol,
                            since=since,
                            limit=self.fetch_limit
                        )
                        if not batch:
                            break
                        
                        # Filter by date range
                        filtered = [
                            t for t in batch
                            if start_time_ms <= t.get('timestamp', 0) <= end_time_ms
                        ]
                        all_trades.extend(filtered)
                        
                        # Update since for next batch
                        if batch:
                            since = batch[-1].get('timestamp', since)
                            if batch[-1].get('timestamp', 0) > end_time_ms:
                                break
                    except Exception as e:
                        logger.error(f"Error fetching Binance trades: {e}")
                        break
                
                return all_trades
            
            try:
                # Run async fetch
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    # If loop is running, we need to use sync wrapper
                    import nest_asyncio
                    nest_asyncio.apply()
                trades = asyncio.run(fetch())
            except RuntimeError:
                # No event loop, create one
                trades = asyncio.run(fetch())
        else:
            logger.warning("Exchange adapter doesn't have ccxt_client")
        
        logger.info(f"Fetched {len(trades)} Binance trades")
        return trades
    
    def _fetch_bitfinex_trades(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime
    ) -> List[Dict]:
        """Fetch trades from Bitfinex."""
        trades = []
        consecutive_errors = 0
        max_errors = 3
        
        start_time_ms = int(start_date.timestamp() * 1000)
        end_time_ms = int(end_date.timestamp() * 1000)
        
        logger.info(f"Fetching Bitfinex trades for {symbol}")
        
        # Bitfinex uses different symbol format
        bitfinex_symbol = symbol
        if not symbol.startswith('t'):
            bitfinex_symbol = f"t{symbol}"
        
        # Use CCXT client directly if available
        if hasattr(self.exchange_adapter, 'ccxt_client'):
            import asyncio
            
            async def fetch():
                all_trades = []
                current_end = end_time_ms
                batch_size_ms = 24 * 60 * 60 * 1000  # 1 day batches
                
                while current_end > start_time_ms and len(all_trades) < self.fetch_limit * 10:
                    current_start = max(start_time_ms, current_end - batch_size_ms)
                    
                    try:
                        batch = await self.exchange_adapter.ccxt_client.fetch_trades(
                            symbol=bitfinex_symbol,
                            since=current_start,
                            limit=self.fetch_limit
                        )
                        
                        if batch:
                            # Filter by date range
                            filtered = [
                                t for t in batch
                                if start_time_ms <= t.get('timestamp', 0) <= end_time_ms
                            ]
                            all_trades.extend(filtered)
                        
                        current_end = current_start
                        
                    except Exception as e:
                        logger.error(f"Error fetching Bitfinex trades: {e}")
                        consecutive_errors += 1
                        if consecutive_errors >= max_errors:
                            break
                        await asyncio.sleep(3)
                
                return all_trades
            
            try:
                trades = asyncio.run(fetch())
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                trades = loop.run_until_complete(fetch())
                loop.close()
        else:
            logger.warning("Exchange adapter doesn't have ccxt_client")
        
        logger.info(f"Fetched {len(trades)} Bitfinex trades")
        return trades
    
    def _fetch_generic_trades(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime
    ) -> List[Dict]:
        """Generic trade fetching fallback."""
        logger.warning("Using generic trade fetching (may have limited functionality)")
        return []
    
    def _process_trade(
        self,
        trade: Dict,
        symbol: str,
        start_date: datetime,
        end_date: datetime
    ) -> Optional[Dict]:
        """
        Process a single trade and apply filters.
        
        Args:
            trade: Raw trade data
            symbol: Trading symbol
            start_date: Start date filter
            end_date: End date filter
            
        Returns:
            Processed trade dict or None if filtered out
        """
        try:
            # Extract timestamp
            timestamp_ms = trade.get('timestamp') or trade.get('time') or trade.get('mts')
            if not timestamp_ms:
                return None
            
            if isinstance(timestamp_ms, str):
                timestamp_ms = int(timestamp_ms)
            
            timestamp = datetime.fromtimestamp(timestamp_ms / 1000, tz=timezone.utc)
            
            # Check date range
            if not (start_date <= timestamp <= end_date):
                return None
            
            # Extract trade details
            price = float(trade.get('price', 0))
            amount = float(trade.get('amount', 0) or trade.get('qty', 0))
            
            if price <= 0 or amount == 0:
                return None
            
            # Calculate USD value
            quote_qty = trade.get('quoteQty') or trade.get('quote_qty')
            if quote_qty:
                usd_value = float(quote_qty)
            else:
                # Calculate from price * amount
                if symbol.endswith('USDT'):
                    usd_value = abs(price * amount)
                elif symbol.endswith('USD'):
                    usd_value = abs(price * amount)
                else:
                    # Estimate (could fetch conversion rate)
                    usd_value = abs(price * amount)
            
            # Apply trade size filter
            if not (self.min_trade_size_usd <= usd_value <= self.max_trade_size_usd):
                return None
            
            # Extract trader information
            trader = trade.get('buyer') or trade.get('owner') or trade.get('trader')
            trade_id = trade.get('id') or trade.get('trade_id')
            
            # Determine trade direction
            is_buyer_maker = trade.get('isBuyerMaker', False)
            trade_direction = 'BUY' if not is_buyer_maker else 'SELL'
            
            # Bitfinex format handling
            if isinstance(trade, list) and len(trade) >= 4:
                trade_id = trade[0]
                timestamp_ms = trade[1]
                amount = float(trade[2])
                price = float(trade[3])
                timestamp = datetime.fromtimestamp(timestamp_ms / 1000, tz=timezone.utc)
                trade_direction = 'BUY' if amount > 0 else 'SELL'
                usd_value = abs(amount * price)
                trader = 'Unknown'  # Bitfinex doesn't expose trader IDs in public API
            
            return {
                'timestamp': timestamp,
                'trade_id': trade_id,
                'symbol': symbol,
                'trader': trader or 'Unknown',
                'trade_direction': trade_direction,
                'price': price,
                'amount': abs(amount),
                'usd_value': usd_value,
                'is_early': False  # Will be determined by ranking
            }
            
        except Exception as e:
            logger.error(f"Error processing trade: {e}")
            return None
    
    def get_top_traders(
        self,
        df: pd.DataFrame,
        top_n: int = 100,
        metric: str = 'total_volume'  # 'total_volume', 'trade_count', 'avg_trade_size'
    ) -> pd.DataFrame:
        """
        Get top traders by specified metric.
        
        Args:
            df: DataFrame with trade analysis
            top_n: Number of top traders to return
            metric: Metric to rank by
            
        Returns:
            DataFrame with top traders
        """
        if df.empty:
            return pd.DataFrame()
        
        # Group by trader
        trader_stats = df.groupby('trader').agg({
            'usd_value': ['sum', 'count', 'mean'],
            'timestamp': 'min'
        }).reset_index()
        
        trader_stats.columns = [
            'trader',
            'total_volume',
            'trade_count',
            'avg_trade_size',
            'first_trade_time'
        ]
        
        # Sort by metric
        if metric == 'total_volume':
            trader_stats = trader_stats.sort_values('total_volume', ascending=False)
        elif metric == 'trade_count':
            trader_stats = trader_stats.sort_values('trade_count', ascending=False)
        elif metric == 'avg_trade_size':
            trader_stats = trader_stats.sort_values('avg_trade_size', ascending=False)
        elif metric == 'early_buyer':
            trader_stats = trader_stats.sort_values('first_trade_time')
        
        # Add ranking
        trader_stats['rank'] = range(1, len(trader_stats) + 1)
        
        return trader_stats.head(top_n)
    
    def save_results(self, df: pd.DataFrame, filename: str):
        """Save analysis results to CSV."""
        try:
            filepath = self.output_dir / filename
            df.to_csv(filepath, index=False)
            logger.info(f"Saved results to {filepath}")
        except Exception as e:
            logger.error(f"Error saving results: {e}")

