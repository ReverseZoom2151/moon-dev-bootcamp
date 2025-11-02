"""
Hyperliquid Utilities Service
Integrates nice_funcs.py functionality into the autonomous trading system
"""

import asyncio
import logging
import pandas as pd
import requests
import aiohttp
import ccxt
import pandas_ta as ta
import eth_account
import warnings
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from hyperliquid.info import Info
from hyperliquid.exchange import Exchange
from hyperliquid.utils import constants
from core.config import get_settings

# Suppress warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class HyperliquidUtilsService:
    """
    Comprehensive Hyperliquid utilities service integrating nice_funcs.py functionality
    """
    
    def __init__(self):
        self.settings = get_settings()
        self.session = None
        self._account = None
        self._exchange = None
        self._info = None
        
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    @property
    def account(self):
        """Lazy initialization of account"""
        if not self._account and self.settings.HYPERLIQUID_PRIVATE_KEY:
            self._account = eth_account.Account.from_key(self.settings.HYPERLIQUID_PRIVATE_KEY)
        return self._account
    
    @property
    def exchange(self):
        """Lazy initialization of exchange"""
        if not self._exchange and self.account:
            self._exchange = Exchange(self.account, constants.MAINNET_API_URL)
        return self._exchange
    
    @property
    def info(self):
        """Lazy initialization of info"""
        if not self._info:
            self._info = Info(constants.MAINNET_API_URL, skip_ws=True)
        return self._info

    async def ask_bid(self, symbol: str) -> Tuple[float, float, List]:
        """Get ask, bid prices and L2 order book data"""
        try:
            url = 'https://api.hyperliquid.xyz/info'
            headers = {'Content-Type': 'application/json'}
            data = {
                'type': 'l2Book',
                'coin': symbol
            }

            if self.session:
                async with self.session.post(url, headers=headers, json=data) as response:
                    l2_data = await response.json()
            else:
                response = requests.post(url, headers=headers, json=data)
                l2_data = response.json()

            levels = l2_data['levels']
            bid = float(levels[0][0]['px'])
            ask = float(levels[1][0]['px'])

            logger.info(f"Retrieved ask/bid for {symbol}: ask={ask}, bid={bid}")
            return ask, bid, levels

        except Exception as e:
            logger.error(f"Error getting ask/bid for {symbol}: {e}")
            raise

    async def spot_price_and_symbol_info(self, symbol: str) -> Tuple[float, str, int, int]:
        """Get spot price and symbol information"""
        try:
            url = "https://api.hyperliquid.xyz/info"
            headers = {"Content-Type": "application/json"}
            body = {"type": "spotMetaAndAssetCtxs"}

            if self.session:
                async with self.session.post(url, headers=headers, json=body) as response:
                    data = await response.json()
            else:
                response = requests.post(url, headers=headers, json=body)
                data = response.json()

            tokens = data[0]['tokens']
            universe = data[0]['universe']
            asset_ctxs = data[1]

            # Find token info
            token_index = None
            sz_decimals = None
            for token in tokens:
                if token['name'] == symbol:
                    token_index = token['index']
                    sz_decimals = token['szDecimals']
                    break

            if token_index is None:
                raise ValueError(f"Token symbol {symbol} not found")

            # Find universe info
            for pair in universe:
                if token_index in pair['tokens']:
                    hoe_ass_symbol = pair['name']
                    universe_index = pair['index']
                    mid_px = float(asset_ctxs[universe_index]['midPx'])
                    
                    # Calculate price decimals
                    mid_px_str = str(mid_px)
                    if '.' in mid_px_str:
                        px_decimals = len(mid_px_str.split('.')[1]) - 1
                    else:
                        px_decimals = 0
                    
                    return mid_px, hoe_ass_symbol, sz_decimals, px_decimals

            raise ValueError(f"Token pair for symbol {symbol} not found in universe")

        except Exception as e:
            logger.error(f"Error getting spot price info for {symbol}: {e}")
            raise

    async def get_sz_px_decimals(self, symbol: str) -> Tuple[int, int]:
        """Get size and price decimals for a symbol"""
        try:
            url = 'https://api.hyperliquid.xyz/info'
            headers = {'Content-Type': 'application/json'}
            data = {'type': 'meta'}

            if self.session:
                async with self.session.post(url, headers=headers, json=data) as response:
                    meta_data = await response.json()
            else:
                response = requests.post(url, headers=headers, json=data)
                meta_data = response.json()

            symbols = meta_data['universe']
            symbol_info = next((s for s in symbols if s['name'] == symbol), None)
            
            if not symbol_info:
                raise ValueError(f'Symbol {symbol} not found')
            
            sz_decimals = symbol_info['szDecimals']
            
            # Get current ask price to determine price decimals
            ask, _, _ = await self.ask_bid(symbol)
            ask_str = str(ask)
            
            if '.' in ask_str:
                px_decimals = len(ask_str.split('.')[1])
            else:
                px_decimals = 0

            logger.info(f"{symbol} - sz_decimals: {sz_decimals}, px_decimals: {px_decimals}")
            return sz_decimals, px_decimals

        except Exception as e:
            logger.error(f"Error getting decimals for {symbol}: {e}")
            raise

    async def spot_limit_order(self, coin: str, is_buy: bool, sz: float, limit_px: float, 
                              sz_decimals: int, px_decimals: int) -> Dict:
        """Place a spot limit order"""
        try:
            if not self.exchange:
                raise ValueError("Exchange not initialized - check private key configuration")

            sz = round(sz, sz_decimals)
            limit_px = round(float(limit_px), px_decimals)
            
            logger.info(f"Placing limit order: {coin} {'BUY' if is_buy else 'SELL'} {sz} @ {limit_px}")
            
            order_result = self.exchange.order(
                coin, is_buy, sz, limit_px, 
                {"limit": {"tif": "Gtc"}}, 
                reduce_only=False
            )
            
            status = order_result['response']['data']['statuses'][0]
            logger.info(f"Order placed: {status}")
            
            return order_result

        except Exception as e:
            logger.error(f"Error placing spot limit order: {e}")
            raise

    async def limit_order(self, coin: str, is_buy: bool, sz: float, limit_px: float, 
                         reduce_only: bool = False) -> Dict:
        """Place a futures limit order"""
        try:
            if not self.exchange:
                raise ValueError("Exchange not initialized - check private key configuration")

            sz_decimals, px_decimals = await self.get_sz_px_decimals(coin)
            sz = round(sz, sz_decimals)
            limit_px = round(float(limit_px), px_decimals)
            
            logger.info(f"Placing limit order: {coin} {'BUY' if is_buy else 'SELL'} {sz} @ {limit_px}")
            
            order_result = self.exchange.order(
                coin, is_buy, sz, limit_px,
                {"limit": {"tif": "Gtc"}},
                reduce_only=reduce_only
            )
            
            status = order_result['response']['data']['statuses'][0]
            logger.info(f"Order placed: {status}")
            
            return order_result

        except Exception as e:
            logger.error(f"Error placing limit order: {e}")
            raise

    async def adjust_leverage(self, symbol: str, leverage: int) -> Dict:
        """Adjust leverage for a symbol"""
        try:
            if not self.exchange:
                raise ValueError("Exchange not initialized - check private key configuration")

            logger.info(f"Setting leverage for {symbol} to {leverage}x")
            result = self.exchange.update_leverage(leverage, symbol)
            logger.info(f"Leverage updated: {result}")
            
            return result

        except Exception as e:
            logger.error(f"Error adjusting leverage for {symbol}: {e}")
            raise

    async def adjust_leverage_size_signal(self, symbol: str, leverage: int) -> Tuple[int, float]:
        """Calculate position size based on 95% of account balance"""
        try:
            if not self.exchange or not self.info or not self.account:
                raise ValueError("Trading components not initialized")

            # Update leverage first
            await self.adjust_leverage(symbol, leverage)
            
            # Get account value
            user_state = self.info.user_state(self.account.address)
            acct_value = float(user_state["marginSummary"]["accountValue"])
            acct_val95 = acct_value * 0.95
            
            # Get current price
            ask, _, _ = await self.ask_bid(symbol)
            
            # Calculate size
            size = (acct_val95 / ask) * leverage
            sz_decimals, _ = await self.get_sz_px_decimals(symbol)
            size = round(size, sz_decimals)
            
            logger.info(f"Calculated size for {symbol}: {size} (95% of ${acct_value})")
            return leverage, size

        except Exception as e:
            logger.error(f"Error calculating leverage size for {symbol}: {e}")
            raise

    async def adjust_leverage_usd_size(self, symbol: str, usd_size: float, leverage: int) -> Tuple[int, float]:
        """Calculate position size based on specific USD amount"""
        try:
            if not self.exchange:
                raise ValueError("Exchange not initialized")

            # Update leverage first
            await self.adjust_leverage(symbol, leverage)
            
            # Get current price
            ask, _, _ = await self.ask_bid(symbol)
            
            # Calculate size
            size = (usd_size / ask) * leverage
            sz_decimals, _ = await self.get_sz_px_decimals(symbol)
            size = round(size, sz_decimals)
            
            logger.info(f"Calculated size for {symbol}: {size} (${usd_size} USD)")
            return leverage, size

        except Exception as e:
            logger.error(f"Error calculating USD size for {symbol}: {e}")
            raise

    async def get_ohlcv_data(self, symbol: str, interval: str, lookback_days: int) -> Optional[List]:
        """Fetch OHLCV candlestick data"""
        try:
            end_time = datetime.now()
            start_time = end_time - timedelta(days=lookback_days)
            
            url = 'https://api.hyperliquid.xyz/info'
            headers = {'Content-Type': 'application/json'}
            data = {
                "type": "candleSnapshot",
                "req": {
                    "coin": symbol,
                    "interval": interval,
                    "startTime": int(start_time.timestamp() * 1000),
                    "endTime": int(end_time.timestamp() * 1000)
                }
            }

            if self.session:
                async with self.session.post(url, headers=headers, json=data) as response:
                    if response.status == 200:
                        snapshot_data = await response.json()
                        return snapshot_data
            else:
                response = requests.post(url, headers=headers, json=data)
                if response.status_code == 200:
                    return response.json()

            logger.error(f"Error fetching OHLCV data for {symbol}")
            return None

        except Exception as e:
            logger.error(f"Error fetching OHLCV data for {symbol}: {e}")
            return None

    def process_data_to_df(self, snapshot_data: List, time_period: int = 20) -> pd.DataFrame:
        """Process snapshot data to DataFrame with support/resistance"""
        try:
            if not snapshot_data:
                return pd.DataFrame()

            columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
            data = []
            
            for snapshot in snapshot_data:
                timestamp = datetime.fromtimestamp(snapshot['t'] / 1000).strftime('%Y-%m-%d %H:%M:%S')
                data.append([
                    timestamp,
                    float(snapshot['o']),
                    float(snapshot['h']),
                    float(snapshot['l']),
                    float(snapshot['c']),
                    float(snapshot['v'])
                ])

            df = pd.DataFrame(data, columns=columns)
            
            # Calculate rolling support and resistance
            df['support'] = df['close'].rolling(window=time_period, min_periods=1).min().shift(1)
            df['resis'] = df['close'].rolling(window=time_period, min_periods=1).max().shift(1)

            return df

        except Exception as e:
            logger.error(f"Error processing data to DataFrame: {e}")
            return pd.DataFrame()

    async def calculate_vwap(self, symbol: str) -> Tuple[pd.DataFrame, float]:
        """Calculate VWAP for a symbol"""
        try:
            snapshot_data = await self.get_ohlcv_data(symbol, '15m', 300)
            df = self.process_data_to_df(snapshot_data)

            if df.empty:
                raise ValueError("No data available for VWAP calculation")

            # Convert timestamp and set as index
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.set_index('timestamp', inplace=True)
            df.sort_index(inplace=True)

            # Calculate VWAP
            df['VWAP'] = ta.vwap(high=df['high'], low=df['low'], close=df['close'], volume=df['volume'])
            latest_vwap = df['VWAP'].iloc[-1]

            logger.info(f"VWAP calculated for {symbol}: {latest_vwap}")
            return df, latest_vwap

        except Exception as e:
            logger.error(f"Error calculating VWAP for {symbol}: {e}")
            raise

    async def get_position(self, symbol: str) -> Tuple[List, bool, float, str, float, float, Optional[bool]]:
        """Get current position information"""
        try:
            if not self.info or not self.account:
                raise ValueError("Info service not initialized")

            user_state = self.info.user_state(self.account.address)
            logger.info(f"Account value: {user_state['marginSummary']['accountValue']}")
            
            positions = []
            
            for position in user_state["assetPositions"]:
                if (position["position"]["coin"] == symbol and 
                    float(position["position"]["szi"]) != 0):
                    
                    positions.append(position["position"])
                    in_pos = True
                    size = float(position["position"]["szi"])
                    pos_sym = position["position"]["coin"]
                    entry_px = float(position["position"]["entryPx"])
                    pnl_perc = float(position["position"]["returnOnEquity"]) * 100
                    
                    long = size > 0 if size != 0 else None
                    
                    logger.info(f"Position found: {symbol} size={size} pnl={pnl_perc}%")
                    return positions, in_pos, size, pos_sym, entry_px, pnl_perc, long
            
            # No position found
            return [], False, 0, None, 0, 0, None

        except Exception as e:
            logger.error(f"Error getting position for {symbol}: {e}")
            raise

    async def get_spot_position(self, symbol: str) -> Tuple[List, bool, float, str, float, float, Optional[bool]]:
        """Get current spot position information"""
        try:
            if not self.info or not self.account:
                raise ValueError("Info service not initialized")

            user_state = self.info.spot_user_state(self.account.address)
            positions = []
            
            for balance in user_state["balances"]:
                if balance["coin"] == symbol:
                    size = float(balance["total"])
                    pos_sym = balance["coin"]
                    entry_px = float(balance["entryNtl"]) / size if size != 0 else 0
                    pnl_perc = 0  # Not available in spot data
                    positions.append(balance)
                    in_pos = size > 1
                    long = size > 1
                    
                    logger.info(f"Spot position found: {symbol} size={size}")
                    return positions, in_pos, size, pos_sym, entry_px, pnl_perc, long
            
            # No position found
            return [], False, 0, None, 0, 0, None

        except Exception as e:
            logger.error(f"Error getting spot position for {symbol}: {e}")
            raise

    async def cancel_all_orders(self) -> None:
        """Cancel all open orders"""
        try:
            if not self.exchange or not self.info or not self.account:
                raise ValueError("Trading components not initialized")

            open_orders = self.info.open_orders(self.account.address)
            logger.info(f"Cancelling {len(open_orders)} open orders")
            
            for order in open_orders:
                self.exchange.cancel(order['coin'], order['oid'])
                logger.info(f"Cancelled order: {order['coin']} {order['oid']}")

        except Exception as e:
            logger.error(f"Error cancelling all orders: {e}")
            raise

    async def cancel_symbol_orders(self, symbol: str) -> None:
        """Cancel all open orders for a specific symbol"""
        try:
            if not self.exchange or not self.info or not self.account:
                raise ValueError("Trading components not initialized")

            open_orders = self.info.open_orders(self.account.address)
            cancelled_count = 0
            
            for order in open_orders:
                if order['coin'] == symbol:
                    self.exchange.cancel(order['coin'], order['oid'])
                    cancelled_count += 1
                    logger.info(f"Cancelled order: {symbol} {order['oid']}")
            
            logger.info(f"Cancelled {cancelled_count} orders for {symbol}")

        except Exception as e:
            logger.error(f"Error cancelling orders for {symbol}: {e}")
            raise

    async def kill_switch(self, symbol: str) -> None:
        """Emergency position close"""
        try:
            positions, in_pos, pos_size, pos_sym, entry_px, pnl_perc, long = await self.get_position(symbol)
            
            while in_pos:
                await self.cancel_all_orders()
                
                ask, bid, _ = await self.ask_bid(pos_sym)
                pos_size = abs(pos_size)
                
                if long:
                    await self.limit_order(pos_sym, False, pos_size, ask, reduce_only=True)
                    logger.info("Kill switch - SELL TO CLOSE submitted")
                else:
                    await self.limit_order(pos_sym, True, pos_size, bid, reduce_only=True)
                    logger.info("Kill switch - BUY TO CLOSE submitted")
                
                await asyncio.sleep(5)
                positions, in_pos, pos_size, pos_sym, entry_px, pnl_perc, long = await self.get_position(symbol)
            
            logger.info("Position successfully closed via kill switch")

        except Exception as e:
            logger.error(f"Error in kill switch for {symbol}: {e}")
            raise

    async def pnl_close(self, symbol: str, target: float, max_loss: float) -> None:
        """Close position based on PnL thresholds"""
        try:
            positions, in_pos, pos_size, pos_sym, entry_px, pnl_perc, long = await self.get_position(symbol)
            
            if not in_pos:
                logger.info(f"No position to close for {symbol}")
                return
            
            if pnl_perc > target:
                logger.info(f"PnL {pnl_perc}% > target {target}% - closing position (WIN)")
                await self.kill_switch(pos_sym)
            elif pnl_perc <= max_loss:
                logger.info(f"PnL {pnl_perc}% <= max loss {max_loss}% - closing position (LOSS)")
                await self.kill_switch(pos_sym)
            else:
                logger.info(f"PnL {pnl_perc}% within range [{max_loss}%, {target}%] - holding position")

        except Exception as e:
            logger.error(f"Error in PnL close for {symbol}: {e}")
            raise

    async def calculate_atr(self, symbol: str, window: int = 14, lookback_days: int = 30) -> Tuple[pd.DataFrame, float]:
        """Calculate Average True Range"""
        try:
            snapshot_data = await self.get_ohlcv_data(symbol, '1h', lookback_days)
            df = self.process_data_to_df(snapshot_data)
            
            if df.empty:
                raise ValueError("No data available for ATR calculation")

            # Calculate true ranges
            df['High-Low'] = df['high'] - df['low']
            df['High-PrevClose'] = abs(df['high'] - df['close'].shift())
            df['Low-PrevClose'] = abs(df['low'] - df['close'].shift())
            
            # True range is max of the three
            df['TrueRange'] = df[['High-Low', 'High-PrevClose', 'Low-PrevClose']].max(axis=1)
            
            # Calculate ATR
            df['ATR'] = df['TrueRange'].rolling(window=window, min_periods=1).mean()
            last_atr = df['ATR'].iloc[-1]
            
            logger.info(f"ATR calculated for {symbol}: {last_atr}")
            return df, last_atr

        except Exception as e:
            logger.error(f"Error calculating ATR for {symbol}: {e}")
            raise

    async def calculate_bollinger_bands(self, symbol: str, length: int = 20, std_dev: int = 2, 
                                      lookback_days: int = 30) -> Tuple[pd.DataFrame, bool, bool]:
        """Calculate Bollinger Bands and classify tight/wide"""
        try:
            snapshot_data = await self.get_ohlcv_data(symbol, '1h', lookback_days)
            df = self.process_data_to_df(snapshot_data)
            
            if df.empty:
                raise ValueError("No data available for Bollinger Bands calculation")

            # Calculate Bollinger Bands
            bollinger_bands = ta.bbands(df['close'], length=length, std=std_dev)
            bollinger_bands = bollinger_bands.iloc[:, [0, 1, 2]]
            bollinger_bands.columns = ['BBL', 'BBM', 'BBU']
            
            df = pd.concat([df, bollinger_bands], axis=1)
            
            # Calculate Band Width
            df['BandWidth'] = df['BBU'] - df['BBL']
            
            # Determine thresholds
            tight_threshold = df['BandWidth'].quantile(0.2)
            wide_threshold = df['BandWidth'].quantile(0.8)
            
            current_band_width = df['BandWidth'].iloc[-1]
            tight = current_band_width <= tight_threshold
            wide = current_band_width >= wide_threshold
            
            logger.info(f"Bollinger Bands for {symbol}: tight={tight}, wide={wide}")
            return df, tight, wide

        except Exception as e:
            logger.error(f"Error calculating Bollinger Bands for {symbol}: {e}")
            raise

    async def get_all_spot_symbols(self) -> List[str]:
        """Get all available spot symbols"""
        try:
            url = "https://api.hyperliquid.xyz/info"
            headers = {"Content-Type": "application/json"}
            body = {"type": "spotMetaAndAssetCtxs"}

            if self.session:
                async with self.session.post(url, headers=headers, json=body) as response:
                    data = await response.json()
            else:
                response = requests.post(url, headers=headers, json=body)
                data = response.json()

            tokens = data[0]['tokens']
            symbols = [token['name'] for token in tokens]
            
            logger.info(f"Retrieved {len(symbols)} spot symbols")
            return symbols

        except Exception as e:
            logger.error(f"Error getting spot symbols: {e}")
            raise

    async def ob_data(self, symbol: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, float, float, float, float]:
        """Get order book data from multiple exchanges"""
        try:
            # Initialize exchanges
            binance = ccxt.binance({'enableRateLimit': True})
            bybit = ccxt.bybit({'enableRateLimit': True})
            coinbasepro = ccxt.coinbasepro({'enableRateLimit': True})

            # Symbol formats for each exchange
            binance_sym = f"{symbol}/USDT"
            bybit_sym = f"{symbol}USDT"
            coinbase_sym = f"{symbol}-USD"

            # Fetch order books
            ob_binance = binance.fetch_order_book(binance_sym)
            ob_bybit = bybit.fetch_order_book(bybit_sym)
            ob_coinbase = coinbasepro.fetch_order_book(coinbase_sym)

            def create_combined_df(ob, exchange):
                bids_df = pd.DataFrame(ob['bids'], columns=['Bid', 'Bid Size'])
                asks_df = pd.DataFrame(ob['asks'], columns=['Ask', 'Ask Size'])
                if exchange == 'coinbasepro' and len(bids_df) > 100:
                    bids_df = bids_df.head(100)
                    asks_df = asks_df.head(100)
                return pd.concat([asks_df.reset_index(drop=True), bids_df.reset_index(drop=True)], axis=1)

            def find_max_sizes(df):
                max_bid_size_row = df.loc[df['Bid Size'].idxmax()]
                max_ask_size_row = df.loc[df['Ask Size'].idxmax()]
                return (max_bid_size_row['Bid'], max_bid_size_row['Bid Size'], 
                       max_ask_size_row['Ask'], max_ask_size_row['Ask Size'])

            # Create DataFrames
            combined_df_binance = create_combined_df(ob_binance, 'binance')
            combined_df_bybit = create_combined_df(ob_bybit, 'bybit')
            combined_df_coinbase = create_combined_df(ob_coinbase, 'coinbasepro')

            # Find max sizes across all exchanges
            max_bid_size = max_ask_size = 0
            max_bid_price = max_ask_price = None

            for df in [combined_df_binance, combined_df_bybit, combined_df_coinbase]:
                bid_price, bid_size, ask_price, ask_size = find_max_sizes(df)
                if bid_size > max_bid_size:
                    max_bid_size, max_bid_price = bid_size, bid_price
                if ask_size > max_ask_size:
                    max_ask_size, max_ask_price = ask_size, ask_price

            # Find prices before biggest
            def find_before_biggest(df, max_price, col_name, is_bid=True):
                if is_bid:
                    sorted_df = df.sort_values(by=col_name, ascending=True)
                    before_biggest_df = sorted_df[sorted_df[col_name] > max_price]
                else:
                    sorted_df = df.sort_values(by=col_name, ascending=False)
                    before_biggest_df = sorted_df[sorted_df[col_name] < max_price]

                if before_biggest_df.empty:
                    return None, None
                else:
                    before_biggest_row = before_biggest_df.iloc[0]
                    return before_biggest_row[col_name], before_biggest_row[col_name + ' Size']

            bid_before_biggest = ask_before_biggest = None
            
            if max_bid_price:
                bid_before, _ = find_before_biggest(combined_df_binance, max_bid_price, 'Bid', is_bid=True)
                bid_before_biggest = bid_before

            if max_ask_price:
                ask_before, _ = find_before_biggest(combined_df_binance, max_ask_price, 'Ask', is_bid=False)
                ask_before_biggest = ask_before

            logger.info(f"Order book analysis for {symbol} completed")
            return (combined_df_binance, combined_df_bybit, combined_df_coinbase, 
                   max_bid_price, max_ask_price, 
                   float(bid_before_biggest) if bid_before_biggest else 0.0,
                   float(ask_before_biggest) if ask_before_biggest else 0.0)

        except Exception as e:
            logger.error(f"Error analyzing order book for {symbol}: {e}")
            raise

    async def open_order_deluxe(self, symbol: str, entry_price: float, stop_loss: float, 
                               take_profit: float, size: float) -> Dict:
        """Place limit order with stop loss and take profit"""
        try:
            if not self.exchange:
                raise ValueError("Exchange not initialized")

            # Cancel existing orders for symbol
            await self.cancel_symbol_orders(symbol)
            
            # Get decimals for rounding
            sz_decimals, px_decimals = await self.get_sz_px_decimals(symbol)
            
            # Round prices appropriately
            if symbol == 'BTC':
                take_profit = int(take_profit)
                stop_loss = int(stop_loss)
            else:
                take_profit = round(take_profit, px_decimals)
                stop_loss = round(stop_loss, px_decimals)

            logger.info(f"Opening deluxe order for {symbol}: entry={entry_price}, sl={stop_loss}, tp={take_profit}")

            # Place main limit order
            order_result = self.exchange.order(
                symbol, True, size, entry_price,
                {"limit": {"tif": "Gtc"}}
            )
            logger.info(f"Limit order placed: {order_result}")

            # Place stop loss order
            stop_order_type = {"trigger": {"triggerPx": stop_loss, "isMarket": True, "tpsl": "sl"}}
            stop_result = self.exchange.order(
                symbol, False, size, stop_loss,
                stop_order_type, reduce_only=True
            )
            logger.info(f"Stop loss order placed: {stop_result}")

            # Place take profit order
            tp_order_type = {"trigger": {"triggerPx": take_profit, "isMarket": True, "tpsl": "tp"}}
            tp_result = self.exchange.order(
                symbol, False, size, take_profit,
                tp_order_type, reduce_only=True
            )
            logger.info(f"Take profit order placed: {tp_result}")

            return {
                "main_order": order_result,
                "stop_loss": stop_result,
                "take_profit": tp_result
            }

        except Exception as e:
            logger.error(f"Error placing deluxe order for {symbol}: {e}")
            raise

    def create_symbol_info(self, symbol: str, price: float, stop_loss_pct: float, 
                          take_profit_pct: float, ideal_size: float) -> Dict:
        """Create symbol info dictionary with calculated prices"""
        try:
            price = float(price)
            
            # Calculate stop loss and take profit prices
            stop_loss = price - (price * abs(stop_loss_pct) / 100)
            take_profit = price + (price * take_profit_pct / 100)

            return {
                "Symbol": symbol,
                "Size": ideal_size,
                "Entry Price": price,
                "Stop Loss": stop_loss,
                "Take Profit": take_profit
            }

        except Exception as e:
            logger.error(f"Error creating symbol info for {symbol}: {e}")
            raise

# Global service instance
hyperliquid_utils_service = HyperliquidUtilsService() 