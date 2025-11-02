# autonomous_trading_system/backend/services/hyperliquid_service.py

import eth_account
import json
import pandas_ta as ta
import requests
import logging
import ccxt
import pandas as pd
import time
from hyperliquid.info import Info
from hyperliquid.exchange import Exchange
from hyperliquid.utils import constants
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class HyperliquidService:
    def __init__(self, settings: Dict[str, Any]):
        """
        Initializes the HyperliquidService.
        Expects 'HYPERLIQUID_SECRET_KEY' in settings.
        """
        private_key = settings.get("HYPERLIQUID_SECRET_KEY")
        if not private_key:
            # Fallback to another key if available, for broader compatibility
            private_key = settings.get("HYPERLIQUID_PRIVATE_KEY")
        
        if not private_key:
            raise ValueError("Hyperliquid private/secret key not found in settings")
            
        self.account = eth_account.Account.from_key(private_key)
        self.exchange = Exchange(self.account, constants.MAINNET_API_URL)
        self.info = Info(constants.MAINNET_API_URL, skip_ws=True)
        self.settings = settings
        logger.info(f"HyperliquidService initialized for account: {self.account.address}")

    def get_order_book(self, symbol: str) -> Tuple[Optional[float], Optional[float], Optional[List]]:
        """Fetches the L2 order book for a given symbol."""
        url = 'https://api.hyperliquid.xyz/info'
        headers = {'Content-Type': 'application/json'}
        data = {'type': 'l2Book', 'coin': symbol}
        try:
            response = requests.post(url, headers=headers, data=json.dumps(data), timeout=10)
            response.raise_for_status()
            l2_data = response.json()['levels']
            if not l2_data or len(l2_data) < 2 or not l2_data[0] or not l2_data[1]:
                logger.warning(f"Incomplete L2 data received for {symbol}")
                return None, None, None
            bid = float(l2_data[0][0]['px'])
            ask = float(l2_data[1][0]['px'])
            return ask, bid, l2_data
        except (requests.exceptions.RequestException, KeyError, IndexError, json.JSONDecodeError) as e:
            logger.error(f"Error fetching L2 book for {symbol}: {e}")
            return None, None, None

    def get_ohlcv(self, symbol: str, interval: str, lookback_days: int) -> pd.DataFrame:
        """Fetches OHLCV data from Hyperliquid."""
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
        try:
            response = requests.post(url, headers=headers, json=data, timeout=15)
            response.raise_for_status()
            snapshot_data = response.json()
            if not snapshot_data:
                return pd.DataFrame()

            df = pd.DataFrame(snapshot_data)
            df['timestamp'] = pd.to_datetime(df['t'], unit='ms')
            df = df.rename(columns={'o': 'open', 'h': 'high', 'l': 'low', 'c': 'close', 'v': 'volume'})
            df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            df.dropna(inplace=True)
            return df
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching candle data for {symbol}: {e}")
            return pd.DataFrame()

    def get_sz_px_decimals(self, symbol: str) -> Tuple[int, int]:
        """Gets the size and price decimals for a given symbol."""
        sz_decimals, px_decimals = 0, 0
        try:
            meta_data = self.info.meta()
            universe = meta_data.get('universe', [])
            symbol_info_item = next((s for s in universe if s.get('name') == symbol), None)
            if symbol_info_item:
                sz_decimals = symbol_info_item.get('szDecimals', 0)
        except Exception as e:
            logger.warning(f"Could not fetch metadata for sz_decimals for {symbol}: {e}")

        try:
            ask, _, _ = self.get_order_book(symbol)
            if ask:
                ask_str = str(ask)
                if '.' in ask_str:
                    px_decimals = len(ask_str.split('.')[1])
        except Exception as e:
            logger.warning(f"Could not fetch ask price for px_decimals for {symbol}: {e}")
        
        logger.debug(f"For symbol '{symbol}': sz_decimals={sz_decimals}, px_decimals={px_decimals}.")
        return sz_decimals, px_decimals

    def limit_order(self, coin: str, is_buy: bool, sz: float, limit_px: float, reduce_only: bool = False):
        """Places a limit order."""
        sz_decimals, px_decimals = self.get_sz_px_decimals(coin)
        sz = round(sz, sz_decimals)
        limit_px = round(float(limit_px), px_decimals)
        
        logger.info(f"Placing order: {coin}, Side: {'BUY' if is_buy else 'SELL'}, Size: {sz}, Price: ${limit_px}, Reduce Only: {reduce_only}")
        try:
            order_result = self.exchange.order(coin, is_buy, sz, limit_px, {"limit": {"tif": "Gtc"}}, reduce_only=reduce_only)
            if isinstance(order_result, dict) and 'response' in order_result:
                status = order_result['response']['data']['statuses'][0]
                logger.info(f"Order placed with status: {status}")
                if 'resting' in status:
                    return status['resting']['oid']
                elif 'filled' in status:
                    return status['filled']['oid']
            else:
                logger.warning(f"Order for {coin} placed with unexpected response: {order_result}")
            return order_result
        except Exception as e:
            logger.error(f"Failed to place limit order for {coin}: {e}", exc_info=True)
            return None

    def adjust_leverage(self, symbol: str, leverage: int, is_cross: bool = True):
        """Adjusts leverage for a given symbol."""
        logger.info(f"Updating leverage for {symbol} to {leverage}x {'(Cross)' if is_cross else ''}.")
        try:
            leverage_type = "cross" if is_cross else "isolated"
            self.exchange.update_leverage(leverage, symbol, is_cross)
            logger.info(f"Successfully updated leverage for {symbol}.")
        except Exception as e:
            logger.error(f"Failed to update leverage for {symbol}: {e}")

    def adjust_leverage_and_get_size_usd(self, symbol: str, usd_size: float, leverage: int) -> Optional[float]:
        """Adjusts leverage and calculates order size based on a USD amount."""
        logger.info(f"Adjusting leverage for {symbol} to {leverage}x for a ${usd_size} position.")
        try:
            self.adjust_leverage(symbol, leverage)
            price, _, _ = self.get_order_book(symbol)
            if not price:
                logger.error(f"Could not fetch price for {symbol} to calculate size.")
                return None
            
            size = (usd_size / price) * leverage
            sz_decimals, _ = self.get_sz_px_decimals(symbol)
            size = round(size, sz_decimals)
            logger.info(f"Calculated size for ${usd_size} at {leverage}x leverage: {size} {symbol}")
            return size
        except Exception as e:
            logger.error(f"Failed to adjust leverage or calculate size for {symbol}: {e}")
            return None

    def adjust_leverage_and_get_size_perc(self, symbol: str, leverage: int, portfolio_perc: float = 0.95) -> Optional[float]:
        """Adjusts leverage and calculates order size based on a percentage of account value."""
        logger.info(f"Calculating position size for {symbol} using {portfolio_perc*100}% of account value at {leverage}x leverage.")
        try:
            self.adjust_leverage(symbol, leverage)
            acct_value = self.get_account_value()
            if acct_value is None:
                logger.error("Could not retrieve account value to calculate size.")
                return None
            
            price, _, _ = self.get_order_book(symbol)
            if not price:
                logger.error(f"Could not fetch price for {symbol} to calculate size.")
                return None

            usd_size = acct_value * portfolio_perc
            size = (usd_size / price) * leverage
            sz_decimals, _ = self.get_sz_px_decimals(symbol)
            size = round(size, sz_decimals)
            logger.info(f"Calculated size for {portfolio_perc*100}% of portfolio: {size} {symbol}")
            return size
        except Exception as e:
            logger.error(f"Failed to calculate percentage-based size for {symbol}: {e}")
            return None

    def get_position(self, symbol: str) -> Dict[str, Any]:
        """Gets perp position details for a symbol."""
        try:
            user_state = self.info.user_state(self.account.address)
            for position in user_state.get("assetPositions", []):
                pos_info = position.get("position", {})
                if pos_info.get("coin") == symbol and float(pos_info.get("szi", 0)) != 0:
                    size = float(pos_info["szi"])
                    return {
                        "in_pos": True,
                        "size": size,
                        "symbol": pos_info["coin"],
                        "entry_px": float(pos_info["entryPx"]),
                        "pnl_perc": float(pos_info.get("returnOnEquity", 0)) * 100,
                        "long": size > 0,
                    }
        except Exception as e:
            logger.error(f"Error getting position for {symbol}: {e}", exc_info=True)
        return {"in_pos": False, "size": 0, "long": None, "entry_px": 0, "pnl_perc": 0}

    def cancel_symbol_orders(self, symbol: str):
        """Cancels all open orders for a specific symbol."""
        logger.info(f"Cancelling open orders for symbol: {symbol}")
        try:
            open_orders = self.info.open_orders(self.account.address)
            for open_order in open_orders:
                if open_order['coin'] == symbol:
                    logger.debug(f"Cancelling order: {open_order}")
                    self.exchange.cancel(open_order['coin'], open_order['oid'])
            logger.info(f"Finished cancelling orders for {symbol}.")
        except Exception as e:
            logger.error(f"Error cancelling orders for {symbol}: {e}", exc_info=True)

    def cancel_all_orders(self):
        """Cancels all open orders for the account."""
        logger.warning("Cancelling ALL open orders for the account.")
        try:
            open_orders = self.info.open_orders(self.account.address)
            if not open_orders:
                logger.info("No open orders to cancel.")
                return
            for open_order in open_orders:
                logger.debug(f"Cancelling order: {open_order}")
                self.exchange.cancel(open_order['coin'], open_order['oid'])
            logger.info("Successfully cancelled all open orders.")
        except Exception as e:
            logger.error(f"Error cancelling all orders: {e}", exc_info=True)

    def kill_switch(self, symbol: str, market: bool = False):
        """Closes a position for a symbol, either with a limit or market-like order."""
        position = self.get_position(symbol)
        if not position.get("in_pos"):
            logger.info(f"No open position for {symbol} to kill.")
            return

        logger.warning(f"KILL SWITCH ACTIVATED for {symbol}. Closing position.")
        self.cancel_symbol_orders(symbol)
        time.sleep(0.5) # Give time for cancellations to process

        pos_size = abs(position["size"])
        side_to_close = not position["long"]

        try:
            price_to_close = None
            if market:
                _, _, l2_data = self.get_order_book(symbol)
                if l2_data:
                    # Use 5th bid/ask for more aggressive closing
                    if side_to_close and len(l2_data[1]) > 4: # We are buying to close a short
                        price_to_close = float(l2_data[1][4]['px']) # 5th ask
                    elif not side_to_close and len(l2_data[0]) > 4: # We are selling to close a long
                        price_to_close = float(l2_data[0][4]['px']) # 5th bid
                if price_to_close:
                    logger.info(f"Market closing position for {symbol} at aggressive price {price_to_close}.")
                    self.limit_order(symbol, side_to_close, pos_size, price_to_close, reduce_only=True)
                else: # Fallback to market order if L2 data is thin
                    logger.warning(f"Thin L2 data for {symbol}, using true market order for kill switch.")
                    self.exchange.order(symbol, side_to_close, pos_size, 0, {"market": True}, reduce_only=True)
            else:
                ask, bid, _ = self.get_order_book(symbol)
                price_to_close = bid if side_to_close else ask
                if price_to_close:
                    logger.info(f"Limit closing position for {symbol} at price {price_to_close}.")
                    self.limit_order(symbol, side_to_close, pos_size, price_to_close, reduce_only=True)
                else:
                    logger.error(f"Could not get order book price for {symbol}, cannot limit close.")
        except Exception as e:
             logger.error(f"Failed during kill switch for {symbol}: {e}. Attempting true market order as fallback.", exc_info=True)
             self.exchange.order(symbol, side_to_close, pos_size, 0, {"market": True}, reduce_only=True)

    def close_all_positions(self, market: bool = False):
        """Closes all open perp positions."""
        logger.warning(f"Closing all open positions ({'market' if market else 'limit'}).")
        self.cancel_all_orders()
        time.sleep(0.5)
        
        all_positions = self.get_all_open_positions()
        if not all_positions:
            logger.info("No open positions found to close.")
            return
            
        for pos in all_positions:
            symbol = pos.get("symbol")
            if symbol:
                logger.info(f"Closing position in {symbol}...")
                self.kill_switch(symbol, market=market)
                time.sleep(1) # Stagger the close orders
        logger.info("Finished closing all positions.")

    def pnl_close(self, symbol: str, target_pnl_perc: float, max_loss_perc: float):
        """Closes a position if PnL targets are met."""
        position = self.get_position(symbol)
        if not position["in_pos"]:
            return

        pnl_perc = position["pnl_perc"]
        if pnl_perc >= target_pnl_perc:
            logger.info(f"Target PnL reached for {symbol} ({pnl_perc:.2f}% >= {target_pnl_perc:.2f}%). Closing position.")
            self.kill_switch(symbol, market=True)
        elif pnl_perc <= -abs(max_loss_perc):
            logger.info(f"Max loss reached for {symbol} ({pnl_perc:.2f}% <= -{abs(max_loss_perc):.2f}%). Closing position.")
            self.kill_switch(symbol, market=True)

    def open_order_deluxe(self, symbol: str, size: float, entry_price: float, sl_price: float, tp_price: float, is_buy: bool = True):
        """Places a limit order with accompanying SL and TP trigger orders."""
        logger.info(f"Placing deluxe order for {symbol}: Entry=${entry_price}, TP=${tp_price}, SL=${sl_price}")
        self.cancel_symbol_orders(symbol)
        time.sleep(0.5)
        
        # Place the main entry order
        entry_oid = self.limit_order(symbol, is_buy, size, entry_price)
        if not entry_oid:
            logger.error(f"Failed to place entry order for {symbol}. Aborting SL/TP placement.")
            return

        # Place the Stop Loss order
        try:
            sl_order_type = {"trigger": {"triggerPx": sl_price, "isMarket": True, "tpsl": "sl"}}
            self.exchange.order(symbol, not is_buy, size, sl_price, sl_order_type, reduce_only=True)
            logger.info(f"Stop loss order placed for {symbol} at {sl_price}")
        except Exception as e:
            logger.error(f"Failed to place SL order for {symbol}: {e}. Entry order OID: {entry_oid}", exc_info=True)

        # Place the Take Profit order
        try:
            tp_order_type = {"trigger": {"triggerPx": tp_price, "isMarket": True, "tpsl": "tp"}}
            self.exchange.order(symbol, not is_buy, size, tp_price, tp_order_type, reduce_only=True)
            logger.info(f"Take profit order placed for {symbol} at {tp_price}")
        except Exception as e:
            logger.error(f"Failed to place TP order for {symbol}: {e}. Entry order OID: {entry_oid}", exc_info=True)

    def get_account_value(self) -> Optional[float]:
        """Gets the total account value."""
        try:
            user_state = self.info.user_state(self.account.address)
            if user_state and 'marginSummary' in user_state:
                return float(user_state['marginSummary']['accountValue'])
            logger.warning("Could not find 'marginSummary' in user_state response.")
            return None
        except Exception as e:
            logger.error(f"Error getting account value: {e}", exc_info=True)
            return None

    def calculate_vwap(self, symbol: str, interval='15m', lookback_days=30) -> Optional[float]:
        """Calculates the latest VWAP for a symbol."""
        df = self.get_ohlcv(symbol, interval, lookback_days)
        if df.empty:
            return None
        try:
            df['VWAP'] = ta.vwap(high=df['high'], low=df['low'], close=df['close'], volume=df['volume'])
            return df['VWAP'].iloc[-1]
        except Exception as e:
            logger.error(f"Error calculating VWAP for {symbol}: {e}")
            return None

    def calculate_atr(self, symbol: str, interval='1h', lookback_days=14, window=14) -> Optional[float]:
        """Calculates the latest ATR for a symbol."""
        df = self.get_ohlcv(symbol, interval, lookback_days + window) # Fetch more data for rolling window
        if df.empty:
            return None
        try:
            df['ATR'] = ta.atr(high=df['high'], low=df['low'], close=df['close'], length=window)
            return df['ATR'].iloc[-1]
        except Exception as e:
            logger.error(f"Error calculating ATR for {symbol}: {e}")
            return None

    def get_bollinger_bands(self, symbol: str, interval='1h', lookback_days=30, length=20, std_dev=2) -> Optional[Dict]:
        """Gets the latest Bollinger Bands values for a symbol."""
        df = self.get_ohlcv(symbol, interval, lookback_days)
        if df.empty:
            return None
        try:
            bbands = ta.bbands(df['close'], length=length, std=std_dev)
            if bbands is None or bbands.empty:
                return None
            latest = bbands.iloc[-1]
            return {
                "lower": latest[f'BBL_{length}_{std_dev}'],
                "middle": latest[f'BBM_{length}_{std_dev}'],
                "upper": latest[f'BBU_{length}_{std_dev}'],
                "bandwidth": latest[f'BBB_{length}_{std_dev}'],
                "percent": latest[f'BBP_{length}_{std_dev}'],
            }
        except Exception as e:
            logger.error(f"Error calculating Bollinger Bands for {symbol}: {e}")
            return None

    def get_aggregated_order_book(self, symbol: str) -> Dict[str, Any]:
        """Aggregates order book data from Binance, Bybit, and Coinbase Pro."""
        logger.info(f"Fetching aggregated order book for {symbol}...")
        results = {"binance": None, "bybit": None, "coinbase": None, "max_bid": None, "max_ask": None}
        clients = {
            "binance": ccxt.binance(),
            "bybit": ccxt.bybit(),
            "coinbase": ccxt.coinbasepro(),
        }
        symbols = {
            "binance": f"{symbol}/USDT",
            "bybit": f"{symbol}USDT",
            "coinbase": f"{symbol}-USD",
        }

        max_bid_size = 0
        max_ask_size = 0
        max_bid_price = None
        max_ask_price = None

        for ex, client in clients.items():
            try:
                ob = client.fetch_order_book(symbols[ex])
                bids_df = pd.DataFrame(ob['bids'], columns=['price', 'size'])
                asks_df = pd.DataFrame(ob['asks'], columns=['price', 'size'])
                results[ex] = {"bids": bids_df.to_dict('records'), "asks": asks_df.to_dict('records')}

                if not bids_df.empty:
                    max_ex_bid = bids_df.iloc[bids_df['size'].idxmax()]
                    if max_ex_bid['size'] > max_bid_size:
                        max_bid_size = max_ex_bid['size']
                        max_bid_price = max_ex_bid['price']

                if not asks_df.empty:
                    max_ex_ask = asks_df.iloc[asks_df['size'].idxmax()]
                    if max_ex_ask['size'] > max_ask_size:
                        max_ask_size = max_ex_ask['size']
                        max_ask_price = max_ex_ask['price']
            except Exception as e:
                logger.warning(f"Could not fetch order book from {ex} for {symbol}: {e}")
        
        results["max_bid"] = {"price": max_bid_price, "size": max_bid_size}
        results["max_ask"] = {"price": max_ask_price, "size": max_ask_size}
        return results

    def get_liquidations(self, lookback_hours: int = 1) -> Optional[Dict]:
        # This function relies on an external CSV. In a real service, this should point to a database or a live feed.
        # For now, it's adapted to show intent.
        liq_file_path = self.settings.get("LIQUIDATION_DATA_FILE", "data/recent_liqs.csv")
        try:
            df = pd.read_csv(liq_file_path)
            # Assuming the CSV has columns: 'Interval', 'Total Liquidations', 'Long Liquidations', 'Short Liquidations'
            return df.set_index('Interval').to_dict('index')
        except FileNotFoundError:
            logger.warning(f"Liquidation data file not found at {liq_file_path}")
            return None
        except Exception as e:
            logger.error(f"Error reading liquidation data: {e}")
            return None

    def get_all_open_positions(self) -> List[Dict[str, Any]]:
        """Gets all open perp positions for the account."""
        open_positions = []
        try:
            user_state = self.info.user_state(self.account.address)
            for position in user_state.get("assetPositions", []):
                pos_info = position.get("position", {})
                if float(pos_info.get("szi", 0)) != 0:
                    open_positions.append(self.get_position(pos_info["coin"]))
        except Exception as e:
            logger.error(f"Error getting all open positions: {e}", exc_info=True)
        return open_positions

    def get_spot_meta_and_ctxs(self) -> Optional[Tuple[List, List, List]]:
        """Fetches spot metadata and asset contexts."""
        url = "https://api.hyperliquid.xyz/info"
        headers = {"Content-Type": "application/json"}
        body = {"type": "spotMetaAndAssetCtxs"}
        try:
            response = requests.post(url, headers=headers, json=body, timeout=10)
            response.raise_for_status()
            data = response.json()
            tokens = data[0]['tokens']
            universe = data[0]['universe']
            asset_ctxs = data[1]
            return tokens, universe, asset_ctxs
        except Exception as e:
            logger.error(f"Error fetching spot meta and asset contexts: {e}")
            return None
            
    def get_spot_meta(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Gets metadata for a specific spot symbol, including price."""
        meta = self.get_spot_meta_and_ctxs()
        if not meta:
            return None
        
        tokens, universe, asset_ctxs = meta
        
        token_info = next((t for t in tokens if t['name'] == symbol), None)
        if not token_info:
            logger.warning(f"Spot token {symbol} not found.")
            return None
            
        token_index = token_info['index']
        
        pair_info = next((p for p in universe if token_index in p['tokens']), None)
        if not pair_info:
             logger.warning(f"Pair info for {symbol} not found.")
             return None

        mid_px = asset_ctxs[pair_info['index']].get('midPx')
        px_decimals = 0
        if mid_px:
            mid_px_str = str(mid_px)
            if '.' in mid_px_str:
                px_decimals = len(mid_px_str.split('.')[1])

        return {
            "symbol": symbol,
            "pair_name": pair_info['name'],
            "sz_decimals": token_info['szDecimals'],
            "mid_price": float(mid_px) if mid_px else None,
            "px_decimals": px_decimals,
        }

    def spot_limit_order(self, coin_pair: str, is_buy: bool, sz: float, limit_px: float):
        """Places a spot limit order."""
        # Note: 'coin_pair' should be the hoe_ass_symbol, e.g., 'PURR/USDC'
        logger.info(f"Placing SPOT order: {coin_pair}, Side: {'BUY' if is_buy else 'SELL'}, Size: {sz}, Price: ${limit_px}")
        try:
            order_result = self.exchange.order(coin_pair, is_buy, sz, limit_px, {"limit": {"tif": "Gtc"}}, is_spot=True)
            if isinstance(order_result, dict) and 'response' in order_result and 'statuses' in order_result['response']['data']:
                logger.info(f"Spot order placed with status: {order_result['response']['data']['statuses'][0]}")
            else:
                 logger.warning(f"Spot order for {coin_pair} placed with unexpected response: {order_result}")
            return order_result
        except Exception as e:
            logger.error(f"Failed to place spot limit order for {coin_pair}: {e}", exc_info=True)
            return None

    def get_spot_position(self, symbol: str) -> Dict[str, Any]:
        """Gets position details for a spot symbol."""
        try:
            user_state = self.info.spot_user_state(self.account.address)
            for balance in user_state.get("balances", []):
                if balance["coin"] == symbol:
                    size = float(balance["total"])
                    if size > 1e-9: # Consider positions with more than dust
                        return {
                            "in_pos": True, "size": size, "symbol": symbol,
                            "entry_px": float(balance["entryNtl"]) / size if size != 0 else 0,
                        }
        except Exception as e:
            logger.error(f"Error getting spot position for {symbol}: {e}", exc_info=True)
        return {"in_pos": False, "size": 0, "symbol": symbol}

    def spot_kill_switch(self, symbol: str, market_order: bool = False):
        """Closes a spot position for a symbol."""
        position = self.get_spot_position(symbol)
        if not position["in_pos"]:
            logger.info(f"No open spot position for {symbol} to kill.")
            return

        meta = self.get_spot_meta(symbol)
        if not meta:
            logger.error(f"Cannot get metadata for {symbol}, aborting spot kill switch.")
            return

        pair_name = meta['pair_name']
        logger.warning(f"SPOT KILL SWITCH ACTIVATED for {symbol} ({pair_name}). Closing position.")
        self.cancel_symbol_orders(pair_name)
        time.sleep(0.5)

        size = position["size"]
        sz_decimals = meta['sz_decimals']
        px_decimals = meta['px_decimals']
        
        size_to_close = round(size, sz_decimals)
        
        try:
            price_to_close = meta['mid_price']
            if market_order and price_to_close:
                # To market sell, we undercut the mid price
                price_to_close *= 0.98
            
            if price_to_close:
                self.spot_limit_order(pair_name, False, size_to_close, round(price_to_close, px_decimals))
            else:
                 logger.error(f"Could not get mid price for {pair_name}, cannot close position.")
        except Exception as e:
             logger.error(f"Failed during spot kill switch for {symbol}: {e}", exc_info=True)

    def get_spot_usdc_balance(self, address: Optional[str] = None) -> float:
        """Gets the USDC balance for a given address, or the service's own account if None."""
        target_address = address if address else self.account.address
        try:
            user_state = self.info.spot_user_state(target_address)
            for balance in user_state.get("balances", []):
                if balance["coin"] == "USDC":
                    return float(balance["total"])
            return 0.0
        except Exception as e:
            # It's common for addresses to have no spot state, so debug level is fine
            logger.debug(f"Could not retrieve USDC balance for {target_address}: {e}")
            return 0.0

    def get_supply_demand_zones(self, symbol: str, interval='1h', lookback_days=30, period=20) -> Optional[Dict[str, float]]:
        """Calculates supply and demand zones based on rolling min/max of high/low."""
        df = self.get_ohlcv(symbol, interval, lookback_days)
        if df.empty:
            return None
        try:
            df['demand_zone_low'] = df['low'].rolling(window=period).min().shift(1)
            df['demand_zone_high'] = df['close'].rolling(window=period).min().shift(1)
            df['supply_zone_low'] = df['close'].rolling(window=period).max().shift(1)
            df['supply_zone_high'] = df['high'].rolling(window=period).max().shift(1)
            
            latest = df.iloc[-1]
            return {
                "demand_low": latest['demand_zone_low'],
                "demand_high": latest['demand_zone_high'],
                "supply_low": latest['supply_zone_low'],
                "supply_high": latest['supply_zone_high'],
            }
        except Exception as e:
            logger.error(f"Error calculating S/D zones for {symbol}: {e}")
            return None

    def get_linear_regression_channel(self, symbol: str, interval='1h', lookback_days=30, length=20) -> Optional[Dict]:
        """Calculates the latest Linear Regression Channel values."""
        df = self.get_ohlcv(symbol, interval, lookback_days)
        if df.empty or len(df) < length:
            return None
        try:
            linreg = ta.linreg(df['close'], length=length)
            if linreg is None: return None
            
            channel_width = df['close'].rolling(window=length).std() * 2
            latest_middle = linreg.iloc[-1]
            latest_width = channel_width.iloc[-1]
            
            return {
                "upper": latest_middle + latest_width,
                "middle": latest_middle,
                "lower": latest_middle - latest_width,
                "slope": linreg.iloc[-1] - linreg.iloc[-2]
            }
        except Exception as e:
            logger.error(f"Error calculating Linear Regression Channel for {symbol}: {e}")
            return None

    def has_volume_spike(self, symbol: str, interval='15m', lookback_days=7, period=20, vol_multiplier=3) -> bool:
        """Checks for a recent volume spike accompanied by a price downtrend."""
        df = self.get_ohlcv(symbol, interval, lookback_days)
        if df.empty or len(df) < period:
            return False
        try:
            df['MA_Volume'] = df['volume'].rolling(window=period).mean()
            df['MA_Close'] = df['close'].rolling(window=period).mean()
            latest = df.iloc[-1]
            return (latest['volume'] > vol_multiplier * latest['MA_Volume']) and (latest['close'] < latest['MA_Close'])
        except Exception:
            return False

    def get_open_interest(self) -> Optional[float]:
        """Gets total open interest from a predefined local CSV file."""
        # This function relies on an external CSV. In a real service, this should point to a database or a live feed.
        oi_file_path = self.settings.get("OPEN_INTEREST_FILE", "data/oi_total.csv")
        try:
            df = pd.read_csv(oi_file_path)
            # Assuming CSV has columns: 'timestamp', 'openInterest'
            return float(df.iloc[-1, 1])
        except (FileNotFoundError, IndexError, ValueError) as e:
            logger.error(f"Error reading open interest file: {e}")
            return None

    def get_funding_rates(self, tokens: List[str]) -> Dict[str, Dict[str, Optional[float]]]:
        """
        Fetches live funding rates from Hyperliquid and Binance for a given list of tokens.
        Returns annualized funding rates.
        """
        funding_rates = {token: {"binance": None, "hyperliquid": None} for token in tokens}
        logger.info(f"Fetching funding rates for: {', '.join(tokens)}")

        # Hyperliquid funding
        try:
            meta = self.info.meta()
            for asset in meta.get("universe", []):
                name = asset.get("name")
                if name in tokens:
                    impact_bps = asset.get("funding", 0)
                    # Hyperliquid funding is per hour, so multiply by 24*365 for annual rate
                    funding_rates[name]["hyperliquid"] = float(impact_bps) * 24 * 365
        except Exception as e:
            logger.error(f"Failed to fetch Hyperliquid funding rates: {e}")

        # Binance funding
        try:
            binance = ccxt.binanceusdm()
            all_funding = binance.fetch_funding_rates()
            for fund_info in all_funding:
                symbol = fund_info.get('symbol', '').replace('/USDT', '')
                if symbol in tokens:
                    # Binance funding is typically per 8 hours, so multiply by 3*365 for annual rate
                    funding_rates[symbol]["binance"] = fund_info.get('fundingRate', 0.0) * 3 * 365 * 100
        except Exception as e:
            logger.error(f"Failed to fetch Binance funding rates: {e}")

        logger.info("Finished fetching funding rates.")
        return funding_rates

    def check_market_conditions(self, symbols: List[str] = None, sma_fast: int = 20, sma_slow: int = 40) -> bool:
        """
        Checks if the price is above both fast and slow SMAs on the daily timeframe for a list of major symbols.
        The market is considered bullish if at least 2 out of 3 major symbols are above both SMAs.
        This logic is adapted from the Day 50 `nice_funcs2.py` script.
        """
        if symbols is None:
            symbols = ['BTC', 'SOL', 'ETH']
        
        bullish_symbols = 0
        lookback_days = sma_slow + 20  # Ensure enough data for the longest SMA
        
        logger.info(f"Checking market conditions for {symbols} using {sma_fast}/{sma_slow} daily SMAs.")

        for symbol in symbols:
            df = self.get_ohlcv(symbol, '1d', lookback_days)
            if df.empty or len(df) < sma_slow:
                logger.warning(f"Not enough daily data for {symbol} to check market conditions.")
                continue

            # Calculate SMAs
            df[f'sma_fast'] = df['close'].rolling(window=sma_fast).mean()
            df[f'sma_slow'] = df['close'].rolling(window=sma_slow).mean()

            # Get the latest values
            latest = df.iloc[-1]
            current_price = latest['close']
            fast_sma_val = latest['sma_fast']
            slow_sma_val = latest['sma_slow']

            if pd.isna(fast_sma_val) or pd.isna(slow_sma_val):
                logger.warning(f"Could not calculate SMAs for {symbol}.")
                continue

            # Check if price is above both SMAs
            if current_price > fast_sma_val and current_price > slow_sma_val:
                logger.info(f"✅ Bullish signal for {symbol}: Price (${current_price:.2f}) > SMA{sma_fast} (${fast_sma_val:.2f}) and SMA{sma_slow} (${slow_sma_val:.2f})")
                bullish_symbols += 1
            else:
                logger.info(f"❌ Bearish/Neutral signal for {symbol}: Price (${current_price:.2f}) vs SMA{sma_fast} (${fast_sma_val:.2f}) and SMA{sma_slow} (${slow_sma_val:.2f})")

        # The market is considered favorable if 2 or more of the 3 major symbols are bullish.
        is_favorable = bullish_symbols >= 2
        logger.info(f"Market Conditions Summary: {bullish_symbols}/{len(symbols)} symbols are bullish. Favorable: {is_favorable}")
        return is_favorable