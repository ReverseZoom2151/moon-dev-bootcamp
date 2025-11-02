import asyncio
import logging
import requests
from datetime import datetime, timedelta
from eth_account import Account
from core.config import get_settings
from hyperliquid.utils import constants
from services.trading_utils import fetch_candle_snapshot, process_data_to_df, adjust_leverage_size_signal, get_sz_px_decimals
from hyperliquid.info import Info
from hyperliquid.exchange import Exchange
from typing import Dict, Any

settings = get_settings()
logger = logging.getLogger(__name__)

class HyperliquidBreakoutService:
    """Service to detect daily-resistance breakouts and place orders on Hyperliquid"""
    def __init__(self):
        if not settings.HYPERLIQUID_PRIVATE_KEY:
            raise ValueError("HYPERLIQUID_PRIVATE_KEY must be set to enable breakout bot")
        # Account and parameters
        self.account = Account.from_key(settings.HYPERLIQUID_PRIVATE_KEY)
        self.order_usd_size = getattr(settings, 'HYPERLIQUID_BREAKOUT_ORDER_USD_SIZE', 10)
        self.leverage = getattr(settings, 'HYPERLIQUID_BREAKOUT_LEVERAGE', 1)
        self.lookback_hours = getattr(settings, 'HYPERLIQUID_BREAKOUT_LOOKBACK_HOURS', 1)
        self.lookback_days = getattr(settings, 'HYPERLIQUID_BREAKOUT_LOOKBACK_DAYS', 20)
        self.tp_percent = getattr(settings, 'HYPERLIQUID_BREAKOUT_TP_PERCENT', 3)
        self.sl_percent = getattr(settings, 'HYPERLIQUID_BREAKOUT_SL_PERCENT', 18)
        self.loop_interval = getattr(settings, 'HYPERLIQUID_BREAKOUT_LOOP_INTERVAL', 60)

    def get_symbols(self):
        """Fetch available symbols from Hyperliquid"""
        url = f"{constants.MAINNET_API_URL}/info"
        resp = requests.post(url, json={'type': 'meta'}, headers={'Content-Type': 'application/json'})
        resp.raise_for_status()
        data = resp.json()
        symbols = [s['name'] for s in data.get('universe', [])]
        logger.info(f"Fetched {len(symbols)} symbols from Hyperliquid")
        return symbols

    def calculate_daily_resistance(self, symbols):
        """Calculate max daily high for each symbol over lookback_days"""
        resistance = {}
        for symbol in symbols:
            try:
                end = datetime.utcnow()
                start = end - timedelta(days=self.lookback_days)
                snapshot = fetch_candle_snapshot(symbol, '1d', start, end)
                if snapshot:
                    highs = [float(day.get('h') or day.get('High')) for day in snapshot]
                    resistance[symbol] = max(highs)
                else:
                    logger.warning(f"No daily candles for {symbol}")
            except Exception as e:
                logger.error(f"Error fetching daily data for {symbol}: {e}")
        return resistance

    def check_breakout(self, symbol, df, resistance_levels):
        """Check if latest close breaks above resistance"""
        level = resistance_levels.get(symbol)
        if level is None:
            return None
        close = float(df['close'].iloc[-1])
        if close > level:
            sl = close * (1 - self.sl_percent / 100)
            tp = close * (1 + self.tp_percent / 100)
            logger.info(f"Breakout {symbol}: close={close}, lvl={level}, SL={sl}, TP={tp}")
            return {'entry_price': close, 'stop_loss': sl, 'take_profit': tp}
        return None

    def _open_order_deluxe(self, symbol_info: Dict[str, Any], size: float):
        """Place entry, stop-loss, and take-profit orders based on symbol_info."""
        symbol = symbol_info["Symbol"]
        entry_price = symbol_info["Entry Price"]
        stop_loss = symbol_info["Stop Loss"]
        take_profit = symbol_info["Take Profit"]
        exchange = Exchange(self.account, constants.MAINNET_API_URL)
        # Determine decimal rounding for price
        _, px_rounding = get_sz_px_decimals(symbol)
        if symbol == "BTC":
            take_profit = int(take_profit)
            stop_loss = int(stop_loss)
        else:
            take_profit = round(take_profit, px_rounding)
            stop_loss = round(stop_loss, px_rounding)
        logger.info(f"Opening order for {symbol}: size={size}, entry={entry_price}, stop_loss={stop_loss}, take_profit={take_profit}")
        # Cancel existing symbol orders
        info = Info(constants.MAINNET_API_URL, skip_ws=True)
        orders = info.open_orders(self.account.address)
        for o in orders:
            if o.get("coin") == symbol:
                try:
                    exchange.cancel(symbol, o["oid"])
                except Exception as e:
                    logger.error(f"Error cancelling order {o.get('oid')} for {symbol}: {e}")
        # Entry limit order
        entry_result = exchange.order(
            symbol,
            True,
            size,
            entry_price,
            {"limit": {"tif": "Gtc"}}
        )
        logger.info(f"Entry order result for {symbol}: {entry_result}")
        # Stop loss order
        stop_type = {"trigger": {"triggerPx": stop_loss, "isMarket": True, "tpsl": "stop_loss"}}
        stop_result = exchange.order(
            symbol,
            False,
            size,
            stop_loss,
            stop_type,
            reduce_only=True
        )
        logger.info(f"Stop loss order result for {symbol}: {stop_result}")
        # Take profit order
        tp_type = {"trigger": {"triggerPx": take_profit, "isMarket": True, "tpsl": "take_profit"}}
        tp_result = exchange.order(
            symbol,
            False,
            size,
            take_profit,
            tp_type,
            reduce_only=True
        )
        logger.info(f"Take profit order result for {symbol}: {tp_result}")

    async def start(self):
        """Main loop to monitor breakouts and place orders"""
        symbols = self.get_symbols()
        resistance = self.calculate_daily_resistance(symbols)
        logger.info("Starting Hyperliquid breakout bot loop")
        while True:
            for symbol in symbols:
                try:
                    end = datetime.utcnow()
                    start = end - timedelta(hours=self.lookback_hours)
                    snapshot = await asyncio.to_thread(fetch_candle_snapshot, symbol, '1h', start, end)
                    df = await asyncio.to_thread(process_data_to_df, snapshot)
                    if df.empty:
                        continue
                    info = self.check_breakout(symbol, df, resistance)
                    if info:
                        # Calculate size based on USD order size and leverage
                        _, size = await asyncio.to_thread(adjust_leverage_size_signal, symbol, self.leverage, self.account)
                        price = info['entry_price']
                        # Place entry, stop-loss, and take-profit orders
                        symbol_info = {
                            "Symbol": symbol,
                            "Entry Price": price,
                            "Stop Loss": info['stop_loss'],
                            "Take Profit": info['take_profit']
                        }
                        await asyncio.to_thread(self._open_order_deluxe, symbol_info, size)
                except Exception as e:
                    logger.error(f"Error in breakout loop for {symbol}: {e}")
            await asyncio.sleep(self.loop_interval) 