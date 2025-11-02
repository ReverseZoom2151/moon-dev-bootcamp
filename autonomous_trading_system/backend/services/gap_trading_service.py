import asyncio
import ccxt
import pandas as pd
import pandas_ta as ta
from core.config import get_settings

class EnhancedEmaTradingService:
    """Service to run the Day 17 Enhanced EMA (Gap) trading bot natively in the ATS"""
    def __init__(self):
        settings = get_settings()
        self.loop_interval = settings.ENHANCED_EMA_LOOP_INTERVAL
        self.demo = getattr(settings, 'SMA_DEMO_MODE', False)
        self.symbol = settings.ENHANCED_EMA_SYMBOL
        self.fast_period = settings.ENHANCED_EMA_FAST_PERIOD
        self.slow_period = settings.ENHANCED_EMA_SLOW_PERIOD
        self.trend_period = settings.ENHANCED_EMA_TREND_PERIOD
        self.stochrsi_rsi_len = settings.ENHANCED_EMA_STOCHRSI_RSI_LEN
        self.atr_period = settings.ENHANCED_EMA_ATR_PERIOD
        self.volume_ma_period = settings.ENHANCED_EMA_VOLUME_MA_PERIOD
        self.risk_per_trade = settings.ENHANCED_EMA_RISK_PER_TRADE
        self.atr_sl_multiplier = settings.ENHANCED_EMA_ATR_SL_MULTIPLIER
        self.partial_profit_factor = settings.ENHANCED_EMA_PARTIAL_PROFIT_FACTOR

        # Advanced exit management state
        self.entry_bar_idx = None
        self.entry_price = None
        self.entry_side = None   # 'long' or 'short'
        self.partial_taken = False
        self.trailing_sl = None
        self.stop_order_id = None
        self.take_profit_order_id = None

        if not self.demo:
            if not (settings.PHEMEX_API_KEY and settings.PHEMEX_SECRET_KEY):
                raise ValueError("PHEMEX_API_KEY and PHEMEX_SECRET_KEY required for Enhanced EMA trading")
            self.exchange = ccxt.phemex({
                'enableRateLimit': True,
                'apiKey': settings.PHEMEX_API_KEY,
                'secret': settings.PHEMEX_SECRET_KEY
            })
        else:
            self.exchange = None

    async def start(self):
        print(f"🚀 Starting Enhanced EMA Trading Service (demo={self.demo})")
        while True:
            await asyncio.to_thread(self._run_cycle)
            await asyncio.sleep(self.loop_interval)

    def _run_cycle(self):
        try:
            if self.demo:
                print("[DEMO] Enhanced EMA cycle skipped in demo mode")
                return

            # Fetch OHLCV data
            limit = max(self.trend_period, self.volume_ma_period) + 10
            bars = self.exchange.fetch_ohlcv(self.symbol, '1h', limit=limit)
            if not bars or len(bars) < limit:
                print(f"❌ Not enough data for Enhanced EMA calculation: {len(bars) if bars else 0}/{limit}")
                return

            df = pd.DataFrame(bars, columns=['timestamp','open','high','low','close','volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)

            price = df['close'].iloc[-1]

            # Calculate indicators
            ema_fast = ta.ema(df['close'], length=self.fast_period)
            ema_slow = ta.ema(df['close'], length=self.slow_period)
            ema_trend = ta.ema(df['close'], length=self.trend_period)
            atr = ta.atr(df['high'], df['low'], df['close'], length=self.atr_period)
            volume_ma = ta.sma(df['volume'], length=self.volume_ma_period)
            obv = ta.obv(df['close'], df['volume'])
            k, d = ta.stochrsi(df['close'], length=self.stochrsi_rsi_len)

            # Ensure we have enough data
            if len(ema_trend) < self.trend_period:
                return

            # Trend and momentum checks
            bullish_trend = price > ema_trend.iloc[-1]
            bearish_trend = price < ema_trend.iloc[-1]
            volume_ok = df['volume'].iloc[-1] > volume_ma.iloc[-1]
            obv_trend = obv.iloc[-1] > obv.iloc[-2]

            ema_cross_up = ema_fast.iloc[-1] > ema_slow.iloc[-1] and ema_fast.iloc[-2] <= ema_slow.iloc[-2]
            ema_cross_down = ema_fast.iloc[-1] < ema_slow.iloc[-1] and ema_fast.iloc[-2] >= ema_slow.iloc[-2]
            stoch_bullish = k.iloc[-1] > d.iloc[-1] and d.iloc[-1] < 40
            stoch_bearish = d.iloc[-1] > k.iloc[-1] and d.iloc[-1] > 60

            # Risk sizing based on ATR
            atr_val = atr.iloc[-1]
            if atr_val <= 0:
                return
            equity = self.exchange.fetch_balance({'type':'swap','code':'USD'})['total']['USD']
            size = (equity * self.risk_per_trade) / (atr_val * self.atr_sl_multiplier)
            size = max(min(size, 0.5 * equity), 0.01 * equity)

            # Entry logic and advanced exit
            if not self._has_position():
                # Enter new position
                if bullish_trend and ema_cross_up and stoch_bullish and volume_ok and obv_trend:
                    sl = price - atr_val * self.atr_sl_multiplier
                    tp = price + atr_val * self.partial_profit_factor
                    self.exchange.create_market_buy_order(self.symbol, size)
                    stop = self.exchange.create_order(self.symbol, 'STOP_MARKET', 'sell', size, None, {'stopPrice': sl})
                    tp_ord = self.exchange.create_limit_sell_order(self.symbol, size, tp)
                    print(f"✅ Enter LONG {self.symbol}: Price={price:.2f}, SL={sl:.2f}, TP={tp:.2f}")
                    # Record entry state
                    self.entry_bar_idx = len(df) - 1
                    self.entry_price = price
                    self.entry_side = 'long'
                    self.partial_taken = False
                    self.trailing_sl = sl
                    self.stop_order_id = stop.get('id') if isinstance(stop, dict) else None
                    self.take_profit_order_id = tp_ord.get('id') if isinstance(tp_ord, dict) else None
                elif bearish_trend and ema_cross_down and stoch_bearish and volume_ok and not obv_trend:
                    sl = price + atr_val * self.atr_sl_multiplier
                    tp = price - atr_val * self.partial_profit_factor
                    self.exchange.create_market_sell_order(self.symbol, size)
                    stop = self.exchange.create_order(self.symbol, 'STOP_MARKET', 'buy', size, None, {'stopPrice': sl})
                    tp_ord = self.exchange.create_limit_buy_order(self.symbol, size, tp)
                    print(f"✅ Enter SHORT {self.symbol}: Price={price:.2f}, SL={sl:.2f}, TP={tp:.2f}")
                    # Record entry state
                    self.entry_bar_idx = len(df) - 1
                    self.entry_price = price
                    self.entry_side = 'short'
                    self.partial_taken = False
                    self.trailing_sl = sl
                    self.stop_order_id = stop.get('id') if isinstance(stop, dict) else None
                    self.take_profit_order_id = tp_ord.get('id') if isinstance(tp_ord, dict) else None
            else:
                # Advanced exit management for open positions
                # Partial profit taking at ATR multiple
                if not self.partial_taken:
                    if self.entry_side == 'long' and price >= self.entry_price + atr_val * self.partial_profit_factor:
                        half = size / 2
                        self.exchange.create_market_sell_order(self.symbol, half)
                        self.partial_taken = True
                        print(f"🟢 Partial profit taken: {half} @ {price:.2f}")
                    elif self.entry_side == 'short' and price <= self.entry_price - atr_val * self.partial_profit_factor:
                        half = size / 2
                        self.exchange.create_market_buy_order(self.symbol, half)
                        self.partial_taken = True
                        print(f"🟢 Partial profit taken (short): {half} @ {price:.2f}")
                # Trailing stop update
                if self.entry_side == 'long':
                    new_sl = price - atr_val * self.atr_sl_multiplier
                    if new_sl > self.trailing_sl:
                        if self.stop_order_id:
                            try: self.exchange.cancel_order(self.stop_order_id, self.symbol)
                            except: pass
                        stop = self.exchange.create_order(self.symbol, 'STOP_MARKET', 'sell', size if not self.partial_taken else size/2, None, {'stopPrice': new_sl})
                        self.trailing_sl = new_sl
                        self.stop_order_id = stop.get('id') if isinstance(stop, dict) else None
                        print(f"🔻 Updated trailing SL to {new_sl:.2f}")
                else:
                    new_sl = price + atr_val * self.atr_sl_multiplier
                    if new_sl < self.trailing_sl:
                        if self.stop_order_id:
                            try: self.exchange.cancel_order(self.stop_order_id, self.symbol)
                            except: pass
                        stop = self.exchange.create_order(self.symbol, 'STOP_MARKET', 'buy', size if not self.partial_taken else size/2, None, {'stopPrice': new_sl})
                        self.trailing_sl = new_sl
                        self.stop_order_id = stop.get('id') if isinstance(stop, dict) else None
                        print(f"🔻 Updated trailing SL to {new_sl:.2f}")
                # Time-based exit (after 5 bars if PnL small)
                if self.entry_bar_idx is not None:
                    duration = (len(df) - 1) - self.entry_bar_idx
                    pnl_pct = abs((price - self.entry_price) / self.entry_price) * 100
                    if duration > 5 and pnl_pct < 0.5:
                        self._cancel_orders()
                        if self.entry_side == 'long':
                            self.exchange.create_market_sell_order(self.symbol, size if not self.partial_taken else size/2)
                        else:
                            self.exchange.create_market_buy_order(self.symbol, size if not self.partial_taken else size/2)
                        print(f"⏱️ Time-based exit: closed position at {price:.2f}")
                        # Reset exit management
                        self.entry_bar_idx = None
                        self.entry_price = None
                        self.entry_side = None
                        self.partial_taken = False
                        self.trailing_sl = None
                        self.stop_order_id = None
                        self.take_profit_order_id = None

        except Exception as e:
            print(f"❌ Error in Enhanced EMA cycle: {e}")

    def _has_position(self) -> bool:
        if self.demo:
            return False
        bal = self.exchange.fetch_balance({'type':'swap','code':'USD'})
        positions = bal['info']['data']['positions']
        for pos in positions:
            if pos.get('symbol') == self.symbol and float(pos.get('size',0)) != 0:
                return True
        return False

    def _cancel_orders(self):
        if self.demo:
            return
        if self.entry_side == 'long':
            if self.stop_order_id:
                try: self.exchange.cancel_order(self.stop_order_id, self.symbol)
                except: pass
            if self.take_profit_order_id:
                try: self.exchange.cancel_order(self.take_profit_order_id, self.symbol)
                except: pass
        else:
            if self.stop_order_id:
                try: self.exchange.cancel_order(self.stop_order_id, self.symbol)
                except: pass
            if self.take_profit_order_id:
                try: self.exchange.cancel_order(self.take_profit_order_id, self.symbol)
                except: pass 