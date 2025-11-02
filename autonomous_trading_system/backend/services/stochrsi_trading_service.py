import asyncio
import ccxt
import pandas as pd
from ta.momentum import StochRSIIndicator
from core.config import get_settings

class StochRSITradingService:
    """Service to run the Day 16 StochRSI trading bot natively in the ATS"""
    def __init__(self):
        settings = get_settings()
        self.loop_interval = settings.STOCHRSI_LOOP_INTERVAL
        self.demo = getattr(settings, 'SMA_DEMO_MODE', False)
        self.symbol = settings.STOCHRSI_SYMBOL
        self.timeframe = settings.STOCHRSI_TIMEFRAME
        self.limit = settings.STOCHRSI_LIMIT
        self.rsi_period = settings.STOCHRSI_RSI_PERIOD
        self.stoch_period = settings.STOCHRSI_K_PERIOD
        self.smooth_k = settings.STOCHRSI_K_PERIOD
        self.smooth_d = settings.STOCHRSI_D_PERIOD
        # Overbought/oversold thresholds
        ob = settings.STOCHRSI_OVERBOUGHT
        os = settings.STOCHRSI_OVERSOLD
        self.overbought = ob * 100 if ob <= 1 else ob
        self.oversold = os * 100 if os <= 1 else os

        if not self.demo:
            if not (settings.PHEMEX_API_KEY and settings.PHEMEX_SECRET_KEY):
                raise ValueError("PHEMEX_API_KEY and PHEMEX_SECRET_KEY required for StochRSI trading")
            self.exchange = ccxt.phemex({
                'enableRateLimit': True,
                'apiKey': settings.PHEMEX_API_KEY,
                'secret': settings.PHEMEX_SECRET_KEY
            })
        else:
            self.exchange = None

        # Order sizing and exit factors
        self.position_size = settings.PHEMEX_ORDER_SIZE
        self.stop_loss_factor = settings.STOCHRSI_STOP_LOSS_FACTOR
        self.take_profit_factor = settings.STOCHRSI_TAKE_PROFIT_FACTOR

    def _has_position(self) -> bool:
        if self.demo:
            return False
        bal = self.exchange.fetch_balance({'type': 'swap', 'code': 'USD'})
        positions = bal['info']['data']['positions']
        for pos in positions:
            if pos.get('symbol') == self.symbol and float(pos.get('size', 0)) != 0:
                return True
        return False

    def _get_price(self) -> float:
        ticker = self.exchange.fetch_ticker(self.symbol)
        return float(ticker.get('last', 0))

    def _cancel_orders(self) -> None:
        orders = self.exchange.fetch_open_orders(self.symbol)
        for o in orders:
            try:
                self.exchange.cancel_order(o['id'], self.symbol)
            except Exception:
                pass

    async def start(self):
        print(f"🚀 Starting StochRSI Trading Service (demo={self.demo})")
        while True:
            await asyncio.to_thread(self._run_cycle)
            await asyncio.sleep(self.loop_interval)

    def _run_cycle(self):
        try:
            if self.demo:
                print("[DEMO] StochRSI cycle skipped in demo mode")
                return

            # Fetch OHLCV data
            bars = self.exchange.fetch_ohlcv(self.symbol, self.timeframe, limit=self.limit)
            if not bars or len(bars) < self.limit:
                print(f"❌ Not enough data for StochRSI calculation: {len(bars) if bars else 0}/{self.limit}")
                return

            df = pd.DataFrame(bars, columns=['timestamp','open','high','low','close','volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            close = df['close']

            # Calculate Stochastic RSI
            indicator = StochRSIIndicator(close, window=self.rsi_period, smooth1=self.smooth_k, smooth2=self.smooth_d)
            k = indicator.stochrsi_k()
            d = indicator.stochrsi_d()

            # Ensure enough data points
            if len(k) < 2 or len(d) < 2:
                return

            prev_k, curr_k = k.iloc[-2], k.iloc[-1]
            prev_d, curr_d = d.iloc[-2], d.iloc[-1]

            bullish = (prev_k <= prev_d) and (curr_k > curr_d)
            bearish = (prev_k >= prev_d) and (curr_k < curr_d)
            is_oversold = curr_k < self.oversold and curr_d < self.oversold
            is_overbought = curr_k > self.overbought and curr_d > self.overbought

            # Entry and exit logic
            if bullish and is_oversold:
                print(f"📈 StochRSI BUY signal for {self.symbol} | K={curr_k:.2f}, D={curr_d:.2f}")
                if not self._has_position():
                    entry_price = self._get_price()
                    # Enter long
                    self.exchange.create_market_buy_order(self.symbol, self.position_size)
                    # Cancel any existing orders
                    self._cancel_orders()
                    # Calculate and place stop-loss and take-profit
                    sl = entry_price * self.stop_loss_factor
                    tp = entry_price * self.take_profit_factor
                    # Stop-loss order
                    self.exchange.create_order(self.symbol, 'STOP_MARKET', 'sell', self.position_size, None, {'stopPrice': sl})
                    # Take-profit limit
                    self.exchange.create_limit_sell_order(self.symbol, self.position_size, tp)
                    print(f"✅ Executed BUY @ {entry_price:.2f}, SL={sl:.2f}, TP={tp:.2f}")
            elif bearish and is_overbought:
                print(f"📉 StochRSI SELL signal for {self.symbol} | K={curr_k:.2f}, D={curr_d:.2f}")
                if self._has_position():
                    # Exit position
                    self._cancel_orders()
                    self.exchange.create_market_sell_order(self.symbol, self.position_size)
                    print(f"❌ Closed position for {self.symbol}")
            else:
                print(f"⏸️ StochRSI HOLD for {self.symbol} | K={curr_k:.2f}, D={curr_d:.2f}")

        except Exception as e:
            print(f"❌ Error in StochRSI cycle: {e}") 