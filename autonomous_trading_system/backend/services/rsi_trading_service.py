import asyncio
import ccxt
import pandas as pd
import time
from ta.momentum import RSIIndicator
from core.config import get_settings

class RsiTradingService:
    """Service to run the Day 7 RSI indicator bot natively in the ATS"""
    def __init__(self):
        settings = get_settings()
        self.loop_interval = settings.RSI_LOOP_INTERVAL
        # Reuse SMA demo flag for RSI demo mode
        self.demo = getattr(settings, 'SMA_DEMO_MODE', False)
        # Default RSI settings from Day 7 script
        self.symbol = 'BTC/USD:BTC'
        self.timeframe = '15m'
        self.limit = 100
        self.rsi_period = 14
        if not self.demo:
            # Initialize Phemex exchange via CCXT
            if not (settings.PHEMEX_API_KEY and settings.PHEMEX_SECRET_KEY):
                raise ValueError("PHEMEX_API_KEY and PHEMEX_SECRET_KEY required for RSI trading")
            self.exchange = ccxt.phemex({
                'enableRateLimit': True,
                'apiKey': settings.PHEMEX_API_KEY,
                'secret': settings.PHEMEX_SECRET_KEY
            })
        else:
            self.exchange = None

    async def start(self):
        print(f"🚀 Starting RSI Trading Service (demo={self.demo})")
        while True:
            await asyncio.to_thread(self._run_cycle)
            await asyncio.sleep(self.loop_interval)

    def _run_cycle(self):
        try:
            # Open position info
            pos_data, has_pos, size, is_long, _ = self._open_positions(self.symbol)
            print(f"Position for {self.symbol}: Open={has_pos}, Size={size}, Long={is_long}")
            # Calculate RSI and signal
            df = self._calculate_rsi(self.symbol)
            if not df.empty and 'rsi' in df.columns:
                latest = df['rsi'].iloc[-1]
                signal = 'BUY' if latest < 30 else 'SELL' if latest > 70 else 'NEUTRAL'
                print(f"Latest RSI for {self.symbol}: {latest:.2f} | Signal: {signal}")
            else:
                print(f"RSI data unavailable for {self.symbol}")
            # PnL-based closure
            try:
                self._pnl_close(self.symbol)
            except Exception as e:
                print(f"Error in pnl_close for {self.symbol}: {e}")
            # Size-based closure
            try:
                self._size_kill()
            except Exception as e:
                print(f"Error in size_kill: {e}")
        except Exception as e:
            print(f"❌ Error in RSI cycle: {e}")

    def _has_position(self, symbol):
        if self.demo:
            return False
        bal = self.exchange.fetch_balance({'type':'swap','code':'USD'})
        positions = bal['info']['data']['positions']
        for pos in positions:
            if pos.get('symbol') == symbol and float(pos.get('size', 0)) != 0:
                return True
        return False

    def _calculate_rsi(self, symbol):
        if self.demo:
            # Simulate RSI
            import random
            rsi = random.uniform(0, 100)
            print(f"[DEMO] RSI for {symbol}: {rsi:.2f}")
            return pd.DataFrame({'rsi': [rsi]})
        # Fetch OHLCV data
        bars = self.exchange.fetch_ohlcv(symbol, self.timeframe, limit=self.limit)
        if not bars or len(bars) < self.rsi_period:
            print("❌ Not enough data for RSI calculation")
            return pd.DataFrame()
        df = pd.DataFrame(bars, columns=['timestamp','open','high','low','close','volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        # Calculate RSI indicator
        indicator = RSIIndicator(df['close'], window=self.rsi_period)
        df['rsi'] = indicator.rsi()
        return df

    def _open_positions(self, symbol):
        """Get open position info for symbol"""
        if self.demo:
            return [], False, 0, None, None
        bal = self.exchange.fetch_balance({'type':'swap','code':'USD'})
        positions = bal['info']['data']['positions']
        for pos in positions:
            if pos.get('symbol') == symbol:
                size = float(pos.get('size', 0))
                has_pos = size != 0
                is_long = True if pos.get('side') == 'Buy' else False
                return positions, has_pos, size, is_long, None
        return positions, False, 0, None, None

    def _ask_bid(self, symbol):
        """Fetch ask and bid prices for symbol"""
        if self.demo:
            return 0, 0
        ob = self.exchange.fetch_order_book(symbol)
        bid = ob['bids'][0][0] if ob.get('bids') else 0
        ask = ob['asks'][0][0] if ob.get('asks') else 0
        return ask, bid

    def _kill_switch(self, symbol):
        """Close open position for symbol using kill switch"""
        if self.demo:
            print(f"[DEMO] RSI kill switch for {symbol}")
            return True
        max_attempts = 5
        attempt = 0
        while attempt < max_attempts:
            attempt += 1
            print(f"Kill switch attempt {attempt}/{max_attempts} for {symbol}")
            try:
                # cancel all orders
                self.exchange.cancel_all_orders(symbol)
                # check position
                _, has_pos, size, is_long, _ = self._open_positions(symbol)
                if not has_pos:
                    print(f"Position closed for {symbol}")
                    return True
                ask, bid = self._ask_bid(symbol)
                if ask == 0 or bid == 0:
                    time.sleep(5)
                    continue
                # place closing order
                if is_long:
                    self.exchange.create_limit_sell_order(symbol, size, ask, self.order_params)
                else:
                    self.exchange.create_limit_buy_order(symbol, size, bid, self.order_params)
                time.sleep(30)
            except Exception as e:
                print(f"Error in kill_switch: {e}")
                time.sleep(5)
        print(f"Failed to close position for {symbol} after {max_attempts} attempts")
        return False

    def _pnl_close(self, symbol):
        """Check profit/loss and close if thresholds hit"""
        positions = self.exchange.fetch_positions({'type':'swap','code':'USD'})
        for pos in positions:
            if pos.get('symbol') == symbol:
                side = pos.get('side')
                entry = float(pos.get('entryPrice', 0))
                leverage = float(pos.get('leverage', 1))
                ask, bid = self._ask_bid(symbol)
                current = bid
                diff = (current - entry) if side == 'long' else (entry - current)
                perc = (diff / entry) * leverage * 100 if entry else 0
                print(f"PnL for {symbol}: {perc:.2f}%")
                if perc >= self.target or perc <= self.max_loss:
                    print(f"PnL out of bounds ({perc:.2f}%), closing {symbol}")
                    self._kill_switch(symbol)
                return

    def _size_kill(self):
        """Close positions exceeding max risk"""
        bal = self.exchange.fetch_balance({'type':'swap','code':'USD'})
        positions = bal['info']['data']['positions']
        for pos in positions:
            cost = float(pos.get('posCost', 0))
            sym = pos.get('symbol')
            if cost > self.max_risk:
                print(f"Position {sym} cost {cost} exceeds max risk {self.max_risk}")
                self._kill_switch(sym) 