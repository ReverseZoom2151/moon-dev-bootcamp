import asyncio
import logging
import ccxt
import time
import pandas as pd
from core.config import get_settings

class VwmaTradingService:
    """Service to run the full Day 9 VWMA trading logic in the ATS"""
    def __init__(self):
        settings = get_settings()
        # configure logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[logging.FileHandler('vwma_trader.log'), logging.StreamHandler()]
        )
        self.logger = logging.getLogger('vwma_trader')

        # trading parameters
        self.demo = settings.VWMA_DEMO_MODE
        self.loop_interval = settings.VWMA_LOOP_INTERVAL
        self.symbol = settings.VWMA_SYMBOL
        self.size = settings.VWMA_ORDER_SIZE
        self.target = settings.VWMA_TARGET_PROFIT_PCT
        self.max_loss = settings.VWMA_MAX_LOSS_PCT
        self.timeframe = settings.VWMA_TIMEFRAME
        self.limit = settings.VWMA_LIMIT
        self.params = settings.VWMA_ORDER_PARAMS
        # Rate limiting setup
        self._last_api_call_time = 0
        self._min_call_interval = 0.2

        # initialize exchange
        if not self.demo:
            if not (get_settings().PHEMEX_API_KEY and get_settings().PHEMEX_SECRET_KEY):
                raise ValueError("PHEMEX_API_KEY and PHEMEX_SECRET_KEY required for VWMA trading")
            self.exchange = ccxt.phemex({
                'enableRateLimit': True,
                'apiKey': get_settings().PHEMEX_API_KEY,
                'secret': get_settings().PHEMEX_SECRET_KEY
            })
        else:
            self.exchange = None

    async def start(self):
        self.logger.info(f"🚀 Starting VWMA Trading Service (demo={self.demo})")
        while True:
            try:
                await asyncio.to_thread(self.run)
            except Exception as e:
                self.logger.error(f"Error in VWMA run cycle: {e}")
            await asyncio.sleep(self.loop_interval)

    def run(self):
        """One trading cycle: risk checks, indicator analysis, manage positions, and (optionally) place orders."""
        # Emergency size-based kill
        self.size_kill()
        # Analyze market and generate VWMA/RSI signals, then manage existing position
        analysis = self.analyze_market()
        is_open, _, _ = self.check_and_manage_positions()
        # Entry logic
        if self.params.get('auto_trade', True) and analysis:
            signal = analysis['signal']
            if not is_open and signal in ['BUY', 'SELL']:
                self.execute_trade(signal)
        self.logger.info("✅ VWMA trading cycle completed")

    def _api_call(self, func, *args, **kwargs):
        """Rate-limited wrapper for CCXT API calls"""
        now = time.time()
        if now - self._last_api_call_time < self._min_call_interval:
            time.sleep(self._min_call_interval - (now - self._last_api_call_time))
        result = func(*args, **kwargs)
        self._last_api_call_time = time.time()
        return result

    # ------ ported methods from 9_vwma.py below ------
    def get_symbol_position_index(self, symbol=None):
        if symbol is None:
            symbol = self.symbol
        mapping = {'BTC/USD:BTC':4,'ETH/USD:ETH':2,'ETHUSD':3,'DOGEUSD':1,'u100000SHIBUSD':0}
        return mapping.get(symbol, None)

    def open_positions(self, symbol=None):
        """Check for open positions"""
        if symbol is None:
            symbol = self.symbol
        self.logger.debug(f"Checking open positions for {symbol}")
        idx = self.get_symbol_position_index(symbol)
        try:
            params = {'type': 'swap', 'code': 'USD'}
            bal = self._api_call(self.exchange.fetch_balance, params)
            positions = bal['info']['data']['positions']
            side = positions[idx]['side']
            size = positions[idx]['size']
            if side == 'Buy': has, is_long = True, True
            elif side == 'Sell': has, is_long = True, False
            else: has, is_long = False, None
            self.logger.info(f"Position check - In position: {has}, Size: {size}, Long: {is_long}, Index: {idx}")
            return positions, has, size, is_long, idx
        except Exception as e:
            self.logger.error(f"Error checking open positions: {e}")
            return [], False, 0, None, idx

    def ask_bid(self, symbol=None):
        """Get the current ask and bid prices"""
        if symbol is None:
            symbol = self.symbol
        self.logger.debug(f"Fetching order book for {symbol}")
        try:
            ob = self._api_call(self.exchange.fetch_order_book, symbol)
            bid = ob['bids'][0][0] if ob['bids'] else None
            ask = ob['asks'][0][0] if ob['asks'] else None
            self.logger.info(f"Current prices for {symbol} - Ask: {ask}, Bid: {bid}")
            return ask, bid
        except Exception as e:
            self.logger.error(f"Error fetching order book: {e}")
            return None, None

    def kill_switch(self, symbol=None):
        """Gracefully close positions using limit orders"""
        if symbol is None:
            symbol = self.symbol
        self.logger.info(f"Starting kill switch for {symbol}")
        try:
            _, has, size, is_long, _ = self.open_positions(symbol)
            self.logger.info(f"Kill switch - In position: {has}, Long: {is_long}, Size: {size}")
            while has:
                self.logger.info("Starting kill switch loop until limit fills...")
                self._api_call(self.exchange.cancel_all_orders, symbol)
                _, has, size, is_long, _ = self.open_positions(symbol)
                size = int(size)
                ask, bid = self.ask_bid(symbol)
                if is_long is False:
                    self._api_call(self.exchange.create_limit_buy_order, symbol, size, bid, self.params)
                    self.logger.info(f"Created BUY to CLOSE order of {size} {symbol} at ${bid}")
                elif is_long is True:
                    self._api_call(self.exchange.create_limit_sell_order, symbol, size, ask, self.params)
                    self.logger.info(f"Created SELL to CLOSE order of {size} {symbol} at ${ask}")
                else:
                    break
                time.sleep(30)
            self.logger.info(f"Kill switch completed for {symbol}")
        except Exception as e:
            self.logger.error(f"Error in kill switch: {e}")

    def pnl_close(self, symbol=None, target=None, max_loss=None):
        """Close positions based on PnL targets"""
        if symbol is None:
            symbol = self.symbol
        if target is None:
            target = self.target
        if max_loss is None:
            max_loss = self.max_loss
        self.logger.info(f"Checking if it's time to exit for {symbol}...")
        try:
            params = {'type': 'swap', 'code': 'USD'}
            pos_list = self._api_call(self.exchange.fetch_positions, params)
            idx = self.open_positions(symbol)[4]
            pos = pos_list[idx]
            side = pos['side']
            contracts = pos.get('contracts', 0)
            entry = float(pos.get('entryPrice', 0))
            lev = float(pos.get('leverage', 1))
            current = self.ask_bid(symbol)[1]
            self.logger.info(f"Position details - Side: {side}, Entry: {entry}, Leverage: {lev}")
            if side.lower() == 'long':
                diff = current - entry; in_long = True
            else:
                diff = entry - current; in_long = False
            perc = 100 * ((diff / entry) * lev) if entry else 0
            self.logger.info(f"For {symbol} current PnL: {perc:.2f}%")
            if perc > target:
                self.logger.info(f"Profit target reached ({perc:.2f}%), exiting")
                self.kill_switch(symbol)
                return True, True, contracts, in_long
            if perc <= max_loss:
                self.logger.warning(f"Stop loss triggered ({perc:.2f}%), exiting")
                self.kill_switch(symbol)
                return False, True, contracts, in_long
            return False, False, contracts, in_long
        except Exception as e:
            self.logger.error(f"Error in PnL calculation: {e}")
            return False, False, 0, None

    def size_kill(self):
        """Emergency kill switch if position size exceeds risk limits"""
        max_risk = 1000
        self.logger.debug("Checking position size against risk limits")
        try:
            params = {'type': 'swap', 'code': 'USD'}
            bal = self._api_call(self.exchange.fetch_balance, params)
            positions = bal['info']['data']['positions']
            cost = float(positions[0].get('posCost', 0))
            self.logger.info(f"Position cost: {cost}")
            if cost > max_risk:
                self.logger.critical(f"EMERGENCY KILL SWITCH: cost {cost} > {max_risk}")
                self.kill_switch(self.symbol)
                time.sleep(300)
                return True
            return False
        except Exception as e:
            self.logger.error(f"Error in size kill: {e}")
            return False

    def get_df_vwma(self, symbol=None, timeframe=None, num_bars=None, cache_time=None):
        """Fetch raw OHLCV data for VWMA calculations"""
        if symbol is None:
            symbol = self.symbol
        if timeframe is None:
            timeframe = self.timeframe
        if num_bars is None:
            num_bars = self.limit
        bars = self._api_call(self.exchange.fetch_ohlcv, symbol, timeframe, num_bars)
        df = pd.DataFrame(bars, columns=['timestamp','open','high','low','close','volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        return df

    def vwma_indi(self, cache_time=None):
        """Calculate VWMA indicators"""
        df = self.get_df_vwma()
        if df.empty:
            return df
        for n in [20, 41, 75]:
            vol = df['volume']
            df[f'VWMA({n})'] = (vol * df['close']).rolling(n).sum() / vol.rolling(n).sum()
        return df

    def analyze_market(self):
        """Analyze the market and generate trading signals based on VWMA crossovers"""
        self.logger.info("Starting market analysis")
        try:
            df = self.vwma_indi()
            if df.empty:
                self.logger.error("Failed to generate VWMA indicators")
                return None
            last = df.iloc[-1]
            vwma20 = last.get('VWMA(20)', 0)
            vwma41 = last.get('VWMA(41)', 0)
            vwma75 = last.get('VWMA(75)', 0)
            _, current = self.ask_bid(self.symbol)
            if vwma20 > vwma41:
                signal = 'BUY'
                strength = 'Strong' if vwma20 > vwma75 else 'Moderate'
            else:
                signal = 'SELL'
                strength = 'Strong' if vwma20 < vwma75 else 'Moderate'
            self.logger.info(f"Trading signal: {signal} ({strength})")
            return {'signal': signal, 'strength': strength, 'vwma20': vwma20, 'vwma41': vwma41, 'vwma75': vwma75, 'current_price': current}
        except Exception as e:
            self.logger.error(f"Error in market analysis: {e}")
            return None

    def check_and_manage_positions(self):
        """Check and manage existing positions"""
        self.logger.info("Checking and managing positions")
        try:
            _, has, size, is_long, _ = self.open_positions(self.symbol)
            if has:
                self.logger.info(f"Position status: {'Long' if is_long else 'Short'}, Size: {size}")
                self.pnl_close(self.symbol, self.target, self.max_loss)
            return has, is_long, size
        except Exception as e:
            self.logger.error(f"Error managing positions: {e}")
            return False, False, 0

    def execute_trade(self, signal):
        """Place orders based on the generated signal"""
        _, has, _, is_long, _ = self.open_positions(self.symbol)
        ask, bid = self.ask_bid(self.symbol)
        if has and ((signal == 'BUY' and not is_long) or (signal == 'SELL' and is_long)):
            self.kill_switch(self.symbol)
        if signal == 'BUY':
            self._api_call(self.exchange.create_limit_buy_order, self.symbol, self.size, bid, self.params)
        elif signal == 'SELL':
            self._api_call(self.exchange.create_limit_sell_order, self.symbol, self.size, ask, self.params) 