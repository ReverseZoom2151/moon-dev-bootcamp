import asyncio
import time
import logging
import ccxt
import pandas as pd
from core.config import get_settings

class Bot1TradingService:
    """Service to run Day 10's Bot1 trading logic in the ATS"""
    def __init__(self):
        settings = get_settings()
        # configure logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[logging.FileHandler('bot1_trader.log'), logging.StreamHandler()]
        )
        self.logger = logging.getLogger('bot1_trader')

        # trading config
        self.loop_interval = settings.BOT1_LOOP_INTERVAL
        self.symbol = settings.BOT1_SYMBOL
        self.pos_size = settings.BOT1_POS_SIZE
        self.target = settings.BOT1_TARGET_PROFIT
        self.max_loss = settings.BOT1_MAX_LOSS
        self.vol_decimal = settings.BOT1_VOL_DECIMAL
        self.params = {'timeInForce': 'PostOnly'}

        # rate limiting
        self._last_api_call_time = 0
        self._min_call_interval = settings.BOT1_MIN_CALL_INTERVAL

        # caches
        self._daily_sma_cache = None
        self._daily_sma_cache_time = 0
        self._f15_sma_cache = None
        self._f15_sma_cache_time = 0

        # initialize exchange
        if not settings.DEBUG_MODE:
            if not (settings.PHEMEX_API_KEY and settings.PHEMEX_SECRET_KEY):
                raise ValueError("PHEMEX_API_KEY and PHEMEX_SECRET_KEY required for Bot1 trading")
            self.exchange = ccxt.phemex({
                'enableRateLimit': True,
                'apiKey': settings.PHEMEX_API_KEY,
                'secret': settings.PHEMEX_SECRET_KEY
            })
        else:
            self.exchange = None

    async def start(self):
        self.logger.info(f"🚀 Starting Bot1 Trading Service (symbol={self.symbol})")
        while True:
            try:
                await asyncio.to_thread(self._bot_cycle)
            except Exception as e:
                self.logger.error(f"Error in Bot1 cycle: {e}")
            await asyncio.sleep(self.loop_interval)

    def _api_call(self, func, *args, **kwargs):
        """Rate-limited wrapper for CCXT API calls"""
        now = time.time()
        if now - self._last_api_call_time < self._min_call_interval:
            time.sleep(self._min_call_interval - (now - self._last_api_call_time))
        result = func(*args, **kwargs)
        self._last_api_call_time = time.time()
        return result

    def ask_bid(self):
        """Get the current ask and bid prices"""
        self.logger.debug(f"Fetching order book for {self.symbol}")
        ob = self._api_call(self.exchange.fetch_order_book, self.symbol)
        bid = ob['bids'][0][0] if ob['bids'] else None
        ask = ob['asks'][0][0] if ob['asks'] else None
        return ask, bid

    def daily_sma(self, cache_time=300):
        """Calculate daily SMA with caching"""
        now = time.time()
        if self._daily_sma_cache and now - self._daily_sma_cache_time < cache_time:
            return self._daily_sma_cache
        df = self._api_call(self.exchange.fetch_ohlcv, self.symbol, '1d', limit=100)
        df = pd.DataFrame(df, columns=['timestamp','open','high','low','close','volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df['sma20_d'] = df['close'].rolling(20).mean()
        ask, bid = self.ask_bid()
        df['sig'] = None
        df.loc[df['sma20_d'] > bid, 'sig'] = 'SELL'
        df.loc[df['sma20_d'] < bid, 'sig'] = 'BUY'
        self._daily_sma_cache, self._daily_sma_cache_time = df, now
        return df

    def f15_sma(self, cache_time=150):
        """Calculate 15-minute SMA and price bands with caching"""
        now = time.time()
        if self._f15_sma_cache and now - self._f15_sma_cache_time < cache_time:
            return self._f15_sma_cache
        bars = self._api_call(self.exchange.fetch_ohlcv, self.symbol, '15m', limit=100)
        df = pd.DataFrame(bars, columns=['timestamp','open','high','low','close','volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df['sma20_15'] = df['close'].rolling(20).mean()
        df['bp_1'] = df['sma20_15'] * 1.001
        df['bp_2'] = df['sma20_15'] * 0.997
        df['sp_1'] = df['sma20_15'] * 0.999
        df['sp_2'] = df['sma20_15'] * 1.003
        self._f15_sma_cache, self._f15_sma_cache_time = df, now
        return df

    def calculate_position_size(self):
        """Scale position size inversely with volatility"""
        df = self.f15_sma()
        vol = df['close'].pct_change().std() * 100
        size = self.pos_size * (1 / (1 + vol))
        return max(int(size), 1)

    def open_positions(self):
        """Check for open positions"""
        bal = self._api_call(self.exchange.fetch_balance, {'type':'swap','code':'USD'})
        pos = bal['info']['data']['positions'][0]
        side = pos['side']; size = pos['size']
        has = side in ['Buy','Sell']; is_long = (side=='Buy')
        return bal['info']['data']['positions'], has, size, is_long

    def kill_switch(self):
        """Close all open positions gracefully"""
        self.logger.info('Starting kill switch')
        _, has, size, is_long = self.open_positions()
        while has:
            self._api_call(self.exchange.cancel_all_orders, self.symbol)
            _, has, size, is_long = self.open_positions()
            ask, bid = self.ask_bid()
            if not is_long:
                self._api_call(self.exchange.create_limit_buy_order, self.symbol, size, bid, self.params)
            else:
                self._api_call(self.exchange.create_limit_sell_order, self.symbol, size, ask, self.params)
            time.sleep(30)

    def sleep_on_close(self):
        """Pause after recent fill"""
        orders = self._api_call(self.exchange.fetch_closed_orders, self.symbol)
        for o in reversed(orders):
            status = o['info']['ordStatus']
            t_ns = int(int(o['info']['transactTimeNs'])/1e9)
            ask, _ = self._api_call(self.exchange.fetch_order_book, self.symbol).values()
            ex_ts = int(self._api_call(self.exchange.fetch_order_book, self.symbol)['timestamp']/1000)
            mins = (ex_ts - t_ns)/60
            if status=='Filled' and mins<59:
                time.sleep(60)
            break

    def ob(self):
        """Order-book volume bias"""
        df = pd.DataFrame()
        for _ in range(11):
            ob = self._api_call(self.exchange.fetch_order_book, self.symbol)
            bid_vol = sum(v for _,v in ob['bids'])
            ask_vol = sum(v for _,v in ob['asks'])
            df = pd.concat([df, pd.DataFrame({'bid_vol':[bid_vol],'ask_vol':[ask_vol]})], ignore_index=True)
            time.sleep(5)
        tb, ta = df['bid_vol'].sum(), df['ask_vol'].sum()
        bullish = tb>ta
        _, has, _, is_long = self.open_positions()
        ratio = (ta/tb if bullish else tb/ta)
        return has and ((is_long and ratio<self.vol_decimal) or (not is_long and ratio<self.vol_decimal))

    def check_circuit_breakers(self):
        """Skip trading on high volatility or spread"""
        df = self.f15_sma()
        vol = df['close'].pct_change().rolling(5).std().iloc[-1]*100
        if vol>5: return False
        ask, bid = self.ask_bid()
        if ((ask-bid)/bid*100)>0.5: return False
        return True

    def pnl_close(self):
        """Monitor PnL and close positions"""
        bal = self._api_call(self.exchange.fetch_positions, {'type':'swap','code':'USD'})
        pos = bal[0]
        side = pos['side']; size = pos['contracts']
        entry = float(pos['entryPrice']); lev = float(pos['leverage'])
        _, current = self.ask_bid()
        diff = (current-entry) if side=='long' else (entry-current)
        perc = (diff/entry)*lev*100 if entry else 0
        if perc>=self.target or perc<=self.max_loss:
            if perc>=self.target and self.ob(): time.sleep(30)
            else: self.kill_switch()
            return True, True, size, (side=='long')
        return False, False, size, (side=='long')

    def _bot_cycle(self):
        """One iteration of the trading bot"""
        if not self.check_circuit_breakers():
            time.sleep(60); return
        pnl_res = self.pnl_close()
        self.sleep_on_close()
        df_d = self.daily_sma(); df_f = self.f15_sma()
        sig = df_d.iloc[-1]['sig']; ask, bid = self.ask_bid()
        size = self.calculate_position_size(); open_size = size//2
        _, in_pos, curr_size, _ = self.open_positions()
        if not in_pos and curr_size<self.pos_size:
            self._api_call(self.exchange.cancel_all_orders, self.symbol)
            last_sma15 = df_f.iloc[-1]['sma20_15']
            if sig=='BUY' and bid>last_sma15:
                bp1, bp2 = df_f.iloc[-1][['bp_1','bp_2']]
                self._api_call(self.exchange.create_limit_buy_order, self.symbol, open_size, bp1, self.params)
                self._api_call(self.exchange.create_limit_buy_order, self.symbol, open_size, bp2, self.params)
                time.sleep(120)
            elif sig=='SELL' and ask<last_sma15:
                sp1, sp2 = df_f.iloc[-1][['sp_1','sp_2']]
                self._api_call(self.exchange.create_limit_sell_order, self.symbol, open_size, sp1, self.params)
                self._api_call(self.exchange.create_limit_sell_order, self.symbol, open_size, sp2, self.params)
                time.sleep(120)
        else:
            time.sleep(600) 