import asyncio
import ccxt
import pandas as pd
import logging
import time
from ta.momentum import RSIIndicator
from core.config import get_settings

class VwapTradingService:
    """Service to run the Day 8 VWAP indicator trading logic natively in the ATS"""
    def __init__(self):
        settings = get_settings()
        # configure logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[logging.FileHandler('vwap_trader.log'), logging.StreamHandler()]
        )
        self.logger = logging.getLogger('vwap_trader')

        self.demo = settings.SMA_DEMO_MODE
        self.loop_interval = settings.VWAP_LOOP_INTERVAL
        self.symbol = settings.VWAP_SYMBOL
        self.timeframe = settings.VWAP_TIMEFRAME
        self.limit = settings.VWAP_LIMIT
        self.target = settings.VWAP_TARGET_PROFIT_PCT
        self.max_loss = settings.VWAP_MAX_LOSS_PCT
        self.max_risk = settings.VWAP_MAX_RISK_AMOUNT
        self.position_size = settings.VWAP_POSITION_SIZE
        self.auto_trade = settings.VWAP_AUTO_TRADE
        self.order_params = settings.VWAP_ORDER_PARAMS

        if not self.demo:
            if not (settings.PHEMEX_API_KEY and settings.PHEMEX_SECRET_KEY):
                raise ValueError("PHEMEX_API_KEY and PHEMEX_SECRET_KEY required for VWAP trading")
            self.exchange = ccxt.phemex({
                'enableRateLimit': True,
                'apiKey': settings.PHEMEX_API_KEY,
                'secret': settings.PHEMEX_SECRET_KEY
            })
        else:
            self.exchange = None

    async def start(self):
        self.logger.info(f"🚀 Starting VWAP Trading Service (demo={self.demo})")
        while True:
            try:
                await asyncio.to_thread(self.run_trading_cycle, self.auto_trade, self.position_size)
            except Exception as e:
                self.logger.error(f"Error in VWAP cycle: {e}")
            await asyncio.sleep(self.loop_interval)

    def get_market_data(self):
        if self.demo:
            return 0, 0
        ob = self.exchange.fetch_order_book(self.symbol)
        bid = ob['bids'][0][0] if ob['bids'] else 0
        ask = ob['asks'][0][0] if ob['asks'] else 0
        return ask, bid

    def get_position_info(self):
        if self.demo:
            return None, False, 0, None, None
        bal = self.exchange.fetch_balance({'type':'swap','code':'USD'})
        positions = bal['info']['data']['positions']
        # map symbol index
        idx_map = {'BTC/USD:BTC':4,'APEUSD':2,'ETHUSD':3,'DOGEUSD':1,'u100000SHIBUSD':0}
        idx = idx_map.get(self.symbol, None)
        if idx is None or idx >= len(positions):
            return positions, False, 0, None, idx
        pos = positions[idx]
        side = pos['side']
        size = float(pos['size'])
        has = size>0
        is_long = True if side=='Buy' else False
        return positions, has, size, is_long, idx

    def close_position(self):
        self.logger.info(f"Starting to close position for {self.symbol}")
        _, has, size, is_long, _ = self.get_position_info()
        if not has:
            self.logger.info("No position to close")
            return True
        attempts = 0
        while has and attempts<5:
            attempts+=1
            self.exchange.cancel_all_orders(self.symbol)
            ask, bid = self.get_market_data()
            if ask==0 or bid==0:
                time.sleep(5); continue
            if is_long:
                self.exchange.create_limit_sell_order(self.symbol, size, ask, self.order_params)
            else:
                self.exchange.create_limit_buy_order(self.symbol, size, bid, self.order_params)
            time.sleep(30)
            _, has, size, is_long, _ = self.get_position_info()
        return not has

    def check_pnl(self):
        bal = self.exchange.fetch_positions({'type':'swap','code':'USD'})
        _, has, size, is_long, idx = self.get_position_info()
        if not has: return False, False, 0, None
        pos = bal[idx]
        entry = float(pos['entryPrice']); lev=float(pos['leverage'])
        current = self.get_market_data()[1]
        diff = current-entry if is_long else entry-current
        perc = (diff/entry)*lev*100 if entry else 0
        self.logger.info(f"P&L: {perc:.2f}%")
        if perc>self.target or perc<=self.max_loss:
            self.close_position(); return True, has, size, is_long
        return False, has, size, is_long

    def check_max_risk(self):
        bal = self.exchange.fetch_balance({'type':'swap','code':'USD'})
        positions = bal['info']['data']['positions']
        cost=sum(float(p.get('posCost',0)) for p in positions)
        if cost>self.max_risk:
            self.logger.warning(f"EMERGENCY KILL SWITCH: Position cost {cost} > {self.max_risk}")
            self.close_position()
            return False
        return True

    def calculate_sma(self):
        bars=self.exchange.fetch_ohlcv(self.symbol,self.timeframe,limit=self.limit)
        df=pd.DataFrame(bars,columns=['timestamp','open','high','low','close','volume'])
        df['timestamp']=pd.to_datetime(df['timestamp'],unit='ms')
        df['sma']=df['close'].rolling(self.period).mean()
        bid=self.get_market_data()[1]
        df['signal']=None
        df.loc[df['sma']>bid,'signal']='SELL'
        df.loc[df['sma']<bid,'signal']='BUY'
        df['support']=df['close'].rolling(self.limit).min()
        df['resistance']=df['close'].rolling(self.limit).max()
        return df

    def calculate_rsi(self):
        bars=self.exchange.fetch_ohlcv(self.symbol,self.timeframe,limit=self.limit)
        df=pd.DataFrame(bars,columns=['timestamp','open','high','low','close','volume'])
        df['timestamp']=pd.to_datetime(df['timestamp'],unit='ms')
        rsi=RSIIndicator(df['close'],window=14)
        df['rsi']=rsi.rsi()
        return df

    def calculate_vwap(self):
        bars=self.exchange.fetch_ohlcv(self.symbol,self.timeframe,limit=self.limit)
        df=pd.DataFrame(bars,columns=['timestamp','open','high','low','close','volume'])
        df['timestamp']=pd.to_datetime(df['timestamp'],unit='ms')
        df['typical_price']=(df['high']+df['low']+df['close'])/3
        df['volume_x_typical']=df['volume']*df['typical_price']
        df['cum_volume']=df['volume'].cumsum()
        df['cum_vol_x_price']=df['volume_x_typical'].cumsum()
        df['vwap']=df['cum_vol_x_price']/df['cum_volume']
        current_price=self.get_market_data()[1]
        last_vwap=df['vwap'].iloc[-1]
        df['vwap_signal']=None
        if current_price>last_vwap:
            df.loc[df.index[-1],'vwap_signal']='BUY'
        else:
            df.loc[df.index[-1],'vwap_signal']='SELL'
        return df

    def get_combined_signal(self):
        sma_df=self.calculate_sma()
        rsi_df=self.calculate_rsi()
        vwap_df=self.calculate_vwap()
        if sma_df.empty or rsi_df.empty or vwap_df.empty:
            return 'NEUTRAL'
        sma_signal=sma_df['signal'].iloc[-1]
        vwap_signal=vwap_df['vwap_signal'].iloc[-1]
        rsi_val=rsi_df['rsi'].iloc[-1]; rsi_signal='BUY' if rsi_val<30 else 'SELL' if rsi_val>70 else 'NEUTRAL'
        signals=[sma_signal,vwap_signal,rsi_signal]
        buy=sum(1 for s in signals if s=='BUY'); sell=sum(1 for s in signals if s=='SELL')
        return 'BUY' if buy>=2 else 'SELL' if sell>=2 else 'NEUTRAL'

    def execute_trade(self, signal, size):
        has_pos=self.get_position_info()[1]
        ask,bid=self.get_market_data()
        if has_pos and ((signal=='BUY' and not self.get_position_info()[3]) or (signal=='SELL' and self.get_position_info()[3])):
            self.close_position()
        if signal=='BUY':
            self.exchange.create_limit_buy_order(self.symbol,size,bid,self.order_params)
        elif signal=='SELL':
            self.exchange.create_limit_sell_order(self.symbol,size,ask,self.order_params)

    def run_trading_cycle(self, auto_trade=False, position_size=1):
        self.check_max_risk()
        _,in_pos,_,_,_=self.get_position_info()
        if in_pos:
            self.check_pnl()
        signal=self.get_combined_signal()
        if auto_trade and signal in ['BUY','SELL']:
            self.execute_trade(signal,position_size)
        return signal 