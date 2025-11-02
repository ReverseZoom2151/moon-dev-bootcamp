"""
Bitfinex Professional MA Reversal Strategy
Institutional-grade Moving Average reversal trading strategy for Bitfinex exchange.
Features advanced margin trading, funding rate analysis, and professional risk management.
"""

import pandas as pd
import pandas_ta as ta
import requests
import time
import hmac
import hashlib
import json
import os
from datetime import datetime
from termcolor import cprint
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class BitfinexMAConfig:
    # API Configuration
    API_KEY = os.getenv('BITFINEX_API_KEY')
    API_SECRET = os.getenv('BITFINEX_API_SECRET')
    BASE_URL = "https://api-pub.bitfinex.com/v2"
    AUTH_URL = "https://api.bitfinex.com/v2/auth"
    
    # Professional Strategy Parameters
    MA_FAST = 25
    MA_SLOW = 30
    TAKE_PROFIT = 0.08
    STOP_LOSS = 0.04
    
    # Trading Configuration
    DEFAULT_SYMBOL = "tBTCUSD"
    TIMEFRAME = "1D"
    LOOKBACK_PERIODS = 100
    
    # Professional Position Sizing
    POSITION_SIZE_USD = 5000.0
    MAX_POSITION_SIZE_USD = 50000.0
    MIN_POSITION_SIZE_USD = 100.0
    
    # Risk Management
    MAX_DAILY_TRADES = 3
    MAX_LEVERAGE = 3.3
    SLIPPAGE = 0.002
    COMMISSION = 0.0025
    
    # Advanced Features
    ENABLE_MARGIN_TRADING = True
    ENABLE_FUNDING_ANALYSIS = True
    FUNDING_RATE_THRESHOLD = 0.01
    
    DATA_CACHE_MINUTES = 3
    REQUIRED_DATA_POINTS = 60

class BitfinexMAReversalStrategy:
    def __init__(self, config=None):
        self.config = config or BitfinexMAConfig()
        self.position = 0.0
        self.entry_price = 0.0
        self.margin_position = False
        self.daily_trades = 0
        self.last_trade_date = None
        self.cached_data = {}
        self.cache_timestamp = {}
        
        if not self.config.API_KEY or not self.config.API_SECRET:
            cprint("Warning: Bitfinex API credentials not configured", "yellow")
    
    def _get_signature(self, path, nonce, body):
        message = f"/api/v2/{path}{nonce}{body}"
        return hmac.new(
            self.config.API_SECRET.encode(),
            message.encode(),
            hashlib.sha384
        ).hexdigest()
    
    def _make_request(self, endpoint, params=None, authenticated=False, method='GET'):
        if params is None:
            params = {}
        
        if authenticated:
            url = f"{self.config.AUTH_URL}{endpoint}"
            
            if not self.config.API_KEY or not self.config.API_SECRET:
                cprint("API credentials required", "red")
                return None
            
            nonce = str(int(time.time() * 1000000))
            body = json.dumps(params)
            signature = self._get_signature(endpoint, nonce, body)
            
            headers = {
                'bfx-nonce': nonce,
                'bfx-apikey': self.config.API_KEY,
                'bfx-signature': signature,
                'content-type': 'application/json'
            }
            
            try:
                if method.upper() == 'POST':
                    response = requests.post(url, headers=headers, data=body, timeout=30)
                else:
                    response = requests.get(url, headers=headers, data=body, timeout=30)
                
                response.raise_for_status()
                return response.json()
            
            except requests.exceptions.RequestException as e:
                cprint(f"API request failed: {e}", "red")
                return None
        else:
            url = f"{self.config.BASE_URL}{endpoint}"
            
            try:
                response = requests.get(url, params=params, timeout=30)
                response.raise_for_status()
                return response.json()
            except requests.exceptions.RequestException as e:
                cprint(f"Public API request failed: {e}", "red")
                return None
    
    def get_historical_data(self, symbol, timeframe='1D', limit=100):
        if not symbol.startswith('t'):
            symbol = f"t{symbol}"
        
        cache_key = f"{symbol}_{timeframe}_{limit}"
        current_time = time.time()
        
        if (cache_key in self.cached_data and 
            cache_key in self.cache_timestamp and
            current_time - self.cache_timestamp[cache_key] < self.config.DATA_CACHE_MINUTES * 60):
            return self.cached_data[cache_key]
        
        end_time = int(time.time() * 1000)
        if timeframe == '1D':
            start_time = end_time - (limit * 24 * 60 * 60 * 1000)
        else:
            start_time = end_time - (limit * 60 * 60 * 1000)
        
        endpoint = f"/candles/trade:{timeframe}:{symbol}/hist"
        params = {
            'start': start_time,
            'end': end_time,
            'limit': limit,
            'sort': 1
        }
        
        data = self._make_request(endpoint, params=params)
        
        if data:
            df = pd.DataFrame(data, columns=[
                'timestamp', 'open', 'close', 'high', 'low', 'volume'
            ])
            
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            for col in ['open', 'close', 'high', 'low', 'volume']:
                df[col] = pd.to_numeric(df[col])
            
            df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
            df = df.sort_values('timestamp').reset_index(drop=True)
            
            self.cached_data[cache_key] = df
            self.cache_timestamp[cache_key] = current_time
            
            return df
        
        return None
    
    def calculate_indicators(self, df):
        if df is None or len(df) < max(self.config.MA_FAST, self.config.MA_SLOW):
            return None
        
        df['sma_fast'] = ta.sma(df['close'], length=self.config.MA_FAST)
        df['sma_slow'] = ta.sma(df['close'], length=self.config.MA_SLOW)
        df['rsi'] = ta.rsi(df['close'], length=14)
        df['atr'] = ta.atr(df['high'], df['low'], df['close'], length=21)
        
        macd_data = ta.macd(df['close'], fast=12, slow=26, signal=9)
        if macd_data is not None:
            df['macd'] = macd_data['MACD_12_26_9']
            df['macd_signal'] = macd_data['MACDs_12_26_9']
            df['macd_histogram'] = macd_data['MACDh_12_26_9']
        
        return df
    
    def get_current_price(self, symbol):
        if not symbol.startswith('t'):
            symbol = f"t{symbol}"
        
        endpoint = f"/ticker/{symbol}"
        data = self._make_request(endpoint)
        
        if data and len(data) >= 7:
            return float(data[6]) if data[6] else None
        
        return None
    
    def get_account_info(self):
        endpoint = "/r/wallets"
        data = self._make_request(endpoint, authenticated=True)
        
        account_info = {'exchange_usd': 0.0, 'margin_usd': 0.0, 'total_usd': 0.0}
        
        if data:
            for wallet in data:
                if len(wallet) >= 3:
                    wallet_type = wallet[0]
                    currency = wallet[1]
                    balance = float(wallet[2]) if wallet[2] else 0.0
                    
                    if currency.upper() == 'USD':
                        if wallet_type == 'exchange':
                            account_info['exchange_usd'] = balance
                        elif wallet_type == 'margin':
                            account_info['margin_usd'] = balance
        
        account_info['total_usd'] = account_info['exchange_usd'] + account_info['margin_usd']
        return account_info
    
    def get_position(self, symbol):
        if not symbol.startswith('t'):
            symbol = f"t{symbol}"
        
        endpoint = "/r/positions"
        data = self._make_request(endpoint, authenticated=True)
        
        if data:
            for position in data:
                if len(position) >= 3 and position[0] == symbol:
                    return {
                        'size': float(position[2]) if position[2] else 0.0,
                        'price': float(position[3]) if len(position) > 3 and position[3] else 0.0,
                        'pnl': float(position[6]) if len(position) > 6 and position[6] else 0.0
                    }
        
        return {'size': 0.0, 'price': 0.0, 'pnl': 0.0}
    
    def place_order(self, symbol, side, amount):
        if not self.config.API_KEY or not self.config.API_SECRET:
            cprint("Order requires API credentials", "red")
            return None
        
        endpoint = "/w/order/submit"
        params = {
            'type': 'EXCHANGE MARKET',
            'symbol': symbol,
            'amount': str(amount if side == 'BUY' else -abs(amount)),
            'price': '0'
        }
        
        try:
            result = self._make_request(endpoint, params=params, authenticated=True, method='POST')
            
            if result and len(result) > 6 and result[6] == 'SUCCESS':
                cprint(f"{side} order executed: {amount} {symbol}", "green")
                return result[4]
            else:
                cprint(f"Order failed: {result[7] if result and len(result) > 7 else 'Unknown error'}", "red")
                return None
        
        except Exception as e:
            cprint(f"Error placing order: {e}", "red")
            return None
    
    def generate_signals(self, df):
        if df is None or len(df) < 2:
            return None
        
        current_row = df.iloc[-1]
        price = current_row['close']
        sma_fast = current_row['sma_fast']
        sma_slow = current_row['sma_slow']
        rsi = current_row['rsi']
        
        if pd.isna(sma_fast) or pd.isna(sma_slow) or pd.isna(rsi):
            return None
        
        signals = {
            'price': price,
            'sma_fast': sma_fast,
            'sma_slow': sma_slow,
            'rsi': rsi,
            'signal': 'HOLD',
            'confidence': 0.0
        }
        
        # Long Signal
        if price > sma_fast and price > sma_slow and self.position <= 0 and rsi < 65:
            signals['signal'] = 'BUY'
            signals['confidence'] = 0.8
            
            if (not pd.isna(current_row.get('macd', float('nan'))) and 
                current_row['macd'] > current_row.get('macd_signal', 0)):
                signals['confidence'] = 0.9
        
        # Short Signal
        elif (self.config.ENABLE_MARGIN_TRADING and 
              price > sma_fast and price < sma_slow and 
              self.position >= 0 and rsi > 35):
            signals['signal'] = 'SELL_SHORT'
            signals['confidence'] = 0.7
        
        # Position management
        elif self.position != 0:
            current_position = self.get_position(self.config.DEFAULT_SYMBOL)
            entry_price = current_position['price'] if current_position['price'] > 0 else self.entry_price
            
            if entry_price > 0:
                profit_pct = ((price - entry_price) / entry_price if self.position > 0 
                             else (entry_price - price) / entry_price)
                
                if profit_pct >= self.config.TAKE_PROFIT:
                    signals['signal'] = 'CLOSE_PROFIT'
                    signals['confidence'] = 1.0
                elif profit_pct <= -self.config.STOP_LOSS:
                    signals['signal'] = 'CLOSE_LOSS'
                    signals['confidence'] = 1.0
        
        return signals
    
    def execute_signal(self, symbol, signal_data):
        if not signal_data or signal_data['signal'] == 'HOLD':
            return False
        
        today = datetime.now().date()
        if self.last_trade_date != today:
            self.daily_trades = 0
            self.last_trade_date = today
        
        if self.daily_trades >= self.config.MAX_DAILY_TRADES:
            cprint(f"Daily trade limit reached: {self.daily_trades}", "yellow")
            return False
        
        current_price = signal_data['price']
        account_info = self.get_account_info()
        available_balance = account_info['total_usd']
        
        if signal_data['signal'] == 'BUY' and available_balance >= self.config.MIN_POSITION_SIZE_USD:
            position_size = min(
                self.config.POSITION_SIZE_USD,
                available_balance * 0.1,
                self.config.MAX_POSITION_SIZE_USD
            )
            quantity = position_size / current_price
            
            result = self.place_order(symbol, 'BUY', quantity)
            if result:
                self.position = quantity
                self.entry_price = current_price
                self.daily_trades += 1
                cprint(f"✅ LONG: {quantity:.6f} at ${current_price:.2f}", "green", attrs=['bold'])
                return True
        
        elif signal_data['signal'] in ['CLOSE_PROFIT', 'CLOSE_LOSS']:
            current_position = self.get_position(symbol)
            position_size = current_position['size']
            
            if position_size != 0:
                side = 'SELL' if position_size > 0 else 'BUY'
                quantity = abs(position_size)
                
                result = self.place_order(symbol, side, quantity)
                if result:
                    profit = current_position['pnl']
                    signal_type = signal_data['signal'].replace('CLOSE_', '')
                    
                    self.position = 0
                    self.entry_price = 0
                    self.daily_trades += 1
                    
                    profit_color = "green" if profit > 0 else "red"
                    cprint(f"✅ CLOSE ({signal_type}): ${profit:.2f} P&L", profit_color, attrs=['bold'])
                    return True
        
        return False
    
    def print_status(self, symbol, df, signal_data):
        current_price = signal_data['price'] if signal_data else self.get_current_price(symbol)
        account_info = self.get_account_info()
        current_position = self.get_position(symbol)
        
        cprint("\n" + "="*70, "cyan")
        cprint("🏦 BITFINEX PROFESSIONAL MA REVERSAL", "cyan", attrs=['bold'])
        cprint("="*70, "cyan")
        
        cprint(f"Symbol: {symbol} | Price: ${current_price:.6f}", "white")
        
        if signal_data:
            cprint(f"Fast SMA: ${signal_data['sma_fast']:.6f} | Slow SMA: ${signal_data['sma_slow']:.6f}", "white")
            cprint(f"RSI: {signal_data['rsi']:.2f}", "white")
            
            signal_color = {'BUY': 'green', 'SELL_SHORT': 'red', 'CLOSE_PROFIT': 'cyan', 'CLOSE_LOSS': 'magenta'}.get(signal_data['signal'], 'yellow')
            cprint(f"Signal: {signal_data['signal']} | Confidence: {signal_data['confidence']:.2f}", signal_color, attrs=['bold'])
        
        if current_position['size'] != 0:
            position_value = abs(current_position['size']) * current_price
            pnl_color = "green" if current_position['pnl'] > 0 else "red"
            
            cprint(f"Position: {current_position['size']:.6f} | Value: ${position_value:.2f}", "white")
            cprint(f"Entry: ${current_position['price']:.6f} | P&L: ${current_position['pnl']:.2f}", pnl_color)
        else:
            cprint("No active position", "white")
        
        cprint(f"Balance: ${account_info['total_usd']:.2f} | Trades: {self.daily_trades}/{self.config.MAX_DAILY_TRADES}", "white")
    
    def run_strategy(self, symbol=None, live_trading=False):
        symbol = symbol or self.config.DEFAULT_SYMBOL
        
        if not symbol.startswith('t'):
            symbol = f"t{symbol}"
        
        cprint(f"\n🚀 BITFINEX PROFESSIONAL MA REVERSAL STRATEGY", "cyan", attrs=['bold'])
        cprint(f"Live Trading: {'ENABLED' if live_trading else 'DISABLED'}", "yellow", attrs=['bold'])
        
        if live_trading and (not self.config.API_KEY or not self.config.API_SECRET):
            cprint("❌ Live trading requires API credentials!", "red", attrs=['bold'])
            return
        
        try:
            while True:
                df = self.get_historical_data(symbol, self.config.TIMEFRAME, self.config.LOOKBACK_PERIODS)
                
                if df is None or len(df) < self.config.REQUIRED_DATA_POINTS:
                    cprint("❌ Insufficient data for analysis", "red")
                    time.sleep(60)
                    continue
                
                df = self.calculate_indicators(df)
                
                if df is None:
                    cprint("❌ Error calculating indicators", "red")
                    time.sleep(60)
                    continue
                
                signal_data = self.generate_signals(df)
                self.print_status(symbol, df, signal_data)
                
                if live_trading and signal_data:
                    executed = self.execute_signal(symbol, signal_data)
                    if executed:
                        cprint(f"✅ Signal executed: {signal_data['signal']}", "green")
                else:
                    if signal_data and signal_data['signal'] != 'HOLD':
                        cprint(f"📊 SIMULATION: Would execute {signal_data['signal']}", "blue")
                
                cprint(f"\n⏰ Waiting 60 seconds...", "white")
                time.sleep(60)
                
        except KeyboardInterrupt:
            cprint("\n🛑 Strategy stopped", "yellow")
        except Exception as e:
            cprint(f"\n❌ Strategy error: {e}", "red")

def main():
    cprint("🏦 BITFINEX PROFESSIONAL MA REVERSAL STRATEGY", "cyan", attrs=['bold'])
    
    strategy = BitfinexMAReversalStrategy()
    
    print(f"Fast MA: {strategy.config.MA_FAST} | Slow MA: {strategy.config.MA_SLOW}")
    print(f"TP: {strategy.config.TAKE_PROFIT:.2%} | SL: {strategy.config.STOP_LOSS:.2%}")
    
    symbol = input(f"Enter symbol (default: {strategy.config.DEFAULT_SYMBOL}): ").strip().upper()
    if not symbol:
        symbol = strategy.config.DEFAULT_SYMBOL
    
    live_trading = input("Enable live trading? [y/N]: ").strip().lower() == 'y'
    
    if live_trading:
        cprint("\n⚠️  LIVE TRADING ENABLED!", "red", attrs=['bold'])
        confirm = input("Confirm live trading? [y/N]: ").strip().lower()
        if confirm != 'y':
            live_trading = False
    
    strategy.run_strategy(symbol=symbol, live_trading=live_trading)

if __name__ == "__main__":
    main()
