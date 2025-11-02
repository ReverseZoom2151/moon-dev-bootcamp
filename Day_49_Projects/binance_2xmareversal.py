"""
Binance MA Reversal Strategy
Professional Moving Average reversal trading strategy adapted for Binance exchange.
Based on dual MA crossover signals with risk management and position sizing.
"""

import pandas as pd
import pandas_ta as ta
import requests
import time
import hmac
import hashlib
import os
from urllib.parse import urlencode
from datetime import datetime
from termcolor import cprint
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# --- Configuration ---
class BinanceMAConfig:
    # API Configuration
    API_KEY = os.getenv('BINANCE_API_KEY')
    API_SECRET = os.getenv('BINANCE_API_SECRET')
    BASE_URL = "https://api.binance.com"
    
    # Strategy Parameters (optimized from original)
    MA_FAST = 25  # Fast moving average period
    MA_SLOW = 30  # Slow moving average period
    TAKE_PROFIT = 0.05  # 5% take profit
    STOP_LOSS = 0.05  # 5% stop loss
    
    # Trading Configuration
    DEFAULT_SYMBOL = "BTCUSDT"
    TIMEFRAME = "1d"  # Daily timeframe
    LOOKBACK_PERIODS = 100  # Periods for analysis
    
    # Position Sizing
    POSITION_SIZE_USD = 1000.0  # Default position size
    MAX_POSITION_SIZE_USD = 10000.0  # Maximum position size
    MIN_POSITION_SIZE_USD = 10.0  # Minimum position size (Binance minimum)
    
    # Risk Management
    MAX_DAILY_TRADES = 5  # Maximum trades per day
    SLIPPAGE = 0.001  # 0.1% slippage tolerance
    COMMISSION = 0.001  # 0.1% trading fee
    
    # Data Configuration
    DATA_CACHE_MINUTES = 5  # Cache market data for 5 minutes
    REQUIRED_DATA_POINTS = 60  # Minimum data points for analysis

class BinanceMAReversalStrategy:
    def __init__(self, config=None):
        """Initialize the Binance MA Reversal Strategy"""
        self.config = config or BinanceMAConfig()
        self.position = 0.0  # Current position (positive for long, negative for short)
        self.entry_price = 0.0  # Entry price for current position
        self.last_signal_time = None
        self.daily_trades = 0
        self.last_trade_date = None
        self.cached_data = {}
        self.cache_timestamp = {}
        
        # Validate API credentials
        if not self.config.API_KEY or not self.config.API_SECRET:
            cprint("Warning: Binance API credentials not configured", "yellow")
    
    def _get_signature(self, params):
        """Generate HMAC SHA256 signature for Binance API"""
        query_string = urlencode(params)
        return hmac.new(
            self.config.API_SECRET.encode('utf-8'),
            query_string.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
    
    def _make_request(self, endpoint, method='GET', params=None, signed=False):
        """Make request to Binance API"""
        if params is None:
            params = {}
        
        url = f"{self.config.BASE_URL}{endpoint}"
        
        if signed:
            params['timestamp'] = int(time.time() * 1000)
            params['signature'] = self._get_signature(params)
            
            headers = {
                'X-MBX-APIKEY': self.config.API_KEY
            }
        else:
            headers = {}
        
        try:
            if method == 'GET':
                response = requests.get(url, params=params, headers=headers)
            elif method == 'POST':
                response = requests.post(url, params=params, headers=headers)
            else:
                raise ValueError(f"Unsupported method: {method}")
            
            response.raise_for_status()
            return response.json()
        
        except requests.exceptions.RequestException as e:
            cprint(f"Binance API request failed: {e}", "red")
            return None
    
    def get_historical_data(self, symbol, interval='1d', limit=100):
        """Get historical kline data from Binance"""
        # Check cache
        cache_key = f"{symbol}_{interval}_{limit}"
        current_time = time.time()
        
        if (cache_key in self.cached_data and 
            cache_key in self.cache_timestamp and
            current_time - self.cache_timestamp[cache_key] < self.config.DATA_CACHE_MINUTES * 60):
            return self.cached_data[cache_key]
        
        endpoint = "/api/v3/klines"
        params = {
            'symbol': symbol,
            'interval': interval,
            'limit': limit
        }
        
        data = self._make_request(endpoint, params=params)
        
        if data:
            df = pd.DataFrame(data, columns=[
                'open_time', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_asset_volume', 'number_of_trades',
                'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
            ])
            
            # Convert to proper data types
            df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
            df['close_time'] = pd.to_datetime(df['close_time'], unit='ms')
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = pd.to_numeric(df[col])
            
            # Cache the data
            self.cached_data[cache_key] = df
            self.cache_timestamp[cache_key] = current_time
            
            return df
        
        return None
    
    def calculate_indicators(self, df):
        """Calculate moving averages and other technical indicators"""
        if df is None or len(df) < max(self.config.MA_FAST, self.config.MA_SLOW):
            return None
        
        # Calculate moving averages using pandas_ta
        df['sma_fast'] = ta.sma(df['close'], length=self.config.MA_FAST)
        df['sma_slow'] = ta.sma(df['close'], length=self.config.MA_SLOW)
        
        # Additional indicators for confirmation
        df['rsi'] = ta.rsi(df['close'], length=14)
        df['atr'] = ta.atr(df['high'], df['low'], df['close'], length=14)
        
        # MACD for trend confirmation
        macd_data = ta.macd(df['close'])
        if macd_data is not None:
            df['macd'] = macd_data['MACD_12_26_9']
            df['macd_signal'] = macd_data['MACDs_12_26_9']
            df['macd_histogram'] = macd_data['MACDh_12_26_9']
        
        return df
    
    def get_account_balance(self):
        """Get account balance"""
        endpoint = "/api/v3/account"
        data = self._make_request(endpoint, signed=True)
        
        if data and 'balances' in data:
            for balance in data['balances']:
                if balance['asset'] == 'USDT':
                    return float(balance['free'])
        
        return 0.0
    
    def get_current_price(self, symbol):
        """Get current price for symbol"""
        endpoint = "/api/v3/ticker/price"
        params = {'symbol': symbol}
        
        data = self._make_request(endpoint, params=params)
        
        if data and 'price' in data:
            return float(data['price'])
        
        return None
    
    def place_market_order(self, symbol, side, quantity):
        """Place a market order"""
        if not self.config.API_KEY or not self.config.API_SECRET:
            cprint("Cannot place order: API credentials not configured", "red")
            return None
        
        endpoint = "/api/v3/order"
        params = {
            'symbol': symbol,
            'side': side,  # 'BUY' or 'SELL'
            'type': 'MARKET',
            'quantity': f"{quantity:.6f}"
        }
        
        try:
            result = self._make_request(endpoint, method='POST', params=params, signed=True)
            
            if result:
                cprint(f"Order placed: {side} {quantity} {symbol}", "green")
                return result
            else:
                cprint(f"Failed to place order: {side} {quantity} {symbol}", "red")
                return None
        
        except Exception as e:
            cprint(f"Error placing order: {e}", "red")
            return None
    
    def generate_signals(self, df):
        """Generate trading signals based on MA reversal strategy"""
        if df is None or len(df) < 2:
            return None
        
        current_row = df.iloc[-1]
        previous_row = df.iloc[-2]
        
        price = current_row['close']
        sma_fast = current_row['sma_fast']
        sma_slow = current_row['sma_slow']
        prev_sma_fast = previous_row['sma_fast']
        prev_sma_slow = previous_row['sma_slow']
        rsi = current_row['rsi']
        
        # Check if we have valid indicator values
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
        
        # Long Signal: Price above both SMAs
        if price > sma_fast and price > sma_slow and self.position <= 0:
            # Additional confirmation: RSI not overbought and MACD bullish
            if rsi < 70:
                signals['signal'] = 'BUY'
                signals['confidence'] = 0.8
                
                # Higher confidence if MACD is also bullish
                if not pd.isna(current_row.get('macd', float('nan'))) and current_row['macd'] > current_row.get('macd_signal', 0):
                    signals['confidence'] = 0.9
        
        # Short Signal: Price above fast MA but below slow MA
        elif price > sma_fast and price < sma_slow and self.position >= 0:
            # Additional confirmation: RSI not oversold
            if rsi > 30:
                signals['signal'] = 'SELL_SHORT'
                signals['confidence'] = 0.7
        
        # Close Short: Price moves above slow MA
        elif self.position < 0 and price > sma_slow:
            signals['signal'] = 'CLOSE_SHORT'
            signals['confidence'] = 0.8
        
        # Close Long: Stop loss or take profit conditions
        elif self.position > 0:
            profit_pct = (price - self.entry_price) / self.entry_price
            
            if profit_pct >= self.config.TAKE_PROFIT:
                signals['signal'] = 'CLOSE_LONG_PROFIT'
                signals['confidence'] = 1.0
            elif profit_pct <= -self.config.STOP_LOSS:
                signals['signal'] = 'CLOSE_LONG_LOSS'
                signals['confidence'] = 1.0
        
        return signals
    
    def execute_signal(self, symbol, signal_data):
        """Execute trading signal"""
        if not signal_data or signal_data['signal'] == 'HOLD':
            return False
        
        # Check daily trade limit
        today = datetime.now().date()
        if self.last_trade_date != today:
            self.daily_trades = 0
            self.last_trade_date = today
        
        if self.daily_trades >= self.config.MAX_DAILY_TRADES:
            cprint(f"Daily trade limit reached: {self.daily_trades}", "yellow")
            return False
        
        current_price = signal_data['price']
        balance = self.get_account_balance()
        
        if signal_data['signal'] == 'BUY':
            if balance >= self.config.MIN_POSITION_SIZE_USD:
                position_size_usd = min(self.config.POSITION_SIZE_USD, balance * 0.1)  # Max 10% of balance
                quantity = position_size_usd / current_price
                
                result = self.place_market_order(symbol, 'BUY', quantity)
                if result:
                    self.position = quantity
                    self.entry_price = current_price
                    self.daily_trades += 1
                    cprint(f"Opened LONG position: {quantity:.6f} at ${current_price:.2f}", "green")
                    return True
        
        elif signal_data['signal'] == 'SELL_SHORT':
            # For spot trading, we can't short directly, so we'll close long positions instead
            if self.position > 0:
                result = self.place_market_order(symbol, 'SELL', self.position)
                if result:
                    profit = (current_price - self.entry_price) * self.position
                    cprint(f"Closed LONG position: {self.position:.6f} at ${current_price:.2f}, Profit: ${profit:.2f}", "cyan")
                    self.position = 0
                    self.entry_price = 0
                    self.daily_trades += 1
                    return True
        
        elif signal_data['signal'] in ['CLOSE_LONG_PROFIT', 'CLOSE_LONG_LOSS', 'CLOSE_SHORT']:
            if self.position != 0:
                side = 'SELL' if self.position > 0 else 'BUY'
                quantity = abs(self.position)
                
                result = self.place_market_order(symbol, side, quantity)
                if result:
                    profit = (current_price - self.entry_price) * self.position if self.position > 0 else (self.entry_price - current_price) * abs(self.position)
                    signal_type = "PROFIT" if "PROFIT" in signal_data['signal'] else "LOSS" if "LOSS" in signal_data['signal'] else "SIGNAL"
                    cprint(f"Closed position ({signal_type}): {quantity:.6f} at ${current_price:.2f}, Profit: ${profit:.2f}", "cyan")
                    self.position = 0
                    self.entry_price = 0
                    self.daily_trades += 1
                    return True
        
        return False
    
    def print_status(self, symbol, df, signal_data):
        """Print current strategy status"""
        current_price = signal_data['price'] if signal_data else self.get_current_price(symbol)
        
        cprint("\n" + "="*60, "cyan")
        cprint("🏆 BINANCE MA REVERSAL STRATEGY STATUS", "cyan", attrs=['bold'])
        cprint("="*60, "cyan")
        
        cprint(f"Symbol: {symbol}", "white")
        cprint(f"Current Price: ${current_price:.2f}", "white")
        
        if signal_data:
            cprint(f"Fast SMA ({self.config.MA_FAST}): ${signal_data['sma_fast']:.2f}", "white")
            cprint(f"Slow SMA ({self.config.MA_SLOW}): ${signal_data['sma_slow']:.2f}", "white")
            cprint(f"RSI: {signal_data['rsi']:.2f}", "white")
            
            signal_color = "green" if signal_data['signal'] in ['BUY'] else "red" if 'SELL' in signal_data['signal'] else "yellow"
            cprint(f"Signal: {signal_data['signal']} (Confidence: {signal_data['confidence']:.2f})", signal_color, attrs=['bold'])
        
        cprint(f"Position: {self.position:.6f} {symbol.replace('USDT', '')}", "white")
        
        if self.position != 0:
            unrealized_pnl = (current_price - self.entry_price) * self.position
            pnl_pct = (current_price - self.entry_price) / self.entry_price * 100
            pnl_color = "green" if unrealized_pnl > 0 else "red"
            cprint(f"Entry Price: ${self.entry_price:.2f}", "white")
            cprint(f"Unrealized P&L: ${unrealized_pnl:.2f} ({pnl_pct:.2f}%)", pnl_color)
        
        balance = self.get_account_balance()
        cprint(f"Available Balance: ${balance:.2f} USDT", "white")
        cprint(f"Daily Trades: {self.daily_trades}/{self.config.MAX_DAILY_TRADES}", "white")
        
    def run_strategy(self, symbol=None, live_trading=False):
        """Run the MA Reversal Strategy"""
        symbol = symbol or self.config.DEFAULT_SYMBOL
        
        cprint(f"\n🚀 Starting Binance MA Reversal Strategy for {symbol}", "cyan", attrs=['bold'])
        cprint(f"Live Trading: {'ENABLED' if live_trading else 'DISABLED'}", "yellow", attrs=['bold'])
        
        if live_trading and (not self.config.API_KEY or not self.config.API_SECRET):
            cprint("❌ Live trading requires API credentials!", "red", attrs=['bold'])
            return
        
        try:
            while True:
                # Get historical data
                df = self.get_historical_data(symbol, self.config.TIMEFRAME, self.config.LOOKBACK_PERIODS)
                
                if df is None or len(df) < self.config.REQUIRED_DATA_POINTS:
                    cprint("❌ Insufficient data for analysis", "red")
                    time.sleep(60)
                    continue
                
                # Calculate indicators
                df = self.calculate_indicators(df)
                
                if df is None:
                    cprint("❌ Error calculating indicators", "red")
                    time.sleep(60)
                    continue
                
                # Generate signals
                signal_data = self.generate_signals(df)
                
                # Print status
                self.print_status(symbol, df, signal_data)
                
                # Execute signal if live trading is enabled
                if live_trading and signal_data:
                    executed = self.execute_signal(symbol, signal_data)
                    if executed:
                        cprint(f"✅ Signal executed: {signal_data['signal']}", "green")
                else:
                    if signal_data and signal_data['signal'] != 'HOLD':
                        cprint(f"📊 SIMULATION: Would execute {signal_data['signal']}", "blue")
                
                # Wait before next iteration
                cprint(f"\n⏰ Waiting 60 seconds before next analysis...", "white")
                time.sleep(60)
                
        except KeyboardInterrupt:
            cprint("\n🛑 Strategy stopped by user", "yellow")
        except Exception as e:
            cprint(f"\n❌ Strategy error: {e}", "red")

def main():
    """Main execution function"""
    cprint("🏦 BINANCE MA REVERSAL STRATEGY", "cyan", attrs=['bold'])
    cprint("Professional Moving Average Reversal Trading System", "white")
    cprint("="*60, "cyan")
    
    # Initialize strategy
    strategy = BinanceMAReversalStrategy()
    
    # Get user preferences
    print("\nStrategy Configuration:")
    print(f"Fast MA Period: {strategy.config.MA_FAST}")
    print(f"Slow MA Period: {strategy.config.MA_SLOW}")
    print(f"Take Profit: {strategy.config.TAKE_PROFIT:.2%}")
    print(f"Stop Loss: {strategy.config.STOP_LOSS:.2%}")
    print(f"Position Size: ${strategy.config.POSITION_SIZE_USD}")
    
    # Get trading symbol
    symbol = input(f"\nEnter trading symbol (default: {strategy.config.DEFAULT_SYMBOL}): ").strip().upper()
    if not symbol:
        symbol = strategy.config.DEFAULT_SYMBOL
    
    # Ask about live trading
    live_trading_input = input("\nEnable live trading? [y/N]: ").strip().lower()
    live_trading = live_trading_input == 'y'
    
    if live_trading:
        cprint("\n⚠️  LIVE TRADING ENABLED - REAL MONEY AT RISK!", "red", attrs=['bold'])
        confirm = input("Are you sure you want to enable live trading? [y/N]: ").strip().lower()
        if confirm != 'y':
            live_trading = False
            cprint("Live trading disabled. Running in simulation mode.", "yellow")
    
    # Run the strategy
    strategy.run_strategy(symbol=symbol, live_trading=live_trading)

if __name__ == "__main__":
    main()
