import logging, sys, os, ccxt, time, pandas as pd
import pandas_ta as ta
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Day_4_Projects.key_file import binance_key, binance_secret

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', handlers=[logging.StreamHandler(), logging.FileHandler(f'binance_vwap_trading_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')])
logger = logging.getLogger('binance_vwap_trader')

class VWAPTrader:
    def __init__(self, symbol='BTCUSDT', timeframe='15m', limit=100, sma_period=20, target_profit=9, max_loss=-8, max_risk=1000):
        self.symbol = symbol
        self.timeframe = timeframe
        self.limit = limit
        self.sma_period = sma_period
        self.target_profit = target_profit
        self.max_loss = max_loss
        self.max_risk = max_risk
        self.order_params = {'timeInForce': 'GTX'}
        self.position_indices = {
            'BTCUSDT': 0,  # Adjust indices based on Binance symbols
            # Add other symbols as needed
        }
        try:
            self.exchange = ccxt.binance({'enableRateLimit': True, 'apiKey': binance_key, 'secret': binance_secret, 'options': {'defaultType': 'future'}})
            logger.info('Connected to Binance')
        except Exception as e:
            logger.error(f'Failed to connect: {e}')
            raise

    def get_market_data(self, symbol=None):
        if symbol is None:
            symbol = self.symbol
        try:
            ob = self.exchange.fetch_order_book(symbol)
            bid = ob['bids'][0][0] if ob['bids'] else 0
            ask = ob['asks'][0][0] if ob['asks'] else 0
            logger.info(f'Prices for {symbol} - Ask: {ask}, Bid: {bid}')
            return ask, bid
        except Exception as e:
            logger.error(f'Error fetching market data: {e}')
            return 0, 0

    def get_position_info(self, symbol=None):
        if symbol is None:
            symbol = self.symbol
        try:
            positions = self.exchange.fetch_positions([symbol])
            if not positions or positions[0]['contracts'] == 0:
                return None, False, 0, None
            pos = positions[0]
            return positions, True, pos['contracts'], pos['side'] == 'long'
        except Exception as e:
            logger.error(f'Error getting position: {e}')
            return None, False, 0, None

    def close_position(self, symbol=None):
        if symbol is None:
            symbol = self.symbol
        logger.info(f'Starting to close position for {symbol}')
        try:
            self.exchange.cancel_all_orders(symbol)
            pos_info = self.get_position_info(symbol)
            has_position = pos_info[1]
            attempts = 0
            max_attempts = 5
            while has_position and attempts < max_attempts:
                attempts += 1
                ask, bid = self.get_market_data(symbol)
                if ask == 0 or bid == 0:
                    time.sleep(5)
                    continue
                size = pos_info[2]
                is_long = pos_info[3]
                side = 'sell' if is_long else 'buy'
                px = ask if side == 'sell' else bid
                self.exchange.create_limit_order(symbol, side, size, px, self.order_params)
                time.sleep(30)
                pos_info = self.get_position_info(symbol)
                has_position = pos_info[1]
            return not has_position
        except Exception as e:
            logger.error(f'Error closing position: {e}')
            return False

    def check_pnl(self):
        pos_info = self.get_position_info()
        if not pos_info[1]: return False, False, 0, None
        ticker = self.exchange.fetch_ticker(self.symbol)
        current = ticker['last']
        entry = pos_info[0][0]['entryPrice']
        leverage = pos_info[0][0]['leverage']
        diff = (current - entry) if pos_info[3] else (entry - current)
        perc = (diff / entry * leverage) * 100
        if perc >= self.target_profit or perc <= self.max_loss:
            self.close_position()
            return True, True, pos_info[2], pos_info[3]
        return False, True, pos_info[2], pos_info[3]

    def check_max_risk(self):
        positions = self.exchange.fetch_positions()
        total_exposure = sum(float(p['notional']) for p in positions if p['notional'] > 0)
        if total_exposure > self.max_risk:
            for p in positions:
                if p['notional'] > 0:
                    self.close_position(p['symbol'])
            return False
        return True

    def calculate_sma(self):
        bars = self.exchange.fetch_ohlcv(self.symbol, self.timeframe, limit=self.limit)
        df = pd.DataFrame(bars, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df[f'sma{self.sma_period}_{self.timeframe}'] = df['close'].rolling(self.sma_period).mean()
        bid = self.get_market_data()[1]
        df.loc[df[f'sma{self.sma_period}_{self.timeframe}'] > bid, 'signal'] = 'SELL'
        df.loc[df[f'sma{self.sma_period}_{self.timeframe}'] < bid, 'signal'] = 'BUY'
        df['support'] = df['low'].rolling(self.limit).min()
        df['resistance'] = df['high'].rolling(self.limit).max()
        return df

    def calculate_rsi(self):
        bars = self.exchange.fetch_ohlcv(self.symbol, self.timeframe, limit=self.limit)
        df = pd.DataFrame(bars, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df['rsi'] = ta.rsi(df['close'])
        return df

    def calculate_vwap(self):
        bars = self.exchange.fetch_ohlcv(self.symbol, self.timeframe, limit=self.limit)
        df = pd.DataFrame(bars, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df['typical_price'] = (df['high'] + df['low'] + df['close']) / 3
        df['volume_x_typical'] = df['volume'] * df['typical_price']
        df['cum_volume'] = df['volume'].cumsum()
        df['cum_vol_x_price'] = df['volume_x_typical'].cumsum()
        df['vwap'] = df['cum_vol_x_price'] / df['cum_volume']
        bid = self.get_market_data()[1]
        last_vwap = df['vwap'].iloc[-1]
        df['vwap_signal'] = None
        if bid > last_vwap:
            df.loc[df.index[-1], 'vwap_signal'] = 'BUY'
        else:
            df.loc[df.index[-1], 'vwap_signal'] = 'SELL'
        return df

    def get_combined_signal(self):
        sma_df = self.calculate_sma()
        rsi_df = self.calculate_rsi()
        vwap_df = self.calculate_vwap()
        if sma_df.empty or rsi_df.empty or vwap_df.empty:
            return 'NEUTRAL'
        sma_signal = sma_df['signal'].iloc[-1] if 'signal' in sma_df.columns else 'NEUTRAL'
        rsi_value = rsi_df['rsi'].iloc[-1]
        rsi_signal = 'BUY' if rsi_value < 30 else 'SELL' if rsi_value > 70 else 'NEUTRAL'
        vwap_signal = vwap_df['vwap_signal'].iloc[-1] if 'vwap_signal' in vwap_df.columns else 'NEUTRAL'
        buy_count = sum(1 for signal in [sma_signal, rsi_signal, vwap_signal] if signal == 'BUY')
        sell_count = sum(1 for signal in [sma_signal, rsi_signal, vwap_signal] if signal == 'SELL')
        return 'BUY' if buy_count >= 2 else 'SELL' if sell_count >= 2 else 'NEUTRAL'

    def execute_trade(self, signal, size=1):
        ask, bid = self.get_market_data()
        if signal == 'BUY':
            self.exchange.create_limit_buy_order(self.symbol, size, bid, self.order_params)
        elif signal == 'SELL':
            self.exchange.create_limit_sell_order(self.symbol, size, ask, self.order_params)

    def run_trading_cycle(self, auto_trade=False, position_size=1):
        self.check_max_risk()
        has_position = self.get_position_info()[1]
        if has_position:
            self.check_pnl()
        signal = self.get_combined_signal()
        if auto_trade and signal in ['BUY', 'SELL']:
            self.execute_trade(signal, position_size)
        return {
            'timestamp': datetime.now(),
            'symbol': self.symbol,
            'has_position': has_position,
            'combined_signal': signal,
            'indicators': {
                'sma': self.calculate_sma().tail(1).to_dict('records')[0] if not self.calculate_sma().empty else {},
                'rsi': self.calculate_rsi().tail(1).to_dict('records')[0] if not self.calculate_rsi().empty else {},
                'vwap': self.calculate_vwap().tail(1).to_dict('records')[0] if not self.calculate_vwap().empty else {}
            },
            'market_data': {
                'ask': self.get_market_data()[0],
                'bid': self.get_market_data()[1]
            }
        }

def main():
    trader = VWAPTrader()
    auto_trade = False
    print('\n====== VWAP Trading System ======')
    print(f'Symbol: {trader.symbol}')
    print(f'Timeframe: {trader.timeframe}')
    print(f'Target profit: {trader.target_profit}%')
    print(f'Max loss: {trader.max_loss}%')
    print(f'Auto-trading: {"Enabled" if auto_trade else "Disabled"}')
    while True:
        print('\nOptions:')
        print('1. Check current position')
        print('2. Check indicators')
        print('3. Calculate VWAP')
        print('4. Run trading cycle')
        print('5. Close position')
        print('6. Toggle auto-trading')
        print('7. Exit')
        choice = input('\nEnter your choice (1-7): ')
        if choice == '1':
            positions, has_position, size, is_long = trader.get_position_info()
            if has_position:
                print(f'\nCurrent position: {"LONG" if is_long else "SHORT"} {size} {trader.symbol}')
                trader.check_pnl()
            else:
                print(f'\nNo open position for {trader.symbol}')
        elif choice == '2':
            print('\nCalculating indicators...')
            sma_df = trader.calculate_sma()
            rsi_df = trader.calculate_rsi()
            if not sma_df.empty and not rsi_df.empty:
                print(f'\nLatest SMA: {sma_df[f"sma{trader.sma_period}_{trader.timeframe}"].iloc[-1]:.2f}')
                print(f'Latest RSI: {rsi_df["rsi"].iloc[-1]:.2f}')
                print(f'SMA Signal: {sma_df["signal"].iloc[-1] if "signal" in sma_df.columns else "N/A"}')
        elif choice == '3':
            print('\nCalculating VWAP...')
            vwap_df = trader.calculate_vwap()
            if not vwap_df.empty:
                print(f'\nLatest VWAP: {vwap_df["vwap"].iloc[-1]:.2f}')
                print(f'VWAP Signal: {vwap_df["vwap_signal"].iloc[-1] if "vwap_signal" in vwap_df.columns else "N/A"}')
        elif choice == '4':
            print('\nRunning trading cycle...')
            results = trader.run_trading_cycle(auto_trade, 1)
            print(f'\nCombined signal: {results["combined_signal"]}')
            if auto_trade and results["combined_signal"] in ["BUY", "SELL"]:
                print(f'Auto-executed {results["combined_signal"]} trade')
        elif choice == '5':
            print('\nClosing position...')
            success = trader.close_position()
            if success:
                print('Position closed successfully')
            else:
                print('Failed to close position')
        elif choice == '6':
            auto_trade = not auto_trade
            print(f'\nAuto-trading: {"Enabled" if auto_trade else "Disabled"}')
        elif choice == '7':
            print('\nExiting VWAP Trading System...')
            break
        else:
            print('\nInvalid choice, please try again')
        input('\nPress Enter to continue...')

if __name__ == '__main__':
    main() 