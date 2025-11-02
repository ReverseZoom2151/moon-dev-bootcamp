import numpy as np
import pandas as pd
import warnings
import os
import sys
from backtesting import Backtest, Strategy

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

warnings.filterwarnings('ignore')

class LiquidationStrategy(Strategy):
    short_liquidation_thresh = 100000
    entry_time_window_mins = 5
    long_liquidation_closure_thresh = 50000
    exit_time_window_mins = 5
    take_profit_pct = 0.02
    stop_loss_pct = 0.01
    def init(self):
        self.short_liquidations = self.data.short_liquidations
        self.long_liquidations = self.data.long_liquidations
        self.trade_entry_bar = -1
    def next(self):
        current_bar_idx = len(self.data.Close) - 1
        if not self.position:
            entry_start_time = self.data.index[-1] - pd.Timedelta(minutes=self.entry_time_window_mins)
            entry_start_idx = np.searchsorted(self.data.index, entry_start_time, side='left')
            recent_short_liquidations = self.short_liquidations[entry_start_idx:].sum()
            if recent_short_liquidations >= self.short_liquidation_thresh:
                sl_price = self.data.Close[-1] * (1 + self.stop_loss_pct)
                tp_price = self.data.Close[-1] * (1 - self.take_profit_pct)
                self.sell(sl=sl_price, tp=tp_price)
                self.trade_entry_bar = current_bar_idx
        elif self.position.is_short:
            exit_start_time = self.data.index[-1] - pd.Timedelta(minutes=self.exit_time_window_mins)
            exit_start_idx = np.searchsorted(self.data.index, exit_start_time, side='left')
            recent_long_liquidations = self.long_liquidations[exit_start_idx:].sum()
            if recent_long_liquidations >= self.long_liquidation_closure_thresh:
                self.position.close(reason='L Liq Threshold Hit')

class DelayedLiquidationStrategy(LiquidationStrategy):
    delay_minutes = 0
    def next(self):
        current_bar_idx = len(self.data.Close) - 1
        if self.position or self.trade_entry_bar == current_bar_idx:
            super().next()
            return
        potential_entry_signal = False
        last_potential_entry_bar = -1
        for lookback in range(1, self.delay_minutes + 1):
            if current_bar_idx - lookback < 0: break
            prev_time = self.data.index[current_bar_idx - lookback]
            entry_start_time = prev_time - pd.Timedelta(minutes=self.entry_time_window_mins)
            entry_start_idx = np.searchsorted(self.data.index, entry_start_time, side='left')
            past_short_liquidations = self.short_liquidations[entry_start_idx : current_bar_idx - lookback + 1].sum()
            if past_short_liquidations >= self.short_liquidation_thresh:
                potential_entry_signal = True
                last_potential_entry_bar = current_bar_idx - lookback
                break
        if potential_entry_signal and self.trade_entry_bar < last_potential_entry_bar:
            bars_since_signal = current_bar_idx - last_potential_entry_bar
            if bars_since_signal >= self.delay_minutes:
                if not self.position:
                    sl_price = self.data.Close[-1] * (1 + self.stop_loss_pct)
                    tp_price = self.data.Close[-1] * (1 - self.take_profit_pct)
                    self.sell(sl=sl_price, tp=tp_price)
                    self.trade_entry_bar = current_bar_idx
                    return
        super().next()

def run_alpha_decay_test(data, strategy_class, delays):
    for delay in delays:
        print(f'\n--- Running backtest with {delay}-minute delay ---')
        strategy_class.delay_minutes = delay
        bt = Backtest(data, strategy_class, cash=100000, commission=0.002)
        try:
            stats = bt.run()
            print(stats)
        except Exception as e:
            print(f'Error during backtest run with delay {delay}: {e}')
        print('\n' + '='*50)

def main():
    symbol = 'SOLUSDT'
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir_relative_to_parent = os.path.join('..', 'Day_21_Projects', 'output_data')
    data_path = os.path.abspath(os.path.join(script_dir, data_dir_relative_to_parent, f'{symbol.replace('USDT', '')}_liq_data.csv'))
    print(f'Loading data for {symbol} from: {data_path}')
    try:
        data = pd.read_csv(data_path, parse_dates=['datetime'], index_col='datetime')
    except FileNotFoundError:
        print(f'Error: Data file not found at {data_path}')
        return
    required_cols = ['LIQ_SIDE', 'price', 'usd_size']
    if not all(col in data.columns for col in required_cols):
        print(f'Error: Input CSV {data_path} missing required columns {required_cols}.')
        return
    data = data[required_cols].copy()
    data['long_liquidations'] = np.where(data['LIQ_SIDE'] == 'L LIQ', data['usd_size'], 0)
    data['short_liquidations'] = np.where(data['LIQ_SIDE'] == 'S LIQ', data['usd_size'], 0)
    agg_funcs = {
        'price': 'mean',
        'long_liquidations': 'sum',
        'short_liquidations': 'sum'
    }
    data_resampled = data.resample('min').agg(agg_funcs)
    data_resampled['Open'] = data_resampled['price']
    data_resampled['High'] = data_resampled['price']
    data_resampled['Low'] = data_resampled['price']
    data_resampled['Close'] = data_resampled['price']
    price_columns = ['price', 'Open', 'High', 'Low', 'Close']
    data_resampled[price_columns] = data_resampled[price_columns].ffill()
    volume_columns = ['long_liquidations', 'short_liquidations']
    data_resampled[volume_columns] = data_resampled[volume_columns].fillna(0)
    data_resampled.sort_index(inplace=True)
    data_resampled.dropna(subset=['Close'], inplace=True)
    if data_resampled.empty:
        print('Error: No data available for backtesting after processing.')
        return
    delays = [0, 1, 2, 5, 10, 15, 30, 60]
    run_alpha_decay_test(data_resampled, DelayedLiquidationStrategy, delays)

if __name__ == '__main__':
    main() 