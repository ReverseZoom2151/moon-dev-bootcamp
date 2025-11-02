import pandas as pd
import pandas_ta as ta
import itertools
import sys, os
from backtesting import Backtest
from backtesting.lib import crossover, TrailingStrategy

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Day_12_Projects.binance_nice_funcs import create_exchange, get_ohlcv2, process_data_to_df

class EnhancedEmaStrategy(TrailingStrategy):
    ema_fast_period = 9
    ema_slow_period = 18
    ema_trend_period = 200
    stochrsi_rsi_len = 14
    atr_period = 14
    volume_ma_period = 20
    risk_per_trade = 0.02
    atr_sl_multiplier = 2.0
    partial_profit_factor = 1.5
    def init(self):
        close = pd.Series(self.data.Close)
        self.ema_fast = self.I(ta.ema, close, self.ema_fast_period)
        self.ema_slow = self.I(ta.ema, close, self.ema_slow_period)
        self.ema_trend = self.I(ta.ema, close, self.ema_trend_period)
        stochrsi = ta.stochrsi(close, length=self.stochrsi_rsi_len, rsi_length=self.stochrsi_rsi_len, k=3, d=3)
        self.stoch_k = self.I(lambda: stochrsi['STOCHRSIk_14_14_3_3'].values)
        self.stoch_d = self.I(lambda: stochrsi['STOCHRSId_14_14_3_3'].values)
        self.atr = self.I(ta.atr, pd.Series(self.data.High), pd.Series(self.data.Low), close, self.atr_period)
        self.volume_ma = self.I(ta.sma, pd.Series(self.data.Volume), self.volume_ma_period)
        self.obv = self.I(ta.obv, close, pd.Series(self.data.Volume))
        self.set_trailing_sl(self.atr_sl_multiplier)
        if not hasattr(self, 'position_entry_bar'):
            self.position_entry_bar = None
        if not hasattr(self, 'position_entry_price'):
            self.position_entry_price = None
    def next(self):
        price = self.data.Close[-1]
        atr_val = self.atr[-1]
        if len(self.data) < max(self.ema_trend_period, self.volume_ma_period) + 10:
            return
        equity = self.equity
        position_size = (equity * self.risk_per_trade) / (atr_val * self.atr_sl_multiplier) if atr_val > 0 else 0
        position_size = min(max(position_size, 0.01), 0.5)
        position_size = round(position_size, 2)
        bullish_trend = price > self.ema_trend[-1]
        bearish_trend = price < self.ema_trend[-1]
        volume_ok = self.data.Volume[-1] > self.volume_ma[-1]
        obv_trend = self.obv[-1] > self.obv[-2]
        ema_cross_up = crossover(self.ema_fast, self.ema_slow)
        ema_cross_down = crossover(self.ema_slow, self.ema_fast)
        stoch_bullish = crossover(self.stoch_k, self.stoch_d) and self.stoch_d[-1] < 40
        stoch_bearish = crossover(self.stoch_d, self.stoch_k) and self.stoch_d[-1] > 60
        if not self.position:
            if all([bullish_trend, ema_cross_up, stoch_bullish, volume_ok, obv_trend]):
                sl = price - atr_val * self.atr_sl_multiplier
                tp = price + atr_val * self.partial_profit_factor
                self.buy(size=position_size, sl=sl, tp=tp)
                self.position_entry_price = price
            elif all([bearish_trend, ema_cross_down, stoch_bearish, volume_ok, not obv_trend]):
                sl = price + atr_val * self.atr_sl_multiplier
                tp = price - atr_val * self.partial_profit_factor
                self.sell(size=position_size, sl=sl, tp=tp)
                self.position_entry_price = price
        elif self.position:
            if self.position.pl_pct >= 1.5:
                self.position.close(0.5)
            current_duration = len(self.data) - self.position_entry_bar if self.position_entry_bar else 0
            if current_duration > 5 and abs(self.position.pl_pct) < 0.5:
                self.position.close()
                self.position_entry_bar = None
        if not self.position and (ema_cross_up or ema_cross_down):
            self.position_entry_bar = len(self.data)

def fetch_data(symbol='SOLUSDT', timeframe='1h', lookback_days=365):
    exchange = create_exchange()
    df = pd.DataFrame(get_ohlcv2(symbol, timeframe, lookback_days))
    df = process_data_to_df(df)
    df.columns = ['Datetime', 'Open', 'High', 'Low', 'Close', 'Volume']
    df.set_index('Datetime', inplace=True)
    df.dropna(inplace=True)
    df = df.sort_index()
    return df

initial_cash = 10000
commission = 0.002
data_scale_factor = 1000.0
opt_params = {
    'ema_fast_period': [7, 9, 12],
    'ema_slow_period': [15, 18, 21],
    'ema_trend_period': [150, 200, 250],
    'stochrsi_rsi_len': [10, 14, 18],
    'atr_sl_multiplier': [1.5, 2.0, 2.5],
    'risk_per_trade': [0.01, 0.02, 0.03],
    'partial_profit_factor': [1.0, 1.5, 2.0],
    'volume_ma_period': [15, 20, 25]
}
timeframes = {
    '1h': '1h',
    '4h': '4h',
    '1d': '1d'
}
overall_best_return = -float('inf')
overall_best_params = None
overall_best_timeframe = None
all_results = {}
for tf_str, tf in timeframes.items():
    print(f"\n\n===== Processing Timeframe: {tf_str} =====")
    try:
        data = fetch_data(lookback_days=365)
        backtest = Backtest(data, EnhancedEmaStrategy, cash=initial_cash, commission=commission)
        print(f"\n--- Running basic backtest ({tf_str}) ---")
        stats_default = backtest.run()
        print(stats_default)
        all_results[tf_str] = {'default': stats_default, 'optimized': None, 'best_params': None}
        print(f"\n--- Running manual grid search ({tf_str}) ---")
        keys, values = zip(*opt_params.items())
        param_combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
        valid_combinations = 0
        combinations_to_test = []
        for params in param_combinations:
             if params['ema_fast_period'] < params['ema_slow_period']:
                 combinations_to_test.append(params)
                 valid_combinations += 1
        total_combinations_to_run = valid_combinations
        current = 0
        timeframe_best_return = -float('inf')
        timeframe_best_params = None
        timeframe_best_stats = None
        for params in combinations_to_test:
            current += 1
            print(f"Testing {tf_str} combination {current}/{total_combinations_to_run}: {params}")
            try:
                result = backtest.run(**params)
                returns = result['Return [%]']
                if returns > timeframe_best_return:
                    timeframe_best_return = returns
                    timeframe_best_params = params
                    timeframe_best_stats = result
                    if returns > overall_best_return:
                        overall_best_return = returns
                        overall_best_params = params
                        overall_best_timeframe = tf_str
            except Exception as e:
                print(f"Error with parameters on {tf_str}: {params}\nError details: {e}")
        print(f"\n--- Optimization Results for {tf_str} ---")
        if timeframe_best_params:
            print(f"Best Return: {timeframe_best_return:.2f}%")
            print(f"Best Params: {timeframe_best_params}")
            all_results[tf_str]['optimized'] = timeframe_best_stats
            all_results[tf_str]['best_params'] = timeframe_best_params
        else:
            print("No profitable combination found during optimization.")
    except Exception as e:
        print(f"An unexpected error occurred processing timeframe {tf_str}: {e}")
print("\n\n===== Overall Results Summary =====")
if overall_best_timeframe:
    print(f"Best Overall Return: {overall_best_return:.2f}%")
    print(f"Best Timeframe: {overall_best_timeframe}")
    print(f"Best Parameters: {overall_best_params}")
    print("\n--- Stats for Overall Best Run ---")
    print(all_results[overall_best_timeframe]['optimized'])
    print("\n--- Plotting Overall Best Result ---")
    try:
        best_tf_data = fetch_data(lookback_days=365)
        best_tf_backtest = Backtest(best_tf_data, EnhancedEmaStrategy, cash=initial_cash, commission=commission)
        best_tf_backtest.run(**overall_best_params)
        best_tf_backtest.plot(plot_volume=False, plot_equity=True)
    except Exception as e:
        print(f"Could not plot overall best result: {e}")
else:
    print("No profitable strategy found across any timeframe during optimization.") 