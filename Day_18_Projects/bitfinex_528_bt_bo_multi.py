import numpy as np, pandas as pd
import os, sys
from backtesting.test import SMA
from datetime import datetime
from backtesting import Backtest, Strategy

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Day_12_Projects.bitfinex_nice_funcs import create_exchange

def BBANDS(data, period=20, std_dev=2):
    middle_band = SMA(data, period)
    std = np.std(data[-period:] if len(data) >= period else data)
    upper_band = middle_band + std_dev * std
    lower_band = middle_band - std_dev * std
    return upper_band, middle_band, lower_band

def fetch_data(symbol='BTC:USDT', period_daily=None, period_hourly=None):
    if not os.path.exists('data'):
        os.makedirs('data')
    daily_file = f'data/{symbol.replace(":", "_")}_daily.csv'
    hourly_file = f'data/{symbol.replace(":", "_")}_hourly.csv'
    print(f"Downloading fresh data for {symbol}...")
    end_date = datetime.now()
    start_date_daily = end_date - pd.Timedelta(days=365)
    start_date_hourly = end_date - pd.Timedelta(days=60)
    end_str = end_date.strftime('%Y-%m-%d')
    start_daily_str = start_date_daily.strftime('%Y-%m-%d')
    start_hourly_str = start_date_hourly.strftime('%Y-%m-%d')
    print(f"Daily data range: {start_daily_str} to {end_str}")
    print(f"Hourly data range: {start_hourly_str} to {end_str}")
    exchange = create_exchange()
    since_daily = int(start_date_daily.timestamp() * 1000)
    since_hourly = int(start_date_hourly.timestamp() * 1000)
    daily_data = pd.DataFrame(exchange.fetch_ohlcv(symbol, timeframe='1d', since=since_daily))
    hourly_data = pd.DataFrame(exchange.fetch_ohlcv(symbol, timeframe='1h', since=since_hourly))
    for data in [daily_data, hourly_data]:
        data.columns = ['timestamp', 'Open', 'High', 'Low', 'Close', 'Volume']
        data['timestamp'] = pd.to_datetime(data['timestamp'], unit='ms')
        data.set_index('timestamp', inplace=True)
        data.dropna(inplace=True)
    if daily_data.empty or hourly_data.empty:
        raise ValueError(f"Could not download data for {symbol}")
    daily_data['resis'] = daily_data['High'].rolling(window=min(20, len(daily_data))).max()
    daily_data['resis'] = daily_data['resis'].fillna(daily_data['High'].expanding().max())
    print(f"\nDaily data shape: {daily_data.shape}")
    print(f"Hourly data shape: {hourly_data.shape}")
    return daily_data, hourly_data

def main():
    symbol = 'BTC:USDT'
    try:
        daily_data, hourly_data = fetch_data(symbol)
        daily_start, daily_end = daily_data.index.min().date(), daily_data.index.max().date()
        hourly_start, hourly_end = hourly_data.index.min().date(), hourly_data.index.max().date()
        if hourly_end < daily_start or hourly_start > daily_end:
            print("\nWARNING: No overlap between daily and hourly data.")
        if daily_start > hourly_start or daily_end < hourly_end:
            print("\nFiltering hourly data to match available daily data range...")
            hourly_data = hourly_data.loc[hourly_data.index.date >= daily_start]
            hourly_data = hourly_data.loc[hourly_data.index.date <= daily_end]
            print(f"Filtered hourly data shape: {hourly_data.shape}")
        if len(hourly_data) < 100:
            raise ValueError("Not enough hourly data points after filtering")
        class ImprovedBreakoutStrategy(Strategy):
            atr_period = 14
            tp_percent = 5
            sl_atr_mult = 1.5
            volume_factor = 1.5
            volume_period = 20
            bb_period = 20
            bb_std = 2
            atr_period_range = [10, 14, 20]
            tp_percent_range = [3, 5, 8]
            sl_atr_mult_range = [1.0, 2.0]
            def init(self):
                self.daily_resistance = daily_data['resis']
                self.atr = self.I(SMA, abs(self.data.High - self.data.Low), self.atr_period)
                self.volume_ma = self.I(SMA, self.data.Volume, self.volume_period)
                def bb_upper(data):
                    return BBANDS(data, self.bb_period, self.bb_std)[0]
                def bb_mid(data):
                    return BBANDS(data, self.bb_period, self.bb_std)[1]
                def bb_lower(data):
                    return BBANDS(data, self.bb_period, self.bb_std)[2]
                self.bb_upper = self.I(bb_upper, self.data.Close)
                self.bb_mid = self.I(bb_mid, self.data.Close)
                self.bb_lower = self.I(bb_lower, self.data.Close)
                self.trade_count = 0
                self.win_count = 0
            def next(self):
                current_time = self.data.index[-1]
                current_date = current_time.date() if not hasattr(current_time, 'tz') else current_time.tz_localize(None).date()
                try:
                    daily_idx_exact = daily_data.index[daily_data.index.date == current_date]
                    if len(daily_idx_exact) > 0:
                        daily_idx = daily_idx_exact[0]
                        daily_resistance = self.daily_resistance.loc[daily_idx]
                    else:
                        prev_days = daily_data.index[daily_data.index.date < current_date]
                        if len(prev_days) > 0:
                            daily_idx = prev_days[-1]
                            daily_resistance = self.daily_resistance.loc[daily_idx]
                        else:
                            daily_idx = daily_data.index[0]
                            daily_resistance = self.daily_resistance.loc[daily_idx]
                except:
                    daily_resistance = self.daily_resistance.iloc[-1]
                current_close = self.data.Close[-1]
                current_volume = self.data.Volume[-1]
                if (np.isnan(daily_resistance) or np.isnan(current_close) or np.isnan(self.bb_upper[-1]) or np.isnan(self.volume_ma[-1]) or not np.isfinite(daily_resistance) or not np.isfinite(current_close) or not np.isfinite(self.bb_upper[-1]) or not np.isfinite(self.volume_ma[-1])):
                    return
                breakout_conditions = (
                    current_close > daily_resistance and
                    current_close > self.bb_upper[-1] and
                    current_volume > self.volume_ma[-1] * self.volume_factor
                )
                if not self.position and breakout_conditions:
                    entry_price = current_close
                    stop_loss = max(0, entry_price - self.atr[-1] * self.sl_atr_mult)
                    take_profit = entry_price * (1 + self.tp_percent / 100)
                    risk_amount = 0.02 * self._broker.equity
                    price_risk = entry_price - stop_loss
                    if price_risk > 0:
                        size = risk_amount / price_risk
                        self.buy(size=size, sl=stop_loss, tp=take_profit)
                        self.trade_count += 1
        bt = Backtest(hourly_data, ImprovedBreakoutStrategy, cash=100000, commission=.002)
        optimize = input("Do you want to optimize parameters? (y/n, default: n): ").lower() == 'y'
        if optimize:
            print("Optimizing strategy parameters...")
            try:
                best_params = {
                    'atr_period': ImprovedBreakoutStrategy.atr_period,
                    'tp_percent': ImprovedBreakoutStrategy.tp_percent,
                    'sl_atr_mult': ImprovedBreakoutStrategy.sl_atr_mult
                }
                best_sharpe = -np.inf
                for atr in ImprovedBreakoutStrategy.atr_period_range:
                    for tp in ImprovedBreakoutStrategy.tp_percent_range:
                        for sl in ImprovedBreakoutStrategy.sl_atr_mult_range:
                            print(f"\nTesting params: ATR={atr}, TP%={tp}, SL={sl}")
                            try:
                                temp_stats = bt.run(atr_period=atr, tp_percent=tp, sl_atr_mult=sl)
                                if not np.isnan(temp_stats['Sharpe Ratio']):
                                    if temp_stats['Sharpe Ratio'] > best_sharpe:
                                        best_sharpe = temp_stats['Sharpe Ratio']
                                        best_params = {'atr_period': atr, 'tp_percent': tp, 'sl_atr_mult': sl}
                            except Exception as e:
                                print(f"Skipped parameters {atr}/{tp}/{sl} due to error: {str(e)}")
                if best_sharpe == -np.inf:
                    print("\nWarning: No valid parameters found during optimization. Using defaults.")
                    stats = bt.run()
                else:
                    print("\nBest parameters found:", best_params)
                    class OptimizedStrategy(ImprovedBreakoutStrategy):
                        atr_period = best_params['atr_period']
                        tp_percent = best_params['tp_percent']
                        sl_atr_mult = best_params['sl_atr_mult']
                    stats = bt.run(strategy=OptimizedStrategy)
            except ValueError as e:
                if "need at most 63 handles" in str(e):
                    print("Optimization failed due to Windows handle limit error. Running backtest without optimization.")
                    stats = None
                else:
                    raise
        else:
            stats = bt.run()
        bt.plot(filename=f"backtest_results_{symbol.replace('-', '_')}.html")
        print("\nBacktest Results:")
        print(stats)
        print("\nBest parameters:")
        print(stats._strategy)
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    import multiprocessing
    multiprocessing.freeze_support()
    main() 