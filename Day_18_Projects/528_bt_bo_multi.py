import numpy as np, pandas as pd, yfinance as yf, os
from backtesting.test import SMA
from datetime import datetime
from backtesting import Backtest, Strategy

# Custom Bollinger Bands implementation
def BBANDS(data, period=20, std_dev=2):
    """Calculate Bollinger Bands for the data
    
    Returns upper band, middle band, lower band
    """
    middle_band = SMA(data, period)
    std = np.std(data[-period:] if len(data) >= period else data)
    upper_band = middle_band + std_dev * std
    lower_band = middle_band - std_dev * std
    return upper_band, middle_band, lower_band

# Function to fetch data from Yahoo Finance
def fetch_data(symbol='BTC-USD', period_daily=None, period_hourly=None):
    # Create data directory if it doesn't exist
    if not os.path.exists('data'):
        os.makedirs('data')
        print(f"Created data directory at: {os.path.abspath('data')}")
    
    # File paths
    daily_file = f'data/{symbol.replace("-", "_")}_daily.csv'
    hourly_file = f'data/{symbol.replace("-", "_")}_hourly.csv'
    
    print(f"Daily file path: {os.path.abspath(daily_file)}")
    print(f"Hourly file path: {os.path.abspath(hourly_file)}")
    
    # Always download fresh data 
    print(f"Downloading fresh data for {symbol}...")
    
    # Use explicit date ranges instead of period
    end_date = datetime.now()
    start_date_daily = end_date - pd.Timedelta(days=365)  # 1 year of daily data
    start_date_hourly = end_date - pd.Timedelta(days=60)  # 60 days of hourly data
    
    # Format dates as strings
    end_str = end_date.strftime('%Y-%m-%d')
    start_daily_str = start_date_daily.strftime('%Y-%m-%d')
    start_hourly_str = start_date_hourly.strftime('%Y-%m-%d')
    
    print(f"Daily data range: {start_daily_str} to {end_str}")
    print(f"Hourly data range: {start_hourly_str} to {end_str}")
    
    try:
        # Get daily data with explicit date range
        daily_data = yf.download(
            symbol, 
            start=start_daily_str, 
            end=end_str, 
            interval='1d',
            progress=False,
            show_errors=False
        )
        
        # Get hourly data with explicit date range
        hourly_data = yf.download(
            symbol, 
            start=start_hourly_str, 
            end=end_str, 
            interval='1h',
            progress=False,
            show_errors=False
        )
    except Exception as e:
        print(f"Error downloading data: {e}")
        # Try alternative method if the first method fails
        print("Trying alternative download method...")
        try:
            daily_data = yf.download(symbol, period="1y", interval='1d', progress=False)
            hourly_data = yf.download(symbol, period="60d", interval='1h', progress=False)
        except Exception as e2:
            print(f"Alternative download also failed: {e2}")
            raise ValueError(f"Could not download data for {symbol}. Try a different symbol.")
    
    # Check if there's a multi-index
    if isinstance(daily_data.columns, pd.MultiIndex):
        print("Detected multi-index columns, flattening...")
        daily_data.columns = [col[0] for col in daily_data.columns]
    if isinstance(hourly_data.columns, pd.MultiIndex):
        hourly_data.columns = [col[0] for col in hourly_data.columns]
    
    # Check for empty data
    if daily_data.empty:
        print(f"Error: No daily data found for {symbol}")
        raise ValueError(f"Could not find daily data for symbol '{symbol}'. Please check if the symbol is valid on Yahoo Finance.")
    
    if hourly_data.empty:
        print(f"Error: No hourly data found for {symbol}")
        raise ValueError(f"Could not find hourly data for symbol '{symbol}'. Please check if the symbol is valid on Yahoo Finance.")
    
    # Convert column names to standard format
    cols_map = {
        'open': 'Open',
        'high': 'High',
        'low': 'Low',
        'close': 'Close',
        'volume': 'Volume',
        'adj close': 'Adj Close',
    }
    
    daily_data.rename(columns={col: cols_map.get(col.lower(), col) for col in daily_data.columns}, inplace=True)
    hourly_data.rename(columns={col: cols_map.get(col.lower(), col) for col in hourly_data.columns}, inplace=True)
    
    # Clean any missing values
    daily_data.dropna(inplace=True)
    hourly_data.dropna(inplace=True)
    
    # Print info about the data
    print(f"\nDaily data shape: {daily_data.shape}")
    print(f"Daily data columns: {daily_data.columns.tolist()}")
    print(f"Hourly data shape: {hourly_data.shape}")
    print(f"Hourly data columns: {hourly_data.columns.tolist()}")
    
    # Make sure we have enough data
    if len(daily_data) < 20:
        print(f"Warning: Only {len(daily_data)} days of daily data available.")
        if len(daily_data) < 5:
            raise ValueError(f"Not enough daily data points for symbol '{symbol}'. Try a more liquid asset.")
    
    if len(hourly_data) < 20:
        print(f"Warning: Only {len(hourly_data)} hours of hourly data available.")
        if len(hourly_data) < 5:
            raise ValueError(f"Not enough hourly data points for symbol '{symbol}'. Try a more liquid asset.")
    
    # Calculate resistance levels for daily data
    daily_data['resis'] = daily_data['High'].rolling(window=min(20, len(daily_data))).max()
    
    # Fill NaN values in resistance with the max High value up to that point
    daily_data['resis'] = daily_data['resis'].fillna(daily_data['High'].expanding().max())
    
    # Print some samples to verify data
    print("\nSample of daily data:")
    print(daily_data.head(3))
    print("\nSample of hourly data:")
    print(hourly_data.head(3))
    
    return daily_data, hourly_data

def main():
    # Get symbol from user or use default
    print("\nSample valid symbols:")
    print("  Cryptocurrencies: BTC-USD, ETH-USD, SOL-USD")
    print("  Stocks: AAPL, MSFT, GOOGL")
    print("  ETFs: SPY, QQQ, GLD")
    print("  Forex: EURUSD=X, GBPUSD=X\n")
    symbol = input("Enter symbol to backtest (default: BTC-USD): ") or "BTC-USD"

    try:
        # Fetch data from Yahoo Finance
        daily_data, hourly_data = fetch_data(symbol)
        
        # Data should already be properly formatted with datetime index
        print(f"\nDaily data shape: {daily_data.shape}")
        print(f"Hourly data shape: {hourly_data.shape}")
        print(f"Daily data date range: {daily_data.index.min()} to {daily_data.index.max()}")
        print(f"Hourly data date range: {hourly_data.index.min()} to {hourly_data.index.max()}")
        
        # Check if there's any overlap between the daily and hourly data timeframes
        daily_start, daily_end = daily_data.index.min(), daily_data.index.max()
        hourly_start, hourly_end = hourly_data.index.min(), hourly_data.index.max()
        
        # Convert to dates for proper comparison if timezone-aware
        if hasattr(hourly_start, 'tz'):
            hourly_start = hourly_start.tz_localize(None).date()
            hourly_end = hourly_end.tz_localize(None).date()
        else:
            hourly_start = hourly_start.date()
            hourly_end = hourly_end.date()
        
        if hasattr(daily_start, 'tz'):
            daily_start = daily_start.tz_localize(None).date()
            daily_end = daily_end.tz_localize(None).date()
        else:
            daily_start = daily_start.date()
            daily_end = daily_end.date()
        
        # Ensure there's an overlap period for the strategy to work with
        if hourly_end < daily_start or hourly_start > daily_end:
            print("\nWARNING: No overlap between daily and hourly data.")
            print("The strategy may not work properly as it relies on daily resistance levels.")
        
        # Filter hourly data to only include dates that have daily data
        if daily_start > hourly_start or daily_end < hourly_end:
            print("\nFiltering hourly data to match available daily data range...")
            hourly_data = hourly_data.loc[hourly_data.index.date >= daily_start]
            hourly_data = hourly_data.loc[hourly_data.index.date <= daily_end]
            print(f"Filtered hourly data shape: {hourly_data.shape}")
            print(f"Filtered hourly data date range: {hourly_data.index.min()} to {hourly_data.index.max()}")
        
        # If we don't have enough hourly data left after filtering, raise an error
        if len(hourly_data) < 100:
            raise ValueError("Not enough hourly data points after filtering to match daily data range.")
        
        class ImprovedBreakoutStrategy(Strategy):
            # Strategy parameters
            atr_period = 14  # Period for ATR calculation
            tp_percent = 5   # Take profit percentage (reduced from 20)
            sl_atr_mult = 1.5  # Stop loss as multiple of ATR
            volume_factor = 1.5  # Volume must be higher than average
            volume_period = 20  # Period for volume average
            bb_period = 20  # Period for Bollinger Bands
            bb_std = 2  # Standard deviation for Bollinger Bands
            
            # Parameter optimization ranges - REDUCED TO AVOID WINDOWS HANDLE LIMIT
            atr_period_range = [10, 14, 20]  # Only 3 values
            tp_percent_range = [3, 5, 8]     # Only 3 values 
            sl_atr_mult_range = [1.0, 2.0]   # Only 2 values
            
            def init(self):
                # Get the daily resistance from our pre-calculated data
                self.daily_resistance = daily_data['resis']
                
                # Calculate indicators
                self.atr = self.I(SMA, abs(self.data.High - self.data.Low), self.atr_period)
                self.volume_ma = self.I(SMA, self.data.Volume, self.volume_period)
                
                # Calculate Bollinger Bands manually
                def bb_upper(data):
                    return BBANDS(data, self.bb_period, self.bb_std)[0]
                
                def bb_mid(data):
                    return BBANDS(data, self.bb_period, self.bb_std)[1]
                
                def bb_lower(data):
                    return BBANDS(data, self.bb_period, self.bb_std)[2]
                
                # Register the custom indicators
                self.bb_upper = self.I(bb_upper, self.data.Close)
                self.bb_mid = self.I(bb_mid, self.data.Close)
                self.bb_lower = self.I(bb_lower, self.data.Close)
                
                # For tracking positions and drawdowns
                self.trade_count = 0
                self.win_count = 0
            
            def next(self):
                # Get current data
                current_time = self.data.index[-1]
                
                # Convert current time to date for comparison
                if hasattr(current_time, 'tz'):
                    current_date = current_time.tz_localize(None).date()
                else:
                    current_date = current_time.date()
                
                # Look for resistance levels from daily data
                try:
                    # Try to find the exact day
                    daily_idx_exact = daily_data.index[daily_data.index.date == current_date]
                    if len(daily_idx_exact) > 0:
                        daily_idx = daily_idx_exact[0]
                        daily_resistance = self.daily_resistance.loc[daily_idx]
                    else:
                        # If not found, look for the closest previous day
                        prev_days = daily_data.index[daily_data.index.date < current_date]
                        if len(prev_days) > 0:
                            daily_idx = prev_days[-1]  # Get the most recent
                            daily_resistance = self.daily_resistance.loc[daily_idx]
                        else:
                            # No previous day data available, use first available day
                            daily_idx = daily_data.index[0]
                            daily_resistance = self.daily_resistance.loc[daily_idx]
                except (KeyError, IndexError):
                    # Fallback: use the latest resistance level
                    daily_resistance = self.daily_resistance.iloc[-1]
                
                # Get current price and volume data
                current_close = self.data.Close[-1]
                current_volume = self.data.Volume[-1]
                
                # Skip if any indicator is NaN or invalid
                if (np.isnan(daily_resistance) or np.isnan(current_close) or 
                    np.isnan(self.bb_upper[-1]) or np.isnan(self.volume_ma[-1]) or
                    not np.isfinite(daily_resistance) or not np.isfinite(current_close) or
                    not np.isfinite(self.bb_upper[-1]) or not np.isfinite(self.volume_ma[-1])):
                    return
                
                # Only print during regular run (not optimization)
                if not hasattr(self._broker, '_cash_adjustments') or not self._broker._cash_adjustments:
                    print(f"Timestamp: {current_time}")
                    print(f"Daily Resistance: {daily_resistance}")
                    print(f"Current Close: {current_close}")
                    print(f"BB Upper: {self.bb_upper[-1]}")
                    print(f"Volume MA: {self.volume_ma[-1]}")
                
                # Check for breakout conditions
                breakout_conditions = (
                    current_close > daily_resistance and  # Price above resistance
                    current_close > self.bb_upper[-1] and  # Price above upper BB
                    current_volume > self.volume_ma[-1] * self.volume_factor  # Increased volume
                )
                
                # Check if we already have a position
                if not self.position and breakout_conditions:
                    # Calculate entry, stop loss and take profit
                    entry_price = current_close  # Enter at market price
                    stop_loss = max(0, entry_price - self.atr[-1] * self.sl_atr_mult)
                    take_profit = entry_price * (1 + self.tp_percent / 100)
                    
                    # Calculate position size (risk 2% of portfolio per trade)
                    risk_amount = 0.02 * self._broker.equity
                    price_risk = entry_price - stop_loss
                    if price_risk > 0:
                        size = risk_amount / price_risk
                        
                        # Log trade details
                        if not hasattr(self._broker, '_cash_adjustments') or not self._broker._cash_adjustments:
                            print(f"BREAKOUT detected at {current_time}")
                            print(f"Entry Price: {entry_price}")
                            print(f"Stop Loss: {stop_loss}")
                            print(f"Take Profit: {take_profit}")
                            print(f"Position Size: {size}")
                        
                        # Enter the trade with size, stop loss and take profit
                        self.buy(size=size, sl=stop_loss, tp=take_profit)
                        self.trade_count += 1
        
        # Run backtest
        print("\nRunning backtest...")
        bt = Backtest(hourly_data, ImprovedBreakoutStrategy, cash=100000, commission=.002)
        
        # Choose whether to optimize or run with default parameters
        optimize = input("Do you want to optimize parameters? (y/n, default: n): ").lower() == 'y'
        
        if optimize:
            print("Optimizing strategy parameters...")
            try:
                # Initialize with default parameters as fallback
                best_params = {
                    'atr_period': ImprovedBreakoutStrategy.atr_period,
                    'tp_percent': ImprovedBreakoutStrategy.tp_percent,
                    'sl_atr_mult': ImprovedBreakoutStrategy.sl_atr_mult
                }
                best_sharpe = -np.inf
                
                # Iterate through all parameter combinations
                for atr in ImprovedBreakoutStrategy.atr_period_range:
                    for tp in ImprovedBreakoutStrategy.tp_percent_range:
                        for sl in ImprovedBreakoutStrategy.sl_atr_mult_range:
                            print(f"\nTesting params: ATR={atr}, TP%={tp}, SL={sl}")
                            
                            try:
                                temp_stats = bt.run(
                                    atr_period=atr,
                                    tp_percent=tp,
                                    sl_atr_mult=sl
                                )
                                
                                # Check if Sharpe Ratio is valid
                                if not np.isnan(temp_stats['Sharpe Ratio']):
                                    if temp_stats['Sharpe Ratio'] > best_sharpe:
                                        best_sharpe = temp_stats['Sharpe Ratio']
                                        best_params = {
                                            'atr_period': atr,
                                            'tp_percent': tp,
                                            'sl_atr_mult': sl
                                        }
                            except Exception as e:
                                print(f"Skipped parameters {atr}/{tp}/{sl} due to error: {str(e)}")
                
                # Verify if any parameters were found
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
        
        # Plot the backtest results
        bt.plot(filename=f"backtest_results_{symbol.replace('-', '_')}.html")
        
        # Print detailed statistics
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
