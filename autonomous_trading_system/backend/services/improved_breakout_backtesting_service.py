import os
import pandas as pd
import yfinance as yf
from datetime import datetime
from backtesting import Backtest
from strategies.improved_breakout_strategy import ImprovedBreakoutStrategy
import logging

DATA_DIR = "data"
logger = logging.getLogger(__name__)

def fetch_data(symbol: str):
    """Fetch and prepare daily and hourly OHLCV data for a symbol via Yahoo Finance"""
    # Ensure data directory exists
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
        logger.info(f"Created data directory at: {os.path.abspath(DATA_DIR)}")
    # File paths for reference
    daily_file = os.path.join(DATA_DIR, f"{symbol.replace('/','_')}_daily.csv")
    hourly_file = os.path.join(DATA_DIR, f"{symbol.replace('/','_')}_hourly.csv")
    logger.info(f"Daily file path: {os.path.abspath(daily_file)}")
    logger.info(f"Hourly file path: {os.path.abspath(hourly_file)}")
    # Define date ranges
    end_date = datetime.utcnow()
    start_daily = end_date - pd.Timedelta(days=365)
    start_hourly = end_date - pd.Timedelta(days=60)
    # Download fresh data
    try:
        logger.info(f"Downloading daily data for {symbol}: {start_daily.date()} to {end_date.date()}")
        daily = yf.download(
            symbol,
            start=start_daily.strftime('%Y-%m-%d'),
            end=end_date.strftime('%Y-%m-%d'),
            interval='1d',
            progress=False,
            show_errors=False
        )
        logger.info(f"Downloading hourly data for {symbol}: {start_hourly.date()} to {end_date.date()}")
        hourly = yf.download(
            symbol,
            start=start_hourly.strftime('%Y-%m-%d'),
            end=end_date.strftime('%Y-%m-%d'),
            interval='1h',
            progress=False,
            show_errors=False
        )
    except Exception as e:
        logger.warning(f"Error downloading data: {e}. Trying fallback method.")
        try:
            daily = yf.download(symbol, period="1y", interval='1d', progress=False)
            hourly = yf.download(symbol, period="60d", interval='1h', progress=False)
        except Exception as e2:
            logger.error(f"Fallback download failed: {e2}")
            raise ValueError(f"Could not download data for {symbol}.")
    # Flatten multi-index columns
    if isinstance(daily.columns, pd.MultiIndex):
        logger.info("Detected multi-index columns in daily data, flattening.")
        daily.columns = [c[0] for c in daily.columns]
    if isinstance(hourly.columns, pd.MultiIndex):
        logger.info("Detected multi-index columns in hourly data, flattening.")
        hourly.columns = [c[0] for c in hourly.columns]
    # Drop NaNs
    daily.dropna(inplace=True)
    hourly.dropna(inplace=True)
    # Validate non-empty data
    if daily.empty:
        logger.error(f"No daily data found for {symbol}")
        raise ValueError(f"No daily data for {symbol}.")
    if hourly.empty:
        logger.error(f"No hourly data found for {symbol}")
        raise ValueError(f"No hourly data for {symbol}.")
    # Rename columns to standard format
    cols_map = {
        'open': 'Open', 'high': 'High', 'low': 'Low',
        'close': 'Close', 'volume': 'Volume', 'adj close': 'Adj Close'
    }
    daily.rename(columns={col: cols_map.get(col.lower(), col) for col in daily.columns}, inplace=True)
    hourly.rename(columns={col: cols_map.get(col.lower(), col) for col in hourly.columns}, inplace=True)
    # Ensure sufficient data length
    if len(daily) < 5:
        raise ValueError(f"Insufficient daily data for {symbol} ({len(daily)} points).")
    if len(hourly) < 5:
        raise ValueError(f"Insufficient hourly data for {symbol} ({len(hourly)} points).")
    # Compute daily resistance
    window = min(20, len(daily))
    daily['resis'] = daily['High'].rolling(window=window).max().fillna(method='ffill')
    # Filter hourly to match daily date range
    daily_start = daily.index.min().date()
    daily_end = daily.index.max().date()
    hourly = hourly[(hourly.index.date >= daily_start) & (hourly.index.date <= daily_end)]
    if hourly.empty:
        logger.warning("No overlap between daily and hourly data after filtering.")
    return daily, hourly


def run_improved_breakout_backtest(
    symbol: str,
    initial_cash: float = 100000,
    commission: float = 0.002,
    optimize: bool = False
) -> dict:
    """Run backtest for ImprovedBreakoutStrategy, optionally optimize parameters"""
    # Fetch data
    daily, hourly = fetch_data(symbol)
    # Attach daily resistance
    ImprovedBreakoutStrategy.daily_resistance = daily['resis']
    # Initialize Backtest
    bt = Backtest(
        hourly,
        ImprovedBreakoutStrategy,
        cash=initial_cash,
        commission=commission
    )
    # Optimization
    best_params = {
        'atr_period': ImprovedBreakoutStrategy.atr_period,
        'tp_percent': ImprovedBreakoutStrategy.tp_percent,
        'sl_atr_mult': ImprovedBreakoutStrategy.sl_atr_mult
    }
    if optimize:
        best_sharpe = -float('inf')
        for atr in ImprovedBreakoutStrategy.atr_period_range:
            for tp in ImprovedBreakoutStrategy.tp_percent_range:
                for sl in ImprovedBreakoutStrategy.sl_atr_mult_range:
                    stats = bt.run(atr_period=atr, tp_percent=tp, sl_atr_mult=sl)
                    sharpe = stats.get('Sharpe Ratio', None)
                    if sharpe is not None and not pd.isna(sharpe):
                        if sharpe > best_sharpe:
                            best_sharpe = sharpe
                            best_params = {'atr_period': atr, 'tp_percent': tp, 'sl_atr_mult': sl}
                            best_stats = stats
        # Fallback to default if none valid
        if best_sharpe == -float('inf'):
            best_stats = bt.run()
    else:
        best_stats = bt.run()
    # Generate plot file
    plot_file = f"backtest_improved_breakout_{symbol.replace('/', '_')}.html"
    bt.plot(filename=plot_file)
    # Serialize stats
    return {
        'strategy_name': 'improved_breakout',
        'symbol': symbol,
        'initial_cash': initial_cash,
        'commission': commission,
        'optimize': optimize,
        'best_params': best_params,
        'stats': best_stats.to_dict(),
        'plot_file': plot_file
    } 