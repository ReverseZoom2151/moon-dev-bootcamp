# binance_rrs_data_processor.py
import pandas as pd
import numpy as np
import logging
from binance_rrs_config import VOLATILITY_LOOKBACK, VOLUME_LOOKBACK

# Setup logger for this module
logger = logging.getLogger(__name__)

def calculate_returns_and_volatility(df: pd.DataFrame, volatility_window: int = VOLATILITY_LOOKBACK) -> pd.DataFrame:
    """
    Calculates various return metrics and volatility for price analysis.
    
    Args:
        df: DataFrame with OHLCV data including 'close' and 'timestamp' columns
        volatility_window: Rolling window size for volatility calculation
    
    Returns:
        DataFrame with added return and volatility columns
    """
    if df.empty:
        logger.warning("Empty DataFrame provided for returns calculation")
        return df
    
    required_columns = ['close', 'timestamp']
    if not all(col in df.columns for col in required_columns):
        logger.error(f"Missing required columns. Need: {required_columns}, Have: {list(df.columns)}")
        return df
    
    logger.debug(f"Calculating returns and volatility for {len(df)} data points")
    
    # Create a copy to avoid modifying original
    result_df = df.copy()
    
    # Ensure data is sorted by timestamp
    result_df = result_df.sort_values('timestamp').reset_index(drop=True)
    
    # Calculate basic returns
    result_df['price_change'] = result_df['close'].diff()
    result_df['simple_return'] = result_df['close'].pct_change()
    result_df['log_return'] = np.log(result_df['close'] / result_df['close'].shift(1))
    
    # Calculate cumulative returns
    result_df['cumulative_return'] = (1 + result_df['simple_return']).cumprod() - 1
    result_df['cumulative_log_return'] = result_df['log_return'].cumsum()
    
    # Calculate volatility (rolling standard deviation of returns)
    result_df['volatility'] = result_df['log_return'].rolling(
        window=volatility_window, min_periods=max(1, volatility_window//2)
    ).std()
    
    # Calculate annualized volatility (assuming different timeframes)
    # This is a rough estimate - adjust multiplier based on actual timeframe
    result_df['annualized_volatility'] = result_df['volatility'] * np.sqrt(252)  # Assume daily-like frequency
    
    # Calculate realized volatility (high-low based)
    result_df['hl_volatility'] = np.log(result_df['high'] / result_df['low'])
    result_df['rolling_hl_volatility'] = result_df['hl_volatility'].rolling(
        window=volatility_window, min_periods=max(1, volatility_window//2)
    ).mean()
    
    # Calculate price momentum indicators
    result_df['momentum_5'] = result_df['close'] / result_df['close'].shift(5) - 1
    result_df['momentum_10'] = result_df['close'] / result_df['close'].shift(10) - 1
    result_df['momentum_20'] = result_df['close'] / result_df['close'].shift(20) - 1
    
    # Calculate moving averages for trend analysis
    result_df['ma_5'] = result_df['close'].rolling(window=5, min_periods=1).mean()
    result_df['ma_20'] = result_df['close'].rolling(window=20, min_periods=1).mean()
    result_df['ma_50'] = result_df['close'].rolling(window=50, min_periods=1).mean()
    
    # Calculate price relative to moving averages
    result_df['price_vs_ma5'] = (result_df['close'] - result_df['ma_5']) / result_df['ma_5']
    result_df['price_vs_ma20'] = (result_df['close'] - result_df['ma_20']) / result_df['ma_20']
    result_df['price_vs_ma50'] = (result_df['close'] - result_df['ma_50']) / result_df['ma_50']
    
    # Handle infinite values and NaNs
    result_df = result_df.replace([np.inf, -np.inf], np.nan)
    
    logger.info(f"Successfully calculated returns and volatility metrics")
    logger.debug(f"Added columns: {[col for col in result_df.columns if col not in df.columns]}")
    
    return result_df

def calculate_volume_metrics(df: pd.DataFrame, volume_window: int = VOLUME_LOOKBACK) -> pd.DataFrame:
    """
    Calculates volume-based metrics for market strength analysis.
    
    Args:
        df: DataFrame with OHLCV data including 'volume' column
        volume_window: Rolling window size for volume calculations
    
    Returns:
        DataFrame with added volume metrics
    """
    if df.empty:
        logger.warning("Empty DataFrame provided for volume calculation")
        return df
    
    if 'volume' not in df.columns:
        logger.error("Volume column not found in DataFrame")
        return df
    
    logger.debug(f"Calculating volume metrics for {len(df)} data points")
    
    # Create a copy to avoid modifying original
    result_df = df.copy()
    
    # Calculate volume metrics
    result_df['volume_ma'] = result_df['volume'].rolling(
        window=volume_window, min_periods=max(1, volume_window//2)
    ).mean()
    
    result_df['volume_ratio'] = result_df['volume'] / result_df['volume_ma']
    
    # Volume-weighted average price (VWAP) approximation
    result_df['typical_price'] = (result_df['high'] + result_df['low'] + result_df['close']) / 3
    result_df['volume_price'] = result_df['typical_price'] * result_df['volume']
    
    result_df['vwap'] = result_df['volume_price'].rolling(
        window=volume_window, min_periods=1
    ).sum() / result_df['volume'].rolling(
        window=volume_window, min_periods=1
    ).sum()
    
    # Volume trend analysis
    result_df['volume_trend'] = result_df['volume'].rolling(
        window=volume_window, min_periods=max(1, volume_window//2)
    ).apply(lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) > 1 else 0, raw=True)
    
    # On-Balance Volume (OBV) approximation
    result_df['obv_factor'] = np.where(result_df['close'] > result_df['close'].shift(1), 1,
                                      np.where(result_df['close'] < result_df['close'].shift(1), -1, 0))
    result_df['obv'] = (result_df['volume'] * result_df['obv_factor']).cumsum()
    
    # Volume spikes detection
    volume_std = result_df['volume'].rolling(
        window=volume_window, min_periods=max(1, volume_window//2)
    ).std()
    result_df['volume_spike'] = (result_df['volume'] - result_df['volume_ma']) / volume_std
    
    # Money Flow Index approximation
    result_df['raw_money_flow'] = result_df['typical_price'] * result_df['volume']
    result_df['positive_money_flow'] = np.where(
        result_df['typical_price'] > result_df['typical_price'].shift(1),
        result_df['raw_money_flow'], 0
    )
    result_df['negative_money_flow'] = np.where(
        result_df['typical_price'] < result_df['typical_price'].shift(1),
        result_df['raw_money_flow'], 0
    )
    
    # Calculate money flow ratio
    positive_mf_sum = result_df['positive_money_flow'].rolling(
        window=volume_window, min_periods=1
    ).sum()
    negative_mf_sum = result_df['negative_money_flow'].rolling(
        window=volume_window, min_periods=1
    ).sum()
    
    result_df['money_flow_ratio'] = positive_mf_sum / (negative_mf_sum + 1e-10)  # Avoid division by zero
    result_df['mfi'] = 100 - (100 / (1 + result_df['money_flow_ratio']))
    
    # Handle infinite values and NaNs
    result_df = result_df.replace([np.inf, -np.inf], np.nan)
    
    # Drop intermediate calculation columns
    cols_to_drop = ['typical_price', 'volume_price', 'obv_factor', 'raw_money_flow', 
                   'positive_money_flow', 'negative_money_flow', 'money_flow_ratio']
    result_df = result_df.drop(columns=[col for col in cols_to_drop if col in result_df.columns])
    
    logger.info(f"Successfully calculated volume metrics")
    
    return result_df

def calculate_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates additional technical indicators for enhanced analysis.
    
    Args:
        df: DataFrame with OHLCV data
    
    Returns:
        DataFrame with added technical indicators
    """
    if df.empty:
        return df
    
    logger.debug(f"Calculating technical indicators for {len(df)} data points")
    
    result_df = df.copy()
    
    # RSI (Relative Strength Index) - 14 period
    delta = result_df['close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    avg_gain = gain.rolling(window=14, min_periods=1).mean()
    avg_loss = loss.rolling(window=14, min_periods=1).mean()
    
    rs = avg_gain / (avg_loss + 1e-10)
    result_df['rsi'] = 100 - (100 / (1 + rs))
    
    # Bollinger Bands
    bb_period = 20
    bb_std = 2
    result_df['bb_middle'] = result_df['close'].rolling(window=bb_period, min_periods=1).mean()
    bb_std_dev = result_df['close'].rolling(window=bb_period, min_periods=1).std()
    result_df['bb_upper'] = result_df['bb_middle'] + (bb_std_dev * bb_std)
    result_df['bb_lower'] = result_df['bb_middle'] - (bb_std_dev * bb_std)
    result_df['bb_position'] = (result_df['close'] - result_df['bb_lower']) / (result_df['bb_upper'] - result_df['bb_lower'])
    
    # MACD (Moving Average Convergence Divergence)
    exp12 = result_df['close'].ewm(span=12, min_periods=1).mean()
    exp26 = result_df['close'].ewm(span=26, min_periods=1).mean()
    result_df['macd_line'] = exp12 - exp26
    result_df['macd_signal'] = result_df['macd_line'].ewm(span=9, min_periods=1).mean()
    result_df['macd_histogram'] = result_df['macd_line'] - result_df['macd_signal']
    
    # Average True Range (ATR)
    high_low = result_df['high'] - result_df['low']
    high_close = np.abs(result_df['high'] - result_df['close'].shift(1))
    low_close = np.abs(result_df['low'] - result_df['close'].shift(1))
    
    true_range = np.maximum(high_low, np.maximum(high_close, low_close))
    result_df['atr'] = true_range.rolling(window=14, min_periods=1).mean()
    
    # Stochastic Oscillator
    stoch_period = 14
    lowest_low = result_df['low'].rolling(window=stoch_period, min_periods=1).min()
    highest_high = result_df['high'].rolling(window=stoch_period, min_periods=1).max()
    result_df['stoch_k'] = 100 * (result_df['close'] - lowest_low) / (highest_high - lowest_low + 1e-10)
    result_df['stoch_d'] = result_df['stoch_k'].rolling(window=3, min_periods=1).mean()
    
    # Williams %R
    result_df['williams_r'] = -100 * (highest_high - result_df['close']) / (highest_high - lowest_low + 1e-10)
    
    # Handle infinite values and NaNs
    result_df = result_df.replace([np.inf, -np.inf], np.nan)
    
    logger.info(f"Successfully calculated technical indicators")
    
    return result_df

def detect_market_regime(df: pd.DataFrame) -> pd.DataFrame:
    """
    Detects market regimes (trending, ranging, volatile) for context.
    
    Args:
        df: DataFrame with price and volatility data
    
    Returns:
        DataFrame with market regime indicators
    """
    if df.empty or len(df) < 50:
        return df
    
    logger.debug(f"Detecting market regimes for {len(df)} data points")
    
    result_df = df.copy()
    
    # Trend detection using moving average slopes
    ma_20_slope = result_df['ma_20'].rolling(window=10, min_periods=1).apply(
        lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) > 1 else 0, raw=True
    )
    
    # Volatility regime
    vol_ma = result_df['volatility'].rolling(window=20, min_periods=1).mean()
    vol_std = result_df['volatility'].rolling(window=20, min_periods=1).std()
    
    # Market regime classification
    result_df['trend_strength'] = np.abs(ma_20_slope)
    result_df['volatility_regime'] = np.where(
        result_df['volatility'] > vol_ma + vol_std, 'high_vol',
        np.where(result_df['volatility'] < vol_ma - vol_std, 'low_vol', 'normal_vol')
    )
    
    result_df['market_regime'] = np.where(
        result_df['trend_strength'] > result_df['trend_strength'].quantile(0.7), 'trending',
        np.where(result_df['trend_strength'] < result_df['trend_strength'].quantile(0.3), 'ranging', 'mixed')
    )
    
    logger.info(f"Successfully detected market regimes")
    
    return result_df

if __name__ == "__main__":
    # Test the data processor
    import logging
    logging.basicConfig(level=logging.INFO)
    
    # Create sample data for testing
    dates = pd.date_range('2024-01-01', periods=100, freq='H')
    np.random.seed(42)
    
    # Generate synthetic OHLCV data
    base_price = 50000
    returns = np.random.normal(0, 0.02, 100)
    prices = [base_price]
    
    for r in returns:
        prices.append(prices[-1] * (1 + r))
    
    test_df = pd.DataFrame({
        'timestamp': dates,
        'open': prices[:-1],
        'high': [p * (1 + abs(np.random.normal(0, 0.01))) for p in prices[:-1]],
        'low': [p * (1 - abs(np.random.normal(0, 0.01))) for p in prices[:-1]],
        'close': prices[1:],
        'volume': np.random.lognormal(10, 1, 100),
        'quote_asset_volume': np.random.lognormal(15, 1, 100),
        'number_of_trades': np.random.randint(100, 1000, 100)
    })
    
    print("Testing Binance RRS data processor...")
    
    # Test returns and volatility calculation
    result_df = calculate_returns_and_volatility(test_df)
    print(f"Returns calculation - added {len(result_df.columns) - len(test_df.columns)} columns")
    
    # Test volume metrics
    result_df = calculate_volume_metrics(result_df)
    print(f"Volume metrics - total columns: {len(result_df.columns)}")
    
    # Test technical indicators
    result_df = calculate_technical_indicators(result_df)
    print(f"Technical indicators - total columns: {len(result_df.columns)}")
    
    # Test market regime detection
    result_df = detect_market_regime(result_df)
    print(f"Market regime - total columns: {len(result_df.columns)}")
    
    print(f"Final DataFrame shape: {result_df.shape}")
    print("Sample of calculated metrics:")
    print(result_df[['timestamp', 'close', 'log_return', 'volatility', 'volume_ratio', 'rsi']].tail())
    
    print("✅ Binance RRS data processor test complete")
