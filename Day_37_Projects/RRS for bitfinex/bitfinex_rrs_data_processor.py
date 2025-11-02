# bitfinex_rrs_data_processor.py
import pandas as pd
import numpy as np
import logging
from bitfinex_rrs_config import VOLATILITY_LOOKBACK, VOLUME_LOOKBACK, PROFESSIONAL_INDICATORS

# Setup logger for this module
logger = logging.getLogger(__name__)

def calculate_professional_returns_and_volatility(df: pd.DataFrame, 
                                                volatility_window: int = VOLATILITY_LOOKBACK) -> pd.DataFrame:
    """
    Calculates professional-grade returns and volatility metrics for Bitfinex trading.
    
    Args:
        df: DataFrame with OHLCV data
        volatility_window: Rolling window for volatility calculations
        
    Returns:
        DataFrame with added return and volatility columns
    """
    if df.empty:
        logger.warning("Empty DataFrame provided for professional returns calculation")
        return df
    
    try:
        df = df.copy()
        
        # Professional return calculations
        df['returns_1'] = df['close'].pct_change()  # 1-period returns
        df['returns_log'] = np.log(df['close'] / df['close'].shift(1))  # Log returns
        df['returns_5'] = df['close'].pct_change(periods=5)  # 5-period returns
        df['returns_10'] = df['close'].pct_change(periods=10)  # 10-period returns
        
        # Professional volatility metrics
        df['volatility'] = df['returns_1'].rolling(window=volatility_window).std()
        df['volatility_annualized'] = df['volatility'] * np.sqrt(365)  # Annualized volatility
        df['volatility_log'] = df['returns_log'].rolling(window=volatility_window).std()
        
        # Professional risk-adjusted metrics
        df['sharpe_ratio_raw'] = df['returns_1'].rolling(window=volatility_window).mean() / df['volatility']
        df['downside_volatility'] = df['returns_1'][df['returns_1'] < 0].rolling(window=volatility_window).std()
        df['sortino_ratio'] = df['returns_1'].rolling(window=volatility_window).mean() / df['downside_volatility']
        
        # Advanced volatility measures for professional trading
        df['garch_volatility'] = df['returns_1'].rolling(window=volatility_window).std() * \
                                np.sqrt(df['returns_1'].rolling(window=5).var())  # GARCH-like
        df['parkinson_volatility'] = np.sqrt((1/(4*np.log(2))) * 
                                           (np.log(df['high']/df['low'])**2).rolling(window=volatility_window).mean())
        
        # Professional momentum indicators
        df['momentum_1'] = df['close'] / df['close'].shift(1) - 1
        df['momentum_5'] = df['close'] / df['close'].shift(5) - 1
        df['momentum_20'] = df['close'] / df['close'].shift(20) - 1
        
        # Rate of change for institutional analysis
        df['roc_5'] = ((df['close'] / df['close'].shift(5)) - 1) * 100
        df['roc_10'] = ((df['close'] / df['close'].shift(10)) - 1) * 100
        df['roc_20'] = ((df['close'] / df['close'].shift(20)) - 1) * 100
        
        logger.info(f"Professional returns and volatility calculated for {len(df)} data points")
        
    except Exception as e:
        logger.error(f"Error calculating professional returns and volatility: {e}")
        
    return df

def calculate_professional_volume_metrics(df: pd.DataFrame, 
                                        volume_window: int = VOLUME_LOOKBACK) -> pd.DataFrame:
    """
    Calculates professional volume metrics for institutional analysis.
    
    Args:
        df: DataFrame with OHLCV data
        volume_window: Rolling window for volume calculations
        
    Returns:
        DataFrame with added volume analysis columns
    """
    if df.empty:
        logger.warning("Empty DataFrame provided for professional volume calculation")
        return df
    
    try:
        df = df.copy()
        
        # Professional volume metrics
        df['volume_sma'] = df['volume'].rolling(window=volume_window).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma']
        df['volume_zscore'] = (df['volume'] - df['volume_sma']) / df['volume'].rolling(window=volume_window).std()
        
        # Volume-weighted metrics for institutional trading
        df['vwap'] = (df['volume'] * (df['high'] + df['low'] + df['close']) / 3).cumsum() / df['volume'].cumsum()
        df['volume_weighted_return'] = df['returns_1'] * df['volume_ratio']
        
        # Professional liquidity metrics
        df['dollar_volume'] = df['volume'] * df['close']  # USD volume
        df['dollar_volume_sma'] = df['dollar_volume'].rolling(window=volume_window).mean()
        df['liquidity_ratio'] = df['dollar_volume'] / df['dollar_volume_sma']
        
        # Advanced volume indicators for margin trading
        df['obv'] = df['volume'].where(df['close'] > df['close'].shift(1), -df['volume']).cumsum()  # On-Balance Volume
        df['cmf'] = ((df['close'] - df['low'] - (df['high'] - df['close'])) / (df['high'] - df['low']) * df['volume']).rolling(window=20).sum() / df['volume'].rolling(window=20).sum()  # Chaikin Money Flow
        df['ad_line'] = (((df['close'] - df['low']) - (df['high'] - df['close'])) / (df['high'] - df['low']) * df['volume']).cumsum()  # Accumulation/Distribution
        
        # Professional volume oscillators
        df['pvo'] = (df['volume'].ewm(span=12).mean() - df['volume'].ewm(span=26).mean()) / df['volume'].ewm(span=26).mean() * 100
        df['volume_rsi'] = calculate_rsi(df['volume'], 14)
        
        # Institutional flow metrics
        df['buying_pressure'] = df['volume'].where(df['close'] > df['open'], 0)
        df['selling_pressure'] = df['volume'].where(df['close'] < df['open'], 0)
        df['net_flow'] = df['buying_pressure'] - df['selling_pressure']
        df['flow_ratio'] = df['buying_pressure'] / (df['buying_pressure'] + df['selling_pressure'])
        
        logger.info(f"Professional volume metrics calculated for {len(df)} data points")
        
    except Exception as e:
        logger.error(f"Error calculating professional volume metrics: {e}")
        
    return df

def calculate_professional_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates professional technical indicators for institutional Bitfinex trading.
    
    Args:
        df: DataFrame with OHLCV data
        
    Returns:
        DataFrame with added technical indicator columns
    """
    if df.empty:
        logger.warning("Empty DataFrame provided for professional technical indicators")
        return df
    
    try:
        df = df.copy()
        
        # Professional moving averages
        for period in PROFESSIONAL_INDICATORS['sma_periods']:
            df[f'sma_{period}'] = df['close'].rolling(window=period).mean()
            df[f'ema_{period}'] = df['close'].ewm(span=period).mean()
        
        # Advanced trend indicators
        df['macd'] = df['close'].ewm(span=12).mean() - df['close'].ewm(span=26).mean()
        df['macd_signal'] = df['macd'].ewm(span=9).mean()
        df['macd_histogram'] = df['macd'] - df['macd_signal']
        
        # Professional oscillators
        df['rsi'] = calculate_rsi(df['close'], 14)
        df['stoch_k'], df['stoch_d'] = calculate_stochastic(df, 14, 3)
        df['williams_r'] = calculate_williams_r(df, 14)
        
        # Bollinger Bands for professional trading
        bb_period = 20
        bb_std = 2
        df['bb_middle'] = df['close'].rolling(window=bb_period).mean()
        bb_std_calc = df['close'].rolling(window=bb_period).std()
        df['bb_upper'] = df['bb_middle'] + (bb_std_calc * bb_std)
        df['bb_lower'] = df['bb_middle'] - (bb_std_calc * bb_std)
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        
        # Professional momentum indicators
        df['cci'] = calculate_cci(df, 20)
        df['atr'] = calculate_atr(df, 14)  # Average True Range
        df['adx'] = calculate_adx(df, 14)  # Directional Movement Index
        
        # Institutional-grade trend analysis
        df['aroon_up'], df['aroon_down'] = calculate_aroon(df, 14)
        df['aroon_oscillator'] = df['aroon_up'] - df['aroon_down']
        
        # Professional support/resistance levels
        df['pivot_point'] = (df['high'] + df['low'] + df['close']) / 3
        df['resistance_1'] = 2 * df['pivot_point'] - df['low']
        df['support_1'] = 2 * df['pivot_point'] - df['high']
        df['resistance_2'] = df['pivot_point'] + (df['high'] - df['low'])
        df['support_2'] = df['pivot_point'] - (df['high'] - df['low'])
        
        # Market structure indicators for derivatives trading
        df['fractal_high'] = ((df['high'].shift(2) < df['high'].shift(1)) & 
                             (df['high'].shift(1) > df['high']) &
                             (df['high'] < df['high'].shift(-1)) & 
                             (df['high'].shift(-1) < df['high'].shift(-2)))
        df['fractal_low'] = ((df['low'].shift(2) > df['low'].shift(1)) & 
                            (df['low'].shift(1) < df['low']) &
                            (df['low'] > df['low'].shift(-1)) & 
                            (df['low'].shift(-1) > df['low'].shift(-2)))
        
        logger.info(f"Professional technical indicators calculated for {len(df)} data points")
        
    except Exception as e:
        logger.error(f"Error calculating professional technical indicators: {e}")
        
    return df

def calculate_professional_market_regime(df: pd.DataFrame) -> pd.DataFrame:
    """
    Determines professional market regime for institutional position sizing.
    
    Args:
        df: DataFrame with calculated indicators
        
    Returns:
        DataFrame with market regime classification
    """
    if df.empty:
        return df
    
    try:
        df = df.copy()
        
        # Trend strength classification
        df['trend_strength'] = 'neutral'
        df.loc[(df['adx'] > 25) & (df['sma_20'] > df['sma_50']), 'trend_strength'] = 'strong_uptrend'
        df.loc[(df['adx'] > 25) & (df['sma_20'] < df['sma_50']), 'trend_strength'] = 'strong_downtrend'
        df.loc[df['adx'] < 15, 'trend_strength'] = 'sideways'
        
        # Volatility regime for risk management
        volatility_percentile = df['volatility'].rolling(window=100).rank(pct=True)
        df['volatility_regime'] = 'normal'
        df.loc[volatility_percentile > 0.8, 'volatility_regime'] = 'high'
        df.loc[volatility_percentile < 0.2, 'volatility_regime'] = 'low'
        
        # Professional market conditions
        df['market_condition'] = 'neutral'
        df.loc[(df['trend_strength'].isin(['strong_uptrend'])) & 
               (df['volatility_regime'] == 'normal'), 'market_condition'] = 'bullish_trending'
        df.loc[(df['trend_strength'].isin(['strong_downtrend'])) & 
               (df['volatility_regime'] == 'normal'), 'market_condition'] = 'bearish_trending'
        df.loc[df['volatility_regime'] == 'high', 'market_condition'] = 'volatile'
        df.loc[(df['trend_strength'] == 'sideways') & 
               (df['volatility_regime'] == 'low'), 'market_condition'] = 'consolidating'
        
        logger.info("Professional market regime analysis completed")
        
    except Exception as e:
        logger.error(f"Error in professional market regime calculation: {e}")
    
    return df

# Helper functions for professional indicators

def calculate_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Calculate professional RSI indicator."""
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calculate_stochastic(df: pd.DataFrame, k_period: int = 14, d_period: int = 3):
    """Calculate professional Stochastic oscillator."""
    lowest_low = df['low'].rolling(window=k_period).min()
    highest_high = df['high'].rolling(window=k_period).max()
    k_percent = 100 * ((df['close'] - lowest_low) / (highest_high - lowest_low))
    d_percent = k_percent.rolling(window=d_period).mean()
    return k_percent, d_percent

def calculate_williams_r(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Calculate professional Williams %R."""
    highest_high = df['high'].rolling(window=period).max()
    lowest_low = df['low'].rolling(window=period).min()
    return -100 * ((highest_high - df['close']) / (highest_high - lowest_low))

def calculate_cci(df: pd.DataFrame, period: int = 20) -> pd.Series:
    """Calculate professional Commodity Channel Index."""
    typical_price = (df['high'] + df['low'] + df['close']) / 3
    sma_tp = typical_price.rolling(window=period).mean()
    mad = typical_price.rolling(window=period).apply(lambda x: np.fabs(x - x.mean()).mean())
    return (typical_price - sma_tp) / (0.015 * mad)

def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Calculate professional Average True Range."""
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    tr = np.maximum(high_low, np.maximum(high_close, low_close))
    return tr.rolling(window=period).mean()

def calculate_adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Calculate professional Average Directional Index."""
    high_diff = df['high'].diff()
    low_diff = -df['low'].diff()
    
    plus_dm = high_diff.where((high_diff > low_diff) & (high_diff > 0), 0)
    minus_dm = low_diff.where((low_diff > high_diff) & (low_diff > 0), 0)
    
    atr = calculate_atr(df, period)
    plus_di = 100 * (plus_dm.rolling(window=period).mean() / atr)
    minus_di = 100 * (minus_dm.rolling(window=period).mean() / atr)
    
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    return dx.rolling(window=period).mean()

def calculate_aroon(df: pd.DataFrame, period: int = 14):
    """Calculate professional Aroon indicators."""
    aroon_up = 100 * (df['high'].rolling(window=period).apply(lambda x: x.argmax()) / (period - 1))
    aroon_down = 100 * (df['low'].rolling(window=period).apply(lambda x: x.argmin()) / (period - 1))
    return aroon_up, aroon_down

if __name__ == "__main__":
    # Test the professional data processor
    logging.basicConfig(level=logging.INFO)
    
    print("Testing Bitfinex professional data processor...")
    
    # Create test data
    dates = pd.date_range('2024-01-01', periods=100, freq='H')
    test_data = pd.DataFrame({
        'timestamp': dates,
        'open': np.random.uniform(45000, 55000, 100),
        'high': np.random.uniform(45000, 55000, 100),
        'low': np.random.uniform(45000, 55000, 100),
        'close': np.random.uniform(45000, 55000, 100),
        'volume': np.random.uniform(1, 100, 100)
    })
    
    # Ensure OHLC logic
    for i in range(len(test_data)):
        high_val = max(test_data.loc[i, 'open'], test_data.loc[i, 'close'])
        low_val = min(test_data.loc[i, 'open'], test_data.loc[i, 'close'])
        test_data.loc[i, 'high'] = max(high_val, test_data.loc[i, 'high'])
        test_data.loc[i, 'low'] = min(low_val, test_data.loc[i, 'low'])
    
    print(f"Test data shape: {test_data.shape}")
    
    # Test professional processing functions
    processed_data = calculate_professional_returns_and_volatility(test_data)
    processed_data = calculate_professional_volume_metrics(processed_data)
    processed_data = calculate_professional_technical_indicators(processed_data)
    processed_data = calculate_professional_market_regime(processed_data)
    
    print(f"Processed data shape: {processed_data.shape}")
    print(f"Professional indicators added: {processed_data.shape[1] - test_data.shape[1]}")
    
    print("✅ Bitfinex professional data processor test complete")
