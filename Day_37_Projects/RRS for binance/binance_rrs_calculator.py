# binance_rrs_calculator.py
import pandas as pd
import numpy as np
import logging
from typing import Dict
from binance_rrs_config import RRS_SMOOTHING_WINDOW, MIN_DATA_POINTS

# Setup logger for this module
logger = logging.getLogger(__name__)

def calculate_rrs(symbol_df: pd.DataFrame, benchmark_df: pd.DataFrame, 
                  symbol_name: str = "Unknown", benchmark_name: str = "BTC") -> pd.DataFrame:
    """Calculates Relative Rotation Strength (RRS) metrics for Binance trading pairs.

    Aligns the symbol data with the benchmark data on timestamp and calculates
    differential return, normalized return, raw RRS, and smoothed RRS.

    Args:
        symbol_df: DataFrame for the symbol, must include 'timestamp', 'log_return',
                   'volatility', and 'volume_ratio' columns.
        benchmark_df: DataFrame for the benchmark, must include 'timestamp' and
                      'log_return' columns.
        symbol_name: Name of the symbol for logging purposes.
        benchmark_name: Name of the benchmark for logging purposes.

    Returns:
        DataFrame with RRS metrics added, or an empty DataFrame if calculation fails
        or no overlapping data exists.
    """
    required_symbol_cols = ['timestamp', 'log_return', 'volatility', 'volume_ratio']
    required_benchmark_cols = ['timestamp', 'log_return']

    if not all(col in symbol_df.columns for col in required_symbol_cols):
        logger.error(f"Symbol DataFrame missing required columns. Need: {required_symbol_cols}, Have: {symbol_df.columns.tolist()}")
        return pd.DataFrame()
    if not all(col in benchmark_df.columns for col in required_benchmark_cols):
        logger.error(f"Benchmark DataFrame missing required columns. Need: {required_benchmark_cols}, Have: {benchmark_df.columns.tolist()}")
        return pd.DataFrame()

    logger.debug(f"Calculating RRS for {symbol_name} vs {benchmark_name}")
    logger.debug(f"Input symbol df shape: {symbol_df.shape}, Benchmark df shape: {benchmark_df.shape}")

    # Ensure timestamps are datetime objects
    symbol_df = symbol_df.copy()
    benchmark_df = benchmark_df.copy()
    
    symbol_df['timestamp'] = pd.to_datetime(symbol_df['timestamp'])
    benchmark_df['timestamp'] = pd.to_datetime(benchmark_df['timestamp'])

    # Align timestamps by setting them as index
    symbol_df = symbol_df.set_index('timestamp')
    benchmark_df = benchmark_df.set_index('timestamp')

    logger.debug(f"After setting index - symbol df shape: {symbol_df.shape}, benchmark df shape: {benchmark_df.shape}")

    # Join DataFrames using inner join to keep only overlapping timestamps
    df_merged = symbol_df.join(benchmark_df[['log_return']], rsuffix='_benchmark', how='inner')

    if df_merged.empty:
        logger.warning(f"No overlapping data between {symbol_name} and {benchmark_name}")
        return pd.DataFrame()

    # Check minimum data points requirement
    if len(df_merged) < MIN_DATA_POINTS:
        logger.warning(f"Insufficient data points for {symbol_name}: {len(df_merged)} < {MIN_DATA_POINTS}")
        return pd.DataFrame()

    logger.debug(f"After join - merged df shape: {df_merged.shape}")

    # Calculate differential returns (symbol vs benchmark)
    df_merged['differential_return'] = df_merged['log_return'] - df_merged['log_return_benchmark']
    
    # Calculate cumulative differential returns
    df_merged['cumulative_differential_return'] = df_merged['differential_return'].cumsum()

    # Calculate rolling statistics for normalization
    rolling_window = min(RRS_SMOOTHING_WINDOW * 4, len(df_merged) // 4)  # Adaptive window
    rolling_window = max(rolling_window, 5)  # Minimum window size
    
    # Rolling mean and std of differential returns
    rolling_mean = df_merged['differential_return'].rolling(
        window=rolling_window, min_periods=max(1, rolling_window//2)
    ).mean()
    
    rolling_std = df_merged['differential_return'].rolling(
        window=rolling_window, min_periods=max(1, rolling_window//2)
    ).std()

    # Avoid division by zero
    rolling_std = rolling_std.fillna(1.0)
    rolling_std = rolling_std.replace(0, 1.0)

    # Calculate normalized differential returns (z-score)
    df_merged['normalized_return'] = (df_merged['differential_return'] - rolling_mean) / rolling_std

    # Calculate raw RRS score
    # Combine normalized returns with volume and volatility factors
    volume_factor = np.log1p(df_merged['volume_ratio'])  # Log transform to reduce outlier impact
    volatility_factor = 1 / (1 + df_merged['volatility'])  # Lower volatility = higher score

    df_merged['raw_rrs'] = (
        df_merged['normalized_return'] * 0.6 +  # 60% weight on returns
        volume_factor * 0.25 +                  # 25% weight on volume
        volatility_factor * 0.15                # 15% weight on volatility adjustment
    )

    # Calculate smoothed RRS using exponential moving average
    smoothing_alpha = 2 / (RRS_SMOOTHING_WINDOW + 1)
    df_merged['smoothed_rrs'] = df_merged['raw_rrs'].ewm(alpha=smoothing_alpha, min_periods=1).mean()

    # Calculate RRS momentum (rate of change in RRS)
    df_merged['rrs_momentum'] = df_merged['smoothed_rrs'].diff(periods=3)  # 3-period momentum
    
    # Calculate RRS percentile ranks (relative ranking)
    df_merged['rrs_percentile'] = df_merged['smoothed_rrs'].rolling(
        window=min(50, len(df_merged)), min_periods=10
    ).rank(pct=True) * 100

    # Calculate additional strength indicators
    # Consistency score - how consistently outperforming
    df_merged['outperformance_ratio'] = (df_merged['differential_return'] > 0).rolling(
        window=rolling_window, min_periods=max(1, rolling_window//2)
    ).mean()

    # Relative strength trend
    df_merged['rrs_trend'] = df_merged['smoothed_rrs'].rolling(
        window=min(10, len(df_merged)//2), min_periods=3
    ).apply(lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) > 2 else 0, raw=True)

    # Risk-adjusted RRS (Sharpe-like ratio)
    returns_std = df_merged['differential_return'].rolling(
        window=rolling_window, min_periods=max(1, rolling_window//2)
    ).std()
    returns_std = returns_std.fillna(1.0).replace(0, 1.0)
    
    df_merged['risk_adjusted_rrs'] = (
        df_merged['cumulative_differential_return'] / returns_std
    )

    # Handle infinite values and NaNs
    df_merged = df_merged.replace([np.inf, -np.inf], np.nan)
    
    # Fill NaN values with appropriate defaults
    numeric_columns = ['differential_return', 'normalized_return', 'raw_rrs', 
                      'smoothed_rrs', 'rrs_momentum', 'rrs_percentile',
                      'outperformance_ratio', 'rrs_trend', 'risk_adjusted_rrs']
    
    for col in numeric_columns:
        if col in df_merged.columns:
            df_merged[col] = df_merged[col].fillna(0)

    # Reset index to make timestamp a column again
    df_merged = df_merged.reset_index()

    # Add metadata
    df_merged['symbol_name'] = symbol_name
    df_merged['benchmark_name'] = benchmark_name
    df_merged['calculation_timestamp'] = pd.Timestamp.utcnow()

    logger.info(f"Successfully calculated RRS for {symbol_name} vs {benchmark_name}")
    logger.info(f"Final RRS range: {df_merged['smoothed_rrs'].min():.4f} to {df_merged['smoothed_rrs'].max():.4f}")
    logger.info(f"Mean RRS: {df_merged['smoothed_rrs'].mean():.4f}, Std: {df_merged['smoothed_rrs'].std():.4f}")

    return df_merged

def calculate_multi_timeframe_rrs(symbol_data_dict: Dict[str, pd.DataFrame], 
                                benchmark_data_dict: Dict[str, pd.DataFrame],
                                symbol_name: str, benchmark_name: str) -> Dict[str, pd.DataFrame]:
    """Calculate RRS across multiple timeframes for comprehensive analysis.
    
    Args:
        symbol_data_dict: Dict mapping timeframes to symbol DataFrames
        benchmark_data_dict: Dict mapping timeframes to benchmark DataFrames  
        symbol_name: Name of the symbol
        benchmark_name: Name of the benchmark
    
    Returns:
        Dict mapping timeframes to RRS DataFrames
    """
    rrs_results = {}
    
    for timeframe in symbol_data_dict.keys():
        if timeframe in benchmark_data_dict:
            logger.info(f"Calculating RRS for {symbol_name} vs {benchmark_name} - {timeframe}")
            
            rrs_df = calculate_rrs(
                symbol_data_dict[timeframe],
                benchmark_data_dict[timeframe],
                f"{symbol_name}_{timeframe}",
                f"{benchmark_name}_{timeframe}"
            )
            
            if not rrs_df.empty:
                rrs_results[timeframe] = rrs_df
            else:
                logger.warning(f"Failed to calculate RRS for {timeframe}")
        else:
            logger.warning(f"No benchmark data for timeframe {timeframe}")
    
    return rrs_results

def rank_symbols_by_rrs(rrs_data_dict: Dict[str, pd.DataFrame], 
                       ranking_metric: str = 'smoothed_rrs') -> pd.DataFrame:
    """Rank multiple symbols by their RRS scores.
    
    Args:
        rrs_data_dict: Dict mapping symbol names to their RRS DataFrames
        ranking_metric: Column to use for ranking ('smoothed_rrs', 'rrs_percentile', etc.)
    
    Returns:
        DataFrame with symbol rankings and key metrics
    """
    rankings = []
    
    for symbol, rrs_df in rrs_data_dict.items():
        if rrs_df.empty or ranking_metric not in rrs_df.columns:
            continue
            
        # Get latest values
        latest_data = rrs_df.iloc[-1]
        
        # Calculate summary statistics
        rrs_values = rrs_df[ranking_metric].dropna()
        if len(rrs_values) == 0:
            continue
            
        ranking_data = {
            'symbol': symbol,
            'current_rrs': latest_data[ranking_metric],
            'rrs_momentum': latest_data.get('rrs_momentum', 0),
            'rrs_trend': latest_data.get('rrs_trend', 0),
            'outperformance_ratio': latest_data.get('outperformance_ratio', 0),
            'risk_adjusted_rrs': latest_data.get('risk_adjusted_rrs', 0),
            'rrs_percentile': latest_data.get('rrs_percentile', 50),
            'volatility': latest_data.get('volatility', 0),
            'volume_ratio': latest_data.get('volume_ratio', 1),
            'data_points': len(rrs_df),
            'timestamp': latest_data['timestamp']
        }
        
        rankings.append(ranking_data)
    
    if not rankings:
        logger.warning("No valid rankings data available")
        return pd.DataFrame()
    
    # Create rankings DataFrame
    rankings_df = pd.DataFrame(rankings)
    
    # Sort by RRS score (descending)
    rankings_df = rankings_df.sort_values('current_rrs', ascending=False).reset_index(drop=True)
    
    # Add ranking position
    rankings_df['rank'] = range(1, len(rankings_df) + 1)
    
    # Add strength categories
    rankings_df['strength_category'] = pd.cut(
        rankings_df['current_rrs'], 
        bins=[-np.inf, -1, -0.5, 0.5, 1, np.inf],
        labels=['Very Weak', 'Weak', 'Neutral', 'Strong', 'Very Strong']
    )
    
    logger.info(f"Generated rankings for {len(rankings_df)} symbols")
    
    return rankings_df

def generate_trading_signals(rankings_df: pd.DataFrame, 
                           strong_threshold: float = 1.0,
                           weak_threshold: float = -1.0) -> pd.DataFrame:
    """Generate trading signals based on RRS analysis.
    
    Args:
        rankings_df: DataFrame with symbol rankings
        strong_threshold: RRS threshold for strong buy signals
        weak_threshold: RRS threshold for strong sell signals
    
    Returns:
        DataFrame with trading signals added
    """
    if rankings_df.empty:
        return rankings_df
    
    signals_df = rankings_df.copy()
    
    # Generate primary signals
    conditions = [
        (signals_df['current_rrs'] >= strong_threshold) & (signals_df['rrs_momentum'] > 0),
        (signals_df['current_rrs'] >= strong_threshold) & (signals_df['rrs_momentum'] <= 0),
        (signals_df['current_rrs'] <= weak_threshold) & (signals_df['rrs_momentum'] < 0),
        (signals_df['current_rrs'] <= weak_threshold) & (signals_df['rrs_momentum'] >= 0),
    ]
    
    choices = ['STRONG_BUY', 'BUY', 'STRONG_SELL', 'SELL']
    
    signals_df['primary_signal'] = np.select(conditions, choices, default='HOLD')
    
    # Generate confidence scores
    signals_df['signal_confidence'] = np.abs(signals_df['current_rrs']) * signals_df['outperformance_ratio']
    
    # Risk assessment
    signals_df['risk_level'] = pd.cut(
        signals_df['volatility'],
        bins=[0, 0.02, 0.05, 0.1, np.inf],
        labels=['Low', 'Medium', 'High', 'Very High']
    )
    
    # Volume confirmation
    signals_df['volume_confirmation'] = signals_df['volume_ratio'] > 1.2
    
    # Trend confirmation
    signals_df['trend_confirmation'] = np.where(
        signals_df['primary_signal'].isin(['STRONG_BUY', 'BUY']),
        signals_df['rrs_trend'] > 0,
        np.where(
            signals_df['primary_signal'].isin(['STRONG_SELL', 'SELL']),
            signals_df['rrs_trend'] < 0,
            True
        )
    )
    
    logger.info(f"Generated trading signals for {len(signals_df)} symbols")
    signal_counts = signals_df['primary_signal'].value_counts()
    logger.info(f"Signal distribution: {signal_counts.to_dict()}")
    
    return signals_df

if __name__ == "__main__":
    # Test the RRS calculator
    import logging
    logging.basicConfig(level=logging.INFO)
    
    print("Testing Binance RRS calculator...")
    
    # Create sample data for testing
    dates = pd.date_range('2024-01-01', periods=200, freq='H')
    np.random.seed(42)
    
    # Generate synthetic data for symbol and benchmark
    def create_test_data(name, trend_factor=0):
        returns = np.random.normal(trend_factor, 0.02, 200)
        prices = [50000]
        for r in returns:
            prices.append(prices[-1] * (1 + r))
        
        return pd.DataFrame({
            'timestamp': dates,
            'close': prices[1:],
            'log_return': returns,
            'volatility': np.abs(np.random.normal(0.02, 0.01, 200)),
            'volume_ratio': np.random.lognormal(0, 0.5, 200)
        })
    
    # Create test data (ETH outperforming BTC)
    symbol_df = create_test_data('ETH', trend_factor=0.0001)
    benchmark_df = create_test_data('BTC', trend_factor=0)
    
    # Test RRS calculation
    rrs_result = calculate_rrs(symbol_df, benchmark_df, 'ETHUSDT', 'BTCUSDT')
    
    if not rrs_result.empty:
        print(f"RRS calculation successful - {len(rrs_result)} data points")
        print("Sample RRS metrics:")
        print(rrs_result[['timestamp', 'differential_return', 'smoothed_rrs', 
                         'rrs_percentile', 'outperformance_ratio']].tail())
        
        # Test ranking (single symbol)
        rankings = rank_symbols_by_rrs({'ETHUSDT': rrs_result})
        print("\nRankings:")
        print(rankings)
        
        # Test signal generation
        signals = generate_trading_signals(rankings)
        print("\nTrading signals:")
        print(signals[['symbol', 'current_rrs', 'primary_signal', 'signal_confidence']])
        
    print("✅ Binance RRS calculator test complete")
