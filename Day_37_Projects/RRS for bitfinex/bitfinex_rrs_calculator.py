# bitfinex_rrs_calculator.py
import pandas as pd
import numpy as np
import logging
from typing import Dict
from bitfinex_rrs_config import RRS_SMOOTHING_WINDOW, MIN_DATA_POINTS, PROFESSIONAL_RISK_THRESHOLDS

# Setup logger for this module
logger = logging.getLogger(__name__)

def calculate_professional_rrs(symbol_df: pd.DataFrame, benchmark_df: pd.DataFrame, 
                              symbol_name: str, benchmark_name: str) -> pd.DataFrame:
    """
    Calculates professional Relative Rotation Strength (RRS) for Bitfinex institutional trading.

    Args:
        symbol_df: DataFrame containing processed symbol data with technical indicators
        benchmark_df: DataFrame containing processed benchmark data
        symbol_name: Name of the symbol being analyzed
        benchmark_name: Name of the benchmark symbol

    Returns:
        DataFrame with RRS calculations and professional trading metrics
    """
    if symbol_df.empty or benchmark_df.empty:
        logger.warning(f"Empty data provided for RRS calculation: {symbol_name} vs {benchmark_name}")
        return pd.DataFrame()

    if len(symbol_df) < MIN_DATA_POINTS or len(benchmark_df) < MIN_DATA_POINTS:
        logger.warning(f"Insufficient data points for professional RRS calculation: {symbol_name}")
        return pd.DataFrame()

    try:
        # Align timestamps for professional analysis
        symbol_df = symbol_df.set_index('timestamp')
        benchmark_df = benchmark_df.set_index('timestamp')
        
        # Find common timestamp range
        common_timestamps = symbol_df.index.intersection(benchmark_df.index)
        if len(common_timestamps) < MIN_DATA_POINTS:
            logger.warning(f"Insufficient overlapping data for {symbol_name} vs {benchmark_name}")
            return pd.DataFrame()

        # Align data on common timestamps
        symbol_aligned = symbol_df.loc[common_timestamps].copy()
        benchmark_aligned = benchmark_df.loc[common_timestamps].copy()

        logger.info(f"Calculating professional RRS for {symbol_name} vs {benchmark_name} "
                   f"with {len(common_timestamps)} aligned data points")

        # Professional RRS calculations
        rrs_df = symbol_aligned[['close', 'volume', 'returns_1', 'volatility']].copy()
        
        # Core RRS: Symbol returns minus benchmark returns
        rrs_df['symbol_returns'] = symbol_aligned['returns_1']
        rrs_df['benchmark_returns'] = benchmark_aligned['returns_1']
        rrs_df['raw_rrs'] = rrs_df['symbol_returns'] - rrs_df['benchmark_returns']
        
        # Professional smoothed RRS for institutional trading
        rrs_df['smoothed_rrs'] = rrs_df['raw_rrs'].rolling(
            window=RRS_SMOOTHING_WINDOW, min_periods=1
        ).mean()
        
        # Exponential smoothing for trend detection
        rrs_df['ema_rrs'] = rrs_df['raw_rrs'].ewm(span=RRS_SMOOTHING_WINDOW).mean()
        
        # Professional momentum calculations
        rrs_df['rrs_momentum'] = rrs_df['smoothed_rrs'].diff(5)  # 5-period momentum
        rrs_df['rrs_velocity'] = rrs_df['smoothed_rrs'].diff(1)  # Velocity (1-period change)
        rrs_df['rrs_acceleration'] = rrs_df['rrs_velocity'].diff(1)  # Acceleration
        
        # Multi-timeframe RRS for institutional analysis
        for window in [5, 10, 20, 50]:
            rrs_df[f'rrs_ma_{window}'] = rrs_df['smoothed_rrs'].rolling(window=window).mean()
            rrs_df[f'rrs_std_{window}'] = rrs_df['smoothed_rrs'].rolling(window=window).std()
            rrs_df[f'rrs_zscore_{window}'] = (rrs_df['smoothed_rrs'] - rrs_df[f'rrs_ma_{window}']) / rrs_df[f'rrs_std_{window}']
        
        # Professional relative performance metrics
        rrs_df['cumulative_rrs'] = rrs_df['raw_rrs'].cumsum()
        rrs_df['relative_outperformance'] = (rrs_df['cumulative_rrs'] > 0).astype(int)
        
        # Institutional volatility-adjusted RRS
        symbol_vol = symbol_aligned['volatility']
        benchmark_vol = benchmark_aligned['volatility']
        rrs_df['vol_adjusted_rrs'] = rrs_df['raw_rrs'] / np.sqrt(symbol_vol**2 + benchmark_vol**2)
        
        # Professional ranking metrics
        rrs_df['rrs_percentile_20'] = rrs_df['smoothed_rrs'].rolling(window=20).rank(pct=True)
        rrs_df['rrs_percentile_50'] = rrs_df['smoothed_rrs'].rolling(window=50).rank(pct=True)
        
        # Add metadata for professional reporting
        rrs_df['symbol'] = symbol_name
        rrs_df['benchmark'] = benchmark_name
        rrs_df['analysis_type'] = 'bitfinex_professional_rrs'
        
        # Current professional values (most recent)
        current_row = rrs_df.iloc[-1]
        rrs_df['current_rrs'] = current_row['smoothed_rrs']
        rrs_df['current_momentum'] = current_row['rrs_momentum']
        rrs_df['current_percentile'] = current_row['rrs_percentile_20']
        
        # Professional risk metrics for institutional trading
        rrs_df = calculate_professional_risk_metrics(rrs_df, symbol_aligned, benchmark_aligned)
        
        logger.info(f"Professional RRS calculation completed for {symbol_name}")
        logger.debug(f"Current professional RRS: {current_row['smoothed_rrs']:.4f}")
        
        return rrs_df.reset_index()
        
    except Exception as e:
        logger.error(f"Error calculating professional RRS for {symbol_name}: {e}")
        return pd.DataFrame()

def calculate_professional_risk_metrics(rrs_df: pd.DataFrame, symbol_df: pd.DataFrame, 
                                       benchmark_df: pd.DataFrame) -> pd.DataFrame:
    """Calculate professional risk metrics for institutional position sizing."""
    try:
        # Professional drawdown analysis
        cumulative_returns = (1 + rrs_df['symbol_returns']).cumprod()
        rolling_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - rolling_max) / rolling_max
        rrs_df['max_drawdown'] = drawdown.expanding().min()
        rrs_df['current_drawdown'] = drawdown
        
        # Professional beta calculation (institutional risk management)
        returns_window = 50
        symbol_returns = rrs_df['symbol_returns'].rolling(window=returns_window)
        benchmark_returns = rrs_df['benchmark_returns'].rolling(window=returns_window)
        
        covariance = symbol_returns.cov(benchmark_returns)
        benchmark_variance = benchmark_returns.var()
        rrs_df['beta'] = covariance / benchmark_variance
        
        # Professional alpha calculation
        risk_free_rate = 0.02 / 365  # Assume 2% annual risk-free rate
        rrs_df['alpha'] = rrs_df['symbol_returns'] - (risk_free_rate + rrs_df['beta'] * 
                                                     (rrs_df['benchmark_returns'] - risk_free_rate))
        
        # Institutional correlation metrics
        rrs_df['correlation'] = symbol_returns.corr(benchmark_returns)
        
        # Professional tracking error for institutional funds
        rrs_df['tracking_error'] = rrs_df['raw_rrs'].rolling(window=returns_window).std() * np.sqrt(365)
        
        # Information ratio for professional analysis
        rrs_df['information_ratio'] = (rrs_df['symbol_returns'].rolling(window=returns_window).mean() - 
                                      rrs_df['benchmark_returns'].rolling(window=returns_window).mean()) / rrs_df['tracking_error']
        
        # Professional VaR calculation (Value at Risk)
        confidence_level = 0.05  # 95% confidence
        rrs_df['var_5pct'] = rrs_df['symbol_returns'].rolling(window=returns_window).quantile(confidence_level)
        rrs_df['cvar_5pct'] = rrs_df['symbol_returns'].rolling(window=returns_window).apply(
            lambda x: x[x <= x.quantile(confidence_level)].mean()
        )
        
        return rrs_df
        
    except Exception as e:
        logger.error(f"Error calculating professional risk metrics: {e}")
        return rrs_df

def rank_symbols_by_professional_rrs(rrs_results: Dict[str, pd.DataFrame], 
                                    ranking_column: str = 'smoothed_rrs') -> pd.DataFrame:
    """
    Ranks symbols by professional RRS performance for institutional decision making.

    Args:
        rrs_results: Dictionary mapping symbol names to their RRS DataFrames
        ranking_column: Column name to use for ranking

    Returns:
        DataFrame with professional ranked results
    """
    if not rrs_results:
        logger.warning("No professional RRS results provided for ranking")
        return pd.DataFrame()

    try:
        ranking_data = []
        
        for symbol_name, rrs_df in rrs_results.items():
            if rrs_df.empty:
                continue
                
            # Get most recent professional metrics
            latest_row = rrs_df.iloc[-1]
            
            # Professional performance metrics
            ranking_record = {
                'symbol': symbol_name,
                'current_rrs': latest_row.get(ranking_column, 0),
                'rrs_momentum': latest_row.get('rrs_momentum', 0),
                'rrs_percentile': latest_row.get('rrs_percentile_20', 0.5),
                'volatility': latest_row.get('volatility', 0),
                'beta': latest_row.get('beta', 1.0),
                'alpha': latest_row.get('alpha', 0),
                'correlation': latest_row.get('correlation', 0),
                'max_drawdown': latest_row.get('max_drawdown', 0),
                'tracking_error': latest_row.get('tracking_error', 0),
                'information_ratio': latest_row.get('information_ratio', 0),
                'var_5pct': latest_row.get('var_5pct', 0),
                'last_price': latest_row.get('close', 0),
                'volume': latest_row.get('volume', 0),
                'exchange': 'Bitfinex'
            }
            
            # Professional trend analysis
            recent_rrs = rrs_df[ranking_column].tail(10)
            ranking_record['trend_consistency'] = (recent_rrs > recent_rrs.mean()).sum() / len(recent_rrs)
            ranking_record['rrs_stability'] = 1 - (recent_rrs.std() / abs(recent_rrs.mean())) if recent_rrs.mean() != 0 else 0
            
            # Professional momentum score
            momentum_score = (latest_row.get('rrs_momentum', 0) * 0.4 + 
                            latest_row.get('rrs_velocity', 0) * 0.3 + 
                            latest_row.get('rrs_acceleration', 0) * 0.3)
            ranking_record['momentum_score'] = momentum_score
            
            ranking_data.append(ranking_record)

        if not ranking_data:
            logger.warning("No valid professional ranking data generated")
            return pd.DataFrame()

        # Create professional ranking DataFrame
        ranking_df = pd.DataFrame(ranking_data)
        
        # Professional composite scoring
        ranking_df['professional_score'] = (
            ranking_df['current_rrs'] * 0.3 +
            ranking_df['momentum_score'] * 0.2 +
            ranking_df['rrs_stability'] * 0.2 +
            ranking_df['information_ratio'].fillna(0) * 0.15 +
            ranking_df['trend_consistency'] * 0.15
        )
        
        # Sort by professional score and assign ranks
        ranking_df = ranking_df.sort_values('professional_score', ascending=False).reset_index(drop=True)
        ranking_df['rank'] = range(1, len(ranking_df) + 1)
        
        logger.info(f"Professional ranking completed for {len(ranking_df)} symbols")
        
        return ranking_df
        
    except Exception as e:
        logger.error(f"Error in professional symbol ranking: {e}")
        return pd.DataFrame()

def generate_professional_trading_signals(ranking_df: pd.DataFrame) -> pd.DataFrame:
    """
    Generates professional trading signals for institutional Bitfinex trading.

    Args:
        ranking_df: DataFrame with professional ranked symbols

    Returns:
        DataFrame with added professional trading signal columns
    """
    if ranking_df.empty:
        logger.warning("Empty ranking DataFrame for professional signal generation")
        return ranking_df

    try:
        signals_df = ranking_df.copy()
        
        # Initialize professional signal columns
        signals_df['primary_signal'] = 'HOLD'
        signals_df['signal_confidence'] = 0.5
        signals_df['risk_level'] = 'Medium'
        signals_df['position_size_suggestion'] = 1.0
        signals_df['professional_grade'] = 'Standard'
        
        # Professional signal generation logic
        for i, row in signals_df.iterrows():
            rrs_score = row['current_rrs']
            momentum = row['momentum_score']
            volatility = row['volatility']
            beta = row['beta']
            max_dd = abs(row['max_drawdown'])
            trend_consistency = row['trend_consistency']
            
            # Professional BUY signals
            if (rrs_score > PROFESSIONAL_RISK_THRESHOLDS['rrs_strong_buy'] and 
                momentum > 0.001 and 
                trend_consistency > 0.6):
                
                signals_df.at[i, 'primary_signal'] = 'STRONG_BUY'
                signals_df.at[i, 'signal_confidence'] = min(0.95, 0.6 + trend_consistency * 0.3 + min(rrs_score * 10, 0.1))
                signals_df.at[i, 'professional_grade'] = 'Institutional'
                
            elif (rrs_score > PROFESSIONAL_RISK_THRESHOLDS['rrs_buy'] and 
                  momentum > 0 and 
                  trend_consistency > 0.5):
                
                signals_df.at[i, 'primary_signal'] = 'BUY'
                signals_df.at[i, 'signal_confidence'] = min(0.85, 0.5 + trend_consistency * 0.25 + min(rrs_score * 8, 0.1))
                signals_df.at[i, 'professional_grade'] = 'Professional'
                
            # Professional SELL signals
            elif (rrs_score < PROFESSIONAL_RISK_THRESHOLDS['rrs_strong_sell'] and 
                  momentum < -0.001 and 
                  trend_consistency < 0.4):
                
                signals_df.at[i, 'primary_signal'] = 'STRONG_SELL'
                signals_df.at[i, 'signal_confidence'] = min(0.95, 0.6 - trend_consistency * 0.2 + min(abs(rrs_score) * 10, 0.1))
                signals_df.at[i, 'professional_grade'] = 'Institutional'
                
            elif (rrs_score < PROFESSIONAL_RISK_THRESHOLDS['rrs_sell'] and 
                  momentum < 0 and 
                  trend_consistency < 0.5):
                
                signals_df.at[i, 'primary_signal'] = 'SELL'
                signals_df.at[i, 'signal_confidence'] = min(0.85, 0.5 - trend_consistency * 0.15 + min(abs(rrs_score) * 8, 0.1))
                signals_df.at[i, 'professional_grade'] = 'Professional'
            
            # Professional risk level determination
            if volatility > PROFESSIONAL_RISK_THRESHOLDS['high_volatility'] or max_dd > 0.15:
                signals_df.at[i, 'risk_level'] = 'High'
                signals_df.at[i, 'position_size_suggestion'] = 0.5  # Reduce size for high risk
            elif volatility < PROFESSIONAL_RISK_THRESHOLDS['low_volatility'] and max_dd < 0.05:
                signals_df.at[i, 'risk_level'] = 'Low'
                signals_df.at[i, 'position_size_suggestion'] = 1.5  # Increase size for low risk
            else:
                signals_df.at[i, 'risk_level'] = 'Medium'
                signals_df.at[i, 'position_size_suggestion'] = 1.0
            
            # Professional beta adjustment
            if beta > 1.5:  # High beta - more volatile than market
                signals_df.at[i, 'position_size_suggestion'] *= 0.8
            elif beta < 0.5:  # Low beta - less volatile than market
                signals_df.at[i, 'position_size_suggestion'] *= 1.2

        # Professional signal distribution summary
        signal_counts = signals_df['primary_signal'].value_counts()
        logger.info(f"Professional signal distribution: {dict(signal_counts)}")
        
        # Add professional timestamp
        signals_df['signal_timestamp'] = pd.Timestamp.utcnow()
        
        return signals_df
        
    except Exception as e:
        logger.error(f"Error generating professional trading signals: {e}")
        return ranking_df

if __name__ == "__main__":
    # Test professional RRS calculator
    logging.basicConfig(level=logging.INFO)
    
    print("Testing Bitfinex professional RRS calculator...")
    
    # Create test data for professional analysis
    dates = pd.date_range('2024-01-01', periods=100, freq='H')
    
    # Symbol test data
    symbol_data = pd.DataFrame({
        'timestamp': dates,
        'close': np.random.uniform(45000, 55000, 100),
        'volume': np.random.uniform(1, 100, 100),
        'returns_1': np.random.normal(0, 0.02, 100),
        'volatility': np.random.uniform(0.01, 0.05, 100)
    })
    
    # Benchmark test data
    benchmark_data = pd.DataFrame({
        'timestamp': dates,
        'close': np.random.uniform(40000, 50000, 100),
        'volume': np.random.uniform(1, 100, 100),
        'returns_1': np.random.normal(0, 0.015, 100),
        'volatility': np.random.uniform(0.01, 0.04, 100)
    })
    
    # Test professional RRS calculation
    rrs_result = calculate_professional_rrs(symbol_data, benchmark_data, 'ETHUSD', 'BTCUSD')
    print(f"Professional RRS calculation result shape: {rrs_result.shape}")
    
    if not rrs_result.empty:
        print(f"Current professional RRS: {rrs_result['current_rrs'].iloc[-1]:.4f}")
        
        # Test professional ranking
        rrs_results = {'ETHUSD': rrs_result}
        ranking_result = rank_symbols_by_professional_rrs(rrs_results)
        print(f"Professional ranking result shape: {ranking_result.shape}")
        
        # Test professional signal generation
        signals_result = generate_professional_trading_signals(ranking_result)
        print(f"Professional signals result shape: {signals_result.shape}")
        
        if not signals_result.empty:
            print(f"Professional signal: {signals_result['primary_signal'].iloc[0]}")
            print(f"Signal confidence: {signals_result['signal_confidence'].iloc[0]:.2f}")
    
    print("✅ Bitfinex professional RRS calculator test complete")
