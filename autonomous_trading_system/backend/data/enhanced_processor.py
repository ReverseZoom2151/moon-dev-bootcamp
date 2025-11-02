"""
Enhanced Data Processing Utilities
Advanced data preprocessing and feature engineering functions
Based on Day 26 enhanced functions implementation
"""

import logging
import numpy as np
import pandas as pd
import warnings
import pandas_ta as pta
from typing import Dict, Any, List
from scipy.signal import find_peaks
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_regression

warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class EnhancedDataProcessor:
    """Advanced data processing and feature engineering"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.scalers = {}
        self.feature_selectors = {}
        
        # Processing parameters
        self.outlier_threshold = self.config.get('outlier_threshold', 3.0)
        self.smoothing_window = self.config.get('smoothing_window', 5)
        self.feature_selection_k = self.config.get('feature_selection_k', 20)
        
        logger.info("🔧 Enhanced Data Processor initialized")
    
    def clean_data(self, df: pd.DataFrame, method: str = 'comprehensive') -> pd.DataFrame:
        """Comprehensive data cleaning"""
        try:
            cleaned_df = df.copy()
            
            if method == 'comprehensive':
                # Remove duplicates
                cleaned_df = self._remove_duplicates(cleaned_df)
                
                # Handle missing values
                cleaned_df = self._handle_missing_values(cleaned_df)
                
                # Remove outliers
                cleaned_df = self._remove_outliers(cleaned_df)
                
                # Fix data types
                cleaned_df = self._fix_data_types(cleaned_df)
                
                # Validate OHLCV data
                if self._is_ohlcv_data(cleaned_df):
                    cleaned_df = self._validate_ohlcv(cleaned_df)
            
            elif method == 'basic':
                cleaned_df = self._handle_missing_values(cleaned_df)
                cleaned_df = self._fix_data_types(cleaned_df)
            
            logger.info(f"Data cleaning complete. Rows: {len(df)} → {len(cleaned_df)}")
            return cleaned_df
            
        except Exception as e:
            logger.error(f"Error cleaning data: {e}")
            return df
    
    def _remove_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove duplicate rows"""
        initial_count = len(df)
        
        # Remove exact duplicates
        df_clean = df.drop_duplicates()
        
        # Remove duplicates based on timestamp if available
        if 'timestamp' in df.columns:
            df_clean = df_clean.drop_duplicates(subset=['timestamp'])
        elif 'time' in df.columns:
            df_clean = df_clean.drop_duplicates(subset=['time'])
        
        removed_count = initial_count - len(df_clean)
        if removed_count > 0:
            logger.info(f"Removed {removed_count} duplicate rows")
        
        return df_clean
    
    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values intelligently"""
        df_clean = df.copy()
        
        for column in df_clean.columns:
            missing_count = df_clean[column].isnull().sum()
            missing_pct = missing_count / len(df_clean)
            
            if missing_pct > 0.5:
                # Drop columns with >50% missing data
                logger.warning(f"Dropping column {column} with {missing_pct:.1%} missing data")
                df_clean = df_clean.drop(columns=[column])
                continue
            
            if missing_count > 0:
                if df_clean[column].dtype in ['float64', 'int64']:
                    # Forward fill for price/volume data
                    if column.lower() in ['open', 'high', 'low', 'close', 'volume']:
                        df_clean[column] = df_clean[column].fillna(method='ffill')
                    else:
                        # Use median for other numeric columns
                        df_clean[column] = df_clean[column].fillna(df_clean[column].median())
                else:
                    # Forward fill for categorical data
                    df_clean[column] = df_clean[column].fillna(method='ffill')
        
        # Drop any remaining rows with NaN values
        initial_count = len(df_clean)
        df_clean = df_clean.dropna()
        removed_count = initial_count - len(df_clean)
        
        if removed_count > 0:
            logger.info(f"Removed {removed_count} rows with missing values")
        
        return df_clean
    
    def _remove_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove statistical outliers"""
        df_clean = df.copy()
        numeric_columns = df_clean.select_dtypes(include=[np.number]).columns
        
        for column in numeric_columns:
            if column.lower() in ['timestamp', 'time']:
                continue
            
            # Use IQR method for outlier detection
            Q1 = df_clean[column].quantile(0.25)
            Q3 = df_clean[column].quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            # For price data, only remove extreme outliers
            if column.lower() in ['open', 'high', 'low', 'close']:
                lower_bound = Q1 - 3 * IQR
                upper_bound = Q3 + 3 * IQR
            
            outlier_mask = (df_clean[column] < lower_bound) | (df_clean[column] > upper_bound)
            outlier_count = outlier_mask.sum()
            
            if outlier_count > 0:
                logger.info(f"Removing {outlier_count} outliers from {column}")
                df_clean = df_clean[~outlier_mask]
        
        return df_clean
    
    def _fix_data_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fix and optimize data types"""
        df_clean = df.copy()
        
        # Convert timestamp columns
        for col in ['timestamp', 'time', 'date']:
            if col in df_clean.columns:
                df_clean[col] = pd.to_datetime(df_clean[col], errors='coerce')
        
        # Optimize numeric columns
        numeric_columns = df_clean.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            if df_clean[col].dtype == 'float64':
                # Try to downcast to float32 if possible
                if df_clean[col].min() >= np.finfo(np.float32).min and df_clean[col].max() <= np.finfo(np.float32).max:
                    df_clean[col] = df_clean[col].astype(np.float32)
            elif df_clean[col].dtype == 'int64':
                # Try to downcast integers
                if df_clean[col].min() >= np.iinfo(np.int32).min and df_clean[col].max() <= np.iinfo(np.int32).max:
                    df_clean[col] = df_clean[col].astype(np.int32)
        
        return df_clean
    
    def _is_ohlcv_data(self, df: pd.DataFrame) -> bool:
        """Check if dataframe contains OHLCV data"""
        required_columns = ['open', 'high', 'low', 'close']
        return all(col in df.columns for col in required_columns)
    
    def _validate_ohlcv(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate and fix OHLCV data consistency"""
        df_clean = df.copy()
        
        # Ensure high >= max(open, close) and low <= min(open, close)
        df_clean['high'] = np.maximum(df_clean['high'], np.maximum(df_clean['open'], df_clean['close']))
        df_clean['low'] = np.minimum(df_clean['low'], np.minimum(df_clean['open'], df_clean['close']))
        
        # Remove rows where high < low (impossible)
        invalid_mask = df_clean['high'] < df_clean['low']
        invalid_count = invalid_mask.sum()
        
        if invalid_count > 0:
            logger.warning(f"Removing {invalid_count} rows with invalid OHLC data")
            df_clean = df_clean[~invalid_mask]
        
        # Ensure volume is non-negative
        if 'volume' in df_clean.columns:
            df_clean['volume'] = np.maximum(df_clean['volume'], 0)
        
        return df_clean
    
    def create_advanced_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create advanced technical and statistical features"""
        try:
            features_df = df.copy()
            
            if not self._is_ohlcv_data(features_df):
                logger.warning("OHLCV data required for advanced features")
                return features_df
            
            # Price-based features
            features_df = self._add_price_features(features_df)
            
            # Volume features
            features_df = self._add_volume_features(features_df)
            
            # Volatility features
            features_df = self._add_volatility_features(features_df)
            
            # Momentum features
            features_df = self._add_momentum_features(features_df)
            
            # Statistical features
            features_df = self._add_statistical_features(features_df)
            
            # Pattern features
            features_df = self._add_pattern_features(features_df)
            
            # Market microstructure features
            features_df = self._add_microstructure_features(features_df)
            
            logger.info(f"Created {len(features_df.columns) - len(df.columns)} advanced features")
            return features_df
            
        except Exception as e:
            logger.error(f"Error creating advanced features: {e}")
            return df
    
    def _add_price_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add price-based features"""
        # Basic price features
        df['hl_avg'] = (df['high'] + df['low']) / 2
        df['hlc_avg'] = (df['high'] + df['low'] + df['close']) / 3
        df['ohlc_avg'] = (df['open'] + df['high'] + df['low'] + df['close']) / 4
        
        # Price ranges
        df['price_range'] = df['high'] - df['low']
        df['price_range_pct'] = df['price_range'] / df['close']
        df['body_size'] = abs(df['close'] - df['open'])
        df['body_size_pct'] = df['body_size'] / df['close']
        
        # Upper and lower shadows
        df['upper_shadow'] = df['high'] - np.maximum(df['open'], df['close'])
        df['lower_shadow'] = np.minimum(df['open'], df['close']) - df['low']
        df['upper_shadow_pct'] = df['upper_shadow'] / df['close']
        df['lower_shadow_pct'] = df['lower_shadow'] / df['close']
        
        # Price position within range
        df['close_position'] = (df['close'] - df['low']) / (df['high'] - df['low'])
        
        return df
    
    def _add_volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add volume-based features"""
        if 'volume' not in df.columns:
            return df
        
        # Volume moving averages
        df['volume_sma_10'] = df['volume'].rolling(window=10).mean()
        df['volume_sma_20'] = df['volume'].rolling(window=20).mean()
        
        # Volume ratios
        df['volume_ratio_10'] = df['volume'] / df['volume_sma_10']
        df['volume_ratio_20'] = df['volume'] / df['volume_sma_20']
        
        # Volume-price features
        df['vwap'] = (df['volume'] * df['hlc_avg']).cumsum() / df['volume'].cumsum()
        df['volume_price_trend'] = df['volume'] * (df['close'] - df['open'])
        
        # On-Balance Volume
        df['price_change'] = df['close'].diff()
        df['obv'] = (np.sign(df['price_change']) * df['volume']).cumsum()
        
        return df
    
    def _add_volatility_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add volatility-based features"""
        # Returns
        df['returns'] = df['close'].pct_change()
        df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
        
        # Rolling volatility
        for window in [5, 10, 20]:
            df[f'volatility_{window}'] = df['returns'].rolling(window=window).std()
            df[f'log_volatility_{window}'] = df['log_returns'].rolling(window=window).std()
        
        # Parkinson volatility (using high-low)
        df['parkinson_vol'] = np.sqrt(
            (1 / (4 * np.log(2))) * np.log(df['high'] / df['low']) ** 2
        ).rolling(window=20).mean()
        
        # Garman-Klass volatility
        df['gk_vol'] = np.sqrt(
            0.5 * np.log(df['high'] / df['low']) ** 2 - 
            (2 * np.log(2) - 1) * np.log(df['close'] / df['open']) ** 2
        ).rolling(window=20).mean()
        
        return df
    
    def _add_momentum_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add momentum-based features"""
        # Rate of change
        for period in [5, 10, 20]:
            df[f'roc_{period}'] = df['close'].pct_change(periods=period)
        
        # Momentum
        for period in [5, 10, 20]:
            df[f'momentum_{period}'] = df['close'] - df['close'].shift(period)
        
        # Acceleration (second derivative)
        df['acceleration'] = df['returns'].diff()
        
        # Relative strength
        gains = df['returns'].where(df['returns'] > 0, 0)
        losses = -df['returns'].where(df['returns'] < 0, 0)
        
        for period in [14, 21]:
            avg_gain = gains.rolling(window=period).mean()
            avg_loss = losses.rolling(window=period).mean()
            rs = avg_gain / avg_loss
            df[f'rsi_{period}'] = 100 - (100 / (1 + rs))
        
        return df
    
    def _add_statistical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add statistical features"""
        # Rolling statistics
        for window in [10, 20, 50]:
            df[f'mean_{window}'] = df['close'].rolling(window=window).mean()
            df[f'std_{window}'] = df['close'].rolling(window=window).std()
            df[f'skew_{window}'] = df['returns'].rolling(window=window).skew()
            df[f'kurt_{window}'] = df['returns'].rolling(window=window).kurt()
            
            # Percentile features
            df[f'percentile_25_{window}'] = df['close'].rolling(window=window).quantile(0.25)
            df[f'percentile_75_{window}'] = df['close'].rolling(window=window).quantile(0.75)
            
            # Z-score
            df[f'zscore_{window}'] = (df['close'] - df[f'mean_{window}']) / df[f'std_{window}']
        
        # Autocorrelation
        for lag in [1, 5, 10]:
            df[f'autocorr_{lag}'] = df['returns'].rolling(window=20).apply(
                lambda x: x.autocorr(lag=lag) if len(x) > lag else np.nan
            )
        
        return df
    
    def _add_pattern_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add pattern recognition features"""
        try:
            # Candlestick patterns using pandas_ta
            df['doji'] = pta.cdl_doji(df['open'], df['high'], df['low'], df['close'])
            df['hammer'] = pta.cdl_hammer(df['open'], df['high'], df['low'], df['close'])
            df['engulfing'] = pta.cdl_engulfing(df['open'], df['high'], df['low'], df['close'])
            df['morning_star'] = pta.cdl_morningstar(df['open'], df['high'], df['low'], df['close'])
            df['evening_star'] = pta.cdl_eveningstar(df['open'], df['high'], df['low'], df['close'])
            
            # Support and resistance levels
            df = self._add_support_resistance(df)
            
            # Trend patterns
            df = self._add_trend_patterns(df)
            
        except Exception as e:
            logger.warning(f"Error adding pattern features: {e}")
        
        return df
    
    def _add_support_resistance(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add support and resistance levels"""
        # Find local peaks and troughs
        high_peaks, _ = find_peaks(df['high'], distance=5)
        low_peaks, _ = find_peaks(-df['low'], distance=5)
        
        # Create support/resistance arrays
        df['resistance_level'] = np.nan
        df['support_level'] = np.nan
        
        if len(high_peaks) > 0:
            df.loc[high_peaks, 'resistance_level'] = df.loc[high_peaks, 'high']
        if len(low_peaks) > 0:
            df.loc[low_peaks, 'support_level'] = df.loc[low_peaks, 'low']
        
        # Forward fill levels
        df['resistance_level'] = df['resistance_level'].fillna(method='ffill')
        df['support_level'] = df['support_level'].fillna(method='ffill')
        
        # Distance to levels
        df['dist_to_resistance'] = (df['resistance_level'] - df['close']) / df['close']
        df['dist_to_support'] = (df['close'] - df['support_level']) / df['close']
        
        return df
    
    def _add_trend_patterns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add trend pattern features"""
        # Higher highs and lower lows
        df['higher_high'] = (df['high'] > df['high'].shift(1)).astype(int)
        df['lower_low'] = (df['low'] < df['low'].shift(1)).astype(int)
        
        # Consecutive patterns
        df['consecutive_up'] = (df['close'] > df['close'].shift(1)).astype(int)
        df['consecutive_down'] = (df['close'] < df['close'].shift(1)).astype(int)
        
        # Count consecutive occurrences
        df['up_streak'] = df['consecutive_up'].groupby(
            (df['consecutive_up'] != df['consecutive_up'].shift()).cumsum()
        ).cumsum()
        
        df['down_streak'] = df['consecutive_down'].groupby(
            (df['consecutive_down'] != df['consecutive_down'].shift()).cumsum()
        ).cumsum()
        
        return df
    
    def _add_microstructure_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add market microstructure features"""
        # Bid-ask spread proxy (using high-low)
        df['spread_proxy'] = (df['high'] - df['low']) / df['close']
        
        # Price impact proxy
        if 'volume' in df.columns:
            df['price_impact'] = abs(df['returns']) / np.log1p(df['volume'])
        
        # Amihud illiquidity measure
        if 'volume' in df.columns:
            df['amihud_illiq'] = abs(df['returns']) / (df['volume'] * df['close'])
        
        # Roll measure (bid-ask spread estimator)
        df['roll_measure'] = 2 * np.sqrt(-df['returns'].rolling(window=20).cov(df['returns'].shift(1)))
        
        return df
    
    def scale_features(self, df: pd.DataFrame, method: str = 'standard', fit: bool = True) -> pd.DataFrame:
        """Scale features using various methods"""
        try:
            numeric_columns = df.select_dtypes(include=[np.number]).columns
            scaled_df = df.copy()
            
            if method not in self.scalers or fit:
                if method == 'standard':
                    self.scalers[method] = StandardScaler()
                elif method == 'minmax':
                    self.scalers[method] = MinMaxScaler()
                elif method == 'robust':
                    self.scalers[method] = RobustScaler()
                else:
                    logger.warning(f"Unknown scaling method: {method}")
                    return df
                
                # Fit the scaler
                self.scalers[method].fit(df[numeric_columns])
            
            # Transform the data
            scaled_values = self.scalers[method].transform(df[numeric_columns])
            scaled_df[numeric_columns] = scaled_values
            
            logger.info(f"Scaled features using {method} method")
            return scaled_df
            
        except Exception as e:
            logger.error(f"Error scaling features: {e}")
            return df
    
    def select_features(self, df: pd.DataFrame, target_column: str, method: str = 'kbest', k: int = None) -> pd.DataFrame:
        """Select best features using various methods"""
        try:
            if target_column not in df.columns:
                logger.error(f"Target column {target_column} not found")
                return df
            
            k = k or self.feature_selection_k
            feature_columns = [col for col in df.columns if col != target_column and df[col].dtype in [np.number]]
            
            if len(feature_columns) <= k:
                return df
            
            X = df[feature_columns]
            y = df[target_column]
            
            # Remove any remaining NaN values
            mask = ~(X.isnull().any(axis=1) | y.isnull())
            X_clean = X[mask]
            y_clean = y[mask]
            
            if method == 'kbest':
                if method not in self.feature_selectors:
                    self.feature_selectors[method] = SelectKBest(score_func=f_regression, k=k)
                    self.feature_selectors[method].fit(X_clean, y_clean)
                
                selected_features = X_clean.columns[self.feature_selectors[method].get_support()]
                
            elif method == 'pca':
                if method not in self.feature_selectors:
                    self.feature_selectors[method] = PCA(n_components=k)
                    self.feature_selectors[method].fit(X_clean)
                
                # Transform features
                X_pca = self.feature_selectors[method].transform(X_clean)
                pca_columns = [f'pca_{i}' for i in range(X_pca.shape[1])]
                
                # Create new dataframe with PCA features
                result_df = df[[target_column]].copy()
                pca_df = pd.DataFrame(X_pca, columns=pca_columns, index=X_clean.index)
                result_df = pd.concat([result_df, pca_df], axis=1)
                
                logger.info(f"Selected {len(pca_columns)} PCA components")
                return result_df
            
            else:
                logger.warning(f"Unknown feature selection method: {method}")
                return df
            
            # Return dataframe with selected features
            result_df = df[[target_column] + list(selected_features)].copy()
            logger.info(f"Selected {len(selected_features)} features using {method}")
            return result_df
            
        except Exception as e:
            logger.error(f"Error selecting features: {e}")
            return df
    
    def create_lagged_features(self, df: pd.DataFrame, columns: List[str], lags: List[int]) -> pd.DataFrame:
        """Create lagged features for time series analysis"""
        try:
            lagged_df = df.copy()
            
            for column in columns:
                if column not in df.columns:
                    continue
                
                for lag in lags:
                    lagged_df[f'{column}_lag_{lag}'] = df[column].shift(lag)
            
            logger.info(f"Created lagged features for {len(columns)} columns with lags {lags}")
            return lagged_df
            
        except Exception as e:
            logger.error(f"Error creating lagged features: {e}")
            return df
    
    def create_rolling_features(self, df: pd.DataFrame, columns: List[str], windows: List[int], functions: List[str]) -> pd.DataFrame:
        """Create rolling window features"""
        try:
            rolling_df = df.copy()
            
            for column in columns:
                if column not in df.columns:
                    continue
                
                for window in windows:
                    for func in functions:
                        if func == 'mean':
                            rolling_df[f'{column}_rolling_{func}_{window}'] = df[column].rolling(window=window).mean()
                        elif func == 'std':
                            rolling_df[f'{column}_rolling_{func}_{window}'] = df[column].rolling(window=window).std()
                        elif func == 'min':
                            rolling_df[f'{column}_rolling_{func}_{window}'] = df[column].rolling(window=window).min()
                        elif func == 'max':
                            rolling_df[f'{column}_rolling_{func}_{window}'] = df[column].rolling(window=window).max()
                        elif func == 'median':
                            rolling_df[f'{column}_rolling_{func}_{window}'] = df[column].rolling(window=window).median()
                        elif func == 'skew':
                            rolling_df[f'{column}_rolling_{func}_{window}'] = df[column].rolling(window=window).skew()
                        elif func == 'kurt':
                            rolling_df[f'{column}_rolling_{func}_{window}'] = df[column].rolling(window=window).kurt()
            
            logger.info(f"Created rolling features for {len(columns)} columns")
            return rolling_df
            
        except Exception as e:
            logger.error(f"Error creating rolling features: {e}")
            return df
    
    def detect_regime_changes(self, df: pd.DataFrame, column: str = 'close', method: str = 'volatility') -> pd.DataFrame:
        """Detect regime changes in time series"""
        try:
            regime_df = df.copy()
            
            if method == 'volatility':
                # Volatility-based regime detection
                returns = df[column].pct_change()
                volatility = returns.rolling(window=20).std()
                
                # Use quantiles to define regimes
                low_vol_threshold = volatility.quantile(0.33)
                high_vol_threshold = volatility.quantile(0.67)
                
                regime_df['volatility_regime'] = pd.cut(
                    volatility,
                    bins=[-np.inf, low_vol_threshold, high_vol_threshold, np.inf],
                    labels=['low_vol', 'medium_vol', 'high_vol']
                )
            
            elif method == 'trend':
                # Trend-based regime detection
                sma_short = df[column].rolling(window=20).mean()
                sma_long = df[column].rolling(window=50).mean()
                
                regime_df['trend_regime'] = np.where(
                    sma_short > sma_long, 'uptrend',
                    np.where(sma_short < sma_long, 'downtrend', 'sideways')
                )
            
            logger.info(f"Detected regime changes using {method} method")
            return regime_df
            
        except Exception as e:
            logger.error(f"Error detecting regime changes: {e}")
            return df
    
    def get_processing_summary(self, original_df: pd.DataFrame, processed_df: pd.DataFrame) -> Dict[str, Any]:
        """Get summary of data processing operations"""
        return {
            'original_shape': original_df.shape,
            'processed_shape': processed_df.shape,
            'features_added': processed_df.shape[1] - original_df.shape[1],
            'rows_removed': original_df.shape[0] - processed_df.shape[0],
            'memory_usage_mb': processed_df.memory_usage(deep=True).sum() / 1024 / 1024,
            'missing_values': processed_df.isnull().sum().sum(),
            'numeric_columns': len(processed_df.select_dtypes(include=[np.number]).columns),
            'categorical_columns': len(processed_df.select_dtypes(include=['object', 'category']).columns)
        } 