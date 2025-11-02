"""
ML-based Technical Indicator Optimizer
Automatically selects optimal technical indicators using machine learning
Based on Day 33 ML indicator analysis implementation
"""

import logging
import numpy as np
import pandas as pd
import pandas_ta as talib
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime
from dataclasses import dataclass
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectKBest, f_regression, RFE
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from strategies.base_strategy import BaseStrategy, StrategySignal, SignalAction, TechnicalIndicatorMixin

logger = logging.getLogger(__name__)

@dataclass
class IndicatorImportance:
    """Importance score for a technical indicator"""
    name: str
    importance: float
    r2_score: float
    mse: float
    rank: int
    category: str


class TechnicalIndicatorLibrary:
    """Comprehensive library of technical indicators"""
    
    def __init__(self):
        self.indicators = {
            # Trend Indicators
            'SMA_10': lambda df: talib.SMA(df['close'], timeperiod=10),
            'SMA_20': lambda df: talib.SMA(df['close'], timeperiod=20),
            'SMA_50': lambda df: talib.SMA(df['close'], timeperiod=50),
            'EMA_10': lambda df: talib.EMA(df['close'], timeperiod=10),
            'EMA_20': lambda df: talib.EMA(df['close'], timeperiod=20),
            'EMA_50': lambda df: talib.EMA(df['close'], timeperiod=50),
            'WMA_10': lambda df: talib.WMA(df['close'], timeperiod=10),
            'WMA_20': lambda df: talib.WMA(df['close'], timeperiod=20),
            'LINEARREG': lambda df: talib.LINEARREG(df['close'], timeperiod=14),
            'TSF': lambda df: talib.TSF(df['close'], timeperiod=14),
            
            # Momentum Indicators
            'RSI_14': lambda df: talib.RSI(df['close'], timeperiod=14),
            'RSI_21': lambda df: talib.RSI(df['close'], timeperiod=21),
            'MACD': lambda df: talib.MACD(df['close'])[0],
            'MACD_SIGNAL': lambda df: talib.MACD(df['close'])[1],
            'MACD_HIST': lambda df: talib.MACD(df['close'])[2],
            'STOCH_K': lambda df: talib.STOCH(df['high'], df['low'], df['close'])[0],
            'STOCH_D': lambda df: talib.STOCH(df['high'], df['low'], df['close'])[1],
            'STOCHRSI_K': lambda df: talib.STOCHRSI(df['close'])[0],
            'STOCHRSI_D': lambda df: talib.STOCHRSI(df['close'])[1],
            'CCI': lambda df: talib.CCI(df['high'], df['low'], df['close'], timeperiod=14),
            'MOM': lambda df: talib.MOM(df['close'], timeperiod=10),
            'ROC': lambda df: talib.ROC(df['close'], timeperiod=10),
            'WILLR': lambda df: talib.WILLR(df['high'], df['low'], df['close'], timeperiod=14),
            
            # Volatility Indicators
            'ATR': lambda df: talib.ATR(df['high'], df['low'], df['close'], timeperiod=14),
            'NATR': lambda df: talib.NATR(df['high'], df['low'], df['close'], timeperiod=14),
            'TRANGE': lambda df: talib.TRANGE(df['high'], df['low'], df['close']),
            'VAR': lambda df: talib.VAR(df['close'], timeperiod=5),
            'STDDEV': lambda df: talib.STDDEV(df['close'], timeperiod=5),
            
            # Volume Indicators
            'OBV': lambda df: talib.OBV(df['close'], df['volume']),
            'AD': lambda df: talib.AD(df['high'], df['low'], df['close'], df['volume']),
            'ADOSC': lambda df: talib.ADOSC(df['high'], df['low'], df['close'], df['volume']),
            
            # Overlap Studies
            'BBANDS_UPPER': lambda df: talib.BBANDS(df['close'])[0],
            'BBANDS_MIDDLE': lambda df: talib.BBANDS(df['close'])[1],
            'BBANDS_LOWER': lambda df: talib.BBANDS(df['close'])[2],
            'SAR': lambda df: talib.SAR(df['high'], df['low']),
            'TEMA': lambda df: talib.TEMA(df['close'], timeperiod=30),
            'TRIMA': lambda df: talib.TRIMA(df['close'], timeperiod=30),
            'KAMA': lambda df: talib.KAMA(df['close'], timeperiod=30),
            
            # Pattern Recognition (selected)
            'CDLDOJI': lambda df: talib.CDLDOJI(df['open'], df['high'], df['low'], df['close']),
            'CDLHAMMER': lambda df: talib.CDLHAMMER(df['open'], df['high'], df['low'], df['close']),
            'CDLENGULFING': lambda df: talib.CDLENGULFING(df['open'], df['high'], df['low'], df['close']),
            'CDLDOJISTAR': lambda df: talib.CDLDOJISTAR(df['open'], df['high'], df['low'], df['close']),
            'CDLHOMINGPIGEON': lambda df: talib.CDLHOMINGPIGEON(df['open'], df['high'], df['low'], df['close']),
        }
        
        self.categories = {
            'trend': ['SMA_10', 'SMA_20', 'SMA_50', 'EMA_10', 'EMA_20', 'EMA_50', 'WMA_10', 'WMA_20', 'LINEARREG', 'TSF'],
            'momentum': ['RSI_14', 'RSI_21', 'MACD', 'MACD_SIGNAL', 'MACD_HIST', 'STOCH_K', 'STOCH_D', 'STOCHRSI_K', 'STOCHRSI_D', 'CCI', 'MOM', 'ROC', 'WILLR'],
            'volatility': ['ATR', 'NATR', 'TRANGE', 'VAR', 'STDDEV'],
            'volume': ['OBV', 'AD', 'ADOSC'],
            'overlap': ['BBANDS_UPPER', 'BBANDS_MIDDLE', 'BBANDS_LOWER', 'SAR', 'TEMA', 'TRIMA', 'KAMA'],
            'pattern': ['CDLDOJI', 'CDLHAMMER', 'CDLENGULFING', 'CDLDOJISTAR', 'CDLHOMINGPIGEON']
        }
    
    def calculate_all_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate all technical indicators"""
        result_df = df.copy()
        
        for name, func in self.indicators.items():
            try:
                result_df[name] = func(df)
            except Exception as e:
                logger.warning(f"Failed to calculate {name}: {e}")
                result_df[name] = np.nan
        
        return result_df
    
    def get_category(self, indicator_name: str) -> str:
        """Get category for an indicator"""
        for category, indicators in self.categories.items():
            if indicator_name in indicators:
                return category
        return 'unknown'


class IndicatorOptimizer:
    """ML-based optimizer for technical indicators"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.max_features = config.get('max_features', 15)
        self.min_r2_score = config.get('min_r2_score', 0.1)
        self.lookback_periods = config.get('lookback_periods', 100)
        
        # ML models
        self.feature_selector = None
        self.importance_model = RandomForestRegressor(
            n_estimators=100,
            random_state=42,
            n_jobs=-1
        )
        
        # Technical indicator library
        self.indicator_lib = TechnicalIndicatorLibrary()
        
        # Results tracking
        self.indicator_rankings = {}
        self.optimization_history = []
        
        logger.info(f"🔬 Indicator Optimizer initialized:")
        logger.info(f"   Max Features: {self.max_features}")
        logger.info(f"   Min R2 Score: {self.min_r2_score}")
    
    async def optimize_indicators(self, symbol: str, market_data: pd.DataFrame) -> List[IndicatorImportance]:
        """Optimize indicators for a specific symbol"""
        try:
            logger.info(f"🔬 Optimizing indicators for {symbol}...")
            
            # Calculate all indicators
            df_with_indicators = self.indicator_lib.calculate_all_indicators(market_data)
            
            # Prepare features and target
            features_df, target = self._prepare_ml_data(df_with_indicators)
            
            if features_df.empty or len(features_df) < 50:
                logger.warning(f"Insufficient data for indicator optimization: {len(features_df)}")
                return []
            
            # Feature selection and importance ranking
            indicator_importance = await self._rank_indicators(features_df, target, symbol)
            
            # Store results
            self.indicator_rankings[symbol] = indicator_importance
            
            # Log top indicators
            logger.info(f"🔬 Top indicators for {symbol}:")
            for i, indicator in enumerate(indicator_importance[:5]):
                logger.info(f"   {i+1}. {indicator.name}: {indicator.importance:.4f} (R²: {indicator.r2_score:.3f})")
            
            return indicator_importance
            
        except Exception as e:
            logger.error(f"❌ Error optimizing indicators for {symbol}: {e}")
            return []
    
    def _prepare_ml_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare data for ML analysis"""
        # Create target variable (future price change)
        df['future_return'] = df['close'].shift(-5) / df['close'] - 1  # 5-period ahead return
        
        # Get indicator columns
        indicator_columns = list(self.indicator_lib.indicators.keys())
        available_indicators = [col for col in indicator_columns if col in df.columns]
        
        # Create features dataframe
        features_df = df[available_indicators].copy()
        target = df['future_return'].copy()
        
        # Remove rows with NaN values
        valid_mask = ~(features_df.isnull().any(axis=1) | target.isnull())
        features_df = features_df[valid_mask]
        target = target[valid_mask]
        
        # Remove infinite values
        features_df = features_df.replace([np.inf, -np.inf], np.nan).dropna()
        target = target[features_df.index]
        
        return features_df, target
    
    async def _rank_indicators(self, features_df: pd.DataFrame, target: pd.Series, symbol: str) -> List[IndicatorImportance]:
        """Rank indicators using multiple ML methods"""
        try:
            # Standardize features
            scaler = StandardScaler()
            features_scaled = scaler.fit_transform(features_df)
            features_scaled_df = pd.DataFrame(features_scaled, columns=features_df.columns, index=features_df.index)
            
            # Method 1: Random Forest Feature Importance
            rf_importance = self._get_rf_importance(features_scaled_df, target)
            
            # Method 2: Univariate Feature Selection
            univariate_scores = self._get_univariate_scores(features_scaled_df, target)
            
            # Method 3: Recursive Feature Elimination
            rfe_ranking = self._get_rfe_ranking(features_scaled_df, target)
            
            # Combine rankings
            combined_rankings = self._combine_rankings(rf_importance, univariate_scores, rfe_ranking)
            
            # Create IndicatorImportance objects
            indicator_importance = []
            
            for i, (indicator_name, combined_score) in enumerate(combined_rankings.items()):
                # Calculate individual R2 score
                r2, mse = self._evaluate_single_indicator(features_df[indicator_name], target)
                
                importance = IndicatorImportance(
                    name=indicator_name,
                    importance=combined_score,
                    r2_score=r2,
                    mse=mse,
                    rank=i + 1,
                    category=self.indicator_lib.get_category(indicator_name)
                )
                
                indicator_importance.append(importance)
            
            # Sort by importance
            indicator_importance.sort(key=lambda x: x.importance, reverse=True)
            
            # Update ranks
            for i, indicator in enumerate(indicator_importance):
                indicator.rank = i + 1
            
            return indicator_importance
            
        except Exception as e:
            logger.error(f"❌ Error ranking indicators: {e}")
            return []
    
    def _get_rf_importance(self, features_df: pd.DataFrame, target: pd.Series) -> Dict[str, float]:
        """Get Random Forest feature importance"""
        try:
            self.importance_model.fit(features_df, target)
            importance_scores = self.importance_model.feature_importances_
            
            return dict(zip(features_df.columns, importance_scores))
        except Exception as e:
            logger.error(f"Error calculating RF importance: {e}")
            return {}
    
    def _get_univariate_scores(self, features_df: pd.DataFrame, target: pd.Series) -> Dict[str, float]:
        """Get univariate feature selection scores"""
        try:
            selector = SelectKBest(score_func=f_regression, k='all')
            selector.fit(features_df, target)
            
            # Normalize scores to [0, 1]
            scores = selector.scores_
            normalized_scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)
            
            return dict(zip(features_df.columns, normalized_scores))
        except Exception as e:
            logger.error(f"Error calculating univariate scores: {e}")
            return {}
    
    def _get_rfe_ranking(self, features_df: pd.DataFrame, target: pd.Series) -> Dict[str, float]:
        """Get Recursive Feature Elimination ranking"""
        try:
            n_features = min(self.max_features, len(features_df.columns))
            rfe = RFE(estimator=RandomForestRegressor(n_estimators=50, random_state=42), n_features_to_select=n_features)
            rfe.fit(features_df, target)
            
            # Convert ranking to importance (lower rank = higher importance)
            max_rank = max(rfe.ranking_)
            importance_scores = [(max_rank - rank + 1) / max_rank for rank in rfe.ranking_]
            
            return dict(zip(features_df.columns, importance_scores))
        except Exception as e:
            logger.error(f"Error calculating RFE ranking: {e}")
            return {}
    
    def _combine_rankings(self, rf_importance: Dict[str, float], univariate_scores: Dict[str, float], rfe_ranking: Dict[str, float]) -> Dict[str, float]:
        """Combine multiple ranking methods"""
        combined = {}
        all_indicators = set(rf_importance.keys()) | set(univariate_scores.keys()) | set(rfe_ranking.keys())
        
        for indicator in all_indicators:
            rf_score = rf_importance.get(indicator, 0)
            uni_score = univariate_scores.get(indicator, 0)
            rfe_score = rfe_ranking.get(indicator, 0)
            
            # Weighted combination
            combined_score = (0.4 * rf_score + 0.3 * uni_score + 0.3 * rfe_score)
            combined[indicator] = combined_score
        
        # Sort by combined score
        return dict(sorted(combined.items(), key=lambda x: x[1], reverse=True))
    
    def _evaluate_single_indicator(self, indicator_series: pd.Series, target: pd.Series) -> Tuple[float, float]:
        """Evaluate single indicator performance"""
        try:
            # Remove NaN values
            valid_mask = ~(indicator_series.isnull() | target.isnull())
            X = indicator_series[valid_mask].values.reshape(-1, 1)
            y = target[valid_mask].values
            
            if len(X) < 10:
                return 0.0, float('inf')
            
            # Simple linear regression
            from sklearn.linear_model import LinearRegression
            model = LinearRegression()
            model.fit(X, y)
            
            y_pred = model.predict(X)
            r2 = r2_score(y, y_pred)
            mse = mean_squared_error(y, y_pred)
            
            return max(0, r2), mse
            
        except Exception as e:
            logger.error(f"Error evaluating indicator: {e}")
            return 0.0, float('inf')


class IndicatorOptimizerStrategy(BaseStrategy, TechnicalIndicatorMixin):
    """
    ML-based Technical Indicator Optimization Strategy
    
    Features:
    - Automatic indicator selection using ML
    - Multi-method feature ranking
    - Market regime-specific optimization
    - Performance tracking and validation
    """
    
    def __init__(self, config: Dict[str, Any], market_data_manager, name: str = "IndicatorOptimizer"):
        super().__init__(config, market_data_manager, name)
        
        # Optimization configuration
        self.optimization_interval = config.get("optimization_interval", 24)  # hours
        self.lookback_days = config.get("lookback_days", 60)
        self.max_features = config.get("max_features", 15)
        self.min_r2_score = config.get("min_r2_score", 0.1)
        self.signal_threshold = config.get("signal_threshold", 0.02)  # 2% price change
        
        # Optimizer
        self.optimizer = IndicatorOptimizer(config)
        
        # Tracking
        self.last_optimization = {}
        self.optimal_indicators = {}
        self.optimization_history = []
        
        logger.info(f"🔬 Indicator Optimizer Strategy initialized:")
        logger.info(f"   Optimization Interval: {self.optimization_interval}h")
        logger.info(f"   Max Features: {self.max_features}")
        logger.info(f"   Symbols: {self.symbols}")
    
    async def _initialize_strategy(self):
        """Initialize indicator optimizer strategy"""
        try:
            # Initialize tracking for each symbol
            for symbol in self.symbols:
                self.last_optimization[symbol] = None
                self.optimal_indicators[symbol] = []
            
            logger.info("✅ Indicator optimizer strategy validation complete")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize indicator optimizer strategy: {e}")
            raise
    
    async def generate_signal(self) -> Optional[StrategySignal]:
        """Generate signal using optimized indicators"""
        try:
            for symbol in self.symbols:
                # Check if optimization is needed
                if await self._should_optimize(symbol):
                    await self._run_optimization(symbol)
                
                # Generate signal using optimal indicators
                signal = await self._generate_optimized_signal(symbol)
                if signal and signal.action != SignalAction.HOLD:
                    return signal
            
            return None
            
        except Exception as e:
            logger.error(f"❌ Error generating optimized signal: {e}", exc_info=True)
            return None
    
    async def _should_optimize(self, symbol: str) -> bool:
        """Check if optimization should be run for symbol"""
        if self.last_optimization[symbol] is None:
            return True
        
        hours_since_optimization = (datetime.utcnow() - self.last_optimization[symbol]).total_seconds() / 3600
        return hours_since_optimization >= self.optimization_interval
    
    async def _run_optimization(self, symbol: str):
        """Run indicator optimization for symbol"""
        try:
            logger.info(f"🔬 Running indicator optimization for {symbol}")
            
            # Get historical data
            market_data = await self._get_market_data(symbol, limit=self.lookback_days * 24)
            if market_data is None:
                return
            
            df = pd.DataFrame(market_data)
            
            # Run optimization
            indicator_importance = await self.optimizer.optimize_indicators(symbol, df)
            
            # Store optimal indicators (top performers above threshold)
            self.optimal_indicators[symbol] = [
                ind for ind in indicator_importance 
                if ind.r2_score >= self.min_r2_score
            ][:self.max_features]
            
            self.last_optimization[symbol] = datetime.utcnow()
            
            # Store in history
            self.optimization_history.append({
                'symbol': symbol,
                'timestamp': datetime.utcnow(),
                'top_indicators': [ind.name for ind in self.optimal_indicators[symbol][:5]],
                'avg_r2_score': np.mean([ind.r2_score for ind in self.optimal_indicators[symbol]]) if self.optimal_indicators[symbol] else 0
            })
            
            logger.info(f"🔬 Optimization complete for {symbol}. Found {len(self.optimal_indicators[symbol])} optimal indicators")
            
        except Exception as e:
            logger.error(f"❌ Error running optimization for {symbol}: {e}")
    
    async def _generate_optimized_signal(self, symbol: str) -> Optional[StrategySignal]:
        """Generate signal using optimized indicators"""
        try:
            if not self.optimal_indicators.get(symbol):
                return None
            
            # Get recent market data
            market_data = await self._get_market_data(symbol, limit=100)
            if market_data is None:
                return None
            
            df = pd.DataFrame(market_data)
            current_price = float(df.iloc[-1]['close'])
            
            # Calculate optimal indicators
            df_with_indicators = self.optimizer.indicator_lib.calculate_all_indicators(df)
            
            # Generate ensemble signal from top indicators
            signal_strength = 0
            total_weight = 0
            
            for indicator in self.optimal_indicators[symbol][:5]:  # Top 5 indicators
                indicator_signal = self._get_indicator_signal(df_with_indicators, indicator.name)
                weight = indicator.importance * indicator.r2_score
                
                signal_strength += indicator_signal * weight
                total_weight += weight
            
            if total_weight == 0:
                return None
            
            # Normalize signal strength
            normalized_signal = signal_strength / total_weight
            
            # Check signal threshold
            if abs(normalized_signal) < self.signal_threshold:
                return None
            
            # Determine action
            action = SignalAction.BUY if normalized_signal > 0 else SignalAction.SELL
            confidence = min(abs(normalized_signal), 1.0)
            
            # Create metadata
            metadata = {
                'signal_strength': normalized_signal,
                'optimal_indicators_used': [ind.name for ind in self.optimal_indicators[symbol][:5]],
                'avg_indicator_importance': np.mean([ind.importance for ind in self.optimal_indicators[symbol][:5]]),
                'avg_r2_score': np.mean([ind.r2_score for ind in self.optimal_indicators[symbol][:5]]),
                'strategy_type': 'indicator_optimizer'
            }
            
            # Create signal
            signal = self._create_signal(
                symbol=symbol,
                action=action,
                price=current_price,
                confidence=confidence,
                metadata=metadata
            )
            
            logger.info(f"🔬 Optimized Signal: {action.value} {symbol} @ {current_price:.4f}")
            logger.info(f"   Signal Strength: {normalized_signal:.3f}, Confidence: {confidence:.2f}")
            logger.info(f"   Top Indicators: {[ind.name for ind in self.optimal_indicators[symbol][:3]]}")
            
            return signal
            
        except Exception as e:
            logger.error(f"❌ Error generating optimized signal for {symbol}: {e}")
            return None
    
    def _get_indicator_signal(self, df: pd.DataFrame, indicator_name: str) -> float:
        """Get normalized signal from an indicator"""
        try:
            if indicator_name not in df.columns:
                return 0
            
            values = df[indicator_name].dropna()
            if len(values) < 2:
                return 0
            
            current_value = values.iloc[-1]
            previous_value = values.iloc[-2]
            
            # Calculate normalized change
            if previous_value != 0:
                change = (current_value - previous_value) / abs(previous_value)
            else:
                change = 0
            
            # Bound the signal
            return np.tanh(change * 10)  # Scale and bound to [-1, 1]
            
        except Exception as e:
            logger.error(f"Error getting signal from {indicator_name}: {e}")
            return 0
    
    def get_optimization_metrics(self) -> Dict[str, Any]:
        """Get optimization performance metrics"""
        return {
            'symbols_optimized': len([s for s in self.symbols if self.optimal_indicators.get(s)]),
            'total_optimizations': len(self.optimization_history),
            'avg_indicators_per_symbol': np.mean([
                len(indicators) for indicators in self.optimal_indicators.values() if indicators
            ]) if self.optimal_indicators else 0,
            'optimization_history': self.optimization_history[-10:],  # Last 10 optimizations
            'last_optimization_times': {
                symbol: time.isoformat() if time else None
                for symbol, time in self.last_optimization.items()
            }
        }
    
    def get_strategy_info(self) -> Dict[str, Any]:
        """Get strategy information and optimization results"""
        return {
            "name": self.name,
            "type": "indicator_optimizer",
            "optimization_interval": self.optimization_interval,
            "max_features": self.max_features,
            "min_r2_score": self.min_r2_score,
            "symbols": self.symbols,
            "optimal_indicators": {
                symbol: [
                    {
                        'name': ind.name,
                        'importance': ind.importance,
                        'r2_score': ind.r2_score,
                        'rank': ind.rank,
                        'category': ind.category
                    }
                    for ind in indicators[:5]  # Top 5 per symbol
                ]
                for symbol, indicators in self.optimal_indicators.items()
            },
            "optimization_metrics": self.get_optimization_metrics(),
            "status": self.status.value,
            "enabled": self.enabled
        } 