"""
Market Regime Detection Strategy
Identifies different market regimes and adapts trading strategies accordingly
Includes trend, volatility, and correlation regime detection
"""

import logging
import numpy as np
import pandas as pd
import pandas_ta as talib
from typing import Dict, Any, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
from strategies.base_strategy import BaseStrategy, StrategySignal, SignalAction, TechnicalIndicatorMixin

logger = logging.getLogger(__name__)

class MarketRegime(Enum):
    BULL_TRENDING = "bull_trending"
    BEAR_TRENDING = "bear_trending"
    SIDEWAYS = "sideways"
    HIGH_VOLATILITY = "high_volatility"
    LOW_VOLATILITY = "low_volatility"
    CRISIS = "crisis"
    RECOVERY = "recovery"
    UNKNOWN = "unknown"


@dataclass
class RegimeState:
    """Current market regime state"""
    regime: MarketRegime
    confidence: float
    duration: int  # periods in current regime
    volatility_percentile: float
    trend_strength: float
    correlation_level: float
    timestamp: datetime


class RegimeDetector:
    """Detects market regimes using multiple methods"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.lookback_periods = config.get('lookback_periods', 100)
        self.volatility_window = config.get('volatility_window', 20)
        self.trend_window = config.get('trend_window', 50)
        self.regime_threshold = config.get('regime_threshold', 0.6)
        
        # Regime detection models
        self.kmeans_model = KMeans(n_clusters=5, random_state=42)
        self.gmm_model = GaussianMixture(n_components=5, random_state=42)
        self.scaler = StandardScaler()
        
        # Regime tracking
        self.current_regime = MarketRegime.UNKNOWN
        self.regime_history = []
        self.regime_probabilities = {}
        
        logger.info(f"🎯 Regime Detector initialized:")
        logger.info(f"   Lookback Periods: {self.lookback_periods}")
        logger.info(f"   Volatility Window: {self.volatility_window}")
    
    def calculate_regime_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate features for regime detection"""
        result_df = df.copy()
        
        # Price-based features
        result_df['returns'] = result_df['close'].pct_change()
        result_df['log_returns'] = np.log(result_df['close'] / result_df['close'].shift(1))
        
        # Volatility features
        result_df['volatility'] = result_df['returns'].rolling(window=self.volatility_window).std()
        result_df['realized_vol'] = result_df['log_returns'].rolling(window=self.volatility_window).std() * np.sqrt(252)
        result_df['vol_of_vol'] = result_df['volatility'].rolling(window=self.volatility_window).std()
        
        # Trend features
        result_df['sma_20'] = talib.SMA(result_df['close'], timeperiod=20)
        result_df['sma_50'] = talib.SMA(result_df['close'], timeperiod=50)
        result_df['ema_12'] = talib.EMA(result_df['close'], timeperiod=12)
        result_df['ema_26'] = talib.EMA(result_df['close'], timeperiod=26)
        
        # Trend strength
        result_df['trend_strength'] = (result_df['close'] - result_df['sma_50']) / result_df['sma_50']
        result_df['price_position'] = (result_df['close'] - result_df['close'].rolling(window=self.trend_window).min()) / \
                                     (result_df['close'].rolling(window=self.trend_window).max() - result_df['close'].rolling(window=self.trend_window).min())
        
        # Momentum features
        result_df['rsi'] = talib.RSI(result_df['close'], timeperiod=14)
        result_df['macd'], result_df['macd_signal'], result_df['macd_hist'] = talib.MACD(result_df['close'])
        result_df['momentum'] = talib.MOM(result_df['close'], timeperiod=10)
        
        # Volume features
        result_df['volume_sma'] = result_df['volume'].rolling(window=20).mean()
        result_df['volume_ratio'] = result_df['volume'] / result_df['volume_sma']
        
        # Skewness and kurtosis
        result_df['returns_skew'] = result_df['returns'].rolling(window=self.volatility_window).skew()
        result_df['returns_kurt'] = result_df['returns'].rolling(window=self.volatility_window).kurt()
        
        # VIX-like fear index (using volatility)
        result_df['fear_index'] = result_df['volatility'].rolling(window=self.volatility_window).quantile(0.8)
        
        return result_df
    
    async def detect_regime(self, market_data: pd.DataFrame, symbol: str = "BTC") -> RegimeState:
        """Detect current market regime"""
        try:
            # Calculate regime features
            df_features = self.calculate_regime_features(market_data)
            
            # Select features for clustering
            feature_columns = [
                'volatility', 'trend_strength', 'price_position', 'rsi',
                'volume_ratio', 'returns_skew', 'returns_kurt', 'fear_index'
            ]
            
            # Get valid data
            valid_data = df_features[feature_columns].dropna()
            
            if len(valid_data) < 50:
                logger.warning(f"Insufficient data for regime detection: {len(valid_data)}")
                return RegimeState(
                    regime=MarketRegime.UNKNOWN,
                    confidence=0.0,
                    duration=0,
                    volatility_percentile=0.5,
                    trend_strength=0.0,
                    correlation_level=0.0,
                    timestamp=datetime.utcnow()
                )
            
            # Standardize features
            features_scaled = self.scaler.fit_transform(valid_data)
            
            # Detect regime using multiple methods
            regime_scores = await self._ensemble_regime_detection(features_scaled, valid_data)
            
            # Determine final regime
            final_regime = max(regime_scores.items(), key=lambda x: x[1])[0]
            confidence = regime_scores[final_regime]
            
            # Calculate additional metrics
            latest_data = valid_data.iloc[-1]
            volatility_percentile = self._calculate_volatility_percentile(df_features['volatility'])
            trend_strength = latest_data['trend_strength']
            
            # Calculate regime duration
            duration = self._calculate_regime_duration(final_regime)
            
            # Create regime state
            regime_state = RegimeState(
                regime=final_regime,
                confidence=confidence,
                duration=duration,
                volatility_percentile=volatility_percentile,
                trend_strength=trend_strength,
                correlation_level=0.5,  # Placeholder for now
                timestamp=datetime.utcnow()
            )
            
            # Update tracking
            self.current_regime = final_regime
            self.regime_history.append(regime_state)
            
            # Keep only recent history
            cutoff_time = datetime.utcnow() - timedelta(days=30)
            self.regime_history = [
                r for r in self.regime_history 
                if r.timestamp > cutoff_time
            ]
            
            logger.info(f"🎯 Detected regime for {symbol}: {final_regime.value} (confidence: {confidence:.2f})")
            
            return regime_state
            
        except Exception as e:
            logger.error(f"❌ Error detecting regime: {e}")
            return RegimeState(
                regime=MarketRegime.UNKNOWN,
                confidence=0.0,
                duration=0,
                volatility_percentile=0.5,
                trend_strength=0.0,
                correlation_level=0.0,
                timestamp=datetime.utcnow()
            )
    
    async def _ensemble_regime_detection(self, features_scaled: np.ndarray, features_df: pd.DataFrame) -> Dict[MarketRegime, float]:
        """Use ensemble of methods to detect regime"""
        regime_scores = {regime: 0.0 for regime in MarketRegime}
        
        # Method 1: Rule-based detection
        rule_based_scores = self._rule_based_detection(features_df)
        
        # Method 2: K-means clustering
        kmeans_scores = self._kmeans_detection(features_scaled, features_df)
        
        # Method 3: Gaussian Mixture Model
        gmm_scores = self._gmm_detection(features_scaled, features_df)
        
        # Combine scores with weights
        weights = {'rule_based': 0.4, 'kmeans': 0.3, 'gmm': 0.3}
        
        for regime in MarketRegime:
            regime_scores[regime] = (
                weights['rule_based'] * rule_based_scores.get(regime, 0) +
                weights['kmeans'] * kmeans_scores.get(regime, 0) +
                weights['gmm'] * gmm_scores.get(regime, 0)
            )
        
        return regime_scores
    
    def _rule_based_detection(self, features_df: pd.DataFrame) -> Dict[MarketRegime, float]:
        """Rule-based regime detection"""
        scores = {regime: 0.0 for regime in MarketRegime}
        
        latest = features_df.iloc[-1]
        
        # Bull trending conditions
        if (latest['trend_strength'] > 0.05 and 
            latest['price_position'] > 0.7 and 
            latest['rsi'] > 50):
            scores[MarketRegime.BULL_TRENDING] = 0.8
        
        # Bear trending conditions
        elif (latest['trend_strength'] < -0.05 and 
              latest['price_position'] < 0.3 and 
              latest['rsi'] < 50):
            scores[MarketRegime.BEAR_TRENDING] = 0.8
        
        # High volatility conditions
        if latest['volatility'] > features_df['volatility'].quantile(0.8):
            scores[MarketRegime.HIGH_VOLATILITY] = 0.7
        
        # Low volatility conditions
        elif latest['volatility'] < features_df['volatility'].quantile(0.2):
            scores[MarketRegime.LOW_VOLATILITY] = 0.7
        
        # Sideways conditions
        if (abs(latest['trend_strength']) < 0.02 and 
            0.3 < latest['price_position'] < 0.7):
            scores[MarketRegime.SIDEWAYS] = 0.6
        
        # Crisis conditions (high volatility + negative trend)
        if (latest['volatility'] > features_df['volatility'].quantile(0.9) and 
            latest['trend_strength'] < -0.1):
            scores[MarketRegime.CRISIS] = 0.9
        
        return scores
    
    def _kmeans_detection(self, features_scaled: np.ndarray, features_df: pd.DataFrame) -> Dict[MarketRegime, float]:
        """K-means clustering regime detection"""
        scores = {regime: 0.0 for regime in MarketRegime}
        
        try:
            # Fit K-means
            clusters = self.kmeans_model.fit_predict(features_scaled)
            current_cluster = clusters[-1]
            
            # Map clusters to regimes based on characteristics
            cluster_features = {}
            for i in range(self.kmeans_model.n_clusters):
                cluster_mask = clusters == i
                if np.sum(cluster_mask) > 0:
                    cluster_data = features_df[cluster_mask]
                    cluster_features[i] = {
                        'avg_volatility': cluster_data['volatility'].mean(),
                        'avg_trend': cluster_data['trend_strength'].mean(),
                        'avg_position': cluster_data['price_position'].mean()
                    }
            
            # Assign regime based on cluster characteristics
            if current_cluster in cluster_features:
                features = cluster_features[current_cluster]
                
                if features['avg_trend'] > 0.03:
                    scores[MarketRegime.BULL_TRENDING] = 0.7
                elif features['avg_trend'] < -0.03:
                    scores[MarketRegime.BEAR_TRENDING] = 0.7
                elif features['avg_volatility'] > features_df['volatility'].quantile(0.7):
                    scores[MarketRegime.HIGH_VOLATILITY] = 0.6
                else:
                    scores[MarketRegime.SIDEWAYS] = 0.5
            
        except Exception as e:
            logger.error(f"Error in K-means detection: {e}")
        
        return scores
    
    def _gmm_detection(self, features_scaled: np.ndarray, features_df: pd.DataFrame) -> Dict[MarketRegime, float]:
        """Gaussian Mixture Model regime detection"""
        scores = {regime: 0.0 for regime in MarketRegime}
        
        try:
            # Fit GMM
            self.gmm_model.fit(features_scaled)
            current_probs = self.gmm_model.predict_proba(features_scaled[-1:].reshape(1, -1))[0]
            dominant_component = np.argmax(current_probs)
            confidence = current_probs[dominant_component]
            
            # Simple mapping based on component index
            regime_mapping = {
                0: MarketRegime.BULL_TRENDING,
                1: MarketRegime.BEAR_TRENDING,
                2: MarketRegime.SIDEWAYS,
                3: MarketRegime.HIGH_VOLATILITY,
                4: MarketRegime.LOW_VOLATILITY
            }
            
            if dominant_component in regime_mapping:
                scores[regime_mapping[dominant_component]] = confidence
            
        except Exception as e:
            logger.error(f"Error in GMM detection: {e}")
        
        return scores
    
    def _calculate_volatility_percentile(self, volatility_series: pd.Series) -> float:
        """Calculate current volatility percentile"""
        try:
            current_vol = volatility_series.iloc[-1]
            return volatility_series.rank(pct=True).iloc[-1]
        except:
            return 0.5
    
    def _calculate_regime_duration(self, current_regime: MarketRegime) -> int:
        """Calculate how long we've been in current regime"""
        if not self.regime_history:
            return 1
        
        duration = 1
        for regime_state in reversed(self.regime_history):
            if regime_state.regime == current_regime:
                duration += 1
            else:
                break
        
        return duration


class RegimeDetectionStrategy(BaseStrategy, TechnicalIndicatorMixin):
    """
    Market Regime Detection Strategy
    
    Features:
    - Multi-method regime detection
    - Adaptive strategy selection based on regime
    - Regime-specific risk management
    - Transition detection and alerts
    """
    
    def __init__(self, config: Dict[str, Any], market_data_manager, name: str = "RegimeDetection"):
        super().__init__(config, market_data_manager, name)
        
        # Regime detection configuration
        self.detection_interval = config.get("detection_interval", 1)  # hours
        self.lookback_days = config.get("lookback_days", 60)
        self.regime_threshold = config.get("regime_threshold", 0.6)
        self.transition_sensitivity = config.get("transition_sensitivity", 0.3)
        
        # Strategy adaptation settings
        self.regime_strategies = config.get("regime_strategies", {
            MarketRegime.BULL_TRENDING.value: {"position_size": 1.2, "stop_loss": 0.05},
            MarketRegime.BEAR_TRENDING.value: {"position_size": 0.8, "stop_loss": 0.03},
            MarketRegime.SIDEWAYS.value: {"position_size": 0.6, "stop_loss": 0.02},
            MarketRegime.HIGH_VOLATILITY.value: {"position_size": 0.4, "stop_loss": 0.02},
            MarketRegime.CRISIS.value: {"position_size": 0.2, "stop_loss": 0.01}
        })
        
        # Regime detector
        self.regime_detector = RegimeDetector(config)
        
        # Tracking
        self.current_regime_state = None
        self.last_detection = {}
        self.regime_transitions = []
        
        logger.info(f"🎯 Regime Detection Strategy initialized:")
        logger.info(f"   Detection Interval: {self.detection_interval}h")
        logger.info(f"   Symbols: {self.symbols}")
    
    async def _initialize_strategy(self):
        """Initialize regime detection strategy"""
        try:
            # Initialize tracking for each symbol
            for symbol in self.symbols:
                self.last_detection[symbol] = None
            
            logger.info("✅ Regime detection strategy validation complete")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize regime detection strategy: {e}")
            raise
    
    async def generate_signal(self) -> Optional[StrategySignal]:
        """Generate regime-aware trading signal"""
        try:
            for symbol in self.symbols:
                # Check if regime detection is needed
                if await self._should_detect_regime(symbol):
                    await self._run_regime_detection(symbol)
                
                # Generate regime-aware signal
                signal = await self._generate_regime_signal(symbol)
                if signal and signal.action != SignalAction.HOLD:
                    return signal
            
            return None
            
        except Exception as e:
            logger.error(f"❌ Error generating regime signal: {e}", exc_info=True)
            return None
    
    async def _should_detect_regime(self, symbol: str) -> bool:
        """Check if regime detection should be run"""
        if self.last_detection[symbol] is None:
            return True
        
        hours_since_detection = (datetime.utcnow() - self.last_detection[symbol]).total_seconds() / 3600
        return hours_since_detection >= self.detection_interval
    
    async def _run_regime_detection(self, symbol: str):
        """Run regime detection for symbol"""
        try:
            logger.info(f"🎯 Running regime detection for {symbol}")
            
            # Get historical data
            market_data = await self._get_market_data(symbol, limit=self.lookback_days * 24)
            if market_data is None:
                return
            
            df = pd.DataFrame(market_data)
            
            # Detect regime
            regime_state = await self.regime_detector.detect_regime(df, symbol)
            
            # Check for regime transition
            if self.current_regime_state and self.current_regime_state.regime != regime_state.regime:
                self._record_regime_transition(self.current_regime_state.regime, regime_state.regime, symbol)
            
            self.current_regime_state = regime_state
            self.last_detection[symbol] = datetime.utcnow()
            
            logger.info(f"🎯 Regime detection complete for {symbol}: {regime_state.regime.value}")
            
        except Exception as e:
            logger.error(f"❌ Error running regime detection for {symbol}: {e}")
    
    def _record_regime_transition(self, old_regime: MarketRegime, new_regime: MarketRegime, symbol: str):
        """Record regime transition"""
        transition = {
            'symbol': symbol,
            'from_regime': old_regime.value,
            'to_regime': new_regime.value,
            'timestamp': datetime.utcnow()
        }
        
        self.regime_transitions.append(transition)
        
        # Keep only recent transitions
        cutoff_time = datetime.utcnow() - timedelta(days=30)
        self.regime_transitions = [
            t for t in self.regime_transitions 
            if t['timestamp'] > cutoff_time
        ]
        
        logger.info(f"🎯 Regime transition detected for {symbol}: {old_regime.value} → {new_regime.value}")
    
    async def _generate_regime_signal(self, symbol: str) -> Optional[StrategySignal]:
        """Generate signal based on current regime"""
        try:
            if not self.current_regime_state:
                return None
            
            # Get current market data
            market_data = await self._get_market_data(symbol, limit=50)
            if market_data is None:
                return None
            
            df = pd.DataFrame(market_data)
            current_price = float(df.iloc[-1]['close'])
            
            # Get regime-specific strategy parameters
            regime_config = self.regime_strategies.get(
                self.current_regime_state.regime.value, 
                {"position_size": 1.0, "stop_loss": 0.03}
            )
            
            # Generate signal based on regime
            signal = await self._regime_specific_signal(df, self.current_regime_state, regime_config)
            
            if not signal:
                return None
            
            # Adjust signal based on regime confidence
            confidence_adjustment = self.current_regime_state.confidence
            adjusted_confidence = signal.confidence * confidence_adjustment
            
            # Create metadata
            metadata = {
                'regime': self.current_regime_state.regime.value,
                'regime_confidence': self.current_regime_state.confidence,
                'regime_duration': self.current_regime_state.duration,
                'volatility_percentile': self.current_regime_state.volatility_percentile,
                'trend_strength': self.current_regime_state.trend_strength,
                'position_size_multiplier': regime_config.get('position_size', 1.0),
                'regime_stop_loss': regime_config.get('stop_loss', 0.03),
                'strategy_type': 'regime_detection'
            }
            
            # Update signal metadata
            signal.confidence = adjusted_confidence
            signal.metadata.update(metadata)
            
            logger.info(f"🎯 Regime Signal: {signal.action.value} {symbol} @ {current_price:.4f}")
            logger.info(f"   Regime: {self.current_regime_state.regime.value} (confidence: {self.current_regime_state.confidence:.2f})")
            logger.info(f"   Duration: {self.current_regime_state.duration} periods")
            
            return signal
            
        except Exception as e:
            logger.error(f"❌ Error generating regime signal for {symbol}: {e}")
            return None
    
    async def _regime_specific_signal(self, df: pd.DataFrame, regime_state: RegimeState, regime_config: Dict[str, Any]) -> Optional[StrategySignal]:
        """Generate signal specific to current regime"""
        try:
            current_price = float(df.iloc[-1]['close'])
            
            # Calculate technical indicators
            df['sma_20'] = talib.SMA(df['close'], timeperiod=20)
            df['rsi'] = talib.RSI(df['close'], timeperiod=14)
            df['macd'], df['macd_signal'], _ = talib.MACD(df['close'])
            
            latest = df.iloc[-1]
            
            # Regime-specific logic
            if regime_state.regime == MarketRegime.BULL_TRENDING:
                # In bull market, buy on dips
                if (latest['close'] > latest['sma_20'] and 
                    latest['rsi'] < 70 and 
                    latest['macd'] > latest['macd_signal']):
                    action = SignalAction.BUY
                    confidence = 0.8
                else:
                    return None
            
            elif regime_state.regime == MarketRegime.BEAR_TRENDING:
                # In bear market, sell on rallies
                if (latest['close'] < latest['sma_20'] and 
                    latest['rsi'] > 30 and 
                    latest['macd'] < latest['macd_signal']):
                    action = SignalAction.SELL
                    confidence = 0.8
                else:
                    return None
            
            elif regime_state.regime == MarketRegime.SIDEWAYS:
                # In sideways market, mean reversion
                if latest['rsi'] < 30:
                    action = SignalAction.BUY
                    confidence = 0.6
                elif latest['rsi'] > 70:
                    action = SignalAction.SELL
                    confidence = 0.6
                else:
                    return None
            
            elif regime_state.regime == MarketRegime.HIGH_VOLATILITY:
                # In high volatility, reduce exposure
                if latest['rsi'] < 25:
                    action = SignalAction.BUY
                    confidence = 0.4
                elif latest['rsi'] > 75:
                    action = SignalAction.SELL
                    confidence = 0.4
                else:
                    return None
            
            elif regime_state.regime == MarketRegime.CRISIS:
                # In crisis, defensive positioning
                if latest['rsi'] < 20:
                    action = SignalAction.BUY
                    confidence = 0.3
                else:
                    return None
            
            else:
                return None
            
            # Create signal
            signal = self._create_signal(
                symbol=df.iloc[-1].get('symbol', 'BTC'),
                action=action,
                price=current_price,
                confidence=confidence,
                metadata={}
            )
            
            return signal
            
        except Exception as e:
            logger.error(f"❌ Error generating regime-specific signal: {e}")
            return None
    
    def get_regime_metrics(self) -> Dict[str, Any]:
        """Get regime detection metrics"""
        if not self.current_regime_state:
            return {}
        
        return {
            'current_regime': self.current_regime_state.regime.value,
            'regime_confidence': self.current_regime_state.confidence,
            'regime_duration': self.current_regime_state.duration,
            'volatility_percentile': self.current_regime_state.volatility_percentile,
            'trend_strength': self.current_regime_state.trend_strength,
            'total_transitions': len(self.regime_transitions),
            'recent_transitions': self.regime_transitions[-5:] if self.regime_transitions else [],
            'regime_distribution': self._calculate_regime_distribution()
        }
    
    def _calculate_regime_distribution(self) -> Dict[str, float]:
        """Calculate distribution of regimes over time"""
        if not self.regime_detector.regime_history:
            return {}
        
        regime_counts = {}
        for regime_state in self.regime_detector.regime_history:
            regime_name = regime_state.regime.value
            regime_counts[regime_name] = regime_counts.get(regime_name, 0) + 1
        
        total = sum(regime_counts.values())
        return {regime: count / total for regime, count in regime_counts.items()}
    
    def get_strategy_info(self) -> Dict[str, Any]:
        """Get strategy information and regime metrics"""
        return {
            "name": self.name,
            "type": "regime_detection",
            "detection_interval": self.detection_interval,
            "regime_threshold": self.regime_threshold,
            "symbols": self.symbols,
            "regime_strategies": self.regime_strategies,
            "regime_metrics": self.get_regime_metrics(),
            "status": self.status.value,
            "enabled": self.enabled
        } 