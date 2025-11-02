"""
LSTM Machine Learning Strategy - Advanced ML-based trading
Uses LSTM neural networks for price prediction and signal generation
"""

import logging
import numpy as np
import pandas as pd
import joblib
import os
from typing import Dict, Any, Optional, Tuple
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from strategies.base_strategy import BaseStrategy, StrategySignal, SignalAction, TechnicalIndicatorMixin
from keras.models import Sequential, load_model
from keras.layers import LSTM, Dense, Dropout, BatchNormalization
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ReduceLROnPlateau

logger = logging.getLogger(__name__)

class LSTMStrategy(BaseStrategy, TechnicalIndicatorMixin):
    """
    LSTM-based Machine Learning Trading Strategy
    
    Features:
    - LSTM neural network for price prediction
    - Multi-timeframe analysis
    - Feature engineering with technical indicators
    - Model retraining and validation
    - Confidence-based signal generation
    """
    
    def __init__(self, config: Dict[str, Any], market_data_manager, name: str = "LSTM_ML"):
        super().__init__(config, market_data_manager, name)
        
        # ML Configuration
        self.sequence_length = config.get("sequence_length", 60)  # 60 periods lookback
        self.prediction_horizon = config.get("prediction_horizon", 5)  # 5 periods ahead
        self.feature_columns = config.get("feature_columns", [
            'close', 'volume', 'high', 'low', 'sma_20', 'ema_12', 'rsi', 'macd'
        ])
        
        # Model parameters
        self.lstm_units = config.get("lstm_units", [50, 50])
        self.dropout_rate = config.get("dropout_rate", 0.2)
        self.learning_rate = config.get("learning_rate", 0.001)
        self.epochs = config.get("epochs", 100)
        self.batch_size = config.get("batch_size", 32)
        
        # Trading parameters
        self.min_confidence = config.get("min_confidence", 0.6)
        self.price_change_threshold = config.get("price_change_threshold", 0.02)  # 2%
        
        # Model storage
        self.models = {}  # symbol -> model
        self.scalers = {}  # symbol -> scaler
        self.model_dir = config.get("model_dir", "models/lstm")
        
        # Performance tracking
        self.prediction_accuracy = {}
        self.last_predictions = {}
        
        # Minimum data points for training
        self.min_training_data = max(self.sequence_length * 3, 200)
        
        # Create model directory
        os.makedirs(self.model_dir, exist_ok=True)
        
        logger.info(f"🤖 LSTM ML Strategy initialized:")
        logger.info(f"   Sequence Length: {self.sequence_length}")
        logger.info(f"   Prediction Horizon: {self.prediction_horizon}")
        logger.info(f"   Features: {self.feature_columns}")
        logger.info(f"   Symbols: {self.symbols}")
    
    async def _initialize_strategy(self):
        """Initialize ML models for each symbol"""
        try:
            # Load existing models if available
            for symbol in self.symbols:
                await self._load_or_create_model(symbol)
            
            logger.info("✅ LSTM ML strategy validation complete")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize LSTM strategy: {e}")
            raise
    
    async def _load_or_create_model(self, symbol: str):
        """Load existing model or create new one"""
        model_path = os.path.join(self.model_dir, f"{symbol}_lstm_model")
        scaler_path = os.path.join(self.model_dir, f"{symbol}_scaler.pkl")
        
        try:
            # Try to load existing model and scaler
            if os.path.exists(f"{model_path}.h5"):
                self.models[symbol] = load_model(f"{model_path}.h5")
                logger.info(f"📁 Loaded existing LSTM model for {symbol}")
            else:
                self.models[symbol] = None
                logger.info(f"🆕 Will create new model for {symbol}")
            
            if os.path.exists(scaler_path):
                self.scalers[symbol] = joblib.load(scaler_path)
                logger.info(f"📁 Loaded existing scaler for {symbol}")
            else:
                self.scalers[symbol] = MinMaxScaler()
                logger.info(f"🆕 Created new scaler for {symbol}")
                
        except Exception as e:
            logger.warning(f"⚠️ Could not load model for {symbol}: {e}")
            self.models[symbol] = None
            self.scalers[symbol] = MinMaxScaler()
    
    def _create_lstm_model(self, input_shape: Tuple[int, int]) -> Any:
        """Create LSTM neural network model"""
        model = Sequential()
        
        # First LSTM layer
        model.add(LSTM(
            units=self.lstm_units[0],
            return_sequences=len(self.lstm_units) > 1,
            input_shape=input_shape
        ))
        model.add(Dropout(self.dropout_rate))
        model.add(BatchNormalization())
        
        # Additional LSTM layers
        for i, units in enumerate(self.lstm_units[1:], 1):
            return_sequences = i < len(self.lstm_units) - 1
            model.add(LSTM(units=units, return_sequences=return_sequences))
            model.add(Dropout(self.dropout_rate))
            model.add(BatchNormalization())
        
        # Dense layers
        model.add(Dense(25, activation='relu'))
        model.add(Dropout(self.dropout_rate))
        model.add(Dense(1))  # Single output for price prediction
        
        # Compile model
        model.compile(
            optimizer=Adam(learning_rate=self.learning_rate),
            loss='mse',
            metrics=['mae']
        )
        
        return model
    
    async def generate_signal(self) -> Optional[StrategySignal]:
        """Generate ML-based trading signal"""
        try:
            for symbol in self.symbols:
                signal = await self._analyze_symbol_ml(symbol)
                if signal and signal.action != SignalAction.HOLD:
                    return signal
            
            return None
            
        except Exception as e:
            logger.error(f"❌ Error generating ML signal: {e}", exc_info=True)
            return None
    
    async def _analyze_symbol_ml(self, symbol: str) -> Optional[StrategySignal]:
        """Analyze symbol using ML model"""
        try:
            # Get market data
            data = await self._get_market_data(symbol, limit=self.min_training_data)
            if data is None or len(data) < self.min_training_data:
                logger.warning(f"⚠️ Insufficient data for ML analysis {symbol}: {len(data) if data is not None else 0}")
                return None
            
            # Prepare features
            features_df = await self._prepare_features(symbol, data)
            if features_df is None or len(features_df) < self.sequence_length:
                return None
            
            # Train or retrain model if needed
            if self.models[symbol] is None or await self._should_retrain(symbol):
                await self._train_model(symbol, features_df)
            
            # Make prediction
            prediction = await self._make_prediction(symbol, features_df)
            if prediction is None:
                return None
            
            # Generate signal based on prediction
            signal = await self._generate_signal_from_prediction(symbol, features_df, prediction)
            return signal
            
        except Exception as e:
            logger.error(f"❌ Error in ML analysis for {symbol}: {e}")
            return None
    
    async def _prepare_features(self, symbol: str, data: Any) -> Optional[pd.DataFrame]:
        """Prepare feature matrix for ML model"""
        try:
            df = pd.DataFrame(data)
            
            # Calculate technical indicators
            df['sma_20'] = self.calculate_sma(df['close'], 20)
            df['ema_12'] = self.calculate_ema(df['close'], 12)
            df['rsi'] = self.calculate_rsi(df['close'], 14)
            
            # MACD
            macd_line, macd_signal, macd_histogram = self.calculate_macd(df['close'])
            df['macd'] = macd_line
            df['macd_signal'] = macd_signal
            df['macd_histogram'] = macd_histogram
            
            # Bollinger Bands
            bb_upper, bb_middle, bb_lower = self.calculate_bollinger_bands(df['close'], 20, 2)
            df['bb_upper'] = bb_upper
            df['bb_lower'] = bb_lower
            df['bb_width'] = (bb_upper - bb_lower) / bb_middle
            
            # Price features
            df['price_change'] = df['close'].pct_change()
            df['high_low_ratio'] = df['high'] / df['low']
            df['volume_sma'] = self.calculate_sma(df['volume'], 20)
            df['volume_ratio'] = df['volume'] / df['volume_sma']
            
            # Volatility
            df['volatility'] = df['price_change'].rolling(window=20).std()
            
            # Time-based features
            df['hour'] = pd.to_datetime(df.index).hour if hasattr(df.index, 'hour') else 12
            df['day_of_week'] = pd.to_datetime(df.index).dayofweek if hasattr(df.index, 'dayofweek') else 1
            
            # Drop NaN values
            df = df.dropna()
            
            # Select feature columns
            available_features = [col for col in self.feature_columns if col in df.columns]
            if len(available_features) < len(self.feature_columns) * 0.8:
                logger.warning(f"⚠️ Missing features for {symbol}: {set(self.feature_columns) - set(available_features)}")
            
            features_df = df[available_features].copy()
            
            logger.debug(f"📊 Prepared {len(features_df)} feature rows for {symbol}")
            return features_df
            
        except Exception as e:
            logger.error(f"❌ Error preparing features for {symbol}: {e}")
            return None
    
    async def _train_model(self, symbol: str, features_df: pd.DataFrame):
        """Train ML model for symbol"""
        try:
            logger.info(f"🎓 Training ML model for {symbol}...")
            
            # Prepare training data
            X, y = self._create_sequences(features_df)
            if len(X) < 50:  # Minimum training samples
                logger.warning(f"⚠️ Insufficient training data for {symbol}: {len(X)} samples")
                return
            
            # Split data
            split_idx = int(len(X) * 0.8)
            X_train, X_test = X[:split_idx], X[split_idx:]
            y_train, y_test = y[:split_idx], y[split_idx:]
            
            # Scale features
            X_train_scaled = self.scalers[symbol].fit_transform(
                X_train.reshape(-1, X_train.shape[-1])
            ).reshape(X_train.shape)
            X_test_scaled = self.scalers[symbol].transform(
                X_test.reshape(-1, X_test.shape[-1])
            ).reshape(X_test.shape)
            
            # Create and train model
            model = self._create_lstm_model((X_train.shape[1], X_train.shape[2]))
            
            # Callbacks
            early_stopping = EarlyStopping(patience=10, restore_best_weights=True)
            reduce_lr = ReduceLROnPlateau(factor=0.5, patience=5, min_lr=1e-6)
            
            # Train model
            history = model.fit(
                X_train_scaled, y_train,
                epochs=self.epochs,
                batch_size=self.batch_size,
                validation_data=(X_test_scaled, y_test),
                callbacks=[early_stopping, reduce_lr],
                verbose=0
            )
            
            # Evaluate model
            train_loss = model.evaluate(X_train_scaled, y_train, verbose=0)
            test_loss = model.evaluate(X_test_scaled, y_test, verbose=0)
            
            logger.info(f"📈 Model training complete for {symbol}")
            logger.info(f"   Train Loss: {train_loss[0]:.6f}")
            logger.info(f"   Test Loss: {test_loss[0]:.6f}")
            
            # Save model
            self.models[symbol] = model
            await self._save_model(symbol)
            
            # Update prediction accuracy
            self.prediction_accuracy[symbol] = {
                'last_updated': datetime.utcnow(),
                'training_samples': len(X_train),
                'test_samples': len(X_test)
            }
            
        except Exception as e:
            logger.error(f"❌ Error training model for {symbol}: {e}")
    
    def _create_sequences(self, features_df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Create sequences for LSTM training"""
        data = features_df.values
        X, y = [], []
        
        for i in range(self.sequence_length, len(data) - self.prediction_horizon):
            X.append(data[i-self.sequence_length:i])
            # Predict price change percentage
            current_price = data[i, 0]  # Assuming first column is close price
            future_price = data[i + self.prediction_horizon, 0]
            price_change = (future_price - current_price) / current_price
            y.append(price_change)
        
        return np.array(X), np.array(y)
    
    async def _make_prediction(self, symbol: str, features_df: pd.DataFrame) -> Optional[float]:
        """Make price prediction using trained model"""
        try:
            if self.models[symbol] is None:
                return None
            
            # Prepare input sequence
            data = features_df.values
            if len(data) < self.sequence_length:
                return None
            
            # Get last sequence
            last_sequence = data[-self.sequence_length:].reshape(1, self.sequence_length, -1)
            
            # Scale input
            last_sequence_scaled = self.scalers[symbol].transform(
                last_sequence.reshape(-1, last_sequence.shape[-1])
            ).reshape(last_sequence.shape)
            
            # Make prediction
            prediction = self.models[symbol].predict(last_sequence_scaled, verbose=0)[0][0]
            
            # Store prediction for validation
            self.last_predictions[symbol] = {
                'prediction': prediction,
                'timestamp': datetime.utcnow(),
                'current_price': data[-1, 0]
            }
            
            return float(prediction)
            
        except Exception as e:
            logger.error(f"❌ Error making prediction for {symbol}: {e}")
            return None
    
    async def _generate_signal_from_prediction(
        self, 
        symbol: str, 
        features_df: pd.DataFrame, 
        prediction: float
    ) -> Optional[StrategySignal]:
        """Generate trading signal from ML prediction"""
        try:
            current_price = float(features_df.iloc[-1, 0])  # Assuming first column is close
            
            # Calculate confidence based on prediction magnitude
            confidence = min(abs(prediction) / self.price_change_threshold, 1.0)
            
            # Only generate signal if confidence is above threshold
            if confidence < self.min_confidence:
                return None
            
            # Determine action
            if prediction > self.price_change_threshold:
                action = SignalAction.BUY
            elif prediction < -self.price_change_threshold:
                action = SignalAction.SELL
            else:
                return None
            
            # Create metadata
            metadata = {
                'ml_prediction': prediction,
                'predicted_price_change_pct': prediction * 100,
                'model_confidence': confidence,
                'sequence_length': self.sequence_length,
                'prediction_horizon': self.prediction_horizon,
                'strategy_type': 'lstm_ml'
            }
            
            # Create signal
            signal = self._create_signal(
                symbol=symbol,
                action=action,
                price=current_price,
                confidence=confidence,
                metadata=metadata
            )
            
            logger.info(f"🤖 ML Signal: {action.value} {symbol} @ {current_price:.4f}")
            logger.info(f"   Prediction: {prediction*100:.2f}% price change")
            logger.info(f"   Confidence: {confidence:.2f}")
            
            return signal
            
        except Exception as e:
            logger.error(f"❌ Error generating signal from prediction for {symbol}: {e}")
            return None
    
    async def _should_retrain(self, symbol: str) -> bool:
        """Determine if model should be retrained"""
        if symbol not in self.prediction_accuracy:
            return True
        
        last_update = self.prediction_accuracy[symbol].get('last_updated')
        if last_update is None:
            return True
        
        # Retrain every 24 hours
        hours_since_update = (datetime.utcnow() - last_update).total_seconds() / 3600
        return hours_since_update > 24
    
    async def _save_model(self, symbol: str):
        """Save trained model and scaler"""
        try:
            model_path = os.path.join(self.model_dir, f"{symbol}_lstm_model")
            scaler_path = os.path.join(self.model_dir, f"{symbol}_scaler.pkl")
            
            # Save model
            if hasattr(self.models[symbol], 'save'):
                self.models[symbol].save(f"{model_path}.h5")
            
            # Save scaler
            joblib.dump(self.scalers[symbol], scaler_path)
            
            logger.debug(f"💾 Saved model and scaler for {symbol}")
            
        except Exception as e:
            logger.warning(f"⚠️ Could not save model for {symbol}: {e}")
    
    def get_strategy_info(self) -> Dict[str, Any]:
        """Get strategy information and performance metrics"""
        return {
            "name": self.name,
            "type": "machine_learning",
            "ml_framework": "tensorflow",
            "sequence_length": self.sequence_length,
            "prediction_horizon": self.prediction_horizon,
            "features": self.feature_columns,
            "symbols": self.symbols,
            "models_trained": len([s for s in self.symbols if self.models.get(s) is not None]),
            "last_predictions": self.last_predictions,
            "prediction_accuracy": self.prediction_accuracy,
            "status": self.status.value,
            "enabled": self.enabled
        } 