"""
DQN Reinforcement Learning Strategy
===================================
Deep Q-Network (DQN) reinforcement learning trading strategy.

Uses RL to learn optimal trading actions through environment interaction.
Features:
- Experience replay for stable learning
- Target network for improved stability
- Continuous learning and adaptation
- Model persistence
"""

import asyncio
import logging
import numpy as np
import pandas as pd
import random
import os
from typing import Dict, Optional, Any, Tuple
from datetime import datetime, timedelta
from collections import deque
from pathlib import Path

# Suppress TensorFlow oneDNN warnings before import
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress INFO and WARNING messages

try:
    import tensorflow as tf  # type: ignore
    from tensorflow import keras  # type: ignore
    Sequential = keras.models.Sequential
    load_model = keras.models.load_model
    Dense = keras.layers.Dense
    Dropout = keras.layers.Dropout
    Adam = keras.optimizers.Adam
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    Sequential = None
    load_model = None
    Dense = None
    Dropout = None
    Adam = None
    tf = None
    keras = None

from .base import BaseStrategy

logger = logging.getLogger(__name__)


class TradingEnvironment:
    """Trading environment for RL agent."""
    
    def __init__(self, data: pd.DataFrame, initial_balance: float = 10000):
        """
        Initialize trading environment.
        
        Args:
            data: DataFrame with OHLCV data and indicators
            initial_balance: Starting balance
        """
        self.data = data.reset_index(drop=True)
        self.initial_balance = initial_balance
        self.reset()
    
    def reset(self) -> np.ndarray:
        """Reset environment to initial state."""
        self.current_step = 0
        self.balance = self.initial_balance
        self.position = 0  # 0: no position, 1: long, -1: short
        self.entry_price = 0
        self.total_reward = 0
        self.trade_history = []
        return self._get_state()
    
    def _get_state(self) -> np.ndarray:
        """Get current state representation."""
        if self.current_step >= len(self.data):
            return np.zeros(10)  # Default state
        
        row = self.data.iloc[self.current_step]
        
        # Technical indicators as state
        state = np.array([
            row.get('close', 0),
            row.get('volume', 0) if 'volume' in row else 0,
            row.get('rsi', 50),
            row.get('macd', 0),
            row.get('bb_upper', 0),
            row.get('bb_lower', 0),
            self.position,
            self.balance / self.initial_balance,
            row.get('price_change', 0),
            row.get('volatility', 0)
        ])
        
        return state.astype(np.float32)
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool]:
        """
        Execute action and return next state, reward, done.
        
        Args:
            action: 0=hold/close, 1=buy, 2=sell
            
        Returns:
            Tuple of (next_state, reward, done)
        """
        if self.current_step >= len(self.data) - 1:
            return self._get_state(), 0, True
        
        current_price = float(self.data.iloc[self.current_step]['close'])
        next_price = float(self.data.iloc[self.current_step + 1]['close'])
        
        reward = 0
        
        # Execute action
        if action == 1 and self.position == 0:  # Buy
            self.position = 1
            self.entry_price = current_price
        elif action == 2 and self.position == 0:  # Sell (short)
            self.position = -1
            self.entry_price = current_price
        elif action == 0 and self.position != 0:  # Close position
            if self.position == 1:  # Close long
                reward = (current_price - self.entry_price) / self.entry_price
            else:  # Close short
                reward = (self.entry_price - current_price) / self.entry_price
            
            self.balance *= (1 + reward)
            self.total_reward += reward
            self.trade_history.append({
                'entry_price': self.entry_price,
                'exit_price': current_price,
                'position': self.position,
                'reward': reward
            })
            self.position = 0
        
        # Holding reward/penalty
        if self.position == 1:  # Long position
            unrealized_pnl = (next_price - self.entry_price) / self.entry_price
            reward += unrealized_pnl * 0.1  # Small reward for unrealized gains
        elif self.position == -1:  # Short position
            unrealized_pnl = (self.entry_price - next_price) / self.entry_price
            reward += unrealized_pnl * 0.1
        
        self.current_step += 1
        next_state = self._get_state()
        done = self.current_step >= len(self.data) - 1
        
        return next_state, reward, done


class DQNAgent:
    """Deep Q-Network agent for trading."""
    
    def __init__(self, state_size: int, action_size: int, learning_rate: float = 0.001):
        """
        Initialize DQN agent.
        
        Args:
            state_size: Size of state vector
            action_size: Number of possible actions (3: hold/close, buy, sell)
            learning_rate: Learning rate for optimizer
        """
        if not TENSORFLOW_AVAILABLE:
            raise ImportError("TensorFlow/Keras not available. Install with: pip install tensorflow")
        
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = learning_rate
        
        self.q_network = self._build_model()
        self.target_network = self._build_model()
        self.update_target_network()
    
    def _build_model(self):
        """Build neural network for Q-learning."""
        model = Sequential([
            Dense(64, input_dim=self.state_size, activation='relu'),
            Dropout(0.2),
            Dense(32, activation='relu'),
            Dropout(0.2),
            Dense(16, activation='relu'),
            Dense(self.action_size, activation='linear')
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=self.learning_rate),
            loss='mse'
        )
        
        return model
    
    def update_target_network(self):
        """Update target network weights."""
        self.target_network.set_weights(self.q_network.get_weights())
    
    def remember(self, state, action, reward, next_state, done):
        """Store experience in replay buffer."""
        self.memory.append((state, action, reward, next_state, done))
    
    def act(self, state: np.ndarray, training: bool = True) -> int:
        """
        Choose action using epsilon-greedy policy.
        
        Args:
            state: Current state vector
            training: Whether agent is in training mode (uses epsilon)
            
        Returns:
            Action index (0=hold/close, 1=buy, 2=sell)
        """
        if training and np.random.random() <= self.epsilon:
            return random.randint(0, self.action_size - 1)
        
        q_values = self.q_network.predict(state.reshape(1, -1), verbose=0)
        return np.argmax(q_values[0])
    
    def replay(self, batch_size: int = 32):
        """Train the model on a batch of experiences."""
        if len(self.memory) < batch_size:
            return
        
        batch = random.sample(self.memory, batch_size)
        states = np.array([e[0] for e in batch])
        actions = np.array([e[1] for e in batch])
        rewards = np.array([e[2] for e in batch])
        next_states = np.array([e[3] for e in batch])
        dones = np.array([e[4] for e in batch])
        
        # Current Q values
        current_q_values = self.q_network.predict(states, verbose=0)
        
        # Next Q values from target network
        next_q_values = self.target_network.predict(next_states, verbose=0)
        
        # Update Q values using Bellman equation
        for i in range(batch_size):
            if dones[i]:
                current_q_values[i][actions[i]] = rewards[i]
            else:
                current_q_values[i][actions[i]] = rewards[i] + 0.95 * np.max(next_q_values[i])
        
        # Train the model
        self.q_network.fit(states, current_q_values, epochs=1, verbose=0)
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


class DQNStrategy(BaseStrategy):
    """
    Deep Q-Network Reinforcement Learning Trading Strategy.
    
    Uses RL to learn optimal trading actions through environment interaction.
    """
    
    def __init__(self, exchange_adapter=None, config: Optional[Dict] = None):
        """
        Initialize DQN RL strategy.
        
        Args:
            exchange_adapter: Exchange adapter instance
            config: Strategy configuration
        """
        super().__init__(name="DQN_RL")
        self.exchange_adapter = exchange_adapter
        self.config = config or {}
        
        # RL Configuration
        self.state_size = self.config.get("state_size", 10)
        self.action_size = self.config.get("action_size", 3)  # 0: hold/close, 1: buy, 2: sell
        self.learning_rate = self.config.get("learning_rate", 0.001)
        self.training_episodes = self.config.get("training_episodes", 100)
        self.batch_size = self.config.get("batch_size", 32)
        
        # Trading parameters
        self.min_confidence = self.config.get("min_confidence", 0.6)
        self.lookback_periods = self.config.get("lookback_periods", 500)
        # Support single symbol or list of symbols (like original)
        symbol_config = self.config.get("symbol", "BTCUSDT")
        if isinstance(symbol_config, list):
            self.symbols = symbol_config
            self.symbol = self.symbols[0]  # Primary symbol for single-symbol operations
        else:
            self.symbols = [symbol_config]
            self.symbol = symbol_config
        self.timeframe = self.config.get("timeframe", "1h")
        
        # Model storage
        self.agents: Dict[str, DQNAgent] = {}  # symbol -> agent
        self.model_dir = Path(self.config.get("model_dir", "./models/dqn"))
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        # Performance tracking
        self.training_history: Dict[str, Dict] = {}
        self.last_actions: Dict[str, Dict] = {}
        self.last_training: Dict[str, datetime] = {}
        
        # Position tracking
        self.current_positions: Dict[str, int] = {}  # symbol -> position (0, 1, -1)
        
        if not TENSORFLOW_AVAILABLE:
            logger.warning("TensorFlow not available. DQN strategy will not work. Install with: pip install tensorflow")
    
    async def initialize(self, config: Dict):
        """Initialize strategy with configuration."""
        await super().initialize(config)
        
        if not TENSORFLOW_AVAILABLE:
            raise ImportError("TensorFlow/Keras required for DQN strategy. Install with: pip install tensorflow")
        
        # Merge config
        self.config.update(config)
        
        # Update symbols if provided
        symbol_config = self.config.get("symbol", self.symbol)
        if isinstance(symbol_config, list):
            self.symbols = symbol_config
            self.symbol = self.symbols[0]
        else:
            self.symbols = [symbol_config]
            self.symbol = symbol_config
        
        # Initialize agents for all symbols (like original)
        for symbol in self.symbols:
            await self._initialize_agent(symbol)
        
        self.logger.info(f"DQN RL Strategy initialized for symbols: {self.symbols}")
    
    async def _initialize_agent(self, symbol: str):
        """Initialize DQN agent for symbol."""
        try:
            # Create agent
            self.agents[symbol] = DQNAgent(
                state_size=self.state_size,
                action_size=self.action_size,
                learning_rate=self.learning_rate
            )
            
            # Try to load existing model
            model_path = self.model_dir / f"{symbol}_dqn_model.h5"
            if model_path.exists():
                self.agents[symbol].q_network = load_model(str(model_path))
                self.agents[symbol].update_target_network()
                self.logger.info(f"📁 Loaded existing DQN model for {symbol}")
            else:
                self.logger.info(f"🆕 Created new DQN agent for {symbol}")
            
            # Initialize training history
            self.training_history[symbol] = {
                'episodes': 0,
                'total_reward': 0,
                'last_trained': None
            }
            
        except Exception as e:
            self.logger.error(f"❌ Error initializing agent for {symbol}: {e}")
            raise
    
    async def execute(self) -> Optional[Dict]:
        """
        Execute DQN strategy logic.
        
        Returns:
            Trading signal or None
        """
        if not TENSORFLOW_AVAILABLE:
            return None
        
        try:
            # Check all symbols and generate signal (like original)
            for symbol in self.symbols:
                # Check if we need to train
                if await self._should_train(symbol):
                    await self._train_agent(symbol)
                
                # Get market data and generate signal
                signal = await self._analyze_symbol_rl(symbol)
                if signal and signal.get('action') != 'HOLD':
                    return signal
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error in DQN strategy execution: {e}")
            return None
    
    async def _analyze_symbol_rl(self, symbol: str) -> Optional[Dict]:
        """Analyze symbol using RL agent."""
        try:
            # Get market data
            if not self.exchange_adapter:
                self.logger.warning("No exchange adapter available")
                return None
            
            # Fetch OHLCV data
            ohlcv_data = await self.exchange_adapter.fetch_ohlcv(
                symbol=symbol,
                timeframe=self.timeframe,
                limit=self.lookback_periods
            )
            
            if not ohlcv_data or len(ohlcv_data) < 100:
                self.logger.warning(f"Insufficient data for RL analysis {symbol}")
                return None
            
            # Prepare features
            features_df = await self._prepare_features(symbol, ohlcv_data)
            if features_df is None or len(features_df) < 50:
                return None
            
            # Get current state
            current_state = self._get_current_state(features_df, symbol)
            if current_state is None:
                return None
            
            # Get action from agent (not in training mode during live trading)
            action = self.agents[symbol].act(current_state, training=False)
            
            # Convert action to signal
            signal = await self._convert_action_to_signal(symbol, action, features_df, current_state)
            return signal
            
        except Exception as e:
            self.logger.error(f"Error in RL analysis for {symbol}: {e}")
            return None
    
    async def _prepare_features(self, symbol: str, data: list) -> Optional[pd.DataFrame]:
        """Prepare feature matrix for RL."""
        try:
            # Convert to DataFrame
            df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            # Calculate technical indicators
            df['sma_20'] = df['close'].rolling(window=20).mean()
            df['ema_12'] = df['close'].ewm(span=12).mean()
            
            # RSI
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df['rsi'] = 100 - (100 / (1 + rs))
            
            # MACD
            ema12 = df['close'].ewm(span=12).mean()
            ema26 = df['close'].ewm(span=26).mean()
            df['macd'] = ema12 - ema26
            
            # Bollinger Bands
            sma = df['close'].rolling(window=20).mean()
            std = df['close'].rolling(window=20).std()
            df['bb_upper'] = sma + (std * 2)
            df['bb_lower'] = sma - (std * 2)
            
            # Price features
            df['price_change'] = df['close'].pct_change()
            df['volatility'] = df['price_change'].rolling(window=20).std()
            
            # Normalize features
            for col in ['close', 'volume', 'rsi', 'macd', 'bb_upper', 'bb_lower']:
                if col in df.columns:
                    mean_val = df[col].mean()
                    std_val = df[col].std()
                    if std_val > 0:
                        df[col] = (df[col] - mean_val) / std_val
            
            # Drop NaN values
            df = df.dropna()
            
            self.logger.debug(f"Prepared {len(df)} feature rows for RL {symbol}")
            return df
            
        except Exception as e:
            self.logger.error(f"Error preparing RL features for {symbol}: {e}")
            return None
    
    def _get_current_state(self, features_df: pd.DataFrame, symbol: str) -> Optional[np.ndarray]:
        """Get current state for RL agent."""
        try:
            if len(features_df) == 0:
                return None
            
            row = features_df.iloc[-1]
            position = self.current_positions.get(symbol, 0)
            
            state = np.array([
                row.get('close', 0),
                row.get('volume', 0),
                row.get('rsi', 50),
                row.get('macd', 0),
                row.get('bb_upper', 0),
                row.get('bb_lower', 0),
                position,
                1.0,  # balance ratio (normalized)
                row.get('price_change', 0),
                row.get('volatility', 0)
            ], dtype=np.float32)
            
            return state
            
        except Exception as e:
            self.logger.error(f"Error getting current state: {e}")
            return None
    
    async def _convert_action_to_signal(
        self,
        symbol: str,
        action: int,
        features_df: pd.DataFrame,
        state: np.ndarray
    ) -> Optional[Dict]:
        """Convert RL action to trading signal."""
        try:
            current_price = float(features_df.iloc[-1]['close'])
            
            # Map actions: 0=hold/close, 1=buy, 2=sell
            if action == 0:
                # Close position if we have one
                if self.current_positions.get(symbol, 0) != 0:
                    return {
                        'action': 'SELL' if self.current_positions[symbol] == 1 else 'BUY',
                        'symbol': symbol,
                        'size': 0,  # Will be calculated by position manager
                        'confidence': 0.7,
                        'metadata': {
                            'rl_action': action,
                            'strategy_type': 'dqn_rl',
                            'reason': 'close_position'
                        }
                    }
                return None  # Hold
            
            elif action == 1:  # Buy
                if self.current_positions.get(symbol, 0) != 0:
                    return None  # Already have position
                
                # Calculate confidence based on Q-values
                q_values = self.agents[symbol].q_network.predict(state.reshape(1, -1), verbose=0)[0]
                max_q = np.max(q_values)
                confidence = min(max_q / 10.0, 1.0) if max_q > 0 else 0.5
                
                if confidence < self.min_confidence:
                    return None
                
                # Store last action (like original)
                self.last_actions[symbol] = {
                    'action': action,
                    'timestamp': datetime.utcnow(),
                    'confidence': confidence
                }
                
                return {
                    'action': 'BUY',
                    'symbol': symbol,
                    'size': self.config.get('position_size', 0.01),
                    'confidence': confidence,
                    'metadata': {
                        'rl_action': action,
                        'rl_confidence': confidence,
                        'agent_epsilon': self.agents[symbol].epsilon,
                        'training_episodes': self.training_history[symbol]['episodes'],
                        'strategy_type': 'dqn_rl'
                    }
                }
            
            elif action == 2:  # Sell (short)
                if self.current_positions.get(symbol, 0) != 0:
                    return None  # Already have position
                
                # Calculate confidence
                q_values = self.agents[symbol].q_network.predict(state.reshape(1, -1), verbose=0)[0]
                max_q = np.max(q_values)
                confidence = min(max_q / 10.0, 1.0) if max_q > 0 else 0.5
                
                if confidence < self.min_confidence:
                    return None
                
                # Store last action (like original)
                self.last_actions[symbol] = {
                    'action': action,
                    'timestamp': datetime.utcnow(),
                    'confidence': confidence
                }
                
                return {
                    'action': 'SELL',
                    'symbol': symbol,
                    'size': self.config.get('position_size', 0.01),
                    'confidence': confidence,
                    'metadata': {
                        'rl_action': action,
                        'rl_confidence': confidence,
                        'agent_epsilon': self.agents[symbol].epsilon,
                        'training_episodes': self.training_history[symbol]['episodes'],
                        'strategy_type': 'dqn_rl'
                    }
                }
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error converting action to signal for {symbol}: {e}")
            return None
    
    async def _should_train(self, symbol: str) -> bool:
        """Determine if agent should be trained."""
        last_trained = self.last_training.get(symbol)
        if last_trained is None:
            return True
        
        # Retrain every 12 hours
        hours_since_training = (datetime.utcnow() - last_trained).total_seconds() / 3600
        return hours_since_training > 12
    
    async def _train_agent(self, symbol: str):
        """Train RL agent on historical data."""
        try:
            self.logger.info(f"🎓 Training RL agent for {symbol}...")
            
            # Get historical data
            if not self.exchange_adapter:
                return
            
            ohlcv_data = await self.exchange_adapter.fetch_ohlcv(
                symbol=symbol,
                timeframe=self.timeframe,
                limit=self.lookback_periods
            )
            
            if not ohlcv_data or len(ohlcv_data) < 100:
                self.logger.warning(f"Insufficient data for training {symbol}")
                return
            
            # Prepare features
            features_df = await self._prepare_features(symbol, ohlcv_data)
            if features_df is None or len(features_df) < 50:
                return
            
            # Create environment
            env = TradingEnvironment(features_df)
            
            total_rewards = []
            
            # Training episodes
            for episode in range(self.training_episodes):
                state = env.reset()
                total_reward = 0
                
                while True:
                    # Choose action
                    action = self.agents[symbol].act(state, training=True)
                    
                    # Execute action
                    next_state, reward, done = env.step(action)
                    
                    # Store experience
                    self.agents[symbol].remember(state, action, reward, next_state, done)
                    
                    state = next_state
                    total_reward += reward
                    
                    if done:
                        break
                
                total_rewards.append(total_reward)
                
                # Train the agent
                if len(self.agents[symbol].memory) > self.batch_size:
                    self.agents[symbol].replay(self.batch_size)
                
                # Update target network periodically
                if episode % 10 == 0:
                    self.agents[symbol].update_target_network()
            
            # Save model
            await self._save_agent(symbol)
            
            # Update training history
            avg_reward = np.mean(total_rewards[-10:]) if total_rewards else 0
            self.training_history[symbol].update({
                'episodes': self.training_history[symbol]['episodes'] + self.training_episodes,
                'total_reward': avg_reward,
                'last_trained': datetime.utcnow()
            })
            self.last_training[symbol] = datetime.utcnow()
            
            self.logger.info(f"📈 RL training complete for {symbol}")
            self.logger.info(f"   Episodes: {self.training_episodes}")
            self.logger.info(f"   Avg Reward: {avg_reward:.4f}")
            self.logger.info(f"   Epsilon: {self.agents[symbol].epsilon:.4f}")
            
        except Exception as e:
            self.logger.error(f"Error training RL agent for {symbol}: {e}")
    
    async def _save_agent(self, symbol: str):
        """Save trained agent."""
        try:
            model_path = self.model_dir / f"{symbol}_dqn_model.h5"
            self.agents[symbol].q_network.save(str(model_path))
            self.logger.debug(f"💾 Saved DQN model for {symbol}")
            
        except Exception as e:
            self.logger.warning(f"Could not save DQN model for {symbol}: {e}")
    
    async def on_position_update(self, event: Dict):
        """Update position tracking."""
        data = event.get('data', {})
        symbol = data.get('symbol', '')
        position = data.get('position', {})
        
        if symbol:
            # Update position tracking (1 for long, -1 for short, 0 for none)
            quantity = float(position.get('quantity', 0))
            if abs(quantity) > 0.0001:
                self.current_positions[symbol] = 1 if quantity > 0 else -1
            else:
                self.current_positions[symbol] = 0
    
    def get_status(self) -> Dict:
        """Get strategy status."""
        status = super().get_status()
        status.update({
            'rl_algorithm': 'dqn',
            'state_size': self.state_size,
            'action_size': self.action_size,
            'training_history': self.training_history,
            'model_dir': str(self.model_dir),
            'tensorflow_available': TENSORFLOW_AVAILABLE
        })
        return status
    
    def get_strategy_info(self) -> Dict[str, Any]:
        """
        Get strategy information and performance metrics.
        Matches original implementation.
        """
        return {
            "name": self.name,
            "type": "reinforcement_learning",
            "rl_algorithm": "dqn",
            "state_size": self.state_size,
            "action_size": self.action_size,
            "symbols": self.symbols,
            "agents_trained": len([s for s in self.symbols if s in self.training_history]),
            "training_history": self.training_history,
            "last_actions": self.last_actions,
            "status": "active" if not self.is_paused else "paused",
            "enabled": True,
            "model_dir": str(self.model_dir),
            "tensorflow_available": TENSORFLOW_AVAILABLE
        }

