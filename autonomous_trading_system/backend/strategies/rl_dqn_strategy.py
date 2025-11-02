"""
Deep Q-Network (DQN) Reinforcement Learning Strategy
Uses RL to learn optimal trading actions through environment interaction
"""

import logging
import numpy as np
import pandas as pd
import random
import os
from typing import Dict, Any, Optional, Tuple
from datetime import datetime
from collections import deque
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout
from keras.optimizers import Adam
from strategies.base_strategy import BaseStrategy, StrategySignal, SignalAction, TechnicalIndicatorMixin

logger = logging.getLogger(__name__)

class TradingEnvironment:
    """Trading environment for RL agent"""
    
    def __init__(self, data: pd.DataFrame, initial_balance: float = 10000):
        self.data = data.reset_index(drop=True)
        self.initial_balance = initial_balance
        self.reset()
    
    def reset(self):
        """Reset environment to initial state"""
        self.current_step = 0
        self.balance = self.initial_balance
        self.position = 0  # 0: no position, 1: long, -1: short
        self.entry_price = 0
        self.total_reward = 0
        self.trade_history = []
        return self._get_state()
    
    def _get_state(self) -> np.ndarray:
        """Get current state representation"""
        if self.current_step >= len(self.data):
            return np.zeros(10)  # Default state
        
        row = self.data.iloc[self.current_step]
        
        # Technical indicators as state
        state = np.array([
            row.get('close', 0),
            row.get('volume', 0),
            row.get('rsi', 50),
            row.get('macd', 0),
            row.get('bb_upper', 0),
            row.get('bb_lower', 0),
            self.position,
            self.balance / self.initial_balance,
            row.get('price_change', 0),
            row.get('volatility', 0)
        ])
        
        return state
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool]:
        """Execute action and return next state, reward, done"""
        if self.current_step >= len(self.data) - 1:
            return self._get_state(), 0, True
        
        current_price = self.data.iloc[self.current_step]['close']
        next_price = self.data.iloc[self.current_step + 1]['close']
        
        reward = 0
        
        # Execute action
        if action == 1 and self.position == 0:  # Buy
            self.position = 1
            self.entry_price = current_price
        elif action == 2 and self.position == 0:  # Sell
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
    """Deep Q-Network agent for trading"""
    
    def __init__(self, state_size: int, action_size: int, learning_rate: float = 0.001):
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
        """Build neural network for Q-learning"""
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
        """Update target network weights"""
        self.target_network.set_weights(self.q_network.get_weights())
    
    def remember(self, state, action, reward, next_state, done):
        """Store experience in replay buffer"""
        self.memory.append((state, action, reward, next_state, done))
    
    def act(self, state):
        """Choose action using epsilon-greedy policy"""
        if np.random.random() <= self.epsilon:
            return random.randint(0, self.action_size - 1)
        
        q_values = self.q_network.predict(state.reshape(1, -1), verbose=0)
        return np.argmax(q_values[0])
    
    def replay(self, batch_size=32):
        """Train the model on a batch of experiences"""
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
        
        # Update Q values
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


class DQNStrategy(BaseStrategy, TechnicalIndicatorMixin):
    """
    Deep Q-Network Reinforcement Learning Trading Strategy
    
    Features:
    - DQN agent learns optimal trading actions
    - Experience replay for stable learning
    - Target network for improved stability
    - Continuous learning and adaptation
    """
    
    def __init__(self, config: Dict[str, Any], market_data_manager, name: str = "DQN_RL"):
        super().__init__(config, market_data_manager, name)
        
        # RL Configuration
        self.state_size = config.get("state_size", 10)
        self.action_size = config.get("action_size", 3)  # 0: hold/close, 1: buy, 2: sell
        self.learning_rate = config.get("learning_rate", 0.001)
        self.training_episodes = config.get("training_episodes", 100)
        self.batch_size = config.get("batch_size", 32)
        
        # Trading parameters
        self.min_confidence = config.get("min_confidence", 0.6)
        self.lookback_periods = config.get("lookback_periods", 500)
        
        # Model storage
        self.agents = {}  # symbol -> agent
        self.environments = {}  # symbol -> environment
        self.model_dir = config.get("model_dir", "models/dqn")
        
        # Performance tracking
        self.training_history = {}
        self.last_actions = {}
        
        # Create model directory
        os.makedirs(self.model_dir, exist_ok=True)
        
        logger.info(f"🎮 DQN RL Strategy initialized:")
        logger.info(f"   State Size: {self.state_size}")
        logger.info(f"   Action Size: {self.action_size}")
        logger.info(f"   Learning Rate: {self.learning_rate}")
        logger.info(f"   Symbols: {self.symbols}")
    
    async def _initialize_strategy(self):
        """Initialize RL agents for each symbol"""
        try:
            for symbol in self.symbols:
                await self._initialize_agent(symbol)
            
            logger.info("✅ DQN RL strategy validation complete")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize DQN strategy: {e}")
            raise
    
    async def _initialize_agent(self, symbol: str):
        """Initialize DQN agent for symbol"""
        try:
            # Create agent
            self.agents[symbol] = DQNAgent(
                state_size=self.state_size,
                action_size=self.action_size,
                learning_rate=self.learning_rate
            )
            
            # Try to load existing model
            model_path = os.path.join(self.model_dir, f"{symbol}_dqn_model.h5")
            if os.path.exists(model_path):
                self.agents[symbol].q_network = load_model(model_path)
                self.agents[symbol].update_target_network()
                logger.info(f"📁 Loaded existing DQN model for {symbol}")
            else:
                logger.info(f"🆕 Created new DQN agent for {symbol}")
            
            # Initialize training history
            self.training_history[symbol] = {
                'episodes': 0,
                'total_reward': 0,
                'last_trained': None
            }
            
        except Exception as e:
            logger.error(f"❌ Error initializing agent for {symbol}: {e}")
    
    async def generate_signal(self) -> Optional[StrategySignal]:
        """Generate RL-based trading signal"""
        try:
            for symbol in self.symbols:
                signal = await self._analyze_symbol_rl(symbol)
                if signal and signal.action != SignalAction.HOLD:
                    return signal
            
            return None
            
        except Exception as e:
            logger.error(f"❌ Error generating RL signal: {e}", exc_info=True)
            return None
    
    async def _analyze_symbol_rl(self, symbol: str) -> Optional[StrategySignal]:
        """Analyze symbol using RL agent"""
        try:
            # Get market data
            data = await self._get_market_data(symbol, limit=self.lookback_periods)
            if data is None or len(data) < 100:
                logger.warning(f"⚠️ Insufficient data for RL analysis {symbol}: {len(data) if data is not None else 0}")
                return None
            
            # Prepare features
            features_df = await self._prepare_features(symbol, data)
            if features_df is None or len(features_df) < 50:
                return None
            
            # Train agent if needed
            if await self._should_train(symbol):
                await self._train_agent(symbol, features_df)
            
            # Get current state
            current_state = self._get_current_state(features_df)
            if current_state is None:
                return None
            
            # Get action from agent
            action = self.agents[symbol].act(current_state)
            
            # Convert action to signal
            signal = await self._convert_action_to_signal(symbol, action, features_df, current_state)
            return signal
            
        except Exception as e:
            logger.error(f"❌ Error in RL analysis for {symbol}: {e}")
            return None
    
    async def _prepare_features(self, symbol: str, data: Any) -> Optional[pd.DataFrame]:
        """Prepare feature matrix for RL"""
        try:
            df = pd.DataFrame(data)
            
            # Calculate technical indicators
            df['sma_20'] = self.calculate_sma(df['close'], 20)
            df['ema_12'] = self.calculate_ema(df['close'], 12)
            df['rsi'] = self.calculate_rsi(df['close'], 14)
            
            # MACD
            macd_line, macd_signal, macd_histogram = self.calculate_macd(df['close'])
            df['macd'] = macd_line
            
            # Bollinger Bands
            bb_upper, bb_middle, bb_lower = self.calculate_bollinger_bands(df['close'], 20, 2)
            df['bb_upper'] = bb_upper
            df['bb_lower'] = bb_lower
            
            # Price features
            df['price_change'] = df['close'].pct_change()
            df['volatility'] = df['price_change'].rolling(window=20).std()
            
            # Normalize features
            for col in ['close', 'volume', 'rsi', 'macd', 'bb_upper', 'bb_lower']:
                if col in df.columns:
                    df[col] = (df[col] - df[col].mean()) / (df[col].std() + 1e-8)
            
            # Drop NaN values
            df = df.dropna()
            
            logger.debug(f"📊 Prepared {len(df)} feature rows for RL {symbol}")
            return df
            
        except Exception as e:
            logger.error(f"❌ Error preparing RL features for {symbol}: {e}")
            return None
    
    async def _train_agent(self, symbol: str, features_df: pd.DataFrame):
        """Train RL agent on historical data"""
        try:
            logger.info(f"🎓 Training RL agent for {symbol}...")
            
            # Create environment
            env = TradingEnvironment(features_df)
            
            total_rewards = []
            
            # Training episodes
            for episode in range(self.training_episodes):
                state = env.reset()
                total_reward = 0
                
                while True:
                    # Choose action
                    action = self.agents[symbol].act(state)
                    
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
            
            logger.info(f"📈 RL training complete for {symbol}")
            logger.info(f"   Episodes: {self.training_episodes}")
            logger.info(f"   Avg Reward: {avg_reward:.4f}")
            logger.info(f"   Epsilon: {self.agents[symbol].epsilon:.4f}")
            
        except Exception as e:
            logger.error(f"❌ Error training RL agent for {symbol}: {e}")
    
    def _get_current_state(self, features_df: pd.DataFrame) -> Optional[np.ndarray]:
        """Get current state for RL agent"""
        try:
            if len(features_df) == 0:
                return None
            
            row = features_df.iloc[-1]
            
            state = np.array([
                row.get('close', 0),
                row.get('volume', 0),
                row.get('rsi', 50),
                row.get('macd', 0),
                row.get('bb_upper', 0),
                row.get('bb_lower', 0),
                0,  # position (will be managed externally)
                1,  # balance ratio
                row.get('price_change', 0),
                row.get('volatility', 0)
            ])
            
            return state
            
        except Exception as e:
            logger.error(f"❌ Error getting current state: {e}")
            return None
    
    async def _convert_action_to_signal(
        self, 
        symbol: str, 
        action: int, 
        features_df: pd.DataFrame,
        state: np.ndarray
    ) -> Optional[StrategySignal]:
        """Convert RL action to trading signal"""
        try:
            current_price = float(features_df.iloc[-1]['close'])
            
            # Map actions: 0=hold/close, 1=buy, 2=sell
            if action == 1:
                signal_action = SignalAction.BUY
            elif action == 2:
                signal_action = SignalAction.SELL
            else:
                return None  # Hold action
            
            # Calculate confidence based on Q-values
            q_values = self.agents[symbol].q_network.predict(state.reshape(1, -1), verbose=0)[0]
            max_q = np.max(q_values)
            confidence = min(max_q / 10.0, 1.0) if max_q > 0 else 0.5
            
            # Only generate signal if confidence is above threshold
            if confidence < self.min_confidence:
                return None
            
            # Store last action
            self.last_actions[symbol] = {
                'action': action,
                'timestamp': datetime.utcnow(),
                'confidence': confidence
            }
            
            # Create metadata
            metadata = {
                'rl_action': action,
                'rl_confidence': confidence,
                'agent_epsilon': self.agents[symbol].epsilon,
                'training_episodes': self.training_history[symbol]['episodes'],
                'strategy_type': 'dqn_rl'
            }
            
            # Create signal
            signal = self._create_signal(
                symbol=symbol,
                action=signal_action,
                price=current_price,
                confidence=confidence,
                metadata=metadata
            )
            
            logger.info(f"🎮 RL Signal: {signal_action.value} {symbol} @ {current_price:.4f}")
            logger.info(f"   Action: {action}, Confidence: {confidence:.2f}")
            
            return signal
            
        except Exception as e:
            logger.error(f"❌ Error converting action to signal for {symbol}: {e}")
            return None
    
    async def _should_train(self, symbol: str) -> bool:
        """Determine if agent should be trained"""
        last_trained = self.training_history[symbol].get('last_trained')
        if last_trained is None:
            return True
        
        # Retrain every 12 hours
        hours_since_training = (datetime.utcnow() - last_trained).total_seconds() / 3600
        return hours_since_training > 12
    
    async def _save_agent(self, symbol: str):
        """Save trained agent"""
        try:
            model_path = os.path.join(self.model_dir, f"{symbol}_dqn_model.h5")
            self.agents[symbol].q_network.save(model_path)
            logger.debug(f"💾 Saved DQN model for {symbol}")
            
        except Exception as e:
            logger.warning(f"⚠️ Could not save DQN model for {symbol}: {e}")
    
    def get_strategy_info(self) -> Dict[str, Any]:
        """Get strategy information and performance metrics"""
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
            "status": self.status.value,
            "enabled": self.enabled
        } 