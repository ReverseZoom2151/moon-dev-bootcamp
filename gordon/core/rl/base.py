"""
RL Base Component
==================
Base class and utilities for all RL components in Gordon.
"""

import logging
import numpy as np
from typing import Dict, Optional, Any, List
from pathlib import Path
from abc import ABC, abstractmethod
from collections import deque
import os

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

logger = logging.getLogger(__name__)


class DQNAgent:
    """Reusable DQN agent for RL components."""
    
    def __init__(
        self,
        state_size: int,
        action_size: int,
        learning_rate: float = 0.001,
        hidden_layers: List[int] = None
    ):
        """
        Initialize DQN agent.
        
        Args:
            state_size: Size of state vector
            action_size: Number of possible actions
            learning_rate: Learning rate for optimizer
            hidden_layers: List of hidden layer sizes (default: [64, 32, 16])
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
        
        if hidden_layers is None:
            hidden_layers = [64, 32, 16]
        
        self.q_network = self._build_model(hidden_layers)
        self.target_network = self._build_model(hidden_layers)
        self.update_target_network()
    
    def _build_model(self, hidden_layers: List[int]):
        """Build neural network for Q-learning."""
        model = Sequential()
        
        # First layer
        model.add(Dense(hidden_layers[0], input_dim=self.state_size, activation='relu'))
        model.add(Dropout(0.2))
        
        # Hidden layers
        for size in hidden_layers[1:]:
            model.add(Dense(size, activation='relu'))
            model.add(Dropout(0.2))
        
        # Output layer
        model.add(Dense(self.action_size, activation='linear'))
        
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
        """Choose action using epsilon-greedy policy."""
        if training and np.random.random() <= self.epsilon:
            return np.random.randint(0, self.action_size)
        
        q_values = self.q_network.predict(state.reshape(1, -1), verbose=0)
        return np.argmax(q_values[0])
    
    def replay(self, batch_size: int = 32, gamma: float = 0.95):
        """Train the model on a batch of experiences."""
        if len(self.memory) < batch_size:
            return
        
        import random
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
                current_q_values[i][actions[i]] = rewards[i] + gamma * np.max(next_q_values[i])
        
        # Train the model
        self.q_network.fit(states, current_q_values, epochs=1, verbose=0)
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def save(self, filepath: str):
        """Save model to file."""
        self.q_network.save(filepath)
    
    def load(self, filepath: str):
        """Load model from file."""
        self.q_network = load_model(filepath)
        self.update_target_network()


class BaseRLComponent(ABC):
    """Base class for all RL components in Gordon."""
    
    def __init__(
        self,
        name: str,
        state_size: int,
        action_size: int,
        model_dir: str = "./models/rl",
        config: Optional[Dict] = None
    ):
        """
        Initialize base RL component.
        
        Args:
            name: Component name
            state_size: Size of state vector
            action_size: Number of possible actions
            model_dir: Directory to save/load models
            config: Configuration dictionary
        """
        self.name = name
        self.state_size = state_size
        self.action_size = action_size
        self.config = config or {}
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        # Training parameters
        self.training = self.config.get('training', False)
        self.training_episodes = self.config.get('training_episodes', 100)
        self.batch_size = self.config.get('batch_size', 32)
        self.learning_rate = self.config.get('learning_rate', 0.001)
        
        # Initialize agent
        if TENSORFLOW_AVAILABLE:
            self.agent = DQNAgent(
                state_size=state_size,
                action_size=action_size,
                learning_rate=self.learning_rate,
                hidden_layers=self.config.get('hidden_layers', [64, 32, 16])
            )
            self._load_model()
        else:
            self.agent = None
            logger.warning(f"TensorFlow not available. {name} will not work.")
        
        # Performance tracking
        self.training_history: List[Dict] = []
        self.last_prediction: Optional[Dict] = None
        
        logger.info(f"{name} RL component initialized")
    
    @abstractmethod
    def get_state(self, context: Dict[str, Any]) -> np.ndarray:
        """
        Extract state vector from context.
        
        Args:
            context: Context dictionary with market data, strategy signals, etc.
            
        Returns:
            State vector as numpy array
        """
        pass
    
    @abstractmethod
    def calculate_reward(self, action: int, context: Dict[str, Any], result: Dict[str, Any]) -> float:
        """
        Calculate reward for action taken.
        
        Args:
            action: Action taken
            context: Context at time of action
            result: Result of action (performance, PnL, etc.)
            
        Returns:
            Reward value
        """
        pass
    
    def predict(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Predict action based on current state.
        
        Args:
            context: Context dictionary
            
        Returns:
            Prediction dictionary with action and confidence
        """
        if not self.agent:
            return {'action': 0, 'confidence': 0.0, 'error': 'TensorFlow not available'}
        
        state = self.get_state(context)
        action = self.agent.act(state, training=self.training)
        
        # Get Q-values for confidence
        q_values = self.agent.q_network.predict(state.reshape(1, -1), verbose=0)[0]
        confidence = float(np.max(q_values) / (np.abs(q_values).max() + 1e-8))
        
        prediction = {
            'action': int(action),
            'confidence': confidence,
            'q_values': q_values.tolist(),
            'state': state.tolist()
        }
        
        self.last_prediction = prediction
        return prediction
    
    def train_step(self, context: Dict[str, Any], action: int, result: Dict[str, Any]):
        """
        Train agent on a single step.
        
        Args:
            context: Context at time of action
            result: Result of action
        """
        if not self.agent:
            return
        
        state = self.get_state(context)
        reward = self.calculate_reward(action, context, result)
        
        # Get next state (simplified - in practice, this would be from next observation)
        next_state = state  # Placeholder - should be updated with actual next state
        done = result.get('done', False)
        
        # Store experience
        self.agent.remember(state, action, reward, next_state, done)
        
        # Train if enough experiences
        if len(self.agent.memory) >= self.batch_size:
            self.agent.replay(self.batch_size)
    
    def train(self, episodes: int = None):
        """
        Train agent for specified episodes.
        
        Args:
            episodes: Number of episodes (uses config default if None)
        """
        if not self.agent:
            logger.warning(f"Cannot train {self.name}: TensorFlow not available")
            return
        
        episodes = episodes or self.training_episodes
        logger.info(f"Training {self.name} for {episodes} episodes...")
        
        # Training should be implemented by subclasses
        # This is a placeholder
        self.training = True
        
        for episode in range(episodes):
            # Subclasses should implement episode logic
            pass
        
        self.training = False
        self._save_model()
        logger.info(f"Training complete for {self.name}")
    
    def _save_model(self):
        """Save model to disk."""
        if not self.agent:
            return
        
        model_path = self.model_dir / f"{self.name}_model.h5"
        self.agent.save(str(model_path))
        logger.info(f"Saved {self.name} model to {model_path}")
    
    def _load_model(self):
        """Load model from disk."""
        if not self.agent:
            return
        
        model_path = self.model_dir / f"{self.name}_model.h5"
        if model_path.exists():
            try:
                self.agent.load(str(model_path))
                logger.info(f"Loaded {self.name} model from {model_path}")
            except Exception as e:
                logger.warning(f"Failed to load {self.name} model: {e}")
    
    def get_status(self) -> Dict[str, Any]:
        """Get component status."""
        return {
            'name': self.name,
            'training': self.training,
            'agent_available': self.agent is not None,
            'memory_size': len(self.agent.memory) if self.agent else 0,
            'epsilon': self.agent.epsilon if self.agent else 0.0,
            'model_path': str(self.model_dir / f"{self.name}_model.h5")
        }

