"""
🌙 Moon Dev's Model Interface
Built with love by Moon Dev 🚀

This module defines the abstract base class for all AI model implementations.
"""

from abc import ABC, abstractmethod
from typing import Dict, Optional, Any
from dataclasses import dataclass, field

@dataclass
class ModelResponse:
    """Standardized response format for all models."""
    content: str
    model_name: str
    raw_response: Any = None # Original response object from the API
    usage: Optional[Dict] = field(default_factory=dict) # e.g., {'prompt_tokens': 10, 'completion_tokens': 50}

class BaseModel(ABC):
    """Abstract Base Class interface for all AI models."""

    def __init__(self, api_key: Optional[str] = None, **kwargs):
        """
        Initializes the base model.

        Args:
            api_key: The API key for the service (Optional for models like Ollama).
            **kwargs: Additional arguments passed to initialize_client.
        """
        self.api_key = api_key
        self.client: Any = None # Type hint as Any, specific client type varies
        self.initialize_client(**kwargs)

    @abstractmethod
    def initialize_client(self, **kwargs) -> None:
        """Initialize the specific API client for the model."""
        pass

    @abstractmethod
    def generate_response(
        self,
        system_prompt: str,
        user_content: str,
        temperature: float = 0.7,
        max_tokens_override: Optional[int] = None # Allow overriding default max_tokens
    ) -> Optional[ModelResponse]:
        """
        Generate a response from the model.

        Args:
            system_prompt: The system prompt for the model.
            user_content: The user's content/prompt.
            temperature: The sampling temperature.
            max_tokens_override: Optional value to override the model's default max_tokens.

        Returns:
            A ModelResponse object containing the generated content and metadata,
            or None if an error occurred.
        """
        # Implementation removed from base class. 
        # Subclasses must implement their specific API call logic.
        # The anti-caching nonce logic previously here should be moved 
        # to specific subclasses if needed.
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Check if the model is available and properly configured (e.g., API key valid)."""
        pass

    @property
    @abstractmethod
    def model_type(self) -> str:
        """Return the general type of the model (e.g., 'openai', 'claude')."""
        pass

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Return the specific model name being used (e.g., 'gpt-4o', 'claude-3-haiku')."""
        pass

    @property
    @abstractmethod
    def max_tokens(self) -> int:
        """Return the default maximum number of tokens the model should generate."""
        # Subclasses should implement this, returning their typical default.
        pass 