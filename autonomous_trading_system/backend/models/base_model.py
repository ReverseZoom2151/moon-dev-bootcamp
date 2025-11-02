"""
ATS-Native AI Model Interface
Defines the abstract base class for all AI model implementations in the Autonomous Trading System.
"""

from abc import ABC, abstractmethod
from typing import Dict, Optional, Any
from dataclasses import dataclass, field

@dataclass
class ModelResponse:
    """Standardized response format for all models."""
    content: str
    model_name: str
    raw_response: Any = None
    usage: Optional[Dict[str, int]] = field(default_factory=dict)

class BaseModel(ABC):
    """Abstract Base Class for all AI models."""

    def __init__(self, api_key: Optional[str] = None, **kwargs):
        """
        Initializes the base model.

        Args:
            api_key: The API key for the service.
            **kwargs: Additional arguments for client initialization.
        """
        self.api_key = api_key
        self.client: Any = None
        self.initialize_client(**kwargs)

    @abstractmethod
    def initialize_client(self, **kwargs) -> None:
        """Initialize the specific API client for the model."""
        pass

    @abstractmethod
    def generate_response(
        self,
        system_prompt: str,
        user_content: any,
        temperature: float = 0.7,
        max_tokens_override: Optional[int] = None
    ) -> Optional[ModelResponse]:
        """
        Generate a response from the model.

        Args:
            system_prompt: The system prompt.
            user_content: The user's content/prompt. Can be a string or a list for multimodal input.
            temperature: The sampling temperature.
            max_tokens_override: Optional override for max tokens.

        Returns:
            A ModelResponse object or None if an error occurred.
        """
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Check if the model is available and properly configured."""
        pass

    @property
    @abstractmethod
    def model_type(self) -> str:
        """Return the general type of the model (e.g., 'openai', 'claude')."""
        pass

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Return the specific model name being used."""
        pass

    @property
    @abstractmethod
    def max_tokens(self) -> int:
        """Return the default maximum number of tokens for generation."""
        pass
