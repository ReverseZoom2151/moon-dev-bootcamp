"""
🌙 Moon Dev's Claude Model Implementation
Built with love by Moon Dev 🚀
"""

import logging
import traceback
from typing import Optional
from anthropic import Anthropic
from termcolor import cprint
from .base_model import BaseModel, ModelResponse

# Use standard logging
logger = logging.getLogger(__name__)

class ClaudeModel(BaseModel):
    """Implementation for Anthropic's Claude models, conforming to BaseModel interface."""

    # Only use the latest Claude 3.7 Sonnet model
    AVAILABLE_MODELS = {
        "claude-3.7-sonnet": "Anthropic's latest hybrid reasoning model (Claude 3.7 Sonnet)"
    }
    # Default to the 3.7 Sonnet model
    DEFAULT_MODEL_NAME = "claude-3.7-sonnet"
    DEFAULT_MAX_TOKENS = 4096 # Default for Claude 3 models

    def __init__(self, api_key: str, model_name: Optional[str] = None, **kwargs):
        """Initializes the Claude model instance."""
        # Use default if no model name provided
        self._model_name = model_name or self.DEFAULT_MODEL_NAME
        # Store default max tokens for this instance
        self._max_tokens = self.DEFAULT_MAX_TOKENS
        super().__init__(api_key, **kwargs) # Base class handles api_key and calls initialize_client

    def initialize_client(self, **kwargs) -> None:
        """Initialize the Anthropic client."""
        try:
            if not self.api_key:
                 raise ValueError("Anthropic API key is required for ClaudeModel.")
            self.client = Anthropic(api_key=self.api_key)
            # Optionally try a simple API call to confirm connectivity/key validity
            # self.client.count_tokens("test") 
            logger.info(f"Successfully initialized Claude client for model: {self._model_name}")
        except Exception as e:
            logger.error(f"Failed to initialize Claude client: {e}\n{traceback.format_exc()}")
            self.client = None

    def generate_response(
        self,
        system_prompt: str,
        user_content: str,
        temperature: float = 0.7,
        max_tokens_override: Optional[int] = None
    ) -> Optional[ModelResponse]:
        """Generate a response using the configured Claude model."""
        if not self.is_available():
            logger.error("Claude client not available. Cannot generate response.")
            return None

        try:
            # Determine the max_tokens for this specific call
            current_max_tokens = max_tokens_override if max_tokens_override is not None else self.max_tokens

            logger.debug(f"Generating Claude response with model={self.model_name}, max_tokens={current_max_tokens}, temp={temperature}")

            response = self.client.messages.create(
                model=self.model_name,
                max_tokens=current_max_tokens,
                temperature=temperature,
                system=system_prompt,
                messages=[
                    {"role": "user", "content": user_content}
                ]
            )

            content = response.content[0].text.strip() if response.content else ""
            usage = {
                "prompt_tokens": response.usage.input_tokens,
                "completion_tokens": response.usage.output_tokens
            }

            logger.debug(f"Claude response received. Usage: {usage}")

            return ModelResponse(
                content=content,
                raw_response=response,
                model_name=self.model_name,
                usage=usage
            )

        except Exception as e:
            logger.error(f"Error during Claude API call: {e}\n{traceback.format_exc()}")
            # Optionally use cprint for console visibility alongside logging
            cprint(f"❌ Claude generation error: {str(e)}", "red")
            return None # Return None as per Optional[ModelResponse]

    def is_available(self) -> bool:
        """Check if the Claude client was initialized successfully."""
        return self.client is not None

    @property
    def model_type(self) -> str:
        """Return the general model type."""
        return "claude"

    @property
    def model_name(self) -> str:
        """Return the specific model name being used."""
        return self._model_name

    @property
    def max_tokens(self) -> int:
        """Return the default maximum number of tokens for this model instance."""
        return self._max_tokens 