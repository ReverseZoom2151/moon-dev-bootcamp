"""
🌙 Moon Dev's DeepSeek Model Implementation
Built with love by Moon Dev 🚀
"""

import logging
import traceback
from typing import Optional
from openai import OpenAI # DeepSeek uses OpenAI-compatible API
from termcolor import cprint # Keep cprint for visible errors if desired
from .base_model import BaseModel, ModelResponse

# Use standard logging
logger = logging.getLogger(__name__)

class DeepSeekModel(BaseModel):
    """Implementation for DeepSeek models, conforming to BaseModel interface."""

    # Common model names for the DeepSeek API endpoint
    # Note: Check DeepSeek documentation for the absolute latest model identifiers.
    AVAILABLE_MODELS = {
        "deepseek-chat": "General chat model (likely V3 based)",
        "deepseek-coder": "Code generation/completion model",
        "deepseek-r1": "Reasoning-focused model (R1 series)" # Adding R1 based on web search
        # "deepseek-v3-..." # Add V3 identifiers if confirmed for the API endpoint
    }
    DEFAULT_MODEL_NAME = "deepseek-chat"
    DEFAULT_MAX_TOKENS = 8192 # DeepSeek models typically support larger contexts
    BASE_URL = "https://api.deepseek.com"

    def __init__(self, api_key: str, model_name: Optional[str] = None, **kwargs):
        """Initializes the DeepSeek model instance."""
        self._model_name = model_name or self.DEFAULT_MODEL_NAME
        self._max_tokens = self.DEFAULT_MAX_TOKENS
        # Pass base_url specific to DeepSeek during initialization
        # The base class __init__ will call initialize_client
        super().__init__(api_key, base_url=self.BASE_URL, **kwargs)

    def initialize_client(self, base_url: str, **kwargs) -> None:
        """Initialize the DeepSeek client (using OpenAI SDK)."""
        try:
            if not self.api_key:
                raise ValueError("DeepSeek API key is required.")
            self.client = OpenAI(
                api_key=self.api_key,
                base_url=base_url # Use the passed base_url
            )
            # Test connection (optional, but good practice)
            # self.client.models.list() # Throws error if key/URL is bad
            logger.info(f"Successfully initialized DeepSeek client for model: {self._model_name} at {base_url}")
        except Exception as e:
            logger.error(f"Failed to initialize DeepSeek client: {e}\n{traceback.format_exc()}")
            self.client = None

    def generate_response(
        self,
        system_prompt: str,
        user_content: str,
        temperature: float = 0.7,
        max_tokens_override: Optional[int] = None
    ) -> Optional[ModelResponse]:
        """Generate a response using the configured DeepSeek model."""
        if not self.is_available():
            logger.error("DeepSeek client not available. Cannot generate response.")
            return None

        try:
            current_max_tokens = max_tokens_override if max_tokens_override is not None else self.max_tokens

            logger.debug(f"Generating DeepSeek response with model={self.model_name}, max_tokens={current_max_tokens}, temp={temperature}")

            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content}
                ],
                temperature=temperature,
                max_tokens=current_max_tokens,
                stream=False
            )

            content = response.choices[0].message.content.strip() if response.choices else ""
            # DeepSeek API usage format matches OpenAI
            usage = response.usage.model_dump() if hasattr(response, 'usage') and response.usage else None

            logger.debug(f"DeepSeek response received. Usage: {usage}")

            return ModelResponse(
                content=content,
                raw_response=response,
                model_name=self.model_name,
                usage=usage
            )

        except Exception as e:
            logger.error(f"Error during DeepSeek API call: {e}\n{traceback.format_exc()}")
            cprint(f"❌ DeepSeek generation error: {str(e)}", "red")
            return None

    def is_available(self) -> bool:
        """Check if the DeepSeek client was initialized successfully."""
        return self.client is not None

    @property
    def model_type(self) -> str:
        """Return the general model type."""
        return "deepseek"

    @property
    def model_name(self) -> str:
        """Return the specific model name being used."""
        return self._model_name

    @property
    def max_tokens(self) -> int:
        """Return the default maximum number of tokens for this model instance."""
        return self._max_tokens 