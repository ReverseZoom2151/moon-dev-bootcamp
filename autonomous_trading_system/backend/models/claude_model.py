"""
ATS-Native Claude Model Implementation
"""

import logging
import traceback
from typing import Optional
from .base_model import BaseModel, ModelResponse

try:
    from anthropic import Anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

logger = logging.getLogger(__name__)

class ClaudeModel(BaseModel):
    """Implementation for Anthropic's Claude models."""

    AVAILABLE_MODELS = {
        "claude-3-5-sonnet-20240620": "Most intelligent model",
        "claude-3-opus-20240229": "Powerful, for complex analysis",
        "claude-3-sonnet-20240229": "Balanced speed and intelligence",
        "claude-3-haiku-20240307": "Fastest, most compact model",
    }
    DEFAULT_MODEL_NAME = "claude-3-5-sonnet-20240620"
    DEFAULT_MAX_TOKENS = 4096

    def __init__(self, api_key: str, model_name: Optional[str] = None, **kwargs):
        self._model_name = model_name or self.DEFAULT_MODEL_NAME
        self._max_tokens = self.DEFAULT_MAX_TOKENS
        super().__init__(api_key, **kwargs)

    def initialize_client(self, **kwargs) -> None:
        """Initialize the Anthropic client."""
        if not ANTHROPIC_AVAILABLE:
            logger.error("Anthropic SDK not installed. Please run 'pip install anthropic'.")
            self.client = None
            return
        try:
            if not self.api_key:
                 raise ValueError("Anthropic API key is required for ClaudeModel.")
            self.client = Anthropic(api_key=self.api_key)
            logger.info(f"✨ Successfully initialized Claude client for model: {self._model_name}")
        except Exception as e:
            logger.error(f"Failed to initialize Claude client: {e}\n{traceback.format_exc()}")
            self.client = None

    def generate_response(
        self,
        system_prompt: str,
        user_content: any,
        temperature: float = 0.7,
        max_tokens_override: Optional[int] = None
    ) -> Optional[ModelResponse]:
        if not self.is_available():
            logger.error("Claude client not available. Cannot generate response.")
            return None

        try:
            current_max_tokens = max_tokens_override if max_tokens_override is not None else self.max_tokens
            
            # Claude API expects a list of dictionaries for the user message
            if isinstance(user_content, str):
                messages = [{"type": "text", "text": user_content}]
            elif isinstance(user_content, list):
                messages = user_content # Assumes it's already in the correct format
            else:
                logger.error(f"Unsupported user_content type for Claude: {type(user_content)}")
                return None

            response = self.client.messages.create(
                model=self.model_name,
                max_tokens=current_max_tokens,
                temperature=temperature,
                system=system_prompt,
                messages=[{"role": "user", "content": messages}]
            )

            content = response.content[0].text.strip() if response.content else ""
            usage = {
                "prompt_tokens": response.usage.input_tokens,
                "completion_tokens": response.usage.output_tokens
            }

            return ModelResponse(
                content=content,
                raw_response=response,
                model_name=self.model_name,
                usage=usage
            )
        except Exception as e:
            logger.error(f"Error during Claude API call: {e}\n{traceback.format_exc()}")
            return None

    def is_available(self) -> bool:
        return self.client is not None

    @property
    def model_type(self) -> str:
        return "claude"

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def max_tokens(self) -> int:
        return self._max_tokens
