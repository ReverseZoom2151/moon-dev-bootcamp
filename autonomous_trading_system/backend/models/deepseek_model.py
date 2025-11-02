"""
ATS-Native DeepSeek Model Implementation
"""

import logging
import traceback
from typing import Optional
from .base_model import BaseModel, ModelResponse

# DeepSeek uses the OpenAI SDK
try:
    from openai import OpenAI
    OPENAI_SDK_AVAILABLE = True
except ImportError:
    OPENAI_SDK_AVAILABLE = False

logger = logging.getLogger(__name__)

class DeepSeekModel(BaseModel):
    """Implementation for DeepSeek models."""

    AVAILABLE_MODELS = {
        "deepseek-chat": "General chat model",
        "deepseek-coder": "Code generation/completion model",
    }
    DEFAULT_MODEL_NAME = "deepseek-chat"
    DEFAULT_MAX_TOKENS = 8192
    BASE_URL = "https://api.deepseek.com"

    def __init__(self, api_key: str, model_name: Optional[str] = None, **kwargs):
        self._model_name = model_name or self.DEFAULT_MODEL_NAME
        self._max_tokens = self.DEFAULT_MAX_TOKENS
        super().__init__(api_key, base_url=self.BASE_URL, **kwargs)

    def initialize_client(self, base_url: str, **kwargs) -> None:
        """Initialize the DeepSeek client (using OpenAI SDK)."""
        if not OPENAI_SDK_AVAILABLE:
            logger.error("OpenAI SDK not installed, which is required for DeepSeek. Please 'pip install openai'.")
            self.client = None
            return
        try:
            if not self.api_key:
                raise ValueError("DeepSeek API key is required.")
            self.client = OpenAI(api_key=self.api_key, base_url=base_url)
            logger.info(f"✨ Successfully initialized DeepSeek client for model: {self._model_name} at {base_url}")
        except Exception as e:
            logger.error(f"Failed to initialize DeepSeek client: {e}\n{traceback.format_exc()}")
            self.client = None

    def generate_response(
        self,
        system_prompt: str,
        user_content: any,
        temperature: float = 0.7,
        max_tokens_override: Optional[int] = None
    ) -> Optional[ModelResponse]:
        if not self.is_available():
            logger.error("DeepSeek client not available. Cannot generate response.")
            return None
        
        if not isinstance(user_content, str):
            logger.error("DeepSeek model currently only supports string user_content.")
            return None

        try:
            current_max_tokens = max_tokens_override if max_tokens_override is not None else self.max_tokens

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
            usage = response.usage.model_dump() if hasattr(response, 'usage') and response.usage else None

            return ModelResponse(
                content=content,
                raw_response=response,
                model_name=self.model_name,
                usage=usage
            )
        except Exception as e:
            logger.error(f"Error during DeepSeek API call: {e}\n{traceback.format_exc()}")
            return None

    def is_available(self) -> bool:
        return self.client is not None

    @property
    def model_type(self) -> str:
        return "deepseek"

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def max_tokens(self) -> int:
        return self._max_tokens
