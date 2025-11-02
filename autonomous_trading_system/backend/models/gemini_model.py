"""
ATS-Native Gemini Model Implementation
"""

import logging
import traceback
from typing import Optional
from .base_model import BaseModel, ModelResponse

try:
    import google.generativeai as genai
    from google.generativeai.types import GenerationConfig
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

logger = logging.getLogger(__name__)

class GeminiModel(BaseModel):
    """Implementation for Google's Gemini models."""

    AVAILABLE_MODELS = {
        "gemini-1.5-pro-latest": "Most capable model for complex tasks",
        "gemini-1.5-flash-latest": "Fast and versatile multimodal model",
    }
    DEFAULT_MODEL_NAME = "gemini-1.5-flash-latest"
    DEFAULT_MAX_TOKENS = 8192

    def __init__(self, api_key: str, model_name: Optional[str] = None, **kwargs):
        self._model_name = model_name or self.DEFAULT_MODEL_NAME
        self._max_tokens = self.DEFAULT_MAX_TOKENS
        super().__init__(api_key, **kwargs)

    def initialize_client(self, **kwargs) -> None:
        """Initialize the Gemini client."""
        if not GEMINI_AVAILABLE:
            logger.error("Google Generative AI SDK not installed. Please 'pip install google-generativeai'.")
            self.client = None
            return
        try:
            if not self.api_key:
                raise ValueError("Google API key (GEMINI_API_KEY) is required.")
            genai.configure(api_key=self.api_key)
            self.client = genai.GenerativeModel(model_name=self._model_name)
            logger.info(f"✨ Successfully initialized Gemini client for model: {self._model_name}")
        except Exception as e:
            logger.error(f"Failed to initialize Gemini client: {e}\n{traceback.format_exc()}")
            self.client = None

    def generate_response(
        self,
        system_prompt: str,
        user_content: any,
        temperature: float = 0.7,
        max_tokens_override: Optional[int] = None
    ) -> Optional[ModelResponse]:
        if not self.is_available():
            logger.error("Gemini client not available. Cannot generate response.")
            return None

        try:
            current_max_tokens = max_tokens_override if max_tokens_override is not None else self.max_tokens

            generation_config = GenerationConfig(
                temperature=temperature,
                max_output_tokens=current_max_tokens
            )
            
            # The user_content can be a string or a list of parts for multimodal
            contents = [user_content]

            response = self.client.generate_content(
                contents=contents,
                generation_config=generation_config,
                # system_instruction is in beta and might not be supported in all environments
                # For broader compatibility, we can prepend it to the user content if needed,
                # but for now we rely on the user_content structure.
                # A more robust solution could check model capabilities.
            )

            if not response.candidates or not hasattr(response.candidates[0], 'content'):
                 logger.warning(f"Gemini response might be empty or blocked.")
                 content = ""
            else:
                # Assuming the response is text-based for now
                content = "".join(part.text for part in response.candidates[0].content.parts)

            usage = {}
            if hasattr(response, 'usage_metadata'):
                usage = {
                    "prompt_tokens": response.usage_metadata.prompt_token_count,
                    "completion_tokens": response.usage_metadata.candidates_token_count,
                    "total_tokens": response.usage_metadata.total_token_count
                }

            return ModelResponse(
                content=content.strip(),
                raw_response=response,
                model_name=self.model_name,
                usage=usage
            )

        except Exception as e:
            logger.error(f"Error during Gemini API call: {e}\n{traceback.format_exc()}")
            return None

    def is_available(self) -> bool:
        return self.client is not None

    @property
    def model_type(self) -> str:
        return "gemini"

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def max_tokens(self) -> int:
        return self._max_tokens
