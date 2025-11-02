"""
ATS-Native Groq Model Implementation
"""

import logging
import traceback
from typing import Optional
from .base_model import BaseModel, ModelResponse

try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False

logger = logging.getLogger(__name__)

class GroqModel(BaseModel):
    """Implementation for Groq's fast inference models."""
    
    AVAILABLE_MODELS = {
        "llama3-70b-8192": "Llama 3 70B",
        "llama3-8b-8192": "Llama 3 8B",
        "mixtral-8x7b-32768": "Mixtral 8x7B",
        "gemma-7b-it": "Gemma 7B",
    }
    
    DEFAULT_MODEL_NAME = "llama3-8b-8192"
    DEFAULT_MAX_TOKENS = 8192

    def __init__(self, api_key: str, model_name: Optional[str] = None, **kwargs):
        self._model_name = model_name or self.DEFAULT_MODEL_NAME
        self._max_tokens = self.DEFAULT_MAX_TOKENS
        super().__init__(api_key, **kwargs)
    
    def initialize_client(self, **kwargs) -> None:
        """Initialize the Groq client."""
        if not GROQ_AVAILABLE:
            logger.error("Groq SDK not installed. Please run 'pip install groq'.")
            self.client = None
            return
        try:
            if not self.api_key:
                raise ValueError("Groq API key is required.")
            self.client = Groq(api_key=self.api_key)
            logger.info(f"✨ Groq client initialized successfully for model: {self._model_name}")
        except Exception as e:
            logger.error(f"Failed to initialize Groq client: {e}\n{traceback.format_exc()}")
            self.client = None
    
    def generate_response(
        self,
        system_prompt: str,
        user_content: any,
        temperature: float = 0.7,
        max_tokens_override: Optional[int] = None
    ) -> Optional[ModelResponse]:
        if not self.is_available():
            logger.error("Groq client not available. Cannot generate response.")
            return None

        if not isinstance(user_content, str):
            logger.error("Groq model currently only supports string user_content.")
            return None

        try:
            current_max_tokens = max_tokens_override if max_tokens_override is not None else self.max_tokens

            response = self.client.chat.completions.create(
                model=self._model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content}
                ],
                temperature=temperature,
                max_tokens=current_max_tokens,
                stream=False
            )
            
            content = response.choices[0].message.content
            usage = response.usage.model_dump() if hasattr(response.usage, 'model_dump') else None
            
            return ModelResponse(
                content=content,
                raw_response=response,
                model_name=self._model_name,
                usage=usage
            )
        except Exception as e:
            logger.error(f"Error generating Groq completion: {e}\n{traceback.format_exc()}")
            return None
    
    def is_available(self) -> bool:
        return self.client is not None
    
    @property
    def model_type(self) -> str:
        return "groq"

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def max_tokens(self) -> int:
        return self._max_tokens