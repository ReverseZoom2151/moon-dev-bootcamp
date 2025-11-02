"""
ATS-Native OpenAI Model Implementation
"""

import logging
from typing import Dict, Any, Optional, List
from .base_model import BaseModel, ModelResponse

# AI and analysis imports
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

logger = logging.getLogger(__name__)

class OpenAIModel(BaseModel):
    """Implementation for OpenAI's models, including GPT-4o, GPT-4, etc."""

    AVAILABLE_MODELS: Dict[str, Dict[str, Any]] = {
        "gpt-4o": {"description": "Most advanced, multimodal model"},
        "gpt-4o-mini": {"description": "Affordable and intelligent small model"},
        "gpt-4-turbo": {"description": "High-performance model for large-scale tasks"},
        "gpt-3.5-turbo": {"description": "Fast, optimized for dialogue"},
    }
    DEFAULT_MODEL = "gpt-4o-mini"
    DEFAULT_MAX_TOKENS = 4096

    def __init__(self, api_key: str, model_name: Optional[str] = None, **kwargs):
        self._model_name = model_name or self.DEFAULT_MODEL
        self._max_tokens = self.DEFAULT_MAX_TOKENS
        
        if self._model_name not in self.AVAILABLE_MODELS:
            logger.warning(f"Model '{self._model_name}' not in defined AVAILABLE_MODELS. Proceeding, but check model name.")
        
        super().__init__(api_key=api_key, **kwargs)

    def initialize_client(self, **kwargs) -> None:
        """Initializes the OpenAI API client."""
        if not OPENAI_AVAILABLE:
            logger.error("OpenAI SDK not installed. Please run 'pip install openai'.")
            self.client = None
            return
        try:
            if not self.api_key:
                raise ValueError("OpenAI API key is required.")
            self.client = openai.OpenAI(api_key=self.api_key)
            logger.info(f"✨ OpenAI client initialized successfully for model: {self._model_name}")
        except Exception as e:
            logger.error(f"❌ Unexpected error initializing OpenAI client: {e}", exc_info=True)
            self.client = None
    
    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def max_tokens(self) -> int:
        return self._max_tokens
        
    def generate_response(
        self, 
        system_prompt: str, 
        user_content: any, 
        temperature: float = 0.7, 
        max_tokens_override: Optional[int] = None
    ) -> Optional[ModelResponse]:
        if not self.is_available():
            logger.error("OpenAI client not initialized. Cannot generate response.")
            return None

        messages: List[Dict[str, Any]] = [{"role": "system", "content": system_prompt}]
        
        if isinstance(user_content, str):
            messages.append({"role": "user", "content": user_content})
        elif isinstance(user_content, list):
            messages.append({"role": "user", "content": user_content})
        else:
            logger.error(f"Unsupported user_content type: {type(user_content)}")
            return None
        
        try:
            current_max_tokens = max_tokens_override if max_tokens_override is not None else self.max_tokens
            
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                max_tokens=current_max_tokens,
                temperature=temperature
            )
            
            if response.choices and response.choices[0].message.content:
                usage_info = response.usage.model_dump() if response.usage else {}
                return ModelResponse(
                    content=response.choices[0].message.content,
                    model_name=self.model_name,
                    raw_response=response.model_dump(),
                    usage=usage_info
                )
            else:
                logger.warning("OpenAI response received, but no choices or message content found.")
                return None

        except openai.APIError as e:
            logger.error(f"❌ OpenAI API error during generation: {e}")
            return None
        except Exception as e:
            logger.error(f"❌ Unexpected error during OpenAI generation: {e}", exc_info=True)
            return None
    
    def is_available(self) -> bool:
        return self.client is not None
    
    @property
    def model_type(self) -> str:
        return "openai"
