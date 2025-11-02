"""
🌙 Moon Dev's OpenAI Model Implementation
Built with love by Moon Dev 🚀

This module provides an interface for interacting with OpenAI models,
including specific handling for O3 and O1 models.
"""

import logging
from typing import Dict, Any, Optional, List
from openai import OpenAI, OpenAIError
from .base_model import BaseModel, ModelResponse

# Configure logging
logger = logging.getLogger(__name__)

class OpenAIModel(BaseModel):
    """Implementation for OpenAI's models with custom handling for O3/O1 types."""
    
    # Descriptions and properties of available OpenAI models supported by this class
    AVAILABLE_MODELS: Dict[str, Dict[str, Any]] = {
        "o3-mini": {
            "description": "Fast reasoning model with problem-solving capabilities",
            "input_price": "$1.10/1m tokens", # Note: Prices may change, check OpenAI docs
            "output_price": "$4.40/1m tokens",
            "supports_reasoning_effort": True
        },
        "o1": {
            "description": "Latest O1 model with reasoning capabilities",
            "input_price": "$0.01/1K tokens",
            "output_price": "$0.03/1K tokens",
            "supports_reasoning_effort": False
        },
        "o1-mini": {
            "description": "Smaller O1 model with reasoning capabilities",
            "input_price": "$0.005/1K tokens",
            "output_price": "$0.015/1K tokens",
            "supports_reasoning_effort": False
        },
        "gpt-4o": {
            "description": "Advanced GPT-4 Optimized model",
            "input_price": "$0.01/1K tokens", # Example pricing, check OpenAI docs
            "output_price": "$0.03/1K tokens",
            "supports_reasoning_effort": False
        },
        "gpt-4o-mini": {
            "description": "Efficient GPT-4 Optimized mini model",
            "input_price": "$0.005/1K tokens", # Example pricing, check OpenAI docs
            "output_price": "$0.015/1K tokens",
            "supports_reasoning_effort": False
        }
    }
    
    DEFAULT_MODEL = "gpt-4o-mini"
    DEFAULT_REASONING_EFFORT = "medium"
    DEFAULT_MAX_TOKENS = 4096

    def __init__(self, api_key: str, model_name: Optional[str] = None, reasoning_effort: Optional[str] = None, **kwargs):
        """Initializes the OpenAIModel.

        Args:
            api_key: The OpenAI API key.
            model_name: The name of the OpenAI model to use (e.g., 'gpt-4o-mini', 'o3-mini').
                        Defaults to DEFAULT_MODEL.
            reasoning_effort: The reasoning effort setting for O3 models ('low', 'medium', 'high').
                                Defaults to DEFAULT_REASONING_EFFORT.
            **kwargs: Additional arguments passed to the base class.
        """
        self._model_name = model_name if model_name else self.DEFAULT_MODEL
        self.reasoning_effort = reasoning_effort if reasoning_effort else self.DEFAULT_REASONING_EFFORT
        self._max_tokens = self.DEFAULT_MAX_TOKENS
        
        if self._model_name not in self.AVAILABLE_MODELS:
            logger.warning(f"Model '{self._model_name}' not in defined AVAILABLE_MODELS. Proceeding, but check model name.")
            # Add a basic entry if not present to avoid KeyErrors later
            self.AVAILABLE_MODELS[self._model_name] = {
                "description": "Custom or undefined model", 
                "supports_reasoning_effort": False
            }
            
        self.client: Optional[OpenAI] = None
        super().__init__(api_key=api_key)

    def initialize_client(self, **kwargs) -> None:
        """Initializes the OpenAI API client.
        
        Sets self.client to an OpenAI instance if successful, otherwise logs an error.
        """
        try:
            if not self.api_key:
                 raise ValueError("OpenAI API key is required but was not provided.")
            self.client = OpenAI(api_key=self.api_key)
            # Perform a simple check to ensure the API key is valid (optional but recommended)
            # self.client.models.list() # This call verifies authentication - costs potentially?
            logger.info(f"✨ OpenAI client initialized successfully for model: {self._model_name}")
            if self._supports_reasoning_effort():
                logger.info(f"🧠 Reasoning effort for '{self._model_name}' set to: {self.reasoning_effort}")
        except OpenAIError as e:
            logger.error(f"❌ OpenAI API error during initialization: {e}")
            self.client = None
        except Exception as e:
            logger.error(f"❌ Unexpected error initializing OpenAI client: {e}", exc_info=True)
            self.client = None
    
    @property
    def model_name(self) -> str:
        """Return the specific model name being used."""
        return self._model_name

    @property
    def max_tokens(self) -> int:
        """Return the default maximum number of tokens the model should generate."""
        return self._max_tokens
        
    def _supports_reasoning_effort(self) -> bool:
        """Checks if the currently configured model supports the 'reasoning_effort' parameter."""
        model_info = self.AVAILABLE_MODELS.get(self._model_name, {})
        return model_info.get('supports_reasoning_effort', False)

    def _prepare_model_kwargs(self, max_tokens_override: Optional[int] = None, **kwargs) -> Dict[str, Any]:
        """Prepares model-specific keyword arguments for the API call.

        Removes or modifies arguments based on the model type (O3, O1, other).

        Args:
            max_tokens_override: Optional value to override the default max_tokens.
            **kwargs: Initial keyword arguments for the API call (e.g., temperature).

        Returns:
            A dictionary of adjusted keyword arguments suitable for the model.
        """
        model_kwargs = kwargs.copy()
        
        current_max_tokens = max_tokens_override if max_tokens_override is not None else self.max_tokens

        if self._supports_reasoning_effort():
            logger.debug(f"Applying reasoning_effort='{self.reasoning_effort}' for O3 model '{self._model_name}'")
            model_kwargs["reasoning_effort"] = self.reasoning_effort
            model_kwargs.pop('max_tokens', None)
            model_kwargs.pop('max_completion_tokens', None)
            model_kwargs.pop('temperature', None)
            logger.debug(f"Removed 'max_tokens'/'max_completion_tokens' and 'temperature' for O3 model.")
        elif self._model_name.startswith('o1'):
            logger.debug(f"Adjusting parameters for O1 model '{self._model_name}'")
            model_kwargs['max_completion_tokens'] = current_max_tokens
            model_kwargs.pop('max_tokens', None)
            model_kwargs.pop('temperature', None)
            model_kwargs.pop('reasoning_effort', None)
            logger.debug(f"Set 'max_completion_tokens' to {current_max_tokens}. Removed 'temperature' and 'reasoning_effort' for O1 model.")
        else:
            logger.debug(f"Applying standard parameters for model '{self._model_name}'")
            model_kwargs['max_tokens'] = current_max_tokens
            model_kwargs.pop('reasoning_effort', None)
            model_kwargs.pop('max_completion_tokens', None)
            logger.debug(f"Set 'max_tokens' to {current_max_tokens}. Removed O1/O3 specific parameters.")
            
        logger.debug(f"Final model kwargs: {model_kwargs}")
        return model_kwargs

    def generate_response(
        self, 
        system_prompt: str, 
        user_content: str, 
        temperature: float = 0.7, 
        max_tokens_override: Optional[int] = None
    ) -> Optional[ModelResponse]:
        """Generates a response using the configured OpenAI model.

        Args:
            system_prompt: The system prompt or instructions.
            user_content: The user's input content.
            temperature: Sampling temperature.
            max_tokens_override: Optional value to override the model's default max_tokens.

        Returns:
            A ModelResponse object, or None if an error occurs.
        """
        if not self.is_available():
            logger.error("OpenAI client not initialized. Cannot generate response.")
            return None

        messages: List[Dict[str, str]]
        
        if self._model_name.startswith('o3'):
            logger.debug("Formatting messages for O3 model (using user content primarily)")
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content}
            ]
        elif self._model_name.startswith('o1'):
            logger.debug("Formatting messages for O1 model (Instructions + Input format)")
            messages = [
                {"role": "user", "content": f"Instructions: {system_prompt}\n\nInput: {user_content}"}
            ]
        else:
            logger.debug("Formatting messages with standard system/user roles")
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content}
            ]
        
        try:
            logger.info(f"Generating response with OpenAI model: {self.model_name}")
            logger.debug(f"Messages: {messages}")
            
            model_kwargs = self._prepare_model_kwargs(max_tokens_override=max_tokens_override, temperature=temperature)
            
            if self.client is None:
                 logger.error("Internal error: OpenAI client is None despite is_available check.")
                 return None

            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                **model_kwargs
            )
            
            logger.info(f"Successfully received response from {self.model_name}")
            if response.choices and response.choices[0].message:
                usage_info = response.usage.model_dump() if response.usage else {}
                return ModelResponse(
                    content=response.choices[0].message.content or "",
                    model_name=self.model_name,
                    raw_response=response.model_dump(),
                    usage=usage_info
                )
            else:
                logger.warning("OpenAI response received, but no choices or message content found.")
                return None

        except OpenAIError as e:
            logger.error(f"❌ OpenAI API error during generation: {e}")
            return None
        except Exception as e:
            logger.error(f"❌ Unexpected error during OpenAI generation: {e}", exc_info=True)
            return None
    
    def is_available(self) -> bool:
        """Checks if the OpenAI client was initialized successfully."""
        return self.client is not None
    
    @property
    def model_type(self) -> str:
        """Returns the type identifier for this model class."""
        return "openai" 