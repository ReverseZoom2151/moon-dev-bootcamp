"""
🌙 Moon Dev's Gemini Model Implementation
Built with love by Moon Dev 🚀
"""

import logging
import traceback
from typing import Optional

import google.generativeai as genai
from google.generativeai.types import GenerationConfig
from termcolor import cprint # Keep for visible errors
from .base_model import BaseModel, ModelResponse

# Use standard logging
logger = logging.getLogger(__name__)

class GeminiModel(BaseModel):
    """Implementation for Google's Gemini models, conforming to BaseModel interface."""

    # Supported Gemini model variants via the Google API
    AVAILABLE_MODELS = {
        # Latest 2.5 generation
        "gemini-2.5-pro": "Gemini 2.5 Pro (most accurate, multimodal)",
        "gemini-2.5-flash": "Gemini 2.5 Flash (best price-performance, multimodal)",
        # Stable 2.0 generation
        "gemini-2.0-flash": "Gemini 2.0 Flash (fast, general-purpose)",
        "gemini-2.0-flash-lite": "Gemini 2.0 Flash-Lite (cost-efficient, low-latency)",
        # Keep 1.5 generation as fallback
        "gemini-1.5-pro-latest": "Gemini 1.5 Pro (text-only, powerful)",
        "gemini-1.5-flash-latest": "Gemini 1.5 Flash (text-only, faster)"
    }
    # Default to Gemini 2.5 Flash for balanced performance and cost
    DEFAULT_MODEL_NAME = "gemini-2.5-flash"
    DEFAULT_MAX_TOKENS = 8192 # Default for most Gemini models

    def __init__(self, api_key: str, model_name: Optional[str] = None, **kwargs):
        """Initializes the Gemini model instance."""
        self._model_name = model_name or self.DEFAULT_MODEL_NAME
        self._max_tokens = self.DEFAULT_MAX_TOKENS
        super().__init__(api_key, **kwargs)

    def initialize_client(self, **kwargs) -> None:
        """Initialize the Gemini client and configure the API key."""
        try:
            if not self.api_key:
                raise ValueError("Google API key (GEMINI_KEY) is required.")
            genai.configure(api_key=self.api_key)
            # Use system_instruction if model supports it (newer models do)
            self.client = genai.GenerativeModel(
                model_name=self._model_name,
                # Set default safety settings to be less restrictive if needed, 
                # but be aware of the implications.
                # safety_settings={ 
                #     HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                #     HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                #     HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                #     HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
                # }
            )
            logger.info(f"Successfully initialized Gemini client for model: {self._model_name}")
            # Optional: Test call
            # self.client.count_tokens("test")
        except Exception as e:
            logger.error(f"Failed to initialize Gemini client: {e}\n{traceback.format_exc()}")
            self.client = None

    def generate_response(
        self,
        system_prompt: str,
        user_content: str,
        temperature: float = 0.7,
        max_tokens_override: Optional[int] = None
    ) -> Optional[ModelResponse]:
        """Generate a response using the configured Gemini model."""
        if not self.is_available():
            logger.error("Gemini client not available. Cannot generate response.")
            return None

        try:
            current_max_tokens = max_tokens_override if max_tokens_override is not None else self.max_tokens

            generation_config = GenerationConfig(
                temperature=temperature,
                max_output_tokens=current_max_tokens
            )

            logger.debug(f"Generating Gemini response with model={self.model_name}, max_tokens={current_max_tokens}, temp={temperature}")

            # Use system_instruction for newer models
            response = self.client.generate_content(
                user_content, # User prompt is the main content
                generation_config=generation_config,
                system_instruction=system_prompt # Use dedicated system instruction field
                # Add safety_settings here if needed per-request override
            )

            # Check for blocked content due to safety settings
            if not response.candidates or not hasattr(response.candidates[0], 'content'):
                 finish_reason = getattr(response.candidates[0], 'finish_reason', 'UNKNOWN')
                 safety_ratings = getattr(response.candidates[0], 'safety_ratings', [])
                 logger.warning(f"Gemini response might be empty or blocked. Finish reason: {finish_reason}. Safety: {safety_ratings}")
                 # Decide if blocked response should be an error or return empty
                 if finish_reason == 'SAFETY':
                     cprint(f"❌ Gemini response blocked due to safety settings. Ratings: {safety_ratings}", "red")
                     # Return None or an empty ModelResponse based on desired behavior
                     return None # Or return ModelResponse(content="", model_name=self.model_name, usage={}, raw_response=response)
                 # Handle other empty reasons if necessary
                 content = "" # Default to empty if not blocked by safety but still no content
            else:
                content = response.text.strip()

            # Extract token usage from metadata
            usage = {}
            if hasattr(response, 'usage_metadata'):
                usage = {
                    "prompt_tokens": response.usage_metadata.prompt_token_count,
                    "completion_tokens": response.usage_metadata.candidates_token_count,
                    "total_tokens": response.usage_metadata.total_token_count
                }

            logger.debug(f"Gemini response received. Usage: {usage}")

            return ModelResponse(
                content=content,
                raw_response=response,
                model_name=self.model_name,
                usage=usage
            )

        except Exception as e:
            logger.error(f"Error during Gemini API call: {e}\n{traceback.format_exc()}")
            cprint(f"❌ Gemini generation error: {str(e)}", "red")
            return None

    def is_available(self) -> bool:
        """Check if the Gemini client was initialized successfully."""
        return self.client is not None

    @property
    def model_type(self) -> str:
        """Return the general model type."""
        return "gemini"

    @property
    def model_name(self) -> str:
        """Return the specific model name being used."""
        return self._model_name

    @property
    def max_tokens(self) -> int:
        """Return the default maximum number of tokens for this model instance."""
        return self._max_tokens 