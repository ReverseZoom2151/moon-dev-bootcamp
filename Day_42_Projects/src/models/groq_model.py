"""
🌙 Moon Dev's Groq Model Implementation
Built with love by Moon Dev 🚀
"""

from groq import Groq
from termcolor import cprint
from .base_model import BaseModel, ModelResponse
import logging
import traceback
from typing import Optional

# Set up module-level logger
logger = logging.getLogger(__name__)

class GroqModel(BaseModel):
    """Implementation for Groq's LLM models, conforming to the BaseModel interface."""
    
    # Updated based on API response from logs (2025-05-12)
    # Note: Prices might need verification from Groq's official documentation
    AVAILABLE_MODELS = {
        "gemma2-9b-it": {
            "description": "Google Gemma 2 9B - Production - 8k context",
            "input_price": "$0.10/1M tokens",
            "output_price": "$0.10/1M tokens"
        },
        "llama-3.3-70b-versatile": {
            "description": "Llama 3.3 70B Versatile - Production - 128k context",
            "input_price": "$0.70/1M tokens",
            "output_price": "$0.90/1M tokens"
        },
        "llama-3.1-8b-instant": {
            "description": "Llama 3.1 8B Instant - Production - 128k context",
            "input_price": "$0.10/1M tokens",
            "output_price": "$0.10/1M tokens"
        },
        "llama-guard-3-8b": {
            "description": "Llama Guard 3 8B - Production - 8k context",
            "input_price": "$0.20/1M tokens",
            "output_price": "$0.20/1M tokens"
        },
        "llama3-70b-8192": {
            "description": "Llama 3 70B - Production - 8k context",
            "input_price": "$0.70/1M tokens",
            "output_price": "$0.90/1M tokens"
        },
        "llama3-8b-8192": {
            "description": "Llama 3 8B - Production - 8k context",
            "input_price": "$0.10/1M tokens",
            "output_price": "$0.10/1M tokens"
        },
        # Preview Models - based on logs, check Groq docs for confirmation
        "deepseek-r1-distill-llama-70b": {
            "description": "DeepSeek R1 Distill Llama 70B - Preview - 128k context",
            "input_price": "$0.70/1M tokens", # Assuming same as llama3-70b
            "output_price": "$0.90/1M tokens" # Assuming same as llama3-70b
        },
        # Other models seen in logs - details TBC from Groq docs
        "compound-beta-mini": {"description": "Groq Compound Beta Mini", "input_price": "?", "output_price": "?"},
        "whisper-large-v3-turbo": {"description": "Whisper Large v3 Turbo (ASR)", "input_price": "?", "output_price": "?"},
        "qwen-qwq-32b": {"description": "Qwen 32B", "input_price": "?", "output_price": "?"},
        "allam-2-7b": {"description": "Allam 2 7B", "input_price": "?", "output_price": "?"},
        "playai-tts-arabic": {"description": "PlayAI TTS Arabic", "input_price": "?", "output_price": "?"},
        "compound-beta": {"description": "Groq Compound Beta", "input_price": "?", "output_price": "?"},
        "meta-llama/llama-guard-4-12b": {"description": "Llama Guard 4 12B", "input_price": "?", "output_price": "?"},
        "playai-tts": {"description": "PlayAI TTS", "input_price": "?", "output_price": "?"},
        "distil-whisper-large-v3-en": {"description": "Distil Whisper Large v3 EN (ASR)", "input_price": "?", "output_price": "?"},
        "meta-llama/llama-4-maverick-17b-128e-instruct": {"description": "Llama 4 Maverick 17B Instruct", "input_price": "?", "output_price": "?"},
        "whisper-large-v3": {"description": "Whisper Large v3 (ASR)", "input_price": "?", "output_price": "?"},
        "meta-llama/llama-4-scout-17b-16e-instruct": {"description": "Llama 4 Scout 17B Instruct", "input_price": "?", "output_price": "?"},
        "mistral-saba-24b": {"description": "Mistral Saba 24B", "input_price": "?", "output_price": "?"},
    }
    
    # Default model and default max tokens for generation
    DEFAULT_MODEL_NAME = "llama3-70b-8192" # Changed from mixtral-8x7b-32768
    DEFAULT_MAX_TOKENS = 2048

    def __init__(self, api_key: str, model_name: Optional[str] = None, **kwargs):
        """Initialize the GroqModel with API key and selected model."""
        try:
            cprint(f"\n🌙 Moon Dev's Groq Model Initialization", "cyan")
            
            # Validate API key
            if not api_key or len(api_key.strip()) == 0:
                raise ValueError("API key is empty or None")
            
            cprint(f"🔑 API Key validation:", "cyan")
            cprint(f"  ├─ Length: {len(api_key)} chars", "cyan")
            cprint(f"  ├─ Contains whitespace: {'yes' if any(c.isspace() for c in api_key) else 'no'}", "cyan")
            cprint(f"  └─ Starts with 'gsk_': {'yes' if api_key.startswith('gsk_') else 'no'}", "cyan")
            
            # Choose model or fallback
            self._model_name = model_name or self.DEFAULT_MODEL_NAME
            if self._model_name not in self.AVAILABLE_MODELS:
                logger.warning(
                    "Invalid Groq model '%s', falling back to default '%s'",
                    self._model_name, self.DEFAULT_MODEL_NAME
                )
                self._model_name = self.DEFAULT_MODEL_NAME
            cprint(f"  └─ ✅ Model name valid", "green")
            
            # Set default max_tokens for this instance
            self._max_tokens = self.DEFAULT_MAX_TOKENS
            
            # Call parent class initialization
            cprint(f"\n📡 Parent class initialization...", "cyan")
            super().__init__(api_key, **kwargs)
            cprint(f"✅ Parent class initialized", "green")
            
        except Exception as e:
            cprint(f"\n❌ Error in Groq model initialization", "red")
            cprint(f"  ├─ Error type: {type(e).__name__}", "red")
            cprint(f"  ├─ Error message: {str(e)}", "red")
            if "api_key" in str(e).lower():
                cprint(f"  ├─ 🔑 This appears to be an API key issue", "red")
                cprint(f"  └─ Please check your GROQ_API_KEY in .env", "red")
            elif "model" in str(e).lower():
                cprint(f"  ├─ 🤖 This appears to be a model name issue", "red")
                cprint(f"  └─ Available models: {list(self.AVAILABLE_MODELS.keys())}", "red")
            raise
    
    def initialize_client(self, **kwargs) -> None:
        """Initialize the Groq client"""
        try:
            cprint(f"\n🔌 Initializing Groq client...", "cyan")
            cprint(f"  ├─ API Key length: {len(self.api_key)} chars", "cyan")
            cprint(f"  ├─ Model name: {self._model_name}", "cyan")
            
            cprint(f"\n  ├─ Creating Groq client...", "cyan")
            self.client = Groq(api_key=self.api_key)
            cprint(f"  ├─ ✅ Groq client created", "green")
            
            # Get list of available models first
            cprint(f"  ├─ Fetching available models from Groq API...", "cyan")
            available_models = self.client.models.list()
            api_models = [model.id for model in available_models.data]
            cprint(f"  ├─ Models available from API: {api_models}", "cyan")
            
            # Validate the selected model against the API list
            if self._model_name not in api_models:
                cprint(f"  ├─ ⚠️ Requested model '{self._model_name}' not found in API. Update code or config.", "red")
                # Optionally, fall back to a known *working* default if desired, 
                # but it's better to fail here if the primary choice is invalid.
                # Example fallback (use a model confirmed available in api_models):
                # fallback_model = "llama3-8b-8192" # Or another available model
                # if fallback_model in api_models:
                #     cprint(f"  ├─ Falling back to available model: {fallback_model}", "yellow")
                #     self._model_name = fallback_model
                # else:
                #     cprint(f"  ├─ ❌ Fallback model {fallback_model} also not available!", "red")
                raise ValueError(f"Requested Groq model '{self._model_name}' is not available in the API.")
            else:
                cprint(f"  ├─ ✅ Requested model '{self._model_name}' is available in API.", "green")

            # Test the connection with a simple completion using the validated model
            cprint(f"  ├─ Testing connection with model: {self._model_name}", "cyan")
            test_response = self.client.chat.completions.create(
                model=self._model_name, # Use the validated model name
                messages=[{"role": "user", "content": "Hello"}],
                max_tokens=10
            )
            cprint(f"  ├─ ✅ Test response received", "green")
            cprint(f"  ├─ Response content: {test_response.choices[0].message.content}", "cyan")
            
            model_info = self.AVAILABLE_MODELS.get(self._model_name, {})
            cprint(f"  ├─ ✨ Groq model initialized: {self._model_name}", "green")
            cprint(f"  ├─ Model info: {model_info.get('description', '')}", "cyan")
            cprint(f"  └─ Pricing: Input {model_info.get('input_price', '')} | Output {model_info.get('output_price', '')}", "yellow")
            
        except Exception as e:
            cprint(f"\n❌ Failed to initialize Groq client", "red")
            cprint(f"  ├─ Error type: {type(e).__name__}", "red")
            cprint(f"  ├─ Error message: {str(e)}", "red")
            
            # Check for specific error types
            if "api_key" in str(e).lower():
                cprint(f"  ├─ 🔑 This appears to be an API key issue", "red")
                cprint(f"  ├─ Make sure your GROQ_API_KEY is correct", "red")
                cprint(f"  └─ Key length: {len(self.api_key)} chars", "red")
            elif "model" in str(e).lower():
                cprint(f"  ├─ 🤖 This appears to be a model name issue", "red")
                cprint(f"  ├─ Requested model: {self._model_name}", "red")
                cprint(f"  └─ Available models: {list(self.AVAILABLE_MODELS.keys())}", "red")
            
            if hasattr(e, 'response'):
                cprint(f"  ├─ Response status: {e.response.status_code}", "red")
                cprint(f"  └─ Response body: {e.response.text}", "red")
            
            if hasattr(e, '__traceback__'):
                import traceback
                cprint(f"\n📋 Full traceback:", "red")
                cprint(traceback.format_exc(), "red")
            
            self.client = None
            raise
    
    def generate_response(
        self,
        system_prompt: str,
        user_content: str,
        temperature: float = 0.7,
        max_tokens_override: Optional[int] = None
    ) -> Optional[ModelResponse]:
        """
        Generate a response using the Groq model.

        Args:
            system_prompt: System-level instruction.
            user_content: User's prompt.
            temperature: Sampling temperature.
            max_tokens_override: Override the default max_tokens if provided.

        Returns:
            A ModelResponse or None if generation failed.
        """
        if not self.is_available():
            logger.error("Groq client not available. Cannot generate response.")
            return None

        try:
            # Determine generation max_tokens
            current_max_tokens = (
                max_tokens_override if max_tokens_override is not None
                else self._max_tokens
            )
            logger.debug(
                "Generating Groq response (model=%s, temp=%s, max_tokens=%s)",
                self._model_name, temperature, current_max_tokens
            )

            response = self.client.chat.completions.create(
                model=self._model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content}
                ],
                temperature=temperature,
                max_tokens=current_max_tokens,
                stream=False  # disable streaming
            )
            # Extract content and usage
            choice = response.choices[0]
            content = choice.message.content
            usage = (
                response.usage.model_dump()
                if hasattr(response.usage, 'model_dump') else response.usage
            )
            return ModelResponse(
                content=content,
                raw_response=response,
                model_name=self._model_name,
                usage=usage
            )
        except Exception as e:
            logger.error(
                "Error generating Groq completion: %s\n%s",
                e, traceback.format_exc()
            )
            return None
    
    def is_available(self) -> bool:
        """Check if Groq is available"""
        return self.client is not None
    
    @property
    def model_type(self) -> str:
        return "groq"

    @property
    def model_name(self) -> str:
        """Return the specific Groq model name being used."""
        return self._model_name

    @property
    def max_tokens(self) -> int:
        """Return the default max_tokens for generation."""
        return self._max_tokens 