"""
ATS-Native AI Model Factory
"""

import logging
from typing import Dict, Optional, List, Type
from core.config import get_settings, Settings
from .base_model import BaseModel
from .claude_model import ClaudeModel, ANTHROPIC_AVAILABLE
from .groq_model import GroqModel, GROQ_AVAILABLE
from .openai_model import OpenAIModel, OPENAI_AVAILABLE
from .gemini_model import GeminiModel, GEMINI_AVAILABLE
from .deepseek_model import DeepSeekModel, OPENAI_SDK_AVAILABLE as DEEPSEEK_AVAILABLE
from .ollama_model import OllamaModel

logger = logging.getLogger(__name__)

class ModelFactory:
    MODEL_IMPLEMENTATIONS: Dict[str, Type[BaseModel]] = {
        "claude": ClaudeModel,
        "groq": GroqModel,
        "openai": OpenAIModel,
        "gemini": GeminiModel,
        "deepseek": DeepSeekModel,
        "ollama": OllamaModel
    }
    
    SDK_AVAILABILITY: Dict[str, bool] = {
        "claude": ANTHROPIC_AVAILABLE,
        "groq": GROQ_AVAILABLE,
        "openai": OPENAI_AVAILABLE,
        "gemini": GEMINI_AVAILABLE,
        "deepseek": DEEPSEEK_AVAILABLE,
        "ollama": True  # Depends on a running service, not a library
    }

    def __init__(self, settings: Optional[Settings] = None):
        logger.info("🏗️ Creating new ModelFactory instance...")
        self.settings = settings or get_settings()
        self._models: Dict[str, BaseModel] = {}
        self._initialize_models()
    
    def _initialize_models(self) -> None:
        """Initialize all available models based on API keys and SDKs."""
        logger.info("🏭 Initializing available AI models...")
        
        # API-based models
        key_map = self._get_api_key_mapping()
        for model_type, key_attr in key_map.items():
            if self.SDK_AVAILABILITY.get(model_type, False):
                api_key = getattr(self.settings, key_attr, None)
                if api_key:
                    try:
                        instance = self.MODEL_IMPLEMENTATIONS[model_type](api_key=api_key)
                        if instance.is_available():
                            self._models[model_type] = instance
                            logger.info(f"✨ Successfully initialized {model_type} model.")
                        else:
                            logger.warning(f"⚠️ {model_type} model initialized but not available. Check API key or service status.")
                    except Exception as e:
                        logger.error(f"❌ Failed to initialize {model_type} model: {e}")
                else:
                    logger.info(f"ℹ️ {model_type} API key not configured, skipping.")
            else:
                logger.info(f"ℹ️ {model_type} SDK not installed, skipping.")

        # Ollama (local service)
        try:
            ollama_instance = OllamaModel()
            if ollama_instance.is_available():
                self._models["ollama"] = ollama_instance
                logger.info("✨ Successfully initialized Ollama model.")
            else:
                logger.info("ℹ️ Ollama server not reachable, skipping.")
        except Exception as e:
            logger.error(f"❌ Failed to initialize Ollama model: {e}")

        logger.info(f"✅ Model initialization complete. Available models: {list(self._models.keys())}")

    def get_model(self, model_type: str) -> Optional[BaseModel]:
        """Get a specific model instance."""
        model = self._models.get(model_type)
        if not model:
            logger.warning(f"Model type '{model_type}' not available or not initialized.")
            return None
        return model
    
    def _get_api_key_mapping(self) -> Dict[str, str]:
        """Maps model types to their corresponding attribute name in the Settings object."""
        return {
            "claude": "ANTHROPIC_API_KEY",
            "groq": "GROQ_API_KEY",
            "openai": "OPENAI_API_KEY",
            "gemini": "GEMINI_API_KEY",
            "deepseek": "DEEPSEEK_API_KEY",
        }
    
    @property
    def available_models(self) -> List[str]:
        return list(self._models.keys())

# Singleton instance
_model_factory_instance: Optional[ModelFactory] = None

def get_model_factory():
    """Provides a singleton instance of the ModelFactory."""
    global _model_factory_instance
    if _model_factory_instance is None:
        _model_factory_instance = ModelFactory()
    return _model_factory_instance
