"""
🌙 Moon Dev's Model Factory
Built with love by Moon Dev 🚀

This module manages all available AI models and provides a unified interface.
"""

import os
import logging
import traceback
from typing import Dict, Optional, List
from termcolor import cprint
from dotenv import load_dotenv
from pathlib import Path
from .base_model import BaseModel
from .claude_model import ClaudeModel
from .groq_model import GroqModel
from .openai_model import OpenAIModel
from .gemini_model import GeminiModel
from .deepseek_model import DeepSeekModel
from .ollama_model import OllamaModel

logger = logging.getLogger(__name__)

class ModelFactory:
    """Factory for creating and managing AI models"""
    
    # Map model types to their implementations
    MODEL_IMPLEMENTATIONS = {
        "claude": ClaudeModel,
        "groq": GroqModel,
        "openai": OpenAIModel,
        "gemini": GeminiModel,
        "deepseek": DeepSeekModel,
        "ollama": OllamaModel
    }
    
    # Default models for each type - updated to latest available models
    DEFAULT_MODELS = {
        "claude": "claude-3-5-sonnet-20240620",  # Latest Claude 3.5 Sonnet model
        "groq": "llama-3-70b-8192",             # Llama 3 70B on Groq
        "openai": "gpt-4o-2024-05-13",          # Latest GPT-4o model with most recent training
        "gemini": "gemini-1.5-pro-latest",      # Latest Gemini 1.5 Pro model
        "deepseek": "deepseek-coder-33b-instruct", # Latest DeepSeek Coder model
        "ollama": "llama3:70b"                  # Latest Llama 3 70B local model
    }
    
    def __init__(self):
        """Initialize the model factory and load available models"""
        logger.info("🏗️ Creating new ModelFactory instance...")
        
        self._load_environment()
        self._models: Dict[str, BaseModel] = {}
        self._initialize_models()
    
    def _load_environment(self) -> None:
        """Load environment variables from .env file"""
        project_root = Path(__file__).parent.parent.parent
        env_path = project_root / '.env'
        logger.info(f"🔍 Loading environment from: {env_path}")
        load_dotenv(dotenv_path=env_path)
        logger.info("✨ Environment loaded")
    
    def _initialize_models(self) -> None:
        """Initialize all available models based on API keys"""
        initialized = False
        
        logger.info("🏭 Moon Dev's Model Factory Initialization")
        logger.info("═" * 50)
        
        # Check and log environment variables
        self._check_environment()
        
        # Initialize API-based models
        for model_type, key_name in self._get_api_key_mapping().items():
            if self._initialize_api_model(model_type, key_name):
                initialized = True
        
        # Initialize Ollama separately since it doesn't need an API key
        if self._initialize_ollama_model():
            initialized = True
        
        self._log_initialization_summary(initialized)
    
    def _check_environment(self) -> None:
        """Check and log available environment variables"""
        logger.info("🔍 Environment Check:")
        for key in ["GROQ_API_KEY", "OPENAI_KEY", "ANTHROPIC_KEY", "GEMINI_KEY", "DEEPSEEK_KEY"]:
            value = os.getenv(key)
            if value and len(value.strip()) > 0:
                logger.info(f"  ├─ {key}: Found ({len(value)} chars)")
            else:
                logger.warning(f"  ├─ {key}: Not found or empty")
    
    def _initialize_api_model(self, model_type: str, key_name: str) -> bool:
        """Initialize a specific API-based model
        
        Args:
            model_type: The type of model to initialize
            key_name: The environment variable name for the API key
            
        Returns:
            bool: True if model was successfully initialized
        """
        logger.info(f"🔄 Initializing {model_type} model...")
        logger.info(f"  ├─ Looking for {key_name}...")
        
        api_key = os.getenv(key_name)
        if not api_key:
            logger.info(f"  └─ ℹ️ {key_name} not found")
            return False
            
        try:
            logger.info(f"  ├─ Found {key_name} ({len(api_key)} chars)")
            logger.info(f"  ├─ Getting model class for {model_type}...")
            
            if model_type not in self.MODEL_IMPLEMENTATIONS:
                logger.error(f"  ├─ ❌ Model type not found in implementations!")
                logger.warning(f"  └─ Available implementations: {list(self.MODEL_IMPLEMENTATIONS.keys())}")
                return False
            
            model_class = self.MODEL_IMPLEMENTATIONS[model_type]
            logger.info(f"  ├─ Using model class: {model_class.__name__}")
            
            # Create instance with more detailed error handling
            try:
                logger.info("  ├─ Creating model instance...")
                logger.info(f"  ├─ Default model name: {self.DEFAULT_MODELS[model_type]}")
                model_instance = model_class(api_key)
                logger.info("  ├─ Model instance created")
                
                # Test if instance is properly initialized
                logger.info("  ├─ Testing model availability...")
                if model_instance.is_available():
                    self._models[model_type] = model_instance
                    logger.info(f"  └─ ✨ Successfully initialized {model_type}")
                    return True
                else:
                    logger.warning("  └─ ⚠️ Model instance created but not available")
                    return False
            except Exception as instance_error:
                logger.warning("  ├─ ⚠️ Error creating model instance")
                logger.warning(f"  ├─ Error type: {type(instance_error).__name__}")
                logger.warning(f"  └─ Error message: {str(instance_error)}")
                logger.debug(traceback.format_exc())
                return False
                
        except Exception as e:
            logger.warning(f"  ├─ ⚠️ Failed to initialize {model_type} model")
            logger.warning(f"  ├─ Error type: {type(e).__name__}")
            logger.warning(f"  └─ Error message: {str(e)}")
            logger.debug(traceback.format_exc())
            return False
    
    def _initialize_ollama_model(self) -> bool:
        """Initialize the Ollama model
        
        Returns:
            bool: True if Ollama model was successfully initialized
        """
        try:
            logger.info("🔄 Initializing Ollama model...")
            model_class = self.MODEL_IMPLEMENTATIONS["ollama"]
            model_instance = model_class(model_name=self.DEFAULT_MODELS["ollama"])
            
            if model_instance.is_available():
                self._models["ollama"] = model_instance
                logger.info("✨ Successfully initialized Ollama")
                return True
            else:
                logger.warning("⚠️ Ollama server not available - make sure 'ollama serve' is running")
                return False
        except Exception as e:
            logger.error(f"❌ Failed to initialize Ollama: {str(e)}")
            return False
    
    def _log_initialization_summary(self, initialized: bool) -> None:
        """Log a summary of the initialization process
        
        Args:
            initialized: Whether at least one model was successfully initialized
        """
        logger.info("\n" + "═" * 50)
        logger.info("📊 Initialization Summary:")
        logger.info(f"  ├─ Models attempted: {len(self._get_api_key_mapping()) + 1}")
        logger.info(f"  ├─ Models initialized: {len(self._models)}")
        logger.info(f"  └─ Available models: {list(self._models.keys())}")
        
        if not initialized:
            logger.warning("⚠️ No AI models available - check API keys and Ollama server")
            
            logger.warning("Required environment variables:")
            for model_type, key_name in self._get_api_key_mapping().items():
                logger.warning(f"  ├─ {key_name} (for {model_type})")
            cprint("  └─ Add these to your .env file 🌙", "yellow")
            cprint("\nFor Ollama:", "yellow")
            cprint("  └─ Make sure 'ollama serve' is running", "yellow")
        else:
            # Print available models
            logger.info("🤖 Available AI Models:")
            for model_type, model in self._models.items():
                logger.info(f"  ├─ {model_type}: {model.model_name}")
            logger.info("  └─ Moon Dev's Model Factory Ready! 🌙")
    
    def get_model(self, model_type: str, model_name: Optional[str] = None) -> Optional[BaseModel]:
        """Get a specific model instance
        
        Args:
            model_type: The type of model to get (e.g., 'claude', 'openai')
            model_name: Optional specific model name to use
            
        Returns:
            BaseModel instance or None if not available
        """
        logger.info(f"🔍 Requesting model: {model_type} ({model_name or 'default'})")
        
        if model_type not in self.MODEL_IMPLEMENTATIONS:
            logger.error(f"❌ Invalid model type: '{model_type}'")
            cprint("Available types:", "yellow")
            for available_type in self.MODEL_IMPLEMENTATIONS.keys():
                cprint(f"  ├─ {available_type}", "yellow")
            return None
            
        if model_type not in self._models:
            key_name = self._get_api_key_mapping().get(model_type)
            if key_name:
                cprint(f"❌ Model type '{model_type}' not available - check {key_name} in .env", "red")
            else:
                cprint(f"❌ Model type '{model_type}' not available", "red")
            return None
        
        # If no specific model requested, return the current one
        if not model_name:
            return self._models[model_type]
            
        # If requested model is different from current, reinitialize
        model = self._models[model_type]
        if model.model_name != model_name:
            return self._reinitialize_model(model_type, model_name)
            
        return model
    
    def _reinitialize_model(self, model_type: str, model_name: str) -> Optional[BaseModel]:
        """Reinitialize a model with a different model name
        
        Args:
            model_type: The type of model to reinitialize
            model_name: The new model name to use
            
        Returns:
            BaseModel instance or None if reinitialization failed
        """
        logger.info(f"🔄 Reinitializing {model_type} with model {model_name}...")
        try:
            # Special handling for Ollama models
            if model_type == "ollama":
                model = self.MODEL_IMPLEMENTATIONS[model_type](model_name=model_name)
            else:
                # For API-based models that need a key
                key_name = self._get_api_key_mapping()[model_type]
                if api_key := os.getenv(key_name):
                    logger.info(f"  ├─ Reinitializing with API key from {key_name}")
                    model = self.MODEL_IMPLEMENTATIONS[model_type](api_key, model_name=model_name)
                else:
                    cprint(f"❌ API key not found for {model_type}", "red")
                    return None
            
            if model.is_available():
                self._models[model_type] = model
                logger.info("✨ Successfully reinitialized with new model")
                return model
            else:
                logger.warning(f"⚠️ Reinitialized model {model_name} not available")
                return None
                
        except Exception as e:
            logger.error(f"❌ Failed to initialize {model_type} with model {model_name}")
            cprint(f"❌ Error type: {type(e).__name__}", "red")
            cprint(f"❌ Error: {str(e)}", "red")
            return None
    
    def _get_api_key_mapping(self) -> Dict[str, str]:
        """Get mapping of model types to their API key environment variable names
        
        Returns:
            Dictionary mapping model types to env var names
        """
        return {
            "claude": "ANTHROPIC_KEY",
            "groq": "GROQ_API_KEY",
            "openai": "OPENAI_KEY",
            "gemini": "GEMINI_KEY",
            "deepseek": "DEEPSEEK_KEY",
            # Ollama doesn't need an API key as it runs locally
        }
    
    @property
    def available_models(self) -> Dict[str, List[str]]:
        """Get all available models and their configurations
        
        Returns:
            Dictionary mapping model types to lists of available model names
        """
        logger.debug("Retrieving available models from factory...")
        return {
            model_type: model.AVAILABLE_MODELS
            for model_type, model in self._models.items()
        }
    
    def is_model_available(self, model_type: str) -> bool:
        """Check if a specific model type is available
        
        Args:
            model_type: The type of model to check
            
        Returns:
            True if model type is available, False otherwise
        """
        return model_type in self._models and self._models[model_type].is_available()

# Create a singleton instance
model_factory = ModelFactory() 