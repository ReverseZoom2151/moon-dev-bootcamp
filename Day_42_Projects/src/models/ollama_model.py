"""
🌙 Moon Dev's Ollama Model Integration
Built with love by Moon Dev 🚀

This module provides integration with locally running Ollama models.
"""

import requests
import json
import logging
from typing import List, Dict, Optional, Any
from .base_model import BaseModel, ModelResponse

# Configure logging
logger = logging.getLogger(__name__)

class OllamaModel(BaseModel):
    """Implementation for local Ollama models using the Ollama REST API"""
    
    # Common popular Ollama models - this is illustrative, 
    # actual availability depends on the user's Ollama setup.
    AVAILABLE_MODELS: List[str] = [
        "llama3:latest",        # Latest Meta Llama 3 (usually 8B)
        "llama3:8b",            # Explicit Meta Llama 3 8B
        "llama3:70b",           # Explicit Meta Llama 3 70B
        "phi3:latest",          # Latest Microsoft Phi-3 (usually mini 3.8B)
        "phi3:medium",          # Microsoft Phi-3 Medium 14B
        "mistral:latest",       # Latest Mistral 7B
        "mixtral:latest",       # Latest Mixtral 8x7B MoE
        "gemma:latest",         # Latest Google Gemma (usually 7B)
        "gemma:2b",             # Google Gemma 2B
        "codellama:latest",     # Latest Code Llama (usually 7B)
        "llava:latest",         # LLaVA multi-modal model
        "qwen:latest",          # Latest Qwen model (e.g., Qwen1.5-7B-Chat)
        "deepseek-coder:latest",# Latest DeepSeek Coder
    ]
    
    DEFAULT_MODEL = "llama3:latest" # A sensible default if none specified
    DEFAULT_MAX_TOKENS = 4096 # Default max tokens for Ollama models

    def __init__(self, api_key: Optional[str] = None, model_name: Optional[str] = None, base_url: str = "http://localhost:11434"):
        """Initialize Ollama model client.

        Args:
            api_key: Not used for Ollama but kept for compatibility.
            model_name: Name of the Ollama model to use (e.g., 'llama3:8b'). Defaults to DEFAULT_MODEL.
            base_url: The base URL for the Ollama API endpoint.
        """
        self.base_url = base_url.rstrip('/') + "/api"
        self._model_name = model_name if model_name else self.DEFAULT_MODEL
        self._max_tokens = self.DEFAULT_MAX_TOKENS # Set default max tokens
        
        # Use a placeholder API key for BaseModel compatibility
        # Pass the determined model name to super()
        super().__init__(api_key="LOCAL_OLLAMA") 
        # self.initialize_client() # initialize_client is called by super().__init__

    # Implement initialize_client as required by BaseModel
    def initialize_client(self, **kwargs) -> None:
        """Checks connection to Ollama and verifies the specified model exists."""
        # The connection check logic previously in __init__ is moved here
        # as this is called by the BaseModel's __init__.
        if not self.is_available():
             # is_available already logs the error, just raise here
            raise ConnectionError(f"Failed to connect to Ollama API at {self.base_url.replace('/api', '')}")
            
        logger.info("✨ Successfully connected to Ollama API.")
        
        available_models = self._get_local_models()
        if available_models:
            local_model_names = [model["name"] for model in available_models]
            logger.info(f"📚 Available local Ollama models: {local_model_names}")
            if self._model_name not in local_model_names:
                logger.warning(f"⚠️ Model '{self._model_name}' not found locally!")
                logger.warning(f"   You may need to run: ollama pull {self._model_name}")
                # Allow initialization but warn the user. API calls will fail later.
        else:
            logger.warning("⚠️ No local models found via Ollama API.")
            logger.warning(f"   Ensure '{self._model_name}' is pulled: ollama pull {self._model_name}")

    def _get_local_models(self) -> Optional[List[Dict[str, Any]]]:
        """Retrieve the list of models available locally from the Ollama API."""
        try:
            response = requests.get(f"{self.base_url}/tags", timeout=10) # Added timeout
            response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
            return response.json().get("models", [])
        except requests.exceptions.RequestException as e:
            logger.error(f"❌ Error fetching local models from Ollama: {e}")
            return None
        except json.JSONDecodeError as e:
             logger.error(f"❌ Error decoding Ollama response: {e}")
             return None

    @property
    def model_type(self) -> str:
        """Return the type of model."""
        return "ollama"
    
    # Implement model_name as a property
    @property
    def model_name(self) -> str:
        """Return the specific model name being used."""
        return self._model_name
    
    # Implement max_tokens as a property
    @property
    def max_tokens(self) -> int:
        """Return the default maximum number of tokens the model should generate."""
        return self._max_tokens
        
    def is_available(self) -> bool:
        """Check if the Ollama API server is running and reachable."""
        try:
            # Use a lightweight endpoint like GET /api/tags or just GET /
            response = requests.get(self.base_url.replace('/api', '/'), timeout=5) # Check base endpoint
            response.raise_for_status()
            return True
        except requests.exceptions.Timeout:
            logger.error(f"❌ Connection timed out when trying to reach Ollama at {self.base_url.replace('/api', '')}.")
            logger.error("💡 Is the Ollama server running? Try 'ollama serve'.")
            return False
        except requests.exceptions.ConnectionError:
            logger.error(f"❌ Connection error when trying to reach Ollama at {self.base_url.replace('/api', '')}.")
            logger.error("💡 Is the Ollama server running? Try 'ollama serve'.")
            return False
        except requests.exceptions.RequestException as e:
            logger.error(f"❌ Error checking Ollama availability: {e}")
            return False
    
    # Update generate_response to return ModelResponse
    def generate_response(
        self, 
        system_prompt: str, 
        user_content: str, 
        temperature: float = 0.7, 
        max_tokens_override: Optional[int] = None
    ) -> Optional[ModelResponse]:
        """Generate a response using the configured Ollama model.

        Args:
            system_prompt: The system prompt to guide the model.
            user_content: The user's input content.
            temperature: Controls randomness (0.0 to 1.0).
            max_tokens_override: Optional maximum number of tokens to generate, overriding default.

        Returns:
            A ModelResponse object, or None if an error occurred.
        """
        if not self.is_available():
            logger.error("Cannot generate response, Ollama server is not available.")
            return None

        endpoint = f"{self.base_url}/chat"
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content}
        ]
        
        # Use override if provided, otherwise use the default max_tokens property
        num_predict = max_tokens_override if max_tokens_override is not None else self.max_tokens

        payload: Dict[str, Any] = {
            "model": self.model_name,
            "messages": messages,
            "stream": False, # We want the full response, not a stream
            "options": {
                "temperature": temperature,
                "num_predict": num_predict # Ollama uses num_predict for max tokens
            }
        }

        try:
            logger.debug(f"Sending request to Ollama: {payload}")
            response = requests.post(endpoint, json=payload, timeout=120) # Increased timeout for generation
            response.raise_for_status() # Raise HTTPError for bad responses

            response_data = response.json()
            content = response_data.get("message", {}).get("content")
            usage_data = {
                'prompt_tokens': response_data.get('prompt_eval_count'),
                'completion_tokens': response_data.get('eval_count'),
                # Add other relevant usage stats if available
            }

            if content:
                 logger.debug(f"Received response from Ollama: {content[:100]}...") # Log snippet
                 return ModelResponse(
                     content=content.strip(), 
                     model_name=self.model_name,
                     raw_response=response_data, # Include the raw response dict
                     usage={k: v for k, v in usage_data.items() if v is not None} # Filter out None values
                 )
            else:
                 logger.error(f"❌ Ollama response missing content: {response_data}")
                 return None

        except requests.exceptions.Timeout:
             logger.error(f"❌ Request timed out while generating response from {self.model_name}.")
             return None
        except requests.exceptions.RequestException as e:
            logger.error(f"❌ Ollama API error during generation: {e}")
            if hasattr(e, 'response') and e.response is not None:
                 logger.error(f"Response status: {e.response.status_code}")
                 logger.error(f"Response body: {e.response.text}")
            return None
        except json.JSONDecodeError as e:
            logger.error(f"❌ Error decoding Ollama JSON response: {e}")
            return None
        except Exception as e:
            logger.error(f"❌ Unexpected error generating response: {e}", exc_info=True)
            return None
    
    def __str__(self) -> str:
        return f"OllamaModel(model={self.model_name}, url={self.base_url.replace('/api', '')})"

    def get_model_parameters(self, model_name: Optional[str] = None) -> Optional[str]:
        """Get parameter details for a specific model using the Ollama API.

        Args:
            model_name: Name of the model to check (e.g., 'llama3:8b'). Defaults to self.model_name.

        Returns:
            String describing parameter size (e.g., "8B", "70B", "7B") or None if unavailable.
        """
        target_model = model_name if model_name else self.model_name
        endpoint = f"{self.base_url}/show"
        payload = {"name": target_model}

        try:
            response = requests.post(endpoint, json=payload, timeout=10)
            response.raise_for_status()
            
            details = response.json()
            param_str = details.get("parameters") # e.g., "8.0 billion", "70 billion"
            
            if param_str:
                 # Try to extract a concise representation (e.g., "8B", "70B")
                 parts = param_str.lower().split()
                 if len(parts) >= 2 and parts[1].startswith("billion"):
                     try:
                         num = float(parts[0])
                         # Format as integer if whole number, else keep decimal
                         return f"{int(num) if num.is_integer() else num}B"
                     except ValueError:
                         return param_str # Return original string if parsing fails
                 elif len(parts) >= 2 and parts[1].startswith("million"):
                      try:
                         num = float(parts[0])
                         return f"{int(num) if num.is_integer() else num}M"
                      except ValueError:
                         return param_str
                 else:
                    return param_str # Return raw string if format unknown
            else:
                logger.warning(f"Parameter details not found for model '{target_model}' in Ollama API response.")
                return None

        except requests.exceptions.RequestException as e:
            logger.error(f"❌ Failed to get model details for '{target_model}' from Ollama: {e}")
            # Check if the error is because the model doesn't exist
            if hasattr(e, 'response') and e.response is not None and e.response.status_code == 404:
                 logger.error(f"   Model '{target_model}' may not be pulled locally.")
            return None
        except json.JSONDecodeError as e:
             logger.error(f"❌ Error decoding Ollama model details response: {e}")
             return None
        except Exception as e:
             logger.error(f"❌ Unexpected error fetching model parameters: {e}", exc_info=True)
             return None 