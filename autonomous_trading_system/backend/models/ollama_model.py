"""
ATS-Native Ollama Model Implementation
"""
import requests
import logging
from typing import List, Dict, Optional, Any
from .base_model import BaseModel, ModelResponse

logger = logging.getLogger(__name__)

class OllamaModel(BaseModel):
    """Implementation for local Ollama models."""
    
    AVAILABLE_MODELS: List[str] = [
        "llama3", "mistral", "phi3", "codellama", "gemma"
    ]
    DEFAULT_MODEL = "llama3"
    DEFAULT_MAX_TOKENS = 4096

    def __init__(self, model_name: Optional[str] = None, base_url: str = "http://localhost:11434", **kwargs):
        self.base_url = base_url.rstrip('/') + "/api"
        self._model_name = model_name or self.DEFAULT_MODEL
        self._max_tokens = self.DEFAULT_MAX_TOKENS
        super().__init__(api_key="LOCAL_OLLAMA", **kwargs)

    def initialize_client(self, **kwargs) -> None:
        """Checks connection to Ollama and verifies the model exists."""
        if not self.is_available():
            logger.error(f"Failed to connect to Ollama API at {self.base_url.replace('/api', '')}")
            return
            
        logger.info("✨ Successfully connected to Ollama API.")
        
        available_models = self._get_local_models()
        if available_models:
            local_model_names = [model["name"] for model in available_models]
            logger.info(f"📚 Available local Ollama models: {local_model_names}")
            if self._model_name not in local_model_names and not any(self._model_name in name for name in local_model_names):
                logger.warning(f"⚠️ Model '{self._model_name}' not found locally. You may need to run 'ollama pull {self._model_name}'.")
        else:
            logger.warning("⚠️ No local models found. Ensure Ollama is running and models are pulled.")

    def _get_local_models(self) -> Optional[List[Dict[str, Any]]]:
        """Retrieve the list of models available locally."""
        try:
            response = requests.get(f"{self.base_url}/tags", timeout=5)
            response.raise_for_status()
            return response.json().get("models", [])
        except requests.exceptions.RequestException as e:
            logger.error(f"❌ Error fetching local models from Ollama: {e}")
            return None

    @property
    def model_type(self) -> str:
        return "ollama"
    
    @property
    def model_name(self) -> str:
        return self._model_name
    
    @property
    def max_tokens(self) -> int:
        return self._max_tokens
        
    def is_available(self) -> bool:
        """Check if the Ollama API server is running."""
        try:
            response = requests.get(self.base_url.replace('/api', '/'), timeout=3)
            return response.status_code == 200
        except requests.exceptions.RequestException:
            return False
    
    def generate_response(
        self, 
        system_prompt: str, 
        user_content: any, 
        temperature: float = 0.7, 
        max_tokens_override: Optional[int] = None
    ) -> Optional[ModelResponse]:
        if not self.is_available():
            logger.error("Cannot generate response, Ollama server is not available.")
            return None
            
        if not isinstance(user_content, str):
            logger.error("Ollama model currently only supports string user_content.")
            return None

        endpoint = f"{self.base_url}/chat"
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content}
        ]
        
        num_predict = max_tokens_override if max_tokens_override is not None else self.max_tokens

        payload: Dict[str, Any] = {
            "model": self.model_name,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": num_predict
            }
        }

        try:
            response = requests.post(endpoint, json=payload, timeout=120)
            response.raise_for_status()

            response_data = response.json()
            content = response_data.get("message", {}).get("content")
            usage_data = {
                'prompt_tokens': response_data.get('prompt_eval_count'),
                'completion_tokens': response_data.get('eval_count'),
            }

            if content:
                 return ModelResponse(
                     content=content.strip(), 
                     model_name=self.model_name,
                     raw_response=response_data,
                     usage={k: v for k, v in usage_data.items() if v is not None}
                 )
            else:
                 logger.error(f"❌ Ollama response missing content: {response_data}")
                 return None

        except requests.exceptions.RequestException as e:
            logger.error(f"❌ Ollama API error during generation: {e}")
            return None
        except Exception as e:
            logger.error(f"❌ Unexpected error generating Ollama response: {e}", exc_info=True)
            return None
