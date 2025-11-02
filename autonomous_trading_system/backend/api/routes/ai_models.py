"""
API Routes for the AI Model Factory
"""

import logging
from fastapi import APIRouter, Depends, HTTPException, Body
from typing import Optional, Any
from models.model_factory import get_model_factory, ModelFactory

logger = logging.getLogger(__name__)
router = APIRouter()

def get_model_factory_dep() -> ModelFactory:
    """Dependency to get the singleton model factory instance."""
    return get_model_factory()

@router.get("/models", summary="List Available AI Models")
async def list_available_models(factory: ModelFactory = Depends(get_model_factory_dep)):
    """
    Retrieves a list of all initialized and available AI models from the factory.
    This is useful for checking which models were successfully loaded based on
    your configuration and available SDKs.
    """
    try:
        available_models = factory.available_models
        return {
            "status": "success",
            "available_models": available_models
        }
    except Exception as e:
        logger.error(f"Error retrieving available models: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to retrieve available models.")

@router.post("/models/{model_type}/generate", summary="Generate Response from an AI Model")
async def generate_response(
    model_type: str,
    system_prompt: str = Body(..., embed=True, description="System-level instructions for the model."),
    user_content: Any = Body(..., embed=True, description="User's prompt. Can be a string or a list for multimodal input."),
    temperature: float = Body(0.7, embed=True, ge=0.0, le=2.0, description="Creativity of the response."),
    max_tokens_override: Optional[int] = Body(None, embed=True, description="Optionally override the model's default max token limit."),
    factory: ModelFactory = Depends(get_model_factory_dep)
):
    """
    Generates a response from the specified AI model.

    - **model_type**: The type of model to use (e.g., 'openai', 'claude', 'groq', 'ollama').
    """
    try:
        model = factory.get_model(model_type)
        if not model:
            raise HTTPException(status_code=404, detail=f"Model type '{model_type}' not available or not configured.")

        response = model.generate_response(
            system_prompt=system_prompt,
            user_content=user_content,
            temperature=temperature,
            max_tokens_override=max_tokens_override
        )

        if not response:
            raise HTTPException(status_code=500, detail=f"Failed to generate response from {model_type}.")

        return {
            "status": "success",
            "model_type": model_type,
            "model_name": response.model_name,
            "response": {
                "content": response.content,
                "usage": response.usage,
            }
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating response from {model_type}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred.")
