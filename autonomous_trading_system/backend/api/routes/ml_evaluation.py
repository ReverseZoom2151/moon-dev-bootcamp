import logging
from fastapi import APIRouter, HTTPException, status
from backend.services.ml_evaluation_service import MLEvaluationService

router = APIRouter()
logger = logging.getLogger(__name__)

@router.get('/run_evaluation')
async def run_evaluation():
    """Run ML evaluation pipeline across generations using technical indicators."""
    try:
        service = MLEvaluationService()
        await service.run_evaluation()
        return {'message': 'ML evaluation completed successfully'}
    except Exception as e:
        logger.error(f"Error running ML evaluation: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e)) 