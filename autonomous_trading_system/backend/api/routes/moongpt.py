from fastapi import APIRouter, Request
from services.moongpt_service import moongpt_service

router = APIRouter()

@router.post('/chat')
async def chat_with_moongpt(request: Request):
    data = await request.json()
    user_prompt = data.get('prompt', '')
    response = await moongpt_service.get_response(user_prompt)
    return {'response': response}
