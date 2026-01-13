from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from auth.security import get_current_user
from db.session import get_db
from chat.service import rag_system

router = APIRouter(prefix="/chat", tags=["Chat"])

@router.post("/stream")
def chat_stream(payload: dict, user=Depends(get_current_user)):
    
    generator = rag_system.generate_stream(
        payload["query"]
    )

    return StreamingResponse(generator, media_type="text/plain")
