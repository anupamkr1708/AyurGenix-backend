from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from db.session import get_db
from db import crud
from auth.security import get_current_user
from chat.schemas import *
from chat.service import run_chat
from usage.deps import enforce_usage_limits

router = APIRouter(prefix="/chat", tags=["Chat"])


@router.post("/", response_model=ChatResponse)
def chat(
    payload: ChatRequest,
    db: Session = Depends(get_db),
    user=Depends(get_current_user),
    usage=Depends(enforce_usage_limits),   # 👈 ADD THIS
):
    return run_chat(db, user, payload.query, payload.conversation_id)



@router.get("/conversations", response_model=list[ConversationOut])
def list_conversations(db: Session = Depends(get_db), user=Depends(get_current_user)):
    return crud.get_user_conversations(db, user.id)


@router.get("/conversation/{conversation_id}", response_model=list[MessageOut])
def get_conversation(conversation_id: str, db: Session = Depends(get_db), user=Depends(get_current_user)):
    conv = crud.get_conversation(db, conversation_id, user.id)
    if not conv:
        raise HTTPException(404, "Conversation not found")

    return crud.get_conversation_messages(db, conversation_id)


@router.post("/", response_model=ChatResponse)
def chat(payload: ChatRequest, db: Session = Depends(get_db), user=Depends(get_current_user)):
    return run_chat(db, user, payload.query, payload.conversation_id)
