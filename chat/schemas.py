from pydantic import BaseModel
from typing import Optional, List
from datetime import datetime
from uuid import UUID


# ----------------------------
# Conversation Schemas
# ----------------------------

class ConversationCreate(BaseModel):
    title: Optional[str] = None


class ConversationOut(BaseModel):
    id: UUID
    title: Optional[str]
    created_at: datetime

    class Config:
        from_attributes = True
        json_encoders = {
            UUID: lambda v: str(v)
        }


# ----------------------------
# Message Schemas
# ----------------------------

class MessageCreate(BaseModel):
    conversation_id: UUID
    content: str


class MessageOut(BaseModel):
    id: UUID
    role: str
    content: str
    created_at: datetime

    class Config:
        from_attributes = True
        json_encoders = {
            UUID: lambda v: str(v)
        }


# ----------------------------
# Chat Schemas
# ----------------------------

class ChatRequest(BaseModel):
    conversation_id: Optional[UUID] = None
    query: str


class ChatResponse(BaseModel):
    conversation_id: UUID
    answer: str
    confidence: float
    sources: list
