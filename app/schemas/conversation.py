# app/schemas/conversation.py

from pydantic import BaseModel, Field
from typing import Optional, Dict, Any
from datetime import datetime


class ConversationMessageCreate(BaseModel):
    """Schema for creating a conversation message"""

    session_id: str
    user_id: str
    role: str = Field(..., pattern="^(user|assistant)$")
    message: str
    classification: Optional[str] = None
    extra_data: Optional[Dict[str, Any]] = None


class ConversationMessageResponse(BaseModel):
    """Schema for conversation message response"""

    id: int
    session_id: str
    user_id: str
    role: str
    message: str
    classification: Optional[str] = None
    extra_data: Optional[Dict[str, Any]] = None
    created_at: datetime

    class Config:
        from_attributes = True


class ConversationSessionCreate(BaseModel):
    """Schema for creating a conversation session"""

    session_id: str
    user_id: str


class ConversationSessionResponse(BaseModel):
    """Schema for conversation session response"""

    id: int
    session_id: str
    user_id: str
    message_count: int
    is_active: bool
    started_at: datetime
    last_activity_at: datetime
    ended_at: Optional[datetime] = None

    class Config:
        from_attributes = True


class ChatRequest(BaseModel):
    """Schema for chat API request"""

    message: str
    user_id: str = Field(
        default="anonymous", description="User identifier for the chat session"
    )
    session_id: Optional[str] = None


class ChatResponse(BaseModel):
    """Schema for chat API response"""

    session_id: str
    reply: str
    classification: Optional[str] = None
    retrieved_docs: Optional[int] = None
    success: bool = True
