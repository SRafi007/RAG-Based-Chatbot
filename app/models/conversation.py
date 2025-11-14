# app/models/conversation.py

from sqlalchemy import Column, Integer, String, Text, DateTime, Boolean, JSON
from sqlalchemy.sql import func
from app.config.db import Base


class Conversation(Base):
    """
    Conversation model for storing chat history in PostgreSQL.

    This stores individual messages in conversations for:
    - Long-term persistence beyond Redis TTL
    - Analytics and reporting
    - Training data collection
    """

    __tablename__ = "conversations"

    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String(255), index=True, nullable=False)
    user_id = Column(String(255), index=True, nullable=False)

    # Message data
    role = Column(String(50), nullable=False)  # "user" or "assistant"
    message = Column(Text, nullable=False)

    # Classification and metadata
    classification = Column(String(50), nullable=True)  # "policy-related", "off-topic"
    extra_data = Column(JSON, nullable=True)  # Store retrieved_docs, context, etc.

    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    def __repr__(self):
        return f"<Conversation(id={self.id}, session={self.session_id}, role={self.role})>"


class ConversationSession(Base):
    """
    Session metadata model for tracking conversation sessions.

    Stores session-level information like start time, end time, message count, etc.
    """

    __tablename__ = "conversation_sessions"

    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String(255), unique=True, index=True, nullable=False)
    user_id = Column(String(255), index=True, nullable=False)

    # Session metadata
    message_count = Column(Integer, default=0)
    is_active = Column(Boolean, default=True)

    # Timestamps
    started_at = Column(DateTime(timezone=True), server_default=func.now())
    last_activity_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    ended_at = Column(DateTime(timezone=True), nullable=True)

    def __repr__(self):
        return f"<ConversationSession(id={self.id}, session={self.session_id}, user={self.user_id})>"
