# app/api/chat_api.py

from fastapi import APIRouter, HTTPException, Depends
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session
import uuid
import asyncio

from app.schemas.conversation import (
    ChatRequest,
    ChatResponse,
    ConversationMessageCreate,
)
from app.models.conversation import Conversation, ConversationSession
from app.orchestrator import orchestrator
from app.config.db import get_db
from app.utils.logger import get_logger

logger = get_logger(__name__)
router = APIRouter()


# ----------------------------
# Standard Chat Endpoint
# ----------------------------
@router.post("/", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest, db: Session = Depends(get_db)):
    """
    Handles standard chat requests for policy-related questions.

    Workflow:
    1. DomainGuard checks if query is policy-related
    2. If off-topic: Returns soft refusal with warning
    3. If policy-related: Retrieves docs and generates response

    Messages are saved to both Redis (STM) and PostgreSQL for persistence.
    """
    try:
        user_id = request.user_id
        session_id = request.session_id or f"{user_id}_{uuid.uuid4().hex[:8]}"

        logger.info(f"Chat request from user {user_id}, session {session_id}")

        # Ensure session exists in PostgreSQL
        session_record = (
            db.query(ConversationSession)
            .filter(ConversationSession.session_id == session_id)
            .first()
        )

        if not session_record:
            session_record = ConversationSession(session_id=session_id, user_id=user_id)
            db.add(session_record)
            db.commit()
            logger.info(f"Created new session in DB: {session_id}")

        # Process through orchestrator (handles Redis STM internally)
        result = await orchestrator.process(
            user_id=user_id,
            message=request.message,
            session_id=session_id,
        )

        # Save user message to PostgreSQL
        user_msg = Conversation(
            session_id=session_id,
            user_id=user_id,
            role="user",
            message=request.message,
            classification=result.get("classification"),
        )
        db.add(user_msg)

        # Save assistant response to PostgreSQL
        assistant_msg = Conversation(
            session_id=session_id,
            user_id=user_id,
            role="assistant",
            message=result["reply"],
            classification=result.get("classification"),
            extra_data={"retrieved_docs": result.get("retrieved_docs", 0)},
        )
        db.add(assistant_msg)

        # Update session metadata
        session_record.message_count += 2
        session_record.is_active = True

        db.commit()

        return ChatResponse(
            session_id=session_id,
            reply=result["reply"],
            classification=result.get("classification"),
            retrieved_docs=result.get("retrieved_docs"),
            success=result.get("success", True),
        )

    except Exception as e:
        logger.error(f"Error in chat_endpoint: {e}", exc_info=True)
        db.rollback()
        raise HTTPException(status_code=500, detail="Internal Server Error")


# ----------------------------
# Streaming Chat Endpoint
# ----------------------------
@router.post("/stream")
async def chat_stream_endpoint(request: ChatRequest, db: Session = Depends(get_db)):
    """
    Streams responses for policy-related questions.

    Workflow (same as standard endpoint, but streams the response):
    1. DomainGuard checks if query is policy-related
    2. If off-topic: Returns soft refusal (non-streamed)
    3. If policy-related: Streams response from SummarizerAgent

    SSE-compatible (text/event-stream).
    """
    try:
        user_id = request.user_id
        session_id = request.session_id or f"{user_id}_{uuid.uuid4().hex[:8]}"

        logger.info(f"Streaming chat request from user {user_id}, session {session_id}")

        # Ensure session exists in PostgreSQL
        session_record = (
            db.query(ConversationSession)
            .filter(ConversationSession.session_id == session_id)
            .first()
        )

        if not session_record:
            session_record = ConversationSession(session_id=session_id, user_id=user_id)
            db.add(session_record)
            db.commit()

        # Save user message to PostgreSQL
        user_msg = Conversation(
            session_id=session_id,
            user_id=user_id,
            role="user",
            message=request.message,
        )
        db.add(user_msg)
        db.commit()

        async def event_generator():
            reply_buffer = ""

            # Stream through orchestrator (handles Redis STM internally)
            async for chunk in orchestrator.stream_process(
                user_id=user_id,
                message=request.message,
                session_id=session_id,
            ):
                reply_buffer += chunk
                yield f"data: {chunk}\n\n"  # SSE format
                await asyncio.sleep(0.01)

            # Save assistant response to PostgreSQL
            assistant_msg = Conversation(
                session_id=session_record.session_id,
                user_id=user_id,
                role="assistant",
                message=reply_buffer,
            )
            db.add(assistant_msg)

            # Update session metadata
            session_record.message_count += 2
            db.commit()

            yield "data: [DONE]\n\n"

        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream",
        )

    except Exception as e:
        logger.error(f"Error in chat_stream_endpoint: {e}", exc_info=True)
        db.rollback()
        raise HTTPException(status_code=500, detail="Internal Server Error")


# ----------------------------
# Get Conversation History
# ----------------------------
@router.get("/history/{session_id}")
async def get_conversation_history(session_id: str, db: Session = Depends(get_db)):
    """
    Retrieve conversation history for a session from PostgreSQL.
    """
    try:
        messages = (
            db.query(Conversation)
            .filter(Conversation.session_id == session_id)
            .order_by(Conversation.created_at.asc())
            .all()
        )

        return {
            "session_id": session_id,
            "message_count": len(messages),
            "messages": [
                {
                    "role": msg.role,
                    "message": msg.message,
                    "classification": msg.classification,
                    "created_at": msg.created_at.isoformat(),
                }
                for msg in messages
            ],
        }

    except Exception as e:
        logger.error(f"Error retrieving history: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal Server Error")
