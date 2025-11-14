# app/orchestrator/state.py

from typing import TypedDict, List, Dict, Optional, Annotated
from langgraph.graph import add_messages


class AgentState(TypedDict):
    """
    State schema for the RAG chatbot orchestrator.
    This represents the state that flows through the LangGraph workflow.
    """

    # User input
    user_id: str
    session_id: str
    message: str

    # Conversation history
    history: Annotated[List[Dict], add_messages]

    # Domain classification
    classification: Optional[str]  # "policy-related" or "off-topic"

    # Retrieved documents
    retrieved_docs: Optional[List[Dict[str, str]]]
    context: Optional[str]

    # Final response
    reply: Optional[str]

    # Metadata
    metadata: Optional[Dict]
