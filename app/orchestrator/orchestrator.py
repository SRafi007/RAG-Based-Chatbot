# app/orchestrator/orchestrator.py

from langgraph.graph import StateGraph, END
from app.orchestrator.state import AgentState
from app.orchestrator.agents import (
    domain_guard_agent,
    retriever_agent,
    summarizer_agent,
)
from app.memory.short_term_memory import (
    create_session,
    append_message,
    get_recent_messages,
    get_session_meta,
)
from app.utils.logger import get_logger

logger = get_logger(__name__)


class RAGOrchestrator:
    """
    LangGraph-based Orchestrator for RAG Chatbot

    Workflow:
    START → DomainGuard → Router
                ├── off-topic → Summarizer → END
                └── policy-related → Retriever → Summarizer → END

    This architecture allows easy plug-and-play of new agents.
    """

    def __init__(self):
        self.graph = self._build_graph()
        logger.info("RAGOrchestrator initialized with LangGraph")

    def _build_graph(self) -> StateGraph:
        """
        Build the LangGraph workflow.

        Returns:
            Compiled StateGraph
        """
        # Create the graph
        workflow = StateGraph(AgentState)

        # Add nodes (agents)
        workflow.add_node("domain_guard", domain_guard_agent)
        workflow.add_node("retriever", retriever_agent)
        workflow.add_node("summarizer", summarizer_agent)

        # Set entry point
        workflow.set_entry_point("domain_guard")

        # Add conditional routing based on classification
        workflow.add_conditional_edges(
            "domain_guard",
            self._route_based_on_classification,
            {
                "off-topic": "summarizer",      # Skip retrieval for off-topic
                "policy-related": "retriever",  # Retrieve docs for policy questions
            }
        )

        # After retrieval, always go to summarizer
        workflow.add_edge("retriever", "summarizer")

        # After summarization, end the workflow
        workflow.add_edge("summarizer", END)

        # Compile the graph
        return workflow.compile()

    def _route_based_on_classification(self, state: AgentState) -> str:
        """
        Routing function to determine next node based on classification.

        Args:
            state: Current agent state

        Returns:
            Next node name
        """
        classification = state.get("classification", "policy-related")
        logger.info(f"Router: Routing to {classification} path")
        return classification

    async def process(
        self,
        user_id: str,
        message: str,
        session_id: str,
    ) -> dict:
        """
        Process a user message through the orchestrator.

        Args:
            user_id: User identifier
            message: User's message
            session_id: Session identifier

        Returns:
            Dictionary with reply and metadata
        """
        try:
            logger.info(f"Processing message for user {user_id}, session {session_id}")

            # Ensure session exists
            meta = await get_session_meta(session_id)
            if not meta:
                await create_session(session_id, user_id)
                logger.info(f"Created new session: {session_id}")

            # Save user message to STM
            await append_message(session_id, "user", message)

            # Get conversation history from STM
            history_raw = await get_recent_messages(session_id, limit=20)

            # Convert Redis format (text) to LangGraph format (content)
            history = [
                {
                    "role": msg.get("role", "user"),
                    "content": msg.get("text", "")
                }
                for msg in history_raw
            ]

            # Prepare initial state
            initial_state: AgentState = {
                "user_id": user_id,
                "session_id": session_id,
                "message": message,
                "history": history,
                "classification": None,
                "retrieved_docs": None,
                "context": None,
                "reply": None,
                "metadata": {}
            }

            # Execute the graph
            final_state = self.graph.invoke(initial_state)

            # Save assistant response to STM
            reply = final_state.get("reply", "I apologize, but I couldn't generate a response.")
            await append_message(session_id, "assistant", reply)

            # Return response
            retrieved_docs = final_state.get("retrieved_docs") or []
            return {
                "reply": reply,
                "classification": final_state.get("classification"),
                "retrieved_docs": len(retrieved_docs) if retrieved_docs else 0,
                "session_id": session_id,
                "success": True
            }

        except Exception as e:
            logger.error(f"Orchestrator error: {e}", exc_info=True)
            return {
                "reply": "I apologize, but I encountered an error processing your request. Please try again.",
                "classification": "error",
                "success": False
            }

    async def stream_process(
        self,
        user_id: str,
        message: str,
        session_id: str,
    ):
        """
        Process a user message with streaming response.

        Args:
            user_id: User identifier
            message: User's message
            session_id: Session identifier

        Yields:
            Response chunks
        """
        try:
            logger.info(f"Streaming process for user {user_id}, session {session_id}")

            # Ensure session exists
            meta = await get_session_meta(session_id)
            if not meta:
                await create_session(session_id, user_id)

            # Save user message
            await append_message(session_id, "user", message)

            # Get history
            history_raw = await get_recent_messages(session_id, limit=20)

            # Convert Redis format (text) to LangGraph format (content)
            history = [
                {
                    "role": msg.get("role", "user"),
                    "content": msg.get("text", "")
                }
                for msg in history_raw
            ]

            # Prepare initial state (for streaming, we run graph partially)
            initial_state: AgentState = {
                "user_id": user_id,
                "session_id": session_id,
                "message": message,
                "history": history,
                "classification": None,
                "retrieved_docs": None,
                "context": None,
                "reply": None,
                "metadata": {}
            }

            # Execute domain guard and retriever (non-streaming parts)
            state = domain_guard_agent(initial_state)

            # Route based on classification
            if state["classification"] == "policy-related":
                state = retriever_agent(state)

            # Stream the summarizer response
            reply_buffer = ""
            async for chunk in summarizer_agent.stream_response(state):
                reply_buffer += chunk
                yield chunk

            # Save complete response to STM
            await append_message(session_id, "assistant", reply_buffer)

        except Exception as e:
            logger.error(f"Orchestrator streaming error: {e}", exc_info=True)
            yield "I apologize, but I encountered an error. Please try again."


# Singleton instance
orchestrator = RAGOrchestrator()
