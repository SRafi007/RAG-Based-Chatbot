# app/orchestrator/agents/domain_guard.py

from app.orchestrator.state import AgentState
from app.utils.llm_client import gemini_client
from app.utils.logger import get_logger

logger = get_logger(__name__)


class DomainGuardAgent:
    """
    Domain Guard Agent - Policy Scope Checker

    LangGraph Node: Classifies user queries as policy-related or off-topic.
    This is the first node in the workflow.
    """

    def __init__(self):
        self.system_prompt = """You are a Domain Guard for a company policy chatbot.

Your job is to classify user queries into TWO categories:
1. "policy-related" - Questions about company policies, procedures, rules, guidelines, HR policies, benefits, etc.
2. "off-topic" - General questions, chit-chat, personal questions, or anything not related to company policies.

Respond with ONLY ONE WORD: either "policy-related" or "off-topic"

Examples:
- "What is the remote work policy?" → policy-related
- "How many vacation days do I get?" → policy-related
- "Tell me a joke" → off-topic
- "What is Python?" → off-topic
- "What's the weather today?" → off-topic
- "Can I work from home?" → policy-related
- "What are the sick leave policies?" → policy-related
"""

    def __call__(self, state: AgentState) -> AgentState:
        """
        LangGraph node execution.

        Args:
            state: Current agent state

        Returns:
            Updated state with classification
        """
        try:
            message = state["message"]
            logger.info(f"DomainGuard: Classifying message: {message[:50]}...")

            prompt = f"{self.system_prompt}\n\nUser message: {message}\n\nClassification:"
            response = gemini_client.generate(prompt)
            classification = response.strip().lower()

            # Validate response
            if "policy-related" in classification:
                state["classification"] = "policy-related"
                logger.info("DomainGuard: Classified as POLICY-RELATED")
            else:
                state["classification"] = "off-topic"
                logger.info("DomainGuard: Classified as OFF-TOPIC")

            return state

        except Exception as e:
            logger.error(f"DomainGuard error: {e}")
            # Default to policy-related to avoid blocking valid queries
            state["classification"] = "policy-related"
            return state


# Singleton instance
domain_guard_agent = DomainGuardAgent()
