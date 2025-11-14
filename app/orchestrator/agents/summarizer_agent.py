# app/orchestrator/agents/summarizer_agent.py

import random
from app.orchestrator.state import AgentState
from app.utils.llm_client import gemini_client
from app.utils.logger import get_logger

logger = get_logger(__name__)


class SummarizerAgent:
    """
    Summarizer Agent - Response Generator

    LangGraph Node: Generates final responses based on:
    1. Retrieved context (for policy-related queries)
    2. Soft refusal with warning (for off-topic queries)
    """

    def __init__(self):
        self.policy_system_prompt = """You are a helpful company policy assistant.

Your role is to answer questions about company policies based on the provided context.

Guidelines:
- Use ONLY the information from the provided context
- Be clear, concise, and professional
- If the context doesn't contain the answer, say "I don't have information about that in the policy documents."
- Cite the source document when possible
- Format your response in a user-friendly way
- Keep responses under 200 words unless more detail is needed
"""

        self.off_topic_responses = [
            "I appreciate your question, but I'm specifically designed to help with company policy-related queries.",
            "That's an interesting question! However, I specialize in answering questions about company policies.",
            "I'd love to help with that, but my expertise is limited to company policy information.",
            "Thanks for asking! I'm focused on providing information about company policies and procedures.",
        ]

    def __call__(self, state: AgentState) -> AgentState:
        """
        LangGraph node execution.

        Args:
            state: Current agent state

        Returns:
            Updated state with final reply
        """
        classification = state.get("classification", "policy-related")

        if classification == "off-topic":
            reply = self._generate_off_topic_response(state["message"])
        else:
            reply = self._generate_policy_response(state)

        state["reply"] = reply
        logger.info(f"SummarizerAgent: Generated {classification} response")
        return state

    def _generate_policy_response(self, state: AgentState) -> str:
        """
        Generate a response for policy-related queries using retrieved context.

        Args:
            state: Agent state with query and context

        Returns:
            Generated response
        """
        try:
            query = state["message"]
            context = state.get("context", "No context available.")
            history = state.get("history", [])

            # Build history context
            history_text = ""
            if history and len(history) > 0:
                recent_history = history[-4:]  # Last 2 exchanges
                formatted_messages = []
                for msg in recent_history:
                    # Handle both dict and LangChain Message objects
                    if isinstance(msg, dict):
                        role = msg.get('role', 'user')
                        content = msg.get('content', '')
                    else:
                        # LangChain Message object
                        role = getattr(msg, 'type', 'user')
                        content = getattr(msg, 'content', '')
                    formatted_messages.append(f"{role}: {content}")

                history_text = "\n".join(formatted_messages)
                history_text = f"\n\nRecent conversation:\n{history_text}\n"

            prompt = f"""{self.policy_system_prompt}

Context from company policy documents:
{context}
{history_text}
User question: {query}

Your response:"""

            response = gemini_client.generate(prompt)
            return response.strip()

        except Exception as e:
            logger.error(f"SummarizerAgent policy response error: {e}")
            return "I apologize, but I'm having trouble generating a response right now. Please try again."

    def _generate_off_topic_response(self, query: str) -> str:
        """
        Generate a soft refusal response for off-topic queries.

        Args:
            query: User's off-topic question

        Returns:
            Soft refusal with warning
        """
        # Select a random polite refusal
        base_response = random.choice(self.off_topic_responses)

        # Add a helpful suggestion
        suggestion = "\n\nFeel free to ask me about:\n- Company policies and procedures\n- HR guidelines and rules\n- Employee benefits and perks\n- Work arrangements and schedules\n- Leave policies and time-off\n- And other policy-related topics!"

        # Add warning
        warning = "\n\n[!] This chatbot is designed for answering company policy-related questions only."

        return f"{base_response}{suggestion}{warning}"

    async def stream_response(self, state: AgentState):
        """
        Stream response generation (for streaming endpoints).

        Args:
            state: Agent state

        Yields:
            Response chunks
        """
        classification = state.get("classification", "policy-related")

        if classification == "off-topic":
            # Off-topic responses are short, yield all at once
            yield self._generate_off_topic_response(state["message"])
        else:
            # Stream policy-related responses
            try:
                query = state["message"]
                context = state.get("context", "No context available.")

                prompt = f"""{self.policy_system_prompt}

Context from company policy documents:
{context}

User question: {query}

Your response:"""

                async for chunk in gemini_client.stream_generate(prompt):
                    yield chunk

            except Exception as e:
                logger.error(f"SummarizerAgent streaming error: {e}")
                yield "I apologize, but I'm having trouble generating a response right now."


# Singleton instance
summarizer_agent = SummarizerAgent()
