# app/utils/llm_client.py

from google import genai
from app.config.settings import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)


class GeminiClient:
    """
    Wrapper around Google Gemini API.
    Provides both standard and streaming generation.
    """

    def __init__(self):
        try:
            self.client = genai.Client(api_key=settings.gemini_api_key)
            self.model = settings.gemini_model_name
            logger.info(f"GeminiClient initialized with model: {self.model}")
        except Exception as e:
            logger.error(f"Failed to initialize GeminiClient: {e}")
            raise

    def generate(self, prompt: str) -> str:
        """
        Generate a complete response (non-streaming).

        Args:
            prompt: The user prompt/question

        Returns:
            Generated text response
        """
        try:
            response = self.client.models.generate_content(
                model=self.model,
                contents=prompt
            )

            if hasattr(response, 'text'):
                return response.text.strip()
            else:
                logger.warning("Response has no text attribute")
                return str(response)

        except Exception as e:
            logger.error(f"[GeminiClient] Generation failed: {e}")
            return f"Error: {str(e)}"

    async def stream_generate(self, prompt: str):
        """
        Async generator yielding Gemini text chunks for streaming responses.

        Args:
            prompt: The user prompt/question

        Yields:
            Text chunks as they are generated
        """
        try:
            response = self.client.models.generate_content_stream(
                model=self.model,
                contents=prompt
            )

            for chunk in response:
                if hasattr(chunk, 'text') and chunk.text:
                    yield chunk.text

        except Exception as e:
            logger.error(f"[GeminiClient] Streaming failed: {e}")
            yield f"Error: {str(e)}"


# Singleton instance
gemini_client = GeminiClient()
