# app/utils/llm_client.py

import asyncio
from google import genai
from google.genai import types
from app.config.settings import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)

class GeminiClient:
    """
    Singleton wrapper around Google Gemini API.
    Provides both standard and streaming generation.
    """

    def __init__(self):
        self.client = genai.Client(api_key=settings.gemini_api_key)
        self.model = settings.gemini_model_name

    def _build_content(self, prompt: str) -> list[types.Content]:
        return [
            types.Content(
                role="user",
                parts=[types.Part.from_text(text=prompt)],
            )
        ]

    def _build_config(self) -> types.GenerateContentConfig:
        return types.GenerateContentConfig(
            thinking_config=types.ThinkingConfig(
                thinking_budget=settings.gemini_thinking_budget,
            ),
            image_config=types.ImageConfig(
                image_size=settings.gemini_image_size,
            ),
        )

    def generate(self, prompt: str) -> str:
        """
        Generate a complete response (non-streaming).
        """
        try:
            contents = self._build_content(prompt)
            config = self._build_config()
            result = self.client.models.generate_content(
                model=self.model,
                contents=contents,
                config=config,
            )
            return result.text.strip() if hasattr(result, "text") else str(result)
        except Exception as e:
            logger.error(f"[GeminiClient] Generation failed: {e}")
            return "Sorry, I couldn't generate a response right now."

    async def stream_generate(self, prompt: str):
        """
        Async generator yielding Gemini text chunks for streaming responses.
        """
        loop = asyncio.get_event_loop()

        def blocking_stream():
            contents = self._build_content(prompt)
            config = self._build_config()
            for chunk in self.client.models.generate_content_stream(
                model=self.model,
                contents=contents,
                config=config,
            ):
                yield chunk.text

        for text_chunk in await loop.run_in_executor(None, lambda: list(blocking_stream())):
            if text_chunk:
                yield text_chunk


# Singleton instance
gemini_client = GeminiClient()
