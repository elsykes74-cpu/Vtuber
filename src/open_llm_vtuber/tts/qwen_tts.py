from typing import Any

from .openai_tts import TTSEngine as OpenAITTSEngine


class TTSEngine(OpenAITTSEngine):
    """Use a local Qwen3-compatible speech endpoint through the OpenAI client."""

    def __init__(
        self,
        model: str = "qwen3-tts-en-single",
        voice: str = "default",
        api_key: str = "not-needed",
        base_url: str = "http://localhost:8000/v1",
        file_extension: str = "wav",
        language: str | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the Qwen TTS engine."""
        self.language = language
        super().__init__(
            model=model,
            voice=voice,
            api_key=api_key,
            base_url=base_url,
            file_extension=file_extension,
            **kwargs,
        )

    def _build_request_kwargs(self, text: str, speed: float) -> dict[str, Any]:
        """Build a Qwen-compatible request payload."""
        request_kwargs = super()._build_request_kwargs(text, speed)
        if self.language:
            request_kwargs["extra_body"] = {"language": self.language}
        return request_kwargs
