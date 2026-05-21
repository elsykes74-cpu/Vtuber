# src/open_llm_vtuber/tts/openai_tts.py
import os
from pathlib import Path
from typing import Any

from loguru import logger
from openai import OpenAI

from .tts_interface import TTSInterface


class TTSEngine(TTSInterface):
    """Use an OpenAI-compatible speech endpoint to generate audio files."""

    def __init__(
        self,
        model: str = "kokoro",
        voice: str = "af_sky+af_bella",
        api_key: str = "not-needed",
        base_url: str = "http://localhost:8880/v1",
        file_extension: str = "mp3",
        **kwargs: Any,
    ) -> None:
        """Initialize the OpenAI-compatible TTS client."""
        self.model = model
        self.voice = voice
        self.base_url = base_url
        self.file_extension = (file_extension or "mp3").lower()
        if self.file_extension not in ["mp3", "wav"]:
            logger.warning(
                f"Unsupported file extension '{self.file_extension}' configured for OpenAI TTS. Defaulting to 'mp3'."
            )
            self.file_extension = "mp3"

        self.new_audio_dir = Path("cache")
        self.temp_audio_file = "temp_openai"
        self.new_audio_dir.mkdir(parents=True, exist_ok=True)

        try:
            self.client = OpenAI(api_key=api_key, base_url=base_url, **kwargs)
            logger.info(
                f"OpenAI-compatible TTS Engine initialized, targeting endpoint: {base_url}"
            )
        except Exception as exc:
            logger.critical(f"Failed to initialize OpenAI client: {exc}")
            self.client = None

    def _build_request_kwargs(self, text: str, speed: float) -> dict[str, Any]:
        """Build the request payload sent to the OpenAI-compatible endpoint."""
        return {
            "model": self.model,
            "voice": self.voice,
            "input": text,
            "response_format": self.file_extension,
            "speed": speed,
        }

    def generate_audio(self, text, file_name_no_ext=None, speed=1.0):
        """Generate speech audio using the configured endpoint."""
        if not self.client:
            logger.error("OpenAI client not initialized. Cannot generate audio.")
            return None

        file_name = self.generate_cache_file_name(file_name_no_ext, self.file_extension)
        speech_file_path = Path(file_name)

        try:
            logger.debug(
                f"Generating audio via {self.base_url} for text: '{text[:50]}...' with voice '{self.voice}' model '{self.model}'"
            )
            with self.client.audio.speech.with_streaming_response.create(
                **self._build_request_kwargs(text, speed)
            ) as response:
                response.stream_to_file(speech_file_path)

            logger.info(
                f"Successfully generated audio file via compatible endpoint: {speech_file_path}"
            )

        except Exception as exc:
            logger.critical(f"Error: OpenAI TTS unable to generate audio: {exc}")
            if speech_file_path.exists():
                try:
                    os.remove(speech_file_path)
                except OSError as remove_error:
                    logger.error(
                        f"Could not remove incomplete file {speech_file_path}: {remove_error}"
                    )
            return None

        return str(speech_file_path)
