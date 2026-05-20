# src/open_llm_vtuber/tts/camb_tts.py
import os
from pathlib import Path

from dotenv import load_dotenv
from loguru import logger
from camb.client import CambAI
from camb.types import StreamTtsOutputConfiguration

from .tts_interface import TTSInterface

# Load .env from parent camb-ai-work directory as fallback
_parent_env = Path(__file__).resolve().parents[4] / ".env"
if _parent_env.exists():
    load_dotenv(_parent_env, override=False)


class TTSEngine(TTSInterface):
    """
    Uses CAMB AI TTS API to generate speech.
    API Reference: https://docs.camb.ai
    """

    def __init__(
        self,
        api_key: str,
        voice_id: int = 147320,
        language: str = "en-us",
        speech_model: str = "mars-flash",
        output_format: str = "wav",
    ):
        self.api_key = api_key if api_key else os.getenv("CAMB_API_KEY", "")
        self.voice_id = voice_id
        self.language = language
        self.speech_model = speech_model
        self.output_format = output_format
        self.file_extension = output_format if output_format in ("wav", "mp3", "flac") else "wav"

        try:
            self.client = CambAI(api_key=api_key)
            logger.info("CAMB AI TTS Engine initialized successfully")
        except Exception as e:
            logger.critical(f"Failed to initialize CAMB AI client: {e}")
            self.client = None
            raise e

    def generate_audio(
        self, text: str, file_name_no_ext: str | None = None
    ) -> str | None:
        if not self.client:
            logger.error("CAMB AI client not initialized. Cannot generate audio.")
            return None

        file_name = self.generate_cache_file_name(file_name_no_ext, self.file_extension)
        speech_file_path = Path(file_name)

        try:
            logger.debug(
                f"Generating audio via CAMB AI for text: '{text[:50]}...' "
                f"with voice {self.voice_id}, model '{self.speech_model}'"
            )

            stream = self.client.text_to_speech.tts(
                text=text,
                language=self.language,
                voice_id=self.voice_id,
                speech_model=self.speech_model,
                output_configuration=StreamTtsOutputConfiguration(
                    format=self.output_format
                ),
            )

            with open(speech_file_path, "wb") as f:
                for chunk in stream:
                    f.write(chunk)

            logger.info(
                f"Successfully generated audio file via CAMB AI: {speech_file_path}"
            )

        except Exception as e:
            logger.critical(f"Error: CAMB AI TTS unable to generate audio: {e}")
            if speech_file_path.exists():
                try:
                    os.remove(speech_file_path)
                except OSError as rm_err:
                    logger.error(
                        f"Could not remove incomplete file {speech_file_path}: {rm_err}"
                    )
            raise e

        return str(speech_file_path)
