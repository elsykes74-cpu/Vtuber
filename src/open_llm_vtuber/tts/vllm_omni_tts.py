import numpy as np
import httpx
from loguru import logger

from .tts_interface import TTSInterface

# Model-specific constants. Contributors: add new models here.
MODEL_PRESETS = {
    "qwen3-tts": {"sample_rate": 24000},
}


class TTSEngine(TTSInterface):
    """
    TTS engine backed by a vLLM-Omni server.

    Streams raw PCM from POST /v1/audio/speech with stream=true and
    response_format=pcm (16-bit signed mono). No local model loading
    — pure HTTP client.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8091/v1",
        model: str = "qwen3-tts",
        voice: str = "vivian",
        language: str = "Auto",
        task_type: str = "Base",
        ref_audio: str = None,
        ref_text: str = None,
        instructions: str = None,
        chunk_size_ms: int = 200,
        **kwargs,
    ):
        preset = MODEL_PRESETS.get(model)
        if preset is None:
            raise ValueError(
                f"Unknown vLLM-Omni model '{model}'. "
                f"Available: {', '.join(MODEL_PRESETS)}"
            )

        self.base_url = base_url.rstrip("/")
        self.voice = voice
        self.language = language
        self.task_type = task_type
        self.ref_audio = ref_audio
        self.ref_text = ref_text
        self.instructions = instructions
        self.sample_rate = preset["sample_rate"]
        # Bytes per yielded chunk: 16-bit PCM, aligned to 2 bytes
        self.chunk_bytes = (int(self.sample_rate * 2 * chunk_size_ms / 1000) // 2) * 2
        logger.info(
            f"vLLM-Omni TTS initialized: {self.base_url} model={model} voice={voice} task_type={task_type}"
        )

    def _build_payload(self, text: str, stream: bool) -> dict:
        payload = {
            "input": text,
            "voice": self.voice,
            "response_format": "pcm" if stream else "wav",
            "stream": stream,
            "language": self.language,
            "task_type": self.task_type,
        }
        if self.task_type == "Base":
            if self.ref_audio:
                payload["ref_audio"] = self.ref_audio
            if self.ref_text:
                payload["ref_text"] = self.ref_text
        if self.task_type == "VoiceDesign" and self.instructions:
            payload["instructions"] = self.instructions
        return payload

    def generate_audio(self, text: str, file_name_no_ext=None) -> str:
        """Synchronous fallback: fetch full WAV and save to cache file."""

        try:
            with httpx.Client(timeout=300.0) as client:
                r = client.post(
                    f"{self.base_url}/audio/speech",
                    json=self._build_payload(text, stream=False),
                )
                r.raise_for_status()
                wav_bytes = r.content

            path = self.generate_cache_file_name(file_name_no_ext, "wav")
            with open(path, "wb") as f:
                f.write(wav_bytes)
            return path
        except Exception as e:
            logger.error(f"vLLM-Omni TTS generate_audio error: {e}")
            return None

    async def async_generate_audio_streaming(self, text: str):
        """
        Async generator yielding (np.ndarray[float32], sample_rate) chunks.

        Streams raw 16-bit signed PCM at self.sample_rate Hz from vLLM-Omni,
        buffering into self.chunk_bytes-sized chunks before yielding.
        """
        buffer = b""
        try:
            async with httpx.AsyncClient(timeout=300.0) as client:
                async with client.stream(
                    "POST",
                    f"{self.base_url}/audio/speech",
                    json=self._build_payload(text, stream=True),
                ) as r:
                    r.raise_for_status()
                    async for raw in r.aiter_bytes(chunk_size=4096):
                        buffer += raw
                        while len(buffer) >= self.chunk_bytes:
                            chunk = buffer[: self.chunk_bytes]
                            buffer = buffer[self.chunk_bytes :]
                            arr = (
                                np.frombuffer(chunk, dtype=np.int16).astype(np.float32)
                                / 32768.0
                            )
                            yield arr, self.sample_rate

            # Flush remainder (align to 2 bytes / one int16 sample)
            remainder = (len(buffer) // 2) * 2
            if remainder >= 2:
                arr = (
                    np.frombuffer(buffer[:remainder], dtype=np.int16).astype(np.float32)
                    / 32768.0
                )
                yield arr, self.sample_rate

        except Exception as e:
            logger.error(f"vLLM-Omni TTS streaming error: {e}")
            raise
