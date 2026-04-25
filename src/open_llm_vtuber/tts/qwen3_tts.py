import torch
import soundfile as sf
from loguru import logger

from .tts_interface import TTSInterface
from .qwen_tts import Qwen3TTSModel


class TTSEngine(TTSInterface):
    """
    Qwen3-TTS engine (Alibaba, Apache-2.0).

    Three generation modes:
      - voice_clone:   clones a voice from a reference audio file (most stable)
      - voice_design:  generates a voice from a natural-language instruction
      - custom_voice:  uses a predefined speaker name

    Model checkpoints (HF repo id or local path):
      - voice_clone   → Qwen/Qwen3-TTS-12Hz-1.7B-Base
      - voice_design  → Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign
      - custom_voice  → Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice
                        speakers: serena, vivian, ryan, aiden, ono_anna, sohee

    Languages: english, spanish, french, german, japanese, korean, chinese,
               portuguese, russian, italian (auto also accepted)
    """

    def __init__(
        self,
        model_path: str,
        model_type: str = "voice_clone",
        language: str = "english",
        # voice_clone params
        ref_audio: str = "",
        ref_text: str = "",
        x_vector_only_mode: bool = True,
        # voice_design params
        instruct: str = "",
        # custom_voice params
        speaker: str = "",
        # runtime
        device: str = "cuda:0",
        temperature: float = 0.9,
        top_k: int = 50,
        top_p: float = 1.0,
        max_new_tokens: int = 2048,
    ):
        self.model_type = model_type
        self.language = language
        self.instruct = instruct
        self.speaker = speaker
        self.x_vector_only_mode = x_vector_only_mode
        self.gen_kwargs = dict(
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            max_new_tokens=max_new_tokens,
        )
        self.file_extension = "wav"

        logger.info(f"Loading Qwen3-TTS [{model_type}] from {model_path} ...")
        self.model = Qwen3TTSModel.from_pretrained(
            model_path,
            device_map=device,
            torch_dtype=torch.bfloat16,
        )
        logger.info("Qwen3-TTS model loaded.")

        # Pre-encode voice clone prompt once at startup (avoids re-encoding on every call)
        self._voice_clone_prompt = None
        if model_type == "voice_clone" and ref_audio:
            logger.info("Encoding voice clone reference audio ...")
            self._voice_clone_prompt = self.model.create_voice_clone_prompt(
                ref_audio=ref_audio,
                ref_text=ref_text if ref_text else None,
                x_vector_only_mode=x_vector_only_mode,
            )
            logger.info("Voice clone prompt ready.")

    def generate_audio(self, text: str, file_name_no_ext=None) -> str:
        file_name = self.generate_cache_file_name(file_name_no_ext, self.file_extension)
        try:
            if self.model_type == "voice_design":
                wavs, fs = self.model.generate_voice_design(
                    text=text,
                    instruct=self.instruct,
                    language=self.language,
                    **self.gen_kwargs,
                )
            elif self.model_type == "custom_voice":
                wavs, fs = self.model.generate_custom_voice(
                    text=text,
                    speaker=self.speaker,
                    language=self.language,
                    **self.gen_kwargs,
                )
            else:  # voice_clone
                wavs, fs = self.model.generate_voice_clone(
                    text=text,
                    language=self.language,
                    voice_clone_prompt=self._voice_clone_prompt,
                    **self.gen_kwargs,
                )
            sf.write(file_name, wavs[0], samplerate=fs, subtype="PCM_16")
            return file_name
        except Exception as e:
            logger.critical(f"Qwen3-TTS generate_audio failed: {e}")
            return None
