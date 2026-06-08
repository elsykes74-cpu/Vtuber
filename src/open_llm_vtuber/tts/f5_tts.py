import os

from loguru import logger

from .tts_interface import TTSInterface


class TTSEngine(TTSInterface):
    """
    F5-TTS engine implementation.

    F5-TTS uses flow matching for faithful speech synthesis with voice cloning
    via a reference audio sample.

    Requires: pip install f5-tts
    """

    def __init__(
        self,
        ref_audio: str = "",
        ref_text: str = "",
        model: str = "F5TTS_v1_Base",
        device: str = "",
        remove_silence: bool = False,
        speed: float = 1.0,
        cross_fade_duration: float = 0.15,
        nfe_step: int = 32,
        cfg_strength: float = 2.0,
        sway_sampling_coef: float = -1.0,
    ):
        """
        Initialize F5-TTS engine.

        Args:
            ref_audio: Path to reference audio file for voice cloning.
            ref_text: Transcription of the reference audio (leave empty for auto-transcription).
            model: Model name (F5TTS_v1_Base, F5TTS_Base, E2TTS_Base).
            device: Device to use (cuda, cpu, mps, xpu). Empty for auto-detect.
            remove_silence: Whether to remove silence from generated audio.
            speed: Speech speed multiplier (1.0 = normal).
            cross_fade_duration: Cross-fade duration in seconds between chunks.
            nfe_step: Number of function evaluations (higher = better quality, slower).
            cfg_strength: Classifier-free guidance strength.
            sway_sampling_coef: Sway sampling coefficient (-1 to disable).
        """
        try:
            from f5_tts.api import F5TTS
        except ImportError:
            raise ImportError(
                "F5-TTS is not installed. Install it with: pip install f5-tts"
            )

        self.ref_audio = ref_audio
        self.ref_text = ref_text
        self.remove_silence = remove_silence
        self.speed = speed
        self.cross_fade_duration = cross_fade_duration
        self.nfe_step = nfe_step
        self.cfg_strength = cfg_strength
        self.sway_sampling_coef = sway_sampling_coef

        device_arg = device if device else None

        logger.info(f"Initializing F5-TTS with model: {model}")
        self.tts = F5TTS(model=model, device=device_arg)
        logger.info("F5-TTS initialized successfully.")

    def generate_audio(self, text: str, file_name_no_ext=None) -> str:
        """
        Generate speech audio file using F5-TTS.

        Args:
            text: Text to synthesize.
            file_name_no_ext: Output filename without extension (optional).

        Returns:
            Path to generated audio file.
        """
        try:
            output_path = self.generate_cache_file_name(file_name_no_ext, "wav")

            if not self.ref_audio:
                raise ValueError(
                    "F5-TTS requires a reference audio file. "
                    "Please set 'ref_audio' in the f5_tts configuration."
                )

            wav, sr, spec = self.tts.infer(
                ref_file=self.ref_audio,
                ref_text=self.ref_text,
                gen_text=text,
                target_rms=0.1,
                cross_fade_duration=self.cross_fade_duration,
                nfe_step=self.nfe_step,
                cfg_strength=self.cfg_strength,
                sway_sampling_coef=self.sway_sampling_coef,
                speed=self.speed,
                remove_silence=self.remove_silence,
                file_wave=output_path,
                seed=None,
            )

            if not os.path.exists(output_path):
                raise FileNotFoundError(
                    f"Failed to generate audio file at {output_path}"
                )

            return output_path

        except Exception as e:
            logger.error(f"F5-TTS failed to generate audio: {e}")
            raise RuntimeError(f"Failed to generate audio with F5-TTS: {e}")
