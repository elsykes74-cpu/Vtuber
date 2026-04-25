import torch
import soundfile as sf
from loguru import logger

from .tts_interface import TTSInterface
from .qwen_tts import Qwen3TTSModel

# Maps (hf_family, size) → HuggingFace repo id.
# model_type in config maps to hf_family: voice_clone→Base, voice_design→VoiceDesign, custom_voice→CustomVoice
_HF_MODEL_MAP = {
    ("Base", "0.6B"): "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
    ("Base", "1.7B"): "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
    ("VoiceDesign", "1.7B"): "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
    ("CustomVoice", "0.6B"): "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
    ("CustomVoice", "1.7B"): "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
}

_MODEL_TYPE_TO_FAMILY = {
    "voice_clone": "Base",
    "voice_design": "VoiceDesign",
    "custom_voice": "CustomVoice",
}


def _resolve_model_path(model_type: str, model_size: str, model_path: str) -> str:
    """Return a fully-downloaded local path.

    If model_path is set, return it directly (assumed complete local directory).
    Otherwise resolve the HF repo id from model_type+model_size and use
    snapshot_download to ensure all files (including speech_tokenizer/) are present.
    """
    import os

    if model_path:
        return model_path

    family = _MODEL_TYPE_TO_FAMILY.get(model_type, "Base")
    key = (family, model_size)
    if key not in _HF_MODEL_MAP:
        raise ValueError(
            f"No HF model for model_type='{model_type}', model_size='{model_size}'. "
            f"Valid sizes: {sorted({k[1] for k in _HF_MODEL_MAP})}. "
            "Or set model_path to a local directory."
        )
    repo_id = _HF_MODEL_MAP[key]

    from huggingface_hub import snapshot_download

    logger.info(f"Ensuring full model snapshot for {repo_id} ...")
    local_dir = snapshot_download(repo_id)
    return local_dir


def _resolve_attention(selection: str) -> str:
    """Return the best available attention implementation.

    Priority for 'auto': sage_attn > flash_attn > sdpa > eager.
    For explicit selections, falls back to sdpa/eager if unavailable.
    Returns one of: 'sage_attn', 'flash_attn', 'sdpa', 'eager'.
    """
    available = ["sdpa", "eager"]
    try:
        import flash_attn  # noqa: F401
        available.insert(0, "flash_attn")
    except ImportError:
        pass
    try:
        from sageattention import sageattn  # noqa: F401
        available.insert(0, "sage_attn")
    except ImportError:
        pass

    if selection == "auto":
        chosen = available[0]
        logger.info(f"Qwen3-TTS attention auto-selected: {chosen}")
        return chosen

    if selection in available:
        return selection

    fallback = "sdpa" if "sdpa" in available else "eager"
    logger.warning(
        f"Qwen3-TTS attention '{selection}' not available, falling back to {fallback}"
    )
    return fallback


def _load_model_with_attention(
    model_path: str, attn: str, device: str
) -> Qwen3TTSModel:
    """Load Qwen3TTSModel with the resolved attention implementation."""
    dtype = torch.bfloat16

    if attn == "sage_attn":
        try:
            from sageattention import sageattn
            model = Qwen3TTSModel.from_pretrained(
                model_path, device_map=device, torch_dtype=dtype
            )
            patched = 0
            for name, module in model.model.named_modules():
                if hasattr(module, "forward") and (
                    "Attention" in type(module).__name__ or "attn" in name.lower()
                ):
                    try:
                        orig = module.forward
                        def _make(orig_fwd):
                            def _sage(*args, **kwargs):
                                if len(args) >= 3:
                                    q, k, v = args[0], args[1], args[2]
                                    mask = kwargs.get("attention_mask", None)
                                    return sageattn(q, k, v, is_causal=False, attn_mask=mask)
                                return orig_fwd(*args, **kwargs)
                            return _sage
                        module.forward = _make(orig)
                        patched += 1
                    except Exception:
                        pass
            logger.info(f"Qwen3-TTS sage_attn patched {patched} attention modules.")
            return model
        except Exception as e:
            logger.warning(f"Qwen3-TTS sage_attn failed ({e}), falling back to sdpa.")
            attn = "sdpa"

    attn_map = {"flash_attn": "flash_attention_2", "sdpa": "sdpa", "eager": "eager"}
    kwargs = {"device_map": device, "torch_dtype": dtype}
    if attn in attn_map:
        kwargs["attn_implementation"] = attn_map[attn]
    return Qwen3TTSModel.from_pretrained(model_path, **kwargs)


class TTSEngine(TTSInterface):
    """
    Qwen3-TTS engine (Alibaba, Apache-2.0).

    Three generation modes:
      - voice_clone:   clones a voice from a reference audio file (most stable)
      - voice_design:  generates a voice from a natural-language instruction
      - custom_voice:  uses a predefined speaker name

    Model is resolved from (model_type, model_size) automatically.
    Set model_path to override with a local directory.

    Languages: english, spanish, french, german, japanese, korean, chinese,
               portuguese, russian, italian (auto also accepted)
    """

    def __init__(
        self,
        model_type: str = "voice_clone",
        model_size: str = "1.7B",
        model_path: str = "",
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
        device: str = "cuda",
        attention: str = "auto",  # auto, sage_attn, flash_attn, sdpa, eager
        temperature: float = 0.9,
        top_k: int = 50,
        top_p: float = 1.0,
        max_new_tokens: int = 2048,
        repetition_penalty: float = 1.05,
        seed: int = -1,
    ):
        self.model_type = model_type
        self.language = language
        self.instruct = instruct
        self.speaker = speaker
        self.x_vector_only_mode = x_vector_only_mode
        self.seed = seed
        self.gen_kwargs = dict(
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            max_new_tokens=max_new_tokens,
            repetition_penalty=repetition_penalty,
        )
        self.file_extension = "wav"

        resolved_path = _resolve_model_path(model_type, model_size, model_path)
        attn = _resolve_attention(attention)
        logger.info(f"Loading Qwen3-TTS [{model_type}] from {resolved_path} (attn={attn}) ...")
        self.model = _load_model_with_attention(resolved_path, attn, device)
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
            if self.seed >= 0:
                torch.manual_seed(self.seed)
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
