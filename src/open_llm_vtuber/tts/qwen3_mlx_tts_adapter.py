"""Qwen3-TTS adapter using MLX (Apple Silicon) 8-bit models.

Follows the qwen3-tts-apple-silicon project: path resolution (project models/
or HF cache e.g. ~/.cache/huggingface/hub), direct load_model(path, lazy=True),
and generate_audio(..., output_path=..., voice=..., instruct=..., speed=...)
producing output_path/audio_000.wav. Implements TTSInterface for Open-LLM-VTuber.
"""

import json
import multiprocessing
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Optional

from loguru import logger

from .tts_interface import TTSInterface

try:
    from mlx_audio.tts.generate import generate_audio
    from mlx_audio.tts import utils as mlx_tts_utils
    from mlx_audio.tts.utils import load_model

    if "qwen3_tts" not in mlx_tts_utils.MODEL_REMAPPING:
        mlx_tts_utils.MODEL_REMAPPING["qwen3_tts"] = "qwen3"
    MLX_AUDIO_AVAILABLE = True
except ImportError as e:
    MLX_AUDIO_AVAILABLE = False
    logger.warning(
        "mlx-audio not available: {}. Install with: uv sync --extra mlx (Apple Silicon only)",
        e,
    )

# Default: 0.6B CustomVoice 8-bit (fast on Mac).
DEFAULT_MODEL_ID = "mlx-community/Qwen3-TTS-12Hz-0.6B-CustomVoice-8bit"

# Top-level keys required by mlx_audio ModelConfig.from_dict.
_LLM_CONFIG_KEYS = (
    "hidden_size",
    "num_hidden_layers",
    "intermediate_size",
    "num_attention_heads",
    "rms_norm_eps",
    "vocab_size",
    "num_key_value_heads",
    "max_position_embeddings",
    "rope_theta",
    "head_dim",
    "tie_word_embeddings",
)


def _prepare_model_dir_for_mlx(
    model_path: str,
) -> tuple[str, Optional[object]]:
    """Prepare model dir for mlx_audio: flatten config and remap weights when needed.

    mlx-community 8-bit: load directly (reference mlx-audio git handles format natively).
    Original Qwen (models--Qwen--*): flatten config and remap talker.model.* ->
    model.*. Returns (path_to_use, tmpdir_handle_or_None). Caller must keep alive.
    """
    if "mlx-community" in model_path:
        return model_path, None

    path = Path(model_path)
    import mlx.core as mx

    config_path = path / "config.json"
    if not config_path.exists():
        return model_path, None

    with open(config_path, encoding="utf-8") as f:
        config = json.load(f)

    talker = config.get("talker_config") or {}
    needs_config_merge = config.get("hidden_size") is None
    if needs_config_merge:
        for key in _LLM_CONFIG_KEYS:
            if key in talker and config.get(key) is None:
                config[key] = talker[key]
        if config.get("vocab_size") is None:
            config["vocab_size"] = talker.get("vocab_size") or talker.get(
                "text_vocab_size"
            )
        if config.get("tie_word_embeddings") is None:
            config["tie_word_embeddings"] = False
        if config.get("head_dim") is None and "head_dim" in talker:
            config["head_dim"] = talker["head_dim"]
    config["model_type"] = "qwen3"

    tmp = tempfile.TemporaryDirectory(prefix="qwen3_mlx_tts_model_")
    tmp_path = Path(tmp.name)

    all_weights: dict[str, mx.array] = {}
    for sf in path.glob("*.safetensors"):
        all_weights.update(mx.load(str(sf)))
    model_prefix = "talker.model."
    subset = {
        k[len("talker.") :]: v
        for k, v in all_weights.items()
        if k.startswith(model_prefix)
    }
    subset.pop("model.text_embedding.weight", None)
    if "model.codec_embedding.weight" in subset:
        subset["model.embed_tokens.weight"] = subset.pop("model.codec_embedding.weight")
    if "model.lm_head.weight" in subset:
        subset["lm_head.weight"] = subset.pop("model.lm_head.weight")
    if "lm_head.weight" not in subset and "model.embed_tokens.weight" in subset:
        subset["lm_head.weight"] = subset["model.embed_tokens.weight"]

    if subset:
        mx.save_safetensors(str(tmp_path / "model.safetensors"), subset)

    for f in path.iterdir():
        if f.is_file() and f.suffix != ".safetensors":
            shutil.copy2(f, tmp_path / f.name)
        elif f.is_dir() and f.name not in ("snapshots", "__pycache__"):
            dest = tmp_path / f.name
            if not dest.exists():
                shutil.copytree(f, dest, dirs_exist_ok=True)

    with open(tmp_path / "config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    return str(tmp_path), tmp


def _hf_cache_dir() -> str:
    """Return Hugging Face hub cache directory (HF_HUB_CACHE or HF_HOME/hub)."""
    cache = os.environ.get("HF_HUB_CACHE") or os.path.join(
        os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface")), "hub"
    )
    return cache


def _resolve_model_path(
    model_id: str, models_dir: Optional[str] = None
) -> Optional[str]:
    """Resolve model_id to a local path (project models/ or HF cache).

    Mirrors qwen3-tts-apple-silicon get_smart_path: (1) project models_dir/
    <folder_name> with optional snapshots/<first_hash>; (2) HF cache with
    models--mlx-community--<folder_name>, models--Qwen--<folder_name>, and for
    -8bit also models--Qwen--<name_without_-8bit>; prefer snapshots when present.

    model_id can be: existing directory path (return as-is), HuggingFace-style
    org/name (e.g. mlx-community/Qwen3-TTS-12Hz-0.6B-CustomVoice-8bit), or
    folder name (e.g. Qwen3-TTS-12Hz-0.6B-CustomVoice-8bit).

    Returns:
        Resolved directory path, or None if not found.
    """
    if not model_id or not model_id.strip():
        return None

    model_id = model_id.strip()

    # Already a path that exists
    if os.path.sep in model_id and os.path.isdir(model_id):
        return model_id

    # Derive folder name from "org/name" or use as folder name
    if "/" in model_id:
        folder_name = model_id.split("/", 1)[-1].strip()
    else:
        folder_name = model_id

    # 1) Project models/ directory (like reference MODELS_DIR)
    if models_dir and os.path.isdir(models_dir):
        full_path = os.path.join(models_dir, folder_name)
        if os.path.exists(full_path):
            snapshots_dir = os.path.join(full_path, "snapshots")
            if os.path.exists(snapshots_dir):
                subfolders = [
                    f for f in os.listdir(snapshots_dir) if not f.startswith(".")
                ]
                if subfolders:
                    return os.path.join(snapshots_dir, subfolders[0])
            return full_path

    # 2) Hugging Face hub cache. run_server.py overrides HF_HOME to project
    # models/, so use models_dir first when provided (user may point to real HF
    # cache); otherwise fall back to _hf_cache_dir().
    hub_candidates: list[str] = []
    if models_dir and os.path.isdir(models_dir):
        hub_candidates.append(models_dir)
    default_hub = _hf_cache_dir()
    if default_hub not in hub_candidates:
        hub_candidates.append(default_hub)

    # For 8bit/mlx-community models: ONLY use mlx-community. Never fall back to
    # models--Qwen--* (original PyTorch format) - that produces noise.
    want_8bit = "8bit" in folder_name or "mlx-community" in model_id
    if want_8bit:
        folder_candidates = [f"models--mlx-community--{folder_name}"]
        if "0.6B" in folder_name:
            alt = folder_name.replace("0.6B", "1.7B")
            folder_candidates.append(f"models--mlx-community--{alt}")
    else:
        folder_candidates = [
            f"models--mlx-community--{folder_name}",
            f"models--Qwen--{folder_name}",
        ]
        if folder_name.endswith("-8bit"):
            folder_candidates.append(f"models--Qwen--{folder_name[:-5]}")

    for hub in hub_candidates:
        if not os.path.isdir(hub):
            continue
        for cache_folder in folder_candidates:
            full_path = os.path.join(hub, cache_folder)
            if not os.path.isdir(full_path):
                continue
            snapshots_dir = os.path.join(full_path, "snapshots")
            if os.path.exists(snapshots_dir):
                subfolders = [
                    f for f in os.listdir(snapshots_dir) if not f.startswith(".")
                ]
                if subfolders:
                    return os.path.join(snapshots_dir, subfolders[0])
            return full_path

    logger.debug(
        "Qwen3-TTS MLX: model not found; model_id=%s, hub_candidates=%s, folder_candidates=%s",
        model_id,
        hub_candidates,
        folder_candidates,
    )
    return None


def _run_tts_in_subprocess_worker(
    model_id: str,
    models_dir: Optional[str],
    speaker: str,
    instruct: str,
    speed: float,
    text: str,
    output_path: str,
) -> None:
    """Run TTS in a subprocess; writes WAV to output_path. Exits 0 on success, 1 on failure.

    Used when run_in_subprocess=True to isolate native crashes (e.g. MLX/Metal).
    """
    try:
        model_path = _resolve_model_path(model_id, models_dir)
        if not model_path:
            sys.exit(1)
        load_path, _tmpdir = _prepare_model_dir_for_mlx(model_path)
        model = load_model(load_path, lazy=True)
        with tempfile.TemporaryDirectory(prefix="qwen3_mlx_tts_") as temp_dir:
            generate_audio(
                model=model,
                text=text,
                voice=speaker,
                instruct=instruct or "Normal tone",
                speed=speed,
                file_prefix=os.path.join(temp_dir, "audio"),
                verbose=False,
            )
            source = os.path.join(temp_dir, "audio_000.wav")
            if os.path.isfile(source):
                shutil.copy(source, output_path)
                sys.exit(0)
        sys.exit(1)
    except Exception:
        sys.exit(1)


class TTSEngine(TTSInterface):
    """TTS engine using MLX 8-bit Qwen3-TTS CustomVoice (Apple Silicon)."""

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL_ID,
        models_dir: Optional[str] = None,
        speaker: str = "Vivian",
        language: str = "Auto",
        instruct: str = "",
        speed: float = 1.0,
        run_in_subprocess: bool = False,
    ) -> None:
        """Initialize the MLX Qwen3-TTS engine.

        Args:
            model_id: Model ID (e.g. mlx-community/Qwen3-TTS-12Hz-0.6B-CustomVoice-8bit)
                or local path. Resolved via project models/ or HF cache (~/.cache/huggingface/hub).
            models_dir: Optional project models directory (e.g. qwen3-tts-apple-silicon/models).
            speaker: CustomVoice speaker (Vivian, Serena, Uncle_Fu, Dylan, Eric, Ryan, Aiden, Ono_Anna, Sohee).
            language: Ignored by MLX CustomVoice; kept for config compatibility.
            instruct: Natural-language emotion/style (e.g. "Normal tone", "happy", "speak with excitement").
            speed: Speech speed (e.g. 1.0, 0.8, 1.3).
            run_in_subprocess: If True, run each TTS in a separate process to isolate
                native crashes (e.g. MLX/Metal) so the server does not exit.
        """
        if not MLX_AUDIO_AVAILABLE:
            raise ImportError(
                "mlx-audio is required for Qwen3 MLX TTS. Install with: uv sync --extra mlx"
            )

        self.model_id = model_id
        self.models_dir = models_dir
        self.speaker = speaker
        self.language = language
        self.instruct = instruct or "Normal tone"
        self.speed = speed
        self._run_in_subprocess = run_in_subprocess

        model_path = _resolve_model_path(model_id, models_dir)
        if not model_path:
            raise FileNotFoundError(
                f"MLX Qwen3-TTS model not found: {model_id}. "
                "Download it (e.g. run the qwen3-tts-apple-silicon project once) or set models_dir."
            )

        self._model = None
        self._model_tmpdir = None
        if not run_in_subprocess:
            load_path, self._model_tmpdir = _prepare_model_dir_for_mlx(model_path)
            self._model = load_model(load_path, lazy=True)
            logger.info(
                "Qwen3-TTS MLX loaded (model_path={}, speaker={})",
                model_path,
                speaker,
            )
        else:
            logger.info(
                "Qwen3-TTS MLX will run in subprocess (model_path={}, speaker={})",
                model_path,
                speaker,
            )

    def generate_audio(
        self, text: str, file_name_no_ext: Optional[str] = None
    ) -> Optional[str]:
        """Generate speech WAV file using MLX Qwen3-TTS CustomVoice.

        Args:
            text: The text to synthesize.
            file_name_no_ext: Optional base name for the output file (no extension).

        Returns:
            Path to the generated WAV file, or None on failure.
        """
        if not text or not text.strip():
            logger.warning("Qwen3-TTS MLX: empty text, skipping synthesis.")
            return None

        path = self.generate_cache_file_name(file_name_no_ext, file_extension="wav")

        if self._run_in_subprocess:
            return self._generate_audio_subprocess(path, text)

        try:
            logger.debug("Qwen3-TTS MLX: generating (len=%d) ...", len(text))
            with tempfile.TemporaryDirectory(prefix="qwen3_mlx_tts_") as temp_dir:
                generate_audio(
                    model=self._model,
                    text=text,
                    voice=self.speaker,
                    instruct=self.instruct,
                    speed=self.speed,
                    file_prefix=os.path.join(temp_dir, "audio"),
                    verbose=False,
                )
                source = os.path.join(temp_dir, "audio_000.wav")
                if os.path.isfile(source):
                    shutil.copy(source, path)
                    logger.debug("Qwen3-TTS MLX: wrote {}", path)
                    return path
                logger.error("Qwen3-TTS MLX: no audio_000.wav produced.")
                return None
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception as e:
            logger.critical(
                "Qwen3-TTS MLX unable to generate audio: {}",
                e,
                exc_info=True,
            )
            return None

    def _generate_audio_subprocess(self, path: str, text: str) -> Optional[str]:
        """Run TTS in a subprocess and write result to path. Returns path or None."""
        with tempfile.NamedTemporaryFile(
            suffix=".wav", prefix="qwen3_mlx_tts_", delete=False
        ) as f:
            out_path = f.name
        try:
            proc = multiprocessing.Process(
                target=_run_tts_in_subprocess_worker,
                args=(
                    self.model_id,
                    self.models_dir,
                    self.speaker,
                    self.instruct,
                    self.speed,
                    text,
                    out_path,
                ),
            )
            proc.start()
            proc.join(timeout=120)
            if os.path.isfile(out_path) and os.path.getsize(out_path) > 0:
                shutil.copy(out_path, path)
                logger.debug("Qwen3-TTS MLX (subprocess): wrote {}", path)
                if proc.exitcode != 0:
                    logger.debug(
                        "Qwen3-TTS MLX subprocess wrote audio but exited with code {}",
                        proc.exitcode,
                    )
                return path
            if proc.exitcode != 0:
                logger.warning(
                    "Qwen3-TTS MLX subprocess exited with code {}", proc.exitcode
                )
            if proc.is_alive():
                proc.terminate()
                proc.join(timeout=5)
            return None
        finally:
            if os.path.isfile(out_path):
                try:
                    os.unlink(out_path)
                except OSError:
                    pass
