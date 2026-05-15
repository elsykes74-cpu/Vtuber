####
# change from xTTS.py
####

import re
import random
import os
import wave
from pathlib import Path
from loguru import logger
from .tts_interface import TTSInterface
import requests


class TTSEngine(TTSInterface):

    def __init__(
        self,
        api_url: str = "http://127.0.0.1:9880/tts",
        text_lang: str = "zh",
        ref_audio_path: str = "",
        prompt_lang: str = "zh",
        prompt_text: str = "",
        text_split_method: str = "cut0",
        batch_size: str = "1",
        media_type: str = "wav",
        streaming_mode: str = "false",
        parallel_infer: str = "true",
        clean_mode: str = "precise",
        custom_regex: str = "",
        speed_factor: str = "1.0",
        top_k: str = "20",
        emotional_tag: str = "",
        emotion_base_dir: str = "",
    ):
        self.api_url = api_url
        self.text_lang = text_lang
        self.ref_audio_path = ref_audio_path
        self.prompt_lang = prompt_lang
        self.prompt_text = prompt_text
        self.text_split_method = text_split_method
        self.batch_size = batch_size
        self.media_type = media_type
        self.streaming_mode = streaming_mode
        self.parallel_infer = parallel_infer
        self.clean_mode = clean_mode
        self.custom_regex = custom_regex
        self.speed_factor = speed_factor
        self.top_k = top_k
        self.emotional_tag = emotional_tag
        self.emotion_base_dir = emotion_base_dir
        self._current_emotion: str | None = None

        self.available_emotions = (
            [e.strip() for e in emotional_tag.split(",") if e.strip()]
            if emotional_tag else []
        )

        if self.available_emotions:
            if emotion_base_dir:
                self._emotion_root = Path(emotion_base_dir)
            elif ref_audio_path:
                self._emotion_root = Path(ref_audio_path).parent
            else:
                self._emotion_root = Path(".")
            # Precompile sentiment tag regular expressions to avoid repeated compilation each time
            self._emotion_tag_re = re.compile(
                r'<(' + '|'.join(re.escape(e) for e in self.available_emotions) + r')>'
            )
            logger.info(
                f"[TTS情感模式] 可用情感: {self.available_emotions} | "
                f"根目录: {self._emotion_root}"
            )

    # --------------------------------------------------------------------------------------------------------- #
    #  语种识别(目前只支持日语、中文)Language recognition(Currently only Japanese and Chinese are supported)    #
    # --------------------------------------------------------------------------------------------------------- #

    def _detect_lang(self, text: str) -> str:
        """根据字符集判断语种，返回 GPT-SoVITS 的 prompt_lang 值。"""
        for ch in text:
            cp = ord(ch)
            if 0x3040 <= cp <= 0x30FF:  # 平假名+片假名
                return "ja"
        for ch in text:
            cp = ord(ch)
            if 0x4E00 <= cp <= 0x9FFF:  # 中文
                return "zh"
        return "en"

    # ------------------------------------------------------------------ #
    #  情感音频解析Emotional Audio Analysis                              #
    # ------------------------------------------------------------------ #

    def _get_emotion_audio(self, emotion: str) -> tuple[str, str, str]:
        """
        从 emotions/<emotion>/ 目录随机取一个 wav。
        文件名即为 prompt_text，自动识别语种。
        返回 (audio_path, prompt_text, prompt_lang)。
        """
        emotion_dir = self._emotion_root / emotion
        if not emotion_dir.exists():
            logger.warning(f"[TTS情感] 目录不存在: {emotion_dir}，回退默认音频")
            return self.ref_audio_path, self.prompt_text, self.prompt_lang

        wav_files = list(emotion_dir.glob("*.wav"))
        if not wav_files:
            logger.warning(f"[TTS情感] 目录为空: {emotion_dir}，回退默认音频")
            return self.ref_audio_path, self.prompt_text, self.prompt_lang

        chosen_wav = random.choice(wav_files)
        prompt = chosen_wav.stem
        lang = self._detect_lang(prompt)
        logger.debug(f"[TTS情感] <{emotion}> → {chosen_wav.name} | lang={lang}")
        return str(chosen_wav.resolve()), prompt, lang  # ← 加 .resolve()
        return str(chosen_wav), prompt, lang

    # ------------------------------------------------------------------------------ #
    #  情感标签提取Emotion tag extraction                                            #
    #  只识别 <emotion> 标准格式（ignore_angle_brackets: false）                     #
    #  Only recognizes the standard <emotion> format (ignore_angle_brackets: false)  #
    # -------------------------------------------------------------------------------#

    def _extract_emotion(self, text: str) -> str | None:
        """从文本中提取第一个情感标签，返回情感名或 None。"""
        m = self._emotion_tag_re.search(text)
        return m.group(1) if m else None

    # ------------------------------------------------------------------ #
    #  文本清洗                                                          #
    #  - 清洗 <emotion> 情感标签（保留标签后的文字）                     #
    #  - 清洗 [xxx] 表情标签                                             #
    # Text Cleaning                                                      #
    # - Clean <emotion> emotion tags (keep the text after the tag)       #
    # - Clean [xxx] emoji tags                                           #
    # ------------------------------------------------------------------ #

    def _clean_text(self, text: str) -> str:
        # 1. 去掉 <emotion> 标签本身，保留后面的文字
        if self.available_emotions:
            text = self._emotion_tag_re.sub('', text)

        # 2. 清洗 [xxx] 表情标签
        if self.clean_mode == "none":
            cleaned = text
        elif self.clean_mode == "aggressive":
            cleaned = re.sub(r"\[.*?\]", "", text)
        elif self.clean_mode == "precise":
            cleaned = re.sub(r'\[[^\s\[\]]+\]', '', text)
        elif self.clean_mode == "custom":
            if self.custom_regex:
                try:
                    cleaned = re.compile(self.custom_regex).sub('', text)
                except re.error as e:
                    logger.error(f"[清洗] 无效正则: {self.custom_regex}，错误: {e}，回退 precise")
                    cleaned = re.sub(r'\[[^\s\[\]]+\]', '', text)
            else:
                logger.warning("[清洗] custom 模式未提供正则，回退 precise")
                cleaned = re.sub(r'\[[^\s\[\]]+\]', '', text)
        else:
            logger.warning(f"[清洗] 未知 clean_mode: {self.clean_mode}，回退 precise")
            cleaned = re.sub(r'\[[^\s\[\]]+\]', '', text)

        return re.sub(r'\s+', ' ', cleaned).strip()

    # ------------------------------------------------------------------ #
    #  单段 TTS API 调用Single-segment TTS API call                      #
    # ------------------------------------------------------------------ #

    def _call_tts_api(
        self,
        text: str,
        ref_audio: str,
        prompt_text: str,
        file_name: str,
        prompt_lang: str = None,
    ) -> str | None:
        data = {
            "text": text,
            "text_lang": self.text_lang,
            "ref_audio_path": ref_audio,
            "prompt_lang": prompt_lang or self.prompt_lang,
            "prompt_text": prompt_text,
            "text_split_method": self.text_split_method,
            "batch_size": self.batch_size,
            "media_type": self.media_type,
            "streaming_mode": self.streaming_mode,
            "parallel_infer": self.parallel_infer,
            "speed_factor": self.speed_factor,
            "top_k": self.top_k,
        }
        logger.debug(
            f"[TTS请求] emotion={self._current_emotion} | "
            f"ref={Path(ref_audio).name} | text={text[:20]}"
        )
        try:
            response = requests.post(self.api_url, json=data, timeout=120)
        except requests.exceptions.Timeout:
            logger.critical("[TTS] API 请求超时")
            return None

        if response.status_code == 200:
            with open(file_name, "wb") as f:
                f.write(response.content)
            return file_name
        else:
            logger.critical(
                f"[TTS] 生成失败，状态码: {response.status_code}，"
                f"响应: {response.text}"
            )
            return None

    # ------------------------------------------------------------------ #
    #  WAV 合并WAV merging                                               #
    # ------------------------------------------------------------------ #

    def _merge_wav_files(self, input_files: list[str], output_file: str):
        params = None
        frames_list = []
        for f in input_files:
            try:
                with wave.open(f, 'rb') as wf:
                    if params is None:
                        params = wf.getparams()
                    frames_list.append(wf.readframes(wf.getnframes()))
            except Exception as e:
                logger.warning(f"[合并] 跳过文件 {f}: {e}")

        if not frames_list or params is None:
            logger.error("[合并] 无可合并的音频片段")
            return

        with wave.open(output_file, 'wb') as out_wf:
            out_wf.setparams(params)
            for frames in frames_list:
                out_wf.writeframes(frames)

        logger.debug(f"[合并] {len(frames_list)} 段 → {output_file}")

    # ------------------------------------------------------------------ #
    #  主入口main entrance                                               #
    # ------------------------------------------------------------------ #

    def generate_audio(self, text, file_name_no_ext=None):
        file_name = self.generate_cache_file_name(file_name_no_ext, self.media_type)
        logger.debug(f"[TTS输入] {text}")

        if self.available_emotions:
            # 检测新情感标签，有则更新持续状态 Detect new sentiment tags and update the status if found.
            new_emotion = self._extract_emotion(text)
            if new_emotion:
                self._current_emotion = new_emotion
                logger.debug(f"[TTS情感] 切换 → {self._current_emotion}")

            # 用持续状态决定参考音频 Use the continuous state to determine the reference audio.
            if self._current_emotion:
                ref_audio, prompt, lang = self._get_emotion_audio(self._current_emotion)
            else:
                ref_audio, prompt, lang = self.ref_audio_path, self.prompt_text, self.prompt_lang

            cleaned = self._clean_text(text)
            if not cleaned:
                logger.warning("[TTS] 清洗后文本为空，跳过")
                return None
            return self._call_tts_api(cleaned, ref_audio, prompt, file_name, prompt_lang=lang)

        # 普通模式 Normal Default Mode
        cleaned_text = self._clean_text(text)
        return self._call_tts_api(
            cleaned_text, self.ref_audio_path, self.prompt_text, file_name
        )