import json
import threading
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

from open_llm_vtuber.config_manager.tts import TTSConfig
from open_llm_vtuber.tts.qwen_tts import TTSEngine as QwenTTSEngine
from open_llm_vtuber.tts.tts_factory import TTSFactory


class _SpeechHandler(BaseHTTPRequestHandler):
    requests: list[dict] = []

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(length)
        self.__class__.requests.append(
            {
                "path": self.path,
                "body": json.loads(body.decode("utf-8")),
            }
        )

        audio = b"RIFF" + b"\x00" * 40
        self.send_response(200)
        self.send_header("Content-Type", "audio/wav")
        self.send_header("Content-Length", str(len(audio)))
        self.end_headers()
        self.wfile.write(audio)

    def log_message(self, format: str, *args) -> None:  # noqa: A003
        return


class _LocalSpeechServer:
    def __enter__(self) -> "_LocalSpeechServer":
        _SpeechHandler.requests.clear()
        self.server = HTTPServer(("127.0.0.1", 0), _SpeechHandler)
        self.port = self.server.server_address[1]
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=1)


def _build_config(base_url: str) -> TTSConfig:
    return TTSConfig.model_validate(
        {
            "tts_model": "qwen_tts",
            "qwen_tts": {
                "model": "qwen3-tts-en-single",
                "voice": "default",
                "api_key": "not-needed",
                "base_url": base_url,
                "file_extension": "wav",
                "language": "English",
            },
        }
    )


class QwenTTSTest(unittest.TestCase):
    def test_factory_returns_qwen_tts_engine(self) -> None:
        config = _build_config("http://127.0.0.1:8000/v1")

        engine = TTSFactory.get_tts_engine(
            config.tts_model,
            **config.qwen_tts.model_dump(),
        )

        self.assertIsInstance(engine, QwenTTSEngine)
        self.assertEqual(engine.base_url, "http://127.0.0.1:8000/v1")
        self.assertEqual(engine.voice, "default")
        self.assertEqual(engine.language, "English")

    def test_qwen_tts_smoke_request(self) -> None:
        with _LocalSpeechServer() as local_server:
            config = _build_config(f"http://127.0.0.1:{local_server.port}/v1")
            engine = TTSFactory.get_tts_engine(
                config.tts_model,
                **config.qwen_tts.model_dump(),
            )

            audio_path = Path(
                engine.generate_audio(
                    "Hello from Open-LLM-VTuber",
                    file_name_no_ext="qwen_tts_smoke",
                )
            )

            try:
                self.assertTrue(audio_path.exists())
                self.assertEqual(audio_path.suffix, ".wav")
                self.assertTrue(_SpeechHandler.requests)

                request = _SpeechHandler.requests[-1]
                self.assertEqual(request["path"], "/v1/audio/speech")
                self.assertEqual(request["body"]["model"], "qwen3-tts-en-single")
                self.assertEqual(request["body"]["voice"], "default")
                self.assertEqual(request["body"]["input"], "Hello from Open-LLM-VTuber")
                self.assertEqual(request["body"]["language"], "English")
                self.assertEqual(request["body"]["response_format"], "wav")
            finally:
                engine.remove_file(str(audio_path), verbose=False)


if __name__ == "__main__":
    unittest.main()
