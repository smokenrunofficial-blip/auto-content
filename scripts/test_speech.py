import io
import json
import os
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request
from unittest.mock import patch

from video_pipeline import process_video, speech


class SpeechTests(unittest.TestCase):
    def test_narration_normalization_removes_markdown_symbols_and_ends_sentences(self) -> None:
        normalized = speech.normalize_narration_text(
            "**La porta** si aprì lentamente 🔥 senza alcun rumore"
        )

        self.assertEqual(normalized, "La porta si aprì lentamente senza alcun rumore.")

    def test_narration_normalization_adds_breaths_to_long_unpunctuated_text(self) -> None:
        text = " ".join(f"parola{index}" for index in range(50))

        normalized = speech.normalize_narration_text(text)

        self.assertGreaterEqual(normalized.count(","), 2)
        self.assertTrue(normalized.endswith("."))

#     def test_synthesis_posts_configured_voice_and_writes_audio(self) -> None:
#         audio = b"mock mp3 audio"
#         with tempfile.TemporaryDirectory() as temporary:
#             destination = Path(temporary) / "narration.mp3"
#             with (
#                 patch.dict(
#                     os.environ,
#                     {
#                         "ELEVENLABS_API_KEY": "test-api-key",
#                         "ELEVENLABS_VOICE_ID": "test-voice-id",
#                     },
#                 ),
#                 patch.object(speech, "urlopen", return_value=io.BytesIO(audio)) as urlopen,
#             ):
#                 speech.synthesize_narration("Ciao mondo", destination)
#                 # request, timeout = urlopen.call_args.args
#                 request = urlopen.call_args.args[0]
#                 timeout = urlopen.call_args.kwargs["timeout"]
#                 self.assertIsInstance(request, Request)
#                 self.assertEqual(
#                     request.full_url,
#                     "https://api.elevenlabs.io/v1/text-to-speech/test-voice-id"
#                     "?output_format=mp3_44100_128",
#                 )
#                 self.assertEqual(timeout, 120)
#                 self.assertEqual(request.get_method(), "POST")
#                 # self.assertEqual(request.get_header("X-api-key"), "test-api-key")
#                 headers = {name.lower(): value for name, value in request.headers.items()}
#                 self.assertEqual(headers["xi-api-key"], "test-api-key")
#                 self.assertEqual(headers["content-type"], "application/json")
#                 self.assertEqual(
#                     json.loads(request.data),
#                     {
#                         "text": "Ciao mondo",
#                         "model_id": "eleven_multilingual_v2",
#                         "voice_settings": {
#                             "stability": 0.45,
#                             "similarity_boost": 0.75,
#                         },
#                     },
#                 )
#                 self.assertEqual(destination.read_bytes(), audio)
#
#     def test_http_error_is_complete_in_pipeline_stderr_not_stdout(self) -> None:
#         api_detail = '{"detail":{"status":"invalid_voice","message":"' + ("x" * 2500) + '"}}'
#         http_error = HTTPError(
#             "https://api.elevenlabs.io/v1/text-to-speech/test-voice-id",
#             422,
#             "Unprocessable Entity",
#             hdrs=None,
#             fp=io.BytesIO(api_detail.encode("utf-8")),
#         )
#         stderr = io.StringIO()
#         stdout = io.StringIO()
#
#         def run_synthesis(*arguments: object) -> None:
#             self.assertEqual(len(arguments), 5)
#             speech.synthesize_narration("Ciao mondo", Path("unused.mp3"))
#
#         with (
#             patch.dict(
#                 os.environ,
#                 {
#                     "ELEVENLABS_API_KEY": "test-api-key",
#                     "ELEVENLABS_VOICE_ID": "test-voice-id",
#                 },
#             ),
#             patch.object(speech, "urlopen", side_effect=http_error),
#             patch.object(
#                 process_video,
#                 "run_pipeline",
#                 side_effect=run_synthesis,
#             ),
#             patch.object(
#                 sys,
#                 "argv",
#                 [
#                     "process_video.py",
#                     "--input",
#                     "input.mp4",
#                     "--video-id",
#                     "video123",
#                     "--brief",
#                     "Un breve spunto",
#                     "--rights-confirmed",
#                 ],
#             ),
#             redirect_stderr(stderr),
#             redirect_stdout(stdout),
#         ):
#             result = process_video.main()
#
#         self.assertEqual(result, 1)
#         self.assertIn("HTTP 422", stderr.getvalue())
#         self.assertIn(api_detail, stderr.getvalue())
#         self.assertEqual(stdout.getvalue(), "")
#
#
    def test_release_whisper_model_moves_cuda_weights_to_cpu(self) -> None:
        class FakeModel:
            device = "cuda:0"

            def __init__(self) -> None:
                self.moves: list[str] = []

            def to(self, device: str) -> None:
                self.moves.append(device)

        model = FakeModel()
        with patch.object(speech, "_clear_cuda_memory", return_value=True) as clear_cache:
            original_device = speech.release_whisper_model(model)

        self.assertEqual(original_device, "cuda:0")
        self.assertEqual(model.moves, ["cpu"])
        clear_cache.assert_called_once_with()

    def test_synthesis_posts_to_local_server_and_writes_wav(self) -> None:
        audio = b"RIFF\x04\x00\x00\x00WAVE"
        response = io.BytesIO(audio)
        response.headers = {"Content-Type": "audio/wav"}
        with tempfile.TemporaryDirectory() as temporary:
            destination = Path(temporary) / "nested" / "narration.wav"
            with (
                patch.dict(os.environ, {"LOCAL_TTS_URL": "http://127.0.0.1:8000/api/tts"}),
                patch.object(speech, "urlopen", return_value=response) as urlopen,
                patch.object(speech, "_report_synthesis_progress"),
            ):
                speech.synthesize_narration("Ciao mondo", destination, "it")
            saved_audio = destination.read_bytes()

        request = urlopen.call_args.args[0]
        self.assertIsInstance(request, Request)
        self.assertEqual(request.full_url, "http://127.0.0.1:8000/api/tts")
        self.assertEqual(request.get_method(), "POST")
        self.assertEqual(
            {name.lower(): value for name, value in request.headers.items()}["content-type"],
            "application/json",
        )
        self.assertEqual(
            json.loads(request.data),
            # {"text": "Ciao mondo", "language": "it"},
            {"text": "Ciao mondo.", "language": "it"},
        )
        self.assertEqual(saved_audio, audio)

    def test_synthesis_reports_local_http_error_details(self) -> None:
        detail = '{"detail":"Piper non disponibile"}'
        http_error = HTTPError(
            "http://127.0.0.1:8000/api/tts",
            500,
            "Internal Server Error",
            hdrs=None,
            fp=io.BytesIO(detail.encode("utf-8")),
        )
        with (
            patch.object(speech, "urlopen", side_effect=http_error),
            patch.object(speech, "_report_synthesis_progress"),
        ):
            with self.assertRaisesRegex(RuntimeError, "HTTP 500") as error:
                speech.synthesize_narration("Ciao mondo", Path("unused.wav"))

        self.assertIn("Piper non disponibile", str(error.exception))

    def test_synthesis_reports_unreachable_local_server(self) -> None:
        with (
            patch.object(
                speech,
                "urlopen",
                side_effect=URLError(ConnectionRefusedError("connection refused")),
            ),
            patch.object(speech, "_report_synthesis_progress"),
        ):
            with self.assertRaisesRegex(
                RuntimeError,
                "Server TTS locale non raggiungibile",
            ):
                speech.synthesize_narration("Ciao mondo", Path("unused.wav"))

    def test_synthesis_rejects_non_wav_response(self) -> None:
        response = io.BytesIO(b"non audio")
        response.headers = {"Content-Type": "application/json"}
        with (
            patch.object(speech, "urlopen", return_value=response),
            patch.object(speech, "_report_synthesis_progress"),
        ):
            with self.assertRaisesRegex(RuntimeError, "WAV valido"):
                speech.synthesize_narration("Ciao mondo", Path("unused.wav"))


if __name__ == "__main__":
    unittest.main()
