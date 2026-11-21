import os
import sys
import tempfile
import unittest
import wave
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend" / "tts"))

try:
    import tts_server
except ModuleNotFoundError as error:
    if error.name in {"fastapi", "pydantic"}:
        raise unittest.SkipTest(f"Dipendenza TTS non installata: {error.name}") from error
    raise


class TTSServerTests(unittest.TestCase):
    def test_italian_preparation_expands_time_percent_abbreviations_and_numbers(self) -> None:
        prepared = tts_server.prepare_italian_tts_text(
            "Sono le 03:14: il valore è 25%, n. 2, ecc."
        )

        self.assertEqual(
            prepared,
            "Sono le tre e quattordici: il valore è venticinque per cento, "
            "numero due, eccetera.",
        )

    def test_italian_preparation_uses_configurable_pronunciation_overrides(self) -> None:
        with patch.dict(
            os.environ,
            {"XTTS_ITALIAN_PRONUNCIATION_OVERRIDES": '{"Qwenword":"cuen uord"}'},
        ):
            prepared = tts_server.prepare_italian_tts_text("Qwenword.")

        self.assertEqual(prepared, "cuen uord.")

    def test_italian_text_splits_at_sentence_and_long_clause_boundaries(self) -> None:
        long_clause = " ".join(f"parola{index}" for index in range(35))
        chunks = tts_server.split_italian_tts_text(
            f"Prima frase. Seconda frase! {long_clause}, poi finisce?"
        )

        self.assertEqual(chunks[:2], ["Prima frase.", "Seconda frase!"])
        self.assertGreaterEqual(len(chunks), 4)
        self.assertTrue(all(chunk[-1] in ".,!?;:…" for chunk in chunks))

    def test_xtts_chunk_synthesis_passes_italian_and_inserts_audio_pauses(self) -> None:
        calls = []

        class FakeModel:
            def tts_to_file(self, *, text, file_path, **options):
                calls.append((text, options))
                with wave.open(file_path, "wb") as audio_file:
                    audio_file.setnchannels(1)
                    audio_file.setsampwidth(2)
                    audio_file.setframerate(24_000)
                    audio_file.writeframes(b"\x10\x00" * 2_400)

        with tempfile.TemporaryDirectory() as temporary:
            output_path = Path(temporary) / "combined.wav"
            tts_server._synthesize_xtts_chunks(
                FakeModel(),
                "Ciao. Come stai?",
                output_path,
                None,
                "Voce di prova",
                0.92,
                {
                    "temperature": 0.65,
                    "top_k": 50,
                    "top_p": 0.85,
                    "repetition_penalty": 2.0,
                },
            )
            with wave.open(str(output_path), "rb") as audio_file:
                duration = audio_file.getnframes() / audio_file.getframerate()

        self.assertEqual([text for text, _options in calls], ["Ciao.", "Come stai?"])
        self.assertTrue(all(options["language"] == "it" for _, options in calls))
        self.assertTrue(all(options["split_sentences"] is False for _, options in calls))
        self.assertTrue(all(options["speed"] == 0.92 for _, options in calls))
        self.assertAlmostEqual(duration, 0.44, places=2)

    def test_xtts_sampling_parameters_are_environment_configurable(self) -> None:
        with patch.dict(
            os.environ,
            {
                "XTTS_TEMPERATURE": "0.7",
                "XTTS_TOP_K": "40",
                "XTTS_TOP_P": "0.9",
                "XTTS_REPETITION_PENALTY": "2.5",
            },
        ):
            options = tts_server._configured_xtts_sampling_options()

        self.assertEqual(
            options,
            {
                "temperature": 0.7,
                "top_k": 40,
                "top_p": 0.9,
                "repetition_penalty": 2.5,
            },
        )

    def test_comma_pause_is_shorter_than_sentence_pause(self) -> None:
        self.assertLess(
            tts_server._pause_after_tts_chunk("una breve pausa,"),
            tts_server._pause_after_tts_chunk("La frase finisce."),
        )

    def test_reference_wav_is_checked_and_logs_duration_and_sample_rate(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            reference = Path(temporary) / "voice.wav"
            with wave.open(str(reference), "wb") as audio_file:
                audio_file.setnchannels(1)
                audio_file.setsampwidth(2)
                audio_file.setframerate(16_000)
                audio_file.writeframes(b"\x80\x00" * 16_000)
            with (
                patch.dict(os.environ, {"VOICE_REFERENCE_WAV": str(reference)}),
                patch.object(tts_server.LOGGER, "info") as logger_info,
            ):
                resolved = tts_server._configured_voice_reference()

        self.assertEqual(resolved, reference.resolve())
        # self.assertIn("16000 Hz", logger_info.call_args.args[0])
        self.assertIn(16_000, logger_info.call_args.args)


if __name__ == "__main__":
    unittest.main()
