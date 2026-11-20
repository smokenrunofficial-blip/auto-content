import io
import json
import re
import sys
import threading
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from typing import Any
from urllib.error import HTTPError
from urllib.request import Request
from unittest.mock import patch

from video_pipeline.captions import write_word_captions
from video_pipeline import captions
from video_pipeline import process_video
from video_pipeline import speech
from video_pipeline.render import render_vertical_short
from video_pipeline import visual_analysis
from video_pipeline.process_video import validate_input
from video_pipeline.visual_analysis import (
    Clip,
    SceneAnalysis,
    VisualMoment,
    _ollama_url,
    choose_retention_clips,
    generate_storytelling_script,
)


def _animated_moment(
    timestamp: float,
    score: float,
    focus_x: float = 0.5,
    suitable: bool = True,
    description: str = "",
) -> VisualMoment:
    return VisualMoment(
        timestamp,
        score,
        focus_x,
        suitable,
        description,
        has_text=False,
        is_animated=True,
    )


class VideoPipelineTests(unittest.TestCase):
    def test_clip_verification_rejects_text_in_an_interior_frame(self) -> None:
        import numpy as np

        class FakeCapture:
            def set(self, *_args: Any) -> None:
                return None

            def read(self) -> tuple[bool, Any]:
                return True, np.full((24, 24, 3), 80, dtype=np.uint8)

            def release(self) -> None:
                return None

        sample = _animated_moment(15.0, 0.9)
        motion = [VisualMoment(time, 0.02, 0.5) for time in range(13, 18)]
        unsafe_analysis = SceneAnalysis(
            "cartoon with visible words",
            (),
            tuple(
                VisualMoment(
                    timestamp,
                    0.9,
                    0.5,
                    True,
                    "cartoon with visible text",
                    has_text=True,
                    is_animated=True,
                )
                for timestamp in (12.5, 13.5, 14.5, 15.5, 16.5, 17.5)
            ),
        )
        with (
            patch.object(
                visual_analysis,
                "_open_video_capture",
                return_value=FakeCapture(),
            ),
            patch.object(visual_analysis, "_call_ollama", return_value={}),
            patch.object(
                visual_analysis,
                "_normalize_vision_response",
                return_value=unsafe_analysis,
            ),
            patch.object(visual_analysis, "_report_frame_progress"),
        ):
            verified = visual_analysis.verify_retention_samples(
                Path("source.mp4"),
                [sample],
                motion,
                video_duration=40.0,
            )

        self.assertFalse(verified[0].suitable)

    def test_pipeline_rejects_repeated_copies_of_one_clean_clip(self) -> None:
        scene = SceneAnalysis(
            "Anime senza testo.",
            (),
            (_animated_moment(15.0, 0.9, description="Anime senza testo."),),
        )
        with tempfile.TemporaryDirectory() as temporary:
            with (
                patch.object(
                    process_video,
                    "validate_input",
                    return_value=Path("source.mp4"),
                ),
                patch.object(process_video, "analyze_video", return_value=(40.0, [], 0.5)),
                patch.object(process_video, "analyze_keyframes", return_value=scene),
                patch.object(
                    process_video,
                    "verify_retention_samples",
                    return_value=list(scene.moments),
                ),
                patch.object(
                    process_video,
                    "choose_retention_clips",
                    return_value=[Clip(12.0, 18.0, 0.9), Clip(12.0, 17.0, 0.9)],
                ),
                patch.object(process_video, "OUTPUT_ROOT", Path(temporary)),
                patch.object(process_video, "load_whisper_model") as load_whisper_model,
            ):
                with self.assertRaises(process_video.CandidateRejectedError):
                    process_video.run_pipeline(
                        Path("source.mp4"),
                        "videoid1",
                        "brief valido",
                        "it",
                        True,
                    )

        load_whisper_model.assert_not_called()

    def test_clip_verification_accepts_animated_text_free_frames(self) -> None:
        import numpy as np

        class FakeCapture:
            def set(self, *_args: Any) -> None:
                return None

            def read(self) -> tuple[bool, Any]:
                return True, np.full((24, 24, 3), 80, dtype=np.uint8)

            def release(self) -> None:
                return None

        sample = _animated_moment(15.0, 0.9)
        motion = [VisualMoment(time, 0.02, 0.5) for time in range(13, 18)]
        safe_analysis = SceneAnalysis(
            "cartoon without visible text",
            (),
            tuple(
                _animated_moment(timestamp, 0.9, description="anime without text")
                for timestamp in (12.5, 13.5, 14.5, 15.5, 16.5, 17.5)
            ),
        )
        with (
            patch.object(
                visual_analysis,
                "_open_video_capture",
                return_value=FakeCapture(),
            ),
            patch.object(visual_analysis, "_call_ollama", return_value={}),
            patch.object(
                visual_analysis,
                "_normalize_vision_response",
                return_value=safe_analysis,
            ),
            patch.object(visual_analysis, "_report_frame_progress"),
        ):
            verified = visual_analysis.verify_retention_samples(
                Path("source.mp4"),
                [sample],
                motion,
                video_duration=40.0,
            )

        self.assertTrue(verified[0].suitable)

    def test_clip_verification_rejects_black_frames_before_rendering(self) -> None:
        import numpy as np

        class FakeCapture:
            def set(self, *_args: Any) -> None:
                return None

            def read(self) -> tuple[bool, Any]:
                return True, np.zeros((24, 24, 3), dtype=np.uint8)

            def release(self) -> None:
                return None

        sample = _animated_moment(15.0, 0.9)
        motion = [VisualMoment(time, 0.02, 0.5) for time in range(13, 18)]
        with (
            patch.object(
                visual_analysis,
                "_open_video_capture",
                return_value=FakeCapture(),
            ),
            patch.object(visual_analysis, "_call_ollama") as call_ollama,
            patch.object(visual_analysis, "_report_frame_progress"),
        ):
            verified = visual_analysis.verify_retention_samples(
                Path("source.mp4"),
                [sample],
                motion,
                video_duration=40.0,
            )

        self.assertFalse(verified[0].suitable)
        call_ollama.assert_not_called()

    def test_render_rejects_silent_narration_before_ffmpeg(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            narration_path = root / "narration.wav"
            narration_path.write_bytes(b"wav")
            with (
                patch("video_pipeline.render._measure_audio_peak", return_value=float("-inf")),
                patch("video_pipeline.render._report_render_progress"),
                patch("video_pipeline.render.subprocess.run") as run,
            ):
                with self.assertRaisesRegex(RuntimeError, "silenzioso"):
                    render_vertical_short(
                        root / "source.mp4",
                        narration_path,
                        root / "captions.ass",
                        root / "final.mp4",
                        [Clip(0.0, 2.0, 0.9)],
                        focus_x=0.5,
                        duration=2.0,
                    )

        run.assert_not_called()

    def test_retention_clips_are_chronological_and_match_target_duration(self) -> None:
        clips = choose_retention_clips(
            # [(2.0, 0.1), (12.0, 0.9), (22.0, 0.4)],
            [
                _animated_moment(2.0, 0.1),
                _animated_moment(12.0, 0.9),
                _animated_moment(22.0, 0.4),
            ],
            video_duration=30.0,
            target_duration=9.0,
        )

        self.assertEqual(clips, sorted(clips, key=lambda clip: clip.start))
        self.assertAlmostEqual(sum(clip.end - clip.start for clip in clips), 9.0)

    # def test_retention_clips_reject_target_longer_than_source(self) -> None:
    #     with self.assertRaises(ValueError):
    #         choose_retention_clips([], video_duration=4.0, target_duration=5.0)

    def test_retention_clips_repeat_short_source_to_match_target_duration(self) -> None:
        clips = choose_retention_clips(
            # [],
            # [_animated_moment(2.0, 0.9)],
            # video_duration=4.0,
            [_animated_moment(7.5, 0.9)],
            video_duration=10.0,
            target_duration=10.0,
        )

        self.assertAlmostEqual(sum(clip.end - clip.start for clip in clips), 10.0)
        # self.assertTrue(all(0 <= clip.start < clip.end <= 4.0 for clip in clips))
        self.assertTrue(all(5.0 <= clip.start < clip.end <= 10.0 for clip in clips))
        self.assertGreater(len(clips), 1)

    def test_render_uses_narration_duration_for_repeated_clips(self) -> None:
        clips = [Clip(0.0, 4.0, 0.0), Clip(0.0, 4.0, 0.0), Clip(0.0, 2.0, 0.0)]
        with tempfile.TemporaryDirectory() as temporary:
            with (
                patch(
                    "video_pipeline.render.MUSIC_DIRECTORY",
                    Path(temporary) / "empty-music",
                ),
                patch("video_pipeline.render._measure_audio_peak", return_value=-12.0),
                patch(
                    "video_pipeline.render._verify_rendered_audio",
                    return_value=("aac", -10.0),
                ),
                patch("video_pipeline.render._report_render_progress"),
                patch("video_pipeline.render.subprocess.run") as run,
            ):
                narration_path = Path(temporary) / "narration.wav"
                narration_path.write_bytes(b"test wav")
                render_vertical_short(
                    Path(temporary) / "source.mp4",
                    # Path(temporary) / "narration.mp3",
                    # Path(temporary) / "narration.wav",
                    narration_path,
                    Path(temporary) / "captions.ass",
                    Path(temporary) / "final.mp4",
                    clips,
                    focus_x=0.5,
                    duration=10.0,
                )

        command = run.call_args.args[0]
        self.assertEqual(command[command.index("-t") + 1], "10.000")
        filter_graph = command[command.index("-filter_complex") + 1]
        self.assertIn("concat=n=3", filter_graph)
        # self.assertIn("[video]subtitles=captions.ass[output]", filter_graph)
        self.assertIn("[video]subtitles=filename='captions.ass':", filter_graph)
        self.assertIn("fontsdir=", filter_graph)
        # self.assertIn("[1:a]anull[finalaudio]", filter_graph)
        self.assertIn("[1:a]loudnorm=I=-16:TP=-1.5:LRA=7[finalaudio]", filter_graph)
        self.assertIn("[finalaudio]", command)
        self.assertNotIn("force_style", filter_graph)

    def test_render_mixes_optional_music_under_narration(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            music_directory = root / "music"
            music_directory.mkdir()
            music_path = music_directory / "horror.mp3"
            music_path.touch()
            narration_path = root / "narration.wav"
            narration_path.write_bytes(b"test wav")
            with (
                patch("video_pipeline.render.MUSIC_DIRECTORY", music_directory),
                patch("video_pipeline.render._measure_audio_peak", return_value=-12.0),
                patch(
                    "video_pipeline.render._verify_rendered_audio",
                    return_value=("aac", -10.0),
                ),
                patch("video_pipeline.render._report_render_progress"),
                patch("video_pipeline.render.subprocess.run") as run,
            ):
                render_vertical_short(
                    root / "source.mp4",
                    # root / "narration.wav",
                    narration_path,
                    root / "captions.ass",
                    root / "final.mp4",
                    [Clip(0.0, 35.0, 0.9)],
                    focus_x=0.5,
                    duration=35.0,
                )

        command = run.call_args.args[0]
        filter_graph = command[command.index("-filter_complex") + 1]
        self.assertEqual(
            command[command.index("-stream_loop") + 1],
            "-1",
        )
        self.assertIn(str(music_path), command)
        self.assertIn("volume=0.1", filter_graph)
        self.assertIn("afade=t=out:st=34.000:d=1.000", filter_graph)
        self.assertIn("[1:a]loudnorm=I=-16:TP=-1.5:LRA=7[voice]", filter_graph)
        self.assertIn("[voice][music]amix=inputs=2", filter_graph)
        self.assertIn("[finalaudio]", command)

    def test_whisper_receives_generated_wav_for_word_timestamps(self) -> None:
        class FakeWhisperModel:
            def transcribe(self, audio_path, **options):
                return {"audio_path": audio_path, "options": options}

        audio_path = Path("narration.wav")
        result = speech.transcribe(
            FakeWhisperModel(),
            audio_path,
            language="it",
            word_timestamps=True,
        )

        self.assertEqual(result["audio_path"], str(audio_path))
        self.assertTrue(result["options"]["word_timestamps"])

    def test_retention_clips_keep_multimodal_focus_position(self) -> None:
        clips = choose_retention_clips(
            # [VisualMoment(timestamp=5.0, score=0.9, focus_x=0.82)],
            [_animated_moment(timestamp=5.0, score=0.9, focus_x=0.82)],
            video_duration=20.0,
            target_duration=4.0,
        )

        self.assertAlmostEqual(clips[0].focus_x, 0.82)

    def test_retention_clips_prefer_suitable_moments_over_rejected_frames(self) -> None:
        clips = choose_retention_clips(
            [
                _animated_moment(3.0, 0.99, suitable=False),
                _animated_moment(20.0, 0.1),
            ],
            video_duration=40.0,
            target_duration=6.0,
        )

        self.assertEqual(clips[0].start, 17.0)

    # def test_retention_clips_fall_back_to_best_interior_candidate(self) -> None:
    def test_retention_clips_reject_all_unsuitable_moments(self) -> None:
        clips = choose_retention_clips(
            [
                _animated_moment(3.0, 0.99, suitable=False),
                _animated_moment(20.0, 0.1, suitable=False),
            ],
            video_duration=40.0,
            target_duration=6.0,
        )

        # self.assertEqual(clips[0].start, 17.0)
        self.assertEqual(clips, [])

    def test_retention_clips_keep_best_interior_sample_when_all_scores_are_low(self) -> None:
        clips = choose_retention_clips(
            [
                _animated_moment(3.0, 0.1),
                _animated_moment(20.0, 0.2),
            ],
            video_duration=40.0,
            target_duration=6.0,
        )

        self.assertEqual(clips[0].start, 17.0)

    def test_retention_fill_avoids_rejected_visual_windows_when_possible(self) -> None:
        rejected_start = 21.0
        rejected_end = 27.0
        clips = choose_retention_clips(
            [
                # _animated_moment(3.0, 0.8),
                _animated_moment(10.0, 0.8),
                _animated_moment(24.0, 0.99, suitable=False),
            ],
            video_duration=48.0,
            target_duration=30.0,
        )

        self.assertAlmostEqual(sum(clip.end - clip.start for clip in clips), 30.0)
        self.assertTrue(
            all(
                not (clip.start < rejected_end and clip.end > rejected_start)
                for clip in clips
            )
        )

    def test_retention_filler_prefers_interior_windows(self) -> None:
        clips = choose_retention_clips(
            # [],
            [_animated_moment(15.0, 0.8)],
            video_duration=30.0,
            target_duration=6.0,
        )

        self.assertEqual(clips[0].start, 12.0)

    def test_retention_never_selects_text_or_live_action_moments(self) -> None:
        clips = choose_retention_clips(
            [
                VisualMoment(
                    14.0,
                    0.99,
                    0.5,
                    True,
                    "cartoon con sottotitoli",
                    has_text=False,
                    is_animated=True,
                ),
                VisualMoment(
                    22.0,
                    0.99,
                    0.5,
                    True,
                    "animazione pulita",
                    has_text=True,
                    is_animated=True,
                ),
                VisualMoment(
                    30.0,
                    0.99,
                    0.5,
                    True,
                    "una persona in una stanza",
                    has_text=False,
                    is_animated=False,
                ),
            ],
            video_duration=44.0,
            target_duration=6.0,
        )

        self.assertEqual(clips, [])

    def test_retention_prefers_clean_interior_frames_over_intro_and_outro(self) -> None:
        clips = choose_retention_clips(
            [
                _animated_moment(3.0, 0.99),
                _animated_moment(15.0, 0.1),
                _animated_moment(37.0, 0.99),
            ],
            video_duration=44.0,
            target_duration=6.0,
        )

        self.assertEqual(clips[0].start, 12.0)

    def test_retention_excludes_first_and_last_five_seconds(self) -> None:
        clips = choose_retention_clips(
            [
                _animated_moment(3.0, 0.99),
                _animated_moment(10.0, 0.8),
                _animated_moment(38.0, 0.99),
            ],
            video_duration=40.0,
            target_duration=6.0,
        )

        self.assertEqual(len(clips), 1)
        self.assertGreaterEqual(clips[0].start, 5.0)
        self.assertLessEqual(clips[0].end, 35.0)

    def test_content_bounds_exclude_outro_only_after_fifteen_seconds(self) -> None:
        self.assertEqual(visual_analysis._content_time_bounds(15.0), (5.0, 15.0))
        self.assertEqual(visual_analysis._content_time_bounds(16.0), (5.0, 11.0))

    def test_verification_frames_are_sampled_within_the_segment(self) -> None:
        frame_times = visual_analysis._verification_frame_times(10.0, 16.0)

        self.assertEqual(frame_times, (12.1, 13.9))

    def test_story_prompt_uses_scene_and_requests_a_hook(self) -> None:
        scene = SceneAnalysis(
            summary="Un artigiano modella una tazza al tornio.",
            actions=("Le mani levigano l'argilla bagnata.",),
            moments=(VisualMoment(3.0, 0.9, 0.6),),
        )
        with patch(
            "video_pipeline.visual_analysis._call_ollama",
            return_value={"narration": "Guarda come nasce questa tazza."},
        ) as ollama:
            narration = generate_storytelling_script(
                "Un lavoro soddisfacente",
                scene,
                "La trascrizione originale",
                target_duration=12.0,
                language="it",
            )

        prompt = ollama.call_args.args[0]
        self.assertEqual(narration, "Guarda come nasce questa tazza.")
        self.assertIn("gancio forte", prompt)
        self.assertIn(scene.summary, prompt)
        self.assertIn(scene.actions[0], prompt)

    def test_story_recovers_text_from_alternative_and_nested_keys(self) -> None:
        scene = SceneAnalysis(
            summary="Una tazza viene modellata al tornio.",
            actions=("Le mani modellano l'argilla.",),
            moments=(VisualMoment(3.0, 0.9, 0.6),),
        )
        with patch(
            "video_pipeline.visual_analysis._call_ollama",
            return_value={"result": {"script": "La tazza prende forma."}},
        ):
            narration = generate_storytelling_script(
                "La ceramica",
                scene,
                "",
                target_duration=12.0,
                language="it",
            )

        self.assertEqual(narration, "La tazza prende forma.")
        self.assertEqual(
            visual_analysis._extract_narration(
                {"result": {"first": "Una frase.", "second": "Un'altra frase."}}
            ),
            "Una frase. Un'altra frase.",
        )

    def test_ollama_uses_text_schema_without_images_and_vision_model_with_images(self) -> None:
        payloads = [
            {"message": {"content": '{"narration":"Testo narrativo."}'}},
            {"message": {"content": '{"scene_summary":"Una scena."}'}},
            {"message": {"content": '{"scene_summary":"Una scena estesa."}'}},
        ]
        with (
            patch.dict(
                "os.environ",
                {
                    "OLLAMA_TEXT_MODEL": "llama3.1:latest",
                    "OLLAMA_VISION_MODEL": "qwen2.5vl:3b",
                    "OLLAMA_CONTEXT_WINDOW": "20000",
                },
            ),
            patch.object(visual_analysis, "_ollama_url", return_value="http://localhost"),
            patch.object(
                visual_analysis,
                "_request_ollama",
                side_effect=payloads,
            ) as request_ollama,
            patch.object(visual_analysis, "_report_frame_progress"),
        ):
            visual_analysis._call_ollama("Scrivi una narrazione.", [])
            visual_analysis._call_ollama("Analizza la scena.", ["encoded-image"])
            visual_analysis._call_ollama(
                "Analizza più immagini.",
                [f"encoded-image-{index}" for index in range(10)],
            )

        text_request = json.loads(request_ollama.call_args_list[0].args[0].data)
        vision_request = json.loads(request_ollama.call_args_list[1].args[0].data)
        multi_image_request = json.loads(request_ollama.call_args_list[2].args[0].data)
        self.assertEqual(text_request["model"], "llama3.1:latest")
        self.assertEqual(text_request["format"], visual_analysis.NARRATION_JSON_SCHEMA)
        self.assertNotIn("images", text_request["messages"][0])
        self.assertEqual(text_request["options"]["num_ctx"], 20000)
        self.assertEqual(text_request["options"]["repeat_penalty"], 1.25)
        self.assertEqual(text_request["options"]["repeat_last_n"], 128)
        self.assertEqual(vision_request["model"], "qwen2.5vl:3b")
        # self.assertEqual(vision_request["format"], "json")
        self.assertEqual(
            vision_request["format"],
            visual_analysis.VISION_JSON_SCHEMA,
        )
        self.assertEqual(vision_request["options"]["repeat_penalty"], 1.25)
        self.assertEqual(vision_request["options"]["repeat_last_n"], 128)
        self.assertEqual(
            len(multi_image_request["messages"][0]["images"]),
            visual_analysis.OLLAMA_MAX_VISION_FRAMES,
        )
        self.assertEqual(visual_analysis.OLLAMA_CONTEXT_WINDOW, 16_384)

    def test_vision_response_uses_fallbacks_for_partial_fields(self) -> None:
        scene = visual_analysis._normalize_vision_response(
            {
                "scene_summary": "",
                "actions": ["", "Persona cammina"],
                "moments": [
                    {"time_seconds": 4.0},
                    {"score": 0.8, "suitable": "true"},
                ],
            },
            duration=20.0,
            timestamps=[0.0, 4.0, 8.0, 12.0, 16.0],
        )

        self.assertIn("non descritta", scene.summary)
        self.assertEqual(scene.actions, ("Persona cammina",))
        self.assertGreaterEqual(len(scene.moments), 2)
        self.assertEqual(scene.moments[0].timestamp, 4.0)
        self.assertEqual(scene.moments[0].score, 0.5)
        self.assertEqual(scene.moments[0].focus_x, 0.5)
        self.assertTrue(scene.moments[1].suitable)
        self.assertTrue(scene.moments[0].has_text)
        self.assertFalse(scene.moments[0].is_animated)

    def test_vision_prompt_requires_text_and_animation_flags(self) -> None:
        prompt = visual_analysis._build_vision_prompt(
            "anime horror",
            ["Immagine 1: t=12.00s."],
        )

        self.assertIn("has_text=true", prompt)
        self.assertIn("is_animated=true", prompt)
        # self.assertIn("lettere, parole, frasi", prompt)
        # self.assertIn("sottotitoli, titoli, crediti, watermark", prompt)
        # self.assertIn("loghi con testo", prompt)
        self.assertIn("area principale centrale", prompt)
        self.assertIn("crop verticale 9:16", prompt)
        self.assertIn("sottotitoli, fumetti con testo", prompt)
        self.assertIn("Ignora piccoli watermark o firme", prompt)
        self.assertIn("dettagli grafici decorativi dello sfondo", prompt)
        self.assertIn("non deve mai attenuare i controlli", prompt)
        # self.assertIn("sottotitoli", prompt)
        # self.assertIn("sottotitolo", prompt)
        self.assertIn("sottotitoli", prompt)
        # self.assertIn("restituisci un momento per ogni immagine", prompt)
        self.assertIn("un elemento per ogni immagine", prompt)
        self.assertIn("scene_summary, actions e moments", prompt)
        self.assertIn("non omettere né rinominare chiavi", prompt)
        self.assertFalse(
            visual_analysis._description_signals_visible_text(
                "Cartoon, no visible text or logo."
            )
        )
        self.assertTrue(
            visual_analysis._description_signals_visible_text(
                "No text except subtitles."
            )
        )
        self.assertFalse(
            visual_analysis._description_signals_visible_text(
                "Anime with a tiny watermark in the extreme upper-right corner."
            )
        )
        self.assertTrue(
            visual_analysis._description_signals_visible_text(
                "Anime with subtitles and a tiny watermark in the extreme corner."
            )
        )
        self.assertFalse(
            visual_analysis._description_signals_visible_text(
                "Decorative background markings in a hand-drawn anime scene."
            )
        )

    def test_vision_parser_accepts_only_explicit_text_free_animation(self) -> None:
        scene = visual_analysis._normalize_vision_response(
            {
                "scene_summary": "Animazione senza testo o loghi.",
                "actions": ["Figure disegnate attraversano la scena."],
                "moments": [
                    {
                        "time_seconds": 15,
                        "description": "Cartoon senza testo o loghi.",
                        "has_text": False,
                        "is_animated": True,
                        "suitable": True,
                    },
                    {
                        "time_seconds": 22,
                        "description": "Sottotitoli visibili.",
                        "has_text": False,
                        "is_animated": True,
                        "suitable": True,
                    },
                ],
            },
            duration=40.0,
            timestamps=[0, 8, 16, 24, 32, 39],
        )

        self.assertTrue(visual_analysis._moment_is_safe_for_retention(scene.moments[0]))
        self.assertTrue(scene.moments[1].has_text)
        self.assertFalse(visual_analysis._moment_is_safe_for_retention(scene.moments[1]))

    def test_text_in_global_summary_does_not_contaminate_clean_frames(self) -> None:
        scene = visual_analysis._normalize_vision_response(
            {
                "scene_summary": "An opening title appears before a clean animated story.",
                "moments": [
                    {
                        "time_seconds": 12,
                        "description": "Clean anime scene without visible text.",
                        "has_text": False,
                        "is_animated": True,
                        "suitable": True,
                    }
                ],
            },
            duration=30.0,
            timestamps=[12.0],
        )

        self.assertFalse(scene.moments[0].has_text)
        self.assertTrue(visual_analysis._moment_is_safe_for_retention(scene.moments[0]))

    def test_vision_response_rejects_only_empty_or_unusable_data(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "vuota o inutilizzabile"):
            visual_analysis._normalize_vision_response(
                {"scene_summary": "", "actions": [], "moments": []},
                duration=20.0,
                timestamps=[0.0, 4.0, 8.0],
            )

    def test_vision_normalizer_accepts_nested_description_aliases(self) -> None:
        scene = visual_analysis._normalize_vision_response(
            {
                "result": {
                    "caption": "A dark hand-drawn hallway.",
                    "frames": [
                        {
                            "timestamp": 4.0,
                            "description": "Cartoon hallway, no visible text.",
                            "has_text": False,
                            "is_animated": True,
                        },
                        "A shadow moves across the wall.",
                    ],
                }
            },
            duration=20.0,
            timestamps=[0.0, 4.0, 8.0],
        )

        self.assertEqual(scene.summary, "A dark hand-drawn hallway.")
        self.assertEqual(scene.moments[0].timestamp, 4.0)
        self.assertTrue(visual_analysis._moment_is_safe_for_retention(scene.moments[0]))
        self.assertTrue(scene.moments[1].has_text)
        # self.assertFalse(scene.moments[1].is_animated)
        # self.assertTrue(scene.moments[1].is_animated)
        self.assertFalse(scene.moments[1].is_animated)

    def test_vision_description_only_moment_is_not_treated_as_empty(self) -> None:
        scene = visual_analysis._normalize_vision_response(
            {
                "moments": [
                    {
                        "description": "A shadow crosses a drawn hallway.",
                        "has_text": False,
                        "is_animated": True,
                    }
                ]
            },
            duration=20.0,
            timestamps=[4.0, 8.0],
        )

        self.assertIn("shadow", scene.summary)
        self.assertEqual(scene.moments[0].timestamp, 4.0)
        self.assertTrue(visual_analysis._moment_is_safe_for_retention(scene.moments[0]))

    def test_partial_summary_supplies_only_explicitly_safe_fallbacks(self) -> None:
        scene = visual_analysis._normalize_vision_response(
            {"scene_summary": "Hand-drawn cartoon, no visible text or logos."},
            duration=20.0,
            timestamps=[4.0, 8.0],
        )

        self.assertEqual(len(scene.moments), 2)
        # self.assertTrue(
        #     all(
        #         visual_analysis._moment_is_safe_for_retention(moment)
        #         for moment in scene.moments
        #     )
        # )
        self.assertFalse(
            all(
                visual_analysis._moment_is_safe_for_retention(moment)
                for moment in scene.moments
            )
        )

    def test_partial_moments_fill_missing_frames_without_weakening_safety(self) -> None:
        scene = visual_analysis._normalize_vision_response(
            {
                "scene_summary": "Hand-drawn cartoon, no visible text.",
                "moments": [{"description": "A shadow crosses the hallway."}],
            },
            duration=30.0,
            timestamps=[4.0, 12.0, 20.0],
        )

        self.assertEqual(len(scene.moments), 3)
        # self.assertTrue(
        #     all(
        #         visual_analysis._moment_is_safe_for_retention(moment)
        #         for moment in scene.moments
        #     )
        # )
        self.assertFalse(
            all(
                visual_analysis._moment_is_safe_for_retention(moment)
                for moment in scene.moments
            )
        )

    def test_ambiguous_partial_summary_keeps_fallback_unsafe(self) -> None:
        scene = visual_analysis._normalize_vision_response(
            {
                "scene_summary": "A dim scene.",
                "moments": [{"description": "A shape moves across the frame."}],
            },
            duration=20.0,
            timestamps=[4.0, 8.0],
        )

        self.assertFalse(visual_analysis._moment_is_safe_for_retention(scene.moments[0]))

    def test_ollama_parser_accepts_markdown_fenced_json(self) -> None:
        response = {
            "message": {
                "content": '```json\n{"caption":"A dark animated scene."}\n```'
            }
        }
        with (
            patch.object(visual_analysis, "_ollama_url", return_value="http://localhost"),
            patch.object(visual_analysis, "_request_ollama", return_value=response),
            patch.object(visual_analysis, "_report_frame_progress"),
        ):
            result = visual_analysis._call_ollama("Analizza la scena.", ["frame"])

        self.assertEqual(result["caption"], "A dark animated scene.")

    def test_ollama_accepts_plain_text_as_vision_description_fallback(self) -> None:
        response = {"message": {"content": "Una stanza buia con una persona ferma."}}
        with (
            patch.object(visual_analysis, "_ollama_url", return_value="http://localhost"),
            patch.object(visual_analysis, "_request_ollama", return_value=response),
            patch.object(visual_analysis, "_report_frame_progress"),
        ):
            result = visual_analysis._call_ollama(
                "Analizza la scena.",
                ["frame"],
            )

        self.assertEqual(
            result["scene_summary"],
            "Una stanza buia con una persona ferma.",
        )

    def test_ollama_preserves_moments_from_top_level_json_array(self) -> None:
        moment = {
            "description": "Cartoon senza testo.",
            "has_text": False,
            "is_animated": True,
            "time_seconds": 4.0,
        }
        response = {"message": {"content": json.dumps([moment])}}
        with (
            patch.object(visual_analysis, "_ollama_url", return_value="http://localhost"),
            patch.object(visual_analysis, "_request_ollama", return_value=response),
            patch.object(visual_analysis, "_report_frame_progress") as report_progress,
        ):
            result = visual_analysis._call_ollama("Analizza la scena.", ["frame"])

        self.assertEqual(result, {"moments": [moment]})
        self.assertTrue(
            any(
                "array JSON" in call.args[1]
                and "Cartoon senza testo." in call.args[1]
                for call in report_progress.call_args_list
            )
        )

    def test_ollama_logs_raw_response_when_moment_count_differs(self) -> None:
        moment = {
            "time_seconds": 4.0,
            "description": "Cartoon senza testo.",
            "has_text": False,
            "is_animated": True,
            "score": 0.8,
            "focus_x": 0.5,
            "suitable": True,
        }
        response = {
            "message": {
                "content": json.dumps(
                    {
                        "scene_summary": "Cartoon senza testo.",
                        "actions": ["Una figura si muove."],
                        "moments": [moment],
                    }
                )
            }
        }
        with (
            patch.object(visual_analysis, "_ollama_url", return_value="http://localhost"),
            patch.object(visual_analysis, "_request_ollama", return_value=response),
            patch.object(visual_analysis, "_report_frame_progress") as report_progress,
        ):
            visual_analysis._call_ollama(
                "Analizza le immagini.",
                ["frame-1", "frame-2"],
            )

        self.assertTrue(
            any(
                "numero di moments diverso" in call.args[1]
                and "Cartoon senza testo." in call.args[1]
                for call in report_progress.call_args_list
            )
        )

    def test_ollama_reads_text_fallback_from_alternate_response_field(self) -> None:
        response = {"response": "Una scena animata e priva di scritte."}
        with (
            patch.object(visual_analysis, "_ollama_url", return_value="http://localhost"),
            patch.object(visual_analysis, "_request_ollama", return_value=response),
            patch.object(visual_analysis, "_report_frame_progress") as report_progress,
        ):
            result = visual_analysis._call_ollama("Analizza la scena.", ["frame"])

        self.assertEqual(
            result["scene_summary"],
            "Una scena animata e priva di scritte.",
        )
        self.assertTrue(report_progress.called)

    def test_ollama_retries_token_repeat_errors_with_increasing_temperature(self) -> None:
        repeat_error = RuntimeError(
            'Ollama non è riuscito a eseguire qwen2.5vl:3b '
            '(HTTP 500: {"error":"prediction aborted, token repeat limit reached"}).'
        )
        response = {"message": {"content": '{"scene_summary":"Una scena."}'}}
        with (
            patch.object(visual_analysis, "_ollama_url", return_value="http://localhost"),
            patch.object(
                visual_analysis,
                "_request_ollama",
                side_effect=[repeat_error, repeat_error, response],
            ) as request_ollama,
            patch.object(visual_analysis, "_report_frame_progress"),
        ):
            result = visual_analysis._call_ollama("Analizza la scena.", ["frame"])

        self.assertEqual(result["scene_summary"], "Una scena.")
        self.assertEqual(request_ollama.call_count, 3)
        temperatures = [
            json.loads(call.args[0].data)["options"]["temperature"]
            for call in request_ollama.call_args_list
        ]
        self.assertEqual(temperatures, [0.2, 0.35, 0.5])
        self.assertTrue(
            all(
                json.loads(call.args[0].data)["options"]["repeat_penalty"] == 1.25
                and json.loads(call.args[0].data)["options"]["repeat_last_n"] == 128
                for call in request_ollama.call_args_list
            )
        )

    def test_ollama_does_not_retry_other_http_errors(self) -> None:
        with (
            patch.object(visual_analysis, "_ollama_url", return_value="http://localhost"),
            patch.object(
                visual_analysis,
                "_request_ollama",
                side_effect=RuntimeError("Ollama error (HTTP 500: model unavailable)."),
            ) as request_ollama,
            patch.object(visual_analysis, "_report_frame_progress"),
        ):
            with self.assertRaisesRegex(RuntimeError, "model unavailable"):
                visual_analysis._call_ollama("Analizza la scena.", ["frame"])

        self.assertEqual(request_ollama.call_count, 1)

    def test_story_prompt_includes_audio_duration_feedback(self) -> None:
        scene = SceneAnalysis(
            summary="Una pressa idraulica schiaccia oggetti.",
            actions=("La pressa comprime una serie di oggetti.",),
            moments=(VisualMoment(3.0, 0.9, 0.5),),
        )
        narration = " ".join(f"parola{index}" for index in range(90))
        feedback = "La traccia precedente durava 28.0s, 2.0s sotto il minimo."
        with patch(
            "video_pipeline.visual_analysis._call_ollama",
            return_value={"narration": narration},
        ) as ollama:
            generate_storytelling_script(
                "Un esperimento soddisfacente",
                scene,
                "",
                target_duration=35.0,
                language="it",
                duration_feedback=feedback,
            )

        self.assertIn(feedback, ollama.call_args.args[0])

    def test_long_story_retries_short_output_and_targets_75_to_115_words(self) -> None:
        scene = SceneAnalysis(
            summary="Una pressa idraulica schiaccia oggetti.",
            actions=("La pressa comprime una serie di oggetti.",),
            moments=(VisualMoment(3.0, 0.9, 0.5),),
        )
        narration = " ".join(f"parola{index}" for index in range(90))
        with patch(
            "video_pipeline.visual_analysis._call_ollama",
            side_effect=[
                {"narration": "Testo troppo breve."},
                {"narration": narration},
            ],
        ) as ollama:
            result = generate_storytelling_script(
                "Un esperimento soddisfacente",
                scene,
                "",
                target_duration=35.0,
                language="it",
            )

        self.assertEqual(result, narration)
        self.assertEqual(ollama.call_count, 2)
        # self.assertIn("75-115 parole", ollama.call_args_list[1].args[0])
        self.assertIn("conteggio esatto della bozza precedente è 3 parole", ollama.call_args_list[1].args[0])
        self.assertIn("È troppo breve", ollama.call_args_list[1].args[0])
        self.assertIn("90-95 parole", ollama.call_args_list[1].args[0])

    def test_long_story_uses_third_attempt_for_word_count_compliance(self) -> None:
        scene = SceneAnalysis(
            summary="Una pressa idraulica schiaccia oggetti.",
            actions=("La pressa comprime un oggetto.",),
            moments=(VisualMoment(3.0, 0.9, 0.5),),
        )
        valid_narration = " ".join(f"parola{index}" for index in range(90))
        too_long_narration = " ".join(f"parola{index}" for index in range(120))
        with patch(
            "video_pipeline.visual_analysis._call_ollama",
            side_effect=[
                {"narration": "Testo troppo breve."},
                {"narration": too_long_narration},
                {"narration": valid_narration},
            ],
        ) as ollama:
            result = generate_storytelling_script(
                "Un esperimento soddisfacente",
                scene,
                "",
                target_duration=35.0,
                language="it",
            )

        self.assertEqual(result, valid_narration)
        self.assertEqual(ollama.call_count, 3)
        third_prompt = ollama.call_args_list[2].args[0]
        self.assertIn("almeno 75 e al massimo 115 parole", third_prompt)
        # self.assertIn("120 parole", third_prompt)
        self.assertIn("conteggio esatto della bozza precedente è 120 parole", third_prompt)
        self.assertIn("È troppo lunga", third_prompt)

    # def test_long_story_accepts_small_word_count_deviation_after_retries(self) -> None:
    def test_long_story_chooses_closest_attempt_after_retries(self) -> None:
        scene = SceneAnalysis(
            summary="Una pressa idraulica schiaccia oggetti.",
            actions=("La pressa comprime un oggetto.",),
            moments=(VisualMoment(3.0, 0.9, 0.5),),
        )
        short_draft = " ".join(f"parola{index}" for index in range(72)) + "."
        long_draft = " ".join(f"parola{index}" for index in range(120)) + "."
        near_limit_draft = " ".join(f"parola{index}" for index in range(118)) + "."
        with patch(
            "video_pipeline.visual_analysis._call_ollama",
            side_effect=[
                {"narration": short_draft},
                {"narration": long_draft},
                {"narration": near_limit_draft},
            ],
        ) as ollama:
            result = generate_storytelling_script(
                "Un esperimento soddisfacente",
                scene,
                "",
                target_duration=35.0,
                language="it",
            )

        # self.assertEqual(result, near_limit_draft)
        self.assertEqual(result, short_draft)
        self.assertEqual(ollama.call_count, 3)
        self.assertIn("72 parole", ollama.call_args_list[1].args[0])
        self.assertIn("120 parole", ollama.call_args_list[2].args[0])

    def test_long_story_accepts_small_shortfall_after_retries(self) -> None:
        scene = SceneAnalysis(
            summary="Una pressa idraulica schiaccia oggetti.",
            actions=("La pressa comprime un oggetto.",),
            moments=(VisualMoment(3.0, 0.9, 0.5),),
        )
        near_minimum_draft = " ".join(f"parola{index}" for index in range(72)) + "."
        with patch(
            "video_pipeline.visual_analysis._call_ollama",
            return_value={"narration": near_minimum_draft},
        ) as ollama:
            result = generate_storytelling_script(
                "Un esperimento soddisfacente",
                scene,
                "",
                target_duration=35.0,
                language="it",
            )

        self.assertEqual(result, near_minimum_draft)
        self.assertEqual(ollama.call_count, 3)

    # def test_long_story_still_rejects_output_far_outside_tolerance(self) -> None:
    def test_long_story_uses_best_available_output_far_below_range(self) -> None:
        scene = SceneAnalysis(
            summary="Una pressa idraulica schiaccia oggetti.",
            actions=("La pressa comprime un oggetto.",),
            moments=(VisualMoment(3.0, 0.9, 0.5),),
        )
        too_short = " ".join(f"parola{index}" for index in range(20))
        with patch(
            "video_pipeline.visual_analysis._call_ollama",
            return_value={"narration": too_short},
        ) as ollama:
            # with self.assertRaisesRegex(RuntimeError, "non rientra in 75-115 parole"):
            #     generate_storytelling_script(
            result = generate_storytelling_script(
                "Un esperimento soddisfacente",
                scene,
                "",
                target_duration=35.0,
                language="it",
            )

        self.assertEqual(result, too_short)
        self.assertEqual(ollama.call_count, 3)

    def test_long_story_selects_closest_attempt_and_truncates_at_sentence(self) -> None:
        scene = SceneAnalysis(
            summary="Una pressa idraulica schiaccia oggetti.",
            actions=("La pressa comprime un oggetto.",),
            moments=(VisualMoment(3.0, 0.9, 0.5),),
        )
        complete_sentence = " ".join(f"parola{index}" for index in range(110)) + "."
        closest_draft = complete_sentence + " " + " ".join(
            f"extra{index}" for index in range(6)
        )
        with patch(
            "video_pipeline.visual_analysis._call_ollama",
            side_effect=[
                {"narration": " ".join(f"a{index}" for index in range(120))},
                {"narration": closest_draft},
                {"narration": " ".join(f"b{index}" for index in range(118))},
            ],
        ) as ollama:
            result = generate_storytelling_script(
                "Un esperimento soddisfacente",
                scene,
                "",
                target_duration=35.0,
                language="it",
            )

        self.assertEqual(result, complete_sentence)
        self.assertEqual(ollama.call_count, 3)
        self.assertEqual(len(result.split()), 110)
        self.assertTrue(result.endswith("."))

    def test_long_narration_truncation_keeps_last_whole_word_if_no_sentence_ends(self) -> None:
        draft = " ".join(f"parola{index}" for index in range(120))

        truncated = visual_analysis._truncate_narration_to_word_limit(draft, 115)

        self.assertEqual(len(truncated.split()), 115)
        self.assertEqual(truncated.split()[-1], "parola114.")

    # def test_pipeline_retries_for_target_audio_and_repeats_short_source(self) -> None:
    def test_pipeline_retries_for_target_audio_using_only_safe_animation(self) -> None:
        scripts = [
            " ".join(f"tentativo{attempt}_parola{index}" for index in range(90))
            for attempt in range(1, 4)
        ]
        scene = SceneAnalysis(
            summary="La pressa schiaccia oggetti.",
            actions=("La pressa scende e comprime.",),
            # moments=(VisualMoment(4.0, 0.9, 0.5),),
            moments=(
                _animated_moment(15.0, 0.9),
                _animated_moment(28.0, 0.8),
            ),
        )

        def write_test_audio(script: str, destination: Path, language: str) -> None:
            destination.write_bytes(script.encode("utf-8"))

        rendered_narration: list[bytes] = []

        def capture_rendered_narration(*args: Any, **kwargs: Any) -> None:
            rendered_narration.append(args[1].read_bytes())

        with tempfile.TemporaryDirectory() as temporary:
            project_root = Path(temporary)
            output_root = project_root / "downloads" / "processed"
            output_root.mkdir(parents=True)
            with (
                # patch.dict(
                #     "os.environ",
                #     {
                #         "ELEVENLABS_API_KEY": "test-key",
                #         "ELEVENLABS_VOICE_ID": "test-voice",
                #     },
                # ),
                patch.object(process_video, "PROJECT_ROOT", project_root),
                patch.object(process_video, "OUTPUT_ROOT", output_root),
                patch.object(
                    process_video,
                    "validate_input",
                    return_value=project_root / "source.mp4",
                ),
                # patch.object(process_video, "analyze_video", return_value=(8.0, [], 0.5)),
                patch.object(process_video, "analyze_video", return_value=(40.0, [], 0.5)),
                patch.object(process_video, "analyze_keyframes", return_value=scene),
                patch.object(
                    process_video,
                    "verify_retention_samples",
                    return_value=list(scene.moments),
                ),
                patch.object(process_video, "load_whisper_model", return_value=object()),
                patch.object(
                    process_video,
                    "transcribe",
                    side_effect=[{"segments": []}, {"segments": []}],
                ),
                patch.object(
                    process_video,
                    "generate_storytelling_script",
                    # side_effect=[script, script],
                    side_effect=scripts,
                ) as generate_script,
                # patch.object(process_video, "synthesize_narration") as synthesize,
                patch.object(
                    process_video,
                    "synthesize_narration",
                    side_effect=write_test_audio,
                ) as synthesize,
                # patch.object(process_video, "audio_duration", side_effect=[4.0, 35.0]),
                patch.object(
                    process_video,
                    "audio_duration",
                    side_effect=[28.0, 42.0, 36.0],
                ),
                patch.object(process_video, "write_word_captions", return_value=90),
                # patch.object(process_video, "render_vertical_short") as render,
                patch.object(
                    process_video,
                    "render_vertical_short",
                    side_effect=capture_rendered_narration,
                ) as render,
                redirect_stdout(io.StringIO()),
            ):
                result = process_video.run_pipeline(
                    project_root / "source.mp4",
                    "video123",
                    "Un breve spunto",
                    "it",
                    rights_confirmed=True,
                )

        # self.assertEqual(result["duration_seconds"], 35.0)
        self.assertEqual(result["duration_seconds"], 36.0)
        self.assertEqual(generate_script.call_args_list[0].args[3], 35.0)
        # self.assertEqual(generate_script.call_args_list[1].args[3], 40.0)
        self.assertEqual(generate_script.call_count, 3)
        self.assertIn("sotto il minimo", generate_script.call_args_list[1].kwargs["duration_feedback"])
        self.assertIn("sopra il massimo", generate_script.call_args_list[2].kwargs["duration_feedback"])
        # self.assertEqual(synthesize.call_count, 2)
        self.assertEqual(synthesize.call_count, 3)
        self.assertEqual(synthesize.call_args.args[1].suffix, ".wav")
        self.assertEqual(synthesize.call_args.args[2], "it")
        # The third attempt is closest to the 35-second target and is rendered.
        self.assertEqual(rendered_narration, [scripts[2].encode("utf-8")])
        rendered_clips = render.call_args.args[4]
        self.assertAlmostEqual(
            sum(clip.end - clip.start for clip in rendered_clips),
            36.0,
        )
        # self.assertTrue(all(clip.end <= 8.0 for clip in rendered_clips))
        self.assertTrue(all(clip.start >= 8.0 and clip.end <= 32.0 for clip in rendered_clips))

    def test_pipeline_reports_candidate_rejection_for_sequential_retry(self) -> None:
        output = io.StringIO()
        with (
            patch.object(
                sys,
                "argv",
                [
                    "process_video",
                    "--input",
                    "source.mp4",
                    "--video-id",
                    "video123",
                    "--brief",
                    "Un breve spunto",
                    "--rights-confirmed",
                    "--allow-candidate-rejection",
                ],
            ),
            patch.object(
                process_video,
                "run_pipeline",
                side_effect=process_video.CandidateRejectedError(
                    "Il video contiene testo visibile."
                ),
            ),
            redirect_stdout(output),
        ):
            result = process_video.main()

        self.assertEqual(result, 0)
        event = json.loads(output.getvalue())
        self.assertEqual(event["type"], "result")
        self.assertTrue(event["candidate_rejected"])

    def test_ollama_endpoint_must_be_local(self) -> None:
        with patch.dict("os.environ", {"OLLAMA_HOST": "http://example.com:11434"}):
            with self.assertRaisesRegex(RuntimeError, "localhost"):
                _ollama_url()

    def test_ollama_hard_timeout_emits_json_progress(self) -> None:
        request_started = threading.Event()
        release_request = threading.Event()
        output = io.StringIO()

        def delayed_urlopen(request: Request, timeout: float) -> io.BytesIO:
            self.assertEqual(
                request.full_url,
                "http://127.0.0.1:11434/api/generate",
            )
            self.assertGreater(timeout, 0)
            request_started.set()
            release_request.wait(timeout=1)
            return io.BytesIO(b"{}")

        with (
            patch.object(visual_analysis, "OLLAMA_REQUEST_TIMEOUT_SECONDS", 0.05),
            patch.object(visual_analysis, "OLLAMA_PROGRESS_INTERVAL_SECONDS", 0.01),
            patch.object(visual_analysis, "urlopen", side_effect=delayed_urlopen),
            redirect_stdout(output),
        ):
            try:
                with self.assertRaises(TimeoutError):
                    visual_analysis._request_ollama(
                        Request("http://127.0.0.1:11434/api/generate"),
                        "qwen2.5vl:3b",
                    )
            finally:
                release_request.set()

        self.assertTrue(request_started.wait(timeout=1))
        events = [json.loads(line) for line in output.getvalue().splitlines()]
        self.assertTrue(events)
        self.assertTrue(all(event["type"] == "progress" for event in events))
        self.assertTrue(all(event["stage"] == "frame_analysis" for event in events))

    def test_ollama_http_error_includes_status(self) -> None:
        error = HTTPError(
            "http://127.0.0.1:11434/api/generate",
            404,
            "model not found",
            hdrs=None,
            fp=io.BytesIO(b""),
        )
        with patch.object(visual_analysis, "urlopen", side_effect=error):
            with self.assertRaisesRegex(RuntimeError, "HTTP 404"):
                visual_analysis._request_ollama(
                    Request("http://127.0.0.1:11434/api/generate"),
                    "qwen2.5vl:3b",
                )

    def test_input_must_be_inside_downloads_and_rights_must_be_confirmed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            downloads = Path(temporary) / "downloads"
            downloads.mkdir()
            video = downloads / "video_123.mp4"
            video.touch()
            original_download_root = process_video.DOWNLOAD_ROOT
            process_video.DOWNLOAD_ROOT = downloads.resolve()
            try:
                self.assertEqual(
                    validate_input(video, "video_123", rights_confirmed=True),
                    video.resolve(),
                )
                with self.assertRaises(ValueError):
                    validate_input(video, "video_123", rights_confirmed=False)
                with self.assertRaises(ValueError):
                    validate_input(video, "other_123", rights_confirmed=True)
            finally:
                process_video.DOWNLOAD_ROOT = original_download_root

    # def test_word_captions_include_word_timestamps_and_highlight(self) -> None:
    def test_word_captions_are_static_and_keep_sentence_timing(self) -> None:
        transcription = {
            "segments": [
                {
                    "words": [
                        # {"word": "Ciao", "start": 0.0, "end": 0.4},
                        {"word": "La", "start": 0.0, "end": 0.4},
                        # {"word": "mondo.", "start": 0.4, "end": 0.9},
                        {"word": "ombra.", "start": 0.4, "end": 0.9},
                        # {"word": "Nuova", "start": 1.0, "end": 1.4},
                        {"word": "Il", "start": 1.0, "end": 1.4},
                        # {"word": "frase.", "start": 1.4, "end": 1.8},
                        {"word": "buio.", "start": 1.4, "end": 1.8},
                    ]
                }
            ]
        }
        with tempfile.TemporaryDirectory() as temporary:
            destination = Path(temporary) / "captions.ass"
            count = write_word_captions(transcription, destination)
            contents = destination.read_text(encoding="utf-8")

        # self.assertEqual(count, 2)
        self.assertEqual(count, 4)
        dialogue_lines = [
            line for line in contents.splitlines() if line.startswith("Dialogue: ")
        ]
        self.assertEqual(len(dialogue_lines), 2)
        # self.assertIn("0:00:00.00,0:00:00.40", contents)
        self.assertIn("0:00:00.00,0:00:01.00", dialogue_lines[0])
        # self.assertIn("Ciao mondo.", dialogue_lines[0])
        self.assertIn(r"la {\fnNosifer\b1}OMBRA.{\r}", dialogue_lines[0])
        # self.assertIn("Nuova frase.", dialogue_lines[1])
        self.assertIn(r"il {\fnSpecial Elite\b1}BUIO.{\r}", dialogue_lines[1])
        # self.assertIn(r"{\c&H0000FFFF&\b1}", contents)
        self.assertNotIn(r"{\c", contents)
        self.assertNotRegex(contents, r"\\(?:k|kf|ko|move|pos)")
        self.assertIn("WrapStyle: 0", contents)
        self.assertIn(
            f"Style: Default,{captions._caption_font_name()},78,",
            contents,
        )
        # self.assertIn(",5,2,2,125,125,610,1", contents)
        self.assertIn(",5,2,2,125,125,640,1", contents)

    def test_word_captions_wrap_lines_and_apply_safe_margins(self) -> None:
        words = [
            {"word": f"parola{index}", "start": index * 0.3, "end": (index + 1) * 0.3}
            for index in range(6)
        ]
        with tempfile.TemporaryDirectory() as temporary:
            destination = Path(temporary) / "captions.ass"
            write_word_captions({"segments": [{"words": words}]}, destination)
            contents = destination.read_text(encoding="utf-8")

        self.assertIn(r"\N", contents)
        # self.assertIn("Default,,125,125,610,,", contents)
        self.assertIn("Default,,125,125,640,,", contents)
        subtitle_text = [
            line.split(",", 9)[9]
            for line in contents.splitlines()
            if line.startswith("Dialogue: ")
        ]
        visible_subtitle_text = [
            re.sub(r"\{[^}]*\}", "", line) for line in subtitle_text
        ]
        self.assertTrue(
            all(
                len(line) <= captions.CAPTION_MAX_LINE_CHARACTERS
                for text in visible_subtitle_text
                for line in text.split(r"\N")
            )
        )


if __name__ == "__main__":
    unittest.main()
