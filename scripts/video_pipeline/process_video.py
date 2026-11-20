import argparse
import json
import os
import re
import sys
import tempfile
import uuid
from pathlib import Path
from typing import Any

from .captions import write_word_captions
from .events import report_progress
from .render import audio_duration, render_vertical_short
from .speech import (
    load_whisper_model,
    synthesize_narration,
    transcribe,
)
from .visual_analysis import (
    analyze_keyframes,
    analyze_video,
    choose_retention_clips,
    generate_storytelling_script,
    verify_retention_samples,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DOWNLOAD_ROOT = (PROJECT_ROOT / "downloads").resolve()
OUTPUT_ROOT = DOWNLOAD_ROOT / "processed"
ALLOWED_EXTENSIONS = {".mp4", ".mkv", ".webm", ".mov", ".m4v"}
ALLOWED_LANGUAGES = {"de", "en", "es", "fr", "it", "pt"}
NARRATION_TARGET_SECONDS = 35.0
NARRATION_MIN_SECONDS = 30.0
NARRATION_MAX_SECONDS = 40.0


class CandidateRejectedError(RuntimeError):
    pass


def validate_input(video_path: Path, video_id: str, rights_confirmed: bool) -> Path:
    if not rights_confirmed:
        raise ValueError("Conferma i diritti di utilizzo del video prima di elaborarlo.")
    if not re.fullmatch(r"[A-Za-z0-9_-]{6,32}", video_id):
        raise ValueError("ID video non valido.")

    resolved = video_path.resolve(strict=True)
    try:
        resolved.relative_to(DOWNLOAD_ROOT)
    except ValueError as error:
        raise ValueError("È possibile elaborare solo file nella cartella downloads/.") from error

    if (
        not resolved.is_file()
        or resolved.suffix.lower() not in ALLOWED_EXTENSIONS
        or not resolved.name.startswith(f"{video_id}.")
    ):
        raise ValueError("Il file deve essere un video scaricato con l'ID indicato.")
    return resolved


def _transcript_text(transcription: dict[str, Any]) -> str:
    return " ".join(
        str(segment.get("text", "")).strip()
        for segment in transcription.get("segments", [])
        if segment.get("text")
    )


def run_pipeline(
    video_path: Path,
    video_id: str,
    brief: str,
    language: str,
    rights_confirmed: bool,
) -> dict[str, Any]:
    source_path = validate_input(video_path, video_id, rights_confirmed)
    if language not in ALLOWED_LANGUAGES:
        raise ValueError("Lingua non supportata. Usa de, en, es, fr, it o pt.")
    if not 5 <= len(brief.strip()) <= 400:
        raise ValueError("Lo spunto narrativo deve contenere da 5 a 400 caratteri.")
    # if not os.environ.get("ELEVENLABS_API_KEY") or not os.environ.get("ELEVENLABS_VOICE_ID"):
    #     raise RuntimeError("Configura ELEVENLABS_API_KEY e ELEVENLABS_VOICE_ID nell'ambiente.")

    # OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    # output_directory = Path(
    #     tempfile.mkdtemp(prefix=f"{video_id}_{uuid.uuid4().hex[:8]}_", dir=OUTPUT_ROOT)
    # )
    report_progress("download", 3, 100, "Video già scaricato: file locale verificato.")

    report_progress(
        "frame_analysis",
        8,
        0,
        "Estrazione dei fotogrammi chiave e analisi multimodale locale con Qwen.",
    )
    source_duration, motion_samples, focus_x = analyze_video(str(source_path))
    if source_duration < 5:
        # raise ValueError("Il video deve durare almeno 5 secondi per generare una narrazione utile.")
        raise CandidateRejectedError("Il video è troppo corto per fornire clip utili.")
    # scene = analyze_keyframes(source_path, source_duration)
    scene = analyze_keyframes(source_path, source_duration, brief=brief)
    # retention_samples = motion_samples + list(scene.moments)
    # retention_samples = list(scene.moments) if scene.moments else motion_samples
    # retention_samples = list(scene.moments)
    retention_samples = verify_retention_samples(
        source_path,
        list(scene.moments),
        motion_samples,
        source_duration,
        brief=brief,
    )
    # initial_clips = choose_retention_clips(
    #     retention_samples,
    #     source_duration,
    #     NARRATION_TARGET_SECONDS,
    # )
    initial_clips = choose_retention_clips(
        retention_samples,
        source_duration,
        NARRATION_TARGET_SECONDS,
    )
    for index, clip in enumerate(initial_clips, start=1):
        report_progress(
            "frame_analysis",
            18,
            99,
            f"Clip selezionata {index}/{len(initial_clips)}: "
            f"{clip.start:.2f}-{clip.end:.2f}s (score {clip.score:.2f}).",
        )
    # unique_clean_clips = {
    #     (clip.start, clip.end)
    #     for clip in initial_clips
    # }
    unique_clean_clip_starts = {
        round(clip.start, 3)
        for clip in initial_clips
    }
    if len(unique_clean_clip_starts) < 2:
        # raise CandidateRejectedError(
        #     "Il video non contiene abbastanza segmenti animati verificati e senza testo."
        # )
        raise CandidateRejectedError(
            "Il video contiene "
            f"{len(unique_clean_clip_starts)} segmenti animati verificati e senza testo; "
            "ne servono almeno 2."
        )
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    output_directory = Path(
        tempfile.mkdtemp(prefix=f"{video_id}_{uuid.uuid4().hex[:8]}_", dir=OUTPUT_ROOT)
    )
    report_progress(
        "frame_analysis",
        18,
        100,
        f"Scena: {scene.summary}",
    )

    report_progress("source_transcription", 20, 0, "Trascrizione della sorgente con Whisper.")
    model = load_whisper_model()
    source_transcription = transcribe(model, source_path, language, word_timestamps=False)
    report_progress("source_transcription", 38, 100, "Trascrizione sorgente completata.")

    # target_duration = min(source_duration * 0.9, 30.0)
    target_duration = NARRATION_TARGET_SECONDS
    # narration_path = output_directory / "narration.mp3"
    narration_path = output_directory / "narration.wav"
    script = ""
    duration = 0.0
    # for attempt in range(3):
    #     report_progress(
    #         "story_generation",
    #         40,
    #         0,
    #         "Creazione di un gancio coerente con le scene osservate.",
    #     )
    #     script = generate_storytelling_script(
    #         brief,
    #         scene,
    #         _transcript_text(source_transcription),
    #         target_duration,
    #         language,
    #     )
    #     report_progress("story_generation", 45, 100, "Script narrativo pronto.")
    #
    #     report_progress(
    #         "elevenlabs_synthesis",
    #         47,
    #         0,
    #         f"Generazione della voce narrante con TTS locale (tentativo {attempt + 1}/3).",
    #     )
    #     # synthesize_narration(script, narration_path)
    #     synthesize_narration(script, narration_path, language)
    #     duration = audio_duration(narration_path)
    #     # if duration <= source_duration:
    #     if NARRATION_MIN_SECONDS <= duration <= NARRATION_MAX_SECONDS:
    #         break
    #     target_duration = min(
    #         NARRATION_MAX_SECONDS,
    #         max(
    #             NARRATION_MIN_SECONDS,
    #             target_duration * NARRATION_TARGET_SECONDS / max(duration, 0.1),
    #         ),
    #     )
    # else:
    #     # raise RuntimeError(
    #     #     "La traccia narrata supera la durata della sorgente dopo tre tentativi."
    #     # )
    #     raise RuntimeError(
    #         "La traccia narrata non è rientrata nel target "
    #         f"{NARRATION_MIN_SECONDS:.0f}-{NARRATION_MAX_SECONDS:.0f}s "
    #         "dopo tre tentativi."
    #     )
    best_attempt: tuple[float, str, Path] | None = None
    duration_feedback: str | None = None
    for attempt in range(3):
        report_progress(
            "story_generation",
            40,
            0,
            "Creazione di un gancio coerente con le scene osservate.",
        )
        script = generate_storytelling_script(
            brief,
            scene,
            _transcript_text(source_transcription),
            target_duration,
            language,
            duration_feedback=duration_feedback,
        )
        report_progress("story_generation", 45, 100, "Script narrativo pronto.")

        report_progress(
            "elevenlabs_synthesis",
            47,
            0,
            f"Generazione della voce narrante con TTS locale (tentativo {attempt + 1}/3).",
        )
        attempt_path = output_directory / f"narration_attempt_{attempt + 1}.wav"
        synthesize_narration(script, attempt_path, language)
        attempt_duration = audio_duration(attempt_path)
        if (
            best_attempt is None
            or abs(attempt_duration - NARRATION_TARGET_SECONDS)
            < abs(best_attempt[0] - NARRATION_TARGET_SECONDS)
        ):
            best_attempt = (attempt_duration, script, attempt_path)

        if abs(attempt_duration - NARRATION_TARGET_SECONDS) < 0.05:
            break

        difference_from_center = abs(attempt_duration - NARRATION_TARGET_SECONDS)
        if attempt_duration < NARRATION_MIN_SECONDS:
            duration_feedback = (
                f"La traccia precedente durava {attempt_duration:.1f}s, "
                f"{NARRATION_MIN_SECONDS - attempt_duration:.1f}s sotto il minimo "
                f"di {NARRATION_MIN_SECONDS:.0f}s. Allunga la narrazione per avvicinarla "
                f"a {NARRATION_TARGET_SECONDS:.0f}s, restando tra "
                f"{NARRATION_MIN_SECONDS:.0f} e {NARRATION_MAX_SECONDS:.0f}s."
            )
        elif attempt_duration > NARRATION_MAX_SECONDS:
            duration_feedback = (
                f"La traccia precedente durava {attempt_duration:.1f}s, "
                f"{attempt_duration - NARRATION_MAX_SECONDS:.1f}s sopra il massimo "
                f"di {NARRATION_MAX_SECONDS:.0f}s. Accorcia la narrazione per avvicinarla "
                f"a {NARRATION_TARGET_SECONDS:.0f}s, restando tra "
                f"{NARRATION_MIN_SECONDS:.0f} e {NARRATION_MAX_SECONDS:.0f}s."
            )
        else:
            direction = "sopra" if attempt_duration > NARRATION_TARGET_SECONDS else "sotto"
            duration_feedback = (
                f"La traccia precedente durava {attempt_duration:.1f}s, "
                f"{difference_from_center:.1f}s {direction} il target centrale di "
                f"{NARRATION_TARGET_SECONDS:.0f}s (range {NARRATION_MIN_SECONDS:.0f}-"
                f"{NARRATION_MAX_SECONDS:.0f}s). Modifica leggermente la lunghezza "
                "per avvicinarti a 35s senza uscire dal range."
            )
        report_progress(
            "story_generation",
            40,
            0,
            f"Feedback durata: {duration_feedback}",
        )
        target_duration = min(
            NARRATION_MAX_SECONDS,
            max(
                NARRATION_MIN_SECONDS,
                target_duration * NARRATION_TARGET_SECONDS / max(attempt_duration, 0.1),
            ),
        )
    if best_attempt is None:
        raise RuntimeError("Non è stato possibile generare una traccia narrata.")
    duration, script, best_attempt_path = best_attempt
    best_attempt_path.replace(narration_path)
    for attempt in range(3):
        unused_attempt_path = output_directory / f"narration_attempt_{attempt + 1}.wav"
        if unused_attempt_path != narration_path and unused_attempt_path.exists():
            unused_attempt_path.unlink()

    # report_progress("elevenlabs_synthesis", 60, 100, "Traccia narrata generata.")
    report_progress("elevenlabs_synthesis", 60, 100, "Voce narrante locale generata.")

    report_progress("narration_transcription", 62, 0, "Whisper allinea le parole della narrazione.")
    narration_transcription = transcribe(
        model,
        narration_path,
        language,
        word_timestamps=True,
    )
    captions_path = output_directory / "captions.ass"
    word_count = write_word_captions(narration_transcription, captions_path)
    report_progress("narration_transcription", 72, 100, f"Timestamp ottenuti per {word_count} parole.")

    # clips = choose_retention_clips(retention_samples, source_duration, duration)
    clips = choose_retention_clips(retention_samples, source_duration, duration)
    report_progress("rendering", 75, 0, "Montaggio 9:16, tagli di retention e sottotitoli.")
    output_path = output_directory / "final.mp4"
    render_vertical_short(
        source_path,
        narration_path,
        captions_path,
        output_path,
        clips,
        focus_x,
        duration,
    )
    report_progress("rendering", 98, 100, "Montaggio completato.")

    relative_output = output_path.relative_to(PROJECT_ROOT)
    return {
        "output_file": str(relative_output),
        "script": script,
        "scene_summary": scene.summary,
        "duration_seconds": round(duration, 2),
        "focus_x": round(focus_x, 3),
        "clips": [
            {"start": round(clip.start, 2), "end": round(clip.end, 2)}
            for clip in clips
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Elabora un video locale in formato verticale.")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--video-id", required=True)
    parser.add_argument("--brief", required=True)
    parser.add_argument("--language", default="it")
    parser.add_argument("--rights-confirmed", action="store_true")
    parser.add_argument(
        "--allow-candidate-rejection",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    args = parser.parse_args()

    try:
        result = run_pipeline(
            args.input,
            args.video_id,
            args.brief,
            args.language,
            args.rights_confirmed,
        )
    except CandidateRejectedError as error:
        if args.allow_candidate_rejection:
            print(
                json.dumps(
                    {
                        "type": "result",
                        "candidate_rejected": True,
                        "reason": str(error),
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
            return 0
        print(f"Elaborazione non riuscita: {error}", file=sys.stderr)
        return 1
    except (OSError, RuntimeError, ValueError) as error:
        print(f"Elaborazione non riuscita: {error}", file=sys.stderr)
        return 1

    print(
        json.dumps({"type": "result", **result}, ensure_ascii=False),
        flush=True,
    )
    report_progress("completed", 100, 100, "Video verticale pronto.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
