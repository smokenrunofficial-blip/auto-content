import json
import logging
import math
import os
import random
import re
import subprocess
from pathlib import Path

from .captions import CAPTION_FONT_DIRECTORY
from .events import report_progress
from .visual_analysis import Clip


MUSIC_DIRECTORY = Path(__file__).resolve().parents[2] / "assets" / "music"
AUDIO_SILENCE_THRESHOLD_DB = -55.0
FACE_DETECTION_MAX_WIDTH = 480
FACE_DETECTION_SAMPLE_INTERVAL_SECONDS = 0.5
FACE_TRACK_SMOOTHING = 0.45
FACE_TRACK_MAX_STEP = 0.12
FACE_TRACK_GRACE_PERIOD_SECONDS = 1.5
FACE_CROP_SAFE_MARGIN_RATIO = 0.2
LOGGER = logging.getLogger(__name__)


def _select_background_music() -> Path | None:
    MUSIC_DIRECTORY.mkdir(parents=True, exist_ok=True)
    music_files = sorted(
        path
        for path in MUSIC_DIRECTORY.iterdir()
        if path.is_file() and path.suffix.casefold() in {".mp3", ".wav"}
    )
    return random.choice(music_files) if music_files else None


def _escape_filter_path(path: Path) -> str:
    return (
        str(path.resolve())
        .replace("\\", "/")
        .replace(":", r"\:")
        .replace("'", r"\'")
        .replace(",", r"\,")
    )


def _horror_overlay_filter_expression() -> str | None:
    enabled_value = os.environ.get("HORROR_OVERLAY_ENABLED", "1").strip().casefold()
    if enabled_value not in {"1", "true", "yes", "on", "0", "false", "no", "off"}:
        raise ValueError("HORROR_OVERLAY_ENABLED deve essere un valore booleano.")
    if enabled_value in {"0", "false", "no", "off"}:
        return None

    mode = os.environ.get("HORROR_OVERLAY_MODE", "analog").strip().casefold()
    if mode == "off":
        return None
    if mode not in {"analog", "noise", "scanlines"}:
        raise ValueError(
            "HORROR_OVERLAY_MODE deve essere 'analog', 'noise', 'scanlines' o 'off'."
        )
    try:
        noise_level = int(os.environ.get("HORROR_OVERLAY_NOISE", "4"))
        scanline_opacity = float(
            os.environ.get("HORROR_OVERLAY_SCANLINE_OPACITY", "0.12")
        )
    except ValueError as error:
        raise ValueError(
            "HORROR_OVERLAY_NOISE e HORROR_OVERLAY_SCANLINE_OPACITY "
            "devono essere valori numerici."
        ) from error
    if not 0 <= noise_level <= 100:
        raise ValueError("HORROR_OVERLAY_NOISE deve essere compreso tra 0 e 100.")
    if not math.isfinite(scanline_opacity) or not 0.0 <= scanline_opacity <= 1.0:
        raise ValueError(
            "HORROR_OVERLAY_SCANLINE_OPACITY deve essere compreso tra 0 e 1."
        )

    filters: list[str] = []
    if mode in {"analog", "noise"} and noise_level:
        filters.append(f"noise=alls={noise_level}:allf=t+u")
    if mode in {"analog", "scanlines"} and scanline_opacity:
        filters.append(
            "drawgrid=width=iw:height=6:thickness=1:"
            f"color=black@{scanline_opacity:.3f}"
        )
    return ",".join(filters) or None


def audio_duration(audio_path: Path) -> float:
    probe = subprocess.run(
        [
            os.environ.get("FFPROBE_EXECUTABLE", "ffprobe"),
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "json",
            str(audio_path),
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    duration = float(json.loads(probe.stdout)["format"]["duration"])
    if duration <= 0:
        raise RuntimeError("La durata dell'audio narrato non è valida.")
    return duration


def _measure_audio_peak(audio_path: Path) -> float:
    probe = subprocess.run(
        [
            os.environ.get("FFMPEG_EXECUTABLE", "ffmpeg"),
            "-hide_banner",
            "-i",
            str(audio_path),
            "-vn",
            "-af",
            "volumedetect",
            "-f",
            "null",
            "-",
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=120,
    )
    if probe.returncode != 0:
        detail = probe.stderr.strip()[-1000:] or "errore FFmpeg non specificato"
        raise RuntimeError(
            f"Impossibile leggere il livello audio di {audio_path}: {detail}"
        )
    match = re.search(
        r"max_volume:\s*(-?(?:\d+(?:\.\d+)?|inf))\s*dB",
        probe.stderr,
        flags=re.IGNORECASE,
    )
    if match is None:
        raise RuntimeError(
            f"FFmpeg non ha restituito il livello di picco per {audio_path}."
        )
    return float(match.group(1))


def _verify_rendered_audio(output_path: Path) -> tuple[str, float]:
    probe = subprocess.run(
        [
            os.environ.get("FFPROBE_EXECUTABLE", "ffprobe"),
            "-v",
            "error",
            "-select_streams",
            "a:0",
            "-show_entries",
            "stream=codec_name",
            "-of",
            "json",
            str(output_path),
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    streams = json.loads(probe.stdout).get("streams", [])
    if not streams:
        raise RuntimeError(
            f"Il video renderizzato non contiene una traccia audio: {output_path}"
        )
    codec = str(streams[0].get("codec_name", "sconosciuto"))
    peak_db = _measure_audio_peak(output_path)
    if peak_db <= AUDIO_SILENCE_THRESHOLD_DB:
        raise RuntimeError(
            "La traccia audio finale è silenziosa o quasi muta "
            f"(picco {peak_db:.1f} dBFS): {output_path}"
        )
    return codec, peak_db


def _report_render_progress(stage_progress: int, message: str) -> None:
    report_progress("rendering", 75, stage_progress, message)


def _bounded_focus_x(value: float) -> float:
    if not math.isfinite(value):
        return 0.5
    return max(0.0, min(1.0, value))


def _face_safe_focus_x(
    face_center: float,
    face_width: float,
    crop_width_ratio: float,
) -> float:
    center = _bounded_focus_x(face_center)
    crop_width = max(0.0, min(1.0, crop_width_ratio))
    face_width = max(0.0, min(1.0, face_width))
    if crop_width >= 1.0 or face_width <= 0.0:
        return center

    half_crop = crop_width / 2
    half_face = face_width / 2
    margin = face_width * FACE_CROP_SAFE_MARGIN_RATIO
    lower_bound = max(
        half_crop,
        center + half_face + margin - half_crop,
    )
    upper_bound = min(
        1.0 - half_crop,
        center - half_face - margin + half_crop,
    )
    if lower_bound > upper_bound:
        return max(half_crop, min(1.0 - half_crop, center))
    return max(lower_bound, min(upper_bound, center))


def _fill_face_detection_gaps(
    sample_times: list[float],
    detected_focus: list[float | None],
    fallback_focus: float,
) -> list[float | None]:
    if len(sample_times) != len(detected_focus):
        raise ValueError(
            "I campioni temporali e i punti focali devono avere la stessa lunghezza."
        )

    filled = list(detected_focus)
    previous_time: float | None = None
    previous_focus: float | None = None
    fallback = _bounded_focus_x(fallback_focus)
    for index, (sample_time, focus) in enumerate(zip(sample_times, detected_focus)):
        if focus is not None:
            previous_time = sample_time
            previous_focus = _bounded_focus_x(focus)
        elif previous_time is not None and previous_focus is not None:
            if sample_time - previous_time <= FACE_TRACK_GRACE_PERIOD_SECONDS:
                filled[index] = previous_focus
            else:
                filled[index] = fallback
    return filled


def _sample_clip_times(duration: float) -> list[float]:
    steps = max(
        2,
        min(24, math.ceil(duration / FACE_DETECTION_SAMPLE_INTERVAL_SECONDS)),
    )
    last_sample_time = duration - min(0.01, duration / 2)
    return [last_sample_time * index / steps for index in range(steps + 1)]


def _smooth_focus_track(
    sample_times: list[float],
    detected_focus: list[float | None],
    fallback_focus: float,
    previous_focus: float,
) -> list[tuple[float, float]]:
    if len(sample_times) != len(detected_focus):
        raise ValueError(
            "I campioni temporali e i punti focali devono avere la stessa lunghezza."
        )
    if not sample_times:
        return []

    fallback = _bounded_focus_x(fallback_focus)
    previous = _bounded_focus_x(previous_focus)
    detected_indexes = [
        index for index, value in enumerate(detected_focus) if value is not None
    ]
    interpolated = [fallback] * len(detected_focus)
    if detected_indexes:
        for index in range(detected_indexes[0]):
            fraction = index / max(1, detected_indexes[0])
            first_detected = _bounded_focus_x(
                float(detected_focus[detected_indexes[0]])
            )
            interpolated[index] = previous + (first_detected - previous) * fraction
        for left_index, right_index in zip(
            detected_indexes,
            detected_indexes[1:],
        ):
            left_value = _bounded_focus_x(float(detected_focus[left_index]))
            right_value = _bounded_focus_x(float(detected_focus[right_index]))
            span = right_index - left_index
            for index in range(left_index, right_index + 1):
                fraction = (index - left_index) / span
                interpolated[index] = left_value + (right_value - left_value) * fraction
        last_detected = detected_indexes[-1]
        for index in range(last_detected, len(interpolated)):
            interpolated[index] = _bounded_focus_x(
                float(detected_focus[last_detected])
            )
    else:
        interpolated = [fallback] * len(detected_focus)

    smoothed: list[tuple[float, float]] = []
    current_focus = previous
    for index, (sample_time, target_focus) in enumerate(
        zip(sample_times, interpolated)
    ):
        if index > 0:
            change = (target_focus - current_focus) * FACE_TRACK_SMOOTHING
            change = max(-FACE_TRACK_MAX_STEP, min(FACE_TRACK_MAX_STEP, change))
            current_focus = _bounded_focus_x(current_focus + change)
        smoothed.append((sample_time, current_focus))
    return smoothed


def _focus_track_expression(track: list[tuple[float, float]]) -> str:
    if not track:
        return "0.5"
    if len(track) == 1:
        return f"{_bounded_focus_x(track[0][1]):.6f}"

    expression = f"{_bounded_focus_x(track[-1][1]):.6f}"
    for (start_time, start_focus), (end_time, end_focus) in reversed(
        list(zip(track, track[1:]))
    ):
        span = max(0.001, end_time - start_time)
        interpolated_focus = (
            f"({_bounded_focus_x(start_focus):.6f}+"
            f"({_bounded_focus_x(end_focus) - _bounded_focus_x(start_focus):.6f})*"
            f"(t-{start_time:.6f})/{span:.6f})"
        )
        expression = (
            f"if(lt(t,{end_time:.6f}),{interpolated_focus},{expression})"
        )
    return expression


def _fallback_focus_tracks(
    clips: list[Clip],
    focus_x: float,
) -> list[list[tuple[float, float]]]:
    previous_focus = _bounded_focus_x(max(0.38, min(0.62, focus_x)))
    tracks: list[list[tuple[float, float]]] = []
    for clip in clips:
        duration = max(0.001, clip.end - clip.start)
        sample_times = _sample_clip_times(duration)
        fallback_focus = max(0.38, min(0.62, _bounded_focus_x(clip.focus_x)))
        track = _smooth_focus_track(
            sample_times,
            [None] * len(sample_times),
            fallback_focus,
            previous_focus,
        )
        tracks.append(track)
        previous_focus = track[-1][1]
    return tracks


def _detect_face_focus_tracks(
    video_path: Path,
    clips: list[Clip],
    focus_x: float,
) -> list[list[tuple[float, float]]]:
    fallback_tracks = _fallback_focus_tracks(clips, focus_x)
    if not Path(video_path).is_file():
        LOGGER.warning(
            "Video sorgente non disponibile per il rilevamento dei volti; "
            "uso il crop centrato."
        )
        return fallback_tracks

    try:
        import cv2
    except ImportError:
        LOGGER.warning(
            "OpenCV non disponibile per il rilevamento dei volti; uso il crop centrato."
        )
        return fallback_tracks

    haarcascade_directory = getattr(getattr(cv2, "data", None), "haarcascades", None)
    if not haarcascade_directory:
        LOGGER.warning(
            "Haar cascade non disponibile; uso il crop centrato."
        )
        return fallback_tracks

    cascade_path = Path(haarcascade_directory) / "haarcascade_frontalface_default.xml"
    try:
        detector = cv2.CascadeClassifier(str(cascade_path))
    except cv2.error as error:
        LOGGER.warning("Impossibile inizializzare Haar cascade: %s", error)
        return fallback_tracks
    if detector.empty():
        LOGGER.warning(
            "Haar cascade vuota o non leggibile; uso il crop centrato."
        )
        return fallback_tracks

    try:
        capture = cv2.VideoCapture(str(video_path))
    except cv2.error as error:
        LOGGER.warning("Impossibile aprire il video per il face tracking: %s", error)
        return fallback_tracks
    if not capture.isOpened():
        capture.release()
        LOGGER.warning(
            "OpenCV non riesce ad aprire il video per il face tracking; "
            "uso il crop centrato."
        )
        return fallback_tracks

    tracks: list[list[tuple[float, float]]] = []
    previous_focus = _bounded_focus_x(max(0.38, min(0.62, focus_x)))
    try:
        for clip_index, clip in enumerate(clips):
            duration = max(0.001, clip.end - clip.start)
            sample_times = _sample_clip_times(duration)
            fallback_focus = max(0.38, min(0.62, _bounded_focus_x(clip.focus_x)))
            detected_focus: list[float | None] = []
            previous_detected = previous_focus
            detection_count = 0
            frame_errors = 0
            last_frame_error = ""
            for relative_time in sample_times:
                try:
                    capture.set(
                        cv2.CAP_PROP_POS_MSEC,
                        (clip.start + relative_time) * 1000,
                    )
                    success, frame = capture.read()
                except cv2.error as error:
                    frame_errors += 1
                    last_frame_error = str(error)
                    detected_focus.append(None)
                    continue
                if not success or frame is None:
                    detected_focus.append(None)
                    continue
                try:
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    frame_height, frame_width = gray.shape[:2]
                    if frame_width <= 0 or frame_height <= 0:
                        detected_focus.append(None)
                        continue
                    resize_scale = min(1.0, FACE_DETECTION_MAX_WIDTH / frame_width)
                    if resize_scale < 1.0:
                        gray = cv2.resize(
                            gray,
                            (
                                max(1, round(frame_width * resize_scale)),
                                max(1, round(frame_height * resize_scale)),
                            ),
                        )
                    resized_width = gray.shape[1]
                    faces = detector.detectMultiScale(
                        gray,
                        scaleFactor=1.1,
                        minNeighbors=4,
                        minSize=(24, 24),
                    )
                except cv2.error as error:
                    frame_errors += 1
                    last_frame_error = str(error)
                    detected_focus.append(None)
                    continue

                if len(faces) == 0:
                    detected_focus.append(None)
                    continue
                selected_face = min(
                    faces,
                    key=lambda face: (
                        abs((face[0] + face[2] / 2) / resized_width - previous_detected),
                        -(face[2] * face[3]),
                    ),
                )
                # face_center = _bounded_focus_x(
                #     (selected_face[0] + selected_face[2] / 2) / resized_width
                # )
                crop_width_ratio = min(
                    1.0,
                    frame_height * 9.0 / (16.0 * frame_width),
                )
                face_center = _face_safe_focus_x(
                    (selected_face[0] + selected_face[2] / 2) / resized_width,
                    selected_face[2] / resized_width,
                    crop_width_ratio,
                )
                detected_focus.append(face_center)
                previous_detected = face_center
                detection_count += 1

            detected_focus = _fill_face_detection_gaps(
                sample_times,
                detected_focus,
                fallback_focus,
            )
            track = _smooth_focus_track(
                sample_times,
                detected_focus,
                fallback_focus,
                previous_focus,
            )
            tracks.append(track)
            previous_focus = track[-1][1] if track else fallback_focus
            if frame_errors:
                LOGGER.warning(
                    "Face tracking ha usato il fallback per %d campioni del clip %d; "
                    "ultimo errore: %s",
                    frame_errors,
                    clip_index + 1,
                    last_frame_error,
                )
            elif detection_count == 0:
                LOGGER.info(
                    "Nessun volto rilevato nel clip %d (%.2f-%.2f s); "
                    "uso il crop centrato.",
                    clip_index + 1,
                    clip.start,
                    clip.end,
                )
    finally:
        capture.release()
    return tracks


def render_vertical_short(
    video_path: Path,
    narration_path: Path,
    captions_path: Path,
    output_path: Path,
    clips: list[Clip],
    focus_x: float,
    duration: float,
) -> None:
    if not clips:
        raise ValueError("Nessun segmento video da montare.")

    # face_tracks = _fallback_focus_tracks(clips, focus_x)
    # face_tracks = _detect_face_focus_tracks(video_path, clips, focus_x)

    if not narration_path.is_file() or narration_path.stat().st_size == 0:
        raise RuntimeError(
            f"File della narrazione mancante o vuoto: {narration_path}"
        )
    narration_peak = _measure_audio_peak(narration_path)
    if narration_peak <= AUDIO_SILENCE_THRESHOLD_DB:
        raise RuntimeError(
            "Il WAV della narrazione è silenzioso o quasi muto "
            f"(picco {narration_peak:.1f} dBFS): {narration_path}"
        )
    _report_render_progress(
        5,
        f"Voce da mixare: {narration_path} "
        f"({narration_path.stat().st_size} byte; picco {narration_peak:.1f} dBFS).",
    )

    background_music = _select_background_music()
    if background_music is None:
        _report_render_progress(
            7,
            f"Nessuna musica valida in {MUSIC_DIRECTORY}; il render userà solo la voce.",
        )
    else:
        music_peak = _measure_audio_peak(background_music)
        if music_peak <= AUDIO_SILENCE_THRESHOLD_DB:
            raise RuntimeError(
                "La musica selezionata è silenziosa o quasi muta "
                f"(picco {music_peak:.1f} dBFS): {background_music}"
            )
        _report_render_progress(
            7,
            f"Musica da mixare: {background_music} "
            f"({background_music.stat().st_size} byte; picco {music_peak:.1f} dBFS; "
            "volume mix 10%).",
        )

    face_tracks = _detect_face_focus_tracks(video_path, clips, focus_x)
    filters: list[str] = []
    labels = "".join(f"[source{index}]" for index in range(len(clips)))
    if len(clips) > 1:
        filters.append(f"[0:v]split={len(clips)}{labels}")

    for index, clip in enumerate(clips):
        source = f"[source{index}]" if len(clips) > 1 else "[0:v]"
        # clip_focus = max(0.0, min(1.0, clip.focus_x))
        focus_expression = _focus_track_expression(face_tracks[index])
        filters.append(
            f"{source}trim=start={clip.start:.3f}:duration={clip.end - clip.start:.3f},"
            "setpts=PTS-STARTPTS,"
            "crop=w='min(iw,ih*9/16)':h=ih:"
            # f"x='max(0,min(iw-ow,iw*{clip_focus:.6f}-ow/2))':y=0,"
            f"x='max(0,min(iw-ow,iw*({focus_expression})-ow/2))':y=0,"
            "scale=1080:1920:force_original_aspect_ratio=increase,"
            "crop=1080:1920,setsar=1"
            f"[clip{index}]"
        )

    concatenated = "".join(f"[clip{index}]" for index in range(len(clips)))
    # filters.append(
    #     f"{concatenated}concat=n={len(clips)}:v=1:a=0[video];"
    #     f"[video]subtitles=filename='captions.ass':"
    #     f"fontsdir='{_escape_filter_path(CAPTION_FONT_DIRECTORY)}'[output]"
    # )
    filters.append(
        f"{concatenated}concat=n={len(clips)}:v=1:a=0[video]"
    )
    subtitle_input = "[video]"
    horror_overlay = _horror_overlay_filter_expression()
    if horror_overlay:
        filters.append(f"[video]{horror_overlay}[analog]")
        subtitle_input = "[analog]"
    filters.append(
        f"{subtitle_input}subtitles=filename='captions.ass':"
        f"fontsdir='{_escape_filter_path(CAPTION_FONT_DIRECTORY)}'[output]"
    )
    if background_music is None:
        # filters.append("[1:a]anull[finalaudio]")
        filters.append("[1:a]loudnorm=I=-16:TP=-1.5:LRA=7[finalaudio]")
    else:
        fade_duration = min(1.0, duration)
        fade_start = max(0.0, duration - fade_duration)
        filters.append(
            # "[1:a]volume=1.0[voice];"
            "[1:a]loudnorm=I=-16:TP=-1.5:LRA=7[voice];"
            f"[2:a]atrim=duration={duration:.3f},asetpts=PTS-STARTPTS,"
            f"volume=0.1,afade=t=out:st={fade_start:.3f}:d={fade_duration:.3f}[music];"
            "[voice][music]amix=inputs=2:duration=first:"
            "dropout_transition=0:normalize=0[finalaudio]"
        )
    filter_graph = ";".join(filters)

    # command = [
    #     os.environ.get("FFMPEG_EXECUTABLE", "ffmpeg"),
    #     "-hide_banner",
    #     "-loglevel",
    #     "error",
    #     "-y",
    #     "-i",
    #     str(video_path),
    #     "-i",
    #     str(narration_path),
    #     "-filter_complex",
    #     filter_graph,
    #     "-map",
    #     "[output]",
    #     "-map",
    #     "1:a:0",
    #     "-c:v",
    #     "libx264",
    #     "-preset",
    #     "veryfast",
    #     "-crf",
    #     "22",
    #     "-c:a",
    #     "aac",
    #     "-b:a",
    #     "192k",
    #     "-t",
    #     f"{duration:.3f}",
    #     "-movflags",
    #     "+faststart",
    #     str(output_path),
    # ]
    command = [
        os.environ.get("FFMPEG_EXECUTABLE", "ffmpeg"),
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(video_path),
        "-i",
        str(narration_path),
        *(
            ["-stream_loop", "-1", "-i", str(background_music)]
            if background_music is not None
            else []
        ),
        "-filter_complex",
        filter_graph,
        "-map",
        "[output]",
        "-map",
        "[finalaudio]",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "22",
        "-c:a",
        "aac",
        "-b:a",
        "192k",
        "-t",
        f"{duration:.3f}",
        "-movflags",
        "+faststart",
        str(output_path),
    ]
    for index, clip in enumerate(clips, start=1):
        _report_render_progress(
            min(25, 8 + index),
            f"Clip renderizzata {index}/{len(clips)}: "
            f"{clip.start:.2f}-{clip.end:.2f}s (score {clip.score:.2f}).",
        )
    _report_render_progress(
        26,
        "Mappa audio FFmpeg: input 1 = voce WAV normalizzata a -16 LUFS; "
        + (
            "input 2 = musica con volume al 10%."
            if background_music is not None
            else "nessuna musica, passthrough della voce."
        ),
    )
    subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
        timeout=3600,
        cwd=output_path.parent,
    )
    codec, output_peak = _verify_rendered_audio(output_path)
    _report_render_progress(
        99,
        f"Verifica audio finale superata: codec {codec}, "
        f"picco {output_peak:.1f} dBFS in {output_path}.",
    )
