import json
import gc
import logging
import os
import sys
from contextlib import redirect_stdout
from datetime import datetime
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
from typing import Any

from .events import report_progress


# ELEVENLABS_REQUEST_TIMEOUT_SECONDS = 120
# LOCAL_TTS_TIMEOUT_SECONDS = 120
LOCAL_TTS_TIMEOUT_SECONDS = 600
LOGGER = logging.getLogger(__name__)


def _clear_cuda_memory() -> bool:
    try:
        import torch
    except ImportError:
        return False
    if not torch.cuda.is_available():
        return False

    torch.cuda.synchronize()
    gc.collect()
    torch.cuda.empty_cache()
    LOGGER.info("Memoria cache CUDA di Whisper rilasciata.")
    return True


def release_whisper_model(model: Any) -> str:
    original_device = str(getattr(model, "device", "cpu"))
    if not original_device.startswith("cuda"):
        LOGGER.warning(
            "VRAM: Whisper usa %s; non è necessario liberare memoria GPU.",
            original_device,
        )
        return original_device

    model.to("cpu")
    _clear_cuda_memory()
    LOGGER.warning("VRAM: Whisper spostato su CPU prima della sintesi TTS.")
    return original_device


def restore_whisper_model(model: Any, device: str) -> str:
    if not device.startswith("cuda"):
        return device
    try:
        import torch
    except ImportError:
        LOGGER.warning("PyTorch non è disponibile; Whisper resta su CPU.")
        return "cpu"
    if not torch.cuda.is_available():
        LOGGER.warning("CUDA non è disponibile; Whisper resta su CPU.")
        return "cpu"

    try:
        model.to(device)
    except RuntimeError as error:
        if "out of memory" not in str(error).casefold():
            raise
        model.to("cpu")
        _clear_cuda_memory()
        LOGGER.warning("VRAM insufficiente per Whisper; il modello resta su CPU.")
        return "cpu"
    LOGGER.warning("VRAM: Whisper ripristinato su %s.", device)
    return device


def _report_synthesis_progress(stage_progress: int, message: str) -> None:
    timestamp = datetime.now().astimezone().isoformat(timespec="milliseconds")
    with redirect_stdout(sys.stderr):
        report_progress(
            "elevenlabs_synthesis",
            47,
            stage_progress,
            f"[{timestamp}] {message}",
        )


def load_whisper_model() -> Any:
    try:
        import whisper
    except ImportError as error:
        raise RuntimeError(
            "Whisper non è installato. Esegui: python -m pip install -r requirements-video.txt"
        ) from error

    return whisper.load_model(os.environ.get("WHISPER_MODEL", "base"))


def transcribe(model: Any, media_path: Path, language: str, word_timestamps: bool) -> dict[str, Any]:
    return model.transcribe(
        str(media_path),
        language=language,
        word_timestamps=word_timestamps,
        verbose=False,
    )


# def synthesize_narration(text: str, destination: Path) -> None:
#     # report_progress("elevenlabs_synthesis", 47, 1, "Lettura configurazione")
#     _report_synthesis_progress(1, "Lettura variabili ElevenLabs dall'ambiente del processo.")
#     # api_key = os.environ.get("ELEVENLABS_API_KEY")
#     # voice_id = os.environ.get("ELEVENLABS_VOICE_ID")
#     api_key_raw = os.environ.get("ELEVENLABS_API_KEY", "")
#     voice_id_raw = os.environ.get("ELEVENLABS_VOICE_ID", "")
#     api_key = api_key_raw.strip()
#     voice_id = voice_id_raw.strip()
#     if not api_key or not voice_id:
#         raise RuntimeError("Configura ELEVENLABS_API_KEY e ELEVENLABS_VOICE_ID nell'ambiente.")
#     if any(character.isspace() for character in api_key):
#         raise RuntimeError("ELEVENLABS_API_KEY contiene spazi o a capo interni.")
#     if any(character.isspace() for character in voice_id):
#         raise RuntimeError("ELEVENLABS_VOICE_ID contiene spazi o a capo interni.")
#
#     whitespace_removed = api_key != api_key_raw or voice_id != voice_id_raw
#     # report_progress("elevenlabs_synthesis", 47, 2, "Configurazione letta")
#     _report_synthesis_progress(
#         2,
#         "Configurazione letta: API key=***; "
#         f"voice_id={voice_id}; spazi esterni rimossi={'sì' if whitespace_removed else 'no'}.",
#     )
#     if len(text) > 5000:
#         raise ValueError("Lo script supera il limite iniziale di 5000 caratteri.")
#
#     url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}?output_format=mp3_44100_128"
#     request = Request(
#         # f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}?output_format=mp3_44100_128",
#         url,
#         data=json.dumps(
#             {
#                 "text": text,
#                 "model_id": "eleven_multilingual_v2",
#                 "voice_settings": {"stability": 0.45, "similarity_boost": 0.75},
#             }
#         ).encode("utf-8"),
#         headers={"xi-api-key": api_key, "Content-Type": "application/json"},
#         method="POST",
#     )
#
#     # report_progress("elevenlabs_synthesis", 47, 3, "Invio POST")
#     _report_synthesis_progress(
#         3,
#         f"Invio POST {url} (voice_id={voice_id}; API key=***; "
#         f"timeout={ELEVENLABS_REQUEST_TIMEOUT_SECONDS}s).",
#     )
#     try:
#         # with urlopen(request, timeout=120) as response:
#         with urlopen(request, timeout=ELEVENLABS_REQUEST_TIMEOUT_SECONDS) as response:
#             response_headers = getattr(response, "headers", {})
#             # report_progress("elevenlabs_synthesis", 47, 4, "Header risposta ricevuti")
#             _report_synthesis_progress(
#                 4,
#                 "Header risposta ElevenLabs ricevuti: "
#                 f"HTTP {getattr(response, 'status', 'non indicato')}, "
#                 f"Content-Type={response_headers.get('Content-Type', 'non indicato')}, "
#                 f"Content-Length={response_headers.get('Content-Length', 'non indicata')}.",
#             )
#             destination.write_bytes(response.read())
#     except HTTPError as error:
#         # report_progress("elevenlabs_synthesis", 47, 4, "Header errore ricevuti")
#         _report_synthesis_progress(
#             4,
#             "Header di errore ElevenLabs ricevuti: "
#             f"HTTP {error.code}, "
#             f"Content-Type={error.headers.get('Content-Type', 'non indicato') if error.headers else 'non indicato'}.",
#         )
#         # detail = error.read(2000).decode("utf-8", errors="replace")
#         detail = error.read().decode("utf-8", errors="replace")
#         raise RuntimeError(f"ElevenLabs ha restituito HTTP {error.code}: {detail}") from error
#     except TimeoutError as error:
#         message = (
#             "Timeout della richiesta ElevenLabs "
#             f"({ELEVENLABS_REQUEST_TIMEOUT_SECONDS}s): {error}"
#         )
#         # report_progress("elevenlabs_synthesis", 47, 4, message)
#         _report_synthesis_progress(4, message)
#         raise RuntimeError(message) from error
#     except URLError as error:
#         message = f"Connessione a ElevenLabs non riuscita: {error.reason}"
#         if isinstance(error.reason, TimeoutError):
#             message = (
#                 "Timeout della richiesta ElevenLabs "
#                 f"({ELEVENLABS_REQUEST_TIMEOUT_SECONDS}s): {error.reason}"
#             )
#         # report_progress("elevenlabs_synthesis", 47, 4, message)
#         _report_synthesis_progress(4, message)
#         raise RuntimeError(message) from error


def synthesize_narration(
    text: str,
    destination: Path,
    language: str = "it",
) -> None:
    if len(text) > 5000:
        raise ValueError("Lo script supera il limite iniziale di 5000 caratteri.")

    url = os.environ.get(
        "LOCAL_TTS_URL",
        "http://127.0.0.1:8000/api/tts",
    ).strip()
    if not url:
        raise RuntimeError("LOCAL_TTS_URL non può essere vuoto.")
    request = Request(
        url,
        data=json.dumps(
            {"text": text, "language": language},
            ensure_ascii=False,
        ).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    _report_synthesis_progress(
        1,
        f"Invio della narrazione al server TTS locale {url}.",
    )
    try:
        with urlopen(request, timeout=LOCAL_TTS_TIMEOUT_SECONDS) as response:
            audio = response.read()
            content_type = response.headers.get("Content-Type", "")
            if not audio.startswith(b"RIFF") or audio[8:12] != b"WAVE":
                raise RuntimeError(
                    "Il server TTS locale non ha restituito un WAV valido "
                    f"(Content-Type: {content_type or 'non indicato'})."
                )
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(audio)
    except HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace").strip()
        raise RuntimeError(
            f"Il server TTS locale ha restituito HTTP {error.code}: "
            f"{detail or error.reason}"
        ) from error
    except TimeoutError as error:
        raise RuntimeError(
            f"Timeout del server TTS locale ({LOCAL_TTS_TIMEOUT_SECONDS}s): {error}"
        ) from error
    except URLError as error:
        if isinstance(error.reason, TimeoutError):
            message = (
                f"Timeout del server TTS locale "
                f"({LOCAL_TTS_TIMEOUT_SECONDS}s): {error.reason}"
            )
        else:
            message = (
                "Server TTS locale non raggiungibile. Avvia backend/tts/tts_server.py "
                f"su 127.0.0.1:8000. Dettaglio: {error.reason}"
            )
        raise RuntimeError(message) from error
