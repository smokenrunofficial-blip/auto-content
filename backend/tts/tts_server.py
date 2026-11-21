from pathlib import Path
# from typing import Optional
from typing import Any, Optional
import gc
import ipaddress
import json
import logging
import math
import os
import shutil
import subprocess
import tempfile
import threading
import wave
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from pydantic import BaseModel

BASE_DIR = Path(__file__).resolve().parent
# PIPER_DIR = BASE_DIR / "piper"
# PROJECT_ROOT = BASE_DIR.parents[1]
# PIPER_DIR = PROJECT_ROOT / "piper"
# LEGACY_PIPER_DIR = BASE_DIR / "piper"
# if not PIPER_DIR.is_dir() and LEGACY_PIPER_DIR.is_dir():
#     PIPER_DIR = LEGACY_PIPER_DIR
# PIPER_EXECUTABLE = shutil.which("piper")
# PIPER_EXE = Path(PIPER_EXECUTABLE) if PIPER_EXECUTABLE else PIPER_DIR / "piper.exe"


def _resolve_piper_paths(
    base_dir: Path,
    configured_executable: str | None = None,
    configured_directory: str | None = None,
    path_executable: str | None = None,
) -> tuple[list[Path], Path, Path]:
    directories = [base_dir / "piper", base_dir.parent / "piper"]
    if base_dir.parent.name.casefold() == "backend":
        directories.append(base_dir.parent.parent / "piper")
    if configured_directory:
        directories.append(Path(configured_directory).expanduser())
    directories = list(dict.fromkeys(directories))

    executable = next(
        (
            candidate
            for directory in directories
            for candidate in (directory / "piper.exe", directory / "piper")
            if candidate.is_file()
        ),
        None,
    )
    if executable is None and configured_executable:
        executable = Path(configured_executable).expanduser()
    if executable is None and path_executable:
        executable = Path(path_executable)
    if executable is None:
        executable = directories[0] / "piper.exe"

    piper_directory = next(
        (directory for directory in directories if directory.is_dir()),
        executable.parent,
    )
    if executable.parent.is_dir():
        piper_directory = executable.parent
    return directories, piper_directory, executable


PROJECT_ROOT = (
    BASE_DIR.parent.parent
    if BASE_DIR.parent.name.casefold() == "backend"
    else BASE_DIR.parent
)
PIPER_DIR_CANDIDATES, PIPER_DIR, PIPER_EXE = _resolve_piper_paths(
    BASE_DIR,
    configured_executable=os.environ.get("PIPER_EXE"),
    configured_directory=os.environ.get("PIPER_DIR"),
    path_executable=shutil.which("piper"),
)
# DEFAULT_MODEL_NAMES = [
#     "it_IT-alina-medium.onnx",
#     "it_IT-paola-medium.onnx",
#     "it_IT-riccardo-medium.onnx",
# ]
DEFAULT_MODEL_NAMES = [
    "it_IT-riccardo-x_low.onnx",
    "it_IT-alina-medium.onnx",
    "it_IT-paola-medium.onnx",
    "it_IT-riccardo-medium.onnx",
]
# DEFAULT_PIPER_LENGTH_SCALE = 1.18
DEFAULT_PIPER_LENGTH_SCALE = 1.28
# DEFAULT_PIPER_PITCH_SEMITONES = -1.5
DEFAULT_XTTS_MODEL_NAME = "tts_models/multilingual/multi-dataset/xtts_v2"
DEFAULT_XTTS_SPEAKER = "Dionisio Schuyler"
DEFAULT_XTTS_SPEED = 0.92
DEFAULT_XTTS_MIN_FREE_VRAM_MB = 5000
XTTS_SUPPORTED_LANGUAGES = {
    "ar", "cs", "de", "en", "es", "fr", "hi", "hu", "it",
    "ja", "ko", "nl", "pl", "pt", "ru", "tr", "zh-cn",
}
DATA_DIR = BASE_DIR / "data"
SETTINGS_FILE = DATA_DIR / "profile_settings.json"
LOGGER = logging.getLogger(__name__)
_TTS_SYNTHESIS_LOCK = threading.Lock()
_XTTS_MODEL: Any | None = None
_XTTS_MODEL_NAME: str | None = None

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:4200",
        "http://127.0.0.1:4200",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class TTSRequest(BaseModel):
    text: str
    language: Optional[str] = "it"


class ProfilePayload(BaseModel):
    email: str
    first_name: Optional[str] = ""
    last_name: Optional[str] = ""


class SettingsPayload(BaseModel):
    persist_memory: bool = True
    allow_summaries: bool = True
    dark_mode: bool = True
    auto_save: bool = True
    notifications: bool = True


def _configured_piper_length_scale() -> float:
    raw_value = os.environ.get(
        "PIPER_LENGTH_SCALE",
        str(DEFAULT_PIPER_LENGTH_SCALE),
    )
    try:
        length_scale = float(raw_value)
    except ValueError as error:
        raise HTTPException(
            status_code=500,
            detail="PIPER_LENGTH_SCALE deve essere un numero positivo.",
        ) from error
    if not math.isfinite(length_scale) or length_scale <= 0:
        raise HTTPException(
            status_code=500,
            detail="PIPER_LENGTH_SCALE deve essere un numero positivo.",
        )
    return length_scale


# def _configured_piper_pitch_semitones() -> float:
#     raw_value = os.environ.get(
#         "PIPER_PITCH_SEMITONES",
#         str(DEFAULT_PIPER_PITCH_SEMITONES),
#     )
#     try:
#         pitch_semitones = float(raw_value)
#     except ValueError as error:
#         raise HTTPException(
#             status_code=500,
#             detail="PIPER_PITCH_SEMITONES deve essere un numero tra -12 e 12.",
#         ) from error
#     if not math.isfinite(pitch_semitones) or not -12 <= pitch_semitones <= 12:
#         raise HTTPException(
#             status_code=500,
#             detail="PIPER_PITCH_SEMITONES deve essere un numero tra -12 e 12.",
#         )
#     return pitch_semitones


class XTTSUnavailableError(RuntimeError):
    pass


def _configured_flag(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    normalized = value.strip().casefold()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise HTTPException(
        status_code=500,
        detail=f"{name} deve essere un valore booleano.",
    )


def _configured_xtts_speed() -> float:
    raw_value = os.environ.get("XTTS_SPEED", str(DEFAULT_XTTS_SPEED))
    try:
        speed = float(raw_value)
    except ValueError as error:
        raise HTTPException(
            status_code=500,
            detail="XTTS_SPEED deve essere un numero tra 0.5 e 2.",
        ) from error
    if not math.isfinite(speed) or not 0.5 <= speed <= 2:
        raise HTTPException(
            status_code=500,
            detail="XTTS_SPEED deve essere un numero tra 0.5 e 2.",
        )
    return speed


def _configured_min_free_vram_mb() -> int:
    raw_value = os.environ.get(
        "XTTS_MIN_FREE_VRAM_MB",
        str(DEFAULT_XTTS_MIN_FREE_VRAM_MB),
    )
    try:
        minimum = int(raw_value)
    except ValueError as error:
        raise HTTPException(
            status_code=500,
            detail="XTTS_MIN_FREE_VRAM_MB deve essere un intero positivo.",
        ) from error
    if minimum <= 0:
        raise HTTPException(
            status_code=500,
            detail="XTTS_MIN_FREE_VRAM_MB deve essere un intero positivo.",
        )
    return minimum


def _ollama_local_base_url() -> str:
    base_url = os.environ.get(
        "OLLAMA_HOST",
        "http://127.0.0.1:11434",
    ).rstrip("/")
    parsed = urlparse(base_url)
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.hostname
        or parsed.username
        or parsed.password
        or parsed.query
        or parsed.fragment
    ):
        raise RuntimeError("OLLAMA_HOST non è un URL HTTP(S) locale valido.")
    try:
        is_loopback = parsed.hostname.casefold() == "localhost" or ipaddress.ip_address(
            parsed.hostname
        ).is_loopback
    except ValueError:
        is_loopback = False
    if not is_loopback:
        raise RuntimeError("OLLAMA_HOST deve puntare a localhost per lo scaricamento modelli.")
    return base_url


def _unload_pipeline_ollama_models() -> None:
    if not _configured_flag("TTS_UNLOAD_OLLAMA_MODELS", True):
        LOGGER.info("Scaricamento modelli Ollama disattivato da configurazione.")
        return
    try:
        base_url = _ollama_local_base_url()
    except RuntimeError as error:
        LOGGER.warning("Scaricamento Ollama saltato: %s", error)
        return

    model_names = {
        os.environ.get("OLLAMA_VISION_MODEL", "qwen2.5vl:3b").strip(),
        os.environ.get("OLLAMA_TEXT_MODEL", "llama3.1:latest").strip(),
    } - {""}
    try:
        with urlopen(
            Request(f"{base_url}/api/ps", method="GET"),
            timeout=3,
        ) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (HTTPError, URLError, TimeoutError, OSError, json.JSONDecodeError) as error:
        LOGGER.info("Nessun modello Ollama da scaricare o server non raggiungibile: %s", error)
        return

    if not isinstance(payload, dict) or not isinstance(payload.get("models"), list):
        LOGGER.warning("Risposta /api/ps di Ollama non valida; nessun modello scaricato.")
        return
    active_models = payload["models"]
    requested_count = 0
    for active_model in active_models:
        if not isinstance(active_model, dict):
            continue
        model_name = str(
            active_model.get("name") or active_model.get("model") or ""
        ).strip()
        if model_name not in model_names:
            continue
        requested_count += 1
        unload_request = Request(
            f"{base_url}/api/generate",
            data=json.dumps(
                {"model": model_name, "keep_alive": 0}
            ).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urlopen(unload_request, timeout=10) as response:
                response.read(1024)
            LOGGER.info("Modello Ollama %s scaricato dalla VRAM.", model_name)
        except (HTTPError, URLError, TimeoutError, OSError) as error:
            LOGGER.warning(
                "Impossibile scaricare il modello Ollama %s: %s",
                model_name,
                error,
            )
    if requested_count == 0:
        LOGGER.info("Nessun modello Ollama della pipeline attivo da scaricare.")


def _clear_xtts_cuda_cache(torch_module: Any, *, unload_model: bool) -> None:
    global _XTTS_MODEL, _XTTS_MODEL_NAME

    try:
        torch_module.cuda.synchronize()
    except RuntimeError as error:
        LOGGER.warning("Sincronizzazione CUDA durante la pulizia non riuscita: %s", error)

    if unload_model and _XTTS_MODEL is not None:
        synthesizer = getattr(_XTTS_MODEL, "synthesizer", None)
        model = getattr(synthesizer, "tts_model", None)
        if model is not None:
            try:
                model.to("cpu")
            except RuntimeError as error:
                LOGGER.warning("Spostamento XTTS su CPU durante lo scaricamento non riuscito: %s", error)
        _XTTS_MODEL = None
        _XTTS_MODEL_NAME = None

    gc.collect()
    try:
        torch_module.cuda.empty_cache()
    except RuntimeError as error:
        LOGGER.warning("Svuotamento cache CUDA non riuscito: %s", error)
    LOGGER.info(
        "Pulizia VRAM XTTS completata (modello scaricato: %s).",
        unload_model,
    )


def _load_xtts_model(torch_module: Any, model_name: str, device: str) -> Any:
    global _XTTS_MODEL, _XTTS_MODEL_NAME

    if _XTTS_MODEL is not None and _XTTS_MODEL_NAME == model_name:
        return _XTTS_MODEL
    if _XTTS_MODEL is not None:
        _clear_xtts_cuda_cache(torch_module, unload_model=True)

    try:
        from TTS.api import TTS
    except ImportError as error:
        raise XTTSUnavailableError(
            "Il pacchetto coqui-tts non è installato."
        ) from error

    device_index = torch_module.device(device).index
    torch_module.cuda.set_device(device_index if device_index is not None else 0)
    model = TTS(model_name=model_name, progress_bar=False, gpu=True)
    _XTTS_MODEL = model
    _XTTS_MODEL_NAME = model_name
    LOGGER.info("Modello XTTS %s caricato su %s.", model_name, device)
    return model


def _configured_voice_reference() -> Path | None:
    configured_path = os.environ.get("VOICE_REFERENCE_WAV", "").strip()
    if not configured_path:
        return None
    reference_path = Path(configured_path).expanduser().resolve()
    if not reference_path.is_file():
        raise HTTPException(
            status_code=500,
            detail=f"VOICE_REFERENCE_WAV non è un file leggibile: {reference_path}",
        )
    try:
        with wave.open(str(reference_path), "rb") as reference_audio:
            if reference_audio.getframerate() <= 0 or reference_audio.getnframes() <= 0:
                raise HTTPException(
                    status_code=500,
                    detail="VOICE_REFERENCE_WAV è vuoto o non valido.",
                )
    except (OSError, wave.Error) as error:
        raise HTTPException(
            status_code=500,
            detail=f"VOICE_REFERENCE_WAV non è un WAV leggibile: {error}",
        ) from error
    return reference_path


def _validate_wav_file(path: Path, label: str) -> None:
    if not path.is_file() or path.stat().st_size < 12:
        raise HTTPException(status_code=500, detail=f"{label} non ha prodotto un WAV valido.")
    try:
        with wave.open(str(path), "rb") as audio_file:
            if audio_file.getframerate() <= 0 or audio_file.getnframes() <= 0:
                raise HTTPException(
                    status_code=500,
                    detail=f"{label} ha prodotto un WAV vuoto o non valido.",
                )
    except wave.Error as error:
        raise HTTPException(
            status_code=500,
            detail=f"{label} ha prodotto un WAV non leggibile: {error}",
        ) from error


def _apply_audio_filters(source_path: Path) -> bytes:
    filters: list[str] = []
    speed = _configured_xtts_speed()
    if speed != 1.0:
        filters.append(f"atempo={speed:g}")
    extra_filter = os.environ.get("TTS_AUDIO_FILTER", "").strip()
    if extra_filter:
        filters.append(extra_filter)
    if not filters:
        _validate_wav_file(source_path, "Il motore TTS")
        return source_path.read_bytes()

    processed_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temporary_file:
            processed_path = Path(temporary_file.name)
        result = subprocess.run(
            [
                os.environ.get("FFMPEG_EXECUTABLE", "ffmpeg"),
                "-hide_banner",
                "-loglevel",
                "error",
                "-nostdin",
                "-y",
                "-i",
                str(source_path),
                "-af",
                ",".join(filters),
                "-c:a",
                "pcm_s16le",
                str(processed_path),
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        if result.returncode != 0:
            detail = result.stderr.decode("utf-8", errors="replace").strip()
            raise HTTPException(
                status_code=500,
                detail=(
                    "Filtri audio TTS FFmpeg non riusciti: "
                    f"{detail or 'errore non specificato'}"
                ),
            )
        _validate_wav_file(processed_path, "FFmpeg")
        return processed_path.read_bytes()
    finally:
        if processed_path and processed_path.exists():
            processed_path.unlink(missing_ok=True)


def _synthesize_with_xtts(req: TTSRequest, language: str) -> bytes:
    unload_after_synthesis = _configured_flag("XTTS_UNLOAD_AFTER_SYNTHESIS", True)
    try:
        import torch
    except ImportError as error:
        raise XTTSUnavailableError("PyTorch non è installato.") from error
    if not torch.cuda.is_available():
        raise XTTSUnavailableError("CUDA non è disponibile.")

    device = os.environ.get("XTTS_DEVICE", "cuda:0").strip()
    if not device.startswith("cuda"):
        raise XTTSUnavailableError(f"Il dispositivo TTS {device!r} non è CUDA.")
    model_name = os.environ.get("XTTS_MODEL_NAME", DEFAULT_XTTS_MODEL_NAME).strip()
    if not model_name:
        raise HTTPException(status_code=500, detail="XTTS_MODEL_NAME non può essere vuoto.")
    reference_path = _configured_voice_reference()
    speaker = os.environ.get("XTTS_SPEAKER", DEFAULT_XTTS_SPEAKER).strip()
    if reference_path is None and not speaker:
        raise HTTPException(status_code=500, detail="XTTS_SPEAKER non può essere vuoto.")

    _unload_pipeline_ollama_models()
    _clear_xtts_cuda_cache(torch, unload_model=False)
    model_is_cached = _XTTS_MODEL is not None and _XTTS_MODEL_NAME == model_name
    if _XTTS_MODEL is not None and not model_is_cached:
        _clear_xtts_cuda_cache(torch, unload_model=True)
    if not model_is_cached:
        try:
            free_bytes, total_bytes = torch.cuda.mem_get_info(torch.device(device))
        except RuntimeError as error:
            raise XTTSUnavailableError(
                f"Impossibile leggere la VRAM disponibile su {device}: {error}"
            ) from error
        free_mb = free_bytes // (1024 * 1024)
        total_mb = total_bytes // (1024 * 1024)
        LOGGER.info(
            "Dispositivo TTS: %s; VRAM libera %d/%d MB.",
            device,
            free_mb,
            total_mb,
        )
        minimum_free_mb = _configured_min_free_vram_mb()
        if free_mb < minimum_free_mb:
            raise XTTSUnavailableError(
                f"VRAM libera insufficiente ({free_mb} MB; richiesti almeno "
                f"{minimum_free_mb} MB)."
            )
    else:
        LOGGER.info("Dispositivo TTS: %s (modello XTTS già caricato).", device)

    load_error_types: tuple[type[BaseException], ...] = (
        ImportError,
        KeyError,
        OSError,
        RuntimeError,
        ValueError,
        HTTPError,
        URLError,
    )
    try:
        import requests
    except ImportError:
        pass
    else:
        load_error_types += (requests.exceptions.RequestException,)
    try:
        model = _load_xtts_model(torch, model_name, device)
    except XTTSUnavailableError:
        _clear_xtts_cuda_cache(torch, unload_model=True)
        raise
    except load_error_types as error:
        detail = str(error)
        error.__traceback__ = None
        _clear_xtts_cuda_cache(torch, unload_model=True)
        raise XTTSUnavailableError(
            f"Modello XTTS non disponibile su {device}: {detail}"
        ) from None

    speakers = getattr(model, "speakers", None)
    if reference_path is None and speakers and speaker not in speakers:
        _clear_xtts_cuda_cache(torch, unload_model=True)
        raise HTTPException(
            status_code=500,
            detail=(
                f"XTTS_SPEAKER {speaker!r} non esiste nel modello. "
                "Configura una voce valida o imposta VOICE_REFERENCE_WAV."
            ),
        )

    output_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temporary_file:
            output_path = Path(temporary_file.name)
        model.tts_to_file(
            text=req.text.strip(),
            file_path=str(output_path),
            language=language,
            speaker=None if reference_path else speaker,
            speaker_wav=str(reference_path) if reference_path else None,
            split_sentences=True,
        )
        _validate_wav_file(output_path, "XTTS")
        return _apply_audio_filters(output_path)
    except RuntimeError as error:
        if "out of memory" not in str(error).casefold():
            raise
        detail = str(error)
        error.__traceback__ = None
        _clear_xtts_cuda_cache(torch, unload_model=True)
        raise XTTSUnavailableError(
            f"Memoria CUDA insufficiente durante XTTS: {detail}"
        ) from None
    finally:
        if output_path and output_path.exists():
            output_path.unlink(missing_ok=True)
        if unload_after_synthesis:
            _clear_xtts_cuda_cache(torch, unload_model=True)
        else:
            _clear_xtts_cuda_cache(torch, unload_model=False)


def ensure_store_file() -> None:
    DATA_DIR.mkdir(exist_ok=True)
    if not SETTINGS_FILE.exists():
        SETTINGS_FILE.write_text(
            json.dumps({
                "profile": {"email": "", "first_name": "", "last_name": ""},
                "settings": {
                    "persist_memory": True,
                    "allow_summaries": True,
                    "dark_mode": True,
                    "auto_save": True,
                    "notifications": True,
                },
            }, indent=2),
            encoding="utf-8",
        )


def read_store() -> dict:
    ensure_store_file()
    try:
        return json.loads(SETTINGS_FILE.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {
            "profile": {"email": "", "first_name": "", "last_name": ""},
            "settings": {
                "persist_memory": True,
                "allow_summaries": True,
                "dark_mode": True,
                "auto_save": True,
                "notifications": True,
            },
        }


def write_store(data: dict) -> None:
    ensure_store_file()
    SETTINGS_FILE.write_text(json.dumps(data, indent=2), encoding="utf-8")


def resolve_model_path() -> str:
    # locations = [
    #     BASE_DIR,
    #     PIPER_DIR,
    #     BASE_DIR / "models",
    #     BASE_DIR / "voice_models",
    #     PIPER_DIR / "models",
    # ]
    locations = [
        BASE_DIR,
        *PIPER_DIR_CANDIDATES,
        BASE_DIR / "models",
        BASE_DIR / "voice_models",
        BASE_DIR.parent / "models",
        PROJECT_ROOT / "models",
        *(directory / "models" for directory in PIPER_DIR_CANDIDATES),
    ]

    for location in locations:
        for name in DEFAULT_MODEL_NAMES:
            candidate = location / name
            if candidate.exists():
                return str(candidate)

    for location in locations:
        if location.exists():
            for match in sorted(location.glob("*.onnx")):
                name_lower = match.name.lower()
                if "it" in name_lower and any(token in name_lower for token in ("alina", "paola", "riccardo")):
                    return str(match)

    raise FileNotFoundError(
        f"Nessun modello italiano .onnx trovato. Controlla che esista nella cartella piper."
    )


@app.get("/api/profile")
def get_profile(email: str = ""):
    store = read_store()
    profile = store.get("profile", {})

    if email:
        profile = {**profile, "email": email}
        if "email" in profile and profile["email"] == "":
            profile["email"] = email

    return {
        "success": True,
        "user": {
            "email": profile.get("email", email or ""),
            "first_name": profile.get("first_name", ""),
            "last_name": profile.get("last_name", ""),
        },
    }


@app.post("/api/profile")
def save_profile(payload: ProfilePayload):
    store = read_store()
    store["profile"] = {
        "email": payload.email or "",
        "first_name": payload.first_name or "",
        "last_name": payload.last_name or "",
    }
    write_store(store)
    return {"success": True, "message": "Profilo salvato."}


@app.get("/api/settings")
def get_settings():
    store = read_store()
    settings = store.get("settings", {
        "persist_memory": True,
        "allow_summaries": True,
        "dark_mode": True,
        "auto_save": True,
        "notifications": True,
    })
    return {"success": True, "settings": settings}


@app.post("/api/settings")
def save_settings(payload: SettingsPayload):
    store = read_store()
    store["settings"] = {
        "persist_memory": bool(payload.persist_memory),
        "allow_summaries": bool(payload.allow_summaries),
        "dark_mode": bool(payload.dark_mode),
        "auto_save": bool(payload.auto_save),
        "notifications": bool(payload.notifications),
    }
    write_store(store)
    return {"success": True, "message": "Impostazioni salvate."}


# @app.post("/api/tts")
# def synth(req: TTSRequest):
def _synthesize_with_piper(req: TTSRequest):
    if not req.text or not req.text.strip():
        raise HTTPException(status_code=400, detail="Testo mancante")

    if not PIPER_EXE.exists():
        raise HTTPException(status_code=500, detail=f"Binario Piper non trovato: {PIPER_EXE}")

    try:
        model_path = resolve_model_path()
        espeak_data = PIPER_DIR / "espeak-ng-data"
        length_scale = _configured_piper_length_scale()
        # pitch_semitones = _configured_piper_pitch_semitones()

        temp_file = None
        # processed_file = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                temp_file = Path(tmp.name)

            command = [
                str(PIPER_EXE),
                "--model",
                str(model_path),
                "--output_file",
                str(temp_file),
            ]
            command.extend(["--length_scale", f"{length_scale:g}"])

            if os.name == "nt" and espeak_data.exists():
                command.extend(["--espeak_data", str(espeak_data)])

            result = subprocess.run(
                command,
                input=req.text.strip().encode("utf-8"),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )

            if result.returncode != 0:
                detail = result.stderr.decode("utf-8", errors="ignore").strip() or "Errore sconosciuto di Piper"
                raise HTTPException(status_code=500, detail=f"Errore Piper: {detail}")

            if not temp_file.exists() or temp_file.stat().st_size < 12:
                raise HTTPException(status_code=500, detail="Piper non ha prodotto un file WAV valido")

            # wav_bytes = temp_file.read_bytes()
            piper_wav_bytes = temp_file.read_bytes()
            # if wav_bytes[:4] != b"RIFF" or wav_bytes[8:12] != b"WAVE":
            if piper_wav_bytes[:4] != b"RIFF" or piper_wav_bytes[8:12] != b"WAVE":
                raise HTTPException(
                    status_code=500,
                    # detail=f"Output non valido: file generato non è un WAV corretto. Anteprima: {wav_bytes[:32]!r}",
                    detail=(
                        "Output non valido: Piper non ha generato un WAV corretto. "
                        f"Anteprima: {piper_wav_bytes[:32]!r}"
                    ),
                )

            # with wave.open(str(temp_file), "rb") as piper_wav:
            #     sample_rate = piper_wav.getframerate()
            # if sample_rate <= 0:
            #     raise HTTPException(
            #         status_code=500,
            #         detail="La frequenza di campionamento del WAV Piper non è valida.",
            #     )
            # pitch_ratio = 2 ** (pitch_semitones / 12)
            # shifted_sample_rate = max(1, round(sample_rate * pitch_ratio))
            # tempo_compensation = sample_rate / shifted_sample_rate
            # audio_filter = (
            #     f"asetrate={shifted_sample_rate},aresample={sample_rate},"
            #     f"atempo={tempo_compensation:.8f},bass=g=3:f=120"
            # )
            # with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            #     processed_file = Path(tmp.name)
            # ffmpeg_result = subprocess.run(
            #     [
            #         os.environ.get("FFMPEG_EXECUTABLE", "ffmpeg"),
            #         "-hide_banner",
            #         "-loglevel",
            #         "error",
            #         "-y",
            #         "-i",
            #         str(temp_file),
            #         "-af",
            #         audio_filter,
            #         "-c:a",
            #         "pcm_s16le",
            #         str(processed_file),
            #     ],
            #     stdout=subprocess.PIPE,
            #     stderr=subprocess.PIPE,
            #     check=False,
            # )
            # if ffmpeg_result.returncode != 0:
            #     detail = (
            #         ffmpeg_result.stderr.decode("utf-8", errors="replace").strip()
            #         or "errore FFmpeg non specificato"
            #     )
            #     raise HTTPException(
            #         status_code=500,
            #         detail=f"Post-processing vocale FFmpeg fallito: {detail}",
            #     )
            # if not processed_file.exists() or processed_file.stat().st_size < 12:
            #     raise HTTPException(
            #         status_code=500,
            #         detail="FFmpeg non ha prodotto un WAV vocale valido.",
            #     )
            # processed_wav_bytes = processed_file.read_bytes()
            # if (
            #     processed_wav_bytes[:4] != b"RIFF"
            #     or processed_wav_bytes[8:12] != b"WAVE"
            # ):
            #     raise HTTPException(
            #         status_code=500,
            #         detail="FFmpeg ha prodotto un file vocale non valido.",
            #     )
            # return Response(content=wav_bytes, media_type="audio/wav")
            LOGGER.warning("Dispositivo TTS selezionato: CPU (fallback Piper).")
            return Response(
                content=_apply_audio_filters(temp_file),
                media_type="audio/wav",
            )
        finally:
            # if processed_file and processed_file.exists():
            #     processed_file.unlink(missing_ok=True)
            if temp_file and temp_file.exists():
                temp_file.unlink(missing_ok=True)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except Exception as exc:
        if isinstance(exc, HTTPException):
            raise exc
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/api/tts")
def synth(req: TTSRequest):
    if not req.text or not req.text.strip():
        raise HTTPException(status_code=400, detail="Testo mancante")
    language = (req.language or "it").strip().casefold()
    if language not in XTTS_SUPPORTED_LANGUAGES:
        raise HTTPException(
            status_code=400,
            detail=f"Lingua TTS non supportata da XTTS: {language}.",
        )
    _configured_xtts_speed()

    with _TTS_SYNTHESIS_LOCK:
        fallback_reason: str | None = None
        try:
            audio = _synthesize_with_xtts(req, language)
        except XTTSUnavailableError as error:
            fallback_reason = str(error)
        if fallback_reason is not None:
            LOGGER.warning(
                "XTTS non disponibile (%s); uso Piper come fallback.",
                fallback_reason,
            )
            if language != "it":
                raise HTTPException(
                    status_code=503,
                    detail=(
                        f"XTTS non disponibile e Piper è configurato solo per l'italiano "
                        f"(lingua richiesta: {language})."
                    ),
                )
            if os.environ.get("VOICE_REFERENCE_WAV", "").strip():
                LOGGER.warning("Piper non supporta VOICE_REFERENCE_WAV; uso la voce predefinita.")
            return _synthesize_with_piper(req)
        return Response(content=audio, media_type="audio/wav")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)