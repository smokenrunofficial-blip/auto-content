from pathlib import Path
# from typing import Optional
from typing import Any, Optional
import gc
import ipaddress
import json
import logging
import math
import os
import re
import shutil
import struct
import subprocess
import tempfile
import threading
import unicodedata
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
DEFAULT_XTTS_TEMPERATURE = 0.65
DEFAULT_XTTS_TOP_K = 50
DEFAULT_XTTS_TOP_P = 0.85
DEFAULT_XTTS_REPETITION_PENALTY = 2.0
DEFAULT_XTTS_MIN_FREE_VRAM_MB = 5000
ITALIAN_PRONUNCIATION_OVERRIDES = {
    "qual'è": "qual è",
    "un po'": "un poco",
    "un pò": "un poco",
    "un po’": "un poco",
}
ITALIAN_ABBREVIATIONS = {
    "p.es.": "per esempio",
    "dott.": "dottore",
    "prof.": "professore",
    "sig.ra": "signora",
    "sig.": "signore",
    "es.": "esempio",
    "ecc.": "eccetera.",
    "etc.": "eccetera.",
}
ITALIAN_ACRONYMS = {
    "AI": "intelligenza artificiale",
    "DNA": "di enne a",
    "GPS": "gi pi esse",
    "IA": "intelligenza artificiale",
    "SMS": "esse emme esse",
    "TV": "tivù",
    "UE": "Unione Europea",
    "USA": "Stati Uniti",
}
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


def _configured_xtts_float(
    name: str,
    default: float,
    minimum: float,
    maximum: float,
) -> float:
    raw_value = os.environ.get(name, str(default))
    try:
        value = float(raw_value)
    except ValueError as error:
        raise HTTPException(
            status_code=500,
            detail=f"{name} deve essere un numero tra {minimum} e {maximum}.",
        ) from error
    if not math.isfinite(value) or not minimum <= value <= maximum:
        raise HTTPException(
            status_code=500,
            detail=f"{name} deve essere un numero tra {minimum} e {maximum}.",
        )
    return value


def _configured_xtts_sampling_options() -> dict[str, float | int]:
    try:
        top_k = int(os.environ.get("XTTS_TOP_K", str(DEFAULT_XTTS_TOP_K)))
    except ValueError as error:
        raise HTTPException(
            status_code=500,
            detail="XTTS_TOP_K deve essere un intero tra 1 e 200.",
        ) from error
    if not 1 <= top_k <= 200:
        raise HTTPException(
            status_code=500,
            detail="XTTS_TOP_K deve essere un intero tra 1 e 200.",
        )
    return {
        "temperature": _configured_xtts_float(
            "XTTS_TEMPERATURE",
            DEFAULT_XTTS_TEMPERATURE,
            0.01,
            2.0,
        ),
        "top_k": top_k,
        "top_p": _configured_xtts_float(
            "XTTS_TOP_P",
            DEFAULT_XTTS_TOP_P,
            0.01,
            1.0,
        ),
        "repetition_penalty": _configured_xtts_float(
            "XTTS_REPETITION_PENALTY",
            DEFAULT_XTTS_REPETITION_PENALTY,
            1.0,
            20.0,
        ),
    }


_ITALIAN_SMALL_NUMBERS = (
    "zero",
    "uno",
    "due",
    "tre",
    "quattro",
    "cinque",
    "sei",
    "sette",
    "otto",
    "nove",
    "dieci",
    "undici",
    "dodici",
    "tredici",
    "quattordici",
    "quindici",
    "sedici",
    "diciassette",
    "diciotto",
    "diciannove",
)
_ITALIAN_TENS = (
    "",
    "",
    "venti",
    "trenta",
    "quaranta",
    "cinquanta",
    "sessanta",
    "settanta",
    "ottanta",
    "novanta",
)


def _italian_under_thousand(number: int) -> str:
    if number < 20:
        return _ITALIAN_SMALL_NUMBERS[number]
    if number < 100:
        tens, unit = divmod(number, 10)
        prefix = _ITALIAN_TENS[tens]
        if unit in {1, 8}:
            prefix = prefix[:-1]
        return prefix + (_ITALIAN_SMALL_NUMBERS[unit] if unit else "")

    hundreds, remainder = divmod(number, 100)
    prefix = "cento" if hundreds == 1 else f"{_ITALIAN_SMALL_NUMBERS[hundreds]}cento"
    if remainder == 8 or 80 <= remainder < 90:
        prefix = prefix[:-1]
    return prefix + (_italian_under_thousand(remainder) if remainder else "")


def _italian_integer_to_words(number: int) -> str:
    if number < 0:
        return f"meno {_italian_integer_to_words(-number)}"
    if number < 1000:
        return _italian_under_thousand(number)
    if number < 1_000_000:
        thousands, remainder = divmod(number, 1000)
        prefix = "mille" if thousands == 1 else f"{_italian_under_thousand(thousands)}mila"
        return prefix + (_italian_under_thousand(remainder) if remainder else "")
    if number < 1_000_000_000:
        millions, remainder = divmod(number, 1_000_000)
        noun = "milione" if millions == 1 else "milioni"
        prefix = "un milione" if millions == 1 else f"{_italian_integer_to_words(millions)} {noun}"
        if remainder:
            prefix += f" {_italian_integer_to_words(remainder)}"
        return prefix
    if number < 1_000_000_000_000:
        billions, remainder = divmod(number, 1_000_000_000)
        noun = "miliardo" if billions == 1 else "miliardi"
        prefix = "un miliardo" if billions == 1 else f"{_italian_integer_to_words(billions)} {noun}"
        if remainder:
            prefix += f" {_italian_integer_to_words(remainder)}"
        return prefix
    return " ".join(_ITALIAN_SMALL_NUMBERS[int(digit)] for digit in str(number))


def _italian_number_to_words(value: str) -> str:
    value = value.strip()
    separator = "," if "," in value else "."
    if separator in value:
        integer_part, fractional_part = value.rsplit(separator, 1)
        if separator == "." and len(fractional_part) == 3:
            value = integer_part + fractional_part
        elif fractional_part.isdigit():
            integer_value = int(integer_part.replace(".", "") or "0")
            spoken_fraction = " ".join(
                _ITALIAN_SMALL_NUMBERS[int(digit)] for digit in fractional_part
            )
            return f"{_italian_integer_to_words(integer_value)} virgola {spoken_fraction}"
    return _italian_integer_to_words(int(value.replace(".", "").replace(",", "")))


def _configured_italian_pronunciation_overrides() -> dict[str, str]:
    overrides = dict(ITALIAN_PRONUNCIATION_OVERRIDES)
    raw_value = os.environ.get("XTTS_ITALIAN_PRONUNCIATION_OVERRIDES", "").strip()
    if not raw_value:
        return overrides
    try:
        configured = json.loads(raw_value)
    except json.JSONDecodeError as error:
        raise HTTPException(
            status_code=500,
            detail="XTTS_ITALIAN_PRONUNCIATION_OVERRIDES deve essere un oggetto JSON.",
        ) from error
    if not isinstance(configured, dict) or any(
        not isinstance(source, str)
        or not source.strip()
        or not isinstance(replacement, str)
        or not replacement.strip()
        for source, replacement in configured.items()
    ):
        raise HTTPException(
            status_code=500,
            detail="Gli override XTTS devono mappare parole non vuote in stringhe.",
        )
    overrides.update(configured)
    return overrides


def prepare_italian_tts_text(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    text = re.sub(r"(?<!\w)n(?:\.|º|°)\s*(?=\d)", "numero ", text, flags=re.IGNORECASE)

    def expand_time(match: re.Match[str]) -> str:
        original_prefix = match.group(1) or ""
        prefix = original_prefix.casefold()
        hour = int(match.group(2))
        minute = int(match.group(3))
        if hour == 0 and minute == 0:
            return "mezzanotte"
        if hour == 0:
            return f"mezzanotte e {_italian_integer_to_words(minute)}"
        hour_words = "una" if hour == 1 else _italian_integer_to_words(hour)
        if minute == 0:
            time_words = f"{hour_words} in punto"
        else:
            time_words = (
                f"{hour_words} e {_italian_integer_to_words(minute)}"
            )
        if hour == 1:
            if prefix in {"alle", "all'"}:
                result = f"all'{time_words}"
            elif prefix == "le" or not prefix:
                result = f"l'{time_words}"
            else:
                result = f"{original_prefix} {time_words}"
            if match.start() == 0 or original_prefix[:1].isupper():
                result = result[0].upper() + result[1:]
            return result
        if prefix:
            return f"{original_prefix} {time_words}"
        prefix_text = "Le" if match.start() == 0 else "le"
        return f"{prefix_text} {time_words}"

    text = re.sub(
        r"(?<!\w)(?:(alle|all'|le|ore)\s+)?(\d{1,2}):([0-5]\d)(?!\w)",
        expand_time,
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        r"(?<!\w)(\d+(?:[.,]\d+)*)\s*%",
        lambda match: f"{_italian_number_to_words(match.group(1))} per cento",
        text,
    )
    text = re.sub(
        r"(?<!\w)(\d+(?:[.,]\d+)*)\s*€",
        lambda match: f"{_italian_number_to_words(match.group(1))} euro",
        text,
    )
    for abbreviation, replacement in sorted(
        ITALIAN_ABBREVIATIONS.items(),
        key=lambda item: len(item[0]),
        reverse=True,
    ):
        text = re.sub(
            rf"(?<!\w){re.escape(abbreviation)}(?=\s|[,!?;:]|$)",
            replacement,
            text,
            flags=re.IGNORECASE,
        )
    for acronym, replacement in sorted(
        ITALIAN_ACRONYMS.items(),
        key=lambda item: len(item[0]),
        reverse=True,
    ):
        text = re.sub(
            rf"(?<!\w){re.escape(acronym)}(?!\w)",
            replacement,
            text,
        )
    text = re.sub(
        r"(?<!\w)\d+(?:[.,]\d+)*(?!\w)",
        lambda match: _italian_number_to_words(match.group(0)),
        text,
    )
    for source, replacement in sorted(
        _configured_italian_pronunciation_overrides().items(),
        key=lambda item: len(item[0]),
        reverse=True,
    ):
        text = re.sub(
            rf"(?<!\w){re.escape(source)}(?!\w)",
            replacement,
            text,
            flags=re.IGNORECASE,
        )

    text = re.sub(r"[-‐‑–—]{2,}", " — ", text)
    text = re.sub(r'["“”]{2,}', '"', text)
    text = "".join(
        character
        for character in text
        if character.isspace()
        or unicodedata.category(character)[0] in {"L", "M", "N"}
        or character in ".,!?;:…'’‘\"“”«»—–-()[]"
    )
    text = re.sub(r"\s+([,.;:!?…])", r"\1", text)
    text = re.sub(r"([,;:!?…])(?=\S)", r"\1 ", text)
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        raise HTTPException(status_code=400, detail="Testo italiano vuoto dopo la normalizzazione.")
    closing_quotes = "\"'’”»)]"
    body = text.rstrip(closing_quotes)
    if not body.endswith((".", "!", "?", "…")):
        suffix = text[len(body) :]
        body = body.rstrip(",;:—–- ") + "."
        # text = body + text[len(body.rstrip(".")) :]
        text = body + suffix
    return text


def _split_sentences_for_tts(text: str) -> list[str]:
    sentences: list[str] = []
    start = 0
    index = 0
    closing_quotes = "\"'’”»)]"
    while index < len(text):
        if text[index] not in ".!?…:;":
            index += 1
            continue
        end = index + 1
        while end < len(text) and text[end] in ".!?…:;":
            end += 1
        while end < len(text) and text[end] in closing_quotes:
            end += 1
        if end == len(text) or text[end].isspace():
            sentence = text[start:end].strip()
            if sentence:
                sentences.append(sentence)
            while end < len(text) and text[end].isspace():
                end += 1
            start = end
            index = end
        else:
            index = end
    if start < len(text):
        remainder = text[start:].strip()
        if remainder:
            sentences.append(remainder)
    return sentences


def _split_long_tts_sentence(sentence: str, max_words: int = 28) -> list[str]:
    if len(sentence.split()) <= max_words:
        return [sentence]
    clauses = re.split(r"(?<=,)\s+", sentence)
    chunks: list[str] = []
    current: list[str] = []
    for clause in clauses:
        words = clause.split()
        if current and len(current) + min(len(words), max_words) > max_words:
            chunks.append(" ".join(current))
            current = []
        while len(words) > max_words:
            piece = words[:max_words]
            if not re.search(r"[,;:]$", piece[-1]):
                piece[-1] += ","
            chunks.append(" ".join(piece))
            words = words[max_words:]
        current.extend(words)
    if current:
        chunks.append(" ".join(current))
    return chunks


def split_italian_tts_text(text: str, max_words: int = 28) -> list[str]:
    prepared = prepare_italian_tts_text(text)
    return [
        chunk
        for sentence in _split_sentences_for_tts(prepared)
        for chunk in _split_long_tts_sentence(sentence, max_words)
    ]


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
            frame_rate = reference_audio.getframerate()
            frame_count = reference_audio.getnframes()
            channel_count = reference_audio.getnchannels()
            sample_width = reference_audio.getsampwidth()
            if frame_rate <= 0 or frame_count <= 0:
                raise HTTPException(
                    status_code=500,
                    detail="VOICE_REFERENCE_WAV è vuoto o non valido.",
                )
            duration_seconds = frame_count / frame_rate
            if not 1.0 <= duration_seconds <= 30.0:
                raise HTTPException(
                    status_code=500,
                    detail="VOICE_REFERENCE_WAV deve durare tra 1 e 30 secondi.",
                )
            if not 16_000 <= frame_rate <= 96_000:
                raise HTTPException(
                    status_code=500,
                    detail="VOICE_REFERENCE_WAV deve avere una frequenza tra 16 e 96 kHz.",
                )
            if channel_count not in {1, 2} or sample_width not in {1, 2, 3, 4}:
                raise HTTPException(
                    status_code=500,
                    detail="VOICE_REFERENCE_WAV deve essere PCM mono o stereo valido.",
                )
            sample_peak, sample_rms, clipped_ratio = _voice_reference_levels(
                reference_audio,
                frame_count,
                channel_count,
                sample_width,
            )
            if sample_rms < 0.001:
                raise HTTPException(
                    status_code=500,
                    detail="VOICE_REFERENCE_WAV è quasi silenzioso o troppo debole.",
                )
            if clipped_ratio > 0.005:
                raise HTTPException(
                    status_code=500,
                    detail="VOICE_REFERENCE_WAV è distorto o presenta clipping eccessivo.",
                )
            LOGGER.info(
                "VOICE_REFERENCE_WAV verificato: durata %.2fs, frequenza %d Hz, "
                "%d canali, %d byte/campione, picco %.3f, RMS %.3f, clipping %.2f%%. "
                "La presenza di musica o riverbero marcato richiede ascolto manuale.",
                duration_seconds,
                frame_rate,
                channel_count,
                sample_width,
                sample_peak,
                sample_rms,
                clipped_ratio * 100,
            )
    except (OSError, wave.Error) as error:
        raise HTTPException(
            status_code=500,
            detail=f"VOICE_REFERENCE_WAV non è un WAV leggibile: {error}",
        ) from error
    return reference_path


def _voice_reference_levels(
    audio_file: wave.Wave_read,
    frame_count: int,
    channel_count: int,
    sample_width: int,
) -> tuple[float, float, float]:
    maximum_sample = (1 << (sample_width * 8 - 1)) - 1
    sample_count = 0
    clipped_count = 0
    squared_sum = 0.0
    peak = 0
    audio_file.rewind()
    while raw_frames := audio_file.readframes(8192):
        if sample_width == 1:
            samples = (value - 128 for value in raw_frames)
            maximum_sample = 128
        elif sample_width == 2:
            samples = (value[0] for value in struct.iter_unpack("<h", raw_frames))
        elif sample_width == 3:
            samples = (
                int.from_bytes(raw_frames[index : index + 3], "little", signed=True)
                for index in range(0, len(raw_frames), 3)
            )
        else:
            samples = (value[0] for value in struct.iter_unpack("<i", raw_frames))
        for sample in samples:
            magnitude = abs(sample)
            peak = max(peak, magnitude)
            squared_sum += magnitude * magnitude
            sample_count += 1
            if magnitude >= maximum_sample * 0.999:
                clipped_count += 1
    if sample_count == 0 or frame_count <= 0 or channel_count <= 0:
        return 0.0, 0.0, 0.0
    return (
        peak / maximum_sample,
        math.sqrt(squared_sum / sample_count) / maximum_sample,
        clipped_count / sample_count,
    )


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


def _pause_after_tts_chunk(text: str) -> float:
    ending = text.rstrip("\"'’”»)]").rstrip()
    if ending.endswith(","):
        return 0.15
    if ending.endswith((":", ";")):
        return 0.20
    if ending.endswith(("...", "…", "?", "!")):
        return 0.28
    return 0.24


def _synthesize_xtts_chunks(
    model: Any,
    text: str,
    output_path: Path,
    reference_path: Path | None,
    speaker: str,
    speed: float,
    sampling_options: dict[str, float | int],
) -> None:
    chunks = split_italian_tts_text(text)
    if not chunks:
        raise HTTPException(status_code=400, detail="Testo italiano vuoto per XTTS.")
    LOGGER.info(
        "Sintesi XTTS italiana in %d segmenti brevi; pause tra i segmenti basate "
        "sulla punteggiatura.",
        len(chunks),
    )

    combined_frames = bytearray()
    output_params: Any | None = None
    with tempfile.TemporaryDirectory(prefix="xtts_chunks_") as temporary_directory:
        for index, chunk in enumerate(chunks):
            chunk_path = Path(temporary_directory) / f"chunk_{index:03d}.wav"
            model.tts_to_file(
                text=chunk,
                file_path=str(chunk_path),
                language="it",
                speaker=None if reference_path else speaker,
                speaker_wav=str(reference_path) if reference_path else None,
                split_sentences=False,
                speed=speed,
                **sampling_options,
            )
            _validate_wav_file(chunk_path, f"XTTS segmento {index + 1}")
            with wave.open(str(chunk_path), "rb") as audio_file:
                params = audio_file.getparams()
                if params.comptype != "NONE":
                    raise HTTPException(
                        status_code=500,
                        detail="XTTS ha prodotto audio WAV compresso non concatenabile.",
                    )
                if output_params is None:
                    output_params = params
                elif (
                    params.nchannels,
                    params.sampwidth,
                    params.framerate,
                    params.comptype,
                ) != (
                    output_params.nchannels,
                    output_params.sampwidth,
                    output_params.framerate,
                    output_params.comptype,
                ):
                    raise HTTPException(
                        status_code=500,
                        detail="I segmenti XTTS hanno parametri audio incompatibili.",
                    )
                combined_frames.extend(
                    audio_file.readframes(audio_file.getnframes())
                )
            if index < len(chunks) - 1 and output_params is not None:
                pause_frames = round(
                    output_params.framerate * _pause_after_tts_chunk(chunk)
                )
                silence_sample = b"\x80" if output_params.sampwidth == 1 else b"\x00"
                combined_frames.extend(
                    silence_sample
                    * pause_frames
                    * output_params.nchannels
                    * output_params.sampwidth
                )

    if output_params is None:
        raise HTTPException(status_code=500, detail="XTTS non ha prodotto audio.")
    with wave.open(str(output_path), "wb") as output_audio:
        output_audio.setnchannels(output_params.nchannels)
        output_audio.setsampwidth(output_params.sampwidth)
        output_audio.setframerate(output_params.framerate)
        output_audio.writeframes(combined_frames)


def _apply_audio_filters(source_path: Path, *, apply_speed: bool = True) -> bytes:
    filters: list[str] = []
    # speed = _configured_xtts_speed()
    speed = _configured_xtts_speed() if apply_speed else 1.0
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
    speed = _configured_xtts_speed()
    sampling_options = _configured_xtts_sampling_options()
    if reference_path is None and not speaker:
        raise HTTPException(status_code=500, detail="XTTS_SPEAKER non può essere vuoto.")
    LOGGER.info(
        "Parametri XTTS: speed=%.2f, temperature=%.2f, top_k=%d, top_p=%.2f, "
        "repetition_penalty=%.2f.",
        speed,
        sampling_options["temperature"],
        sampling_options["top_k"],
        sampling_options["top_p"],
        sampling_options["repetition_penalty"],
    )

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
        if language == "it":
            # model.tts_to_file(
            #     text=req.text.strip(),
            #     file_path=str(output_path),
            #     language=language,
            #     speaker=None if reference_path else speaker,
            #     speaker_wav=str(reference_path) if reference_path else None,
            #     split_sentences=True,
            # )
            _synthesize_xtts_chunks(
                model,
                req.text.strip(),
                output_path,
                reference_path,
                speaker,
                speed,
                sampling_options,
            )
        else:
            model.tts_to_file(
                text=req.text.strip(),
                file_path=str(output_path),
                language=language,
                speaker=None if reference_path else speaker,
                speaker_wav=str(reference_path) if reference_path else None,
                split_sentences=True,
                speed=speed,
                **sampling_options,
            )
        _validate_wav_file(output_path, "XTTS")
        # return _apply_audio_filters(output_path)
        return _apply_audio_filters(output_path, apply_speed=False)
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