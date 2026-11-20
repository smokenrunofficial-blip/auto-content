from pathlib import Path
from typing import Optional
import json
import math
import os
import shutil
import subprocess
import tempfile

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
DEFAULT_PIPER_LENGTH_SCALE = 1.18
DATA_DIR = BASE_DIR / "data"
SETTINGS_FILE = DATA_DIR / "profile_settings.json"

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


@app.post("/api/tts")
def synth(req: TTSRequest):
    if not req.text or not req.text.strip():
        raise HTTPException(status_code=400, detail="Testo mancante")

    if not PIPER_EXE.exists():
        raise HTTPException(status_code=500, detail=f"Binario Piper non trovato: {PIPER_EXE}")

    try:
        model_path = resolve_model_path()
        espeak_data = PIPER_DIR / "espeak-ng-data"
        length_scale = _configured_piper_length_scale()

        temp_file = None
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

            wav_bytes = temp_file.read_bytes()
            if wav_bytes[:4] != b"RIFF" or wav_bytes[8:12] != b"WAVE":
                raise HTTPException(
                    status_code=500,
                    detail=f"Output non valido: file generato non è un WAV corretto. Anteprima: {wav_bytes[:32]!r}",
                )

            return Response(content=wav_bytes, media_type="audio/wav")
        finally:
            if temp_file and temp_file.exists():
                temp_file.unlink(missing_ok=True)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except Exception as exc:
        if isinstance(exc, HTTPException):
            raise exc
        raise HTTPException(status_code=500, detail=str(exc)) from exc


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)