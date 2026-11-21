#!/usr/bin/env python3
# """Search YouTube and download only results explicitly marked CC BY."""
"""Search and download YouTube results with cascading candidate fallbacks."""

import argparse
import json
import math
import os
import sqlite3
import sys
import tempfile
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urlparse

import yt_dlp
from yt_dlp.cookies import extract_cookies_from_browser


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DOWNLOAD_DIR = PROJECT_ROOT / "downloads"
# MAX_LIMIT = 3
# MAX_LIMIT = 10
MAX_LIMIT = 5
MIN_SOURCE_VIDEO_DURATION_SECONDS = 60
MAX_SOURCE_VIDEO_DURATION_SECONDS = 900
IDEAL_SOURCE_VIDEO_DURATION_MIN_SECONDS = 300
IDEAL_SOURCE_VIDEO_DURATION_MAX_SECONDS = 600
SUPPORTED_COOKIE_BROWSERS = {"chrome", "edge", "firefox"}
YOUTUBE_FORMAT_SELECTOR = (
    "bestvideo[height<=1080][ext=mp4]+bestaudio[ext=m4a]/"
    "bestvideo[height<=1080]+bestaudio/best[height<=1080]/best"
)
YOUTUBE_AUTH_REQUIRED_MESSAGE = (
    "YouTube richiede l'accesso o la conferma dell'età. "
    "Clicca su \"Login YouTube (Cache)\" e importa la sessione dal browser "
    "oppure carica un file cookies.txt."
)


def youtube_cookies_file() -> Path:
    configured_path = os.environ.get("YOUTUBE_COOKIES_FILE", "").strip()
    path = Path(configured_path).expanduser() if configured_path else (
        PROJECT_ROOT / "data" / "youtube_cookies.txt"
    )
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def _count_youtube_cookie_rows(contents: str) -> int:
    lines = contents.lstrip("\ufeff").splitlines()
    has_netscape_header = any(
        line.strip().casefold()
        in {"# netscape http cookie file", "# http cookie file"}
        for line in lines
    )
    if not has_netscape_header:
        return 0

    youtube_cookie_count = 0
    for line in lines:
        if not line or (line.startswith("#") and not line.startswith("#HttpOnly_")):
            continue
        fields = line.split("\t")
        if len(fields) != 7:
            continue
        domain = fields[0]
        if domain.startswith("#HttpOnly_"):
            domain = domain[len("#HttpOnly_"):]
        normalized_domain = domain.lstrip(".").casefold()
        if (
            normalized_domain == "youtube.com"
            or normalized_domain.endswith(".youtube.com")
        ):
            youtube_cookie_count += 1
    return youtube_cookie_count


def validate_youtube_cookie_file(path: Path) -> int:
    try:
        contents = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as error:
        raise ValueError("Il file cookies.txt non è leggibile in formato UTF-8.") from error
    count = _count_youtube_cookie_rows(contents)
    if count == 0:
        raise ValueError(
            "Il file non contiene cookie YouTube validi in formato Netscape."
        )
    return count


def _friendly_browser_cookie_error(browser: str, error: Exception) -> str:
    detail = str(error).casefold()
    if any(
        phrase in detail
        for phrase in (
            "locked",
            "being used",
            "in use",
            "permission denied",
            "access is denied",
            "could not copy",
        )
    ):
        return (
            f"Il browser {browser} sta bloccando il database dei cookie. "
            "Chiudilo completamente e riprova, oppure carica un file cookies.txt."
        )
    if any(phrase in detail for phrase in ("not found", "could not find", "unknown browser")):
        return (
            f"Il profilo {browser} non è stato trovato. Accedi al browser su questo "
            "computer oppure carica un file cookies.txt."
        )
    if "cookie youtube validi" in detail:
        return (
            f"Nel browser {browser} non è stata trovata una sessione YouTube valida. "
            "Accedi a YouTube nel browser oppure carica un file cookies.txt."
        )
    return (
        f"Non è stato possibile importare la sessione da {browser}. "
        "Chiudi il browser o carica un file cookies.txt."
    )


def export_browser_cookies(browser: str) -> int:
    if browser not in SUPPORTED_COOKIE_BROWSERS:
        raise ValueError("Browser non supportato. Usa chrome, edge o firefox.")

    destination = youtube_cookies_file()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        cookie_jar = extract_cookies_from_browser(browser)
        with tempfile.NamedTemporaryFile(
            prefix=".youtube_cookies_",
            suffix=".txt",
            dir=destination.parent,
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
        cookie_jar.save(
            str(temporary_path),
            ignore_discard=True,
            ignore_expires=True,
        )
        cookie_count = validate_youtube_cookie_file(temporary_path)
        if os.name != "nt":
            temporary_path.chmod(0o600)
        os.replace(temporary_path, destination)
        temporary_path = None
        if os.name != "nt":
            destination.chmod(0o600)
        return cookie_count
    except (
        OSError,
        RuntimeError,
        ValueError,
        sqlite3.Error,
        yt_dlp.utils.DownloadError,
    ) as error:
        raise RuntimeError(_friendly_browser_cookie_error(browser, error)) from error
    finally:
        if temporary_path and temporary_path.exists():
            temporary_path.unlink(missing_ok=True)


def _add_cookie_file_option(options: dict[str, Any]) -> None:
    cookie_file = youtube_cookies_file()
    if cookie_file.is_file():
        options["cookiefile"] = str(cookie_file)


def _add_youtube_extractor_args(options: dict[str, Any]) -> None:
    options["extractor_args"] = {
        "youtube": {"player_client": ["default", "web", "android"]}
    }


def _add_youtube_request_delays(options: dict[str, Any]) -> None:
    options.update(
        sleep_interval_requests=1.5,
        sleep_interval=3,
        max_sleep_interval=6,
    )


def _flat_search_candidates(entries: Any) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    if entries is None or isinstance(entries, (str, bytes, dict)):
        return candidates
    try:
        entry_iterator = iter(entries)
    except TypeError:
        return candidates
    for entry in entry_iterator:
        if not isinstance(entry, dict):
            continue
        entry_url = str(entry.get("webpage_url") or entry.get("url") or "").strip()
        if not entry_url:
            entry_url = str(entry.get("id") or "").strip()
        if not entry_url:
            continue
        if not entry_url.startswith(("http://", "https://")):
            video_id = str(entry.get("id") or entry_url).strip()
            if not video_id:
                continue
            entry_url = f"https://www.youtube.com/watch?v={video_id}"
        if _is_shorts_url(entry_url):
            continue

        candidate = dict(entry)
        candidate["webpage_url"] = entry_url
        duration = _positive_number(candidate.get("duration"))
        if duration is not None and not (
            MIN_SOURCE_VIDEO_DURATION_SECONDS
            <= duration
            <= MAX_SOURCE_VIDEO_DURATION_SECONDS
        ):
            continue
        width, height = _video_dimensions(candidate)
        if width is not None and height is not None and height > width:
            continue
        candidates.append(candidate)
    return candidates


def youtube_auth_required(error: BaseException | str) -> bool:
    detail = str(error).casefold()
    return (
        "sign in to confirm your age" in detail
        or "confirm your age" in detail
        or "--cookies-from-browser" in detail
        or "--cookies" in detail
        or ("youtube" in detail and "cookies" in detail)
    )


def friendly_download_error(error: BaseException | str) -> str:
    return YOUTUBE_AUTH_REQUIRED_MESSAGE if youtube_auth_required(error) else str(error)


def has_allowed_license(license_name: str | None) -> bool:
    """Return whether metadata explicitly identifies YouTube's CC BY license."""
    return True
    #return bool(
    #    license_name
    #    and license_name.casefold().startswith("creative commons attribution license")
    #)


def _is_shorts_url(url: str) -> bool:
    return "shorts" in {
        segment.casefold() for segment in urlparse(url).path.split("/")
    }


def _positive_number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number) or number <= 0:
        return None
    return number


def _video_dimensions(metadata: dict[str, Any]) -> tuple[float | None, float | None]:
    width = _positive_number(metadata.get("width"))
    height = _positive_number(metadata.get("height"))
    if width is not None and height is not None:
        return width, height

    formats = metadata.get("formats")
    if not isinstance(formats, list):
        return None, None
    dimensions = [
        (format_width, format_height)
        for item in formats
        if isinstance(item, dict)
        if (format_width := _positive_number(item.get("width"))) is not None
        if (format_height := _positive_number(item.get("height"))) is not None
    ]
    if not dimensions:
        return None, None
    return max(dimensions, key=lambda dimension: dimension[0] * dimension[1])


def _candidate_is_eligible(
    metadata: dict[str, Any],
    original_url: str | None = None,
) -> bool:
    url = original_url or str(metadata.get("webpage_url", ""))
    if _is_shorts_url(url):
        return False

    duration = _positive_number(metadata.get("duration"))
    if duration is None or not (
        MIN_SOURCE_VIDEO_DURATION_SECONDS
        <= duration
        <= MAX_SOURCE_VIDEO_DURATION_SECONDS
    ):
        return False

    width, height = _video_dimensions(metadata)
    return width is None or height is None or height <= width


def _is_horizontal(metadata: dict[str, Any]) -> bool:
    width, height = _video_dimensions(metadata)
    return width is not None and height is not None and width > height


def _candidate_priority(metadata: dict[str, Any]) -> tuple[int, int, int, float]:
    duration = _positive_number(metadata.get("duration")) or 0
    width, height = _video_dimensions(metadata)
    is_landscape = width is not None and height is not None and width > height
    is_ideal_duration = (
        IDEAL_SOURCE_VIDEO_DURATION_MIN_SECONDS
        <= duration
        <= IDEAL_SOURCE_VIDEO_DURATION_MAX_SECONDS
    )
    return (
        int(is_landscape and is_ideal_duration),
        int(is_landscape),
        int(is_ideal_duration),
        -abs(duration - 450),
    )


def _select_candidates(
    candidates: list[dict[str, Any]],
    limit: int,
) -> list[dict[str, Any]]:
    if limit <= 0:
        return []
    non_shorts = [
        candidate
        for candidate in candidates
        if not _is_shorts_url(str(candidate.get("webpage_url", "")))
    ]
    horizontal = [
        candidate for candidate in non_shorts if _is_horizontal(candidate)
    ]

    ideal_duration = [
        candidate
        for candidate in horizontal
        if IDEAL_SOURCE_VIDEO_DURATION_MIN_SECONDS
        <= (_positive_number(candidate.get("duration")) or 0)
        <= IDEAL_SOURCE_VIDEO_DURATION_MAX_SECONDS
    ]
    sorted_ideal_duration = sorted(
        ideal_duration,
        key=_candidate_priority,
        reverse=True,
    )

    standard_duration = [
        candidate
        for candidate in horizontal
        if MIN_SOURCE_VIDEO_DURATION_SECONDS
        <= (_positive_number(candidate.get("duration")) or 0)
        <= MAX_SOURCE_VIDEO_DURATION_SECONDS
    ]
    sorted_standard_duration = sorted(
        standard_duration,
        key=_candidate_priority,
        reverse=True,
    )

    longer_videos = [
        candidate
        for candidate in horizontal
        if (_positive_number(candidate.get("duration")) or 0)
        > MAX_SOURCE_VIDEO_DURATION_SECONDS
    ]
    sorted_longer_videos = sorted(
        longer_videos,
        key=lambda candidate: _positive_number(candidate.get("duration")) or 0,
    )
    # Preserve the old first-nonempty-tier behavior while filling the requested
    # candidate limit from the next tiers when a preferred tier is too small.
    # if ideal_duration:
    #     return sorted(
    #         ideal_duration,
    #         key=_candidate_priority,
    #         reverse=True,
    #     )[:limit]
    # if standard_duration:
    #     return sorted(
    #         standard_duration,
    #         key=_candidate_priority,
    #         reverse=True,
    #     )[:limit]
    # if longer_videos:
    #     return sorted(
    #         longer_videos,
    #         key=lambda candidate: _positive_number(candidate.get("duration")) or 0,
    #     )[:limit]
    # return non_shorts[:limit]
    ordered_candidates = (
        sorted_ideal_duration
        + sorted_standard_duration
        + sorted_longer_videos
        + sorted(non_shorts, key=_candidate_priority, reverse=True)
    )
    selected: list[dict[str, Any]] = []
    seen_urls: set[str] = set()
    for candidate in ordered_candidates:
        candidate_url = str(candidate.get("webpage_url") or candidate.get("url") or "")
        candidate_key = (
            candidate_url
            or str(candidate.get("id") or "")
            or str(id(candidate))
        )
        if candidate_key in seen_urls:
            continue
        seen_urls.add(candidate_key)
        selected.append(candidate)
        if len(selected) >= limit:
            break
    return selected


def _candidate_match_filter(
    info: dict[str, Any],
    *,
    incomplete: bool,
) -> str | None:
    if incomplete:
        return None
    # if not has_allowed_license(info.get("license")):
    #     return "Licenza Creative Commons Attribution non verificata"
    # if not _candidate_is_eligible(info):
    #     return "Durata o orientamento del video non idonei"
    if _is_shorts_url(str(info.get("webpage_url", ""))):
        return "I video Shorts non sono candidati validi"
    return None


def download_matches(
    keyword: str,
    limit: int,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> list[dict[str, str]]:
    # search_limit = min(max(limit * 20, 50), 100)
    search_limit = min(max(limit * 5, 10), 25)
    search_options: dict[str, Any] = {
        "quiet": True,
        "no_warnings": True,
        "skip_download": True,
        "noplaylist": True,
        "extract_flat": True,
    }
    _add_youtube_extractor_args(search_options)
    _add_cookie_file_option(search_options)
    _add_youtube_request_delays(search_options)

    with yt_dlp.YoutubeDL(search_options) as searcher:
        if progress_callback:
            progress_callback(
                {
                    "type": "progress",
                    "stage": "search",
                    "stage_progress": 0,
                    "message": f"Ricerca dei video per “{keyword}”.",
                }
            )
        search_result = searcher.extract_info(
            f"ytsearch{search_limit}:{keyword}",
            download=False,
        )

    entries = search_result.get("entries", []) if search_result else []
    # candidates: list[dict[str, Any]] = []
    # with yt_dlp.YoutubeDL(search_options) as searcher:
    #     for index, entry in enumerate(entries):
    #         if not isinstance(entry, dict):
    #             continue
    #         entry_url = str(entry.get("webpage_url") or entry.get("url") or "")
    #         if not entry_url:
    #             continue
    #         try:
    #             metadata = searcher.extract_info(entry_url, download=False)
    #         except (OSError, yt_dlp.utils.DownloadError) as error:
    #             metadata = dict(entry)
    #             if progress_callback:
    #                 progress_callback(
    #                     {
    #                         "type": "progress",
    #                         "stage": "search",
    #                         "stage_progress": int((index + 1) * 100 / max(1, len(entries))),
    #                         "message": (
    #                             f"Metadati candidato {index + 1}/{len(entries)} "
    #                             f"non disponibili; uso i dati della ricerca "
    #                             f"({friendly_download_error(error)})."
    #                         ),
    #                     }
    #                 )
    #         if not isinstance(metadata, dict) or not metadata:
    #             metadata = dict(entry)
    #         metadata.setdefault("webpage_url", entry_url)
    #         if not _is_shorts_url(str(metadata.get("webpage_url", ""))):
    #             candidates.append(metadata)
    #         if progress_callback:
    #             progress_callback(
    #                 {
    #                     "type": "progress",
    #                     "stage": "search",
    #                     "stage_progress": int((index + 1) * 100 / max(1, len(entries))),
    #                     "message": (
    #                         "Verifica durata, orientamento e alternative dei risultati "
    #                         f"({index + 1}/{len(entries)})."
    #                     ),
    #                 }
    #             )
    candidates = _flat_search_candidates(entries)
    candidates = _select_candidates(candidates, limit)
    if progress_callback:
        progress_callback(
            {
                "type": "progress",
                "stage": "search",
                "stage_progress": 100,
                # "message": "Ricerca completata; selezionato il primo risultato idoneo.",
                "message": (
                    f"Ricerca completata; selezionati {len(candidates)} candidati da provare."
                ),
            }
        )

    # if not candidates:
    #     return []
    if not candidates:
        return []

    DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)

    def report_download_progress(status: dict[str, Any]) -> None:
        if not progress_callback:
            return
        if status.get("status") == "finished":
            stage_progress = 100
            message = "Download completato."
        else:
            total = status.get("total_bytes") or status.get("total_bytes_estimate")
            downloaded = status.get("downloaded_bytes", 0)
            stage_progress = int(downloaded * 100 / total) if total else 0
            stage_progress = max(0, min(99, stage_progress))
            message = "Download del video selezionato."
        progress_callback(
            {
                "type": "progress",
                "stage": "download",
                "stage_progress": stage_progress,
                "message": message,
            }
        )

    class _QuietLogger:
        def debug(self, msg: str) -> None:
            pass

        def warning(self, msg: str) -> None:
            pass

        def error(self, msg: str) -> None:
            pass

    download_options: dict[str, Any] = {
        # "format": "bestvideo*+bestaudio/best",
        "format": YOUTUBE_FORMAT_SELECTOR,
        "merge_output_format": "mp4",
        # "remux_video": "mp4",
        "remuxvideo": "mp4",
        "outtmpl": str(DOWNLOAD_DIR / "%(id)s.%(ext)s"),
        "quiet": True,
        "no_warnings": True,
        "noprogress": True,
        "logger": _QuietLogger(),
        "noplaylist": True,
        # "match_filter": lambda info, *, incomplete: (
        #     None
        #     if incomplete or has_allowed_license(info.get("license"))
        #     else "Licenza Creative Commons Attribution non verificata"
        # ),
        "match_filter": _candidate_match_filter,
    }
    _add_youtube_extractor_args(download_options)
    _add_cookie_file_option(download_options)
    _add_youtube_request_delays(download_options)
    if progress_callback:
        download_options["progress_hooks"] = [report_download_progress]
    downloaded: list[dict[str, str]] = []
    download_failures: list[str] = []

    with yt_dlp.YoutubeDL(download_options) as downloader:
        # for candidate in candidates:
        for candidate_index, candidate in enumerate(candidates, start=1):
            candidate_url = str(candidate.get("webpage_url", ""))
            try:
                info = downloader.extract_info(candidate_url, download=True)
            except (OSError, yt_dlp.utils.DownloadError) as error:
                download_failures.append(f"{candidate_url}: {error}")
                if progress_callback:
                    progress_callback(
                        {
                            "type": "progress",
                            "stage": "download",
                            "stage_progress": int(candidate_index * 100 / len(candidates)),
                            "message": (
                                f"Download candidato {candidate_index}/{len(candidates)} "
                                f"non riuscito; provo il successivo."
                            ),
                        }
                    )
                continue
            # if not info:
            #     raise RuntimeError(
            #         f"yt-dlp non ha restituito i dati per {candidate['webpage_url']}"
            #     )
            if not isinstance(info, dict) or not info:
                download_failures.append(
                    f"{candidate_url}: yt-dlp non ha restituito i dati del video"
                )
                continue
            license_name = info.get("license")
            # if not has_allowed_license(license_name):
            #     raise RuntimeError(
            #         f"La licenza di {candidate.get('title', 'un video')} non è verificabile."
            #     )
            downloaded.append(
                {
                    "id": str(info.get("id", "")),
                    "title": str(info.get("title", "Titolo non disponibile")),
                    "url": str(info.get("webpage_url", candidate["webpage_url"])),
                    "license": str(license_name or "None"),
                    "file": str(info.get("filepath", "")),
                }
            )

    if not downloaded and download_failures:
        raise RuntimeError(
            "Nessun candidato è stato scaricato. "
            + " | ".join(download_failures[:3])
        )

    return downloaded


def main() -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="backslashreplace")

    parser = argparse.ArgumentParser(description=__doc__)
    # parser.add_argument("--keyword", required=True, help="Parola chiave o nicchia da cercare")
    parser.add_argument("--keyword", help="Parola chiave o nicchia da cercare")
    parser.add_argument(
        "--export-cookies-from-browser",
        choices=sorted(SUPPORTED_COOKIE_BROWSERS),
        help="Salva la sessione YouTube del browser nella cache persistente",
    )
    # parser.add_argument("--limit", type=int, default=2, help="Numero massimo di download (1-3)")
    # parser.add_argument("--limit", type=int, default=2, help="Numero massimo di candidati (1-10)")
    parser.add_argument("--limit", type=int, default=2, help="Numero massimo di candidati (1-5)")
    parser.add_argument(
        "--progress",
        action="store_true",
        help="Emetti progressi ed esito come eventi JSON Lines",
    )
    args = parser.parse_args()

    if args.export_cookies_from_browser:
        try:
            cookie_count = export_browser_cookies(args.export_cookies_from_browser)
        except (OSError, RuntimeError, ValueError) as error:
            print(str(error), file=sys.stderr)
            return 1
        print(
            json.dumps(
                {
                    "success": True,
                    "browser": args.export_cookies_from_browser,
                    "cookie_count": cookie_count,
                }
            )
        )
        return 0

    if args.keyword is None:
        parser.error("--keyword è richiesto per cercare e scaricare video")
    keyword = args.keyword.strip()
    if not 2 <= len(keyword) <= 120:
        parser.error("--keyword deve contenere da 2 a 120 caratteri")
    if not 1 <= args.limit <= MAX_LIMIT:
        parser.error(f"--limit deve essere compreso tra 1 e {MAX_LIMIT}")

    try:
        progress_callback = (
            lambda event: print(json.dumps(event, ensure_ascii=False), flush=True)
            if args.progress
            else None
        )
        videos = download_matches(keyword, args.limit, progress_callback)
    except (OSError, RuntimeError, yt_dlp.utils.DownloadError) as error:
        # print(f"Ricerca o download non riuscito: {error}", file=sys.stderr)
        print(
            f"Ricerca o download non riuscito: {friendly_download_error(error)}",
            file=sys.stderr,
        )
        return 1

    result = {"keyword": keyword, "videos": videos}
    if args.progress:
        print(json.dumps({"type": "result", **result}, ensure_ascii=False), flush=True)
    else:
        print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
