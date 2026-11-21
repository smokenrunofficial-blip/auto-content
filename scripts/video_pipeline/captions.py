# import os
import difflib
import math
import re
from pathlib import Path
from typing import Any


CAPTION_FONT_SIZE = 78
CAPTION_OUTLINE = 5
CAPTION_SHADOW = 2
CAPTION_ALIGNMENT = 2
CAPTION_MARGIN_L = 125
CAPTION_MARGIN_R = 125
# CAPTION_MARGIN_V = 610
CAPTION_MARGIN_V = 640
CAPTION_MAX_LINE_CHARACTERS = 20
CAPTION_MAX_EVENT_CHARACTERS = CAPTION_MAX_LINE_CHARACTERS * 2
CAPTION_MAX_WORDS_PER_EVENT = 3
CAPTION_EARLY_OFFSET_SECONDS = 0.05
CAPTION_MIN_EVENT_DURATION_SECONDS = 0.01
CAPTION_FONT_DIRECTORY = Path(__file__).resolve().parents[2] / "backend" / "tts" / "fonts"
CAPTION_KEYWORDS = {
    "buio", "sangue", "ombra", "ombre", "morte", "morto", "morta", "urlo", "urla",
    "silenzio", "paura", "presenza", "occhi", "dietro", "passi", "freddo",
    "sussurro", "sussurra", "rumore", "vuoto", "notte", "fantasma", "mostro",
    "sparito", "sparita", "seguito", "sola", "solo",
}
CAPTION_STOP_WORDS = {
    "a", "ad", "al", "alla", "alle", "allo", "che", "chi", "ci", "con", "da",
    "dal", "dalla", "de", "del", "della", "di", "e", "ed", "era", "è", "gli",
    "ha", "hai", "hanno", "i", "il", "in", "io", "la", "le", "li", "lo", "ma",
    "mi", "nei", "nel", "nella", "no", "non", "o", "per", "poi", "qui", "se",
    "si", "sia", "sono", "su", "tra", "tu", "un", "una", "uno", "vi",
}


def _caption_font_name() -> str:
    # fonts_directory = Path(os.environ.get("WINDIR", r"C:\Windows")) / "Fonts"
    # try:
    #     installed_stems = {
    #         path.stem.casefold()
    #         for path in fonts_directory.iterdir()
    #         if path.is_file()
    #     }
    # except OSError:
    #     return "Georgia"
    #
    # preferences = (
    #     ("Chiller", "chiller"),
    #     ("Creepster", "creepster"),
    #     ("Cinzel", "cinzel"),
    #     ("Impact", "impact"),
    #     ("Georgia", "georgia"),
    # )
    # for family, stem in preferences:
    #     if any(
    #         installed == stem or installed.startswith(f"{stem}-")
    #         for installed in installed_stems
    #     ):
    #         return family
    # return "Georgia"
    return _caption_font_names()[0]


def _caption_font_names() -> list[str]:
    if not CAPTION_FONT_DIRECTORY.is_dir():
        return ["Arial"]
    font_files = sorted(
        path
        for path in CAPTION_FONT_DIRECTORY.iterdir()
        if path.is_file() and path.suffix.casefold() in {".ttf", ".otf"}
    )
    if not font_files:
        return ["Arial"]

    return [
        re.sub(
            r"(?<=[a-z])(?=[A-Z])",
            " ",
            re.sub(r"-(?:Regular|Bold|Italic)$", "", path.stem),
        )
        for path in font_files
    ]


def _caption_keyword_indices(words: list[str]) -> set[int]:
    candidates: list[tuple[int, int, int]] = []
    for index, word in enumerate(words):
        normalized = re.sub(r"^[^\w]+|[^\w]+$", "", word.casefold())
        if not normalized or normalized in CAPTION_STOP_WORDS:
            continue
        priority = int(normalized in CAPTION_KEYWORDS)
        if priority or len(normalized) >= 5:
            candidates.append((priority, len(normalized), -index))

    if not candidates and words:
        candidates = [
            (
                0,
                len(re.sub(r"^[^\w]+|[^\w]+$", "", word)),
                -index,
            )
            for index, word in enumerate(words)
            if re.sub(r"^[^\w]+|[^\w]+$", "", word)
        ]

    candidates.sort(reverse=True)
    maximum_keywords = min(3, max(1, (len(words) + 3) // 4))
    return {-candidate[2] for candidate in candidates[:maximum_keywords]}


def _ass_time(seconds: float) -> str:
    centiseconds = max(0, round(seconds * 100))
    hours, remainder = divmod(centiseconds, 360000)
    minutes, remainder = divmod(remainder, 6000)
    whole_seconds, fraction = divmod(remainder, 100)
    return f"{hours}:{minutes:02}:{whole_seconds:02}.{fraction:02}"


def _escape_ass_text(text: str) -> str:
    return text.replace("\\", "\\\\").replace("{", "\\{").replace("}", "\\}")


def _normalize_alignment_word(word: str) -> str:
    return "".join(character.casefold() for character in word if character.isalnum())


def align_transcription_to_script(
    transcription: dict[str, Any],
    script_text: str,
) -> list[dict[str, Any]]:
    script_words = re.findall(r"\S+", script_text.strip())
    if not script_words:
        raise ValueError("Lo script della narrazione non contiene parole da sottotitolare.")

    spoken_words = [
        {
            "word": str(word["word"]).strip(),
            "start": float(word["start"]),
            "end": float(word["end"]),
        }
        for segment in transcription.get("segments", [])
        for word in segment.get("words", [])
        if word.get("word")
        and math.isfinite(float(word.get("start", 0)))
        and math.isfinite(float(word.get("end", 0)))
        and float(word.get("end", 0)) > float(word.get("start", 0))
    ]
    spoken_words = [
        word for word in spoken_words if _normalize_alignment_word(word["word"])
    ]
    if not spoken_words:
        raise RuntimeError("Whisper non ha prodotto timestamp validi per le parole.")

    aligned: list[dict[str, Any] | None] = [None] * len(script_words)
    matcher = difflib.SequenceMatcher(
        a=[_normalize_alignment_word(word["word"]) for word in spoken_words],
        b=[_normalize_alignment_word(word) for word in script_words],
        autojunk=False,
    )
    for (
        operation,
        spoken_start,
        spoken_end,
        script_start,
        script_end,
    ) in matcher.get_opcodes():
        if operation == "equal":
            for offset in range(script_end - script_start):
                source_word = spoken_words[spoken_start + offset]
                aligned[script_start + offset] = {
                    "word": script_words[script_start + offset],
                    "start": source_word["start"],
                    "end": source_word["end"],
                }
        elif operation == "replace" and spoken_start < spoken_end:
            source_words = spoken_words[spoken_start:spoken_end]
            target_count = script_end - script_start
            if target_count <= 0:
                continue
            if len(source_words) == target_count:
                for offset, source_word in enumerate(source_words):
                    aligned[script_start + offset] = {
                        "word": script_words[script_start + offset],
                        "start": source_word["start"],
                        "end": source_word["end"],
                    }
            else:
                span_start = source_words[0]["start"]
                span_end = source_words[-1]["end"]
                step = max(0.0, span_end - span_start) / target_count
                for offset in range(target_count):
                    aligned[script_start + offset] = {
                        "word": script_words[script_start + offset],
                        "start": span_start + step * offset,
                        "end": span_start + step * (offset + 1),
                    }

    missing_index = 0
    average_spoken_duration = sum(
        word["end"] - word["start"] for word in spoken_words
    ) / len(spoken_words)
    while missing_index < len(aligned):
        if aligned[missing_index] is not None:
            missing_index += 1
            continue
        missing_end = missing_index
        while missing_end < len(aligned) and aligned[missing_end] is None:
            missing_end += 1
        missing_count = missing_end - missing_index
        previous_word = aligned[missing_index - 1] if missing_index > 0 else None
        next_word = aligned[missing_end] if missing_end < len(aligned) else None
        span_start = (
            float(previous_word["end"])
            if previous_word is not None
            else spoken_words[0]["start"]
        )
        span_end = (
            float(next_word["start"])
            if next_word is not None
            else spoken_words[-1]["end"]
        )
        if span_end <= span_start:
            span_end = span_start + average_spoken_duration * missing_count
        step = (span_end - span_start) / missing_count
        for offset in range(missing_count):
            script_index = missing_index + offset
            aligned[script_index] = {
                "word": script_words[script_index],
                "start": span_start + step * offset,
                "end": span_start + step * (offset + 1),
            }
        missing_index = missing_end

    result: list[dict[str, Any]] = []
    previous_end = 0.0
    for word in aligned:
        if word is None:
            raise RuntimeError("Impossibile allineare tutte le parole dello script.")
        start = max(previous_end, 0.0, float(word["start"]))
        end = max(start + CAPTION_MIN_EVENT_DURATION_SECONDS, float(word["end"]))
        result.append({"word": word["word"], "start": start, "end": end})
        previous_end = end
    return result


# def _wrap_caption_words(words: list[str], active_index: int) -> str:
#     lines: list[str] = []
#     current_line: list[str] = []
#     line_characters = 0
#
#     for index, word in enumerate(words):
#         word = _escape_ass_text(word.strip())
#         if not word:
#             continue
#         additional_characters = len(word) + (1 if current_line else 0)
#         if (
#             current_line
#             and line_characters + additional_characters
#             > CAPTION_MAX_LINE_CHARACTERS
#         ):
#             lines.append(" ".join(current_line))
#             current_line = []
#             line_characters = 0
#
#         if index == active_index:
#             word = r"{\c&H0000FFFF&\b1}" + word + r"{\r}"
#         if current_line:
#             line_characters += 1
#         current_line.append(word)
#         line_characters += len(words[index].strip())
#
#     if current_line:
#         lines.append(" ".join(current_line))
#     return r"\N".join(lines)
# def _wrap_caption_words(words: list[str]) -> str:
#     lines: list[str] = []
#     current_line: list[str] = []
#     line_characters = 0
#
#     for word in words:
#         word = _escape_ass_text(word.strip())
#         if not word:
#             continue
#         additional_characters = len(word) + (1 if current_line else 0)
#         if (
#             current_line
#             and line_characters + additional_characters
#             > CAPTION_MAX_LINE_CHARACTERS
#         ):
#             lines.append(" ".join(current_line))
#             current_line = []
#             line_characters = 0
#
#         if current_line:
#             line_characters += 1
#         current_line.append(word)
#         line_characters += len(word)
#
#     if current_line:
#         lines.append(" ".join(current_line))
#     return r"\N".join(lines)
def _wrap_caption_words(
    words: list[str],
    highlighted_indices: set[int] | None = None,
    highlight_fonts: list[str] | None = None,
    keyword_offset: int = 0,
) -> str:
    lines: list[str] = []
    current_line: list[str] = []
    line_characters = 0
    highlighted_count = 0
    fonts = highlight_fonts or []
    highlighted_indices = highlighted_indices or set()

    for index, raw_word in enumerate(words):
        # plain_word = raw_word.strip().lower()
        plain_word = raw_word.strip()
        if not plain_word:
            continue
        visible_length = len(plain_word)
        word = _escape_ass_text(plain_word)
        if index in highlighted_indices and fonts:
            font_name = fonts[(keyword_offset + highlighted_count) % len(fonts)]
            highlighted_count += 1
            # word = rf"{{\fn{_escape_ass_text(font_name)}\b1}}{word}{{\r}}"
            word = (
                rf"{{\fn{_escape_ass_text(font_name)}\b1}}"
                # f"{_escape_ass_text(plain_word.upper())}{{\\r}}"
                f"{_escape_ass_text(plain_word)}{{\\r}}"
            )

        additional_characters = visible_length + (1 if current_line else 0)
        if (
            current_line
            and line_characters + additional_characters
            > CAPTION_MAX_LINE_CHARACTERS
        ):
            lines.append(" ".join(current_line))
            current_line = []
            line_characters = 0

        if current_line:
            line_characters += 1
        current_line.append(word)
        line_characters += visible_length

    if current_line:
        lines.append(" ".join(current_line))
    return r"\N".join(lines)


def _caption_word_groups(words: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    groups: list[list[dict[str, Any]]] = []
    current_group: list[dict[str, Any]] = []
    sentence_endings = (".", "!", "?", "…")

    for word in words:
        text = str(word["word"]).strip()
        if not text:
            continue
        candidate_texts = [
            str(item["word"]).strip() for item in current_group
        ] + [text]
        candidate_caption = _wrap_caption_words(candidate_texts)
        # len(" ".join(candidate_texts)) > CAPTION_MAX_EVENT_CHARACTERS
        if current_group and (
            # len(current_group) >= CAPTION_MAX_WORDS_PER_EVENT
            len(current_group) >= CAPTION_MAX_WORDS_PER_EVENT
            or len(" ".join(candidate_texts)) > CAPTION_MAX_EVENT_CHARACTERS
            or candidate_caption.count(r"\N") >= 2
            or str(current_group[-1]["word"]).strip().endswith(sentence_endings)
        ):
            groups.append(current_group)
            current_group = []

        current_group.append(word)
        if text.endswith(sentence_endings):
            groups.append(current_group)
            current_group = []

    if current_group:
        groups.append(current_group)
    return groups


def _caption_event_times(
    groups: list[list[dict[str, Any]]],
) -> list[tuple[float, float]]:
    event_times: list[tuple[float, float]] = []
    previous_end = 0.0
    for index, group in enumerate(groups):
        start = max(
            previous_end,
            0.0,
            float(group[0]["start"]) - CAPTION_EARLY_OFFSET_SECONDS,
        )
        end = max(
            start + CAPTION_MIN_EVENT_DURATION_SECONDS,
            float(group[-1]["end"]),
        )
        if index + 1 < len(groups):
            next_start = max(
                0.0,
                float(groups[index + 1][0]["start"])
                - CAPTION_EARLY_OFFSET_SECONDS,
            )
            end = min(end, max(start + CAPTION_MIN_EVENT_DURATION_SECONDS, next_start))
        event_times.append((start, end))
        previous_end = end
    return event_times


# def write_word_captions(transcription: dict[str, Any], destination: Path) -> int:
#     words = [
#         word
#         for segment in transcription.get("segments", [])
#         for word in segment.get("words", [])
#         if word.get("word") and float(word.get("end", 0)) > float(word.get("start", 0))
#     ]
def write_word_captions(
    transcription: dict[str, Any],
    destination: Path,
    script_text: str | None = None,
) -> int:
    if script_text is None:
        words = [
            word
            for segment in transcription.get("segments", [])
            for word in segment.get("words", [])
            if word.get("word")
            and float(word.get("end", 0)) > float(word.get("start", 0))
        ]
    else:
        words = align_transcription_to_script(transcription, script_text)
    if not words:
        raise RuntimeError("Whisper non ha prodotto timestamp per le parole della narrazione.")

    lines = [
        "[Script Info]",
        "ScriptType: v4.00+",
        "PlayResX: 1080",
        "PlayResY: 1920",
        # "WrapStyle: 2",
        "WrapStyle: 0",
        "",
        "[V4+ Styles]",
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, "
        "Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, "
        "Shadow, Alignment, MarginL, MarginR, MarginV, Encoding",
        # "Style: Default,Arial,62,&H00FFFFFF,&H000000FF,&H00000000,&H7F000000,1,0,0,0,100,100,0,0,1,4,1,2,70,70,170,1",
        (
            f"Style: Default,{_caption_font_name()},{CAPTION_FONT_SIZE},"
            "&H00FFFFFF,&H000000FF,&H00000000,&H7F000000,1,0,0,0,100,100,0,0,1,"
            f"{CAPTION_OUTLINE},{CAPTION_SHADOW},{CAPTION_ALIGNMENT},"
            f"{CAPTION_MARGIN_L},{CAPTION_MARGIN_R},{CAPTION_MARGIN_V},1"
        ),
        "",
        "[Events]",
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text",
    ]

    # for index, word in enumerate(words):
    #     first = max(0, index - 2)
    #     last = min(len(words), index + 3)
    #     caption_words = [
    #         _escape_ass_text(str(item["word"]).strip())
    #         for item in words[first:last]
    #     ]
    #     active = index - first
    #     caption_words[active] = (
    #         r"{\c&H0000FFFF&\b1}" + caption_words[active] + r"{\r}"
    #     )
    #     text = " ".join(part for part in caption_words if part)
    #     start = float(word["start"])
    #     end = float(word["end"])
    #     lines.append(
    #         f"Dialogue: 0,{_ass_time(start)},{_ass_time(end)},Default,,"
    #         f"{CAPTION_MARGIN_L},{CAPTION_MARGIN_R},{CAPTION_MARGIN_V},,{text}"
    #     )
    # for index, word in enumerate(words):
    #     first = max(0, index - 2)
    #     last = min(len(words), index + 3)
    #     caption_words = [
    #         str(item["word"]).strip()
    #         for item in words[first:last]
    #     ]
    #     active = index - first
    #     text = _wrap_caption_words(caption_words, active)
    #     start = float(word["start"])
    #     end = float(word["end"])
    #     lines.append(
    #         f"Dialogue: 0,{_ass_time(start)},{_ass_time(end)},Default,,"
    #         f"{CAPTION_MARGIN_L},{CAPTION_MARGIN_R},{CAPTION_MARGIN_V},,{text}"
    #     )
    # caption_groups = _caption_word_groups(words)
    # for index, group in enumerate(caption_groups):
    #     caption_words = [str(word["word"]).strip() for word in group]
    #     text = _wrap_caption_words(caption_words)
    caption_groups = _caption_word_groups(words)
    event_times = _caption_event_times(caption_groups)
    font_names = _caption_font_names()
    keyword_fonts = font_names[1:] or font_names[:1]
    next_keyword_index = 0
    # for index, group in enumerate(caption_groups):
    for index, (group, (start, end)) in enumerate(
        zip(caption_groups, event_times)
    ):
        caption_words = [str(word["word"]).strip() for word in group]
        keyword_indices = _caption_keyword_indices(caption_words)
        text = _wrap_caption_words(
            caption_words,
            highlighted_indices=keyword_indices,
            highlight_fonts=keyword_fonts,
            keyword_offset=next_keyword_index,
        )
        next_keyword_index += len(keyword_indices)
        # start = float(group[0]["start"])
        # end = float(group[-1]["end"])
        # if index + 1 < len(caption_groups):
        #     next_start = float(caption_groups[index + 1][0]["start"])
        #     end = max(end, next_start)
        lines.append(
            f"Dialogue: 0,{_ass_time(start)},{_ass_time(end)},Default,,"
            f"{CAPTION_MARGIN_L},{CAPTION_MARGIN_R},{CAPTION_MARGIN_V},,{text}"
        )

    destination.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return len(words)
