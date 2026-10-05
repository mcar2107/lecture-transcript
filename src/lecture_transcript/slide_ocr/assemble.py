"""Сборка результата слайда в Markdown (задачи 4.5, 4.6).

Правила, принятые здесь и зафиксированные явно:

1. Порядок строк — сверху вниз, внутри визуальной строки слайда — слева
   направо. Боксы PaddleOCR по вертикали «дрожат» на 1-3 px, поэтому строка
   определяется не точным равенством `bbox.y`, а перекрытием по вертикали:
   фрагменты попадают в одну строку, если их центры расходятся не больше
   чем на `line_tolerance_ratio` от высоты бокса. Сортировка устойчивая:
   фрагменты без bbox (например, от VLM-бэкенда, который отдаёт готовый
   Markdown) сохраняют порядок выдачи бэкенда и ставятся после фрагментов
   с боксами. Если боксов нет ни у одного фрагмента — порядок бэкенда
   сохраняется целиком.

2. Ссылка на PNG обязательна ВСЕГДА, в том числе для слайда, на котором
   ничего не распозналось (требование «Ссылка на изображение обязательна»).

3. Фрагмент помечается неуверенным при `confidence < low_confidence_threshold`
   (по умолчанию 0.60, TODO(7.6) — подобрать эмпирически). Уже выставленный
   бэкендом флаг `low_confidence=True` не снимается.

4. Слайд помечается `unreliable=True`, если распознанные фрагменты есть и
   доля неуверенных среди них >= `unreliable_ratio` (по умолчанию 0.5).
   Частный случай из спецификации — «ни один фрагмент не распознан с
   достаточной уверенностью» (доля 1.0) — этим правилом покрыт. Пустой
   слайд ненадёжным НЕ считается: пустой результат ошибкой не является.
   Исключение — бэкенд сам отбросил строки (гибрид выбрасывает похожее на
   рукописное, D5): такой слайд ненадёжен всегда, иначе размытый кроп, все
   строки которого ушли под порог, выглядел бы честно пустым слайдом, и
   человек не узнал бы, что надо смотреть в PNG. Что считать отброшенным
   содержимым, решает бэкенд: гибрид не сообщает об одиночных символах —
   это иконки и картинки, прочитанные как буква.
   Стадия выставляет только флаг; видимую пометку по нему рисует рендер
   транскрипта (transcript-assembly), в `SlideOcr.markdown` её нет.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Sequence

from ..contracts import OcrFragment, Slide, SlideOcr

#: Пометка неуверенного фрагмента в Markdown.
LOW_CONFIDENCE_MARK = "_(?)_"


@dataclass(frozen=True)
class AssembleConfig:
    """Пороги сборки. TODO(7.6): подобрать эмпирически на эталонной записи."""

    #: Ниже этой уверенности фрагмент помечается как неуверенный.
    low_confidence_threshold: float = 0.60
    #: Доля неуверенных фрагментов, с которой слайд считается ненадёжным.
    unreliable_ratio: float = 0.5
    #: Отмечать неуверенные фрагменты прямо в Markdown.
    mark_in_markdown: bool = True
    #: Доля высоты бокса, в пределах которой фрагменты считаются лежащими
    #: на одной визуальной строке слайда (устойчивость к джиттеру боксов).
    line_tolerance_ratio: float = 0.5


#: Символы, которые в тексте слайда должны остаться текстом, а не разметкой.
_MD_ESCAPE_RE = re.compile(r"([\\`*_{}\[\]$|])")
#: То же для символов, которые опасны только в начале строки.
_MD_LINE_START_RE = re.compile(r"^([#>+\-=])")
#: Символы пути, ломающие Markdown-ссылку `![...](путь)`.
_PATH_ESCAPES = {" ": "%20", "(": "%28", ")": "%29", "<": "%3C", ">": "%3E"}


DEFAULT_ASSEMBLE = AssembleConfig()


class RecognizedFragments(tuple):
    """Результат `recognize()`: фрагменты + число строк, отброшенных бэкендом.

    Контракт `OcrBackend.recognize` — последовательность `OcrFragment`;
    кортеж с дополнительным атрибутом ему удовлетворяет, а сборка по
    `dropped` помечает слайд ненадёжным (см. правило 4).
    """

    dropped: int = 0

    def __new__(cls, fragments: Sequence[OcrFragment] = (), dropped: int = 0):
        instance = super().__new__(cls, fragments)
        instance.dropped = dropped
        return instance


def sort_fragments(
    fragments: Sequence[OcrFragment],
    config: AssembleConfig = DEFAULT_ASSEMBLE,
) -> list[OcrFragment]:
    """Упорядочить фрагменты сверху вниз, слева направо (см. правило 1).

    Фрагменты сначала группируются в визуальные строки слайда по вертикали
    (допуск — доля высоты бокса), и только внутри строки идёт сортировка по
    `x`. Без группировки разница в 1-3 px по `y`, обычная для реальных боксов
    PaddleOCR, полностью переопределяла бы порядок слева направо.
    """
    items = list(fragments)
    if not any(fragment.bbox is not None for fragment in items):
        return items

    boxed = [(index, item) for index, item in enumerate(items) if item.bbox is not None]
    loose = [(index, item) for index, item in enumerate(items) if item.bbox is None]
    boxed.sort(key=lambda pair: (pair[1].bbox.y, pair[1].bbox.x, pair[0]))

    rows: list[dict] = []
    for index, fragment in boxed:
        bbox = fragment.bbox
        center = bbox.y + bbox.height / 2.0
        row = rows[-1] if rows else None
        if row is not None:
            tolerance = max(row["height"], bbox.height) * config.line_tolerance_ratio
            if abs(center - row["center"]) <= tolerance:
                row["items"].append((index, fragment))
                row["center"] = sum(
                    item.bbox.y + item.bbox.height / 2.0 for _, item in row["items"]
                ) / len(row["items"])
                row["height"] = max(row["height"], bbox.height)
                continue
        rows.append({"center": center, "height": bbox.height, "items": [(index, fragment)]})

    ordered: list[OcrFragment] = []
    for row in rows:
        row["items"].sort(key=lambda pair: (pair[1].bbox.x, pair[0]))
        ordered.extend(fragment for _, fragment in row["items"])
    ordered.extend(fragment for _, fragment in loose)
    return ordered


def mark_low_confidence(
    fragments: Sequence[OcrFragment],
    config: AssembleConfig = DEFAULT_ASSEMBLE,
) -> list[OcrFragment]:
    """Проставить `low_confidence` по порогу уверенности (см. правило 3)."""
    marked: list[OcrFragment] = []
    for fragment in fragments:
        low = fragment.low_confidence or fragment.confidence < config.low_confidence_threshold
        marked.append(replace(fragment, low_confidence=low))
    return marked


def is_unreliable(
    fragments: Sequence[OcrFragment],
    config: AssembleConfig = DEFAULT_ASSEMBLE,
) -> bool:
    """Считать ли слайд ненадёжным (см. правило 4)."""
    if not fragments:
        return False
    low = sum(1 for fragment in fragments if fragment.low_confidence)
    return low / len(fragments) >= config.unreliable_ratio


def escape_markdown(text: str) -> str:
    """Экранировать разметку в тексте слайда.

    Стадия сама вводит соглашение «`$...$` — это LaTeX», поэтому доллар в
    печатном тексте («100$ за штуку») обязан остаться долларом, а не открыть
    формулу. То же для `#`, `|`, `*`, `_`, `>` и прочей разметки, которую
    OCR читает как обычные символы слайда.
    """
    escaped = _MD_ESCAPE_RE.sub(r"\\\1", text)
    return _MD_LINE_START_RE.sub(r"\\\1", escaped)


def image_link(slide_index: int, image_path: Path) -> str:
    """Markdown-ссылка на PNG слайда — обязательная часть результата.

    Пробелы и скобки в пути кодируются процентами: иначе каталог вида
    «Лекция 3 (запись)» разваливает ссылку на изображение.
    """
    path = Path(image_path).as_posix()
    for char, replacement in _PATH_ESCAPES.items():
        path = path.replace(char, replacement)
    return f"![слайд {slide_index}]({path})"


def render_markdown(
    slide_index: int,
    image_path: Path,
    fragments: Sequence[OcrFragment],
    config: AssembleConfig = DEFAULT_ASSEMBLE,
) -> str:
    """Собрать Markdown слайда: строки в порядке слайда + ссылка на PNG.

    Строки слайда разделяются пустой строкой: одиночный перевод строки в
    CommonMark — soft break, и структура строк слайда при рендере схлопнулась
    бы в один абзац.

    Пометки уровня слайда («распознано неуверенно») здесь нет намеренно:
    это представление, и его владелец — рендер транскрипта, который рисует
    её по флагу `SlideOcr.unreliable`. Иначе пометка выходит дважды.
    """
    blocks: list[str] = []
    for fragment in fragments:
        text = fragment.text.strip()
        if not text:
            continue
        if fragment.kind == "text":
            text = escape_markdown(text)
        if config.mark_in_markdown and fragment.low_confidence:
            text = f"{text} {LOW_CONFIDENCE_MARK}"
        blocks.append(text)
    blocks.append(image_link(slide_index, image_path))
    return "\n\n".join(blocks).strip() + "\n"


def assemble_slide_ocr(
    slide_index: int,
    image_path: Path,
    fragments: Sequence[OcrFragment],
    backend: str = "",
    config: AssembleConfig = DEFAULT_ASSEMBLE,
    dropped: int = 0,
) -> SlideOcr:
    """Собрать `SlideOcr` из сырых фрагментов бэкенда.

    `dropped` — сколько строк бэкенд отбросил сам; берётся и из атрибута
    `RecognizedFragments`, если фрагменты переданы как есть.
    """
    dropped = max(dropped, getattr(fragments, "dropped", 0))
    ordered = mark_low_confidence(sort_fragments(fragments, config), config)
    unreliable = is_unreliable(ordered, config) or dropped > 0
    markdown = render_markdown(slide_index, image_path, ordered, config)
    return SlideOcr(
        slide_index=slide_index,
        image_path=Path(image_path),
        fragments=tuple(ordered),
        markdown=markdown,
        unreliable=unreliable,
        backend=backend,
    )


def recognize_slide(
    slide: Slide,
    backend,
    config: AssembleConfig = DEFAULT_ASSEMBLE,
) -> SlideOcr:
    """Распознать один слайд выбранным бэкендом и собрать результат."""
    fragments = backend.recognize(slide.image_path)
    return assemble_slide_ocr(
        slide_index=slide.index,
        image_path=slide.image_path,
        fragments=tuple(fragments),
        backend=getattr(backend, "name", ""),
        config=config,
        dropped=getattr(fragments, "dropped", 0),
    )


def recognize_slides(
    slides: Sequence[Slide],
    backend_name: str = "hybrid",
    config: AssembleConfig = DEFAULT_ASSEMBLE,
    unload: bool = True,
) -> list[SlideOcr]:
    """Этап OCR слайдов целиком.

    Доступность бэкенда проверяется ДО начала прогона (требование
    «Недоступный бэкенд»), офлайн-режим включается до загрузки моделей
    (задача 4.9), модель выгружается по завершении этапа (design D7).
    """
    from .offline import enforce_offline
    from .registry import ensure_available

    enforce_offline()
    backend = ensure_available(backend_name)
    try:
        return [recognize_slide(slide, backend, config) for slide in slides]
    finally:
        if unload:
            backend.unload()
