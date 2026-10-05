"""Гибридный бэкенд по умолчанию: PaddleOCR + роутинг + pix2tex (design D5).

Схема: PaddleOCR отдаёт строки с боксами и уверенностью -> `routing`
решает, какие строки формульные -> кроп такой строки уходит в pix2tex и
возвращается как `$...$`. Рукописное не распознаётся сознательно (D5):
доступ к нему обеспечивает ссылка на PNG слайда в собранном Markdown.

Оба вложенных бэкенда инъектируются в конструктор — это позволяет
тестировать логику гибрида на фейках, без весов моделей.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol, Sequence

from ..contracts import Availability, OcrFragment, Rect
from .assemble import RecognizedFragments
from .paddle_backend import PaddleTextBackend
from .pix2tex_backend import DEFAULT_FORMULA_CONFIDENCE, Pix2TexBackend, latex_to_markdown
from .routing import (
    DEFAULT_ROUTING,
    RoutingConfig,
    looks_like_formula,
    looks_like_handwriting,
    route_line,
)

#: На сколько пикселей расширять кроп строки перед подачей в pix2tex:
#: бокс PaddleOCR режет надстрочные индексы вплотную.
CROP_PADDING_PX = 6

#: Сколько букв и цифр должно быть в отброшенной строке, чтобы она считалась
#: потерянным содержимым и делала слайд ненадёжным.
MIN_DROPPED_ALNUM = 2


def _alnum_count(text: str) -> int:
    return sum(1 for ch in text if ch.isalnum())


class FormulaRecognizer(Protocol):
    """Минимальный контракт формульного распознавателя для гибрида."""

    def check_availability(self) -> Availability: ...

    def latex_from_image(self, image: Any) -> str: ...

    def unload(self) -> None: ...


def _crop(image: Any, bbox: Rect | None, padding: int = CROP_PADDING_PX) -> Any:
    """Кроп строки с небольшим запасом; без бокса — изображение целиком."""
    if bbox is None:
        return image
    left = max(0, bbox.x - padding)
    top = max(0, bbox.y - padding)
    right = min(image.width, bbox.x + bbox.width + padding)
    bottom = min(image.height, bbox.y + bbox.height + padding)
    if right <= left or bottom <= top:
        return image
    return image.crop((left, top, right, bottom))


class HybridBackend:
    """Бэкенд по умолчанию: печатный текст + печатные формулы."""

    name = "hybrid"

    def __init__(
        self,
        text_backend: Any | None = None,
        formula_backend: Any | None = None,
        routing: RoutingConfig = DEFAULT_ROUTING,
    ) -> None:
        self.text_backend = text_backend if text_backend is not None else PaddleTextBackend()
        self.formula_backend = (
            formula_backend if formula_backend is not None else Pix2TexBackend()
        )
        self.routing = routing

    def check_availability(self) -> Availability:
        """Доступен, только если доступны оба вложенных бэкенда."""
        reasons: list[str] = []
        for backend in (self.text_backend, self.formula_backend):
            availability = backend.check_availability()
            if not availability.available:
                reasons.append(f"{getattr(backend, 'name', backend)}: {availability.reason}")
        if reasons:
            return Availability(False, "; ".join(reasons))
        return Availability(True)

    def recognize(self, image_path: Path) -> Sequence[OcrFragment]:
        """Строки слайда: текстом либо формулой по решению роутера.

        Строки, опознанные как рукописные (D5, категория [3]), выбрасываются
        целиком: ни текстом, ни формулой они в результат не идут, доступ к ним
        даёт обязательная ссылка на PNG слайда.
        """
        lines: list[OcrFragment] = []
        dropped = 0
        for fragment in self.text_backend.recognize(image_path):
            if not looks_like_handwriting(fragment.text, fragment.confidence, self.routing):
                lines.append(fragment)
            elif _alnum_count(fragment.text) >= MIN_DROPPED_ALNUM:
                # Отброшенное не теряется молча: сборка пометит слайд
                # ненадёжным. Одиночный символ — не потерянная пометка, а
                # иконка или картинка, прочитанная как буква: на записи
                # wr_20261005_1350 значок логотипа («<», 0.27) делал
                # ненадёжной половину слайдов.
                dropped += 1
        if not lines:
            return RecognizedFragments((), dropped)

        formula_indexes = [
            index
            for index, fragment in enumerate(lines)
            if route_line(fragment.text, fragment.confidence, self.routing) == "formula"
        ]
        if not formula_indexes:
            return RecognizedFragments(lines, dropped)

        from PIL import Image

        result = list(lines)
        with Image.open(image_path) as raw:
            page = raw.convert("RGB")
        for index in formula_indexes:
            fragment = lines[index]
            latex = self.formula_backend.latex_from_image(_crop(page, fragment.bbox))
            markdown = latex_to_markdown(latex)
            if not markdown:
                continue  # формулу не прочитали — оставляем строку как текст
            confidence = min(DEFAULT_FORMULA_CONFIDENCE, fragment.confidence)
            # Строка, ушедшая в pix2tex только по низкой уверенности, сама по
            # себе формулой не выглядит — результат обязан нести пометку
            # «сверьтесь с изображением», а не выдавать себя за надёжный.
            routed_by_confidence = not looks_like_formula(fragment.text, self.routing)
            result[index] = OcrFragment(
                text=markdown,
                kind="formula",
                confidence=confidence,
                bbox=fragment.bbox,
                low_confidence=fragment.low_confidence or routed_by_confidence,
            )
        return RecognizedFragments(result, dropped)

    def unload(self) -> None:
        """Выгрузить обе модели (design D7)."""
        self.text_backend.unload()
        self.formula_backend.unload()
