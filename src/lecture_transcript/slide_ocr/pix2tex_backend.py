"""Бэкенд распознавания печатных формул в LaTeX — pix2tex (задача 4.4).

Импорт `pix2tex` ленивый. Модель не даёт собственной оценки уверенности,
поэтому формульным фрагментам присваивается `DEFAULT_FORMULA_CONFIDENCE`,
ограниченная сверху уверенностью исходной строки текстового OCR
(см. `hybrid_backend`): размытая строка не может стать «уверенной формулой».

Веса. Сам pix2tex качает чекпойнты внутрь своего пакета
(`site-packages/pix2tex/model/checkpoints`), то есть в слой контейнера: при
`docker compose run --rm` они пропадают вместе с контейнером, и офлайн-прогон
остаётся без весов. Поэтому бэкенд держит веса в кэше моделей
(`$XDG_CACHE_HOME/pix2tex`, в образе — том `/cache`), сам докачивает их туда
и передаёт путь к чекпойнту в `LatexOCR`.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import re
import urllib.request
from pathlib import Path
from typing import Any, Sequence

from ..contracts import Availability, OcrFragment, Rect

#: Уверенность формульного фрагмента по умолчанию — pix2tex своей не даёт
#: (design D5: точность на печатных формулах ~85%).
DEFAULT_FORMULA_CONFIDENCE = 0.85

_INSTALL_HINT = (
    "установите pix2tex: pip install 'pix2tex[gui]' или pip install pix2tex "
    "(веса ~100 МБ скачиваются один раз при доступной сети)"
)


#: Чекпойнты pix2tex: те же файлы и тот же релиз, что качает сама библиотека
#: (`pix2tex.model.checkpoints.get_latest_checkpoint`, тег зашит как v0.0.1).
_WEIGHTS_URL = "https://github.com/lukas-blecher/LaTeX-OCR/releases/download/v0.0.1/{name}"
_WEIGHTS_FILES = ("weights.pth", "image_resizer.pth")


def cache_weights_dir() -> Path:
    """Каталог весов pix2tex в кэше моделей (в образе — `/cache/pix2tex`)."""
    base = os.environ.get("XDG_CACHE_HOME") or str(Path.home() / ".cache")
    return Path(base) / "pix2tex"


def package_weights_dirs() -> tuple[Path, ...]:
    """Каталоги, куда pix2tex кладёт чекпойнты сам (внутрь пакета)."""
    try:
        spec = importlib.util.find_spec("pix2tex")
    except Exception:  # pragma: no cover — битая установка
        return ()
    locations = list(getattr(spec, "submodule_search_locations", None) or []) if spec else []
    return tuple(Path(location) / "model" / "checkpoints" for location in locations)


def local_weights_dirs() -> tuple[Path, ...]:
    """Где искать веса: сначала кэш моделей, затем каталог пакета.

    Каталог пакета оставлен для установок вне образа, где pix2tex уже
    скачал веса себе сам.
    """
    return (cache_weights_dir(), *package_weights_dirs())


def find_weights_dir() -> Path | None:
    """Первый каталог с основным чекпойнтом — проверка без сети (задача 4.9)."""
    for path in local_weights_dirs():
        try:
            if (path / "weights.pth").is_file():
                return path
        except OSError:  # pragma: no cover — недоступный каталог
            continue
    return None


def has_local_weights() -> bool:
    """Есть ли локальные веса — проверка без сети (задача 4.9)."""
    return find_weights_dir() is not None


def download_weights(target: Path) -> Path:
    """Скачать чекпойнты в `target`. Файл появляется под своим именем только
    целиком: оборванная загрузка не выдаёт себя за готовые веса."""
    target.mkdir(parents=True, exist_ok=True)
    for name in _WEIGHTS_FILES:
        final = target / name
        if final.is_file():
            continue
        partial = final.with_name(name + ".part")
        with urllib.request.urlopen(_WEIGHTS_URL.format(name=name)) as source, partial.open(
            "wb"
        ) as output:
            while chunk := source.read(1 << 20):
                output.write(chunk)
        os.replace(partial, final)
    return target


def _is_blank(image: Any) -> bool:
    """Однотонная картинка: у всех каналов минимум совпадает с максимумом.

    pix2tex на такой падает внутри своей нормализации (деление на
    `max - min == 0` -> NaN -> пустой кроп -> `cv2.cvtColor ... !_src.empty()`).
    Формулы на однотонном кропе нет, распознавать нечего.
    """
    extrema = image.getextrema()
    if extrema and not isinstance(extrema[0], tuple):  # одноканальное изображение
        extrema = (extrema,)
    return all(low == high for low, high in extrema)


def latex_to_markdown(latex: str) -> str:
    """LaTeX -> Markdown-фрагмент `$...$` согласно контракту `OcrFragment`."""
    text = (latex or "").strip()
    text = re.sub(r"^\$+|\$+$", "", text).strip()
    text = re.sub(r"^\\\[|\\\]$", "", text).strip()
    text = re.sub(r"^\\\(|\\\)$", "", text).strip()
    text = " ".join(text.split())
    if not text:
        return ""
    return f"${text}$"


class Pix2TexBackend:
    """Печатные формулы -> LaTeX."""

    name = "pix2tex"

    def __init__(self) -> None:
        self._model: Any | None = None

    def check_availability(self) -> Availability:
        """Проверка до прогона: установлен ли pix2tex и лежат ли веса рядом."""
        try:
            if importlib.util.find_spec("pix2tex") is None:
                return Availability(False, f"библиотека 'pix2tex' не установлена — {_INSTALL_HINT}")
            if importlib.util.find_spec("torch") is None:
                return Availability(False, "библиотека 'torch' не установлена — нужна для pix2tex")
        except Exception as exc:  # битая установка: find_spec бросает наружу
            return Availability(False, f"установка pix2tex повреждена: {exc}")
        if not has_local_weights():
            return Availability(
                False,
                "веса pix2tex не найдены локально — скачайте их один раз "
                f"при доступной сети: {_INSTALL_HINT}",
            )
        return Availability(True)

    def _load(self) -> Any:
        if self._model is None:
            from .offline import enforce_offline

            enforce_offline()
            from pix2tex.cli import LatexOCR  # ленивый импорт

            # Без весов в кэше качаем их туда сами: собственная загрузка
            # pix2tex положила бы их в пакет, мимо тома /cache. В боевом
            # прогоне сюда не дойдёт — check_availability() отвергнет бэкенд.
            weights_dir = find_weights_dir() or download_weights(cache_weights_dir())
            # Остальные параметры — умолчания LatexOCR(); относительный путь
            # к конфигу разрешается от каталога модели внутри пакета.
            arguments = argparse.Namespace(
                config="settings/config.yaml",
                checkpoint=str(weights_dir / "weights.pth"),
                no_cuda=True,
                no_resize=False,
            )
            self._model = LatexOCR(arguments)
        return self._model

    def latex_from_image(self, image: Any) -> str:
        """LaTeX по PIL-изображению (кроп строки или весь слайд)."""
        if _is_blank(image):
            return ""
        model = self._load()
        return str(model(image))

    def recognize(self, image_path: Path) -> Sequence[OcrFragment]:
        """Распознать изображение целиком как одну формулу."""
        from PIL import Image  # входит в зависимости проекта

        with Image.open(image_path) as image:
            image = image.convert("RGB")
            latex = self.latex_from_image(image)
        markdown = latex_to_markdown(latex)
        if not markdown:
            return ()
        return (
            OcrFragment(
                text=markdown,
                kind="formula",
                confidence=DEFAULT_FORMULA_CONFIDENCE,
                bbox=Rect(0, 0, image.width, image.height),
            ),
        )

    def unload(self) -> None:
        """Выгрузить модель (design D7)."""
        self._model = None
