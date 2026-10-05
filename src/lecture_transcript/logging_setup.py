"""Настройка логирования пайплайна.

Формат ориентирован на чтение прогона глазами: видно, какая стадия идёт и
сколько она заняла (design D7 — стадии последовательные, тайминг по каждой
нужен для сверки с бюджетом ~15 минут).
"""

from __future__ import annotations

import logging
import sys
import time
from contextlib import contextmanager
from typing import Iterator

__all__ = ["LOG_LEVELS", "setup_logging", "stage_timer", "format_duration"]

LOG_LEVELS: tuple[str, ...] = ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")

_FORMAT = "%(asctime)s %(levelname)-7s %(name)s: %(message)s"
_DATE_FORMAT = "%H:%M:%S"
_PACKAGE_LOGGER = "lecture_transcript"


def setup_logging(level: str = "INFO", *, stream=None) -> None:
    """Настроить корневой логгер. Повторный вызов переустанавливает handler."""
    resolved = getattr(logging, level.upper(), None)
    if not isinstance(resolved, int):
        raise ValueError(
            f"неизвестный уровень логирования {level!r}; "
            f"доступны: {', '.join(LOG_LEVELS)}"
        )
    handler = logging.StreamHandler(stream if stream is not None else sys.stderr)
    handler.setFormatter(logging.Formatter(_FORMAT, datefmt=_DATE_FORMAT))
    root = logging.getLogger()
    for existing in list(root.handlers):
        root.removeHandler(existing)
    root.addHandler(handler)
    root.setLevel(resolved)
    # Уровень пакета задаётся отдельно от корня: библиотеки моделей меняют
    # уровень корневого логгера при импорте (`import paddle` ставит WARNING,
    # pix2tex — CRITICAL), и логи стадий после OCR пропадали.
    logging.getLogger(_PACKAGE_LOGGER).setLevel(resolved)


def format_duration(seconds: float) -> str:
    """Человекочитаемая длительность: `12.3 с` / `2 мин 05 с`."""
    if seconds < 60:
        return f"{seconds:.1f} с"
    minutes, rest = divmod(int(seconds), 60)
    return f"{minutes} мин {rest:02d} с"


@contextmanager
def stage_timer(name: str, logger: logging.Logger | None = None) -> Iterator[None]:
    """Замерить и залогировать длительность стадии."""
    log = logger or logging.getLogger("lecture_transcript.pipeline")
    log.info("стадия %s: старт", name)
    started = time.perf_counter()
    try:
        yield
    except BaseException:
        elapsed = time.perf_counter() - started
        log.error("стадия %s: прервана через %s", name, format_duration(elapsed))
        raise
    else:
        elapsed = time.perf_counter() - started
        log.info("стадия %s: готово за %s", name, format_duration(elapsed))
