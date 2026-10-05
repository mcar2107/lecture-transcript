"""Восстановление пунктуации и регистра как отдельная стадия (D6).

Стадия включается только для бэкендов, объявивших ``provides_punctuation =
False``. Ключевое ограничение D6: **таймкоды берутся из ASR и не
пересчитываются**. Модель пунктуации возвращает голый текст без времён,
поэтому её результат обязан быть выровнен обратно на исходные слова.

Правило выравнивания
--------------------

Пунктуатор имеет право менять только регистр и знаки препинания — поток
букв и цифр он менять не должен. На этом и строится выравнивание:

1. Из исходных слов и из пунктуированного текста извлекается «ядро» —
   поток символов ``isalnum()``, приведённый к нижнему регистру, с ``ё`` -> ``е``.
2. Потоки сравниваются. Не совпали — значит модель добавила, съела или
   переписала слова; пунктуация **не применяется**, выдаётся предупреждение,
   слова остаются исходными. Это осознанный выбор: лучше текст без
   пунктуации, чем поехавшая привязка ко времени.
3. Совпали — каждое исходное слово забирает из пунктуированного текста ровно
   столько символов ядра, сколько было у него, и получает подстроку от своего
   первого до последнего символа ядра, расширенную примыкающими знаками
   препинания (справа — точки, запятые; слева — открывающие кавычки и скобки).

Правило устойчиво к тому, что модель склеила или разбила слова пробелами:
состав и количество слов, а с ними и все ``start_s`` / ``end_s``, сохраняются
по построению.
"""

from __future__ import annotations

import importlib.util
import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from typing import Any, Protocol, runtime_checkable

from ..contracts import Availability, BackendUnavailableError, Word
from .offline import hf_model_cached, is_offline

logger = logging.getLogger(__name__)

__all__ = [
    "Punctuator",
    "PunctuationResult",
    "align_punctuated",
    "restore_punctuation",
    "normalize_core",
    "register_punctuator",
    "get_punctuator",
    "check_punctuator",
    "list_punctuator_names",
    "SbertPuncCaseRu",
    "DEFAULT_PUNCTUATOR_NAME",
    "INSTALL_HINT",
    "NO_PUNCTUATION_WARNING",
    "PARTIAL_PUNCTUATION_PREFIX",
    "partial_punctuation_warning",
]

#: Сколько слов уходит в модель за один вызов — контекст модели ограничен.
DEFAULT_CHUNK_WORDS = 64


@runtime_checkable
class Punctuator(Protocol):
    """Контракт модели пунктуации: голый текст -> текст со знаками и регистром."""

    name: str

    def check_availability(self) -> Availability: ...

    def restore(self, text: str) -> str: ...

    def unload(self) -> None:
        """Выгрузить модель и освободить память (D7)."""


@dataclass(frozen=True)
class PunctuationResult:
    """Результат стадии: слова с теми же таймкодами и отчёт о применении."""

    words: tuple[Word, ...]
    applied: bool
    warnings: tuple[str, ...] = ()


# --------------------------------------------------------------------------
# Выравнивание
# --------------------------------------------------------------------------


def normalize_core(text: str) -> str:
    """Ядро строки: только буквы и цифры, нижний регистр, ``ё`` -> ``е``."""
    return "".join(
        ("е" if ch == "ё" else ch)
        for ch in (c.lower() for c in text)
        if ch.isalnum()
    )


def align_punctuated(
    words: Sequence[Word], punctuated_text: str
) -> tuple[list[Word], list[str]]:
    """Наложить пунктуированный текст на исходные слова, сохранив их таймкоды.

    Возвращает (слова, предупреждения). При расхождении потоков символов
    возвращаются исходные слова без изменений.
    """
    original = list(words)
    if not original:
        return [], []

    cores = [normalize_core(word.text) for word in original]
    stream = [(idx, ch) for idx, ch in enumerate(_normalized_chars(punctuated_text)) if ch]

    expected = "".join(cores)
    got = "".join(ch for _idx, ch in stream)
    if expected != got:
        return original, [
            "пунктуатор изменил состав слов "
            f"(символов было {len(expected)}, стало {len(got)}) — "
            "пунктуация не применена, таймкоды и текст оставлены как есть"
        ]

    aligned: list[Word] = []
    pointer = 0
    consumed_until = 0
    last: tuple[int, int] | None = None  # (индекс слова, начало его подстроки)
    for word, core in zip(original, cores, strict=True):
        if not core:
            aligned.append(word)
            continue
        span_start = stream[pointer][0]
        span_end = stream[pointer + len(core) - 1][0] + 1
        pointer += len(core)
        span_start = _extend_left(punctuated_text, span_start, consumed_until)
        span_end = _extend_right(punctuated_text, span_end)
        consumed_until = span_end
        text = punctuated_text[span_start:span_end].strip()
        last = (len(aligned), span_start)
        aligned.append(
            Word(
                text=text or word.text,
                start_s=word.start_s,
                end_s=word.end_s,
                confidence=word.confidence,
            )
        )
    # Хвост из одних знаков (например, открывающая кавычка в самом конце)
    # правому слову не достался — отдаём его последнему, а не теряем.
    if last is not None and punctuated_text[consumed_until:].strip():
        index, span_start = last
        aligned[index] = replace(
            aligned[index], text=punctuated_text[span_start:].strip()
        )
    return aligned, []


def _normalized_chars(text: str) -> list[str]:
    """Посимвольная нормализация с сохранением позиций: не-ядро -> пустая строка."""
    result: list[str] = []
    for ch in text:
        lowered = ch.lower()
        if lowered == "ё":
            lowered = "е"
        result.append(lowered if lowered.isalnum() else "")
    return result


#: Открывающие знаки: отдельно стоящая группа с таким знаком принадлежит
#: СЛЕДУЮЩЕМУ слову («Он « да »», «См. ( два )»). Всё прочее отдельно стоящее
#: (тире, многоточие, закрывающие кавычки и скобки, «%») — предыдущему.
_OPENING = frozenset("«„“\"'([{‹")


def _is_mark(ch: str) -> bool:
    return not ch.isspace() and not ch.isalnum()


def _extend_right(text: str, end: int) -> int:
    """Прихватить знаки препинания справа — примыкающие и отдельно стоящие.

    Знак, окружённый пробелами, раньше не доставался ни левому слову (стоп на
    пробеле), ни правому (там тоже стоп на пробеле) — и молча пропадал из
    результата. Теперь каждая отдельно стоящая группа знаков без открывающих
    достаётся левому слову; с открывающими — остаётся правому.
    """
    while True:
        while end < len(text) and _is_mark(text[end]):
            end += 1
        probe = end
        while probe < len(text) and text[probe].isspace():
            probe += 1
        if probe == end or probe == len(text):
            return end
        group_end = probe
        while group_end < len(text) and _is_mark(text[group_end]):
            group_end += 1
        if group_end == probe:  # дальше начинается слово
            return end
        if group_end < len(text) and not text[group_end].isspace():
            return end  # группа примыкает к следующему слову — она его
        if any(ch in _OPENING for ch in text[probe:group_end]):
            return end  # открывающий знак — следующему слову
        end = group_end


def _extend_left(text: str, start: int, floor: int) -> int:
    """Прихватить знаки слева — примыкающие и все отдельно стоящие до ``floor``.

    До ``floor`` предыдущее слово уже забрало всё своё, поэтому оставшиеся
    между словами группы (открывающие кавычки, скобки, тире реплики) — наши.
    """
    while True:
        while start > floor and _is_mark(text[start - 1]):
            start -= 1
        probe = start
        while probe > floor and text[probe - 1].isspace():
            probe -= 1
        if probe == start or probe == floor:
            return start
        group_start = probe
        while group_start > floor and _is_mark(text[group_start - 1]):
            group_start -= 1
        if group_start == probe:  # слева слово
            return start
        start = group_start


# --------------------------------------------------------------------------
# Стадия
# --------------------------------------------------------------------------


def restore_punctuation(
    words: Sequence[Word],
    punctuator: Punctuator,
    *,
    chunk_words: int = DEFAULT_CHUNK_WORDS,
) -> PunctuationResult:
    """Прогнать слова через модель пунктуации, сохранив их таймкоды.

    Слова режутся на куски по ``chunk_words``, каждый кусок пунктуируется и
    выравнивается независимо: ошибка модели на одном куске не портит остальные.
    """
    original = list(words)
    if not original:
        return PunctuationResult(words=(), applied=False)

    availability = punctuator.check_availability()
    if not availability.available:
        raise BackendUnavailableError(
            f"пунктуатор {punctuator.name!r} недоступен: {availability.reason}"
        )

    result: list[Word] = []
    warnings: list[str] = []
    step = max(1, chunk_words)
    total = 0
    failed = 0
    for start in range(0, len(original), step):
        chunk = original[start : start + step]
        total += 1
        raw = " ".join(word.text for word in chunk)
        try:
            punctuated = punctuator.restore(raw)
        except Exception as exc:  # noqa: BLE001 — стадия не должна ронять прогон
            message = f"пунктуатор {punctuator.name!r} упал на куске: {exc!r}"
            logger.warning(message)
            warnings.append(message)
            result.extend(chunk)
            failed += 1
            continue
        if punctuated.strip() == raw.strip():
            # Модель вернула вход как есть: знаков и регистра не прибавилось,
            # считать кусок «пунктуированным» — самообман.
            warnings.append(
                f"пунктуатор {punctuator.name!r} вернул текст без изменений"
            )
            result.extend(chunk)
            failed += 1
            continue
        aligned, chunk_warnings = align_punctuated(chunk, punctuated)
        if chunk_warnings:
            for message in chunk_warnings:
                logger.warning("%s: %s", punctuator.name, message)
            warnings.extend(chunk_warnings)
            failed += 1
        result.extend(aligned)

    applied = failed < total
    if failed:
        # Отказ громкий независимо от масштаба: и полный, и частичный.
        marker = (
            NO_PUNCTUATION_WARNING
            if not applied
            else partial_punctuation_warning(failed, total)
        )
        logger.error("%s", marker)
        warnings.append(marker)

    return PunctuationResult(
        words=tuple(result), applied=applied, warnings=tuple(dict.fromkeys(warnings))
    )


#: Единый маркер того, что пунктуация не восстановлена. Ставится при любой
#: причине (нет пакета, модель упала, выравнивание отвергло результат, стадия
#: выключена), чтобы отказ нельзя было проглядеть среди частных предупреждений.
NO_PUNCTUATION_WARNING = (
    "ПУНКТУАЦИЯ НЕ ВОССТАНОВЛЕНА: итоговый текст без знаков препинания "
    "и без заглавных букв — требование «Пунктуация и регистр» не выполнено"
)

#: Начало маркера частичного отказа (часть кусков осталась без пунктуации).
PARTIAL_PUNCTUATION_PREFIX = "ПУНКТУАЦИЯ ВОССТАНОВЛЕНА ЧАСТИЧНО"


def partial_punctuation_warning(failed: int, total: int) -> str:
    """Маркер частичного отказа с числом кусков — оно не теряется при дедупликации."""
    return (
        f"{PARTIAL_PUNCTUATION_PREFIX}: {failed} из {total} кусков текста "
        "остались без знаков препинания и заглавных букв — требование "
        "«Пунктуация и регистр» выполнено не полностью"
    )


# --------------------------------------------------------------------------
# Реестр пунктуаторов
# --------------------------------------------------------------------------

DEFAULT_PUNCTUATOR_NAME = "sbert_punc_case_ru"

_PUNCTUATORS: dict[str, Callable[[], Punctuator]] = {}
_INSTANCES: dict[str, Punctuator] = {}


def register_punctuator(
    name: str, factory: Callable[[], Punctuator], *, replace: bool = False
) -> None:
    """Зарегистрировать модель пунктуации под именем."""
    if not replace and name in _PUNCTUATORS:
        raise ValueError(f"пунктуатор {name!r} уже зарегистрирован")
    _PUNCTUATORS[name] = factory
    _INSTANCES.pop(name, None)


def list_punctuator_names() -> list[str]:
    return sorted(_PUNCTUATORS)


def get_punctuator(name: str = DEFAULT_PUNCTUATOR_NAME) -> Punctuator:
    """Вернуть экземпляр пунктуатора по имени (кэшируется)."""
    if name not in _PUNCTUATORS:
        known = ", ".join(list_punctuator_names()) or "(реестр пуст)"
        raise BackendUnavailableError(
            f"пунктуатор {name!r} не зарегистрирован; доступные: {known}"
        )
    if name not in _INSTANCES:
        _INSTANCES[name] = _PUNCTUATORS[name]()
    return _INSTANCES[name]


def check_punctuator(name: str = DEFAULT_PUNCTUATOR_NAME) -> Availability:
    """Проверить доступность пунктуатора до начала прогона."""
    try:
        punctuator = get_punctuator(name)
    except BackendUnavailableError as exc:
        return Availability(False, str(exc))
    except Exception as exc:  # noqa: BLE001 — упавшая фабрика = недоступность
        return Availability(False, f"не удалось создать пунктуатор {name!r}: {exc!r}")
    try:
        return punctuator.check_availability()
    except Exception as exc:  # noqa: BLE001
        return Availability(False, f"проверка доступности {name!r} упала: {exc!r}")


#: Как поставить пакет пунктуатора. Важно: на PyPI пакета
#: ``sbert_punc_case_ru`` НЕТ (`pip install sbert_punc_case_ru` даёт 404), и
#: репозитория github.com/kontur-ai/sbert_punc_case_ru тоже больше нет.
#: Код лежит внутри репозитория модели на Hugging Face, там же setup.py.
INSTALL_HINT = (
    "пакета sbert_punc_case_ru нет на PyPI, ставится только из репозитория "
    "модели: `pip install torch transformers` и "
    "`GIT_LFS_SKIP_SMUDGE=1 pip install "
    "'sbert_punc_case_ru @ git+https://huggingface.co/kontur-ai/"
    "sbert_punc_case_ru@f778dc6c63bb0ec235a220488862509810e54583'` "
    "(веса модели пунктуации ~0.7 ГБ качаются при первой загрузке)"
)

#: Репозиторий весов; совпадает с ``sbert_punc_case_ru.sbertpunccase.MODEL_REPO``,
#: но берётся не оттуда: импорт пакета тянет transformers.
MODEL_REPO = "kontur-ai/sbert_punc_case_ru"


class SbertPuncCaseRu:
    """Модель восстановления пунктуации и регистра для русского текста.

    Тяжёлые зависимости (``transformers``, ``torch``) импортируются лениво;
    без них :meth:`check_availability` объясняет, чего не хватает, и стадия
    пунктуации корректно пропускается с предупреждением.
    """

    name = DEFAULT_PUNCTUATOR_NAME

    def __init__(self, device: str | None = None) -> None:
        # Идентификатор модели пакет не принимает: он зашит в
        # ``sbert_punc_case_ru.sbertpunccase.MODEL_REPO``. Настраивается только
        # устройство (``None`` — cuda, если доступна, иначе cpu).
        self._device = device
        self._model: Any | None = None

    def check_availability(self) -> Availability:
        for package in ("torch", "transformers", "sbert_punc_case_ru"):
            if importlib.util.find_spec(package) is None:
                return Availability(
                    False,
                    f"не установлен пакет {package!r}: {INSTALL_HINT}",
                )
        # В офлайне transformers не скачает веса, и restore() упадёт на каждом
        # куске — лучше сказать об этом до прогона.
        if is_offline() and not hf_model_cached(MODEL_REPO):
            return Availability(
                False,
                f"веса {MODEL_REPO} не скачаны, а офлайн-режим включён — "
                "прогрейте кэш: python -m lecture_transcript.warmup",
            )
        return Availability(True)

    def restore(self, text: str) -> str:
        if not text.strip():
            return text
        if self._model is None:
            self._model = self._load()
        return str(self._model.punctuate(text))

    def _load(self) -> Any:
        """Создать модель по реальному API пакета (коммит f778dc6).

        ``class SbertPuncCase(nn.Module)``: ``__init__(self)`` без аргументов —
        токенизатор и веса ``kontur-ai/sbert_punc_case_ru`` грузятся внутри
        конструктора через ``transformers``; перенос на устройство —
        ``.to(device)``; пунктуация — ``.punctuate(text) -> str``.
        Метода ``from_pretrained`` у класса НЕТ.
        """
        from sbert_punc_case_ru import SbertPuncCase  # noqa: PLC0415 — ленивый импорт

        model = SbertPuncCase()
        device = self._resolve_device()
        if device is not None:
            model = model.to(device)
        return model

    def _resolve_device(self) -> str | None:
        if self._device is not None:
            return self._device
        if importlib.util.find_spec("torch") is None:
            return None
        import torch  # noqa: PLC0415 — ленивый импорт

        return "cuda" if torch.cuda.is_available() else None

    def unload(self) -> None:
        """Освободить память под следующую стадию (D7)."""
        if self._model is None:
            return
        self._model = None
        if importlib.util.find_spec("torch") is not None:
            import torch  # noqa: PLC0415

            if torch.cuda.is_available():  # pragma: no cover — требует CUDA
                torch.cuda.empty_cache()


register_punctuator(DEFAULT_PUNCTUATOR_NAME, SbertPuncCaseRu)
