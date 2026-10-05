"""Прогрев кэша моделей: `python -m lecture_transcript.warmup`.

Образ по умолчанию работает с `HF_HUB_OFFLINE=1`: веса моделей из сети не
тянутся, их отсутствие даёт честную ошибку. Значит, веса нужно один раз
скачать заранее — это и делает прогрев:

1. снимает офлайн-режим на время прогрева (переключатели `*_OFFLINE=0`);
2. по очереди загружает модели бэкендов их же собственными загрузчиками и
   сразу выгружает каждую (design D7: одна модель в памяти за раз);
3. возвращает окружение как было и в явно включённом офлайн-режиме
   вызывает `check_availability()` каждого прогретого бэкенда;
4. печатает сводку; код 1, если хоть одна запрошенная модель не загрузилась
   или не прошла проверку.

Публичного «загрузить модель без распознавания» пакеты не дают, поэтому
модель поднимается минимальным фиктивным вызовом на синтетическом входе:
PNG с короткой формулой для OCR, секунда тишины для ASR и VAD, два слова
для пунктуатора. Картинка намеренно не однотонная: pix2tex на однотонной
падает в собственной нормализации, а бэкенд такую и вовсе не отдаёт модели.

Каталоги кэша моделей warmup не выбирает: их задаёт окружение образа
(`HF_HOME`, `TORCH_HOME`, `PADDLE_PDX_CACHE_HOME`, `XDG_CACHE_HOME`).
"""

from __future__ import annotations

import argparse
import ast
import logging
import os
import sys
import tempfile
import time
import wave
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping, Sequence

from .config import ConfigError, PipelineConfig, load_config
from .contracts import AudioArtifact, Availability, PipelineError, SpeechInterval
from .logging_setup import LOG_LEVELS, format_duration, setup_logging

__all__ = [
    "CACHE_ENV_VARS",
    "WarmTarget",
    "TargetResult",
    "WarmupError",
    "offline_env",
    "online_window",
    "offline_window",
    "plan_targets",
    "run_warmup",
    "format_summary",
    "build_parser",
    "main",
]

LOGGER = logging.getLogger("lecture_transcript.warmup")

#: Переменные, которыми образ задаёт каталоги кэша моделей (docker/Dockerfile).
CACHE_ENV_VARS: tuple[str, ...] = (
    "XDG_CACHE_HOME",
    "HF_HOME",
    "TORCH_HOME",
    "PADDLE_PDX_CACHE_HOME",
)

#: OCR-бэкенды, которые не греются без явной просьбы: VLM весит ~6 ГБ.
HEAVY_OCR_BACKENDS: tuple[str, ...] = ("vlm",)

#: Откуда берётся список офлайн-переменных — ровно те модули, что их задают.
_OFFLINE_SOURCES: tuple[str, ...] = (
    "slide_ocr/offline.py",
    "speech_transcription/offline.py",
)

_SAMPLE_RATE = 16000


class WarmupError(PipelineError):
    """Прогрев не может начаться: не найден список офлайн-переменных и т. п."""


# --------------------------------------------------------------------------
# Офлайн-режим
# --------------------------------------------------------------------------


def offline_env() -> dict[str, str]:
    """Объединённый список офлайн-переменных из `offline.py` пакетов стадий.

    Читается разбором исходника, а НЕ импортом: импорт пакета стадии может
    потянуть библиотеки хаба, а `huggingface_hub` фиксирует `HF_HUB_OFFLINE`
    в момент своего импорта — снимать офлайн после этого было бы бесполезно.
    """
    base = Path(__file__).resolve().parent
    merged: dict[str, str] = {}
    for relative in _OFFLINE_SOURCES:
        path = base / relative
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except OSError as exc:
            raise WarmupError(f"не читается {path}: {exc}") from exc
        found: Any = None
        for node in tree.body:
            if isinstance(node, ast.AnnAssign):
                target, value = node.target, node.value
            elif isinstance(node, ast.Assign) and len(node.targets) == 1:
                target, value = node.targets[0], node.value
            else:
                continue
            if isinstance(target, ast.Name) and target.id == "OFFLINE_ENV" and value:
                found = ast.literal_eval(value)
        if not isinstance(found, dict):
            raise WarmupError(f"{path}: не найден словарь-литерал OFFLINE_ENV")
        merged.update({str(k): str(v) for k, v in found.items()})
    return merged


def _is_offline_switch(name: str) -> bool:
    """Переключатель офлайна, а не гигиена (телеметрия, проверка обновлений)."""
    return name.endswith("_OFFLINE")


#: Значение переключателя `*_OFFLINE` в окне прогрева. Именно "0", а не
#: удалённая переменная: бэкенды при загрузке зовут `enforce_offline()`,
#: который заполняет ПУСТЫЕ переключатели единицей. Без явного "0" первый же
#: OCR-бэкенд включил бы офлайн обратно, и все модели после него грелись бы
#: уже без сети (а huggingface_hub ещё и запомнил бы это на импорте).
_ONLINE_VALUE = "0"


def _snapshot(keys: Sequence[str]) -> dict[str, str | None]:
    return {key: os.environ.get(key) for key in keys}


def _restore(saved: Mapping[str, str | None]) -> None:
    for key, value in saved.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value


@contextmanager
def online_window(env_spec: Mapping[str, str]) -> Iterator[None]:
    """Окно прогрева: переключатели `*_OFFLINE` явно выключены ("0"), запреты
    телеметрии и проверки обновлений включены. На выходе окружение
    восстанавливается ровно таким, каким было, — при любом исходе."""
    saved = _snapshot(list(env_spec))
    try:
        for key, value in env_spec.items():
            os.environ[key] = _ONLINE_VALUE if _is_offline_switch(key) else value
        yield
    finally:
        _restore(saved)


@contextmanager
def offline_window(env_spec: Mapping[str, str]) -> Iterator[None]:
    """Явно включённый офлайн-режим — для финальной проверки доступности."""
    saved = _snapshot(list(env_spec))
    try:
        os.environ.update(env_spec)
        yield
    finally:
        _restore(saved)


# --------------------------------------------------------------------------
# Цели прогрева
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class WarmInputs:
    """Синтетические входы для фиктивных вызовов."""

    image_path: Path
    audio: AudioArtifact

    def image(self) -> Any:
        from PIL import Image  # noqa: PLC0415

        with Image.open(self.image_path) as raw:
            return raw.convert("RGB")


@dataclass(frozen=True)
class WarmTarget:
    """Одна модель к прогреву.

    `load` отдаёт объект бэкенда (дёшево, без весов), `warm` минимальным
    вызовом заставляет его загрузить веса. У объекта обязаны быть
    `check_availability()` и `unload()` — контракт бэкендов.
    """

    kind: str
    name: str
    load: Callable[[], Any]
    warm: Callable[[Any, WarmInputs], None]
    note: str = ""

    @property
    def label(self) -> str:
        return f"{self.kind}:{self.name}"


@dataclass
class TargetResult:
    target: WarmTarget
    loaded: bool = False
    error: str = ""
    seconds: float = 0.0
    available_offline: bool | None = None
    offline_reason: str = ""
    backend: Any = field(default=None, repr=False)

    @property
    def ok(self) -> bool:
        return self.loaded and self.available_offline is True


def _warm_ocr(backend: Any, inputs: WarmInputs) -> None:
    text = getattr(backend, "text_backend", None)
    formula = getattr(backend, "formula_backend", None)
    if text is not None and formula is not None:
        # Гибрид: роутер может не счесть пробную строку формульной, и тогда
        # pix2tex не загрузится — поэтому поднимаем обе части явно.
        text.recognize(inputs.image_path)
        formula.latex_from_image(inputs.image())
        return
    backend.recognize(inputs.image_path)


def _ocr_target(name: str) -> WarmTarget:
    def load() -> Any:
        from .slide_ocr import registry  # noqa: PLC0415 — после снятия офлайна

        return registry.get_backend(name)

    return WarmTarget(
        "ocr", name, load, _warm_ocr,
        note="recognize() PNG с формулой; у hybrid — обе части по отдельности",
    )


def _asr_target(name: str) -> WarmTarget:
    def load() -> Any:
        from .speech_transcription import registry  # noqa: PLC0415

        return registry.get_backend(registry.resolve_name(name))

    def warm(backend: Any, inputs: WarmInputs) -> None:
        # Непустой интервал обязателен: при пустом модель не вызывается.
        backend.transcribe(
            inputs.audio, [SpeechInterval(0.0, inputs.audio.duration_s)]
        )

    return WarmTarget("asr", name, load, warm, note="transcribe() секунды тишины")


class _SileroVad:
    """Держатель silero-vad: модель живёт внутри вызова, выгружать нечего."""

    name = "silero"

    def check_availability(self) -> Availability:
        from .speech_transcription import check_vad_backend  # noqa: PLC0415

        return check_vad_backend("silero")

    def unload(self) -> None:
        """Модель локальна внутри detect_speech и отпускается по выходу."""


def _vad_target() -> WarmTarget:
    def warm(_: Any, inputs: WarmInputs) -> None:
        from .speech_transcription import detect_speech  # noqa: PLC0415

        detect_speech(inputs.audio, None, backend="silero")

    return WarmTarget(
        "vad", "silero", _SileroVad, warm, note="detect_speech() секунды тишины"
    )


def _punctuation_target() -> WarmTarget:
    def load() -> Any:
        from .speech_transcription import get_punctuator  # noqa: PLC0415

        return get_punctuator()

    def warm(punctuator: Any, inputs: WarmInputs) -> None:
        # Не restore_punctuation(): стадия глотает ошибки модели (прогон не
        # должен падать из-за пунктуации), и несостоявшаяся загрузка весов
        # выглядела бы как успешный прогрев. Здесь отказ обязан быть громким.
        raw = "проверка связи"
        if punctuator.restore(raw).strip() == raw:
            raise RuntimeError("пунктуатор вернул текст без изменений — модель не отработала")

    return WarmTarget("punctuation", "default", load, warm, note="restore() двух слов")


def plan_targets(
    config: PipelineConfig, *, all_backends: bool = False, with_vlm: bool = False
) -> list[WarmTarget]:
    """Что греть: выбранное в конфиге либо всё зарегистрированное.

    Вызывается ВНУТРИ окна прогрева: для `--all` нужны реестры, а их импорт
    должен случиться уже после снятия офлайн-режима.
    """
    targets: list[WarmTarget] = []

    if all_backends:
        from .slide_ocr import registry as ocr_registry  # noqa: PLC0415
        from .speech_transcription import registry as asr_registry  # noqa: PLC0415

        ocr_names = [
            name
            for name in ocr_registry.list_backend_names()
            if name not in HEAVY_OCR_BACKENDS or with_vlm
        ]
        asr_names = list(asr_registry.list_backend_names())
    else:
        ocr_names = [config.slide_ocr.backend]
        if with_vlm:
            ocr_names += [n for n in HEAVY_OCR_BACKENDS if n not in ocr_names]
        asr_names = [config.speech_transcription.asr_backend]

    targets += [_ocr_target(name) for name in ocr_names]
    targets += [_asr_target(name) for name in asr_names]

    vad = config.speech_transcription.vad_backend
    if all_backends or vad == "silero":
        targets.append(_vad_target())
    elif vad == "auto":
        LOGGER.info(
            "vad_backend=auto: silero не греется; без его весов прогон уйдёт в energy-VAD"
        )

    if all_backends or config.speech_transcription.punctuation != "never":
        targets.append(_punctuation_target())
    return targets


# --------------------------------------------------------------------------
# Прогрев
# --------------------------------------------------------------------------


def _make_inputs(workdir: Path) -> WarmInputs:
    from PIL import Image, ImageDraw  # noqa: PLC0415

    workdir.mkdir(parents=True, exist_ok=True)
    image_path = workdir / "formula.png"
    image = Image.new("RGB", (160, 48), (255, 255, 255))
    # Встроенный растровый шрифт PIL: файлы шрифтов в образе не нужны.
    ImageDraw.Draw(image).text((8, 16), "x = a + b", fill=(0, 0, 0))
    image.save(image_path)

    wav_path = workdir / "silence.wav"
    with wave.open(str(wav_path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(_SAMPLE_RATE)
        wav.writeframes(b"\x00\x00" * _SAMPLE_RATE)
    audio = AudioArtifact(
        path=wav_path, sample_rate=_SAMPLE_RATE, duration_s=1.0, source_track_index=0
    )
    return WarmInputs(image_path=image_path, audio=audio)


def run_warmup(
    plan: Callable[[], Sequence[WarmTarget]],
    workdir: Path,
    *,
    env_spec: Mapping[str, str] | None = None,
) -> list[TargetResult]:
    """Прогреть цели по очереди и проверить их доступность в офлайне."""
    spec = offline_env() if env_spec is None else dict(env_spec)
    results: list[TargetResult] = []

    with online_window(spec):
        inputs = _make_inputs(workdir)
        for target in plan():
            result = TargetResult(target)
            results.append(result)
            LOGGER.info("прогрев %s: старт", target.label)
            started = time.perf_counter()
            backend: Any = None
            try:
                backend = target.load()
                target.warm(backend, inputs)
                result.loaded = True
                result.backend = backend
            except Exception as exc:  # noqa: BLE001 — один отказ не рушит остальных
                result.error = f"{type(exc).__name__}: {exc}"
                LOGGER.error("прогрев %s: не загрузилась — %s", target.label, result.error)
            finally:
                result.seconds = time.perf_counter() - started
                if backend is not None:
                    try:
                        backend.unload()  # D7: одна модель в памяти за раз
                    except Exception as exc:  # noqa: BLE001
                        LOGGER.warning("прогрев %s: unload() упал — %s", target.label, exc)
            if result.loaded:
                LOGGER.info(
                    "прогрев %s: готово за %s", target.label, format_duration(result.seconds)
                )

    # Окружение уже восстановлено; офлайн включается явно, даже если снаружи
    # было HF_HUB_OFFLINE=0 — проверяется ровно то, как пойдёт боевой прогон.
    with offline_window(spec):
        for result in results:
            if not result.loaded:
                continue
            try:
                availability = result.backend.check_availability()
            except Exception as exc:  # noqa: BLE001
                result.available_offline = False
                result.offline_reason = f"{type(exc).__name__}: {exc}"
                continue
            result.available_offline = bool(availability.available)
            result.offline_reason = availability.reason
    return results


def format_summary(results: Sequence[TargetResult]) -> str:
    lines = ["Сводка прогрева:"]
    if not results:
        lines.append("  нечего греть — ни одна модель не выбрана")
    for result in results:
        label = result.target.label
        if not result.loaded:
            lines.append(f"  FAIL  {label}: не загрузилась — {result.error}")
        elif result.available_offline is not True:
            reason = result.offline_reason or "check_availability() вернул available=False"
            lines.append(f"  FAIL  {label}: загрузилась, но недоступна в офлайне — {reason}")
        else:
            lines.append(f"  OK    {label} ({format_duration(result.seconds)})")
    lines.append("Каталоги кэша моделей:")
    for name in CACHE_ENV_VARS:
        lines.append(f"  {name}={os.environ.get(name) or '(не задан — умолчание библиотеки)'}")
    return "\n".join(lines)


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m lecture_transcript.warmup",
        description=(
            "Скачать веса моделей в кэш образа, пока офлайн-режим снят, "
            "и проверить, что в офлайне бэкенды доступны"
        ),
    )
    parser.add_argument(
        "--config", type=Path, default=None, help="YAML поверх дефолтной конфигурации"
    )
    parser.add_argument(
        "--all",
        dest="all_backends",
        action="store_true",
        help="греть все зарегистрированные бэкенды, а не только выбранные в конфиге",
    )
    parser.add_argument(
        "--with-vlm",
        action="store_true",
        help="греть и VLM-бэкенд OCR (~6 ГБ); без флага он пропускается",
    )
    parser.add_argument("--log-level", choices=LOG_LEVELS, default="INFO")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    setup_logging(args.log_level)
    try:
        config = load_config(args.config)
    except ConfigError as exc:
        LOGGER.error("%s", exc)
        return 2

    try:
        with tempfile.TemporaryDirectory(prefix="lecture-warmup-") as tmp:
            results = run_warmup(
                lambda: plan_targets(
                    config, all_backends=args.all_backends, with_vlm=args.with_vlm
                ),
                Path(tmp),
            )
    except WarmupError as exc:
        LOGGER.error("%s", exc)
        return 2

    print(format_summary(results))
    return 0 if all(result.ok for result in results) else 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
