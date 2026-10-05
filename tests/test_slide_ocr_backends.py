"""Бэкенды распознавания: текст, формулы, гибрид и VLM (4.2, 4.4, 4.8)."""

from __future__ import annotations

import importlib.util
import inspect
import sys
import types
from dataclasses import fields
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pytest
from PIL import Image, ImageDraw

from lecture_transcript.contracts import (
    Availability,
    OcrBackend,
    OcrFragment,
    Rect,
    SlideOcr,
)
from lecture_transcript.slide_ocr import HybridBackend, PaddleTextBackend, Pix2TexBackend, VlmBackend
from lecture_transcript.slide_ocr.assemble import assemble_slide_ocr
from lecture_transcript.slide_ocr.paddle_backend import parse_paddle_output
from lecture_transcript.slide_ocr import pix2tex_backend
from lecture_transcript.slide_ocr.pix2tex_backend import latex_to_markdown
from lecture_transcript.slide_ocr.vlm_backend import parse_vlm_markdown

FORMULA_LATEX = r"\sqrt{x^2-x} = -5"


# --------------------------------------------------------------------------
# 4.2 — разбор выдачи текстового бэкенда (боксы строк + confidence)
# --------------------------------------------------------------------------


def test_разбор_классической_выдачи_paddleocr():
    raw = [
        [
            [[[10, 20], [210, 20], [210, 60], [10, 60]], ("Как называются эти числа", 0.98)],
            [[[10, 80], [180, 80], [180, 120], [10, 120]], ("Пример 2.", 0.45)],
        ]
    ]
    fragments = parse_paddle_output(raw)
    assert [f.text for f in fragments] == ["Как называются эти числа", "Пример 2."]
    assert fragments[0].kind == "text"
    assert fragments[0].bbox == Rect(x=10, y=20, width=200, height=40)
    assert fragments[0].confidence == pytest.approx(0.98)
    assert fragments[0].low_confidence is False
    assert fragments[1].low_confidence is True


@pytest.mark.parametrize("as_array", [False, True], ids=["списки", "numpy"])
def test_разбор_словарной_выдачи_paddleocr_3x(as_array: bool):
    # PaddleX отдаёт rec_scores/rec_polys как numpy.ndarray, а не как списки:
    # на списках дефект «bool(ndarray) — ValueError» невидим.
    wrap = np.array if as_array else (lambda value: value)
    raw = [
        {
            "rec_texts": ["Пример 2.", "  "],
            "rec_scores": wrap([0.97, 0.9]),
            "rec_polys": wrap(
                [[[5, 5], [105, 5], [105, 45], [5, 45]], [[0, 0], [1, 0], [1, 1], [0, 1]]]
            ),
        }
    ]
    fragments = parse_paddle_output(raw)
    assert len(fragments) == 1  # пустые строки отбрасываются
    assert fragments[0].bbox == Rect(x=5, y=5, width=100, height=40)
    assert fragments[0].confidence == pytest.approx(0.97)


def test_разбор_словарной_выдачи_без_боксов():
    # dt_polys как запасной ключ и полностью отсутствующие боксы.
    assert parse_paddle_output({"rec_texts": ["a"], "rec_scores": np.array([0.9])})[0].bbox is None
    fragments = parse_paddle_output(
        {"rec_texts": ["Пример"], "rec_scores": np.array([0.9]), "dt_polys": np.array(
            [[[5, 5], [105, 5], [105, 45], [5, 45]]]
        )}
    )
    assert fragments[0].bbox == Rect(x=5, y=5, width=100, height=40)


def test_пустая_выдача_не_ошибка():
    assert parse_paddle_output(None) == []
    assert parse_paddle_output([]) == []
    assert parse_paddle_output([[]]) == []


@pytest.mark.skipif(
    importlib.util.find_spec("paddleocr") is not None,
    reason="paddleocr установлен — проверяем ветку 'библиотеки нет'",
)
def test_текстовый_бэкенд_честно_сообщает_об_отсутствии_библиотеки():
    availability = PaddleTextBackend().check_availability()
    assert availability.available is False
    assert "paddle" in availability.reason and "pip install" in availability.reason


# --------------------------------------------------------------------------
# 4.4 — формульный бэкенд
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw, expected",
    [
        (FORMULA_LATEX, f"${FORMULA_LATEX}$"),
        (f"${FORMULA_LATEX}$", f"${FORMULA_LATEX}$"),
        (f"$${FORMULA_LATEX}$$", f"${FORMULA_LATEX}$"),
        (r"\[ x = 1 \]", "$x = 1$"),
        ("   ", ""),
    ],
)
def test_latex_приводится_к_markdown_фрагменту(raw: str, expected: str):
    assert latex_to_markdown(raw) == expected


@pytest.mark.skipif(
    importlib.util.find_spec("pix2tex") is not None,
    reason="pix2tex установлен — проверяем ветку 'библиотеки нет'",
)
def test_формульный_бэкенд_честно_сообщает_об_отсутствии_библиотеки():
    availability = Pix2TexBackend().check_availability()
    assert availability.available is False
    assert "pix2tex" in availability.reason or "torch" in availability.reason


# --------------------------------------------------------------------------
# Гибрид: текст -> роутинг -> формулы (design D5)
# --------------------------------------------------------------------------


class _FakeTextBackend:
    name = "fake-text"

    def __init__(self, fragments: Sequence[OcrFragment]) -> None:
        self._fragments = tuple(fragments)
        self.unloaded = False

    def check_availability(self) -> Availability:
        return Availability(True)

    def recognize(self, image_path: Path) -> Sequence[OcrFragment]:
        return self._fragments

    def unload(self) -> None:
        self.unloaded = True


class _FakeFormulaBackend:
    name = "fake-formula"

    def __init__(self, latex: str = FORMULA_LATEX) -> None:
        self.latex = latex
        self.crops: list[tuple[int, int]] = []
        self.unloaded = False

    def check_availability(self) -> Availability:
        return Availability(True)

    def latex_from_image(self, image: Any) -> str:
        self.crops.append((image.width, image.height))
        return self.latex

    def unload(self) -> None:
        self.unloaded = True


def _slide_png(tmp_path: Path) -> Path:
    image = Image.new("RGB", (800, 400), "white")
    ImageDraw.Draw(image).text((20, 20), "slide", fill="black")
    path = tmp_path / "slide.png"
    image.save(path)
    return path


def _lines() -> list[OcrFragment]:
    return [
        OcrFragment(text="Пример 2.", kind="text", confidence=0.97, bbox=Rect(20, 20, 200, 50)),
        OcrFragment(text="√(x²−x) = −5", kind="text", confidence=0.88, bbox=Rect(20, 120, 400, 60)),
    ]


def test_гибрид_отправляет_формульную_строку_в_pix2tex(tmp_path: Path):
    formula_backend = _FakeFormulaBackend()
    backend = HybridBackend(_FakeTextBackend(_lines()), formula_backend)

    fragments = list(backend.recognize(_slide_png(tmp_path)))

    assert fragments[0].kind == "text" and fragments[0].text == "Пример 2."
    assert fragments[1].kind == "formula"
    assert fragments[1].text == f"${FORMULA_LATEX}$"
    assert fragments[1].bbox == Rect(20, 120, 400, 60)
    # В pix2tex ушёл кроп строки, а не весь слайд.
    assert len(formula_backend.crops) == 1
    assert formula_backend.crops[0] != (800, 400)


def test_гибрид_оставляет_строку_текстом_если_формула_не_прочиталась(tmp_path: Path):
    backend = HybridBackend(_FakeTextBackend(_lines()), _FakeFormulaBackend(latex="  "))
    fragments = list(backend.recognize(_slide_png(tmp_path)))
    assert [f.kind for f in fragments] == ["text", "text"]


def test_гибрид_недоступен_если_недоступен_любой_вложенный_бэкенд():
    class _Unavailable(_FakeTextBackend):
        def check_availability(self) -> Availability:
            return Availability(False, "весов нет")

    backend = HybridBackend(_Unavailable(()), _FakeFormulaBackend())
    availability = backend.check_availability()
    assert availability.available is False
    assert "весов нет" in availability.reason


def test_гибрид_выгружает_обе_модели():
    text_backend, formula_backend = _FakeTextBackend(()), _FakeFormulaBackend()
    HybridBackend(text_backend, formula_backend).unload()
    assert text_backend.unloaded and formula_backend.unloaded


# --------------------------------------------------------------------------
# Рукописное не распознаётся (спека «Рукописные пометки не распознаются»,
# design D5, категория [3]: «решается не моделью, а ссылкой» на PNG)
# --------------------------------------------------------------------------


#: Так PaddleOCR читает пометку маркером: буквоподобный мусор с низкой
#: уверенностью и без единого математического символа.
HANDWRITING_NOISE = "гoу лшв кya"


def _lines_with_handwriting() -> list[OcrFragment]:
    return [
        OcrFragment(
            text="Найдите все корни уравнения",
            kind="text",
            confidence=0.97,
            bbox=Rect(20, 20, 600, 50),
        ),
        OcrFragment(
            text=HANDWRITING_NOISE,
            kind="text",
            confidence=0.31,
            bbox=Rect(300, 250, 200, 90),
        ),
    ]


def test_слайд_с_рукописным_решением_печатное_распознано_рукописное_нет(tmp_path: Path):
    formula_backend = _FakeFormulaBackend()
    backend = HybridBackend(_FakeTextBackend(_lines_with_handwriting()), formula_backend)
    image_path = _slide_png(tmp_path)

    recognized = backend.recognize(image_path)
    fragments = list(recognized)
    result = assemble_slide_ocr(1, image_path, recognized, backend=backend.name)

    # Печатное условие задачи распознано.
    assert [f.text for f in fragments] == ["Найдите все корни уравнения"]
    # Рукописное не попало в текст ни как текст, ни как галлюцинация pix2tex.
    assert HANDWRITING_NOISE not in result.markdown
    assert FORMULA_LATEX not in result.markdown
    assert formula_backend.crops == []
    # Отброшенное не теряется молча: слайд помечен, человек идёт в PNG.
    assert result.unreliable is True
    # Доступ к рукописному даёт обязательная ссылка на PNG.
    assert f"![слайд 1]({image_path.as_posix()})" in result.markdown


def test_плохо_прочитанная_печатная_формула_в_pix2tex_всё_ещё_уходит(tmp_path: Path):
    # Обратная сторона правила: низкая уверенность на строке с математикой —
    # это категория [2] из D5, её и задумано отдавать pix2tex.
    lines = [
        OcrFragment(text="Пример 2.", kind="text", confidence=0.97, bbox=Rect(20, 20, 200, 50)),
        OcrFragment(text="V(x2-x)=-5", kind="text", confidence=0.35, bbox=Rect(20, 120, 400, 60)),
    ]
    formula_backend = _FakeFormulaBackend()
    backend = HybridBackend(_FakeTextBackend(lines), formula_backend)
    fragments = list(backend.recognize(_slide_png(tmp_path)))

    assert fragments[1].kind == "formula"
    assert fragments[1].text == f"${FORMULA_LATEX}$"
    assert len(formula_backend.crops) == 1
    # Уверенность исходной строки протаскивается (решение 2.5.3), поэтому на
    # сборке фрагмент получает пометку «сверьтесь с изображением».
    result = assemble_slide_ocr(1, _slide_png(tmp_path), fragments)
    assert result.fragments[1].low_confidence is True


def test_строка_ушедшая_в_pix2tex_только_по_уверенности_помечается(tmp_path: Path):
    # Строка состоит из слов и формулой по своему тексту не выглядит: в
    # pix2tex она ушла только из-за низкой уверенности, и выдавать её
    # результат за надёжный нельзя.
    lines = [
        OcrFragment(
            text="Haйдитe kopни ypaвнeния x2-5x+6=0",
            kind="text",
            confidence=0.45,
            bbox=Rect(20, 120, 400, 60),
        )
    ]
    backend = HybridBackend(_FakeTextBackend(lines), _FakeFormulaBackend())
    fragments = list(backend.recognize(_slide_png(tmp_path)))
    assert fragments[0].kind == "formula"
    assert fragments[0].low_confidence is True


# --------------------------------------------------------------------------
# Контракт OcrBackend: проверяется поведением, а не наличием атрибутов
# --------------------------------------------------------------------------


def _contract_violations(backend: Any) -> list[str]:
    """Нарушения контракта `OcrBackend`: имя, сигнатуры, поведение.

    `isinstance(..., OcrBackend)` для runtime_checkable Protocol проверяет
    только наличие атрибутов — класс с любыми сигнатурами и любым типом
    возврата такую проверку проходит. Здесь сверяются сигнатуры.
    """
    problems: list[str] = []
    name = getattr(backend, "name", None)
    if not isinstance(name, str) or not name:
        problems.append(f"name: ожидалась непустая строка, получено {name!r}")
    for method, parameters in (
        ("check_availability", 0),
        ("recognize", 1),
        ("unload", 0),
    ):
        function = getattr(backend, method, None)
        if not callable(function):
            problems.append(f"{method}: не метод")
            continue
        try:
            signature = inspect.signature(function)
        except (TypeError, ValueError):  # pragma: no cover — экзотические объекты
            problems.append(f"{method}: сигнатура недоступна")
            continue
        required = [
            parameter
            for parameter in signature.parameters.values()
            if parameter.default is inspect.Parameter.empty
            and parameter.kind
            in (parameter.POSITIONAL_ONLY, parameter.POSITIONAL_OR_KEYWORD)
        ]
        if len(required) != parameters:
            problems.append(
                f"{method}: ожидалось {parameters} обязательных аргументов, "
                f"объявлено {len(required)}"
            )
    availability = None
    try:
        availability = backend.check_availability()
    except Exception as exc:
        problems.append(f"check_availability бросил исключение вместо Availability: {exc!r}")
    if availability is not None and not isinstance(availability, Availability):
        problems.append(f"check_availability вернул {type(availability).__name__}")
    return problems


def _recognize_violations(backend: Any, image_path: Path) -> list[str]:
    """Поведение `recognize`: последовательность корректных `OcrFragment`."""
    problems: list[str] = []
    fragments = backend.recognize(image_path)
    if isinstance(fragments, (str, bytes)) or not isinstance(fragments, Sequence):
        return [f"recognize вернул {type(fragments).__name__}, а не последовательность"]
    for fragment in fragments:
        if not isinstance(fragment, OcrFragment):
            problems.append(f"элемент {type(fragment).__name__}, а не OcrFragment")
            continue
        if fragment.kind not in ("text", "formula"):
            problems.append(f"kind={fragment.kind!r}")
        if not 0.0 <= fragment.confidence <= 1.0:
            problems.append(f"confidence={fragment.confidence}")
        if fragment.bbox is not None and not isinstance(fragment.bbox, Rect):
            problems.append(f"bbox={type(fragment.bbox).__name__}")
    return problems


def test_штатные_бэкенды_удовлетворяют_контракту():
    for backend in (HybridBackend(), VlmBackend(), PaddleTextBackend(), Pix2TexBackend()):
        assert isinstance(backend, OcrBackend)
        assert _contract_violations(backend) == [], backend


def test_проверка_контракта_ловит_класс_с_неверными_сигнатурами():
    class _Nonsense:
        name = 123

        def check_availability(self):
            raise RuntimeError("boom")

        def recognize(self):
            return "мусор"

        def unload(self, a, b):
            return None

    # Protocol такой класс пропускает — значит проверять надо не им.
    assert isinstance(_Nonsense(), OcrBackend)
    problems = _contract_violations(_Nonsense())
    assert any("name" in problem for problem in problems)
    assert any(problem.startswith("recognize") for problem in problems)
    assert any(problem.startswith("unload") for problem in problems)
    assert any("check_availability" in problem for problem in problems)


def test_поведение_recognize_соответствует_контракту(tmp_path: Path):
    image_path = _slide_png(tmp_path)
    hybrid = HybridBackend(_FakeTextBackend(_lines()), _FakeFormulaBackend())
    assert _recognize_violations(hybrid, image_path) == []
    assert _recognize_violations(_FakeVlmBackend(), image_path) == []


def test_проверка_поведения_ловит_бэкенд_с_мусорным_результатом(tmp_path: Path):
    class _Broken(_FakeTextBackend):
        def recognize(self, image_path: Path):
            return (
                OcrFragment(text="a", kind="таблица", confidence=7.0),  # type: ignore[arg-type]
            )

    problems = _recognize_violations(_Broken(()), _slide_png(tmp_path))
    assert any("kind" in problem for problem in problems)
    assert any("confidence" in problem for problem in problems)


# --------------------------------------------------------------------------
# 4.8 — альтернативный VLM-бэкенд за тем же контрактом
# --------------------------------------------------------------------------


def test_разбор_ответа_vlm_в_фрагменты():
    answer = "```markdown\nПример 2.\n$\\sqrt{x^2-x} = -5$\n\nОтвет: корней нет\n```"
    fragments = parse_vlm_markdown(answer)
    assert [f.kind for f in fragments] == ["text", "formula", "text"]
    assert fragments[1].text == f"${FORMULA_LATEX}$"
    assert all(0.0 <= f.confidence <= 1.0 for f in fragments)


class _FakeVlmBackend:
    """VLM за тем же контрактом: один проход слайд -> Markdown+LaTeX."""

    name = "vlm"

    def check_availability(self) -> Availability:
        return Availability(True)

    def recognize(self, image_path: Path) -> Sequence[OcrFragment]:
        return tuple(parse_vlm_markdown(f"Пример 2.\n${FORMULA_LATEX}$"))

    def unload(self) -> None:
        return None


def _type_name(value: Any) -> str:
    # bbox по контракту `Rect | None` — обе формы равноправны.
    return "Rect|None" if value is None or isinstance(value, Rect) else type(value).__name__


def _schema(result: SlideOcr) -> dict[str, Any]:
    return {
        "slide_ocr": {
            field.name: _type_name(getattr(result, field.name)) for field in fields(result)
        },
        "fragment": [
            {field.name: _type_name(getattr(fragment, field.name)) for field in fields(fragment)}
            for fragment in result.fragments[:1]
        ],
        "kinds": sorted({fragment.kind for fragment in result.fragments}),
    }


def test_переключение_на_vlm_даёт_результат_того_же_формата(tmp_path: Path):
    image_path = _slide_png(tmp_path)
    hybrid = HybridBackend(_FakeTextBackend(_lines()), _FakeFormulaBackend())
    vlm = _FakeVlmBackend()

    hybrid_result = assemble_slide_ocr(1, image_path, hybrid.recognize(image_path), backend=hybrid.name)
    vlm_result = assemble_slide_ocr(1, image_path, vlm.recognize(image_path), backend=vlm.name)

    assert _schema(hybrid_result) == _schema(vlm_result)
    assert hybrid_result.backend == "hybrid" and vlm_result.backend == "vlm"
    # Оба варианта дают текст, формулу и обязательную ссылку на PNG.
    for result in (hybrid_result, vlm_result):
        assert "Пример 2." in result.markdown
        assert f"${FORMULA_LATEX}$" in result.markdown
        assert f"![слайд 1]({image_path.as_posix()})" in result.markdown


def test_vlm_бэкенд_честно_говорит_о_требованиях_к_vram(monkeypatch, tmp_path: Path):
    # Окружение задаётся явно: на реальной машине причина зависела бы от того,
    # есть ли CUDA и скачаны ли веса Qwen, а не от проверяемой ветки.
    no_cuda = _fake_torch(free_gb=0.0, total_gb=0.0)
    no_cuda.cuda.is_available = lambda: False  # type: ignore[attr-defined]
    _prepare_vlm_env(monkeypatch, tmp_path, no_cuda)

    availability = VlmBackend().check_availability()
    assert availability.available is False
    assert "CUDA" in availability.reason and "VRAM" in availability.reason


# --------------------------------------------------------------------------
# Доступность: проверка ДО прогона, а не падение в его середине
# (спека, сценарий «Недоступный бэкенд»)
# --------------------------------------------------------------------------


def _pretend_installed(monkeypatch, installed: dict[str, Any]) -> None:
    """Сделать вид, что перечисленные библиотеки установлены."""
    real = importlib.util.find_spec

    def fake(name: str, *args, **kwargs):
        if name in installed:
            return installed[name]
        return real(name, *args, **kwargs)

    monkeypatch.setattr(importlib.util, "find_spec", fake)


def test_paddle_сообщает_об_отсутствии_весов_до_прогона(monkeypatch, tmp_path: Path):
    # Библиотека установлена, весов нет: офлайн-режим (4.9) не даст скачать
    # их на ходу, значит сказать надо сейчас, а не в середине распознавания.
    _pretend_installed(monkeypatch, {"paddle": object(), "paddleocr": object()})
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.delenv("PADDLE_PDX_CACHE_HOME", raising=False)

    availability = PaddleTextBackend().check_availability()
    assert availability.available is False
    assert "вес" in availability.reason

    (tmp_path / ".paddleocr" / "whl").mkdir(parents=True)
    (tmp_path / ".paddleocr" / "whl" / "det.tar").write_bytes(b"x")
    assert PaddleTextBackend().check_availability().available is True


def test_paddle_находит_веса_в_кэше_paddlex_из_переменной_окружения(monkeypatch, tmp_path: Path):
    # Так устроен Docker-образ: PADDLE_PDX_CACHE_HOME=/cache/paddlex, а
    # PaddleX 3.2.0 кладёт модели в <CACHE_DIR>/official_models.
    home = tmp_path / "home"
    home.mkdir()
    cache = tmp_path / "cache" / "paddlex"
    _pretend_installed(monkeypatch, {"paddle": object(), "paddleocr": object()})
    monkeypatch.setattr(Path, "home", lambda: home)
    monkeypatch.setenv("PADDLE_PDX_CACHE_HOME", str(cache))

    # Служебные каталоги PaddleX весами не считаются.
    (cache / "locks").mkdir(parents=True)
    (cache / "func_ret").mkdir()
    assert PaddleTextBackend().check_availability().available is False

    model = cache / "official_models" / "eslav_PP-OCRv5_mobile_rec"
    model.mkdir(parents=True)
    (model / "inference.pdiparams").write_bytes(b"x")
    assert PaddleTextBackend().check_availability().available is True
    assert HybridBackend(formula_backend=_FakeFormulaBackend()).check_availability().available


class _OcrResult320(dict):
    """Как `OCRResult` PaddleX 3.2.0: `BaseResult(dict, ...)`, массивы numpy."""


class _FakePaddleOCR320:
    """Фейк, повторяющий API paddleocr==3.2.0 (пин Docker-образа).

    Сигнатуры перенесены из колеса: `paddleocr/_pipelines/ocr.py`
    (`PaddleOCR.__init__`, `predict`, устаревший `ocr`) и
    `paddleocr/_common_args.py` (`parse_common_args` — всё, что не явный
    параметр, не устаревшее имя и не общий аргумент, роняет конструктор).
    """

    _DEPRECATED = {
        "det_model_dir": "text_detection_model_dir",
        "det_limit_side_len": "text_det_limit_side_len",
        "det_limit_type": "text_det_limit_type",
        "det_db_thresh": "text_det_thresh",
        "det_db_box_thresh": "text_det_box_thresh",
        "det_db_unclip_ratio": "text_det_unclip_ratio",
        "rec_model_dir": "text_recognition_model_dir",
        "rec_batch_num": "text_recognition_batch_size",
        "use_angle_cls": "use_textline_orientation",
        "cls_model_dir": "textline_orientation_model_dir",
        "cls_batch_num": "textline_orientation_batch_size",
    }
    _COMMON = {
        "device", "enable_hpi", "use_tensorrt", "precision",
        "enable_mkldnn", "mkldnn_cache_capacity", "cpu_threads",
    }
    instances: list["_FakePaddleOCR320"] = []

    def __init__(
        self,
        doc_orientation_classify_model_name=None,
        doc_orientation_classify_model_dir=None,
        doc_unwarping_model_name=None,
        doc_unwarping_model_dir=None,
        text_detection_model_name=None,
        text_detection_model_dir=None,
        textline_orientation_model_name=None,
        textline_orientation_model_dir=None,
        textline_orientation_batch_size=None,
        text_recognition_model_name=None,
        text_recognition_model_dir=None,
        text_recognition_batch_size=None,
        use_doc_orientation_classify=None,
        use_doc_unwarping=None,
        use_textline_orientation=None,
        text_det_limit_side_len=None,
        text_det_limit_type=None,
        text_det_thresh=None,
        text_det_box_thresh=None,
        text_det_unclip_ratio=None,
        text_det_input_shape=None,
        text_rec_score_thresh=None,
        return_word_box=None,
        text_rec_input_shape=None,
        lang=None,
        ocr_version=None,
        **kwargs,
    ):
        self.params = {"lang": lang, "use_textline_orientation": use_textline_orientation}
        self.models = {
            name: value
            for name, value in locals().items()
            if name.endswith(("_model_name", "_model_dir")) and value is not None
        }
        for name, value in kwargs.items():
            if name in self._DEPRECATED:
                self.params[self._DEPRECATED[name]] = value
            elif name not in self._COMMON:
                raise ValueError(f"Unknown argument: {name}")
        _FakePaddleOCR320.instances.append(self)

    def predict(
        self,
        input,
        *,
        use_doc_orientation_classify=None,
        use_doc_unwarping=None,
        use_textline_orientation=None,
        text_det_limit_side_len=None,
        text_det_limit_type=None,
        text_det_thresh=None,
        text_det_box_thresh=None,
        text_det_unclip_ratio=None,
        text_rec_score_thresh=None,
        return_word_box=None,
    ):
        return [
            _OcrResult320(
                input_path=input,
                rec_texts=["Пример 2."],
                rec_scores=np.array([0.97]),
                rec_polys=np.array([[[5, 5], [105, 5], [105, 45], [5, 45]]]),
            )
        ]

    def ocr(self, img, **kwargs):  # в 3.2.0 помечен @deprecated
        return self.predict(img, **kwargs)


class _FakePaddleOCR2x:
    """PaddleOCR 2.x: нет `predict`, `ocr(img, cls=...)`, принимает `show_log`."""

    def __init__(self, lang="ch", use_angle_cls=False, show_log=True, **kwargs):
        self.lang = lang

    def ocr(self, img, det=True, rec=True, cls=True):
        return [[[[[10, 20], [210, 20], [210, 60], [10, 60]], ("Пример 2.", 0.98)]]]


def _install_fake_paddleocr(monkeypatch, engine_cls) -> None:
    module = types.ModuleType("paddleocr")
    module.PaddleOCR = engine_cls  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "paddleocr", module)
    from lecture_transcript.slide_ocr import offline

    monkeypatch.setattr(offline, "enforce_offline", lambda: None)


def test_paddle_бэкенд_работает_с_api_paddleocr_3_2_0(monkeypatch, tmp_path: Path):
    # show_log в 3.2.0 роняет конструктор: ValueError: Unknown argument.
    _FakePaddleOCR320.instances.clear()
    _install_fake_paddleocr(monkeypatch, _FakePaddleOCR320)
    monkeypatch.setenv("PADDLE_PDX_CACHE_HOME", str(tmp_path / "empty-cache"))

    fragments = PaddleTextBackend().recognize(_slide_png(tmp_path))

    assert [f.text for f in fragments] == ["Пример 2."]
    assert fragments[0].bbox == Rect(x=5, y=5, width=100, height=40)
    engine = _FakePaddleOCR320.instances[-1]
    assert engine.params == {"lang": "ru", "use_textline_orientation": True}
    assert engine.models == {}, "моделей в кэше нет — их выбирает и качает PaddleX"


def _cache_paddle_models(root: Path, *names: str) -> None:
    for name in names:
        (root / "official_models" / name).mkdir(parents=True)
        (root / "official_models" / name / "inference.yml").write_text(
            f"Global:\n  model_name: {name}\n", encoding="utf-8"
        )


_ALL_RU_MODELS = (
    "PP-LCNet_x1_0_doc_ori",
    "UVDoc",
    "PP-LCNet_x1_0_textline_ori",
    "PP-OCRv5_server_det",
    "eslav_PP-OCRv5_mobile_rec",
)


def test_paddle_бэкенд_берёт_скачанные_модели_с_диска(monkeypatch, tmp_path: Path):
    # PaddleX 3.2.0 без явного *_model_dir идёт к хостингу моделей даже за
    # скачанной моделью и без сети падает: «No available model hosting platforms».
    _FakePaddleOCR320.instances.clear()
    _install_fake_paddleocr(monkeypatch, _FakePaddleOCR320)
    cache = tmp_path / "paddlex"
    _cache_paddle_models(cache, *_ALL_RU_MODELS)
    monkeypatch.setenv("PADDLE_PDX_CACHE_HOME", str(cache))

    PaddleTextBackend().recognize(_slide_png(tmp_path))

    engine = _FakePaddleOCR320.instances[-1]
    models = cache / "official_models"
    assert engine.models == {
        "doc_orientation_classify_model_name": "PP-LCNet_x1_0_doc_ori",
        "doc_orientation_classify_model_dir": str(models / "PP-LCNet_x1_0_doc_ori"),
        "doc_unwarping_model_name": "UVDoc",
        "doc_unwarping_model_dir": str(models / "UVDoc"),
        "textline_orientation_model_name": "PP-LCNet_x1_0_textline_ori",
        "textline_orientation_model_dir": str(models / "PP-LCNet_x1_0_textline_ori"),
        "text_detection_model_name": "PP-OCRv5_server_det",
        "text_detection_model_dir": str(models / "PP-OCRv5_server_det"),
        "text_recognition_model_name": "eslav_PP-OCRv5_mobile_rec",
        "text_recognition_model_dir": str(models / "eslav_PP-OCRv5_mobile_rec"),
    }
    # модели текста заданы явно — lang не передаётся, PaddleOCR его бы проигнорировал
    assert engine.params["lang"] is None


def test_paddle_бэкенд_не_задаёт_детекцию_без_распознавания(monkeypatch, tmp_path: Path):
    # Заданная детекция отключает выбор моделей по lang: без пары распознавание
    # осталось бы без русской модели.
    _FakePaddleOCR320.instances.clear()
    _install_fake_paddleocr(monkeypatch, _FakePaddleOCR320)
    cache = tmp_path / "paddlex"
    _cache_paddle_models(cache, "PP-OCRv5_server_det", "UVDoc")
    monkeypatch.setenv("PADDLE_PDX_CACHE_HOME", str(cache))

    PaddleTextBackend().recognize(_slide_png(tmp_path))

    engine = _FakePaddleOCR320.instances[-1]
    assert set(engine.models) == {"doc_unwarping_model_name", "doc_unwarping_model_dir"}
    assert engine.params["lang"] == "ru"


def test_paddle_бэкенд_по_прежнему_работает_с_api_paddleocr_2x(monkeypatch, tmp_path: Path):
    _install_fake_paddleocr(monkeypatch, _FakePaddleOCR2x)
    fragments = PaddleTextBackend().recognize(_slide_png(tmp_path))
    assert [f.text for f in fragments] == ["Пример 2."]


def _pretend_pix2tex(monkeypatch, tmp_path: Path) -> tuple[Path, Path]:
    """pix2tex «установлен» в tmp_path; кэш моделей тоже там. -> (пакет, кэш)."""
    package = tmp_path / "pix2tex"
    package.mkdir()
    spec = types.SimpleNamespace(submodule_search_locations=[str(package)])
    _pretend_installed(monkeypatch, {"pix2tex": spec, "torch": object()})
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    return package, tmp_path / "cache" / "pix2tex"


def _write_pix2tex_weights(directory: Path, names=("weights.pth", "image_resizer.pth")) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for name in names:
        (directory / name).write_bytes(b"x")


def test_pix2tex_без_ресайзера_веса_не_считаются_скачанными(monkeypatch, tmp_path: Path):
    # Загрузка оборвалась между файлами: LatexOCR молча работал бы без
    # ресайзера, а проверка и прогрев показывали бы OK.
    _, cache = _pretend_pix2tex(monkeypatch, tmp_path)
    _write_pix2tex_weights(cache, names=("weights.pth",))
    availability = Pix2TexBackend().check_availability()
    assert availability.available is False and "вес" in availability.reason


def test_pix2tex_сообщает_об_отсутствии_весов_до_прогона(monkeypatch, tmp_path: Path):
    package, _ = _pretend_pix2tex(monkeypatch, tmp_path)

    availability = Pix2TexBackend().check_availability()
    assert availability.available is False
    assert "вес" in availability.reason

    checkpoints = package / "model" / "checkpoints"
    checkpoints.mkdir(parents=True)
    _write_pix2tex_weights(checkpoints)
    assert Pix2TexBackend().check_availability().available is True


def test_pix2tex_находит_веса_в_кэше_моделей(monkeypatch, tmp_path: Path):
    # В образе это /cache/pix2tex на томе: веса в пакете живут только
    # до конца `docker compose run --rm`.
    _, cache = _pretend_pix2tex(monkeypatch, tmp_path)
    _write_pix2tex_weights(cache)
    assert Pix2TexBackend().check_availability().available is True
    assert pix2tex_backend.find_weights_dir() == cache


def test_pix2tex_грузит_модель_из_кэша_и_качает_туда(monkeypatch, tmp_path: Path):
    _, cache = _pretend_pix2tex(monkeypatch, tmp_path)
    downloads: list[Path] = []

    def fake_download(target: Path) -> Path:
        downloads.append(target)
        _write_pix2tex_weights(target)
        return target

    created: list[Any] = []

    class FakeLatexOCR:
        def __init__(self, arguments=None):
            created.append(arguments)

        def __call__(self, image):
            return r"x = a + b"

    cli = types.ModuleType("pix2tex.cli")
    cli.LatexOCR = FakeLatexOCR  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "pix2tex", types.ModuleType("pix2tex"))
    monkeypatch.setitem(sys.modules, "pix2tex.cli", cli)
    monkeypatch.setattr(pix2tex_backend, "download_weights", fake_download)
    monkeypatch.setattr("lecture_transcript.slide_ocr.offline.enforce_offline", lambda: {})

    image = Image.new("RGB", (40, 20), "white")
    ImageDraw.Draw(image).line((2, 10, 38, 10), fill="black")
    assert Pix2TexBackend().latex_from_image(image) == "x = a + b"

    assert downloads == [cache]
    assert created[0].checkpoint == str(cache / "weights.pth")


def test_pix2tex_не_отдаёт_модели_однотонную_картинку(monkeypatch):
    # pix2tex делит на (max - min) и падает на cv2.cvtColor(!_src.empty()).
    backend = Pix2TexBackend()
    monkeypatch.setattr(backend, "_load", lambda: pytest.fail("модель не должна грузиться"))
    assert backend.latex_from_image(Image.new("RGB", (64, 32), "white")) == ""
    assert backend.latex_from_image(Image.new("L", (64, 32), 0)) == ""


def _fake_torch(free_gb: float, total_gb: float) -> types.ModuleType:
    module = types.ModuleType("torch")
    free_bytes = int(free_gb * 1024 ** 3)
    total_bytes = int(total_gb * 1024 ** 3)

    class _Cuda:
        @staticmethod
        def is_available() -> bool:
            return True

        @staticmethod
        def mem_get_info() -> tuple[int, int]:
            return free_bytes, total_bytes

        @staticmethod
        def get_device_properties(index: int):
            return types.SimpleNamespace(total_memory=total_bytes)

        @staticmethod
        def empty_cache() -> None:
            return None

    module.cuda = _Cuda()  # type: ignore[attr-defined]
    return module


def _prepare_vlm_env(monkeypatch, tmp_path: Path, torch_module, with_weights: bool = True):
    _pretend_installed(
        monkeypatch,
        {"torch": object(), "transformers": object(), "bitsandbytes": object()},
    )
    monkeypatch.setenv("HF_HOME", str(tmp_path))
    for variable in ("HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE"):
        monkeypatch.delenv(variable, raising=False)
    if with_weights:
        model_dir = "models--" + VlmBackend().model_id.replace("/", "--")
        (tmp_path / "hub" / model_dir).mkdir(parents=True)
    monkeypatch.setitem(sys.modules, "torch", torch_module)


def test_vlm_меряет_свободную_а_не_общую_видеопамять(monkeypatch, tmp_path: Path):
    # Карта на 8 ГБ, из которых 7 уже занято: по общей памяти проверка бы
    # прошла, а загрузка модели упала бы OOM в середине прогона.
    _prepare_vlm_env(monkeypatch, tmp_path, _fake_torch(free_gb=1.0, total_gb=8.0))
    availability = VlmBackend().check_availability()
    assert availability.available is False
    assert "VRAM" in availability.reason and "1.0" in availability.reason


def test_vlm_доступен_когда_свободной_видеопамяти_хватает(monkeypatch, tmp_path: Path):
    _prepare_vlm_env(monkeypatch, tmp_path, _fake_torch(free_gb=7.5, total_gb=8.0))
    assert VlmBackend().check_availability().available is True


def test_vlm_сообщает_об_отсутствии_весов_до_прогона(monkeypatch, tmp_path: Path):
    _prepare_vlm_env(
        monkeypatch, tmp_path, _fake_torch(free_gb=7.5, total_gb=8.0), with_weights=False
    )
    availability = VlmBackend().check_availability()
    assert availability.available is False
    assert "вес" in availability.reason and "Qwen" in availability.reason


@pytest.mark.parametrize(
    "backend_factory, module",
    [
        (PaddleTextBackend, "paddle"),
        (Pix2TexBackend, "pix2tex"),
        (VlmBackend, "torch"),
    ],
)
def test_битая_установка_не_роняет_проверку_доступности(monkeypatch, backend_factory, module):
    # find_spec бросает ModuleNotFoundError/ValueError, если родительский
    # пакет установлен, но повреждён; контракт требует вернуть Availability.
    def explode(name: str, *args, **kwargs):
        raise ModuleNotFoundError(f"{name} повреждён")

    monkeypatch.setattr(importlib.util, "find_spec", explode)
    availability = backend_factory().check_availability()
    assert isinstance(availability, Availability)
    assert availability.available is False and availability.reason


def test_гибрид_не_падает_на_битой_установке_вложенного_бэкенда(monkeypatch):
    def explode(name: str, *args, **kwargs):
        raise ValueError(f"{name} повреждён")

    monkeypatch.setattr(importlib.util, "find_spec", explode)
    availability = HybridBackend().check_availability()
    assert availability.available is False and "повреждена" in availability.reason


# --------------------------------------------------------------------------
# Проверки на реальных весах — запускаются на целевой машине (-m models)
#
# Кроп слайда режется из эталонной записи по таймкодам, найденным глазами:
#   t=425  — слайд «Как называются эти числа 1, 2, 3, 4, 5, 6, ...?» (4.2),
#            печатный текст, рукописных пометок ещё нет;
#   t=4210 — слайд «Пример 2. sqrt(x^2-x) = -5» из design.md (4.4). Он в
#            эталонной записи есть: интервал t≈4198–4225 с (группа #35,
#            покадровый проход стадии slide-extraction), берём середину.
# --------------------------------------------------------------------------

TEXT_SLIDE_S = 425.0
FORMULA_SLIDE_S = 4210.0


def _crop_reference_slide(reference: Path, manifest: dict, timestamp: float, out: Path) -> Path:
    import subprocess

    region = manifest["slide_region_1080p"]
    subprocess.run(
        [
            "ffmpeg", "-nostdin", "-loglevel", "error", "-ss", str(timestamp),
            "-i", str(reference), "-frames:v", "1",
            "-vf", f"crop={region['width']}:{region['height']}:{region['x']}:{region['y']}",
            str(out), "-y",
        ],
        check=True,
    )
    return out


@pytest.mark.models
@pytest.mark.reference
def test_печатный_текст_распознан_без_ошибок(reference_mp4, fixtures_manifest, tmp_path: Path):
    """4.2: слайд «Как называются эти числа ...» распознан без ошибок."""
    backend = PaddleTextBackend()
    availability = backend.check_availability()
    if not availability.available:
        pytest.skip(availability.reason)

    crop = _crop_reference_slide(reference_mp4, fixtures_manifest, TEXT_SLIDE_S, tmp_path / "text.png")
    fragments = list(backend.recognize(crop))
    joined = " ".join(fragment.text for fragment in fragments)

    assert "Как называются эти числа" in joined
    assert "Обозначение множества" in joined
    assert all(fragment.bbox is not None for fragment in fragments)
    assert all(fragment.confidence >= 0.8 for fragment in fragments)


@pytest.mark.models
@pytest.mark.reference
def test_печатная_формула_распознана_в_корректный_latex(
    reference_mp4, fixtures_manifest, tmp_path: Path
):
    """4.4: печатная формула слайда «Пример 2.» отдаётся корректным LaTeX."""
    backend = HybridBackend()
    availability = backend.check_availability()
    if not availability.available:
        pytest.skip(availability.reason)

    crop = _crop_reference_slide(reference_mp4, fixtures_manifest, FORMULA_SLIDE_S, tmp_path / "formula.png")
    fragments = list(backend.recognize(crop))
    formulas = [fragment.text for fragment in fragments if fragment.kind == "formula"]

    assert formulas, "ни одна строка слайда не распознана как формула"
    assert any("sqrt" in formula for formula in formulas)
    for formula in formulas:
        body = formula.strip("$")
        assert body, "пустая формула"
        assert body.count("{") == body.count("}")  # LaTeX рендерится
        assert body.count("(") == body.count(")")
    assert any("Пример" in fragment.text for fragment in fragments if fragment.kind == "text")


def test_pix2tex_загрузка_весов_не_висит_бесконечно(monkeypatch, tmp_path: Path):
    import io

    timeouts: list[float | None] = []

    def fake_urlopen(url, timeout=None):
        timeouts.append(timeout)
        return io.BytesIO(b"weights")

    monkeypatch.setattr(pix2tex_backend.urllib.request, "urlopen", fake_urlopen)
    target = pix2tex_backend.download_weights(tmp_path / "pix2tex")

    assert timeouts and all(t is not None and t > 0 for t in timeouts)
    assert sorted(p.name for p in target.iterdir()) == ["image_resizer.pth", "weights.pth"]
