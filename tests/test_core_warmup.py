"""Тесты прогрева кэша моделей (`python -m lecture_transcript.warmup`)."""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

from lecture_transcript import warmup
from lecture_transcript.config import load_config
from lecture_transcript.contracts import Availability

REPO_ROOT = Path(__file__).resolve().parent.parent


# --------------------------------------------------------------------------
# Фейковые бэкенды
# --------------------------------------------------------------------------


class FakeBackend:
    """Журналирует load/warm/unload/check и состояние офлайна в каждый момент."""

    def __init__(self, name: str, events: list[str], *, available: bool = True) -> None:
        self.name = name
        self.events = events
        self.available = available
        self.env_at_warm: dict[str, str | None] = {}
        self.env_at_check: dict[str, str | None] = {}

    def warm(self) -> None:
        self.events.append(f"warm:{self.name}")
        self.env_at_warm = {k: os.environ.get(k) for k in SPEC}

    def check_availability(self) -> Availability:
        self.events.append(f"check:{self.name}")
        self.env_at_check = {k: os.environ.get(k) for k in SPEC}
        return Availability(self.available, "" if self.available else "весов нет")

    def unload(self) -> None:
        self.events.append(f"unload:{self.name}")


SPEC = warmup.offline_env()


def fake_target(backend: FakeBackend, *, fail_load: bool = False, fail_warm: bool = False):
    def load():
        backend.events.append(f"load:{backend.name}")
        if fail_load:
            raise RuntimeError(f"нет сети для {backend.name}")
        return backend

    def warm(obj, inputs):
        obj.warm()
        assert inputs.image_path.is_file() and inputs.audio.path.is_file()
        if fail_warm:
            raise OSError(f"обрыв загрузки {backend.name}")

    return warmup.WarmTarget("fake", backend.name, load, warm)


@pytest.fixture()
def offline_env_set(monkeypatch: pytest.MonkeyPatch) -> dict[str, str]:
    """Окружение как в образе: офлайн включён. monkeypatch вернёт всё после теста."""
    for key, value in SPEC.items():
        monkeypatch.setenv(key, value)
    return dict(SPEC)


# --------------------------------------------------------------------------
# Офлайн-режим
# --------------------------------------------------------------------------


def test_offline_env_is_the_union_of_package_lists() -> None:
    """Список не выдуман: ровно объединение OFFLINE_ENV обоих пакетов."""
    from lecture_transcript.slide_ocr.offline import OFFLINE_ENV as OCR_ENV
    from lecture_transcript.speech_transcription.offline import OFFLINE_ENV as ASR_ENV

    assert warmup.offline_env() == {**ASR_ENV, **OCR_ENV}
    assert "HF_HUB_OFFLINE" in SPEC and "TRANSFORMERS_OFFLINE" in SPEC


def test_offline_lifted_during_warmup_and_restored_after(
    tmp_path: Path, offline_env_set: dict[str, str]
) -> None:
    events: list[str] = []
    backend = FakeBackend("a", events)
    before = {k: os.environ.get(k) for k in SPEC}

    results = warmup.run_warmup(lambda: [fake_target(backend)], tmp_path)

    # во время прогрева все переключатели *_OFFLINE явно выключены
    for key in SPEC:
        if key.endswith("_OFFLINE"):
            assert backend.env_at_warm[key] == "0", f"{key} не снят на время прогрева"
    # а гигиена (телеметрия и т. п.) осталась включённой
    assert backend.env_at_warm["HF_HUB_DISABLE_TELEMETRY"] == "1"
    # после прогрева окружение ровно как было
    assert {k: os.environ.get(k) for k in SPEC} == before
    assert results[0].ok


def test_backend_enforce_offline_does_not_end_warmup_window(
    tmp_path: Path, offline_env_set: dict[str, str]
) -> None:
    """Pix2TexBackend._load() зовёт enforce_offline(): раньше это возвращало
    HF_HUB_OFFLINE=1 посреди прогрева, и следующие модели не скачивались."""
    from lecture_transcript.slide_ocr.offline import enforce_offline
    from lecture_transcript.speech_transcription.offline import is_offline

    events: list[str] = []
    first, second = FakeBackend("ocr", events), FakeBackend("asr", events)
    offline_seen: list[bool] = []

    def warm_ocr(obj, inputs):
        enforce_offline()
        obj.warm()

    def warm_asr(obj, inputs):
        offline_seen.append(is_offline())
        obj.warm()

    warmup.run_warmup(
        lambda: [
            warmup.WarmTarget("fake", "ocr", lambda: first, warm_ocr),
            warmup.WarmTarget("fake", "asr", lambda: second, warm_asr),
        ],
        tmp_path,
    )
    assert offline_seen == [False]
    assert second.env_at_warm["HF_HUB_OFFLINE"] == "0"


def test_environment_restored_even_when_warmup_explodes(
    tmp_path: Path, offline_env_set: dict[str, str]
) -> None:
    before = {k: os.environ.get(k) for k in SPEC}

    def broken_plan():
        raise KeyboardInterrupt  # не Exception: окно обязано вернуть окружение

    with pytest.raises(KeyboardInterrupt):
        warmup.run_warmup(broken_plan, tmp_path)
    assert {k: os.environ.get(k) for k in SPEC} == before


def test_final_check_runs_in_offline_mode_even_if_outside_was_online(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Как в compose: снаружи HF_HUB_OFFLINE=0. Проверка всё равно офлайн,
    а после неё снова «0», как было."""
    for key in SPEC:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("HF_HUB_OFFLINE", "0")

    events: list[str] = []
    backend = FakeBackend("a", events)
    warmup.run_warmup(lambda: [fake_target(backend)], tmp_path)

    for key, value in SPEC.items():
        assert backend.env_at_check[key] == value, f"{key} при проверке не офлайн"
    assert os.environ.get("HF_HUB_OFFLINE") == "0"
    assert os.environ.get("TRANSFORMERS_OFFLINE") is None


# --------------------------------------------------------------------------
# Порядок и отказы
# --------------------------------------------------------------------------


def test_each_backend_loaded_and_unloaded_in_turn(
    tmp_path: Path, offline_env_set: dict[str, str]
) -> None:
    events: list[str] = []
    a, b = FakeBackend("a", events), FakeBackend("b", events)
    warmup.run_warmup(lambda: [fake_target(a), fake_target(b)], tmp_path)
    assert events == [
        "load:a", "warm:a", "unload:a",
        "load:b", "warm:b", "unload:b",
        "check:a", "check:b",
    ]


def test_failed_loader_gives_nonzero_and_does_not_stop_others(
    tmp_path: Path, offline_env_set: dict[str, str], monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    events: list[str] = []
    ok_first = FakeBackend("first", events)
    no_net = FakeBackend("no-net", events)
    torn = FakeBackend("torn", events)
    ok_last = FakeBackend("last", events)

    monkeypatch.setattr(
        warmup,
        "plan_targets",
        lambda config, **kw: [
            fake_target(ok_first),
            fake_target(no_net, fail_load=True),
            fake_target(torn, fail_warm=True),
            fake_target(ok_last),
        ],
    )
    code = warmup.main(["--log-level", "WARNING"])
    out = capsys.readouterr().out

    assert code == 1
    assert "warm:last" in events, "отказ одного бэкенда остановил остальных"
    assert "unload:torn" in events, "упавший после загрузки бэкенд не выгружен"
    assert "check:no-net" not in events and "check:torn" not in events
    assert "OK    fake:first" in out and "OK    fake:last" in out
    assert "FAIL  fake:no-net: не загрузилась — RuntimeError: нет сети для no-net" in out
    assert "FAIL  fake:torn: не загрузилась — OSError: обрыв загрузки torn" in out


def test_backend_unavailable_offline_after_warmup_is_a_failure(
    tmp_path: Path, offline_env_set: dict[str, str], monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    events: list[str] = []
    ghost = FakeBackend("ghost", events, available=False)
    monkeypatch.setattr(warmup, "plan_targets", lambda config, **kw: [fake_target(ghost)])
    assert warmup.main(["--log-level", "WARNING"]) == 1
    assert "недоступна в офлайне — весов нет" in capsys.readouterr().out


def test_all_ok_gives_zero_and_prints_cache_dirs(
    tmp_path: Path, offline_env_set: dict[str, str], monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    events: list[str] = []
    monkeypatch.setenv("HF_HOME", "/cache/huggingface")
    monkeypatch.setattr(
        warmup, "plan_targets", lambda config, **kw: [fake_target(FakeBackend("a", events))]
    )
    assert warmup.main(["--log-level", "WARNING"]) == 0
    out = capsys.readouterr().out
    assert "OK    fake:a" in out
    assert "HF_HOME=/cache/huggingface" in out


# --------------------------------------------------------------------------
# Что греется
# --------------------------------------------------------------------------


def _labels(targets) -> list[str]:
    return [t.label for t in targets]


def test_plan_follows_config_by_default() -> None:
    config = load_config()
    labels = _labels(warmup.plan_targets(config))
    assert labels == [
        f"ocr:{config.slide_ocr.backend}",
        f"asr:{config.speech_transcription.asr_backend}",
        "vad:silero",
        "punctuation:default",
    ]


def test_plan_all_skips_vlm_unless_asked() -> None:
    from lecture_transcript.slide_ocr import registry as ocr_registry
    from lecture_transcript.speech_transcription import registry as asr_registry

    config = load_config()
    labels = _labels(warmup.plan_targets(config, all_backends=True))
    assert "ocr:vlm" not in labels
    for name in ocr_registry.list_backend_names():
        if name != "vlm":
            assert f"ocr:{name}" in labels
    for name in asr_registry.list_backend_names():
        assert f"asr:{name}" in labels

    with_vlm = _labels(warmup.plan_targets(config, all_backends=True, with_vlm=True))
    assert "ocr:vlm" in with_vlm


def test_hybrid_warms_both_text_and_formula_parts(tmp_path: Path) -> None:
    """Роутер мог бы не позвать pix2tex на пробной строке — греем обе части явно."""
    calls: list[str] = []

    class Text:
        def recognize(self, path):
            calls.append("text")

    class Formula:
        def latex_from_image(self, image):
            calls.append(f"formula:{image.size}")

    class Hybrid:
        text_backend = Text()
        formula_backend = Formula()

    inputs = warmup._make_inputs(tmp_path)  # noqa: SLF001
    warmup._warm_ocr(Hybrid(), inputs)  # noqa: SLF001
    assert calls == ["text", "formula:(160, 48)"]


def test_warm_image_is_not_blank(tmp_path: Path) -> None:
    """На однотонной картинке pix2tex падает (cvtColor !_src.empty()), а бэкенд
    такую модели и не отдаёт — прогрев не загрузил бы веса."""
    from lecture_transcript.slide_ocr.pix2tex_backend import _is_blank

    inputs = warmup._make_inputs(tmp_path)  # noqa: SLF001
    assert not _is_blank(inputs.image())


class _Punctuator:
    def __init__(self, restore) -> None:
        self.restore = restore

    def check_availability(self) -> Availability:
        return Availability(True)

    def unload(self) -> None:
        pass


def _hub_unreachable(text: str) -> str:
    raise OSError("huggingface.co недоступен")


@pytest.mark.parametrize(
    "restore, error",
    [
        (_hub_unreachable, "OSError: huggingface.co недоступен"),
        (lambda text: text, "без изменений"),
    ],
    ids=["модель-упала", "текст-без-изменений"],
)
def test_punctuation_warmup_fails_loudly(
    tmp_path: Path, offline_env_set: dict[str, str], monkeypatch: pytest.MonkeyPatch,
    restore, error: str,
) -> None:
    """restore_punctuation() глотает ошибки модели; прогрев так делать не должен."""
    from lecture_transcript import speech_transcription

    monkeypatch.setattr(speech_transcription, "get_punctuator", lambda: _Punctuator(restore))
    results = warmup.run_warmup(lambda: [warmup._punctuation_target()], tmp_path)  # noqa: SLF001
    assert not results[0].loaded
    assert error in results[0].error


def test_punctuation_warmup_ok_when_model_punctuates(
    tmp_path: Path, offline_env_set: dict[str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    from lecture_transcript import speech_transcription

    punctuator = _Punctuator(lambda text: "Проверка связи.")
    monkeypatch.setattr(speech_transcription, "get_punctuator", lambda: punctuator)
    results = warmup.run_warmup(lambda: [warmup._punctuation_target()], tmp_path)  # noqa: SLF001
    assert results[0].ok


# --------------------------------------------------------------------------
# Стык с docker/compose.yaml
# --------------------------------------------------------------------------


def test_compose_invocation_is_accepted_by_warmup() -> None:
    compose = (REPO_ROOT / "docker" / "compose.yaml").read_text(encoding="utf-8")
    match = re.search(r"python -m lecture_transcript\.warmup([^\n]*)", compose)
    assert match, "compose.yaml не вызывает lecture_transcript.warmup"
    extra = match.group(1).split()
    args = warmup.build_parser().parse_args(extra)  # SystemExit, если не принимает
    assert args.all_backends is False


def test_module_runs_as_script_help() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "lecture_transcript.warmup", "--help"],
        capture_output=True, text=True, timeout=60, check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "--all" in result.stdout and "--with-vlm" in result.stdout
