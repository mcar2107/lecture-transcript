"""Тесты стадии speech-transcription, не требующие весов моделей.

Проверяется вся логика, которая не зависит от конкретной модели: реестр и
отсев недоступного бэкенда до прогона, детекция речи и абсолютность
таймкодов, деградация без глоссария, условное включение стадии пунктуации
и сохранность таймкодов после неё, единство формата результата между
бэкендами, работа без сети.

Аудио — синтетическое (numpy + wave), бэкенды — фейковые.
"""

from __future__ import annotations

import dataclasses
import importlib.util
import inspect
import logging
import os
import random
import shutil
import socket
import sys
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pytest

from lecture_transcript.contracts import (
    AsrBackend,
    AudioArtifact,
    Availability,
    BackendUnavailableError,
    Glossary,
    SpeechInterval,
    Transcription,
    Word,
)
from lecture_transcript.speech_transcription import (
    DEFAULT_PUNCTUATOR_NAME,
    GIGAAM_MAX_CHUNK_S,
    NO_PUNCTUATION_WARNING,
    GigaAmBackend,
    IntervalAsrBackend,
    SpeechTranscriptionConfig,
    VadConfig,
    align_punctuated,
    SbertPuncCaseRu,
    available_backend_names,
    check_backend,
    check_punctuator,
    check_vad_backend,
    clear_audio_cache,
    ensure_available,
    is_monotonic,
    list_backend_names,
    offline_mode,
    register,
    reset_registry,
    restore_punctuation,
    run_vad,
    split_interval,
    transcribe_speech,
    validate_words,
    write_wav_mono,
)
from lecture_transcript.speech_transcription import audio_io, vad as vad_module
from lecture_transcript.speech_transcription import punctuation as punctuation_module
from lecture_transcript.speech_transcription.offline import OFFLINE_ENV
from lecture_transcript.speech_transcription.punctuation import (
    PARTIAL_PUNCTUATION_PREFIX,
    normalize_core,
    partial_punctuation_warning,
)

SAMPLE_RATE = 16_000

# Раскладка синтетической записи, секунды от начала:
#   [0.0, 1.0) тишина, [1.0, 2.0) речь, [2.0, 5.0) ДЛИТЕЛЬНАЯ тишина,
#   [5.0, 6.0) речь, [6.0, 6.5) тишина
SPEECH_SPANS = ((1.0, 2.0), (5.0, 6.0))
TOTAL_S = 6.5


# --------------------------------------------------------------------------
# Синтетическое аудио
# --------------------------------------------------------------------------


def make_audio(
    path: Path,
    spans: Sequence[tuple[float, float]] = SPEECH_SPANS,
    total_s: float = TOTAL_S,
) -> AudioArtifact:
    """Собрать WAV «тишина — речь — длительная тишина — речь»."""
    rng = np.random.default_rng(20260909)
    count = int(round(total_s * SAMPLE_RATE))
    # Фоновый шум заметно ниже порога тишины — так ведёт себя реальная запись.
    samples = rng.normal(0.0, 0.0005, count).astype(np.float32)
    time = np.arange(count, dtype=np.float32) / SAMPLE_RATE
    for start_s, end_s in spans:
        mask = (time >= start_s) & (time < end_s)
        samples[mask] += 0.5 * np.sin(2 * np.pi * 220.0 * time[mask])
    write_wav_mono(path, samples, SAMPLE_RATE)
    return AudioArtifact(
        path=path, sample_rate=SAMPLE_RATE, duration_s=total_s, source_track_index=0
    )


@pytest.fixture
def audio(tmp_path: Path) -> AudioArtifact:
    return make_audio(tmp_path / "lecture.wav")


@pytest.fixture
def energy_config() -> SpeechTranscriptionConfig:
    """Конфиг без сети и без весов: энергетический VAD, пунктуатор задаётся тестом."""
    return SpeechTranscriptionConfig(vad_backend="energy", vad=VadConfig())


@pytest.fixture(autouse=True)
def clean_registry():
    """Реестр возвращается к штатному составу после каждого теста."""
    yield
    reset_registry()


# --------------------------------------------------------------------------
# Фейковые бэкенды и пунктуаторы
# --------------------------------------------------------------------------


class FakeBackend(IntervalAsrBackend):
    """Фейковый ASR: одно слово на 0.5 с куска, времена́ — от начала куска."""

    def __init__(
        self,
        name: str = "fake",
        *,
        provides_punctuation: bool = False,
        supports_glossary: bool = False,
        available: bool = True,
        reason: str = "",
        word_s: float = 0.5,
        vocabulary: Sequence[str] = ("привет", "мир", "это", "тест", "лекции"),
    ) -> None:
        super().__init__()
        self.name = name
        self.provides_punctuation = provides_punctuation
        self.supports_glossary = supports_glossary
        self._available = available
        self._reason = reason
        self._word_s = word_s
        self._vocabulary = tuple(vocabulary)
        self.transcribe_calls = 0
        self.seen_glossaries: list[Glossary | None] = []
        self.unloaded = False

    def check_availability(self) -> Availability:
        return Availability(self._available, self._reason)

    def transcribe(self, audio, intervals, glossary=None):
        self.transcribe_calls += 1
        self.seen_glossaries.append(glossary)
        return super().transcribe(audio, intervals, glossary)

    def _transcribe_chunk(self, samples, sample_rate, glossary):
        duration_s = samples.shape[0] / sample_rate
        words: list[Word] = []
        cursor = 0.0
        index = 0
        while cursor < duration_s - 1e-9:
            end = min(duration_s, cursor + self._word_s)
            text = self._vocabulary[index % len(self._vocabulary)]
            if glossary and index == 0 and glossary.terms:
                text = glossary.terms[0]
            words.append(Word(text=text, start_s=cursor, end_s=end))
            cursor = end
            index += 1
        return words

    def unload(self) -> None:
        self.unloaded = True


class FakePunctuator:
    """Фейковый пунктуатор, повторяющий приёмы настоящей русской модели.

    Дружелюбного «capitalize + запятая через слово» мало: реальная модель
    ставит ещё и отдельно стоящее тире, кавычки вокруг слова и «ё» вместо
    «е». Именно на таком входе выравнивание раньше молча теряло символы,
    поэтому фейк обязан их выдавать.
    """

    name = "fake-punctuator"

    def __init__(self, *, available: bool = True, drop_word: bool = False) -> None:
        self._available = available
        self._drop_word = drop_word
        self.calls = 0
        self.unloaded = False

    def check_availability(self) -> Availability:
        return Availability(self._available, "" if self._available else "нет весов")

    def restore(self, text: str) -> str:
        self.calls += 1
        tokens = text.split()
        if self._drop_word and len(tokens) > 1:
            tokens = tokens[:-1]
        out: list[str] = []
        last = len(tokens) - 1
        for index, token in enumerate(tokens):
            piece = token.capitalize() if index == 0 else token
            if index == 1:
                piece = piece.replace("е", "ё", 1)
            if index == 2:
                piece = f"«{piece}»"
            if index % 2 == 1 and index != last:
                piece += ","
            if index == last and last >= 2:
                out.append("—")
            out.append(piece)
        return " ".join(out) + "." if out else ""

    def unload(self) -> None:
        self.unloaded = True


# --------------------------------------------------------------------------
# 5.1 — реестр и отсев недоступного бэкенда до прогона
# --------------------------------------------------------------------------


def test_registry_lists_builtin_backends():
    assert "gigaam-v2" in list_backend_names()
    assert "whisper-large-v3" in list_backend_names()
    # Реальные веса здесь не стоят, поэтому список доступных — подмножество.
    assert set(available_backend_names()) <= set(list_backend_names())


def test_unknown_backend_rejected_with_reason():
    with pytest.raises(BackendUnavailableError) as exc:
        ensure_available("нет-такого")
    assert "не зарегистрирован" in str(exc.value)
    assert "gigaam-v2" in str(exc.value)


def test_registry_resolves_aliases():
    """Псевдонимы ведут к тому же бэкенду, но в списке имён только канонические."""
    assert check_backend("gigaam").reason == check_backend("gigaam-v2").reason
    assert check_backend("whisper").reason == check_backend("whisper-large-v3").reason
    assert "gigaam" not in list_backend_names()


def test_unavailable_backend_rejected_before_run(audio, energy_config):
    """5.1: недоступный бэкенд отвергается ДО прогона — transcribe не вызывается."""
    backend = FakeBackend(
        "broken", available=False, reason="не установлен пакет 'torch'"
    )
    register("broken", lambda: backend)

    assert check_backend("broken").available is False
    with pytest.raises(BackendUnavailableError) as exc:
        transcribe_speech(
            audio, config=dataclasses.replace(energy_config, asr_backend="broken")
        )

    assert "не установлен пакет" in str(exc.value)
    assert backend.transcribe_calls == 0
    assert "broken" not in available_backend_names()


def test_backend_refuses_direct_transcribe_when_unavailable(audio):
    backend = FakeBackend("broken", available=False, reason="нет весов")
    with pytest.raises(BackendUnavailableError):
        backend.transcribe(audio, [SpeechInterval(0.0, 1.0)])


# --------------------------------------------------------------------------
# 5.2 — VAD и абсолютные таймкоды
# --------------------------------------------------------------------------


def test_vad_finds_speech_and_cuts_long_silence(audio):
    result = run_vad(audio, VadConfig(), "energy")

    assert result.backend == "energy"
    assert len(result.intervals) == 2
    first, second = result.intervals
    assert first.start_s == pytest.approx(1.0, abs=0.25)
    assert first.end_s == pytest.approx(2.0, abs=0.25)
    assert second.start_s == pytest.approx(5.0, abs=0.25)
    assert second.end_s == pytest.approx(6.0, abs=0.25)
    # Длительная тишина (2.0–5.0) в распознавание не попадает.
    assert second.start_s - first.end_s > 2.0


def test_vad_keeps_short_pauses_inside_speech(tmp_path):
    """Короткая пауза не разрывает речь и не теряет слова по обе стороны."""
    audio = make_audio(
        tmp_path / "short_pause.wav", spans=((1.0, 1.8), (2.1, 3.0)), total_s=4.0
    )
    intervals = run_vad(audio, VadConfig(min_silence_s=1.0), "energy").intervals
    assert len(intervals) == 1
    assert intervals[0].start_s == pytest.approx(1.0, abs=0.25)
    assert intervals[0].end_s == pytest.approx(3.0, abs=0.25)


def test_words_after_long_pause_have_absolute_timecodes(audio, energy_config):
    """5.2 (главный тест): слова после длинной паузы отсчитываются от начала записи.

    Проверка идёт против **независимо посчитанного** ожидания: фейковый
    бэкенд кладёт слова строго по ``word_s`` от начала куска, значит i-е
    слово второго интервала обязано стоять на ``intervals[1].start_s +
    i * word_s``. Раньше тест сверял только минимум с началом интервала —
    и проходил даже при полностью убранном пересчёте offset, потому что
    кламп в границы интервала подтягивал локальные времена к его началу
    (все слова схлопывались в одну точку, чего тест не замечал).
    """
    word_s = 0.5
    backend = FakeBackend("fake-abs", word_s=word_s)
    intervals = run_vad(audio, energy_config.vad, "energy").intervals
    assert len(intervals) == 2

    result = transcribe_speech(
        audio, config=energy_config, backend=backend, punctuator=FakePunctuator()
    )

    words = list(result.words)
    assert words, "фейковый бэкенд обязан выдать слова"
    assert is_monotonic(words)

    # Ни одно слово не выродилось в точку: схлопывание таймкодов — дефект,
    # а не «просто короткое слово».
    assert all(w.end_s > w.start_s for w in words)

    after_pause = [w for w in words if w.start_s >= intervals[1].start_s - 1e-6]
    before_pause = [w for w in words if w.start_s < intervals[1].start_s - 1e-6]
    assert before_pause and after_pause

    # Точная раскладка обоих кусков, посчитанная из параметров фейка,
    # а не из выхода стадии.
    for interval, chunk_words in ((intervals[0], before_pause), (intervals[1], after_pause)):
        duration_s = interval.end_s - interval.start_s
        expected_count = int(np.ceil(duration_s / word_s - 1e-9))
        assert len(chunk_words) == expected_count
        for index, word in enumerate(chunk_words):
            assert word.start_s == pytest.approx(
                interval.start_s + index * word_s, abs=0.01
            ), f"слово {index} интервала {interval} стоит не на своём абсолютном времени"
            assert word.end_s == pytest.approx(
                min(interval.start_s + (index + 1) * word_s, interval.end_s), abs=0.01
            )

    # Если бы таймкоды остались локальными, слова второго куска начались бы с 0.0.
    assert min(w.start_s for w in after_pause) > 4.0
    assert max(w.end_s for w in after_pause) <= TOTAL_S + 1e-6

    # В длительной тишине слов нет вообще.
    gap_start, gap_end = intervals[0].end_s, intervals[1].start_s
    assert not [w for w in words if gap_start < w.start_s < gap_end]


def test_transcribe_without_intervals_covers_whole_audio(audio):
    """Интервалы не заданы (None) — распознаётся вся запись целиком."""
    backend = FakeBackend("fake-full")
    result = backend.transcribe(audio, None)
    assert result.words
    assert result.words[0].start_s == pytest.approx(0.0, abs=1e-6)
    assert result.words[-1].end_s == pytest.approx(TOTAL_S, abs=0.05)
    assert not any("речь не обнаружена" in w for w in result.warnings)


def test_empty_vad_result_is_not_treated_as_whole_audio(audio):
    """Пустой результат VAD — это «речи нет», а не «распознать всё аудио»."""
    backend = FakeBackend("fake-empty")
    result = backend.transcribe(audio, [])
    assert result.words == ()
    assert any("речь не обнаружена" in w for w in result.warnings)


def test_silence_only_recording_yields_empty_transcription(tmp_path, energy_config, caplog):
    """Запись без речи не уходит в ASR целиком, а даёт пустой транскрипт с предупреждением."""
    silence = np.zeros(int(10.0 * SAMPLE_RATE), dtype=np.float32)
    path = write_wav_mono(tmp_path / "silence.wav", silence, SAMPLE_RATE)
    audio = AudioArtifact(
        path=path, sample_rate=SAMPLE_RATE, duration_s=10.0, source_track_index=0
    )
    backend = FakeBackend("fake-silence")

    assert run_vad(audio, energy_config.vad, "energy").intervals == ()

    with caplog.at_level(logging.WARNING):
        result = transcribe_speech(
            audio, config=energy_config, backend=backend, punctuator=FakePunctuator()
        )

    assert result.words == (), "тишина не должна порождать слов"
    assert any("речь не обнаружена" in w for w in result.warnings)
    assert "речь не обнаружена" in caplog.text


# --------------------------------------------------------------------------
# 5.2 — предел длины куска: длинный интервал VAD режется до входа в модель
# --------------------------------------------------------------------------


class ChunkSpyBackend(FakeBackend):
    """Фейк, который запоминает длительность каждого поданного куска."""

    def __init__(self, name: str = "chunk-spy", *, max_chunk_s: float | None = None) -> None:
        super().__init__(name)
        self.max_chunk_s = max_chunk_s
        self.chunk_durations_s: list[float] = []

    def _transcribe_chunk(self, samples, sample_rate, glossary):
        duration_s = samples.shape[0] / sample_rate
        self.chunk_durations_s.append(duration_s)
        # Настоящая gigaam 0.1.0 на длинном куске бросает ValueError.
        if self.max_chunk_s is not None and duration_s > 25.0:
            raise ValueError("Too long wav file, use 'transcribe_longform' method.")
        return super()._transcribe_chunk(samples, sample_rate, glossary)


def make_long_speech_audio(path: Path, total_s: float = 90.0) -> AudioArtifact:
    """Непрерывная речь без длинных пауз: VAD отдаст ОДИН интервал на всю запись."""
    rng = np.random.default_rng(20260910)
    count = int(round(total_s * SAMPLE_RATE))
    time = np.arange(count, dtype=np.float32) / SAMPLE_RATE
    samples = (0.5 * np.sin(2 * np.pi * 220.0 * time)).astype(np.float32)
    # Короткие (0.2 с) паузы между «словами»: они короче min_silence_s,
    # интервал не разрывают, но дают VAD-нарезке место для разреза.
    for start_s in np.arange(1.9, total_s, 2.0):
        mask = (time >= start_s) & (time < start_s + 0.2)
        samples[mask] = 0.0
    samples += rng.normal(0.0, 0.0005, count).astype(np.float32)
    write_wav_mono(path, samples, SAMPLE_RATE)
    return AudioArtifact(
        path=path, sample_rate=SAMPLE_RATE, duration_s=total_s, source_track_index=0
    )


def test_long_vad_interval_is_split_below_model_limit(tmp_path, energy_config):
    """Один длинный интервал VAD не уходит в модель целиком (иначе gigaam падает)."""
    audio = make_long_speech_audio(tmp_path / "long.wav", total_s=90.0)
    intervals = run_vad(audio, energy_config.vad, "energy").intervals
    assert len(intervals) == 1
    assert intervals[0].end_s - intervals[0].start_s > 25.0, (
        "фикстура обязана давать интервал длиннее предела модели"
    )

    backend = ChunkSpyBackend("chunk-spy", max_chunk_s=GIGAAM_MAX_CHUNK_S)
    result = transcribe_speech(
        audio, config=energy_config, backend=backend, punctuator=FakePunctuator()
    )

    assert backend.chunk_durations_s, "в модель должен был уйти хотя бы один кусок"
    assert max(backend.chunk_durations_s) <= GIGAAM_MAX_CHUNK_S + 1e-6
    assert result.words
    assert is_monotonic(result.words)
    # Таймкоды абсолютные и покрывают весь интервал, а не только первый кусок.
    assert result.words[0].start_s == pytest.approx(intervals[0].start_s, abs=0.6)
    assert result.words[-1].end_s == pytest.approx(intervals[0].end_s, abs=0.6)
    assert not any(w.end_s > audio.duration_s + 0.05 for w in result.words)


def test_split_interval_covers_interval_without_gaps():
    """Куски идут встык, покрывают интервал целиком и не длиннее предела."""
    samples = np.zeros(int(120.0 * SAMPLE_RATE), dtype=np.float32)
    interval = SpeechInterval(0.0, 5459.2)  # ровно то, что дал VAD на 91 минуте

    pieces = split_interval(samples, SAMPLE_RATE, interval, 20.0)

    assert len(pieces) > 250
    assert all(p.end_s - p.start_s <= 20.0 + 1e-6 for p in pieces)
    assert pieces[0].start_s == interval.start_s
    assert pieces[-1].end_s == interval.end_s
    for previous, current in zip(pieces[:-1], pieces[1:], strict=True):
        assert current.start_s == previous.end_s


def test_split_interval_prefers_pauses_between_words(tmp_path):
    """Разрез ищется по минимуму энергии — то есть попадает в паузу, а не в слово."""
    total_s = 12.0
    count = int(round(total_s * SAMPLE_RATE))
    time = np.arange(count, dtype=np.float32) / SAMPLE_RATE
    samples = (0.5 * np.sin(2 * np.pi * 220.0 * time)).astype(np.float32)
    pause_start, pause_end = 4.6, 4.9  # единственная пауза в окне поиска
    samples[(time >= pause_start) & (time < pause_end)] = 0.0

    pieces = split_interval(samples, SAMPLE_RATE, SpeechInterval(0.0, total_s), 5.0)

    assert len(pieces) == 3
    assert pause_start <= pieces[0].end_s <= pause_end, (
        f"разрез {pieces[0].end_s} попал не в паузу {pause_start}–{pause_end}"
    )


def test_split_interval_disabled_by_default():
    """Без предела длины интервал не режется — поведение бэкендов без лимита."""
    samples = np.zeros(SAMPLE_RATE * 10, dtype=np.float32)
    interval = SpeechInterval(0.0, 600.0)
    assert split_interval(samples, SAMPLE_RATE, interval, None) == [interval]
    assert IntervalAsrBackend.max_chunk_s is None


def test_gigaam_chunk_limit_below_model_threshold():
    """Предел GigaAM выбран с запасом к порогу 25 с из gigaam 0.1.0."""
    assert GigaAmBackend.max_chunk_s == GIGAAM_MAX_CHUNK_S
    assert 0 < GIGAAM_MAX_CHUNK_S <= 25.0 - 2.0


# --------------------------------------------------------------------------
# 5.5 — деградация при бэкенде без поддержки глоссария
# --------------------------------------------------------------------------


def test_backend_without_glossary_warns_and_succeeds(audio, energy_config, caplog):
    """5.5: прогон успешен, глоссарий не подан, предупреждение выдано."""
    backend = FakeBackend("no-glossary", supports_glossary=False)
    glossary = Glossary(terms=("гомоморфизм", "PostgreSQL"))

    with caplog.at_level(logging.WARNING):
        result = transcribe_speech(
            audio,
            glossary,
            config=energy_config,
            backend=backend,
            punctuator=FakePunctuator(),
        )

    assert result.words
    assert result.used_glossary is False
    assert any("не поддерживает глоссарий" in w for w in result.warnings)
    assert any("не поддерживает глоссарий" in r.message for r in caplog.records)
    assert backend.seen_glossaries == [None]


def test_backend_with_glossary_receives_it(audio, energy_config):
    backend = FakeBackend("with-glossary", supports_glossary=True)
    glossary = Glossary(terms=("гомоморфизм",))

    result = transcribe_speech(
        audio,
        glossary,
        config=energy_config,
        backend=backend,
        punctuator=FakePunctuator(),
    )

    assert result.used_glossary is True
    assert backend.seen_glossaries == [glossary]
    assert not any("не поддерживает глоссарий" in w for w in result.warnings)


def test_empty_glossary_runs_without_warning(audio, energy_config):
    backend = FakeBackend("no-glossary", supports_glossary=False)
    result = transcribe_speech(
        audio,
        Glossary(),
        config=energy_config,
        backend=backend,
        punctuator=FakePunctuator(),
    )
    assert result.words
    assert result.used_glossary is False
    assert not any("глоссарий" in w for w in result.warnings)


# --------------------------------------------------------------------------
# 5.6 — стадия пунктуации: текст меняется, таймкоды нет
# --------------------------------------------------------------------------


def test_punctuation_preserves_word_timecodes(audio, energy_config):
    """5.6: множество и таймкоды слов до и после стадии совпадают, текст меняется."""
    backend = FakeBackend("no-punct", provides_punctuation=False)
    punctuator = FakePunctuator()

    before = backend.transcribe(audio, run_vad(audio, energy_config.vad, "energy").intervals)
    after = transcribe_speech(
        audio, config=energy_config, backend=backend, punctuator=punctuator
    )

    assert punctuator.calls > 0
    assert after.has_punctuation is True
    assert len(after.words) == len(before.words)

    times_before = [(w.start_s, w.end_s) for w in before.words]
    times_after = [(w.start_s, w.end_s) for w in after.words]
    assert times_after == times_before

    # Сравнение по «ядру» (буквы/цифры, нижний регистр, ё->е): пунктуатор
    # имеет право добавить знаки и поменять регистр, но не состав слов.
    cores_before = [normalize_core(w.text) for w in before.words]
    cores_after = [normalize_core(w.text) for w in after.words]
    assert cores_after == cores_before

    joined = " ".join(w.text for w in after.words)
    assert joined != " ".join(w.text for w in before.words)
    assert any(ch in joined for ch in ".,")
    assert after.words[0].text[0].isupper()


def test_punctuation_skipped_for_backend_with_own_punctuation(audio, energy_config):
    """5.6: бэкенд со своей пунктуацией второй раз не пунктуируется."""
    backend = FakeBackend(
        "has-punct", provides_punctuation=True, vocabulary=("Привет,", "мир.")
    )
    punctuator = FakePunctuator()

    result = transcribe_speech(
        audio, config=energy_config, backend=backend, punctuator=punctuator
    )

    assert punctuator.calls == 0
    assert result.has_punctuation is True
    assert {w.text for w in result.words} == {"Привет,", "мир."}


def test_punctuation_skipped_when_punctuator_unavailable(audio, energy_config, caplog):
    backend = FakeBackend("no-punct")
    config = dataclasses.replace(energy_config, punctuator="нет-такого")

    with caplog.at_level(logging.WARNING):
        result = transcribe_speech(audio, config=config, backend=backend)

    assert result.words
    assert result.has_punctuation is False
    assert any("пунктуатор" in w for w in result.warnings)


def test_align_rejects_punctuator_that_changed_words():
    """Правило выравнивания: изменился состав слов -> пунктуация не применяется."""
    words = (
        Word("привет", 0.0, 0.4),
        Word("мир", 0.4, 0.8),
    )
    aligned, warnings = align_punctuated(words, "Привет!")
    assert aligned == list(words)
    assert warnings and "состав слов" in warnings[0]


def test_align_survives_merged_and_split_tokens():
    """Склейка/разбиение токенов моделью не меняет ни состав слов, ни таймкоды."""
    words = (
        Word("пере", 0.0, 0.2),
        Word("менная", 0.2, 0.5),
        Word("икс", 0.5, 0.9),
    )
    aligned, warnings = align_punctuated(words, "Переменная, икс.")
    assert warnings == []
    assert [(w.start_s, w.end_s) for w in aligned] == [(0.0, 0.2), (0.2, 0.5), (0.5, 0.9)]
    assert "".join(w.text for w in aligned).replace(" ", "").lower().startswith("переменная,")
    assert aligned[-1].text == "икс."


@pytest.mark.parametrize(
    ("tokens", "punctuated"),
    [
        (["это", "важно"], "Это — важно."),
        (["это", "важно"], "Это - важно."),
        (["ну", "да"], "Ну ... да."),
        (["привет", "мир"], "— Привет, мир."),
        (["он", "сказал", "нет"], "Он «сказал»: нет."),
        (["мы", "пришли", "увидели"], "Мы пришли — увидели!"),
        (["еж", "бежит"], "Ёж бежит..."),
        (["итак", "начнём"], "Итак: начнём?"),
        (["раз", "два", "три"], "Раз, два — три."),
        # входы аудита 2: закрывающие знаки и хвосты, раньше терявшиеся молча
        (["см", "рисунок", "два"], "См. рисунок ( два )."),
        (["он", "сказал", "да", "и", "ушел"], "Он сказал « да » и ушёл."),
        (["он", "сказал", "да"], "Он сказал „ да “ ."),
        (["и", "так"], "... И так."),
        (["и", "так"], "… и так."),
    ],
)
def test_align_keeps_every_punctuation_mark(tokens, punctuated):
    """Ни один знак препинания не пропадает при выравнивании (в т.ч. отдельно стоящий).

    Тире, дефис и многоточие, окружённые пробелами, раньше не доставались
    ни левому слову, ни правому и исчезали из текста молча.
    """
    words = tuple(
        Word(text=token, start_s=index * 0.4, end_s=index * 0.4 + 0.4)
        for index, token in enumerate(tokens)
    )

    aligned, warnings = align_punctuated(words, punctuated)

    assert warnings == []
    assert [(w.start_s, w.end_s) for w in aligned] == [(w.start_s, w.end_s) for w in words]
    joined = " ".join(w.text for w in aligned)
    assert normalize_core(joined) == normalize_core(punctuated)
    marks_in = sorted(ch for ch in punctuated if not ch.isalnum() and not ch.isspace())
    marks_out = sorted(ch for ch in joined if not ch.isalnum() and not ch.isspace())
    assert marks_out == marks_in, f"знаки потеряны или размножены: {joined!r}"


def test_restore_punctuation_reports_failed_chunk():
    words = [Word("привет", 0.0, 0.4), Word("мир", 0.4, 0.8)]
    result = restore_punctuation(words, FakePunctuator(drop_word=True))
    assert result.applied is False
    assert result.words == tuple(words)
    assert result.warnings


def test_punctuation_failure_is_loud(audio, energy_config, caplog):
    """Отказ пунктуации виден в warnings отдельным маркером и логируется как ошибка.

    Единственный зарегистрированный пунктуатор требует пакета, которого в
    боевом образе нет; без явного маркера транскрипт уезжает без знаков
    препинания, а причина теряется среди прочих предупреждений.
    """
    backend = FakeBackend("no-punct")
    config = dataclasses.replace(energy_config, punctuator="нет-такого")

    with caplog.at_level(logging.WARNING):
        result = transcribe_speech(audio, config=config, backend=backend)

    assert result.has_punctuation is False
    assert NO_PUNCTUATION_WARNING in result.warnings
    assert NO_PUNCTUATION_WARNING in caplog.text
    assert any(r.levelno >= logging.ERROR for r in caplog.records)


def test_default_punctuator_reason_names_real_install_source():
    """Причина недоступности обязана вести туда, где пакет реально лежит."""
    from lecture_transcript.speech_transcription.punctuation import SbertPuncCaseRu

    if importlib.util.find_spec("sbert_punc_case_ru") is not None:  # pragma: no cover
        # в образе пакет стоит; без весов в офлайне причина будет про веса
        pytest.skip("пакет пунктуатора установлен")
    availability = SbertPuncCaseRu().check_availability()
    # `pip install sbert_punc_case_ru` не работает: пакета нет на PyPI.
    assert "huggingface.co/kontur-ai/sbert_punc_case_ru" in availability.reason


def test_punctuated_flag_true_only_when_stage_worked(audio, energy_config):
    """Маркер отказа не ставится, когда пунктуация реально применена."""
    result = transcribe_speech(
        audio,
        config=energy_config,
        backend=FakeBackend("no-punct"),
        punctuator=FakePunctuator(),
    )
    assert result.has_punctuation is True
    assert NO_PUNCTUATION_WARNING not in result.warnings


# --------------------------------------------------------------------------
# 5.3 — used_glossary отражает факт подачи, а не декларацию бэкенда
# --------------------------------------------------------------------------


class _ModelWithoutHotwords:
    """Модель с сигнатурой gigaam 0.1.0: параметра hotwords нет вообще."""

    def transcribe(self, wav_file: str) -> str:  # noqa: ARG002
        return "привет мир"


def test_gigaam_drops_glossary_flag_when_hotwords_unsupported(tmp_path):
    """5.3: у gigaam 0.1.0 нет hotwords — used_glossary обязан стать False."""
    backend = GigaAmBackend()
    backend._model = _ModelWithoutHotwords()  # noqa: SLF001 — подмена модели вместо весов
    backend._hotwords_supported = backend._detect_hotwords_support(backend._model)  # noqa: SLF001
    assert backend._hotwords_supported is False  # noqa: SLF001

    backend._run_warnings = []  # noqa: SLF001
    backend._glossary_used = True  # noqa: SLF001 — состояние после старта прогона
    samples = np.zeros(SAMPLE_RATE, dtype=np.float32)

    text = backend._recognize(samples, SAMPLE_RATE, Glossary(terms=("гомоморфизм",)))  # noqa: SLF001

    assert text == "привет мир"
    assert backend._glossary_used is False, (  # noqa: SLF001
        "флаг обязан отражать факт подачи hotwords, а не supports_glossary"
    )
    assert any("hotwords" in w for w in backend._run_warnings)  # noqa: SLF001


def test_backend_that_cannot_pass_glossary_reports_false(audio, energy_config):
    """Бэкенд, объявивший поддержку, но не подавший глоссарий, не врёт в отчёте."""

    class DeclaredButNotPassed(FakeBackend):
        def _transcribe_chunk(self, samples, sample_rate, glossary):
            if glossary:
                self._glossary_rejected(
                    "версия библиотеки не принимает hotwords — глоссарий не подан"
                )
            return super()._transcribe_chunk(samples, sample_rate, None)

    backend = DeclaredButNotPassed("declared", supports_glossary=True)
    result = transcribe_speech(
        audio,
        Glossary(terms=("гомоморфизм",)),
        config=energy_config,
        backend=backend,
        punctuator=FakePunctuator(),
    )

    assert result.used_glossary is False
    assert any("не подан" in w for w in result.warnings)


# --------------------------------------------------------------------------
# Доступность silero и экономия памяти
# --------------------------------------------------------------------------


def test_silero_unavailable_without_its_package(monkeypatch):
    """torch есть, пакета silero-vad нет — бэкенд недоступен, а не «доступен через torch.hub»."""
    real_find_spec = importlib.util.find_spec
    seen: list[str] = []

    def fake_find_spec(name, *args, **kwargs):
        seen.append(name)
        if name == "torch":
            return object()  # torch «установлен» — типичная частичная установка
        if name == "silero_vad":
            return None
        return real_find_spec(name, *args, **kwargs)

    monkeypatch.setattr(vad_module.importlib.util, "find_spec", fake_find_spec)

    availability = check_vad_backend("silero")

    assert availability.available is False
    assert "silero-vad" in availability.reason
    # Проверка доступности не должна импортировать torch: find_spec на
    # подмодуле ("torch.hub") тянет родительский пакет в память.
    assert all("." not in name for name in seen), seen


def test_auto_vad_falls_back_to_energy_when_silero_missing(audio, monkeypatch):
    """Решение 2.3.2: в auto без silero берётся energy, а не падение в середине прогона."""
    real_find_spec = importlib.util.find_spec

    def fake_find_spec(name, *args, **kwargs):
        if name == "torch":
            return object()
        if name == "silero_vad":
            return None
        return real_find_spec(name, *args, **kwargs)

    monkeypatch.setattr(vad_module.importlib.util, "find_spec", fake_find_spec)

    result = run_vad(audio, VadConfig(), "auto")

    assert result.backend == "energy"
    assert any("silero-vad недоступен" in w for w in result.warnings)


def test_audio_is_read_from_disk_once_per_run(audio, energy_config, monkeypatch):
    """WAV читается один раз за прогон: VAD и ASR берут одни и те же сэмплы."""
    clear_audio_cache()
    opens: list[str] = []
    real_open = audio_io.wave.open

    def counting_open(path, mode="rb"):
        opens.append(str(path))
        return real_open(path, mode)

    monkeypatch.setattr(audio_io.wave, "open", counting_open)

    transcribe_speech(
        audio,
        config=energy_config,
        backend=FakeBackend("fake-read"),
        punctuator=FakePunctuator(),
    )

    assert opens.count(str(audio.path)) == 1, opens


def test_audio_cache_notices_changed_file(tmp_path):
    """Кэш привязан к mtime и размеру: подменённый файл читается заново."""
    clear_audio_cache()
    path = tmp_path / "clip.wav"
    write_wav_mono(path, np.zeros(SAMPLE_RATE, dtype=np.float32), SAMPLE_RATE)
    first, _ = audio_io.read_wav_mono(path)
    assert first.shape[0] == SAMPLE_RATE

    write_wav_mono(path, np.zeros(2 * SAMPLE_RATE, dtype=np.float32), SAMPLE_RATE)
    second, _ = audio_io.read_wav_mono(path)
    assert second.shape[0] == 2 * SAMPLE_RATE
    clear_audio_cache()


def test_energy_vad_does_not_materialize_squares(audio, monkeypatch):
    """Энергия считается без float64-копии всей записи и без массива квадратов."""

    def forbidden(*args, **kwargs):
        raise AssertionError("энергия не должна материализовать квадраты всей записи")

    monkeypatch.setattr(np, "square", forbidden)

    result = run_vad(audio, VadConfig(), "energy")

    assert len(result.intervals) == 2


def test_pipeline_releases_audio_cache(audio, energy_config):
    """После ASR сэмплы записи отпускаются — модель пунктуации грузится не поверх них."""
    transcribe_speech(
        audio,
        config=energy_config,
        backend=FakeBackend("fake-mem"),
        punctuator=FakePunctuator(),
    )
    assert audio_io._CACHE_VALUE is None  # noqa: SLF001


# --------------------------------------------------------------------------
# Проверки временно́й шкалы
# --------------------------------------------------------------------------


def test_validate_words_flags_zero_duration_words():
    """Слово, схлопнутое в точку, — дефект таймкодов, а не «просто короткое слово»."""
    words = [Word("привет", 1.0, 1.4), Word("мир", 2.0, 2.0)]
    problems = validate_words(words, duration_s=6.5)
    assert any("нулевой длительности" in p for p in problems)
    assert validate_words([Word("привет", 1.0, 1.4)], duration_s=6.5) == []


def test_sample_rate_mismatch_is_reported(tmp_path):
    """WAV не 16 кГц при заявленных 16 кГц — таймкоды поедут, об этом надо сказать."""
    path = tmp_path / "wrong_rate.wav"
    write_wav_mono(path, np.zeros(8000, dtype=np.float32), 8000)
    audio = AudioArtifact(
        path=path, sample_rate=SAMPLE_RATE, duration_s=1.0, source_track_index=0
    )
    clear_audio_cache()

    result = FakeBackend("fake-rate").transcribe(audio, [SpeechInterval(0.0, 1.0)])

    assert any("частота дискретизации" in w for w in result.warnings)


# --------------------------------------------------------------------------
# 5.7 — единство формата выхода между бэкендами
# --------------------------------------------------------------------------


def test_backends_share_output_schema(audio, energy_config):
    """5.7: два бэкенда с разными флагами дают идентичную схему результата."""
    plain = FakeBackend("plain", provides_punctuation=False, supports_glossary=True)
    punctuated = FakeBackend(
        "punctuated",
        provides_punctuation=True,
        supports_glossary=False,
        vocabulary=("Привет,", "мир.", "Это", "тест."),
    )
    glossary = Glossary(terms=("гомоморфизм",))

    results = [
        transcribe_speech(
            audio,
            glossary,
            config=energy_config,
            backend=backend,
            punctuator=FakePunctuator(),
        )
        for backend in (plain, punctuated)
    ]

    schema = {f.name: f.type for f in dataclasses.fields(Transcription)}
    for result in results:
        assert isinstance(result, Transcription)
        assert {f.name: f.type for f in dataclasses.fields(result)} == schema
        assert isinstance(result.words, tuple)
        assert isinstance(result.backend, str) and result.backend
        assert isinstance(result.has_punctuation, bool)
        assert isinstance(result.used_glossary, bool)
        assert isinstance(result.warnings, tuple)
        assert result.words, "оба бэкенда обязаны выдать слова"

        previous_end = 0.0
        for word in result.words:
            assert isinstance(word, Word)
            assert isinstance(word.text, str) and word.text
            assert isinstance(word.start_s, float) and isinstance(word.end_s, float)
            assert word.start_s >= previous_end - 1e-6, "слова не должны пересекаться"
            assert word.start_s <= word.end_s
            assert 0.0 <= word.start_s <= audio.duration_s
            assert word.end_s <= audio.duration_s + 1e-6
            previous_end = word.end_s
        assert is_monotonic(result.words)

    # Схема одна, а флаги — свои у каждого бэкенда.
    assert [r.has_punctuation for r in results] == [True, True]
    assert [r.used_glossary for r in results] == [True, False]
    assert len({len(r.words) for r in results}) == 1


def test_fake_backend_satisfies_protocol():
    assert isinstance(FakeBackend("proto"), AsrBackend)


def test_backend_unloaded_after_run(audio, energy_config):
    """D7: модель ASR выгружается до загрузки модели пунктуации."""
    backend = FakeBackend("unloadable")
    punctuator = FakePunctuator()
    transcribe_speech(audio, config=energy_config, backend=backend, punctuator=punctuator)
    assert backend.unloaded is True
    assert punctuator.unloaded is True


# --------------------------------------------------------------------------
# 5.8 — отсутствие сетевых вызовов
# --------------------------------------------------------------------------


class _NetworkBlocked(AssertionError):
    """Попытка выйти в сеть на этапе распознавания речи."""


@pytest.fixture
def no_network(monkeypatch):
    """Любое обращение к сокетам роняет тест."""

    def deny(*args, **kwargs):
        raise _NetworkBlocked("этап распознавания речи попытался выйти в сеть")

    monkeypatch.setattr(socket, "socket", deny)
    monkeypatch.setattr(socket, "create_connection", deny)
    monkeypatch.setattr(socket, "getaddrinfo", deny)
    return deny


def test_stage_runs_with_network_blocked(audio, energy_config, no_network):
    """5.8: при заблокированных сокетах стадия отрабатывает успешно."""
    backend = FakeBackend("offline-fake", supports_glossary=True)
    result = transcribe_speech(
        audio,
        Glossary(terms=("гомоморфизм",)),
        config=energy_config,
        backend=backend,
        punctuator=FakePunctuator(),
    )
    assert result.words
    assert result.has_punctuation is True
    assert is_monotonic(result.words)


def test_offline_mode_sets_and_restores_env(monkeypatch):
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    with offline_mode():
        assert all(key in os.environ for key in OFFLINE_ENV)
        assert os.environ["HF_HUB_OFFLINE"] == "1"
    assert "HF_HUB_OFFLINE" not in os.environ


def test_pipeline_enables_offline_mode(audio, energy_config, monkeypatch):
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    transcribe_speech(
        audio,
        config=energy_config,
        backend=FakeBackend("offline-fake"),
        punctuator=FakePunctuator(),
    )
    assert os.environ["HF_HUB_OFFLINE"] == "1"


# --------------------------------------------------------------------------
# Аудит, итерация 2
# --------------------------------------------------------------------------

#: Фейк пакета ``sbert_punc_case_ru``, повторяющий API пинованного коммита
#: f778dc6c (``sbert_punc_case_ru/sbertpunccase.py``). Прежний адаптер звал
#: ``SbertPuncCase.from_pretrained`` — у настоящего класса такого метода нет,
#: и удобный фейк с ``from_pretrained`` скрывал, что пунктуация не работает.
_SBERT_PACKAGE_SOURCE = '''
MODEL_REPO = "kontur-ai/sbert_punc_case_ru"
created = []


class _Module:
    """Как torch.nn.Module: to() возвращает self, from_pretrained нет."""

    def to(self, device):
        self.device = device
        return self


class SbertPuncCase(_Module):
    def __init__(self):  # без аргументов: веса MODEL_REPO грузятся внутри
        super().__init__()
        self.device = "cpu"
        created.append(self)

    def forward(self, input_ids, attention_mask):
        raise NotImplementedError

    def punctuate(self, text):
        words = text.strip().lower().split()
        if not words:
            return ""
        out = [w.capitalize() if i == 0 else w for i, w in enumerate(words)]
        out[-1] += "."
        return " ".join(out)
'''


@pytest.fixture
def fake_sbert_package(tmp_path, monkeypatch):
    """Подложить пакет с реальной сигнатурой ``SbertPuncCase`` вместо весов."""
    package = tmp_path / "fakepkgs" / "sbert_punc_case_ru"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text(
        "from .sbertpunccase import SbertPuncCase\n", encoding="utf-8"
    )
    (package / "sbertpunccase.py").write_text(_SBERT_PACKAGE_SOURCE, encoding="utf-8")
    monkeypatch.syspath_prepend(str(package.parent))
    for name in [m for m in sys.modules if m.startswith("sbert_punc_case_ru")]:
        monkeypatch.delitem(sys.modules, name)
    import sbert_punc_case_ru.sbertpunccase as module  # noqa: PLC0415

    yield module
    for name in [m for m in list(sys.modules) if m.startswith("sbert_punc_case_ru")]:
        sys.modules.pop(name, None)


def test_sbert_adapter_matches_real_package_api(fake_sbert_package):
    """Адаптер создаёт модель так, как её создаёт настоящий пакет: ``SbertPuncCase()``."""
    cls = fake_sbert_package.SbertPuncCase
    # Фейк обязан повторять реальную сигнатуру — иначе тест снова ничего не ловит.
    assert not hasattr(cls, "from_pretrained")
    assert list(inspect.signature(cls.__init__).parameters) == ["self"]

    adapter = SbertPuncCaseRu(device="cpu")
    assert adapter.restore("привет мир") == "Привет мир."
    assert adapter.restore("это лекция") == "Это лекция."
    assert len(fake_sbert_package.created) == 1, "модель грузится один раз"
    assert fake_sbert_package.created[0].device == "cpu"
    adapter.unload()


def test_sbert_adapter_applies_punctuation_on_every_chunk(fake_sbert_package, monkeypatch):
    """Через стадию: ни один кусок не падает, пунктуация применена."""
    adapter = SbertPuncCaseRu(device="cpu")
    monkeypatch.setattr(adapter, "check_availability", lambda: Availability(True))
    words = [
        Word(token, index * 0.4, index * 0.4 + 0.4)
        for index, token in enumerate(["привет", "мир", "это", "лекция"] * 40)
    ]

    result = restore_punctuation(words, adapter)

    assert result.applied is True
    assert not any("упал" in w for w in result.warnings), result.warnings
    assert not any(PARTIAL_PUNCTUATION_PREFIX in w for w in result.warnings)
    assert result.words[0].text == "Привет"


@pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="нет ffmpeg/ffprobe в окружении",
)
def test_cli_speech_stages_use_package_entrypoints(
    clip_speech, tmp_path, monkeypatch, capsys
):
    """Боевой путь через CLI (``--only-stage``): точки входа пакета, громкий отказ, очистка кэша.

    Адаптеры cli зовут доменные функции мимо ``transcribe_speech`` — на этом
    пути отказ пунктуации был тихим, а сэмплы записи висели в кэше после ASR.
    """
    from lecture_transcript import cli  # noqa: PLC0415
    import lecture_transcript.speech_transcription as speech_pkg  # noqa: PLC0415

    backend = FakeBackend("fake-cli-asr")
    register("fake-cli-asr", lambda: backend)

    class BrokenPunctuator(FakePunctuator):
        name = "broken-punctuator"

        def restore(self, text: str) -> str:
            self.calls += 1
            raise RuntimeError("модель пунктуации не загрузилась")

    broken = BrokenPunctuator()
    monkeypatch.setitem(
        punctuation_module._PUNCTUATORS, DEFAULT_PUNCTUATOR_NAME, lambda: broken
    )
    monkeypatch.delitem(
        punctuation_module._INSTANCES, DEFAULT_PUNCTUATOR_NAME, raising=False
    )

    def adapter_called(ctx):
        raise AssertionError(
            f"стадия {ctx.stage}: вызван адаптер cli, а не точка входа пакета"
        )

    for stage in ("vad", "asr", "punctuation"):
        monkeypatch.setitem(cli._STAGE_ADAPTERS, stage, adapter_called)

    calls: list[str] = []
    payloads: dict[str, dict] = {}
    cache_after_asr: list[bool] = []
    for attr in ("run_vad_stage", "run_asr_stage", "run_punctuation_stage"):
        real = getattr(speech_pkg, attr)

        def spy(ctx, _real=real, _attr=attr):
            calls.append(_attr)
            payload = _real(ctx)
            payloads[_attr] = payload
            if _attr == "run_asr_stage":
                cache_after_asr.append(audio_io._CACHE_VALUE is None)
            return payload

        monkeypatch.setattr(speech_pkg, attr, spy)

    # Ветка слайдов здесь не проверяется: заглушки вместо frames..glossary.
    original = cli._stage_callable
    stubs = {
        "frames": lambda ctx: {"frames": None},
        "slides": lambda ctx: {"slides": []},
        "ocr": lambda ctx: {"ocr": []},
        "glossary": lambda ctx: {"terms": ["гомоморфизм"]},
    }
    monkeypatch.setattr(
        cli,
        "_stage_callable",
        lambda stage: stubs[stage] if stage in stubs else original(stage),
    )
    monkeypatch.delitem(cli._STAGE_MODEL_HOLDERS, "ocr", raising=False)

    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "speech_transcription:\n  vad_backend: energy\n  asr_backend: fake-cli-asr\n",
        encoding="utf-8",
    )
    base = [
        str(clip_speech), "--out-dir", str(tmp_path / "out"),
        "--config", str(config_path), "--log-level", "INFO",
    ]
    for stage in ("audio", "frames", "slides", "ocr", "glossary"):
        assert cli.main([*base, "--only-stage", stage]) == 0, stage
    capsys.readouterr()

    assert cli.main([*base, "--only-stage", "vad"]) == 0
    assert cli.main([*base, "--only-stage", "asr"]) == 0
    asr_log = capsys.readouterr().err
    assert cli.main([*base, "--only-stage", "punctuation"]) == 0
    punctuation_log = capsys.readouterr().err

    assert calls == ["run_vad_stage", "run_asr_stage", "run_punctuation_stage"]
    assert cache_after_asr == [True], "сэмплы записи остались в кэше после ASR"
    assert audio_io._CACHE_VALUE is None
    assert backend.transcribe_calls == 1
    assert broken.calls > 0

    # Форма нагрузок — как у адаптеров cli: её читает merge.
    assert set(payloads["run_vad_stage"]) == {"intervals"}
    assert payloads["run_vad_stage"]["intervals"]
    transcription_keys = {"backend", "has_punctuation", "used_glossary", "warnings", "words"}
    assert set(payloads["run_asr_stage"]) == transcription_keys
    assert set(payloads["run_punctuation_stage"]) == transcription_keys

    # 5.5 на боевом пути: глоссарий из слайдов есть, бэкенд его не умеет.
    assert "не поддерживает глоссарий" in asr_log
    assert payloads["run_asr_stage"]["used_glossary"] is False

    # Громкий отказ пунктуации дошёл и до лога, и до нагрузки.
    assert NO_PUNCTUATION_WARNING in punctuation_log
    assert "ERROR" in punctuation_log
    assert payloads["run_punctuation_stage"]["has_punctuation"] is False
    assert NO_PUNCTUATION_WARNING in payloads["run_punctuation_stage"]["warnings"]


class _FlakyPunctuator(FakePunctuator):
    """Отрабатывает первый кусок и падает на остальных (как при OOM посреди прогона)."""

    name = "flaky"

    def __init__(self) -> None:
        super().__init__()
        self.attempts = 0

    def restore(self, text: str) -> str:
        self.attempts += 1
        if self.attempts > 1:
            raise RuntimeError("CUDA out of memory")
        return super().restore(text)


def test_partial_punctuation_failure_is_as_loud_as_total(caplog):
    """Часть кусков упала — маркер с числом кусков в warnings и ERROR в логе."""
    words = [Word(f"слово{i}", i * 0.3, i * 0.3 + 0.3) for i in range(64 * 5)]

    with caplog.at_level(logging.WARNING):
        result = restore_punctuation(words, _FlakyPunctuator())

    assert result.applied is True
    assert partial_punctuation_warning(4, 5) in result.warnings
    assert any(
        r.levelno >= logging.ERROR and PARTIAL_PUNCTUATION_PREFIX in r.getMessage()
        for r in caplog.records
    )


def test_punctuator_returning_input_unchanged_is_a_failure():
    """Модель вернула текст как есть — это не «пунктуация применена»."""

    class Echo(FakePunctuator):
        def restore(self, text: str) -> str:
            return text

    words = [Word("привет", 0.0, 0.4), Word("мир", 0.4, 0.8)]
    result = restore_punctuation(words, Echo())
    assert result.applied is False
    assert NO_PUNCTUATION_WARNING in result.warnings


def test_unavailable_passed_punctuator_does_not_lose_asr_result(audio, energy_config):
    """Недоступный подставной пунктуатор не роняет стадию после ASR."""
    result = transcribe_speech(
        audio,
        config=energy_config,
        backend=FakeBackend("keep-asr"),
        punctuator=FakePunctuator(available=False),
    )
    assert result.words
    assert result.has_punctuation is False
    assert NO_PUNCTUATION_WARNING in result.warnings


def test_broken_punctuator_factory_is_reported_as_unavailable(monkeypatch):
    def factory():
        raise RuntimeError("ctor failed")

    monkeypatch.setitem(punctuation_module._PUNCTUATORS, "boom", factory)
    availability = check_punctuator("boom")
    assert availability.available is False
    assert "ctor failed" in availability.reason


class _GigaAmWithoutWeights(GigaAmBackend):
    """GigaAmBackend с моделью сигнатуры gigaam 0.1.0 вместо весов."""

    def check_availability(self) -> Availability:
        return Availability(True)

    def _prepare(self, sample_rate: int) -> None:
        self._model = _ModelWithoutHotwords()
        self._hotwords_supported = self._detect_hotwords_support(self._model)


def test_gigaam_runs_without_glossary_with_degradation_warning(audio, energy_config):
    """Решение по 5.3: gigaam 0.1.0 глоссарий не принимает — работает сценарий 5.5."""
    assert GigaAmBackend.supports_glossary is False

    result = transcribe_speech(
        audio,
        Glossary(terms=("гомоморфизм",)),
        config=energy_config,
        backend=_GigaAmWithoutWeights(),
        punctuator=FakePunctuator(),
    )

    assert result.words
    assert result.used_glossary is False
    assert any("не поддерживает глоссарий" in w for w in result.warnings)


def _pretend_packages(monkeypatch, *names: str) -> None:
    real = importlib.util.find_spec
    monkeypatch.setattr(
        importlib.util,
        "find_spec",
        lambda name, *a, **kw: object() if name in names else real(name, *a, **kw),
    )


def test_gigaam_offline_without_weights_is_unavailable(monkeypatch, tmp_path):
    """gigaam качает с CDN Сбера мимо HF: HF_HUB_OFFLINE его не остановит,
    поэтому отсутствие весов в офлайне ловится до прогона."""
    _pretend_packages(monkeypatch, "torch", "gigaam")
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))

    with offline_mode():
        availability = GigaAmBackend().check_availability()
    assert availability.available is False
    assert str(tmp_path / "gigaam" / "v2_rnnt.ckpt") in availability.reason

    # онлайн (окно прогрева) — доступен: веса докачаются при загрузке
    monkeypatch.setenv("HF_HUB_OFFLINE", "0")
    assert GigaAmBackend().check_availability().available is True

    (tmp_path / "gigaam").mkdir()
    (tmp_path / "gigaam" / "v2_rnnt.ckpt").write_bytes(b"x")
    with offline_mode():
        assert GigaAmBackend().check_availability().available is True


def test_gigaam_downloads_into_model_cache(monkeypatch, tmp_path):
    """По умолчанию gigaam пишет в ~/.cache/gigaam — в контейнере это слой,
    который `run --rm` выбрасывает. Каталог обязан идти из XDG_CACHE_HOME."""
    import types

    calls: list[dict] = []
    fake = types.ModuleType("gigaam")
    fake.load_model = lambda name, **kwargs: calls.append(kwargs) or _ModelWithoutHotwords()
    monkeypatch.setitem(sys.modules, "gigaam", fake)
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))

    GigaAmBackend()._prepare(SAMPLE_RATE)  # noqa: SLF001
    assert calls == [{"download_root": str(tmp_path / "gigaam")}]


def test_punctuator_offline_without_weights_is_unavailable(monkeypatch, tmp_path):
    _pretend_packages(monkeypatch, "torch", "transformers", "sbert_punc_case_ru")
    monkeypatch.setenv("HF_HOME", str(tmp_path))
    for variable in ("HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE"):
        monkeypatch.delenv(variable, raising=False)

    with offline_mode():
        availability = SbertPuncCaseRu().check_availability()
    assert availability.available is False
    assert "kontur-ai/sbert_punc_case_ru" in availability.reason
    assert "warmup" in availability.reason

    snapshot = tmp_path / "hub" / "models--kontur-ai--sbert_punc_case_ru" / "snapshots" / "f778dc6"
    snapshot.mkdir(parents=True)
    with offline_mode():
        assert SbertPuncCaseRu().check_availability().available is True


def test_used_glossary_false_when_nothing_was_recognized(audio):
    """Флаг ставится в момент подачи: нет кусков — глоссарий никуда не подан."""
    backend = FakeBackend("glossary-empty", supports_glossary=True)
    result = backend.transcribe(audio, [], Glossary(terms=("гомоморфизм",)))
    assert result.used_glossary is False


def test_asr_releases_audio_cache_on_direct_call(audio):
    """Кэш сэмплов отпускает сам ASR — при любом пути вызова, не только в transcribe_speech."""
    clear_audio_cache()
    audio_io.read_wav_mono(audio.path)
    assert audio_io._CACHE_VALUE is not None

    FakeBackend("direct").transcribe(audio, [SpeechInterval(1.0, 2.0)])

    assert audio_io._CACHE_VALUE is None


def test_asr_releases_audio_cache_on_failure(audio):
    class Exploding(FakeBackend):
        def _transcribe_chunk(self, samples, sample_rate, glossary):
            raise RuntimeError("модель упала")

    clear_audio_cache()
    with pytest.raises(RuntimeError):
        Exploding("exploding").transcribe(audio, [SpeechInterval(1.0, 2.0)])
    assert audio_io._CACHE_VALUE is None


def test_audio_cache_not_fooled_by_same_size_and_mtime(tmp_path):
    """Подмена файла с тем же размером и восстановленным mtime читается заново."""
    clear_audio_cache()
    path = tmp_path / "clip.wav"
    time = np.arange(2 * SAMPLE_RATE, dtype=np.float32) / SAMPLE_RATE
    write_wav_mono(path, (0.5 * np.sin(2 * np.pi * 220.0 * time)).astype(np.float32), SAMPLE_RATE)
    original = path.stat()
    first, _ = audio_io.read_wav_mono(path)
    assert float(np.abs(first).max()) > 0.4

    write_wav_mono(path, np.zeros(2 * SAMPLE_RATE, dtype=np.float32), SAMPLE_RATE)
    os.utime(path, ns=(original.st_atime_ns, original.st_mtime_ns))
    assert path.stat().st_size == original.st_size
    assert path.stat().st_mtime_ns == original.st_mtime_ns

    second, _ = audio_io.read_wav_mono(path)
    assert float(np.abs(second).max()) == 0.0
    clear_audio_cache()


def test_split_interval_finds_pause_far_from_chunk_end():
    """Пауза в середине окна, а не в последних 0.75 с, тоже находится."""
    total_s = 30.0
    count = int(round(total_s * SAMPLE_RATE))
    time = np.arange(count, dtype=np.float32) / SAMPLE_RATE
    samples = (0.5 * np.sin(2 * np.pi * 220.0 * time)).astype(np.float32)
    pause_start, pause_end = 12.0, 12.3
    samples[(time >= pause_start) & (time < pause_end)] = 0.0

    pieces = split_interval(samples, SAMPLE_RATE, SpeechInterval(0.0, total_s), 20.0)

    assert pause_start <= pieces[0].end_s <= pause_end, pieces
    assert all(p.end_s - p.start_s <= 20.0 + 1e-6 for p in pieces)


def test_align_keeps_all_marks_on_generated_inputs():
    """Свойство «ни один знак не потерян» на сгенерированных входах, а не только на таблице."""
    rng = random.Random(20260910)
    vocabulary = ["это", "важно", "мир", "лекция", "тест", "он", "сказал", "да", "ёж"]
    marks = [",", ".", "!", "?", ":", ";", "...", "…", "—", "-", "«", "»", "„", "“",
             "(", ")", '"', "%", "*"]
    for _ in range(2000):
        words = [
            Word(text=token.replace("ё", "е"), start_s=i * 0.4, end_s=i * 0.4 + 0.4)
            for i, token in enumerate(rng.choices(vocabulary, k=rng.randint(1, 6)))
        ]
        parts: list[str] = []
        for word in words:
            if rng.random() < 0.3:
                parts.append(rng.choice(marks))
            token = word.text.capitalize() if rng.random() < 0.3 else word.text
            if rng.random() < 0.3:
                token = rng.choice(marks) + token
            if rng.random() < 0.4:
                token += rng.choice(marks)
            parts.append(token)
            if rng.random() < 0.3:
                parts.append(rng.choice(marks))
        text = " ".join(parts)

        aligned, warnings = align_punctuated(words, text)

        assert warnings == [], text
        assert [(w.start_s, w.end_s) for w in aligned] == [(w.start_s, w.end_s) for w in words]
        joined = " ".join(w.text for w in aligned)
        assert normalize_core(joined) == normalize_core(text), (text, joined)
        marks_in = sorted(ch for ch in text if not ch.isalnum() and not ch.isspace())
        marks_out = sorted(ch for ch in joined if not ch.isalnum() and not ch.isspace())
        assert marks_out == marks_in, (text, joined)

