"""Логирование пайплайна: логи стадий не зависят от уровня корневого логгера."""

from __future__ import annotations

import io
import logging

import pytest

from lecture_transcript.logging_setup import setup_logging


@pytest.fixture
def restore_logging():
    root = logging.getLogger()
    package = logging.getLogger("lecture_transcript")
    saved = (list(root.handlers), root.level, package.level)
    yield
    root.handlers[:] = saved[0]
    root.setLevel(saved[1])
    package.setLevel(saved[2])


def test_логи_стадий_переживают_смену_уровня_корня_библиотекой(restore_logging):
    # `import paddle` ставит корню WARNING, pix2tex — CRITICAL: на реальном
    # прогоне после старта OCR пропадали все логи стадий.
    stream = io.StringIO()
    setup_logging("INFO", stream=stream)

    logging.getLogger().setLevel(logging.CRITICAL)
    logging.getLogger("lecture_transcript.cli").info("стадия asr: старт")

    assert "стадия asr: старт" in stream.getvalue()


def test_уровень_пакета_задаётся_флагом(restore_logging):
    stream = io.StringIO()
    setup_logging("WARNING", stream=stream)

    logger = logging.getLogger("lecture_transcript.cli")
    logger.info("скрыто")
    logger.warning("видно")

    assert "скрыто" not in stream.getvalue()
    assert "видно" in stream.getvalue()
