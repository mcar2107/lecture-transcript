"""Проверки артефактов окружения из `docker/` (задачи 1.1 и 1.2).

Тесты статические: они не собирают образ и не требуют CUDA — только следят,
что артефакты на месте, синтаксически валидны и что пины версий не размылись
до `>=` при будущих правках. Фактическая проверка «nvidia-smi видит RTX 4060 /
torch видит GPU» живёт в `docker/verify_env.sh` и выполняется на целевой машине.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import pytest
import yaml

DOCKER_DIR = Path(__file__).resolve().parent.parent / "docker"

REQUIREMENT_FILES = (
    "requirements.txt",
    "requirements-torch.txt",
    "requirements-paddle.txt",
    "requirements-dev.txt",
)

# Строка требования: имя[extras]==версия. Комментарии и опции индекса отсеиваются.
_REQ_LINE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*(\[[A-Za-z0-9,._-]+\])?==")

# VCS-требование считается зафиксированным, только если ссылается на полный
# хэш коммита: имя @ git+<url>@<40 hex>. Ветка или тег — не пин.
_VCS_LINE = re.compile(
    r"^[A-Za-z0-9][A-Za-z0-9._-]*\s+@\s+git\+\S+@[0-9a-f]{40}$"
)


def _requirement_lines(path: Path) -> list[str]:
    lines = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or line.startswith("-"):
            continue
        lines.append(line)
    return lines


def test_all_artifacts_present() -> None:
    expected = {
        "Dockerfile",
        "compose.yaml",
        "verify_env.sh",
        "README.md",
        *REQUIREMENT_FILES,
    }
    missing = sorted(name for name in expected if not (DOCKER_DIR / name).is_file())
    assert not missing, f"нет артефактов окружения: {missing}"


@pytest.mark.parametrize("name", REQUIREMENT_FILES)
def test_versions_are_pinned_exactly(name: str) -> None:
    """Каждая зависимость зафиксирована: `==` либо git-ссылка на хэш коммита."""
    loose = [
        line
        for line in _requirement_lines(DOCKER_DIR / name)
        if not (_REQ_LINE.match(line) or _VCS_LINE.match(line))
    ]
    assert not loose, f"{name}: не зафиксированы жёстко: {loose}"


def test_external_indexes_declared() -> None:
    """Пакеты не с PyPI лежат в отдельных файлах со своим индексом."""
    torch_txt = (DOCKER_DIR / "requirements-torch.txt").read_text(encoding="utf-8")
    assert "--index-url https://download.pytorch.org/whl/cu124" in torch_txt
    assert "torch==2.5.1+cu124" in torch_txt

    paddle_txt = (DOCKER_DIR / "requirements-paddle.txt").read_text(encoding="utf-8")
    assert "--index-url https://www.paddlepaddle.org.cn/packages/stable/cu126/" in paddle_txt

    # В основном файле внешних индексов быть не должно.
    main_txt = (DOCKER_DIR / "requirements.txt").read_text(encoding="utf-8")
    assert "--index-url" not in main_txt
    assert "--extra-index-url" not in main_txt


def test_compose_is_valid_and_requests_gpu() -> None:
    compose = yaml.safe_load((DOCKER_DIR / "compose.yaml").read_text(encoding="utf-8"))
    service = compose["services"]["lecture-transcript"]

    assert service["gpus"] == "all"

    caps = service["environment"]["NVIDIA_DRIVER_CAPABILITIES"]
    # Без `video` в контейнер не попадает libnvcuvid и nvdec не работает.
    assert "video" in caps.split(",")

    joined = " ".join(str(v) for v in service["volumes"])
    assert "/data/input" in joined, "не смонтирован входной каталог"
    assert "/cache" in joined, "не смонтирован каталог кэша моделей"


def _dockerfile_instructions(text: str) -> list[tuple[int, str]]:
    """Логические инструкции Dockerfile: переносы через `\\` склеены.

    Возвращает пары (номер первой физической строки, текст инструкции).
    Комментарии внутри продолжений выбрасываются — Docker их игнорирует.
    """
    instructions: list[tuple[int, str]] = []
    current: list[str] = []
    start = 0
    for number, raw in enumerate(text.splitlines()):
        line = raw.strip()
        if not current:
            if not line or line.startswith("#"):
                continue
            start = number
        elif line.startswith("#"):
            continue
        continued = line.endswith("\\")
        current.append(line[:-1].strip() if continued else line)
        if not continued:
            instructions.append((start, " ".join(current)))
            current = []
    if current:
        instructions.append((start, " ".join(current)))
    return instructions


def test_dockerfile_pins_base_image_and_nvdec_env() -> None:
    text = (DOCKER_DIR / "Dockerfile").read_text(encoding="utf-8")
    assert "nvidia/cuda:12.6.3-cudnn-runtime-ubuntu22.04" in text
    assert "NVIDIA_DRIVER_CAPABILITIES=compute,utility,video" in text


def test_torch_is_installed_before_pypi_set() -> None:
    """torch с индекса PyTorch ставится РАНЬШЕ набора с PyPI.

    Нарушение порядка означает установку CPU-сборки torch и неработающий GPU.
    Сравниваются позиции именно `RUN pip install`-строк: `COPY` перечисляет
    requirements-torch.txt первым всегда, и сравнение с ним ничего не проверяет.
    """
    text = (DOCKER_DIR / "Dockerfile").read_text(encoding="utf-8")
    # Инструкции переносятся через `\`, поэтому склеиваем логические строки:
    # шаг установки torch занимает две физические.
    instructions = _dockerfile_instructions(text)
    run_lines = [
        (i, line)
        for i, line in instructions
        if line.startswith("RUN ") and "pip install" in line
    ]

    def position(fragment: str) -> int:
        matches = [i for i, line in run_lines if fragment in line]
        assert matches, f"нет шага `RUN pip install` с {fragment!r}"
        assert len(matches) == 1, f"шагов с {fragment!r} больше одного: {matches}"
        return matches[0]

    torch_step = position("requirements-torch.txt")
    pypi_step = position("/tmp/reqs/requirements.txt")
    paddle_step = position("requirements-paddle.txt")

    assert torch_step < pypi_step, (
        "torch с индекса PyTorch должен ставиться до набора с PyPI, "
        f"сейчас строки {torch_step} и {pypi_step}"
    )
    assert torch_step < paddle_step

    # opencv-contrib переустанавливается последним — иначе победитель в `cv2`
    # зависит от порядка резолва pip.
    cv_step = position("opencv-contrib-python==4.10.0.84")
    assert cv_step > pypi_step and cv_step > paddle_step


def test_git_dependency_installed_without_lfs_payload() -> None:
    """VCS-пин sbert_punc_case_ru ставится с GIT_LFS_SKIP_SMUDGE=1.

    Иначе pip затянет в слой образа model.safetensors (~1.7 GB) из того же
    репозитория модели на HuggingFace.
    """
    requirements = (DOCKER_DIR / "requirements.txt").read_text(encoding="utf-8")
    assert "sbert_punc_case_ru @ git+https://huggingface.co/" in requirements

    dockerfile = (DOCKER_DIR / "Dockerfile").read_text(encoding="utf-8")
    install_line = next(
        line
        for line in dockerfile.splitlines()
        if line.startswith("RUN ") and "/tmp/reqs/requirements.txt" in line
    )
    assert "GIT_LFS_SKIP_SMUDGE=1" in install_line
    # Сборка из git требует самого git в образе.
    assert re.search(r"^\s+git \\$", dockerfile, re.MULTILINE)


def test_build_context_is_trimmed() -> None:
    """Есть .dockerignore, и он отсекает тяжёлое из контекста сборки.

    BuildKit читает `<путь-к-Dockerfile>.dockerignore` раньше корневого
    `.dockerignore`, поэтому файл лежит рядом с Dockerfile.
    """
    ignore = DOCKER_DIR / "Dockerfile.dockerignore"
    assert ignore.is_file(), "нет docker/Dockerfile.dockerignore"
    patterns = {
        line.strip()
        for line in ignore.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    }
    for required in (".venv/", ".git/", "tests/fixtures/", "**/__pycache__/"):
        assert required in patterns, f"в .dockerignore нет {required}"


@pytest.mark.skipif(shutil.which("bash") is None, reason="нет bash")
def test_verify_env_script_syntax() -> None:
    script = DOCKER_DIR / "verify_env.sh"
    result = subprocess.run(
        ["bash", "-n", str(script)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


# --------------------------------------------------------------------------
# Поведенческие проверки verify_env.sh
#
# Скрипт запускается с подменённым PATH: вместо nvidia-smi/ffmpeg подставляются
# заглушки, которые воспроизводят конкретный сценарий отказа. Проверяется код
# возврата и то, что нужная проверка даёт именно [FAIL], а не [WARN].
# GPU для этого не нужна — что и позволяет гонять тесты на dev-машине.
# --------------------------------------------------------------------------

_NVIDIA_SMI_OK = """#!/bin/sh
case " $* " in
    *" -L "*) echo "GPU 0: NVIDIA GeForce RTX 4060 (UUID: GPU-0000)"; exit 0;;
esac
echo "NVIDIA GeForce RTX 4060, 8188 MiB, 560.00"
exit 0
"""

_LDCONFIG_OK = """#!/bin/sh
echo "	libnvcuvid.so.1 (libc6,x86-64) => /usr/lib/x86_64-linux-gnu/libnvcuvid.so.1"
exit 0
"""

# hwaccels и decoders в порядке, а любой реальный вызов (энкод/декод) падает.
_FFMPEG_NO_ENCODE = """#!/bin/sh
case " $* " in
    *" -hwaccels "*) echo "Hardware acceleration methods:"; echo "cuda"; exit 0;;
    *" -decoders "*) echo " V..... h264_cuvid  Nvidia CUVID H264 decoder"; exit 0;;
esac
echo "Unknown encoder 'libx264'" >&2
exit 1
"""

# cuda отсутствует в списке hwaccels.
_FFMPEG_NO_CUDA = """#!/bin/sh
case " $* " in
    *" -hwaccels "*) echo "Hardware acceleration methods:"; echo "vaapi"; exit 0;;
    *" -decoders "*) echo " V..... h264  H.264 decoder"; exit 0;;
esac
exit 0
"""


#: Утилиты, наличие которых проверяет verify_env.sh. В тестах они есть только
#: как заглушки: внутри GPU-контейнера настоящие nvidia-smi и ffmpeg лежат в
#: /usr/bin, и «пустое окружение» без этой изоляции видело бы их.
_HOST_TOOLS_HIDDEN = frozenset({"nvidia-smi", "ffmpeg", "ffprobe"})


def _system_bin(tmp_path: Path) -> Path:
    """Копия /usr/bin и /bin из симлинков — без утилит из `_HOST_TOOLS_HIDDEN`."""
    sysbin = tmp_path / "sysbin"
    sysbin.mkdir(exist_ok=True)
    for source in (Path("/usr/bin"), Path("/bin")):
        if not source.is_dir():
            continue
        for tool in source.iterdir():
            link = sysbin / tool.name
            if tool.name in _HOST_TOOLS_HIDDEN or link.exists() or link.is_symlink():
                continue
            link.symlink_to(tool)
    return sysbin


def _run_verify_env(tmp_path: Path, stubs: dict[str, str]) -> subprocess.CompletedProcess:
    stub_dir = tmp_path / "bin"
    stub_dir.mkdir(parents=True, exist_ok=True)
    for name, body in stubs.items():
        stub = stub_dir / name
        stub.write_text(body, encoding="utf-8")
        stub.chmod(0o755)
    env = {
        "PATH": f"{stub_dir}:{_system_bin(tmp_path)}",
        "HOME": str(tmp_path),
        "TMPDIR": str(tmp_path),
    }
    return subprocess.run(
        ["bash", str(DOCKER_DIR / "verify_env.sh")],
        capture_output=True,
        text=True,
        env=env,
        timeout=180,
        check=False,
    )


def _fail_lines(output: str) -> list[str]:
    return [line for line in output.splitlines() if "[FAIL]" in line]


@pytest.mark.skipif(shutil.which("bash") is None, reason="нет bash")
def test_verify_env_exits_nonzero_without_gpu(tmp_path: Path) -> None:
    """Пустое окружение: ни nvidia-smi, ни ffmpeg — код возврата 1."""
    result = _run_verify_env(tmp_path, {})
    assert result.returncode == 1, result.stdout
    assert any("nvidia-smi" in line for line in _fail_lines(result.stdout))


@pytest.mark.skipif(shutil.which("bash") is None, reason="нет bash")
def test_verify_env_fails_when_hwaccels_lacks_cuda(tmp_path: Path) -> None:
    """`ffmpeg -hwaccels` без cuda — провал проверки задачи 1.1."""
    result = _run_verify_env(
        tmp_path,
        {
            "nvidia-smi": _NVIDIA_SMI_OK,
            "ldconfig": _LDCONFIG_OK,
            "ffmpeg": _FFMPEG_NO_CUDA,
            "ffprobe": "#!/bin/sh\nexit 0\n",
        },
    )
    assert result.returncode == 1
    assert any("hwaccels" in line for line in _fail_lines(result.stdout)), result.stdout


@pytest.mark.skipif(shutil.which("bash") is None, reason="нет bash")
def test_verify_env_fails_when_nvdec_smoke_test_cannot_run(tmp_path: Path) -> None:
    """Невыполнимая сквозная проверка nvdec — FAIL, а не WARN.

    Косвенные проверки (-hwaccels, -decoders, ldconfig) здесь проходят: именно
    в этом сценарии раньше скрипт мог завершиться нулём, не потрогав nvdec.
    """
    result = _run_verify_env(
        tmp_path,
        {
            "nvidia-smi": _NVIDIA_SMI_OK,
            "ldconfig": _LDCONFIG_OK,
            "ffmpeg": _FFMPEG_NO_ENCODE,
            "ffprobe": "#!/bin/sh\nexit 0\n",
        },
    )
    assert result.returncode == 1
    smoke_fail = [line for line in _fail_lines(result.stdout) if "ролик" in line]
    assert smoke_fail, f"сквозная проверка nvdec не дала [FAIL]:\n{result.stdout}"
    assert not [
        line
        for line in result.stdout.splitlines()
        if "[WARN]" in line and "ролик" in line
    ], "сквозная проверка nvdec деградировала в WARN"


@pytest.mark.skipif(shutil.which("bash") is None, reason="нет bash")
def test_verify_env_checks_punctuator_and_vlm_imports() -> None:
    """Скрипт проверяет пакеты, без которых стадии 5.6 и 4.8 нерабочие."""
    text = (DOCKER_DIR / "verify_env.sh").read_text(encoding="utf-8")
    for probe in (
        "sbert_punc_case_ru",
        "Qwen2_5_VLForConditionalGeneration",
        "accelerate",
        "bitsandbytes",
    ):
        assert probe in text, f"в verify_env.sh нет проверки импорта: {probe}"
