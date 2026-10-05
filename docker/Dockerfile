# syntax=docker/dockerfile:1.7
#
# Образ окружения пайплайна lecture-transcript.
# Цель: WSL2 + Docker на Windows, одна GPU RTX 4060 8 GB (Ada, sm_89).
#
# Собирается из корня репозитория:
#   docker build -f docker/Dockerfile -t lecture-transcript:0.1.0 .
# Исходники в образ НЕ копируются — репозиторий монтируется на /app
# (см. docker/compose.yaml), PYTHONPATH указывает на /app/src.
# Так пересборка образа нужна только при смене зависимостей.

# --------------------------------------------------------------------------
# База
# --------------------------------------------------------------------------
# nvidia/cuda:12.6.3-cudnn-runtime-ubuntu22.04
#   - runtime (не devel): компилятор и заголовки CUDA не нужны, все колёса
#     ставятся готовыми — образ втрое меньше
#   - cudnn: в теге ветки 12.4+ это cuDNN 9. Он обязателен для ctranslate2>=4.5
#     (CT2 4.5.0 перешёл на cuDNN 9 и потерял совместимость с cuDNN 8)
#   - ubuntu22.04: на 22.04 фактически проверено, что apt-овый ffmpeg умеет
#     nvdec (см. блок про ffmpeg ниже)
# Тег вынесен в ARG, чтобы можно было переехать на другую CUDA без правки файла.
ARG CUDA_IMAGE=nvidia/cuda:12.6.3-cudnn-runtime-ubuntu22.04
FROM ${CUDA_IMAGE}

ARG PYTHON_VERSION=3.11
# INSTALL_DEV=1 — доставить requirements-dev.txt (pytest и пр.)
ARG INSTALL_DEV=1

ENV DEBIAN_FRONTEND=noninteractive \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Ключевая строка для nvdec.
# nvidia-container-toolkit по умолчанию прокидывает в контейнер только
# capabilities `compute,utility`, и тогда libnvcuvid.so.1 / libnvidia-encode.so.1
# внутрь НЕ попадают: `ffmpeg -hwaccels` покажет cuda, а декодирование упадёт
# на "Cannot load libnvcuvid.so.1". Capability `video` добавляет их.
ENV NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility,video

# --------------------------------------------------------------------------
# Системные пакеты
# --------------------------------------------------------------------------
# ffmpeg берётся из apt Ubuntu 22.04 — сборка из исходников не нужна.
# Проверено фактически (запуск ubuntu:22.04 amd64, apt-get install ffmpeg):
#   ffmpeg 4.4.2-0ubuntu0.22.04.1
#   ffmpeg -hwaccels  -> vdpau, cuda, vaapi, qsv, drm, opencl
#   ffmpeg -decoders  -> h264_cuvid, hevc_cuvid, av1_cuvid, mjpeg_cuvid, ...
# Флагов --enable-nvdec/--enable-cuvid в строке configure нет потому, что
# ffmpeg включает их сам при наличии заголовков ffnvcodec (они в build-deps
# Debian/Ubuntu), а не потому, что поддержка отсутствует. libnvcuvid грузится
# через dlopen в рантайме — из драйвера хоста, см. NVIDIA_DRIVER_CAPABILITIES.
#
# Чего в apt-сборке НЕТ: --enable-cuda-nvcc, то есть CUDA-фильтров
# (scale_cuda, yadif_cuda) не будет. Пайплайну они не нужны: D7 требует только
# декодирование nvdec с последующей выгрузкой кадров в системную память.
#
# libgl1/libglib2.0-0/libsm6/libxext6 — для не-headless сборки OpenCV,
# которую тянет paddlex (opencv-contrib-python).
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
        git \
        gnupg \
        software-properties-common \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
        python${PYTHON_VERSION} \
        python${PYTHON_VERSION}-dev \
        python${PYTHON_VERSION}-venv \
        ffmpeg \
        libgl1 \
        libglib2.0-0 \
        libgomp1 \
        libsm6 \
        libxext6 \
    && apt-get purge -y --auto-remove software-properties-common \
    && rm -rf /var/lib/apt/lists/*

# --------------------------------------------------------------------------
# Python-окружение
# --------------------------------------------------------------------------
# В Ubuntu 22.04 системный python — 3.10; 3.11 приходит из deadsnakes.
# Venv вместо системного python: не ломает apt-овые скрипты и даёт
# один предсказуемый префикс для всех колёс.
ENV VIRTUAL_ENV=/opt/venv
RUN python${PYTHON_VERSION} -m venv ${VIRTUAL_ENV}
ENV PATH="${VIRTUAL_ENV}/bin:${PATH}"
RUN pip install --upgrade "pip==24.3.1" "setuptools==75.6.0" "wheel==0.45.1"

# --------------------------------------------------------------------------
# Зависимости
# --------------------------------------------------------------------------
COPY docker/requirements-torch.txt \
     docker/requirements.txt \
     docker/requirements-paddle.txt \
     docker/requirements-dev.txt \
     /tmp/reqs/

# 1) torch с индекса PyTorch (cu124). Ставится первым: остальные пакеты
#    объявляют torch зависимостью, и при таком порядке pip не пойдёт за
#    CPU-сборкой torch на PyPI. Локальная метка +cu124 удовлетворяет
#    ограничению gigaam `torch<=2.5.1` — по PEP 440 локальная метка
#    игнорируется, если её нет в самом спецификаторе.
RUN pip install --index-url https://download.pytorch.org/whl/cu124 \
        -r /tmp/reqs/requirements-torch.txt

# 2) Основной набор с PyPI.
#    GIT_LFS_SKIP_SMUDGE=1 — из-за единственной VCS-зависимости
#    `sbert_punc_case_ru @ git+https://huggingface.co/...`: пакета нет на PyPI,
#    его код лежит внутри репозитория модели на HuggingFace, а рядом с кодом —
#    model.safetensors на ~1.7 GB под git-lfs. Без этой переменной pip утянул бы
#    веса прямо в слой образа; веса нужны в рантайме и качаются в /cache.
RUN GIT_LFS_SKIP_SMUDGE=1 pip install -r /tmp/reqs/requirements.txt

# 3) PaddlePaddle GPU — только со своего индекса, GPU-колёс 3.x на PyPI нет.
#    --no-deps не нужен: у колеса лёгкий набор зависимостей, уже покрытый выше.
RUN pip install --index-url https://www.paddlepaddle.org.cn/packages/stable/cu126/ \
        -r /tmp/reqs/requirements-paddle.txt

# 4) Детерминированный победитель в модуле `cv2`.
#    opencv-python-headless (pix2tex, albumentations) и opencv-contrib-python
#    (paddlex) кладут файлы в один и тот же пакет cv2; кто из них окажется
#    последним при обычной установке — зависит от порядка резолва.
#    Явно фиксируем contrib-сборку как итоговую: она надмножество по функциям,
#    а нужные ей libGL уже стоят.
RUN pip install --no-deps --force-reinstall opencv-contrib-python==4.10.0.84

# 5) Dev-инструменты — отдельным слоем, отключаются через --build-arg INSTALL_DEV=0.
RUN if [ "${INSTALL_DEV}" = "1" ]; then \
        pip install -r /tmp/reqs/requirements-dev.txt; \
    fi \
    && rm -rf /tmp/reqs

# --------------------------------------------------------------------------
# Кэш моделей
# --------------------------------------------------------------------------
# Все веса качаются в /cache, который монтируется томом: пересборка образа
# не должна приводить к повторной выкачке нескольких гигабайт.
# XDG_CACHE_HOME читают и бэкенды проекта: pix2tex и GigaAM сами пишут веса
# внутрь контейнера, поэтому бэкенды передают им /cache/pix2tex и /cache/gigaam.
ENV XDG_CACHE_HOME=/cache \
    HF_HOME=/cache/huggingface \
    TORCH_HOME=/cache/torch \
    PADDLE_PDX_CACHE_HOME=/cache/paddlex \
    MPLCONFIGDIR=/cache/matplotlib
RUN mkdir -p /cache /data/input /data/output

# --------------------------------------------------------------------------
# Код и точка входа
# --------------------------------------------------------------------------
# Репозиторий монтируется на /app снаружи, поэтому COPY исходников здесь нет.
ENV PYTHONPATH=/app/src
WORKDIR /app

COPY docker/verify_env.sh /usr/local/bin/verify_env.sh
RUN chmod +x /usr/local/bin/verify_env.sh

CMD ["bash"]
