# Окружение пайплайна `lecture-transcript`

Артефакты по задачам **1.1** (образ с CUDA-runtime и ffmpeg с nvdec) и **1.2**
(зафиксированные зависимости Python) из
`openspec/changes/lecture-transcript-pipeline/tasks.md`.

Целевая машина: **WSL2 + Docker на Windows, одна RTX 4060 8 GB**.
Разработка велась на macOS arm64 без CUDA, поэтому часть проверок физически
невыполнима здесь — см. раздел «Что не проверено».

## Состав

| Файл | Назначение |
|---|---|
| `Dockerfile` | образ: CUDA 12.6 + cuDNN 9, ffmpeg с nvdec, Python 3.11, все зависимости |
| `Dockerfile.dockerignore` | отсечение контекста сборки (BuildKit читает его вместо корневого `.dockerignore`) |
| `requirements.txt` | runtime-зависимости с PyPI, версии зафиксированы через `==` |
| `requirements-torch.txt` | `torch`/`torchvision`/`torchaudio` с индекса PyTorch (cu124) |
| `requirements-paddle.txt` | `paddlepaddle-gpu` с индекса PaddlePaddle (cu126) |
| `requirements-dev.txt` | `pytest` и прочий инструментарий |
| `compose.yaml` | сервис с `gpus: all`, монтированием входа, выхода и кэша |
| `verify_env.sh` | самопроверка окружения внутри контейнера |

## Как собрать

Контекст сборки — **корень репозитория**, не `docker/`.

```bash
# из корня репозитория
docker compose -f docker/compose.yaml build

# или без compose
docker build -f docker/Dockerfile -t lecture-transcript:0.1.0 .
```

Сборка без dev-инструментов:

```bash
docker build -f docker/Dockerfile --build-arg INSTALL_DEV=0 -t lecture-transcript:0.1.0 .
```

Исходники в образ не копируются: репозиторий монтируется на `/app`,
`PYTHONPATH=/app/src`. Правка кода не требует пересборки — пересборка нужна
только при изменении зависимостей.

## Как запустить самопроверку

```bash
docker compose -f docker/compose.yaml run --rm lecture-transcript verify_env.sh
```

Скрипт возвращает `0`, только если прошли все проверки:

1. Python 3.11, `ffmpeg`/`ffprobe` в `PATH`
2. `nvidia-smi` видит GPU; печатаются имя, объём VRAM, версия драйвера
   (отсутствие «4060» в списке — предупреждение, не провал)
3. `ffmpeg -hwaccels` перечисляет `cuda`; собран декодер `h264_cuvid`;
   доступна `libnvcuvid.so`; **сквозная проверка** — пробный ролик кодируется
   `libx264` и декодируется через `-hwaccel cuda -c:v h264_cuvid`
4. `torch.cuda.is_available()`, печать имени GPU, VRAM и compute capability,
   пробное матричное умножение на устройстве
5. импорт всех ключевых пакетов пайплайна, включая `sbert_punc_case_ru`
   (без него стадия 5.6 деградирует и транскрипт GigaAM останется без
   пунктуации) и связку для VLM-бэкенда:
   `Qwen2_5_VLForConditionalGeneration`, `accelerate`, `bitsandbytes`
6. `paddle.is_compiled_with_cuda()` и число видимых устройств
7. `pip check` — конфликтов зависимостей нет

Каталоги входа и выхода задаются переменными окружения:

```bash
LECTURE_INPUT_DIR=/mnt/d/lectures LECTURE_OUTPUT_DIR=./out \
  docker compose -f docker/compose.yaml run --rm lecture-transcript verify_env.sh
```

Веса моделей качаются в `/cache` (именованный том `model-cache`) — пересборка
образа не приводит к повторной выкачке нескольких гигабайт. Раскладка после
прогрева конфигурации по умолчанию (~1.5 ГБ):

| Каталог | Модель | Откуда качается |
|---|---|---|
| `/cache/paddlex/official_models` | PaddleOCR (5 моделей, ~140 МБ) | `huggingface.co`, при отказе — `modelscope.cn` |
| `/cache/pix2tex` | pix2tex (~115 МБ) | `github.com` (релиз LaTeX-OCR v0.0.1) |
| `/cache/gigaam` | GigaAM-v2 RNNT (~450 МБ) | `cdn.chatwm.opensmodel.sberdevices.ru` |
| `/cache/huggingface` | пунктуатор `kontur-ai/sbert_punc_case_ru` (~820 МБ) | `huggingface.co` |

silero-vad весов не качает: модель лежит внутри pip-пакета. pix2tex и GigaAM
сами по умолчанию пишут веса внутрь контейнера (каталог пакета и `~/.cache`),
и при `run --rm` они пропадают. Поэтому бэкенды явно передают им каталоги
внутри `$XDG_CACHE_HOME`. PaddleX 3.2.0 даже за скачанной моделью идёт к хостингу и
без сети падает, поэтому бэкенд передаёт PaddleOCR каталоги моделей из кэша
(`*_model_dir`). Сообщение «No model hoster is available» при прогоне без сети
безвредно: это проверка хостингов при импорте PaddleX.

**Сеть для сборки и прогрева.** Сборка: `download.pytorch.org`, `pypi.org`,
`huggingface.co` (git-зависимость пунктуатора), `www.paddlepaddle.org.cn` и
`paddle-whl.cdn.bcebos.com` (колесо Paddle). Прогрев: хосты из таблицы выше.
Китайские хосты через VPN могут не открываться: сборка тогда виснет на шаге
с Paddle до таймаута.

**Прогрев кэша.** По умолчанию сервис запускается с `HF_HUB_OFFLINE=1`: задачи
4.9 и 5.8 требуют прогона без сети, и при отсутствии весов нужна внятная ошибка,
а не тихая загрузка из интернета. Поэтому первый (и только первый) запуск, когда
кэш пуст, делается с явным переопределением:

```bash
HF_HUB_OFFLINE=0 docker compose -f docker/compose.yaml run --rm lecture-transcript \
    python -m lecture_transcript.warmup
```

Проверить, что прогретого кэша хватает для работы без сети, можно прогревом в
контейнере с отключённой сетью. Он должен завершиться с кодом 0:

```bash
docker run --rm --gpus all --network none --shm-size 2g \
    -e NVIDIA_DRIVER_CAPABILITIES=compute,utility,video -e HF_HUB_OFFLINE=1 \
    -v "$PWD":/app -v lecture-transcript_model-cache:/cache \
    lecture-transcript:0.1.0 python -m lecture_transcript.warmup
```

`verify_env.sh` сети не требует: он только импортирует пакеты, веса не грузит.

## Принятые решения

### Базовый образ: `nvidia/cuda:12.6.3-cudnn-runtime-ubuntu22.04`

- `runtime`, а не `devel`: все колёса ставятся готовыми, компилятор CUDA
  не нужен, образ существенно меньше.
- `cudnn` в теге ветки 12.4+ означает **cuDNN 9**. Это жёсткое требование
  `ctranslate2>=4.5.0`: в release notes CTranslate2 v4.5.0 прямо сказано, что
  пакет перешёл на cuDNN 9 и больше не совместим с cuDNN 8.
- `ubuntu22.04`: именно на 22.04 фактически проверено, что apt-овый ffmpeg
  умеет nvdec (ниже).

### ffmpeg берётся из apt, а не собирается из исходников

Это проверено фактом, а не предположением. Запуск `ubuntu:22.04` (amd64,
через эмуляцию) с `apt-get install ffmpeg` даёт:

```
ffmpeg version 4.4.2-0ubuntu0.22.04.1
ffmpeg -hwaccels -> vdpau, cuda, vaapi, qsv, drm, opencl
ffmpeg -decoders -> h264_cuvid, hevc_cuvid, av1_cuvid, mjpeg_cuvid, vp9_cuvid, ...
```

То есть `cuda` в списке hwaccels и все `*_cuvid` декодеры присутствуют.
Флагов `--enable-nvdec` / `--enable-cuvid` в строке `configure` нет потому,
что ffmpeg включает их автоматически при наличии заголовков `ffnvcodec`
(они в build-deps пакета), а не потому, что поддержки нет. Сама `libnvcuvid`
не линкуется, а грузится через `dlopen` в рантайме — из драйвера хоста.

**Вывод:** сборка ffmpeg из исходников не нужна, `apt install ffmpeg` достаточно.

Ограничение apt-сборки: нет `--enable-cuda-nvcc`, значит недоступны CUDA-фильтры
(`scale_cuda`, `yadif_cuda`, `hwupload_cuda`). Пайплайну они не требуются:
по design D7 нужен только декод nvdec с выгрузкой кадров в системную память,
кроп и ресайз делаются на CPU/OpenCV.

### `NVIDIA_DRIVER_CAPABILITIES=compute,utility,video`

Самая неочевидная строка во всём образе. `nvidia-container-toolkit` по умолчанию
прокидывает только `compute,utility`, и тогда `libnvcuvid.so.1` внутрь **не
попадает**: `ffmpeg -hwaccels` покажет `cuda`, `nvidia-smi` отработает, а
декодирование упадёт на `Cannot load libnvcuvid.so.1`. Capability `video`
это чинит. Переменная продублирована и в `Dockerfile`, и в `compose.yaml`.

### Python 3.11 из deadsnakes

В Ubuntu 22.04 системный Python — 3.10, а `pyproject.toml` требует `>=3.11`.
PPA deadsnakes даёт 3.11 без пересборки. Пакеты ставятся в venv `/opt/venv`,
чтобы не ломать apt-овые скрипты и иметь один предсказуемый префикс.

### Верхняя граница стека задана GigaAM

`gigaam==0.1.0` объявляет `torch<=2.5.1` и `torchaudio<=2.5.1`. Это фиксирует
весь стек: torch 2.5.1 → самая свежая CUDA-сборка на индексе PyTorch — `+cu124`,
torchvision — `0.20.1`. Остальные пакеты выровнены на релизы того же периода.

Локальная метка `+cu124` не мешает ограничению `torch<=2.5.1`: по PEP 440
локальная метка кандидата игнорируется, если её нет в самом спецификаторе.

Базовый образ при этом CUDA 12.6, а torch собран под 12.4 — это нормально:
linux-колёса torch тянут собственные `nvidia-*-cu12` библиотеки и зависят
только от драйвера, а драйвер 12.6 обратно совместим с бинарями под 12.4.

### PaddleOCR 3.2.0, а не 3.0.x/3.1.x и не 2.x

- `paddleocr==3.2.0` — первая версия, где зависимость на `paddlex` ограничена
  сверху (`paddlex[ocr-core]<3.3.0,>=3.2.0`) и использует лёгкий extra
  `ocr-core`. У 3.0.x/3.1.x там `paddlex[ie,multimodal,ocr]>=3.0.3` —
  без верхней границы и с втрое большим графом зависимостей.
- Ветка 2.x отпадает по другой причине: единственная GPU-сборка Paddle на PyPI —
  `paddlepaddle-gpu==2.6.2`, собранная под CUDA 11.x, и она несовместима
  с базовым образом на CUDA 12.6 + cuDNN 9.
- Русский включается параметром `lang="ru"` (кириллические модели распознавания).

### Конфликт вокруг модуля `cv2`

Три пакета экосистемы поставляют один и тот же модуль `cv2`:
`pix2tex` и `albumentations` требуют `opencv-python-headless`,
`paddlex[ocr-core]` жёстко пинит `opencv-contrib-python==4.10.0.84`.
Обе версии выровнены на `4.10.0.84`, а последним шагом сборки
`opencv-contrib-python` переустанавливается принудительно — чтобы победитель
в `cv2` был детерминированным, а не зависел от порядка резолва pip.
Не-headless сборке нужны `libGL`/`libgthread`, они ставятся apt-ом.

### Пунктуатор ставится из git по хэшу коммита

Стадия 5.6 (D6) обязательна для GigaAM — он не даёт пунктуации сам. Реализация
в `speech_transcription/punctuation.py` импортирует пакет `sbert_punc_case_ru`,
а его **нет на PyPI** (`pip install sbert_punc_case_ru` → 404), и репозитория
`github.com/kontur-ai/sbert_punc_case_ru` тоже нет: код лежит внутри
репозитория модели на HuggingFace. Отсюда единственный VCS-пин в проекте:

```
sbert_punc_case_ru @ git+https://huggingface.co/kontur-ai/sbert_punc_case_ru@f778dc6c63bb0ec235a220488862509810e54583
```

Что проверено на dev-машине:

- `api/models/kontur-ai/sbert_punc_case_ru/revision/f778dc6c...` → 200,
  возвращённый `sha` совпадает с пином (значит это существующий коммит,
  а не ветка, которая может уехать);
- в дереве коммита есть `setup.py` и пакет `sbert_punc_case_ru/`;
- `setup.py`: `name="sbert_punc_case_ru"`, `version="0.2"`,
  `install_requires=["transformers>=4.36.2", "torch", "numpy"]` — с пинами
  образа не конфликтует;
- `sbert_punc_case_ru/__init__.py` не имеет побочных эффектов (веса грузятся
  в `SbertPuncCase.__init__`), поэтому импорт безопасно проверять
  в `verify_env.sh` при `HF_HUB_OFFLINE=1`;
- `uv pip compile` на этом пине проходит: uv клонирует репозиторий, собирает
  метаданные из `setup.py` и выдаёт разрешённое требование.

Ставится с `GIT_LFS_SKIP_SMUDGE=1` (см. Dockerfile): рядом с кодом лежит
`model.safetensors` на ~1.7 GB под git-lfs, и без этого он попал бы в слой
образа. Веса нужны в рантайме и качаются в `/cache`. Замер: кэш клона
у `uv` после резолва — 7.8 MB, то есть LFS-объекты не тянутся.

**Оговорка:** `setup.py` пакета использует `distutils`, удалённый в Python 3.12.
На Python 3.11 (наш случай) это работает; при переезде на 3.12+ пин сломается.
Запасной путь на этот случай — реализовать пунктуатор поверх `transformers`
без отдельного пакета: модель `kontur-ai/sbert_punc_case_ru` — обычный
`AutoModelForTokenClassification`, а весь код пакета это ~150 строк разметки
меток (`LABELS_CASE` × `LABELS_PUNC`) и склейки токенов. Альтернативная
модель того же назначения — `RUPunct/RUPunct_big`.

### VLM-бэкенд (Qwen2.5-VL 4bit) — работоспособен

Задача 4.8 требует рабочего альтернативного OCR-бэкенда. Для этого пришлось
поднять `transformers` с 4.46.3 до **4.49.0**: класса
`Qwen2_5_VLForConditionalGeneration` в 4.46.3 нет. Проверено по исходникам:

```
transformers v4.48.3 .../models/qwen2_5_vl/__init__.py -> 404
transformers v4.49.0 .../models/qwen2_5_vl/__init__.py -> 200
```

Следом подтянулся `tokenizers` 0.20.3 → **0.21.0** (4.49.0 требует
`tokenizers>=0.21,<0.22`), и добавлены `bitsandbytes==0.45.5` (4-битная
квантизация; ветка 0.45.x собрана под CUDA 12.4 — совпадает с `torch+cu124`)
и `accelerate==1.4.0` (без него не работает `device_map="auto"`).

Ограничение `gigaam → torch<=2.5.1` при этом **не тронуто**: `transformers`
не пинит torch сверху. Весь набор из 103 пакетов вместе с `torch==2.5.1`
разрешается без конфликтов — проверено `uv pip compile`.

## Что проверено на dev-машине

| Проверка | Команда | Результат |
|---|---|---|
| синтаксис Dockerfile + линт | `docker buildx build --check -f docker/Dockerfile .` | `Check complete, no warnings found` |
| существование базового образа | тот же вызов, шаг `load metadata` | тег `nvidia/cuda:12.6.3-cudnn-runtime-ubuntu22.04` разрешился в registry |
| валидность compose | `docker compose -f docker/compose.yaml config` | OK |
| синтаксис скрипта | `bash -n docker/verify_env.sh` | OK |
| ffmpeg из apt умеет nvdec | запуск `ubuntu:22.04` amd64 + `apt install ffmpeg` | `cuda` в hwaccels, `h264_cuvid` в decoders |
| разрешимость `requirements.txt` | `uv pip compile --python-version 3.11 --python-platform x86_64-unknown-linux-gnu` | разрешается без конфликтов |
| то же с пином `torch==2.5.1` | тот же вызов + `torch==2.5.1`, `torchvision==0.20.1`, `torchaudio==2.5.1` | разрешается без конфликтов |
| разрешимость `requirements-torch.txt` | `uv pip compile` с индексом PyTorch | `torch==2.5.1+cu124`, `torchvision==0.20.1+cu124`, `torchaudio==2.5.1+cu124` |
| разрешимость `requirements-dev.txt` | `uv pip compile` | разрешается без конфликтов |
| наличие каждой версии на PyPI | запросы к `https://pypi.org/pypi/<pkg>/<ver>/json` | все пины отвечают `200` |
| коммит пунктуатора существует | HF API `revision/f778dc6c...` | `200`, `sha` совпадает; в дереве есть `setup.py` |
| VCS-пин собирается | `GIT_LFS_SKIP_SMUDGE=1 uv pip compile` только на git-строке | метаданные собраны, кэш клона 7.8 MB (LFS не тянется) |
| `qwen2_5_vl` есть в transformers | исходники на GitHub, теги `v4.48.3` и `v4.49.0` | `404` / `200` |
| `.dockerignore` работает | пробная сборка `busybox` + `COPY . /ctx` с копией правил | контекст `279 MB → 956 kB`, внутри только нужные каталоги |
| тест порядка установки ловит поломку | мутация: два `RUN pip install` переставлены местами | `test_torch_is_installed_before_pypi_set` падает; файл восстановлен байт в байт |
| тест сквозной проверки nvdec ловит поломку | мутация: `fail` → `warn` в `verify_env.sh` | `test_verify_env_fails_when_nvdec_smoke_test_cannot_run` падает; файл восстановлен байт в байт |
| поведение `verify_env.sh` без GPU | запуск с подменённым `PATH` и заглушками | код возврата `1`, нужные строки `[FAIL]` |
| статические проверки артефактов | `.venv/bin/pytest tests/test_docker_artifacts.py` | 16 passed |

## Что не проверено

Всё ниже требует физической RTX 4060 и выполняется **на целевой машине**,
одной командой `verify_env.sh`. Задачи 1.1 и 1.2 нельзя считать закрытыми,
пока она не вернула `0`.

1. **Сборка образа целиком.** На dev-машине arm64 базовый образ
   `nvidia/cuda:*` под amd64 не запускается штатно, а сборка через эмуляцию
   заняла бы часы и всё равно ничего не доказала бы про GPU.
2. **`nvidia-smi` внутри контейнера** — нет NVIDIA runtime у локального Docker.
   Проверка: шаг 2 в `verify_env.sh`.
3. **Фактическое декодирование через nvdec.** Наличие `cuda` в `-hwaccels`
   проверено, но `libnvcuvid` приходит из драйвера хоста. Проверка: шаг 3
   в `verify_env.sh` (сквозной энкод + декод пробного ролика).
4. **`torch.cuda.is_available()`, имя GPU, объём VRAM.** Проверка: шаг 4.
5. **Наличие `paddlepaddle-gpu==3.2.0` на индексе PaddlePaddle.** Индекс
   `paddlepaddle.org.cn` недоступен с dev-машины (таймаут соединения), поэтому
   существование именно этого файла подтвердить не удалось. Косвенное
   основание: CPU-двойник `paddlepaddle==3.2.0` на PyPI есть, колесо
   `cp311-manylinux1_x86_64` присутствует. Это **самый рискованный пин
   в проекте**. Если сборка упадёт на этом шаге — взять ближайшую доступную
   версию из каталога
   `https://www.paddlepaddle.org.cn/packages/stable/cu126/paddlepaddle-gpu/`
   и обновить `requirements-paddle.txt`.
6. **Реальная установка в чистом образе.** GPU-колёса (`torch+cu124`,
   `paddlepaddle-gpu`) не существуют для arm64-macOS, поэтому здесь проверена
   только *разрешимость* графа версий для платформы `x86_64-unknown-linux-gnu`,
   без скачивания и установки.
7. **Замеры VRAM и времени стадий** из design D7 — задачи 7.2 и 7.4,
   вне скоупа 1.1/1.2.
8. **Рантайм-совместимость `pix2tex` с `transformers` 4.49.0.** Граф версий
   разрешается (`pix2tex` требует лишь `transformers>=4.18`), но 4.49 старше
   `pix2tex` 0.1.4 на два года, и удалённые за это время API теоретически
   могут всплыть при импорте. Проверка: строка `pix2tex` в шаге 5
   `verify_env.sh` — она импортирует `LatexOCR`, то есть ловит это сразу.
9. **Загрузка Qwen2.5-VL в 4bit.** `bitsandbytes` подбирает CUDA-бинарник
   в рантайме; проверить можно только на GPU. Шаг 5 `verify_env.sh`
   подтверждает наличие всех трёх составляющих (класс модели, `accelerate`,
   `bitsandbytes`), но не сам прогон квантизованной модели — это задача 4.8.

## После первой успешной сборки

Снять полный лок окружения — здесь пинуются прямые зависимости и рискованные
транзитивные, но не весь граф:

```bash
docker compose -f docker/compose.yaml run --rm lecture-transcript \
  pip freeze > docker/requirements.lock.txt
```
