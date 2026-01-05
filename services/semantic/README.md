# Сервис семантического поиска по OpenAlex

Сервис для семантического поиска научных публикаций на основе корпуса OpenAlex. Тексты статей кодируются трансформером E5, векторный индекс строится в FAISS, метаданные и связи хранятся в PostgreSQL. Доступ к поиску предоставляется по gRPC.

Ключевые особенности
- Многоязычная модель: `intfloat/multilingual-e5-large` (поддержка LoRA-адаптера)
- Векторный индекс FAISS (IVF+PQ или Flat), хранение эмбеддингов в memmap
- PostgreSQL схема с авторами, организациями, идентификаторами, ссылками и двунаправленными/однонаправленными связями между работами
- Пайплайн: загрузка OpenAlex → очистка/нормализация → загрузка в БД → генерация эмбеддингов → построение FAISS индекса
- gRPC API с методом семантического поиска статей


## Быстрый старт (Docker Compose)

Предустановки
- Docker и Docker Compose
- Внешняя сеть для gRPC (если нужна): `docker network create grpc_network`

1) Отредактируйте `.env` (см. пример в корне репозитория). Важно указать доступ к БД и пути к индексу FAISS:
   - `FAISS_INDEX_PATH=data/index/faiss_both.index`
   - `FAISS_DOC_IDS_PATH=data/index/doc_ids_both.npy`

2) Запустите стэк:
   - `docker compose up -d --build`

3) Проверка состояния:
   - Сервис gRPC доступен на `localhost:5104`
   - Логи контейнера: `docker logs -f semantic-search`

В Compose включено:
- `postgres` (PostgreSQL 17)
- `migrator` — применяет миграции (`db/migrations`) перед запуском сервиса
- `semantic-search` — сам gRPC‑сервис (см. `cmd/main.py`)
- `pipeline-worker` (опционально) — фоновый воркер пайплайна (ожидает сигнал) 

Примонтированные тома
- `./data` → `/app/data` (индексы и артефакты пайплайна)
- `hf-cache` → `/cache/huggingface` (кэш моделей HF)


## Локальный запуск (без Docker)

Требования
- Python 3.12
- Postgres 14+ (в примерах — 17)

1) Установите зависимости
```
python -m venv .venv
. .venv/Scripts/activate  # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -U pip
pip install -r requirements.txt
```

2) Примените миграции БД (варианты)
- Docker‑миграции (проще): `docker compose run --rm migrator`
- Локально: примените файлы из `db/migrations` вручную или используйте любой совместимый мигратор

3) Укажите переменные окружения (`.env` или env):
- `DB_HOST`, `DB_PORT`, `DB_USER`, `DB_PASSWORD`, `DB_NAME`
- `FAISS_INDEX_PATH`, `FAISS_DOC_IDS_PATH` (см. раздел «Данные и индексы»)

4) Запустите сервис
```
python cmd/main.py
```


## Данные и индексы

Готовые артефакты (пример):
- `data/index/faiss_both.index` — FAISS‑индекс
- `data/index/doc_ids_both.npy` — соответствия позиций в индексе → `paper_id`

Если индексов нет, постройте их пайплайном (см. ниже) или положите подготовленные файлы в `data/index` и укажите пути в `.env`.


## Пайплайн подготовки данных

Шаги пайплайна формализованы скриптами в `src/parser` и оркестратором в `scripts/run_pipeline.py`.

1) Загрузка OpenAlex (через API)
```
python src/parser/openalex_csv_parser.py \
  --email you@example.com \
  --outdir data/raw \
  --en 1000000 --ru 550000 \
  --chunk-size 50000 \
  --gzip --resume
```

2) Очистка и нормализация
```
python src/parser/clean_openalex.py \
  --indir data/raw \
  --outdir data/processed
```

3) Загрузка в PostgreSQL
```
python src/parser/load_openalex_to_db.py \
  --indir data/processed
```

4) Генерация эмбеддингов (E5)
```
python src/parser/e5_embed_corpus.py \
  --outdir data/index \
  [--where "language = 'en'" | другие SQL‑фильтры] \
  [--limit 100000] \
  [--model intfloat/multilingual-e5-large] \
  [--lora-dir path/to/lora]
```
Результат: `doc_embeddings.f16.memmap` (+ `.shape.json`) и `doc_ids.npy` в указанной директории.

5) Построение FAISS‑индекса
```
python src/parser/e5_build_faiss.py \
  --emb-dir data/index \
  --index-type ivfpq \
  --metric ip \
  --nlist 4096 --m 32 --nbits 8
```
Результат: `faiss.index` (+ `.meta.json`) в `data/index`. Эти пути затем указываются в `.env`.

Оркестратор пайплайна
- `scripts/run_pipeline.py --mode once` запустит последовательность шагов из `src/pipeline/runner.py`.
- `scripts/run_pipeline.py --mode serve` запустит `PipelineWorker`, который ждёт POSIX‑сигнал и по событию запускает пайплайн.


## gRPC API

Proto: `proto/service.proto`
- Пакет: `semantic`
- Сервис: `SemanticService`
- RPC:
  - `SearchPaper(SearchRequest) -> PapersResponse`
  - `AddPaper(AddRequest) -> AddPaperResponse`

Сообщения
- `SearchRequest` — поле `Input_data: string` (текст запроса)
- `PaperResponse` — `ID, Title, Abstract, Year, Best_oa_location`
- `PapersResponse` — repeated `PaperResponse Papers`
- `AddRequest` — поля статьи + списки `Referenced_works`/`Related_works` (идентификаторы OpenAlex). Сервер пока не реализует сохранение, метод возвращает `Error = ""`.

Пример клиента (Python)
```
import grpc
from src.http.grpc import service_pb2, service_pb2_grpc

channel = grpc.insecure_channel("localhost:5104")
stub = service_pb2_grpc.SemanticServiceStub(channel)
resp = stub.SearchPaper(service_pb2.SearchRequest(Input_data="graph neural networks for chemistry"))
for p in resp.Papers:
    print(p.Title, p.Year, p.Best_oa_location)
```


## Конфигурация (переменные окружения)

Основные переменные (`.env`):
- Логи и сервис: `LOG_LEVEL`, `LOGSTASH_HOST`, `LOGSTASH_PORT`, `SEMANTIC_PORT`
- БД: `DB_HOST`, `DB_PORT`, `DB_USER`, `DB_PASSWORD`, `DB_NAME`, `SSLMode`
- Индекс и модель: `FAISS_INDEX_PATH`, `FAISS_DOC_IDS_PATH`, `EMBEDDING_MODEL_NAME`, `EMBEDDING_BATCH_SIZE`, `EMBEDDING_LORA_PATH`
- Пайплайн: `DATA_ROOT`, `PIPELINE_RAW_DIR`, `PIPELINE_PROCESSED_DIR`, `PIPELINE_INDEX_DIR`, `OPENALEX_EMAIL`, `PIPELINE_OPENALEX_EN`, `PIPELINE_OPENALEX_RU`, `PIPELINE_OPENALEX_CHUNK`

Значения по умолчанию можно посмотреть в `src/config/config.py` и `src/pipeline/settings.py`.


## Архитектура и ключевые компоненты

- Вход: текст запроса → кодируется E5 (`SemanticEncoder`) → поиск ближайших соседей в FAISS (`FaissIndex`) → извлечение метаданных из Postgres (`PaperRepository`) → ответ gRPC.

Исходники:
- Точка входа сервиса: `cmd/main.py`
- gRPC сервер и обработчики: `src/http/grpc/grpc_server.py`, `src/http/grpc/grpc_handler.py`, `proto/service.proto`
- Поиск: `src/services/search/faiss_index.py`, `src/services/search/faiss_searcher.py`, `src/services/search/search.py`
- Модель E5: `src/al_models/e5/encoder.py`
- Доступ к БД: `src/storage/paper_repository.py`, `src/db/connection.py`
- Пайплайн: `src/pipeline/runner.py`, `src/pipeline/worker.py`, скрипты в `src/parser/*`
- Схема БД и миграции: `db/migrations`, (также референс `crebas.sql`)


## Тонкости и производительность

- GPU: при наличии CUDA PyTorch выберет GPU автоматически; на CPU генерация эмбеддингов заметно медленнее. Для CUDA‑колёс используйте документацию PyTorch (см. комментарий в `requirements.txt`).
- EMBEDDING_BATCH_SIZE: можно увеличить на GPU, уменьшать на CPU при нехватке памяти.
- Тип индекса: для больших корпусов используйте `ivfpq` с тренировкой на поднаборах (`--train-size`). Для максимальной точности — `flat`.


## Тестирование и отладка

- Быстрая проверка индекса оффлайн: `src/parser/e5_test_search.py`
- Healthcheck сервиса: `scripts/healthcheck.py`


## Структура репозитория (кратко)

- `cmd/main.py` — запуск gRPC‑сервиса
- `proto/service.proto` — описание API
- `src/services/search/*` — поиск (FAISS, ранжирование, сервис)
- `src/al_models/e5/*` — обёртка над моделью E5
- `src/storage/*`, `src/db/*` — работа с БД
- `src/parser/*` — скрипты пайплайна (OpenAlex → БД → эмбеддинги → индекс)
- `db/migrations` — миграции PostgreSQL
- `docker-compose.yml`, `Dockerfile` — контейнеризация
- `scripts/*` — утилиты (pipeline runner, healthcheck)


## Ограничения и планы

- `AddPaper` пока не реализован сервером (заглушка в `grpc_handler.py`), запись в БД/индекс требует отдельной реализации.
- Реализация создания чатов их истории и ее получения
- Постройка индекса с опорой на внешее цитирование, а не только на текстовое содержание


## Ссылки на исходники

- Точка входа: `cmd/main.py:1`
- gRPC proto: `proto/service.proto:1`
- FAISS обёртка: `src/services/search/faiss_index.py:1`
- Поисковик: `src/services/search/faiss_searcher.py:1`
- Сервис поиска: `src/services/search/search.py:1`
- Репозиторий статей: `src/storage/paper_repository.py:1`
- Конфигурация: `src/config/config.py:1`
- Пайплайн раннер: `src/pipeline/runner.py:1`
- Сборка эмбеддингов: `src/parser/e5_embed_corpus.py:1`
- Построение индекса: `src/parser/e5_build_faiss.py:1`

