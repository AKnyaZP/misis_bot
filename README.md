# misis_bot
AI ассистент для студентов МИСИС с использованием GPT-oSS-20b, LangChain и Qdrant

## Описание

Telegram-бот для помощи студентам и абитуриентам Университета МИСИС. Бот использует:
- **GPT-oSS-20b** (или другую модель HuggingFace) для генерации ответов
- **LangChain** для работы с LLM и RAG
- **Qdrant** для векторного хранилища и семантического поиска
- **Парсер сайта misis.ru** для сбора информации о университете

## Установка и запуск с Docker (рекомендуется)

### Быстрый старт

1. Клонируйте репозиторий:
```bash
git clone <repository-url>
cd misis_bot
```

2. Создайте файл `.env` в корне проекта (обязательно перед запуском):
```bash
cp env.sample .env
```

   Или создайте файл `.env` вручную в корне проекта со следующим содержимым.

3. Заполните `.env` файл необходимыми переменными (обязательно BOT_TOKEN и HF_TOKEN):
```env
BOT_TOKEN=your_telegram_bot_token
HF_MODEL=Qwen/Qwen2.5-7B-Instruct  # или GPT-oSS-20b
HF_TOKEN=your_huggingface_token

# Qdrant будет доступен через docker-compose, URL настроен автоматически
QDRANT_URL=http://qdrant:6333
QDRANT_API_KEY=
QDRANT_COLLECTION_NAME=misis_bot_vectors

MAX_NEW_TOKENS=512
TEMPERATURE=0.7
```

4. Запустите все сервисы с помощью Docker Compose:
```bash
docker-compose up -d
```

Это запустит:
- **Qdrant** - векторное хранилище
- **Bot** - Telegram бот

5. Запустите парсер для загрузки данных в Qdrant:
```bash
docker-compose run --rm parser
```

Или используйте Makefile:
```bash
make parser
```

6. Просмотр логов:
```bash
# Все логи
docker-compose logs -f

# Только бот
docker-compose logs -f bot

# Только парсер
docker-compose logs -f parser
```

7. Остановка сервисов:
```bash
docker-compose down
```

### Использование Makefile (опционально)

Для удобства можно использовать Makefile:

```bash
make build      # Собрать образы
make up         # Запустить сервисы
make down       # Остановить сервисы
make logs       # Показать логи
make logs-bot   # Показать логи только бота
make parser     # Запустить парсер
make rebuild    # Пересобрать и перезапустить
make status     # Показать статус сервисов
make clean      # Остановить и удалить все (включая volumes)
make help       # Показать все доступные команды
```

### Парсинг данных (если нужно перезагрузить)

Если нужно перезагрузить данные в Qdrant:
```bash
docker-compose run --rm parser
# или
make parser
```

### Обновление бота после изменений кода

```bash
docker-compose build bot
docker-compose up -d bot
# или
make rebuild
```

## Локальная установка (без Docker)

1. Установите зависимости:
```bash
pip install -r requirements.txt
```

2. Запустите Qdrant через Docker:
```bash
docker run -d -p 6333:6333 -p 6334:6334 --name qdrant qdrant/qdrant
```

3. Создайте файл `bot/.env` на основе `bot/env.sample`:
```bash
cp bot/env.sample bot/.env
```

4. Заполните `bot/.env` файл:
```env
BOT_TOKEN=your_telegram_bot_token
HF_MODEL=Qwen/Qwen2.5-7B-Instruct
HF_TOKEN=your_huggingface_token
QDRANT_URL=http://localhost:6333
QDRANT_API_KEY=
QDRANT_COLLECTION_NAME=misis_bot_vectors
MAX_NEW_TOKENS=512
TEMPERATURE=0.7
```

5. Запустите парсер для загрузки данных:
```bash
python bot/scripts/parse_misis.py
```

6. Запустите бота:
```bash
python bot/run.py
```

## Использование

1. Запустите бота в Telegram
2. Отправьте команду `/start`
3. Задавайте вопросы об университете МИСИС

Бот использует семантический поиск по данным с сайта misis.ru для формирования ответов.

## Структура проекта

```
.
├── Dockerfile                    # Docker образ для бота
├── docker-compose.yml            # Конфигурация Docker Compose
├── .dockerignore                 # Игнорируемые файлы для Docker
├── requirements.txt              # Python зависимости
├── README.md                     # Документация
└── bot/
    ├── app.py                    # Главный файл приложения
    ├── config.py                 # Настройки
    ├── handlers.py               # Обработчики сообщений
    ├── middleware.py             # Middleware для передачи сервиса
    ├── requests.py               # Функции для работы с API
    ├── run.py                    # Точка входа
    ├── env.sample                # Пример файла с переменными окружения
    ├── parsers/
    │   └── misis_parser.py       # Парсер сайта misis.ru
    ├── scripts/
    │   └── parse_misis.py        # Скрипт для парсинга и загрузки
    └── services/
        └── hf_service.py         # Сервис с LangChain и Qdrant
```

## Особенности

- **RAG (Retrieval-Augmented Generation)**: Бот использует семантический поиск по данным с сайта МИСИС
- **Асинхронный парсинг**: Эффективный обход сайта с использованием aiohttp
- **Чанкинг**: Автоматическое разбиение текста на оптимальные чанки для векторного поиска
- **Многоязычные эмбеддинги**: Используется модель для русского языка

## Лицензия

MIT
