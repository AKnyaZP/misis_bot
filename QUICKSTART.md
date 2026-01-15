# Быстрый старт

## Запуск с Docker (рекомендуется)

1. **Клонируйте репозиторий и перейдите в директорию:**
   ```bash
   cd misis_bot
   ```

2. **Создайте файл `.env` в корне проекта:**
   ```bash
   cp env.sample .env
   ```

3. **Откройте `.env` и заполните необходимые переменные:**
   ```env
   BOT_TOKEN=ваш_токен_телеграм_бота
   HF_TOKEN=ваш_токен_huggingface
   ```

4. **Запустите все сервисы:**
   ```bash
   docker-compose up -d
   ```

5. **Загрузите данные в Qdrant:**
   ```bash
   docker-compose run --rm parser
   ```

6. **Проверьте статус:**
   ```bash
   docker-compose ps
   docker-compose logs -f bot
   ```

Готово! Бот должен быть запущен и готов к работе.

## Проверка работы

1. Найдите вашего бота в Telegram
2. Отправьте команду `/start`
3. Задайте вопрос об университете МИСИС

## Полезные команды

```bash
# Просмотр логов
docker-compose logs -f bot

# Остановка
docker-compose down

# Перезапуск
docker-compose restart

# Пересборка после изменений кода
docker-compose build bot
docker-compose up -d bot
```

## Альтернативный способ с Makefile

Если у вас установлен Make:

```bash
make up      # Запустить
make parser  # Загрузить данные
make logs    # Просмотр логов
make down    # Остановить
```

## Решение проблем

### Бот не запускается
- Проверьте логи: `docker-compose logs bot`
- Убедитесь, что BOT_TOKEN и HF_TOKEN заполнены в `.env`

### Парсер не может подключиться к Qdrant
- Дождитесь полного запуска Qdrant: `docker-compose ps`
- Проверьте, что Qdrant здоров: `docker-compose logs qdrant`

### Данные не загружаются
- Запустите парсер вручную: `docker-compose run --rm parser`
- Проверьте логи парсера: `docker-compose logs parser`
