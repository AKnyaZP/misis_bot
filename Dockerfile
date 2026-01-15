FROM python:3.11-slim

WORKDIR /app

# Установка системных зависимостей
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Копируем requirements.txt и устанавливаем зависимости
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Копируем код приложения
COPY bot/ ./bot/

# Создаем директорию для логов
RUN mkdir -p /app/logs

# Устанавливаем рабочую директорию
WORKDIR /app

# Команда по умолчанию
CMD ["python", "-m", "bot.run"]
