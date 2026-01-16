from aiogram import Dispatcher, Bot
import logging
from bot.config import settings
from bot.handlers import router



logger = logging.getLogger(__name__)

async def on_startup(bot: Bot) -> None:
    bot_info = await bot.get_me()
    logging.info(f"Bot @{bot_info.username} is up and running")

async def app() -> None:
    import os
    from pathlib import Path
    
    logger.info("Проверка конфигурации...")
    
    # Диагностика: показываем откуда загружается токен
    env_file = Path(__file__).parent / ".env"
    env_token = os.getenv("BOT_TOKEN")
    
    logger.info(f"Файл .env существует: {env_file.exists()}")
    if env_file.exists():
        try:
            with open(env_file, 'r') as f:
                env_content = f.read()
                bot_token_line = [line for line in env_content.split('\n') if line.strip().startswith('BOT_TOKEN=')]
                if bot_token_line:
                    # Показываем только первые и последние символы токена для безопасности
                    token_preview = bot_token_line[0][:20] + "..." if len(bot_token_line[0]) > 20 else bot_token_line[0]
                    logger.info(f"Строка BOT_TOKEN в .env: {token_preview}")
                    if 'BOT_TOKEN=' in bot_token_line[0] and len(bot_token_line[0].split('=', 1)) > 1:
                        token_value = bot_token_line[0].split('=', 1)[1].strip()
                        if token_value:
                            logger.info(f"✓ Токен найден в .env (длина: {len(token_value)} символов)")
                        else:
                            logger.warning("⚠ BOT_TOKEN= указан в .env, но значение пустое!")
                    else:
                        logger.warning("⚠ BOT_TOKEN= указан в .env, но значение отсутствует!")
        except Exception as e:
            logger.warning(f"Не удалось прочитать .env файл: {e}")
    
    if env_token:
        logger.info(f"✓ BOT_TOKEN найден в переменных окружения (длина: {len(env_token)} символов)")
    
    logger.info(f"Загруженный BOT_TOKEN из настроек: {'установлен' if settings.bot_token else 'НЕ установлен'}")
    if settings.bot_token:
        logger.info(f"Длина токена: {len(settings.bot_token)} символов")
        logger.info(f"Начинается с: {settings.bot_token[:10]}...")
    
    # Проверка токена перед запуском
    if settings.bot_token is None or (isinstance(settings.bot_token, str) and settings.bot_token.strip() == ""):
        logger.error("=" * 50)
        logger.error("ОШИБКА: BOT_TOKEN не установлен!")
        logger.error("=" * 50)
        logger.error("Установите реальный токен одним из способов:")
        logger.error("1. В файле bot/.env: BOT_TOKEN=ваш_токен_здесь")
        logger.error("2. В переменных окружения: export BOT_TOKEN=ваш_токен_здесь")
        logger.error("")
        logger.error("Получите токен у @BotFather в Telegram:")
        logger.error("1. Откройте Telegram и найдите @BotFather")
        logger.error("2. Отправьте команду /newbot")
        logger.error("3. Следуйте инструкциям")
        logger.error("4. Скопируйте полученный токен")
        logger.error("=" * 50)
        return
    
    logger.info("Подключение к Telegram...")
    try:
        bot = Bot(token=settings.bot_token)
        # Проверяем валидность токена
        bot_info = await bot.get_me()
        logger.info("=" * 50)
        logger.info(f"✓ Бот успешно подключен: @{bot_info.username}")
        logger.info(f"✓ Имя бота: {bot_info.first_name}")
        logger.info("=" * 50)
    except Exception as e:
        logger.error("=" * 50)
        logger.error(f"ОШИБКА подключения к Telegram: {e}")
        logger.error("Проверьте правильность BOT_TOKEN в bot/.env")
        logger.error("=" * 50)
        return
    
    logger.info("Инициализация сервисов...")
    from bot.services.hf_service import HFService
    from bot.middleware import HFServiceMiddleware
    
    try:
        hf_service = HFService(settings)
        logger.info("✓ HFService инициализирован")
    except Exception as e:
        logger.warning(f"⚠ Ошибка инициализации HFService: {e}")
        logger.warning("Бот будет работать без RAG функциональности")
        hf_service = None

    logger.info("Настройка диспетчера...")
    dp = Dispatcher(on_startup=on_startup)
    if hf_service:
        dp.message.middleware(HFServiceMiddleware(hf_service))
        logger.info("✓ RAG функциональность включена")
    dp.include_router(router)
    logger.info("✓ Роутеры подключены")

    logger.info("=" * 50)
    logger.info("Бот запущен и готов к работе!")
    logger.info("Нажмите Ctrl+C для остановки")
    logger.info("=" * 50)
    
    await dp.start_polling(bot)