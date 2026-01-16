import asyncio
import logging
import sys
from bot.app import app

if __name__ == "__main__":
    # Настраиваем логирование в файл и консоль
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler("bot.log"),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info("=" * 50)
    logger.info("Запуск бота МИСИС...")
    logger.info("=" * 50)
    
    try:
        asyncio.run(app())
    except KeyboardInterrupt:
        logger.info("Бот остановлен пользователем")
    except Exception as e:
        logger.exception(f"Критическая ошибка: {e}")
        sys.exit(1)