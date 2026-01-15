import asyncio
import logging
from bot.app import app

if __name__ == "__main__":
    logging.basicConfig(
        filename="bot.log",
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    try:
        asyncio.run(app())
    except KeyboardInterrupt:
        logging.info("Bot stopped by user")
    except Exception as e:
        logging.exception(e)