from aiogram import Dispatcher, Bot
import logging
from bot.config import settings
from bot.handlers import router



logger = logging.getLogger(__name__)

async def on_startup(bot: Bot) -> None:
    bot_info = await bot.get_me()
    logging.info(f"Bot @{bot_info.username} is up and running")

async def app() -> None:
    bot = Bot(token=settings.bot_token)
    
    from bot.services.hf_service import HFService
    from bot.middleware import HFServiceMiddleware
    
    hf_service = HFService(settings)

    dp = Dispatcher(on_startup=on_startup)
    dp.message.middleware(HFServiceMiddleware(hf_service))
    dp.include_router(router)

    await dp.start_polling(bot)