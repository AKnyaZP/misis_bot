from aiogram import Dispatcher, Bot
import logging
from cw.config import settings
from cw.handlers import router
from huggingface_hub import InferenceClient



logger = logging.getLogger(__name__)

async def on_startup(bot: Bot) -> None:
    bot_info = await bot.get_me()
    logging.info(f"Bot @{bot_info.username} is up and running")

async def app() -> None:
    bot = Bot(token=settings.bot_token)
    hf_client=InferenceClient(model=settings.hf_model, token=settings.hf_token)

    dp=Dispatcher(on_startup=on_startup, hf_client=hf_client)
    dp.include_router(router)

    await dp.start_polling(bot)