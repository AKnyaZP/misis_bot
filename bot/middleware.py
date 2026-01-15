from typing import Callable, Dict, Any, Awaitable
from aiogram import BaseMiddleware
from aiogram.types import TelegramObject
from bot.services.hf_service import HFService


class HFServiceMiddleware(BaseMiddleware):
    """Middleware для передачи HFService в handlers."""
    
    def __init__(self, hf_service: HFService):
        self.hf_service = hf_service

    async def __call__(
        self,
        handler: Callable[[TelegramObject, Dict[str, Any]], Awaitable[Any]],
        event: TelegramObject,
        data: Dict[str, Any]
    ) -> Any:
        data["hf_service"] = self.hf_service
        return await handler(event, data)
