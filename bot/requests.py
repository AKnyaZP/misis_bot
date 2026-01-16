import logging
from typing import Optional
from bot.services.hf_service import HFService

logger = logging.getLogger(__name__)

async def get_hf_response(prompt: str, hf_service: Optional[HFService], use_rag: bool = True) -> str:
    """Получает ответ от сервиса HuggingFace."""
    if hf_service is None:
        return "Извините, сервис генерации ответов недоступен. Проверьте настройки BOT_TOKEN и HF_TOKEN в bot/.env"
    
    try:
        result = await hf_service.generate(prompt, use_rag=use_rag)
        return result
    except RuntimeError as e:
        error_msg = str(e)
        logger.error(f"Ошибка генерации ответа: {error_msg}")
        if "Токен HuggingFace не имеет прав" in error_msg:
            return (
                "⚠️ Ошибка доступа к модели.\n\n"
                "Токен HuggingFace не имеет достаточных прав для использования Inference API.\n\n"
                "Для исправления:\n"
                "1. Перейдите на https://huggingface.co/settings/tokens\n"
                "2. Создайте новый токен с правами 'read'\n"
                "3. Обновите HF_TOKEN в файле bot/.env\n"
                "4. Перезапустите бота"
            )
        elif "Неверный токен" in error_msg:
            return (
                "⚠️ Ошибка аутентификации.\n\n"
                "Неверный токен HuggingFace.\n\n"
                "Проверьте правильность HF_TOKEN в файле bot/.env"
            )
        else:
            return f"⚠️ Ошибка при генерации ответа:\n\n{error_msg}"
    except Exception as e:
        logger.error(f"Неожиданная ошибка генерации ответа: {e}")
        return (
            "⚠️ Произошла неожиданная ошибка при генерации ответа.\n\n"
            "Попробуйте позже или обратитесь к администратору."
        )