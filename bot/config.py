from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional


class TgBotSettings(BaseSettings):
    bot_token: Optional[str] = None  # Будет загружено из .env или переменных окружения

    hf_model: str = "Qwen/Qwen2.5-7B-Instruct"  # GPT-oSS-20b будет использоваться через langchain
    hf_token: Optional[str] = None  # Будет загружено из .env или переменных окружения

    # Qdrant settings
    # В Docker используйте http://qdrant:6333, локально http://localhost:6333
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str = ""
    qdrant_collection_name: str = "misis_bot_vectors"

    # Generation settings
    max_new_tokens: int = 512
    temperature: float = 0.7

    model_config = SettingsConfigDict(
        env_file = Path(__file__).parent.joinpath(".env"), 
        env_file_encoding="utf-8",
        env_ignore_empty=True  # Игнорируем пустые значения из .env
    )

# Создаем настройки и обрабатываем пустые значения
_settings_raw = TgBotSettings()

# Если токены пустые или None, устанавливаем их в None явно
settings = TgBotSettings(
    bot_token=_settings_raw.bot_token if _settings_raw.bot_token and _settings_raw.bot_token.strip() else None,
    hf_token=_settings_raw.hf_token if _settings_raw.hf_token and _settings_raw.hf_token.strip() else None,
    hf_model=_settings_raw.hf_model,
    qdrant_url=_settings_raw.qdrant_url,
    qdrant_api_key=_settings_raw.qdrant_api_key,
    qdrant_collection_name=_settings_raw.qdrant_collection_name,
    max_new_tokens=_settings_raw.max_new_tokens,
    temperature=_settings_raw.temperature,
)