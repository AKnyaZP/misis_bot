from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class TgBotSettings(BaseSettings):
    bot_token: str

    hf_model: str = "Qwen/Qwen2.5-7B-Instruct"  # GPT-oSS-20b будет использоваться через langchain
    hf_token: str

    # Qdrant settings
    # В Docker используйте http://qdrant:6333, локально http://localhost:6333
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str = ""
    qdrant_collection_name: str = "misis_bot_vectors"

    # Generation settings
    max_new_tokens: int = 512
    temperature: float = 0.7

    model_config = SettingsConfigDict(
        env_file = Path(__file__).parent.joinpath(".env"), env_file_encoding="utf-8"
    )

settings = TgBotSettings()